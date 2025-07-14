import time
from pathlib import Path
from typing import List

import click
import yfinance as yf
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

# Default tickers for Russell 2000 and S&P 500 indices on Yahoo Finance
DEFAULT_TICKERS = ["^RUT", "^GSPC"]


def chunked(iterable: List[str], size: int):
    """Yield successive chunks from *iterable* of length *size*."""
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("data"),
    show_default=True,
    help="Directory where CSV files will be stored.",
)
@click.option(
    "--start-date",
    type=str,
    default=None,
    help="Date string (YYYY-MM-DD) to begin downloading from. Defaults to Yahoo's max history.",
)
@click.option(
    "--end-date",
    type=str,
    default=None,
    help="Date string (YYYY-MM-DD) to end downloading at (inclusive). Defaults to today.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Number of tickers to request in a single Yahoo query. Use small values to reduce rate limits.",
)
@click.option(
    "--sleep",
    "sleep_seconds",
    type=float,
    default=1.5,
    show_default=True,
    help="Seconds to wait between each batch to avoid hitting rate limits.",
)
@click.option(
    "--ticker",
    "tickers",
    multiple=True,
    help="Specify extra tickers to download in addition to Russell 2000 (^RUT) and S&P 500 (^GSPC). Can be passed multiple times.",
)
def cli(
    output_dir: Path,
    start_date: str,
    end_date: str,
    batch_size: int,
    sleep_seconds: float,
    tickers: List[str],
):
    """Download daily OHLC data for Russell 2000 and S&P 500 (plus optional extra tickers) using Yahoo Finance.

    The script writes one CSV per ticker so that re-running the command will automatically skip any ticker that was
    already downloaded. Use --start-date / --end-date to restrict the fetched period.
    """

    # Collect final ticker list and ensure uniqueness + original order
    tickers_list: List[str] = []
    seen = set()
    for tkr in DEFAULT_TICKERS + list(tickers):
        if tkr not in seen:
            tickers_list.append(tkr)
            seen.add(tkr)

    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(tickers_list)
    click.echo(f"Preparing to download {total} tickers to {output_dir.resolve()}")

    # Iterate in batches
    for batch_idx, batch in enumerate(chunked(tickers_list, batch_size), start=1):
        # Filter out tickers that already have data saved
        pending = [tkr for tkr in batch if not (output_dir / f"{tkr.replace('^', '')}.csv").exists()]
        if not pending:
            click.echo(f"Batch {batch_idx}: all {len(batch)} tickers already downloaded, skipping.")
            continue

        click.echo(f"Batch {batch_idx}: downloading {pending} …")
        
        try:
            data = yf.download(
                tickers=pending,
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            click.echo(f"Error downloading batch {batch_idx}: {e}")
            continue

        # Handle different return formats from yfinance
        if data.empty:
            click.echo(f"Warning: no data returned for batch {pending}")
            continue

        # Process the data based on number of tickers
        if len(pending) == 1:
            # Single ticker - data is a DataFrame
            ticker = pending[0]
            if not data.empty:
                csv_path = output_dir / f"{ticker.replace('^', '')}.csv"
                # Ensure the index is properly formatted as date
                data.index = pd.to_datetime(data.index).date
                data.to_csv(csv_path)
                click.echo(f"Saved {ticker} → {csv_path}")
        else:
            # Multiple tickers - data has MultiIndex columns
            for ticker in pending:
                try:
                    ticker_data = data[ticker]
                    if not ticker_data.empty:
                        csv_path = output_dir / f"{ticker.replace('^', '')}.csv"
                        # Ensure the index is properly formatted as date
                        ticker_data.index = pd.to_datetime(ticker_data.index).date
                        ticker_data.to_csv(csv_path)
                        click.echo(f"Saved {ticker} → {csv_path}")
                except (KeyError, AttributeError) as e:
                    click.echo(f"Warning: no data for {ticker}: {e}")

        if batch_idx * batch_size < total:
            time.sleep(sleep_seconds)

    click.echo("Download completed.")


if __name__ == "__main__":
    cli() 