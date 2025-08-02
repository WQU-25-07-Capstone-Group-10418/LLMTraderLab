import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple, Dict
import itertools
import re
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

# Import the enums and helper functions from get_predict
from get_predict import Index, Model, Temperature, HistoryRange, _extract_first_float

def fetch_real_time_data(days_back: int = 60) -> Dict[str, pd.DataFrame]:
    """
    Fetch real-time market data for the past N days.

    Args:
        days_back: Number of days of historical data to fetch (default: 60 to ensure we have enough trading days)

    Returns:
        Dictionary with index symbols as keys and DataFrames as values
    """
    print("Fetching real-time market data...")

    # Calculate date range
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    print(f"Date range: {start_date} to {end_date}")

    # Map our enum values to Yahoo Finance tickers
    ticker_map = {
        "GSPC": "^GSPC",
        "RUT": "^RUT"
    }

    tickers = list(ticker_map.values())
    print(f"Downloading data for: {tickers}")

    try:
        # Download data from Yahoo Finance
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )

        if data.empty:
            raise ValueError("No data returned from Yahoo Finance")

        print(f"Successfully downloaded data: {len(data)} rows")

        # Process and organize data
        processed_data = {}

        if len(tickers) == 1:
            # Single ticker case
            ticker_symbol = tickers[0]
            index_key = ticker_symbol.replace("^", "")
            processed_data[index_key] = data
            print(f"Processed {index_key}: {len(data)} trading days")
        else:
            # Multiple tickers case
            for ticker in tickers:
                try:
                    index_key = ticker.replace("^", "")
                    ticker_data = data[ticker]
                    if not ticker_data.empty:
                        processed_data[index_key] = ticker_data
                        print(f"Processed {index_key}: {len(ticker_data)} trading days")
                except (KeyError, AttributeError) as e:
                    print(f"Warning: no data for {ticker}: {e}")

        return processed_data

    except Exception as e:
        raise RuntimeError(f"Failed to fetch real-time data: {str(e)}")

def get_historical_prices_from_data(data: pd.DataFrame, history_range: HistoryRange, predict_date: date) -> List[float]:
    """
    Extract historical prices from fetched data instead of reading from CSV.

    Args:
        data: DataFrame with market data
        history_range: Number of days of history to use
        predict_date: The prediction date (should be today)

    Returns:
        List of closing prices for the specified history range
    """
    if data.empty:
        raise ValueError("Market data is empty")

    data_filtered = data[data.index.date < predict_date]

    if data_filtered.empty:
        raise ValueError(f"No historical data available before {predict_date}")

    recent_data = data_filtered.tail(int(history_range))

    if "Close" not in recent_data.columns:
        raise ValueError("Expected 'Close' column in market data")

    prices = recent_data["Close"].tolist()

    if len(prices) < int(history_range):
        print(f"Warning: Only {len(prices)} days of data available, requested {int(history_range)}")

    return prices

def get_predict_realtime(index: Index, model: Model, temperature: Temperature, 
                        history_range: HistoryRange, predict_date: date, 
                        market_data: Dict[str, pd.DataFrame]) -> Tuple[float, int, int, int]:
    """
    Make a prediction using real-time market data instead of CSV files.

    Args:
        index: Market index to predict
        model: LLM model to use
        temperature: Temperature setting
        history_range: Number of days of historical data
        predict_date: Date to make prediction for
        market_data: Dictionary of market data

    Returns:
        Tuple of (prediction, prompt_tokens, completion_tokens, total_tokens)
    """

    if index.value not in market_data:
        raise ValueError(f"No market data available for {index.value}")

    history = get_historical_prices_from_data(market_data[index.value], history_range, predict_date)

    if not history:
        raise ValueError("Historical data is empty; cannot make prediction")

    prices_str = ", ".join(f"{p:.2f}" for p in history)
    last_date_str = predict_date.isoformat()

    system_prompt = (
        "You are a seasoned financial analyst. Use quantitative reasoning and the provided historical prices "
        "to forecast the next trading day's closing price. Return ONLY the predicted price (a number)."
    )

    user_prompt = (
        f"Historical daily closing prices for {index.name.replace('_', ' ')} (oldest → newest):\n"
        f"{prices_str}\n\n"
        f"Please predict the closing price for {index.name.replace('_', ' ')} on {last_date_str}.\n"
        "Return only the numeric value, no additional text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print(f"Making prediction for {index.value} using {len(history)} days of data")

    response = completion(
        model=model.value,
        messages=messages,
        temperature=float(temperature),
    )

    content: str
    try:
        content = response["choices"][0]["message"]["content"]
    except (TypeError, KeyError):
        content = str(response)

    # Extract token usage information
    try:
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
    except (TypeError, KeyError):
        prompt_tokens = completion_tokens = total_tokens = 0

    prediction = _extract_first_float(content)
    return prediction, prompt_tokens, completion_tokens, total_tokens

def save_backtest_result(index: Index, model: Model, temperature: Temperature,
                        history_range: HistoryRange, predict_date: date,
                        prediction: float):
    """Save a single backtest result to the CSV file."""
    backtest_csv_path = Path("data") / "backtest.csv"
    backtest_csv_path.parent.mkdir(exist_ok=True)

    new_row = {
        'index': index.value,
        'model': model.value,
        'temperature': float(temperature),
        'history_range': int(history_range),
        'predict_date': predict_date.isoformat(),
        'prediction': prediction
    }

    if backtest_csv_path.exists():
        df = pd.read_csv(backtest_csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(backtest_csv_path, index=False)

def save_backtest_token_cost(index: Index, model: Model, temperature: Temperature,
                            history_range: HistoryRange, prompt_tokens: int,
                            completion_tokens: int, total_tokens: int):
    """Save token usage information for backtest to CSV file."""
    token_csv_path = Path("data") / "backtest_token_cost.csv"
    token_csv_path.parent.mkdir(exist_ok=True)

    # Create unique ID
    unique_id = f"{index.value}-{model.value}-{float(temperature)}-{int(history_range)}"

    new_row = {
        'id': unique_id,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens
    }

    if token_csv_path.exists():
        df = pd.read_csv(token_csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(token_csv_path, index=False)

def run_backtest_predictions():
    """
    Run real-time predictions for all parameter combinations using current market data.

    This function:
    1. Fetches real-time market data once
    2. Uses today as the prediction date
    3. Runs all parameter combinations
    4. Saves results to ./data/backtest.csv
    """
    predict_date = date.today()
    print(f"Running backtest predictions for {predict_date}")
    print("=" * 80)

    try:
        market_data = fetch_real_time_data()
    except Exception as e:
        print(f"Failed to fetch market data: {str(e)}")
        return

    if not market_data:
        print("No market data available for prediction")
        return

    print(f"Market data fetched successfully for: {list(market_data.keys())}")
    print()

    # Get all possible parameter combinations
    all_indices = list(Index)
    all_models = list(Model)
    all_temperatures = list(Temperature)
    all_history_ranges = list(HistoryRange)

    total_combinations = len(all_indices) * len(all_models) * len(all_temperatures) * len(all_history_ranges)
    print(f"Total combinations to process: {total_combinations}")
    print()

    processed = 0
    errors = 0
    successful_predictions = []

    # Iterate through all combinations
    for index, model, temperature, history_range in itertools.product(
        all_indices, all_models, all_temperatures, all_history_ranges
    ):
        processed += 1

        try:
            print(f"[{processed}/{total_combinations}] Processing {index.value}-{model.value}-{temperature}-{history_range}")

            prediction, prompt_tokens, completion_tokens, total_tokens = get_predict_realtime(
                index, model, temperature, history_range, predict_date, market_data
            )
            save_backtest_result(index, model, temperature, history_range, 
                               predict_date, prediction)
            save_backtest_token_cost(index, model, temperature, history_range, 
                                   prompt_tokens, completion_tokens, total_tokens)

            successful_predictions.append({
                'index': index.value,
                'model': model.value,
                'prediction': prediction,
                'tokens': total_tokens
            })

            print(f"  Prediction: {prediction:.2f}")
            print(f"  Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

        except Exception as e:
            errors += 1
            print(f"  Error: {str(e)}")
            continue

    # Final summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(f"Prediction date: {predict_date}")
    print(f"Total combinations processed: {processed}")
    print(f"Successful predictions: {len(successful_predictions)}")
    print(f"Errors: {errors}")
    print(f"Success rate: {len(successful_predictions)/processed*100:.1f}%")

    if successful_predictions:
        print(f"\nResults saved to: ./data/backtest.csv")
        print(f"Token costs saved to: ./data/backtest_token_cost.csv")

        print(f"\nPrediction Summary:")
        for index in Index:
            index_predictions = [p for p in successful_predictions if p['index'] == index.value]
            if index_predictions:
                avg_prediction = sum(p['prediction'] for p in index_predictions) / len(index_predictions)
                print(f"  {index.value}: {len(index_predictions)} predictions, average: {avg_prediction:.2f}")

if __name__ == "__main__":
    run_backtest_predictions()
