import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Dict, Any
from datetime import date, timedelta

def load_backtest_data_with_actual_values():
    """
    Load backtest data and supplement it with actual market values.
    
    Returns:
        DataFrame with columns: index, model, temperature, history_range, predict_date, 
                               prediction, actual_value, prev_actual_value
    """
    # Load backtest predictions
    backtest_csv_path = Path("data") / "backtest.csv"
    if not backtest_csv_path.exists():
        raise FileNotFoundError("Backtest data file not found: data/backtest.csv")
    
    df = pd.read_csv(backtest_csv_path, parse_dates=['predict_date'])
    
    if df.empty:
        raise ValueError("No backtest data available for evaluation")
    
    print(f"Loaded {len(df)} backtest prediction records")
    print(f"Date range: {df['predict_date'].min()} to {df['predict_date'].max()}")
    print("Fetching actual market data for comparison...")
    
    # Get unique dates and indices to minimize API calls
    unique_dates = sorted(df['predict_date'].dt.date.unique())
    unique_indices = df['index'].unique()
    
    print(f"Need to fetch data for {len(unique_dates)} dates and {len(unique_indices)} indices")
    
    # Calculate date range for data download (add buffer for previous day data)
    start_date = min(unique_dates) - timedelta(days=5)  # Buffer for weekends
    end_date = max(unique_dates) + timedelta(days=1)
    
    # Map index values to Yahoo Finance tickers
    ticker_map = {
        "GSPC": "^GSPC",
        "RUT": "^RUT"
    }
    
    # Download market data for all required indices
    market_data = {}
    for index_name in unique_indices:
        if index_name not in ticker_map:
            raise ValueError(f"Unknown index: {index_name}")
        
        ticker = ticker_map[index_name]
        print(f"Downloading {ticker} data from {start_date} to {end_date}")
        
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            market_data[index_name] = data
            print(f"Downloaded {len(data)} trading days for {index_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to download data for {ticker}: {str(e)}")
    
    # Add actual_value and prev_actual_value to each row
    print("Processing actual values...")
    
    actual_values = []
    prev_actual_values = []
    
    for _, row in df.iterrows():
        index_name = row['index']
        predict_date = row['predict_date'].date()
        
        try:
            actual_val, prev_actual_val = get_actual_values_from_data(
                market_data[index_name], predict_date
            )
            actual_values.append(actual_val)
            prev_actual_values.append(prev_actual_val)
        except Exception as e:
            print(f"Warning: Could not get actual values for {index_name} on {predict_date}: {e}")
            actual_values.append(np.nan)
            prev_actual_values.append(np.nan)
    
    df['actual_value'] = actual_values
    df['prev_actual_value'] = prev_actual_values
    
    # Remove rows with missing actual values
    original_count = len(df)
    df = df.dropna(subset=['actual_value', 'prev_actual_value'])
    removed_count = original_count - len(df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} rows due to missing market data")
    
    print(f"Final dataset: {len(df)} predictions with complete data")
    return df

def get_actual_values_from_data(market_data: pd.DataFrame, predict_date: date) -> tuple[float, float]:
    """
    Extract actual_value and prev_actual_value from market data.
    
    Args:
        market_data: DataFrame with market data
        predict_date: The prediction date
        
    Returns:
        Tuple of (actual_value, prev_actual_value)
    """
    if market_data.empty:
        raise ValueError("Market data is empty")
    
    # Filter data up to and including the predict_date
    relevant_data = market_data[market_data.index.date <= predict_date]
    
    if relevant_data.empty:
        raise ValueError(f"No market data available up to {predict_date}")
    
    # Get the exact date
    actual_data = relevant_data[relevant_data.index.date == predict_date]
    if actual_data.empty:
        raise ValueError(f"No market data found for {predict_date}")
    
    actual_value = float(actual_data["Close"].iloc[0])
    
    # Get previous day's data (last trading day before predict_date)
    prev_data = relevant_data[relevant_data.index.date < predict_date]
    if prev_data.empty:
        raise ValueError(f"No previous day data found before {predict_date}")
    
    prev_actual_value = float(prev_data["Close"].iloc[-1])  # Last available day before predict_date
    
    return actual_value, prev_actual_value

def evaluate_backtest():
    """
    Evaluate backtest prediction accuracy using ./data/backtest.csv
    
    Two types of evaluation:
    1. Directional accuracy: Whether predicted direction matches actual direction
    2. Prediction deviation: (prediction - prev_actual_value) / (actual_value - prev_actual_value)
    
    Aggregates results by: index, model, temperature, history_range
    """
    
    # Load backtest data and supplement with actual values
    df = load_backtest_data_with_actual_values()
    
    print()
    
    # Calculate evaluation metrics (reuse from evaluation.py)
    df = calculate_metrics(df)
    
    # Generate aggregated reports
    print("=" * 80)
    print("BACKTEST EVALUATION RESULTS")
    print("=" * 80)
    print()
    
    # 1. Aggregate by Index
    print("1. RESULTS BY INDEX")
    print("-" * 40)
    index_results = aggregate_by_dimension(df, 'index')
    print_results_table(index_results, "Index")
    print()
    
    # 2. Aggregate by Model
    print("2. RESULTS BY MODEL")
    print("-" * 40)
    model_results = aggregate_by_dimension(df, 'model')
    print_results_table(model_results, "Model")
    print()
    
    # 3. Aggregate by Temperature
    print("3. RESULTS BY TEMPERATURE")
    print("-" * 40)
    temperature_results = aggregate_by_dimension(df, 'temperature')
    print_results_table(temperature_results, "Temperature")
    print()
    
    # 4. Aggregate by History Range
    print("4. RESULTS BY HISTORY RANGE")
    print("-" * 40)
    history_results = aggregate_by_dimension(df, 'history_range')
    print_results_table(history_results, "History Range")
    print()
    
    # Overall summary
    print("5. OVERALL SUMMARY")
    print("-" * 40)
    overall_stats = calculate_overall_stats(df)
    print_overall_summary(overall_stats)
    
    return {
        'by_index': index_results,
        'by_model': model_results,
        'by_temperature': temperature_results,
        'by_history_range': history_results,
        'overall': overall_stats
    }

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate directional accuracy and prediction deviation metrics."""
    
    # Calculate directional accuracy
    predicted_direction = df['prediction'] > df['prev_actual_value']  # True = up, False = down
    actual_direction = df['actual_value'] > df['prev_actual_value']   # True = up, False = down
    df['direction_correct'] = predicted_direction == actual_direction
    
    # Calculate prediction deviation
    # Formula: (prediction - prev_actual_value) / (actual_value - prev_actual_value)
    numerator = df['prediction'] - df['prev_actual_value']
    denominator = df['actual_value'] - df['prev_actual_value']
    
    # Handle division by zero (when actual_value == prev_actual_value, i.e., no change)
    df['prediction_deviation'] = np.where(
        denominator != 0,
        numerator / denominator,
        np.nan  # Set to NaN when there's no actual change
    )
    
    # Calculate absolute prediction error percentage
    df['abs_error_pct'] = np.abs(df['prediction'] - df['actual_value']) / df['actual_value'] * 100
    
    return df

def aggregate_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate results by a specific dimension."""
    
    grouped = df.groupby(dimension)
    
    results = []
    for name, group in grouped:
        # Filter out NaN values for deviation calculation
        valid_deviations = group['prediction_deviation'].dropna()
        
        stats = {
            dimension: name,
            'count': len(group),
            'directional_accuracy': group['direction_correct'].mean() * 100,  # Convert to percentage
            'avg_prediction_deviation': valid_deviations.mean() if len(valid_deviations) > 0 else np.nan,
            'std_prediction_deviation': valid_deviations.std() if len(valid_deviations) > 0 else np.nan,
            'median_prediction_deviation': valid_deviations.median() if len(valid_deviations) > 0 else np.nan,
            'avg_abs_error_pct': group['abs_error_pct'].mean(),
            'median_abs_error_pct': group['abs_error_pct'].median(),
        }
        results.append(stats)
    
    return pd.DataFrame(results)

def calculate_overall_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overall statistics across all predictions."""
    
    valid_deviations = df['prediction_deviation'].dropna()
    
    return {
        'total_predictions': len(df),
        'directional_accuracy': df['direction_correct'].mean() * 100,
        'avg_prediction_deviation': valid_deviations.mean() if len(valid_deviations) > 0 else np.nan,
        'std_prediction_deviation': valid_deviations.std() if len(valid_deviations) > 0 else np.nan,
        'median_prediction_deviation': valid_deviations.median() if len(valid_deviations) > 0 else np.nan,
        'avg_abs_error_pct': df['abs_error_pct'].mean(),
        'median_abs_error_pct': df['abs_error_pct'].median(),
        'best_directional_accuracy': df['direction_correct'].max() * 100,
        'worst_directional_accuracy': df['direction_correct'].min() * 100,
    }

def print_results_table(results_df: pd.DataFrame, dimension_name: str):
    """Print formatted results table."""
    
    print(f"{'Value':<20} {'Count':<8} {'Dir.Acc(%)':<12} {'Avg.Dev':<10} {'Med.Dev':<10} {'Avg.Err(%)':<12} {'Med.Err(%)':<12}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        deviation_avg = f"{row['avg_prediction_deviation']:.3f}" if not pd.isna(row['avg_prediction_deviation']) else "N/A"
        deviation_med = f"{row['median_prediction_deviation']:.3f}" if not pd.isna(row['median_prediction_deviation']) else "N/A"
        
        print(f"{str(row[dimension_name.lower().replace(' ', '_')]):<20} "
              f"{row['count']:<8} "
              f"{row['directional_accuracy']:<12.1f} "
              f"{deviation_avg:<10} "
              f"{deviation_med:<10} "
              f"{row['avg_abs_error_pct']:<12.2f} "
              f"{row['median_abs_error_pct']:<12.2f}")

def print_overall_summary(stats: Dict[str, Any]):
    """Print overall summary statistics."""
    
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Overall Directional Accuracy: {stats['directional_accuracy']:.1f}%")
    print(f"Average Prediction Deviation: {stats['avg_prediction_deviation']:.3f}" if not pd.isna(stats['avg_prediction_deviation']) else "Average Prediction Deviation: N/A")
    print(f"Median Prediction Deviation: {stats['median_prediction_deviation']:.3f}" if not pd.isna(stats['median_prediction_deviation']) else "Median Prediction Deviation: N/A")
    print(f"Standard Deviation of Prediction Deviation: {stats['std_prediction_deviation']:.3f}" if not pd.isna(stats['std_prediction_deviation']) else "Standard Deviation of Prediction Deviation: N/A")
    print(f"Average Absolute Error: {stats['avg_abs_error_pct']:.2f}%")
    print(f"Median Absolute Error: {stats['median_abs_error_pct']:.2f}%")

def explain_metrics():
    """Explain what the evaluation metrics mean."""
    
    print("METRIC EXPLANATIONS")
    print("=" * 50)
    print()
    print("1. Directional Accuracy (%)")
    print("   - Percentage of predictions where direction matches reality")
    print("   - Direction: UP (prediction > prev_value) vs DOWN (prediction < prev_value)")
    print("   - Higher is better (100% = perfect directional prediction)")
    print()
    print("2. Prediction Deviation")
    print("   - Formula: (prediction - prev_actual) / (actual - prev_actual)")
    print("   - 1.0 = Perfect prediction")
    print("   - 0.0 = No change predicted (prediction = prev_actual)")
    print("   - Negative = Wrong direction")
    print("   - >1.0 = Overestimated change magnitude")
    print()
    print("3. Absolute Error (%)")
    print("   - |prediction - actual| / actual * 100")
    print("   - Measures magnitude of prediction error")
    print("   - Lower is better (0% = perfect prediction)")
    print()

if __name__ == "__main__":
    # Run backtest evaluation
    try:
        results = evaluate_backtest()
        print("\n" + "=" * 80)
        explain_metrics()
    except Exception as e:
        print(f"Error during backtest evaluation: {str(e)}")
