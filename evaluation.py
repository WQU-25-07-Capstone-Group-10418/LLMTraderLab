import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

def evaluate():
    """
    Evaluate prediction accuracy using ./data/predict.csv

    Two types of evaluation:
    1. Directional accuracy: Whether predicted direction matches actual direction
    2. Prediction deviation: (prediction - prev_actual_value) / (actual_value - prev_actual_value)

    Aggregates results by: index, model, temperature, history_range
    """

    predict_csv_path = Path("data") / "predict.csv"
    if not predict_csv_path.exists():
        raise FileNotFoundError("Prediction data file not found: data/predict.csv")

    df = pd.read_csv(predict_csv_path, parse_dates=['predict_date'])

    if df.empty:
        print("No prediction data available for evaluation")
        return

    print(f"Loaded {len(df)} prediction records for evaluation")
    print(f"Date range: {df['predict_date'].min()} to {df['predict_date'].max()}")
    print()

    df = calculate_metrics(df)

    # Generate aggregated reports
    print("=" * 80)
    print("EVALUATION RESULTS")
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

    predicted_direction = df['prediction'] > df['prev_actual_value']  # True = up, False = down
    actual_direction = df['actual_value'] > df['prev_actual_value']   # True = up, False = down
    df['direction_correct'] = predicted_direction == actual_direction

    # Calculate prediction deviation
    # Formula: (prediction - prev_actual_value) / (actual_value - prev_actual_value)
    numerator = df['prediction'] - df['prev_actual_value']
    denominator = df['actual_value'] - df['prev_actual_value']

    df['prediction_deviation'] = np.where(
        denominator != 0,
        numerator / denominator,
        np.nan  # Error handler
    )

    df['abs_error_pct'] = np.abs(df['prediction'] - df['actual_value']) / df['actual_value'] * 100

    return df

def aggregate_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate results by a specific dimension."""

    grouped = df.groupby(dimension)

    results = []
    for name, group in grouped:
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
    try:
        results = evaluate()
        print("\n" + "=" * 80)
        explain_metrics()
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
