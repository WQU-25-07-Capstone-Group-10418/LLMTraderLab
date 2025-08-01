from datetime import date
from enum import Enum
from litellm import completion
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import re
from typing import List, Tuple
import itertools
import os

load_dotenv()

# Define the parameters
class Index(str, Enum):
    SP500 = "GSPC"
    RUSSELL_2000 = "RUT"

class Model(str, Enum):
    GEMINI = "gemini/gemini-2.5-flash"
    NOVA = "bedrock/us.amazon.nova-lite-v1:0"
    GPT = "gpt-3.5-turbo"

class Temperature(float, Enum):
    LOW = 0.0
    MEDIUM = 0.5
    HIGH = 1.0

class HistoryRange(int, Enum):
    ONE_WEEK = 7
    ONE_MONTH = 30

def _extract_first_float(text: str) -> float:
    """Extract the first floating-point number from *text*.

    Raises ValueError if none found.
    """
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError("No numeric value found in LLM response")
    return float(match.group(0))

def get_predict(index: Index, model: Model, temperature: Temperature, history_range: HistoryRange, predict_date: date) -> float:
    history: List[float] = get_index_data(index, history_range, predict_date)
    if not history:
        raise ValueError("Historical data is empty; cannot make prediction")

    prices_str = ", ".join(f"{p:.2f}" for p in history)
    last_date_str = (predict_date).isoformat()  # predict_date references target date

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

    # Debugging log
    print(messages)

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

    return _extract_first_float(content)

def get_index_data(index: Index, history_range: HistoryRange, predict_date: date) -> list[float]:
    data_dir = Path("data")
    csv_path = data_dir / f"{index.value}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    # Skip first row (ticker info) and use second row as header
    df = pd.read_csv(csv_path, skiprows=1, parse_dates=[0], index_col=0).sort_index()
    
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df[df.index.date < predict_date]

    recent = df.tail(int(history_range))

    if "Close" not in recent.columns:
        raise ValueError("Expected 'Close' column in data file")

    return recent["Close"].tolist()

def get_actual_value(index: Index, predict_date: date) -> Tuple[float, float]:
    """Get the actual closing price for the given index and date, plus previous day's price.

    Args:
        index: The market index
        predict_date: The date to get the actual value for

    Returns:
        Tuple of (actual_value, prev_actual_value) - current day and previous day closing prices

    Raises:
        ValueError: If the actual value cannot be found
    """
    data_dir = Path("data")
    csv_path = data_dir / f"{index.value}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path, skiprows=1, parse_dates=[0], index_col=0).sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    relevant_data = df[df.index.date <= predict_date]

    if relevant_data.empty:
        raise ValueError(f"No data found for {index.value} up to {predict_date}")

    if "Close" not in relevant_data.columns:
        raise ValueError("Expected 'Close' column in data file")

    actual_data = relevant_data[relevant_data.index.date == predict_date]
    if actual_data.empty:
        raise ValueError(f"No actual data found for {index.value} on {predict_date}")

    actual_value = float(actual_data["Close"].iloc[0])

    prev_data = relevant_data[relevant_data.index.date < predict_date]
    if prev_data.empty:
        raise ValueError(f"No previous day data found for {index.value} before {predict_date}")

    prev_actual_value = float(prev_data["Close"].iloc[-1])

    return actual_value, prev_actual_value

def load_existing_predictions() -> pd.DataFrame:
    """Load existing predictions from CSV file, or return empty DataFrame if file doesn't exist."""
    predict_csv_path = Path("data") / "predict.csv"

    if predict_csv_path.exists():
        return pd.read_csv(predict_csv_path, parse_dates=['predict_date'])
    else:
        # Create empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'index', 'model', 'temperature', 'history_range',
            'predict_date', 'prediction', 'actual_value', 'prev_actual_value'
        ])

def save_prediction_result(index: Index, model: Model, temperature: Temperature,
                         history_range: HistoryRange, predict_date: date,
                         prediction: float, actual_value: float, prev_actual_value: float):
    """Save a single prediction result to the CSV file."""
    predict_csv_path = Path("data") / "predict.csv"
    predict_csv_path.parent.mkdir(exist_ok=True)

    new_row = {
        'index': index.value,
        'model': model.value,
        'temperature': float(temperature),
        'history_range': int(history_range),
        'predict_date': predict_date.isoformat(),
        'prediction': prediction,
        'actual_value': actual_value,
        'prev_actual_value': prev_actual_value
    }

    if predict_csv_path.exists():
        df = pd.read_csv(predict_csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(predict_csv_path, index=False)

def prediction_exists(existing_df: pd.DataFrame, index: Index, model: Model,
                     temperature: Temperature, history_range: HistoryRange,
                     predict_date: date) -> bool:
    """Check if a prediction already exists for the given parameters."""
    if existing_df.empty:
        return False

    predict_date_str = predict_date.isoformat()

    mask = (
        (existing_df['index'] == index.value) &
        (existing_df['model'] == model.value) &
        (existing_df['temperature'] == float(temperature)) &
        (existing_df['history_range'] == int(history_range)) &
        (existing_df['predict_date'] == predict_date_str)
    )

    return mask.any()

def run_all_predictions(predict_date: date):
    """Run predictions for all parameter combinations for a specific date.

    Args:
        predict_date: The date to make predictions for
    """
    # Load existing predictions to avoid duplicates
    existing_predictions = load_existing_predictions()

    print(f"Starting predictions for {predict_date}")
    print(f"Found {len(existing_predictions)} existing predictions")

    # Get all possible parameter combinations
    all_indices = list(Index)
    all_models = list(Model)
    all_temperatures = list(Temperature)
    all_history_ranges = list(HistoryRange)

    total_combinations = len(all_indices) * len(all_models) * len(all_temperatures) * len(all_history_ranges)
    print(f"Total combinations to process: {total_combinations}")

    processed = 0
    skipped = 0
    errors = 0

    # Iterate through all combinations
    for index, model, temperature, history_range in itertools.product(
        all_indices, all_models, all_temperatures, all_history_ranges
    ):
        processed += 1

        # Check if this combination already exists
        if prediction_exists(existing_predictions, index, model, temperature, history_range, predict_date):
            skipped += 1
            print(f"[{processed}/{total_combinations}] Skipping {index.value}-{model.value}-{temperature}-{history_range} (already exists)")
            continue

        try:
            print(f"[{processed}/{total_combinations}] Processing {index.value}-{model.value}-{temperature}-{history_range}")

            # Make prediction
            prediction = get_predict(index, model, temperature, history_range, predict_date)

            # Get actual value and previous day's actual value
            actual_value, prev_actual_value = get_actual_value(index, predict_date)

            # Save result
            save_prediction_result(index, model, temperature, history_range,
                                 predict_date, prediction, actual_value, prev_actual_value)

            print(f"  Prediction: {prediction:.2f}, Actual: {actual_value:.2f}, Prev: {prev_actual_value:.2f}")

        except Exception as e:
            errors += 1
            print(f"  Error: {str(e)}")
            continue

    print(f"\nCompleted processing for {predict_date}")
    print(f"Total processed: {processed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors: {errors}")
    print(f"Successfully completed: {processed - skipped - errors}")

if __name__ == "__main__":
    run_all_predictions(date(2024, 12, 17))

    # print("Testing LLM connectivity...")

    # try:
    #     gemini_response = completion(
    #         model="gemini/gemini-2.5-flash",
    #         messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
    #     )
    #     print(f"✅ Gemini: {gemini_response['choices'][0]['message']['content'][:50]}...")
    # except Exception as e:
    #     print(f"❌ Gemini error: {e}")

    # try:
    #     nova_response = completion(
    #         model="bedrock/us.amazon.nova-lite-v1:0",
    #         messages=[{ "content": "Hello, how are you?","role": "user"}]
    #     )
    #     print(f"✅ Nova: {nova_response['choices'][0]['message']['content'][:50]}...")
    # except Exception as e:
    #     print(f"❌ Nova error: {e}")

    # try:
    #     gpt_response = completion(
    #         model="gpt-3.5-turbo",
    #         messages=[{ "content": "Hello, how are you?","role": "user"}]
    #     )
    #     print(f"✅ GPT: {gpt_response['choices'][0]['message']['content'][:50]}...")
    # except Exception as e:
    #     print(f"❌ GPT error: {e}")
