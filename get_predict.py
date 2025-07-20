from datetime import date
from enum import Enum
from litellm import completion
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import re
from typing import List, Tuple

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

    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0).sort_index()

    df = df[df.index.date < predict_date]

    recent = df.tail(int(history_range))

    if "Close" not in recent.columns:
        raise ValueError("Expected 'Close' column in data file")

    return recent["Close"].tolist()

if __name__ == "__main__":
    # Test
    gemini_response = completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
    )

    nova_response = completion(
    model="bedrock/us.amazon.nova-lite-v1:0",
    messages=[{ "content": "Hello, how are you?","role": "user"}]
    )

    gpt_response = completion(
    model="gpt-3.5-turbo",
    messages=[{ "content": "Hello, how are you?","role": "user"}]
    )

    print(f"gemini_response: {gemini_response}")
    print(f"nova_response: {nova_response}")
    print(f"gpt_response: {gpt_response}")
