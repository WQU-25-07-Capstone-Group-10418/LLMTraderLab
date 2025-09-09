# LLMTraderLab

A comprehensive LLM-based trading prediction system that uses multiple AI models to forecast market movements and evaluate prediction accuracy.

## Features

- **Multi-Model Support**: Gemini, Amazon Nova, and GPT-3.5-turbo
- **Real-time Data**: Yahoo Finance integration for live market data
- **Comprehensive Evaluation**: Directional accuracy and prediction deviation metrics
- **Automated Backtesting**: Daily predictions with GitHub Actions
- **Historical Analysis**: Batch prediction and evaluation capabilities

## Prerequisites

1. Python version **3.12** or above
2. [Poetry](https://python-poetry.org/) as the package management
3. API Keys for:
   - OpenAI (GPT models)
   - Google Gemini
   - AWS Bedrock (Nova models)

## Installation

1. Clone the repository and install the packages

    ```bash
    git clone https://github.com/WQU-25-07-Capstone-Group-10418/LLMTraderLab.git
    cd LLMTraderLab
    poetry install
    ```

2. Enter Poetry Shell

    ```bash
    poetry shell
    # or: source .venv/bin/activate
    ```

3. Set up environment variables

    Create a `.env` file with your API keys:

    ```bash
    OPENAI_API_KEY=sk-proj-xxxxxxxxx
    GEMINI_API_KEY=xxxxxxxxx
    AWS_ACCESS_KEY_ID=xxx
    AWS_SECRET_ACCESS_KEY=xxxxxx
    ```

## Usage

### 1. Data Downloader

Download historical market data for S&P 500 and Russell 2000 indices:

```bash
python -m download_indices --help
```

### 2. Historical Predictions

Run predictions for multiple historical dates:

```bash
python -m get_predict
```

### 3. Real-time Backtesting

Generate daily predictions using current market data:

```bash
python -m backtest
```

### 4. Evaluation

Evaluate prediction accuracy:

```bash
# Evaluate historical predictions
python -m evaluation

# Evaluate backtest predictions
python -m backtest_evaluation
```

## Project Structure

```bash
LLMTraderLab/
├── data/                          # Data storage
│   ├── GSPC.csv                   # S&P 500 historical data
│   ├── RUT.csv                    # Russell 2000 historical data
│   ├── predict.csv                # Historical prediction results
│   ├── backtest.csv               # Real-time prediction results
│   ├── token_cost.csv             # Token usage tracking
│   └── backtest_token_cost.csv    # Backtest token usage
├── .github/workflows/             # GitHub Actions
│   └── daily-backtest.yml         # Automated daily predictions
├── get_predict.py                 # Historical prediction system
├── backtest.py                    # Real-time prediction system
├── evaluation.py                  # Historical evaluation
├── backtest_evaluation.py         # Backtest evaluation
├── download_indices.py            # Data downloader
└── pyproject.toml                 # Poetry configuration
```

## Model Configuration

The system supports multiple AI models with different parameters:

- **Models**: Gemini 2.5 Flash, Amazon Nova Lite, GPT-3.5-turbo
- **Temperatures**: 0.0, 0.5, 1.0
- **History Ranges**: 7 days, 30 days
- **Indices**: S&P 500 (GSPC), Russell 2000 (RUT)

Total combinations: 2 indices × 3 models × 3 temperatures × 2 ranges = 36 predictions per date

## Evaluation Metrics

### 1. Directional Accuracy

Percentage of predictions where direction matches reality (up/down)

### 2. Prediction Deviation

Formula: `(prediction - prev_actual) / (actual - prev_actual)`

- 1.0 = Perfect prediction
- 0.0 = No change predicted
- Negative = Wrong direction
- >1.0 = Overestimated change magnitude

### 3. Absolute Error

Percentage error: `|prediction - actual| / actual * 100`

## Automated Daily Predictions

The system includes GitHub Actions for automated daily predictions:

1. **Schedule**: Monday-Friday at 9:30 AM UTC
2. **Process**: Fetches real-time data, runs all model combinations
3. **Output**: Saves results to `data/backtest.csv`
4. **Commit**: Automatically commits results to main branch

### Setup GitHub Actions

1. Add repository secrets:
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

2. Enable Actions in repository settings
3. Workflow will run automatically

## Output Files

- **predict.csv**: Historical prediction results with actual values
- **backtest.csv**: Real-time prediction results (no actual values)
- **token_cost.csv**: Token usage tracking for cost analysis
- **backtest_token_cost.csv**: Token usage for backtest predictions

## License

This project is part of the WQU Capstone Project for Group 10418.
