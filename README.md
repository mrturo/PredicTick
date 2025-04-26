# 📈 PredicTick - Financial Market Prediction Framework

**Author:** Arturo Mendoza ([arturo.amb89@gmail.com](mailto:arturo.amb89@gmail.com))

---

## 🚀 Overview

PredicTick is a modular framework for forecasting market direction — Down, Neutral, or Up — based on historical price action and technical indicators.

The name blends "Predict" and "Tick", reflecting its core mission: predicting the next market tick with precision, speed, and intelligence.

PredicTick combines modern machine learning with financial expertise to support daily predictions, training pipelines, backtesting simulations, and dashboards.

It includes:

- **Boosting models** (`XGBoost`) optimized via `Optuna` for multiclass classification.
- **Centralized configuration** using `ParameterLoader` for full control and reproducibility.
- **Technical feature engineering including** (RSI, MACD, Bollinger Bands, Stochastic RSI, and more).
- **Backtesting engine for historical evaluation** of prediction strategies.
- **Interactive dashboards** built with `Streamlit` and `Optuna` visualizations.

<p align="center"><img src="images/logo-1.png" alt="Logo" style="width: 50%; height: auto;"></p>

---

## 🧱 Project Structure

```bash
root/
├── src/
│   ├── backtesting/      # Historical simulations
│   ├── dashboard/        # Interactive dashboard
│   ├── evaluation/       # Performance evaluation
│   ├── market_data/      # Download and enrichment of market data
│   ├── prediction/       # Daily predictions
│   ├── training/         # Model training
│   └── utils/            # Global parameters and utilities
├── .github/workflows/    # CI/CD pipelines
├── config/               # Symbol lists and parameter files
├── utils/                # Global parameters and utilities
├── images/               # App logo and other project images
├── Dockerfile            # Cloud Run optimized container
├── README.md             # Project documentation
├── requirements-dev.txt  # Project dev dependencies
├── requirements.txt      # Project prod dependencies
├── envtool.sh            # Project setup and cleaning script
└── run_tasks.sh          # Task automation script
```

---

## ⚙️ Requirements & Setup

> **Minimum Python version required: `3.10`**
> The framework uses advanced features such as `from __future__ import annotations` and enhanced type hinting that require Python ≥3.10.
> If multiple versions are installed, ensure the virtual environment uses Python 3.10 or newer.

To set up the environment (choose the installation mode `prod` or `dev`):

```bash
bash envtool.sh install prod   # production mode
bash envtool.sh install dev    # development mode
```

`envtool.sh` will fail if the mode argument is omitted.

This script will:

- Create a Python virtual environment `.venv` if missing.
- Upgrade `pip`.
- Install dependencies from `requirements.txt`, and additionally from `requirements-dev.txt` if in development mode.

### 📦 Dependencies

> **Note:** For the full and up-to-date list of dependencies (including exact versions), please refer to `requirements.txt` and `requirements-dev.txt`.

**Main dependencies:**

- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`.
- `optuna`, `ta`, `pandas_market_calendars`.
- `yfinance`, `streamlit`, `holidays`.
- `joblib`, `matplotlib`, `seaborn`, `plotly`.
- Google API & environment: `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`, `python-dotenv`.

**Development tools:**

- `black`, `isort`, `bandit`, `pylint`, `autoflake`, `pydocstringformatter`, `coverage`, `pytest`.

---

## 🛠️ Environment Utility Script (`envtool.sh`)

`envtool.sh` centralizes installation, maintenance and quality‑assurance tasks.
All available commands are listed below.

| Command | Example | Brief description |
|---------|---------|-------------------|
| `install {prod-dev}` | `bash envtool.sh install prod` | Create/activate `.venv`, upgrade `pip`, always install `requirements.txt`, and additionally `requirements-dev.txt` if in development mode. |
| `reinstall {prod-dev}` | `bash envtool.sh reinstall dev` | Remove environment and caches, then perform a fresh `install`. |
| `uninstall` | `bash envtool.sh uninstall` | Remove **everything**: `.venv`, caches and build artifacts. |
| `clean-env` | `bash envtool.sh clean-env` | Delete only the `.venv` directory. |
| `clean-cache` | `bash envtool.sh clean-cache` | Delete `__pycache__`, `.pytest_cache`, `.mypy_cache`, build artifacts, logs and temporary files. |
| `code-check [paths…]` | `bash envtool.sh code-check src/ tests/` | Run *isort*, *autoflake*, *pydocstringformatter*, *black*, *bandit* and *pylint* over the specified paths (default `src/ tests/`). |
| `status` | `bash envtool.sh status` | Show environment status: `.venv` presence, Python/pip versions, requirement files. |
| `test` | `bash envtool.sh test` | Activate `.venv`, run *pytest* with *coverage*, generate HTML report (`htmlcov/`). |

---

## 🛠️ Main Workflows (`run_tasks.sh`)

Use the task‑runner to launch common workflows.

| Task | Command | Action |
|------|---------|--------|
| Update Market Data | `bash run_tasks.sh update` | Download and enrich the latest market data. |
| Auto‑Update Hourly | `bash run_tasks.sh auto-update` | Continuous hourly data updates (uses `caffeinate`). |
| Train Model | `bash run_tasks.sh train` | Train the XGBoost model with current configuration. |
| Evaluate Model | `bash run_tasks.sh evaluate` | Evaluate model performance on the evaluation set. |
| Daily Prediction | `bash run_tasks.sh predict` | Generate daily forecasts. |
| Backtesting | `bash run_tasks.sh backtest` | Run historical simulations with stored models. |
| Launch Dashboard | `bash run_tasks.sh dashboard` | Start the Streamlit dashboard with Optuna visualizations. |
| Full Pipeline | `bash run_tasks.sh all` | Sequentially execute Update → Train → Evaluate → Predict. |

---

## 🐳 Docker Usage

The repository ships with a hardened multi-stage `Dockerfile` optimized for execution in **Google Cloud Run**. It builds the project virtual environment and runs the update workflow by default. After installing Docker, you can containerize the project as follows:

### 1. Build the image

```bash
docker build -t predictick .
```

Rebuild the image whenever dependencies or source files change so the container stays in sync with the repository.

### 2. Run the default update workflow

```bash
docker run --rm predictick
```

The image’s default command executes `bash run_tasks.sh update`. Mount any required configuration or environment files (e.g. `.env`, `config/`) if they are not baked into the image or contain secrets that should stay outside of version control.

### 3. Launch alternative workflows

Override the container command to trigger any other subcommand from `run_tasks.sh` without rebuilding the image:

```bash
docker run --rm predictick bash run_tasks.sh train
docker run --rm predictick bash run_tasks.sh backtest
docker run --rm predictick bash run_tasks.sh dashboard
```

Use the same pattern for additional flows such as `auto-update`, `predict`, or `all`. You can also pass environment variables or
bind mounts (`-v`) to provide external credentials, data directories, or output locations as needed by each task.

---

## ⚙️ CI/CD Deployment to Cloud Run

This repository includes a **GitHub Actions pipeline** configured to deploy automatically to **Google Cloud Run**:

* On every push to the **`main`** branch, the pipeline builds the Docker image, pushes it to **Artifact Registry**, and updates the **Cloud Run Job**.
* Manual approval is required before publishing to Artifact Registry for production safety.
* The deployment uses **Workload Identity Federation (WIF/OIDC)** for secure authentication to Google Cloud Platform.

Workflow file: `.github/workflows/deploy_updater.yml`.

---

## 🧠 Centralized Configuration

All pipeline components rely on a shared `ParameterLoader` that controls:

- Symbol lists for training/prediction (`training_symbols`, `correlative_symbols`).
- Indicator windows and thresholds (e.g., RSI, MACD).
- Artifact paths (models, scalers, plots, JSON data).
- Market timezone and FED event days.
- Training/evaluation cutoff date.

This guarantees **consistency and reproducibility** across modules.

---

## 🔀 Modeling Strategy

This framework uses a **multi‑asset model** approach:

- A model is trained with multiple symbols, leveraging cross‑asset signals such as correlation with benchmark indices (e.g., *SPY*).
- Shared learning improves generalization and reduces data‑sparsity issues.

---

## 🧠 Prediction Logic

- **Multiclass classification (3 classes):**

  - `↓ Down` (target = 0)
  - `→ Neutral` (target = 1)
  - `↑ Up` (target = 2)

- **Input features include:**

  - `rsi`, `macd`, `volume`, `bb_width`
  - `stoch_rsi`, `obv`, `atr`, `williams_r`, `hour`, `weekdays_current`, `is_fed_event`, `is_holiday`
  - Cross-asset features: `spread_vs_SPY`, `corr_5d_SPY`

- **Model:** `XGBoostClassifier` with `multi:softprob` objective.
- **Optimization:** `Optuna` + `TimeSeriesSplit` + `RandomUnderSampler` for class balance.

---

## 🧾 Feature Dictionary

**Price & Volume:**

- `open`: Opening price of the session.
- `low`: Lowest intraday price.
- `high`: Highest intraday price.
- `close`: Closing price of the session.
- `adj_close`: Adjusted close price accounting for splits/dividends.
- `volume`: Number of shares traded during the session.

**Technical Indicators:**

- `adx_14d`: Average Directional Index over 14-days; measures trend strength.
- `atr`: Average True Range; daily price volatility.
- `atr_14d`: Smoothed ATR using 14-days EMA.
- `macd`: Difference between 12- and 26-period EMAs; momentum signal.
- `rsi`: Relative Strength Index; measures recent price gains/losses.
- `stoch_rsi`: Normalized RSI oscillator (0 to 1).
- `williams_r`: Momentum oscillator (0 to -100); indicates overbought/oversold.

**Time-based Fractions:**

- `time_of_day`: Fraction of the current day (0 = 00:00, 1 = 00:00 next day).
- `time_of_week`: Fraction of the week (0 = Monday at 00:00, 1 = next Monday at 00:00).
- `time_of_month`: Fraction of the month (0 = first day at 00:00, 1 = first day of next month at 00:00).
- `time_of_year`: Fraction of the year (0 = Jan 1st at 00:00, 1 = Jan 1st of next year at 00:00).

**Derived Price Metrics:**

- `average_price`: (High + Low + Close) / 3.
- `typical_price`: Weighted mean of price range.
- `bb_width`: Width of Bollinger Bands; reflects volatility.
- `bollinger_pct_b`: Current price’s percentile inside the Bollinger Bands.
- `price_change`: Difference between close and open prices.
- `range`: Difference between daily high and low.

**Returns & Volatility:**

- `intraday_return`: Return from open to close.
- `overnight_return`: Return from previous close to today’s open.
- `return`: Overall return over the session.
- `volatility`: Estimated session volatility.

**Volume Dynamics:**

- `obv`: On‑Balance Volume; cumulative volume based on price direction.
- `volume_change`: Percent change in volume vs. prior session.
- `volume_rvol_20d`: Relative volume compared to 20‑day average.
- `relative_volume`: Ratio of current volume to its N‑day moving average (configurable window).

**Price Derivatives & Candle Patterns:**

- `price_derivative`: First‑order derivative of the price series, capturing instantaneous momentum.
- `smoothed_derivative`: Moving‑average‑smoothed derivative, reducing noise while preserving trend shifts.
- `candle_pattern`: Encoded single‑candle Japanese candlestick pattern identifier (e.g., hammer).
- `multi_candle_pattern`: Encoded multi‑candle pattern identifier (e.g., morning star).

**Event Windows & Calendar Flags:**

- `is_pre_fed_event`: Trading days before a scheduled Fed event.
- `is_fed_event`: The day of a scheduled Fed event (FOMC, minutes release, etc.).
- `is_post_fed_event`: Trading days after a Fed event.
- `is_pre_holiday`: Trading days before a market holiday.
- `is_holiday`: The calendar day of a market holiday.
- `is_post_holiday`: Trading days after a market holiday.

---

## 📊 Example Outputs

- ✅ Normalized confusion matrix.
- ✅ F1‑score per class.
- ✅ Expected return score by symbol.
- ✅ Prediction logs:

```text
🟢 SUCC | Best Symbol for 2024-06-20: AAPL → UP (Score: 1.245)
```

---

## 📁 Logging Directory:

All console logs are persisted under `logs/` with timestamps for easy audit and debugging.

---

## 📈 Optuna Dashboard

An interactive Streamlit-based dashboard allows you to inspect the hyperparameter optimization results:

```bash
bash run_tasks.sh dashboard
```

Features:

- Optimization history
- Parallel coordinate plots
- Hyperparameter importance
- Slice plots per trial

---

## ⏰ Timezone & Calendars

- All timestamps processed in **America/New_York**.
- FED event days and US holidays injected as features.
- Training cutoff date set dynamically via `ParameterLoader`.

---

## 🪪 Data Integrity & Artifacts

- ✅ Data validated via `validate_data()` on every update.
- ✅ Artifacts saved with timestamps for reproducibility.
- ✅ All critical components are modular and tested.

---

## 📌 Naming Conventions

- ✅ Classes: `PascalCase`.
- ✅ Functions/variables: `snake_case` (PEP‑8).
- ✅ Constants: `ALL_CAPS_SNAKE`.
- ✅ No camelCase.
- ✅ Linting via `black`, `pylint`, `isort`, `bandit`, `autoflake`.

---

## ⚠️ Initial Setup Required

For the project to operate correctly, you will need to create and populate certain configuration files. Please ensure the following are set up:

- **`.env` file:** Create this file in the root directory. It is used for vulnerable variables. (Note: `envtool.sh` does not automatically create this file.)

- **Configuration Directory and Files:**

  - Create a directory named `config` in the root of the project.
  - Inside `config/`, create `event_dates.json`. This file should define important economic event dates, such as FED meeting days, which can be used as features in the model.
  - Inside `config/`, create `symbols.json`. This file should define the symbols for training and prediction.
  - Inside `config/`, create `symbols_invalid.json`. This file should list any symbols to be excluded.
  - Create a subdirectory `gcp` inside `config/` (i.e., `config/gcp/`).
  - Inside `config/gcp/`, the `credentials.json` file must be placed (explained later in detail under 'How to Generate credentials.json'). This file is needed for Google Cloud Platform interactions, such as uploading or downloading artifacts from Google Drive.

- **Example Configuration Files:**

Below are sample contents for required JSON configuration files:

- config/event_dates.json

```bash
{
  "fed_event_days": [
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-06",
    "2024-12-18"
  ]
}
```

- config/symbols.json

```bash
{
  "training": ["AAPL", "BTC-USD", "EURUSD=X", "VOO", "GLD"],
  "correlative": ["VOO", "GLD"],
  "prediction_groups": [
    {
      "name": "group-1",
      "symbols": ["BTC-USD"]
    },
    {
      "name": "group-2",
      "symbols": ["AAPL", "VOO", "GLD"]
    }
  ]
}
```

- config/symbols_invalid.json

```bash
["TSLA"]
```

- .env

```bash
GDRIVE_FOLDER_ID=xxx  # Folder ID in Google Drive where artifacts will be uploaded/downloaded
```

### 🔐 How to Generate `credentials.json`

To use Google Drive with this project (for uploading or downloading models and artifacts), follow these steps:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Enable the **Google Drive API** for the project.
4. Create credentials:
   - Choose **OAuth 2.0 Client ID** for interactive use **or**
   - Choose **Service Account** for headless/scripted access.
5. Download the resulting `credentials.json` file.
6. Save it in the path: `config/gcp/credentials.json`.

This enables secure, authenticated access to Google Drive resources from within the PredicTick framework.

## 🤝 AI GUIDE

[Open the AI Guide](AI_GUIDE.md)

---

## 🤝 Contributions

Contributions are welcome! 

Please follow the existing project structure and coding standards. To propose improvements or new features (e.g. model variants, indicators, dashboards), open a PR or issue.

---

> _“The future belongs to those who anticipate it.”_
