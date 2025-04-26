# 📈 PredicTick - Financial Market Prediction Framework

**Author:** Arturo Mendoza ([arturo.amb89@gmail.com](mailto:arturo.amb89@gmail.com))

---

## 🚀 Overview

PredicTick is a modular framework for forecasting market direction — Down, Neutral, or Up — based on historical price action and technical indicators.

The name blends "Predict" and "Tick", reflecting its core mission: predicting the next market tick with precision, speed, and intelligence.

PredicTick combines modern machine learning with financial expertise to support daily predictions, training pipelines, backtesting simulations, and dashboards.

It includes:

* **Boosting models** (`XGBoost`) optimized via `Optuna` for multiclass classification.
* **Centralized configuration** using `ParameterLoader` for full control and reproducibility.
* **Technical feature engineering including** (RSI, MACD, Bollinger Bands, Stochastic RSI, and more).
* **Backtesting engine for historical evaluation** of prediction strategies.
* **Interactive dashboards** built with `Streamlit` and `Optuna` visualizations.

<p align="center"><img src="images/logo-1.png" alt="Logo" style="width: 50%; height: auto;"></p>

---

## 🧱 Project Structure

```bash
root/
├── src/
│   ├── backtesting/      # Historical simulations
│   ├── dashboard/        # Interactive dashboard
│   ├── evaluation/       # Performance evaluation
│   ├── market_data/      # Market data downloading and updating
│   ├── prediction/       # Daily predictions
│   ├── training/         # Model training
│   └── utils/            # Global parameters and utilities
├── config/               # Symbol lists and parameter files
├── data/                 # Data artifacts and models
├── images/               # App logo and other project images
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

To set up the environment:

```bash
bash envtool.sh install
```

This script will:

* Create a Python virtual environment `.venv` if missing.
* Upgrade `pip`.
* Install dependencies from `requirements.txt` and `requirements-dev.txt`.

### 📦 Dependencies

> **Note:** For the full and up-to-date list of dependencies (including exact versions), please refer to `requirements.txt` and `requirements-dev.txt`.

**Main dependencies:**

* `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`
* `optuna`, `ta`, `pandas_market_calendars`
* `yfinance`, `streamlit`, `holidays`
* `joblib`, `matplotlib`, `seaborn`, `plotly`
* Google API & environment: `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`, `python-dotenv`

**Development tools:**

* `black`, `isort`, `bandit`, `pylint`, `autoflake`, `pydocstringformatter`, `coverage`, `pytest`

---

## 🧠 Centralized Configuration

All pipeline components use a centralized `ParameterLoader` object that handles:

* Symbols for training/prediction (`training_symbols`, `correlative_symbols`)
* Thresholds and indicator windows (e.g. RSI, MACD)
* Paths to models, scalers, plots, and JSON data
* Market timezone and FED event days
* Cutoff date for training/evaluation

This ensures **consistency, maintainability and reproducibility** across modules.

---

## 🛠️ Main Workflows

Use the `run_tasks.sh` automation script to launch any task:

### 1. 📥 Update Market Data

```bash
bash run_tasks.sh update
```

### 2. 🧠 Train Model

```bash
bash run_tasks.sh train
```

### 3. 📊 Evaluate Model

```bash
bash run_tasks.sh evaluate
```

### 4. 🔮 Daily Prediction

```bash
bash run_tasks.sh predict
```

### 5. 🕰️ Backtesting

```bash
bash run_tasks.sh backtest
```

### 6. 📈 Launch Dashboard

```bash
bash run_tasks.sh dashboard
```

### 7. 🚀 Full Pipeline (Update → Train → Evaluate → Predict)

```bash
bash run_tasks.sh all
```

---

## 🔀 Modeling Strategy

This framework uses a **Multi-active model** approach:

* A model is trained using multiple symbols as input, leveraging cross-symbol features such as correlation with benchmark assets (e.g., SPY).
* This allows for shared learning across assets, improving generalization and efficiency.

---

## 🧠 Prediction Logic

* **Multiclass classification (3 classes):**

  * `↓ Down` (target = 0)
  * `→ Neutral` (target = 1)
  * `↑ Up` (target = 2)

* **Input features include:**

  * `return_1h`, `volatility_3h`, `momentum_3h`, `rsi`, `macd`, `volume`, `bb_width`
  * `stoch_rsi`, `obv`, `atr`, `williams_r`, `hour`, `day_of_week`, `is_fed_day`, `is_before_holiday`
  * Cross-asset features: `spread_vs_SPY`, `corr_5d_SPY`

* **Model:** `XGBoostClassifier` with `multi:softprob` objective

* **Optimization:** `Optuna` + `TimeSeriesSplit` + `RandomUnderSampler` for class balance

---

## 📊 Example Outputs

* ✅ Confusion matrix (normalized %)
* ✅ F1-score per class (bar plot)
* ✅ Expected return score for best symbol
* ✅ Prediction logs:

```text
🟢 SUCC | Best Symbol for 2024-06-20: AAPL → UP (Score: 1.245)
```

---

## 📁 Logging Directory:

All logs printed to the console during execution are also stored persistently in the logs/ directory. This includes messages from data updates, training runs, predictions, and backtests. The log files are timestamped and named accordingly to support debugging, auditing, and monitoring workflows.

---

## 📈 Optuna Dashboard

An interactive Streamlit-based dashboard allows you to inspect the hyperparameter optimization results:

```bash
bash run_tasks.sh dashboard
```

Features:

* Optimization history
* Parallel coordinate plots
* Hyperparameter importance
* Slice plots per trial

---

## ⏰ Timezone & Calendars

* All dates are converted and processed in **America/New\_York** timezone.
* FED event days and US holidays are injected into the feature set.
* Training cutoff date is dynamically defined in `ParameterLoader`.

---

## 🪪 Data Integrity & Artifacts

* ✅ Market data is validated via `validate_market_data()` on each update.
* ✅ Artifacts (models, scalers, Optuna studies) are saved with timestamps for reproducibility.
* ✅ All critical components are testable and modular.

---

## 📌 Naming Conventions

* ✅ Classes: `PascalCase`
* ✅ Functions & variables: `snake_case` (PEP8)
* ✅ Constants: `ALL_CAPS_SNAKE`
* ✅ No usage of camelCase
* ✅ Linting tools: `black`, `pylint`, `isort`, `bandit`, `autoflake`

---

## ⚠️ Initial Setup Required

For the project to operate correctly, you will need to create and populate certain configuration files. Please ensure the following are set up:

* **`.env` file:** Create this file in the root directory. It is used for vulnerable variables. (Note: `envtool.sh` does not automatically create this file.)

* **Configuration Directory and Files:**

  * Create a directory named `config` in the root of the project.
  * Inside `config/`, create `symbols.json`. This file should define the symbols for training and prediction.
  * Inside `config/`, create `symbols_invalid.json`. This file should list any symbols to be excluded.
  * Create a subdirectory `gcp` inside `config/` (i.e., `config/gcp/`).
  * Inside `config/gcp/`, the `credentials.json` file must be placed (explained later in detail under 'How to Generate credentials.json'). This file is needed for Google Cloud Platform interactions, such as uploading or downloading artifacts from Google Drive.

* **Example Configuration Files:**

Below are sample contents for required JSON configuration files:

  * config/symbols.json
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

  * config/symbols\_invalid.json
```bash
["TSLA"]
```

  * .env
```bash
GDRIVE_FOLDER_ID=xxx  # Folder ID in Google Drive where artifacts will be uploaded/downloaded
```

### 🔐 How to Generate `credentials.json`

To use Google Drive with this project (for uploading or downloading models and artifacts), follow these steps:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Enable the **Google Drive API** for the project.
4. Create credentials:
   * Choose **OAuth 2.0 Client ID** for interactive use **or**
   * Choose **Service Account** for headless/scripted access.
5. Download the resulting `credentials.json` file.
6. Save it in the path: `config/gcp/credentials.json`.

This enables secure, authenticated access to Google Drive resources from within the PredicTick framework.

## 🤝 AI GUIDE

[Open the AI Guide](AI_GUIDE.md)

---

## 🤝 Contributions

Contributions are welcome!

Please follow the existing project structure and coding standards.
To propose improvements or new features (e.g. model variants, indicators, dashboards), open a PR or issue.

---

> *“The future belongs to those who anticipate it.”*
