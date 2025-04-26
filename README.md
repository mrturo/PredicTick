# 📈 PredicTick - Financial Market Prediction Framework

**Author:** Arturo Mendoza (arturo.amb89@gmail.com)

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
│   ├── updater/          # Data downloading and updating
│   ├── training/         # Model training
│   ├── evaluation/       # Performance evaluation
│   ├── prediction/       # Daily predictions
│   ├── backtesting/      # Historical simulations
│   ├── dashboard/        # Interactive dashboard
│   └── util/             # Global parameters and utilities
├── config/               # Symbol lists and parameter files
├── data/                 # Data artifacts and models
├── images/               # App logo and other project images
├── README.md             # Project documentation
├── requirements.txt      # Project dependencies
├── envtool.sh            # Project setup and cleaning script
└── run_tasks.sh          # Task automation script
```

---

## ⚙️ Requirements & Setup

To set up the environment:

```bash
bash envtool.sh install
```

This script will:

- Create a Python virtual environment `.venv` if missing.
- Upgrade `pip`.
- Install dependencies from `requirements.txt`.

### 📦 Dependencies

- `pandas`, `numpy`, `scikit-learn`
- `xgboost`, `imbalanced-learn`
- `optuna`, `ta`, `pandas_market_calendars`
- `yfinance`, `streamlit`, `holidays`
- `joblib`, `matplotlib`, `seaborn`, `plotly`
- Dev tools: `black`, `isort`, `bandit`, `pylint`, `autoflake`, `pydocstringformatter`

---

## 🧠 Centralized Configuration

All pipeline components use a centralized `ParameterLoader` object that handles:

- Symbols for training/prediction (`training_symbols`, `correlative_symbols`)
- Thresholds and indicator windows (e.g. RSI, MACD)
- Paths to models, scalers, plots, and JSON data
- Market timezone and FED event days
- Cutoff date for training/evaluation

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

- A model is trained using multiple symbols as input, leveraging cross-symbol features such as correlation with benchmark assets (e.g., SPY).
- This allows for shared learning across assets, improving generalization and efficiency.

---

## 🧠 Prediction Logic

- **Multiclass classification (3 classes):**
  - `↓ Down` (target = 0)
  - `→ Neutral` (target = 1)
  - `↑ Up` (target = 2)

- **Input features include:**
  - `return_1h`, `volatility_3h`, `momentum_3h`, `rsi`, `macd`, `volume`, `bb_width`
  - `stoch_rsi`, `obv`, `atr`, `williams_r`, `hour`, `day_of_week`, `is_fed_day`, `is_before_holiday`
  - Cross-asset features: `spread_vs_SPY`, `corr_5d_SPY`

- **Model:** `XGBoostClassifier` with `multi:softprob` objective
- **Optimization:** `Optuna` + `TimeSeriesSplit` + `RandomUnderSampler` for class balance

---

## 📊 Example Outputs

- ✅ Confusion matrix (normalized %)
- ✅ F1-score per class (bar plot)
- ✅ Expected return score for best symbol
- ✅ Prediction logs:
  
```text
✅ SUCC | Best Symbol for 2024-06-20: AAPL → UP (Score: 1.245)
```

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

- All dates are converted and processed in **America/New_York** timezone.
- FED event days and US holidays are injected into the feature set.
- Training cutoff date is dynamically defined in `ParameterLoader`.

---

## 🧪 Data Integrity & Artifacts

- ✅ Market data is validated via `validate_market_data()` on each update.
- ✅ Artifacts (models, scalers, Optuna studies) are saved with timestamps for reproducibility.
- ✅ All critical components are testable and modular.

---

## 📌 Naming Conventions

- ✅ Classes: `PascalCase`
- ✅ Functions & variables: `snake_case` (PEP8)
- ✅ Constants: `ALL_CAPS_SNAKE`
- ✅ No usage of camelCase
- ✅ Linting tools: `black`, `pylint`, `isort`, `bandit`, `autoflake`

---

## ⚠️ Initial Setup Required

Ensure the following configuration files are created and populated:

- `config/config_parameters.json`
- `config/symbols/symbols_training.json`
- `config/symbols/symbols_correlative.json`

These define training symbols, model parameters, technical indicator settings, and data paths.

---

## 🤝 Contributions

Contributions are welcome!

Please follow the existing project structure and coding standards.
To propose improvements or new features (e.g. model variants, indicators, dashboards), open a PR or issue.

---

> _“The future belongs to those who anticipate it.”_
