"""
Module for evaluating trained financial market prediction models.

Loads trained artifacts, applies feature engineering, generates evaluation metrics
(confusion matrix, classification report, F1-score plots), and saves visual outputs
for a multi-class (Down/Neutral/Up) classification model.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from market_data.gateway import Gateway
from utils.calendar_manager import CalendarManager
from utils.feature_engineering import FeatureEngineering
from utils.logger import Logger
from utils.parameters import ParameterLoader
from utils.plots import Plots

_PARAMS = ParameterLoader(Gateway.get_last_update())
_CONFUSION_MATRIX_FIGSIZE = _PARAMS.get("confusion_matrix_figsize")
_CONF_MATRIX_PLOT_FILEPATH = _PARAMS.get("conf_matrix_plot_filepath")
_ALL_SYMBOLS = _PARAMS.get("all_symbols")
_CUTOFF_DATE = _PARAMS.get("cutoff_date")
_EVALUATION_REPORT_BASEPATH = _PARAMS.get("evaluation_report_basepath")
_F1_SCORE_FIGSIZE = _PARAMS.get("f1_score_figsize")
_F1_SCORE_PLOT_FILEPATH = _PARAMS.get("f1_score_plot_filepath")
_FEATURES = _PARAMS.get("features")
_FED_EVENT_DAYS = _PARAMS.get("fed_event_days")
_MARKET_TZ = _PARAMS.get("market_tz")
_MODEL_FILEPATH = _PARAMS.get("model_filepath")

TARGET = "target"
CROSS_FEATURES = ["spread_vs_spy", "corr_5d_spy"]
SYMBOL_COLUMN = "symbol"
DATETIME_COLUMN = "datetime"
CATEGORICAL_FEATURES = [SYMBOL_COLUMN]


def load_model():
    """
    Load the trained classification model from the predefined file path.

    Returns:
        Trained model object loaded via joblib.
    """
    Logger.info("Loading model artifact...")
    return joblib.load(_MODEL_FILEPATH)


def build_calendar_context():
    """
    Build context for evaluation by retrieving US market holidays and FED event days.

    Returns:
        tuple: A tuple containing two lists:
            - us_holidays (list of dates)
            - fed_event_days_dt (list of dates)
    """
    Logger.info("Building calendar context (US holidays & FED events)...")
    _, us_holidays, fed_event_days_dt = CalendarManager.build_market_calendars(
        _FED_EVENT_DAYS
    )
    return us_holidays, fed_event_days_dt


def load_and_process_data(symbols: list) -> pd.DataFrame:
    """
    Load market data, perform feature engineering, and label target values.

    Args:
        symbols (list): List of symbols to include.
        us_holidays (list): Dates of US holidays.
        fed_event_days (list): Dates of FED event days.

    Returns:
        pd.DataFrame: Processed DataFrame ready for model input.
    """
    Logger.info("Loading and preprocessing data for evaluation...")
    market_data = Gateway.load()["symbols"]
    combined = []
    cutoff_date = pd.to_datetime(_CUTOFF_DATE).tz_localize(_MARKET_TZ)

    for symbol, entry in market_data.items():
        if symbol not in symbols:
            continue

        df = pd.DataFrame(entry["historical_prices"])
        if df.empty or not {"close", "high", "low", "volume"}.issubset(df.columns):
            Logger.warning(f"Skipping {symbol}: incomplete data.")
            continue

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
        df = df[df.index <= cutoff_date]
        df.sort_index(inplace=True)
        df["symbol"] = symbol

        df = FeatureEngineering.enrich_with_common_features(df, symbol)

        combined.append(df)

    if not combined:
        Logger.error("No valid dataframes generated. Aborting evaluation.")
        return pd.DataFrame()

    return pd.concat(combined).reset_index()


def engineer_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate cross-asset features like spread vs SPY and rolling correlation.

    Args:
        df (pd.DataFrame): DataFrame containing return_1h and SPY data.

    Returns:
        pd.DataFrame: DataFrame with additional cross features.
    """
    df = df.copy()
    spy_df = df[df[SYMBOL_COLUMN] == "SPY"].set_index(DATETIME_COLUMN)
    spy_returns = spy_df["return_1h"].rename("spy_return")

    df = df.set_index(DATETIME_COLUMN)
    df = df.join(spy_returns, on=DATETIME_COLUMN)
    df["spread_vs_spy"] = df["return_1h"] - df["spy_return"]
    df["corr_5d_spy"] = (
        df["return_1h"].rolling(window=5).corr(df["spy_return"]).astype(np.float32)
    )
    return df.drop(columns=["spy_return"]).reset_index()


def generate_reports(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Generate evaluation metrics and visualizations.

    Save them to the output folder.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    """
    Logger.info("Generating classification and F1-score reports...")
    os.makedirs(_EVALUATION_REPORT_BASEPATH, exist_ok=True)

    conf_matrix_filepath = os.path.join(
        _EVALUATION_REPORT_BASEPATH, _CONF_MATRIX_PLOT_FILEPATH
    )
    f1_score_filepath = os.path.join(
        _EVALUATION_REPORT_BASEPATH, _F1_SCORE_PLOT_FILEPATH
    )

    report = classification_report(
        y_true, y_pred, target_names=["Down", "Neutral", "Up"]
    )
    Logger.debug(report)

    report_dict = classification_report(
        y_true, y_pred, output_dict=True, target_names=["Down", "Neutral", "Up"]
    )

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=_CONFUSION_MATRIX_FIGSIZE)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Down", "Neutral", "Up"],
        yticklabels=["Down", "Neutral", "Up"],
    )
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(conf_matrix_filepath)
    Logger.success(f"Confusion matrix saved at {conf_matrix_filepath}")

    f1_scores = [report_dict[label]["f1-score"] for label in ["Down", "Neutral", "Up"]]
    plt.figure(figsize=_F1_SCORE_FIGSIZE)
    plt.bar(["Down", "Neutral", "Up"], f1_scores)

    Plots.format_f1_score_plot("F1-Score per Class")
    plt.savefig(f1_score_filepath)
    Logger.success(f"F1-score plot saved at {f1_score_filepath}")


def evaluate_model():
    """
    Main evaluation pipeline for multi-class market model.

    Loads model, processes data, performs prediction and saves evaluation metrics.
    """
    Logger.info("🚀 Starting model evaluation...")
    model = load_model()

    df = load_and_process_data(_ALL_SYMBOLS)
    if df.empty:
        Logger.error("Evaluation aborted: no data available.")
        return

    df = engineer_cross_features(df)
    df.dropna(
        subset=_FEATURES + CROSS_FEATURES + CATEGORICAL_FEATURES + [TARGET],
        inplace=True,
    )

    if df.empty:
        Logger.error(
            "Evaluation aborted: no data left after dropping NaNs in required columns."
        )
        return

    x_eval = df[_FEATURES + CROSS_FEATURES + CATEGORICAL_FEATURES]
    y_true = df[TARGET].astype(int)

    Logger.info("📈 Running predictions...")
    y_pred = model.predict(x_eval)

    Logger.success("✅ Evaluation completed. Compiling reports...")
    generate_reports(y_true, y_pred)


if __name__ == "__main__":
    evaluate_model()
