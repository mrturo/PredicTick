"""
Core utilities shared by prediction pipelines.

Includes:
- PredictionResult dataclass
- ArtifactLoader for loading persisted models/scalers
- MarketDataHandler for centralized market data access
- Common utilities like expected return calculation and prediction thresholding
"""

import datetime
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from updater.data_updater import FileManager
from utils.calendar import build_market_calendars
from utils.logger import Logger
from utils.parameters import ParameterLoader

# Constants
SYMBOL_COLUMN = "symbol"
DATETIME_COLUMN = "Datetime"
TARGET_COLUMN = "target"

# Centralized parameter object
parameters = ParameterLoader(FileManager.last_update())


@dataclass
class PredictionResult:
    """Represents the output of a model prediction for a symbol."""

    symbol: str
    next_date: datetime.date
    direction: str
    expected_return_score: float


class ArtifactLoader:
    """Helper class for loading model or scaler artifacts with variant suffixes."""

    @staticmethod
    def load(filepath: str, suffix: str = "") -> Any:
        """Loads artifact with suffix."""
        try:
            full_path = ArtifactLoader._validate_filepath(filepath, suffix)
            return joblib.load(full_path)
        except Exception as e:
            Logger.error(f"Failed to load artifact from {full_path}: {e}")
            raise

    @staticmethod
    def exists(filepath: str, suffix: str = "") -> bool:
        """Checks if a suffixed artifact path exists."""
        full_path = ArtifactLoader._validate_filepath(filepath, suffix)
        return os.path.exists(full_path)

    @staticmethod
    def _validate_filepath(filepath: str, suffix: str) -> str:
        """Constructs a suffixed path and checks existence."""
        base, ext = os.path.splitext(filepath)
        return f"{base}{suffix}{ext}" if suffix else filepath


class MarketDataHandler:
    """Provides access to updated market data."""

    @staticmethod
    def load_market_data() -> dict:
        """Loads latest market data and organizes by symbol."""
        market_data = FileManager.load()
        if not market_data:
            Logger.error("Market data is empty. Cannot proceed.")
            raise ValueError("Market data is empty.")
        return {entry["symbol"]: entry["historical_prices"] for entry in market_data}

    @staticmethod
    def has_data() -> bool:
        """Checks if current market data is non-empty."""
        return bool(FileManager.load())


def get_next_business_day(current_date: datetime.date) -> datetime.date:
    """Returns the next valid business day after the given date."""
    return (pd.Timestamp(current_date) + BDay(1)).date()


def calculate_expected_return(df: pd.DataFrame) -> float:
    """Estimates expected return using ATR and latest close."""
    try:
        last_atr = df["atr"].iloc[-1]
        last_close = df["Close"].iloc[-1]
        return (
            0.0
            if last_close == 0 or pd.isna(last_atr) or pd.isna(last_close)
            else last_atr / last_close
        )
    except (KeyError, IndexError, ValueError) as e:
        Logger.error(f"Failed to calculate expected return: {e}")
        return 0.0


def custom_predict_class(
    probabilities: np.ndarray, up_threshold: float, down_threshold: float
) -> dict:
    """Classifies the signal as UP or DOWN based on thresholded probability vector."""
    return (
        {"value": 1, "probabilities": probabilities[2]}
        if probabilities[2] > up_threshold
        else (
            {"value": 0, "probabilities": probabilities[0]}
            if probabilities[0] > down_threshold
            else {"value": 1, "probabilities": probabilities[2]}
        )
    )


def get_market_metadata() -> tuple:
    """Loads US market holidays and FED event days from calendar utilities."""
    _, us_holidays, fed_event_days = build_market_calendars(
        parameters["fed_event_days"]
    )
    return us_holidays, fed_event_days


def filter_market_data_by_symbol(
    market_data_dict: dict, symbol: str, current_date: Optional[datetime.date] = None
) -> Optional[List[dict]]:
    """Filters symbol data from market_data_dict up to a specific date."""
    if symbol not in market_data_dict:
        return None

    records = market_data_dict[symbol]
    if current_date:
        records = [
            r for r in records if pd.to_datetime(r["Datetime"]).date() <= current_date
        ]

    return records if records else None


def print_prediction_result(
    label: str,
    prediction_fn: Callable[[Optional[List[str]]], Optional[PredictionResult]],
    symbols: Optional[List[str]] = None,
):
    """Unified printer for prediction results with timing."""
    start = time.perf_counter()
    prediction = prediction_fn(symbols)
    elapsed = time.perf_counter() - start

    if prediction:
        if prediction.next_date < datetime.date.today():
            Logger.warning(f"{label} | Prediction is outdated: {prediction.next_date}")
        else:
            Logger.success(
                f"{label} | Best Symbol for {prediction.next_date}: {prediction.symbol} → "
                f"{prediction.direction} (Score: {prediction.expected_return_score:.2f}) "
                f"| Time: {elapsed:.2f}s"
            )
    else:
        Logger.warning(f"{label} | No valid prediction found. | Time: {elapsed:.2f}s")
