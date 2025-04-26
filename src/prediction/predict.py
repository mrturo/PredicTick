"""
Core utilities shared by prediction pipelines.

Includes:
- PredictionResult dataclass
- ArtifactLoader for loading persisted models/scalers
- Handler for centralized market data access
- Common utilities like expected return calculation and prediction thresholding
"""

import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from market_data.gateway import Gateway
from training.data_preparation import engineer_cross_features
from utils.calendar_manager import CalendarManager
from utils.feature_engineering import FeatureEngineering
from utils.logger import Logger
from utils.parameters import ParameterLoader

# Constants
SYMBOL_COLUMN = "symbol"
DATETIME_COLUMN = "datetime"
TARGET_COLUMN = "target"

# Centralized parameter object
_PARAMS = ParameterLoader(Gateway.get_last_update())
_FED_EVENT_DAYS = _PARAMS.get("fed_event_days")
_GROUP_1_SYMBOLS = _PARAMS.get("group-1")
_GROUP_2_SYMBOLS = _PARAMS.get("group-2")
_FEATURES = _PARAMS.get("features")
_CORRELATIVE_SYMBOLS = _PARAMS.get("correlative_symbols")
_UP_THRESHOLD = _PARAMS.get("up_threshold")
_DOWN_THRESHOLD = _PARAMS.get("down_threshold")
_PREDICTION_WORKERS = _PARAMS.get("prediction_workers")
_MODEL_FILEPATH = _PARAMS.get("model_filepath")
_SYMBOL_COLUMN = "symbol"


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
            full_filepath = ArtifactLoader._validate_filepath(filepath, suffix)
            return joblib.load(full_filepath)
        except Exception as e:
            Logger.error(f"Failed to load artifact from {full_filepath}: {e}")
            raise

    @staticmethod
    def exists(filepath: str, suffix: str = "") -> bool:
        """Checks if a suffixed artifact path exists."""
        full_filepath = ArtifactLoader._validate_filepath(filepath, suffix)
        return os.path.exists(full_filepath)

    @staticmethod
    def _validate_filepath(filepath: str, suffix: str) -> str:
        """Constructs a suffixed path and checks existence."""
        base, ext = os.path.splitext(filepath)
        return f"{base}{suffix}{ext}" if suffix else filepath


class Handler:
    """Provides access to updated market data."""

    @staticmethod
    def load_market_data() -> dict:
        """Loads latest market data and organizes by symbol."""
        market_data = Gateway.load()["symbols"]
        if not market_data:
            Logger.error("Market data is empty. Cannot proceed.")
            raise ValueError("Market data is empty.")
        return Gateway.load()["symbols"]

    @staticmethod
    def has_data() -> bool:
        """Checks if current market data is non-empty."""
        return bool(Gateway.load()["symbols"])


def get_next_business_day(current_date: datetime.date) -> datetime.date:
    """Returns the next valid business day after the given date."""
    return (pd.Timestamp(current_date) + BDay(1)).date()


def calculate_expected_return(df: pd.DataFrame) -> float:
    """Estimates expected return using ATR and latest close."""
    try:
        last_atr = df["atr"].iloc[-1]
        last_close = df["close"].iloc[-1]
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
    _, us_holidays, fed_event_days = CalendarManager.build_market_calendars(
        _FED_EVENT_DAYS
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
            r for r in records if pd.to_datetime(r["datetime"]).date() <= current_date
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


# pylint: disable=too-few-public-methods
class SymbolPredictorConfig:
    """Encapsulates configuration parameters for the predictor."""

    def __init__(self):
        self.model = ArtifactLoader.load(_MODEL_FILEPATH)


class SymbolPredictor:
    """Handles predictions across financial symbols using cross-feature modeling."""

    def __init__(self, market_data: Optional[dict] = None):
        self.config = SymbolPredictorConfig()
        self.market_data_dict = market_data or Handler.load_market_data()

    def prepare_features(
        self, symbol: str, current_date: Optional[datetime.date]
    ) -> Optional[pd.DataFrame]:
        """Prepares and enriches feature dataframe for a single symbol."""
        records = filter_market_data_by_symbol(
            self.market_data_dict, symbol, current_date
        )
        if not records:
            return None

        # Ensure all records are dicts
        records = [r for r in records if isinstance(r, dict)]
        if not records:
            return None

        df = pd.DataFrame(records)
        if df.empty:
            return None

        df = FeatureEngineering.prepare_raw_dataframe(df, symbol)

        return df.reset_index()

    def engineer_cross_features(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """Adds cross-symbol features using Trainer."""
        return engineer_cross_features(full_df, _CORRELATIVE_SYMBOLS)

    def _prepare_batch_data(
        self,
        symbols: List[str],
        current_date: Optional[datetime.date],
    ) -> pd.DataFrame:
        """Loads and concatenates prepared DataFrames for each symbol in parallel."""
        processed = []

        def process(symbol):
            df = self.prepare_features(symbol, current_date)
            return df if df is not None and not df.empty else None

        with ThreadPoolExecutor(max_workers=_PREDICTION_WORKERS) as executor:
            futures = {executor.submit(process, s): s for s in symbols}
            for future in as_completed(futures):
                df = future.result()
                if df is not None:
                    processed.append(df)

        return pd.concat(processed) if processed else pd.DataFrame()

    def predict_next_day(
        self,
        symbols: Optional[List[str]] = None,
        current_date: Optional[datetime.date] = None,
    ) -> Optional[PredictionResult]:
        """Predicts the best symbol with a strong UP signal for the next business day."""
        symbols = symbols or list(self.market_data_dict.keys())
        combined_df = self._prepare_batch_data(symbols, current_date)
        if combined_df.empty:
            return None

        combined_df = self.engineer_cross_features(combined_df)
        latest_rows, feature_cols = self._build_feature_matrix(combined_df)
        if latest_rows.empty:
            return None

        input_df = latest_rows[feature_cols]
        predictions = self.config.model.predict_proba(input_df)

        results = self._evaluate_predictions(latest_rows, predictions)
        return max(results, key=lambda r: r.expected_return_score) if results else None

    def print_prediction(self, label: str, symbols: Optional[List[str]] = None):
        """Prints the best UP signal with probability and score."""
        print_prediction_result(label, self.predict_next_day, symbols)

    def _build_feature_matrix(self, combined_df: pd.DataFrame) -> tuple:
        """Filters rows and builds the input feature matrix."""
        cross_features = [
            col
            for col in combined_df.columns
            if col.startswith("spread_vs_") or col.startswith("corr_5d_")
        ]
        all_features = _FEATURES + cross_features
        feature_cols = [col for col in all_features if col in combined_df.columns]

        filtered = combined_df[
            combined_df[feature_cols].notna().sum(axis=1)
            >= int(len(feature_cols) * 0.8)
        ]
        latest_rows = (
            filtered.groupby(_SYMBOL_COLUMN).tail(1).dropna(subset=feature_cols)
        )
        return latest_rows, feature_cols

    def _evaluate_predictions(
        self, latest_rows: pd.DataFrame, predictions: np.ndarray
    ) -> List[PredictionResult]:
        """Generates PredictionResult list for symbols with UP signal."""
        results = []
        for (_, row), probs in zip(latest_rows.iterrows(), predictions):
            if probs[2] <= _UP_THRESHOLD and probs[0] <= _DOWN_THRESHOLD:
                continue
            prediction = custom_predict_class(probs, _UP_THRESHOLD, _DOWN_THRESHOLD)
            if prediction["value"] != 1:
                continue
            expected_return = calculate_expected_return(row.to_frame().T)
            score = expected_return * (probs[2] - max(probs[0], probs[1]))
            results.append(
                PredictionResult(
                    symbol=row[_SYMBOL_COLUMN],
                    next_date=get_next_business_day(row["date"]),
                    direction="UP",
                    expected_return_score=score,
                )
            )
        return results


if __name__ == "__main__":
    predictor = SymbolPredictor()
    predictor.print_prediction("group-1", _GROUP_1_SYMBOLS)
    predictor.print_prediction("group-2", _GROUP_2_SYMBOLS)
