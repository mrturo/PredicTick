"""
Feature engineering utilities for financial market models.

Provides centralized logic to enrich OHLCV dataframes with technical indicators,
session-based flags, calendar placeholders, and target label generation
for multi-class classification tasks (Down, Neutral, Up).
"""

from typing import Optional

import pandas as pd # type: ignore

from utils.parameters import ParameterLoader


class FeatureEngineering:
    """Applies domain-specific transformations and target labeling to financial time series."""

    _PARAMS = ParameterLoader()
    _DEFAULT_CLASSIFICATION_THRESHOLD = _PARAMS.get("classification_threshold")
    _DEFAULT_PERIODS_AHEAD = _PARAMS.get("classification_periods_ahead")

    @staticmethod
    def enrich_with_common_features(
        df: pd.DataFrame, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Applies common feature engineering steps including:

        * Technical indicators
        * Session flags
        * Holiday/fed placeholders
        * Target labeling

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
            symbol (str): Optional symbol to tailor features.

        Returns:
            pd.DataFrame: Enriched DataFrame with new features.
        """
        df = df.copy()
        df.sort_index(inplace=True)

        if symbol:
            df["symbol"] = symbol

        if "datetime" in df.columns:
            df.set_index("datetime", inplace=True)

        # Ensure index is a DatetimeIndex before extracting hour and minute
        # if not isinstance(df.index, pd.DatetimeIndex):
        #     df.index = pd.to_datetime(df.index)
        # df["hour"] = df.index.hour + df.index.minute / 60

        # df["date"] = df.index.date
        # daily_close = df.groupby("date")["close"].last()
        # prev_map = {
        #    k2: k1 for k1, k2 in zip(daily_close.index, list(daily_close.index)[1:])
        # }
        # df["prev_day"] = df["date"].map(prev_map)
        # df["close_yesterday"] = df["prev_day"].map(daily_close.to_dict())
        # df["today_close"] = df["date"].map(daily_close.to_dict())

        df["target"] = FeatureEngineering.classify_target(df).astype("Int8")

        return df

    @staticmethod
    def prepare_raw_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Standardizes timestamp parsing, indexing and calls enrich_with_common_features.

        Args:
            df (pd.DataFrame): Raw input dataframe.
            symbol (str): Symbol to assign.

        Returns:
            pd.DataFrame: Preprocessed and enriched dataframe.
        """
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        df["symbol"] = symbol

        return FeatureEngineering.enrich_with_common_features(df, symbol)

    @staticmethod
    def classify_target(
        df: pd.DataFrame,
        forward_col: str = "close",
        periods_ahead: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> pd.Series:
        """
        Classifies returns into 3 categories based on thresholds.

        Args:
            df (pd.DataFrame): Input DataFrame with price data.
            forward_col (str): Column used to compute future return.
            periods_ahead (int): Horizon in periods to compute forward return.
            threshold (float): Threshold for classifying Up/Down/Neutral.

        Returns:
            pd.Series: Series of integer labels (0=Down, 1=Neutral, 2=Up).
        """

        periods_ahead = (
            periods_ahead
            if periods_ahead is not None
            else FeatureEngineering._DEFAULT_PERIODS_AHEAD
        )
        if periods_ahead is None:
            periods_ahead = 1

        threshold = (
            threshold
            if threshold is not None
            else FeatureEngineering._DEFAULT_CLASSIFICATION_THRESHOLD
        )

        future = df[forward_col].shift(-periods_ahead)
        returns = (future - df[forward_col]) / df[forward_col]

        def label_fn(x: float) -> Optional[int]:
            if pd.isna(x) or threshold is None:
                return None
            if x < -threshold:
                return 0
            if x > threshold:
                return 2
            return 1

        return returns.astype("float32").apply(label_fn)
