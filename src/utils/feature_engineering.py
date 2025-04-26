"""
Feature engineering utilities for financial market models.

Provides centralized logic to enrich OHLCV dataframes with technical indicators,
session-based flags, calendar placeholders, and target label generation
for multi-class classification tasks (Down, Neutral, Up).
"""

import pandas as pd

from utils.indicators import IndicatorCalculator
from utils.parameters import ParameterLoader


class FeatureEngineering:
    """Applies domain-specific transformations and target labeling to financial time series."""

    _PARAMS = ParameterLoader()
    _DEFAULT_CLASSIFICATION_THRESHOLD = _PARAMS.get("classification_threshold")
    _DEFAULT_PERIODS_AHEAD = _PARAMS.get("classification_periods_ahead")
    _PRE_MARKET_END = _PARAMS.get("pre_market_end")
    _POST_MARKET_START = _PARAMS.get("post_market_start")

    @staticmethod
    def enrich_with_common_features(
        df: pd.DataFrame, symbol: str = None
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

        df = IndicatorCalculator.compute_technical_indicators(df)

        if "datetime" in df.columns:
            df.set_index("datetime", inplace=True)

        df["hour"] = df.index.hour + df.index.minute / 60
        df["is_pre_market"] = (
            (df["hour"] >= 0) & (df["hour"] < FeatureEngineering._PRE_MARKET_END)
        ).astype(int)
        df["is_post_market"] = (
            (df["hour"] >= FeatureEngineering._POST_MARKET_START) & (df["hour"] <= 24)
        ).astype(int)
        df["day_of_week"] = df.index.dayofweek
        df["is_fed_day"] = 0
        df["is_holiday"] = 0
        df["is_before_holiday"] = 0

        df["date"] = df.index.date
        daily_close = df.groupby("date")["close"].last()
        prev_map = {
            k2: k1 for k1, k2 in zip(daily_close.index, list(daily_close.index)[1:])
        }
        df["prev_day"] = df["date"].map(prev_map)
        df["close_yesterday"] = df["prev_day"].map(daily_close.to_dict())
        df["today_close"] = df["date"].map(daily_close.to_dict())

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
        periods_ahead: int = None,
        threshold: float = None,
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
        threshold = (
            threshold
            if threshold is not None
            else FeatureEngineering._DEFAULT_CLASSIFICATION_THRESHOLD
        )

        future = df[forward_col].shift(-periods_ahead)
        returns = (future - df[forward_col]) / df[forward_col]

        def label_fn(x: float) -> int:
            if pd.isna(x):
                return pd.NA
            if x < -threshold:
                return 0
            if x > threshold:
                return 2
            return 1

        return returns.astype("float32").apply(label_fn)
