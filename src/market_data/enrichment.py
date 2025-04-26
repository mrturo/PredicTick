"""Module for enriching market price data with technical and temporal features."""

import numpy as np
import pandas as pd
import ta

from market_data.schedule_builder import ScheduleBuilder
from utils.logger import Logger
from utils.parameters import ParameterLoader


class Enrichment:
    """Class responsible for enriching market data with technical and temporal features.

    This includes calculating indicators such as returns, volatility, volume metrics,
    and pattern recognition, as well as calendar-based progress features.
    """

    _PARAMS = ParameterLoader()
    _VOLUME_WINDOW = _PARAMS.get("volume_window")

    @staticmethod
    def enrich_historical_prices(df: pd.DataFrame) -> pd.DataFrame:
        """Enriches historical prices with computed features."""
        if df.empty or any(
            col not in df.columns for col in ["close", "open", "volume"]
        ):
            Logger.warning(
                "    Cannot enrich historical prices: essential columns missing."
            )
            return df

        df = df.sort_values("datetime").copy()
        df.reset_index(drop=True, inplace=True)

        # Interpolate or forward-fill volume = 0
        if "volume" in df.columns:
            zero_volume_mask = df["volume"] == 0
            if zero_volume_mask.any():
                Logger.warning(
                    "    Detected zero volume entries. Applying interpolation."
                )
                df["volume"] = df["volume"].replace(0, pd.NA)
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                df["volume"] = df["volume"].infer_objects(copy=False)
                df["volume"] = df["volume"].interpolate(
                    method="linear", limit_direction="forward"
                )
                df["volume"] = df["volume"].bfill()
                df["volume"] = df["volume"].infer_objects(copy=False)

        # Basic features
        df["return"] = df["close"].pct_change(fill_method=None)
        df["volatility"] = (df["high"] - df["low"]) / df["open"].replace(0, pd.NA)
        df["price_change"] = df["close"] - df["open"]
        df["volume_change"] = df["volume"].pct_change(fill_method=None)
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["average_price"] = (df["high"] + df["low"]) / 2
        epsilon = 1e-6  # Ajusta según la escala de tus datos
        conditions = [
            (df["close"] - df["open"]) > epsilon,
            (df["open"] - df["close"]) > epsilon,
        ]
        choices = [1, -1]
        df["candle"] = np.select(conditions, choices, default=0)
        df["range"] = df["high"] - df["low"]
        df["relative_volume"] = (
            df["volume"]
            / df["volume"]
            .rolling(window=Enrichment._VOLUME_WINDOW, min_periods=1)
            .mean()
        )

        min_required_rows = 252
        if len(df) < min_required_rows:
            Logger.warning(
                f"    Skipping technical indicators: insufficient data (has {len(df)} "
                f"rows, needs {min_required_rows})."
            )
            return df

        # New features
        df["atr_14"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).average_true_range()

        df["overnight_return"] = df["open"].pct_change(fill_method=None).fillna(0)
        df["intraday_return"] = (df["close"] / df["open"]) - 1

        df["volume_rvol_20d"] = (
            df["volume"] / df["volume"].rolling(window=20, min_periods=1).mean()
        )
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=df["close"], volume=df["volume"]
        ).on_balance_volume()

        adx = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=14
        )
        df["adx_14"] = adx.adx()

        boll = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bollinger_pct_b"] = boll.bollinger_pband()

        df["pattern_bullish_engulfing"] = (
            (df["close"] > df["open"])
            & (df["close"].shift(1) < df["open"].shift(1))
            & (df["close"] > df["open"].shift(1))
            & (df["open"] < df["close"].shift(1))
        ).astype(int)
        df["pattern_bearish_engulfing"] = (
            (df["close"] < df["open"])
            & (df["close"].shift(1) > df["open"].shift(1))
            & (df["open"] > df["close"].shift(1))
            & (df["close"] < df["open"].shift(1))
        ).astype(int)

        max_52w = df["close"].rolling(window=252, min_periods=1).max()
        df["pct_from_52w_high"] = (max_52w - df["close"]) / max_52w

        df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

        Logger.debug("    Enriched historical prices with additional metadata.")
        return df

    @staticmethod
    def add_month_and_year_days(
        df: pd.DataFrame, calendar_type: int = 7
    ) -> pd.DataFrame:
        """Add month/day progress tracking columns using calendar type logic."""
        df = df.copy()
        df["local_datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["local_date"] = df["local_datetime"].dt.date
        df["local_year"] = df["local_datetime"].dt.year
        df["local_month"] = df["local_datetime"].dt.month

        unique_dates = (
            df[["local_year", "local_month", "local_date"]]
            .drop_duplicates()
            .sort_values("local_date")
        )

        month_day_map, year_day_map = ScheduleBuilder.build_day_index_maps(
            unique_dates, calendar_type
        )

        unique_dates["local_month_days_current"] = unique_dates.apply(
            lambda row: month_day_map.get(
                (row["local_year"], row["local_month"]), {}
            ).get(row["local_date"], np.nan),
            axis=1,
        )
        unique_dates["local_year_days_current"] = unique_dates.apply(
            lambda row: year_day_map.get(row["local_year"], {}).get(
                row["local_date"], np.nan
            ),
            axis=1,
        )

        merged = ScheduleBuilder.merge_and_estimate_totals(unique_dates, calendar_type)

        df = df.merge(
            merged, on=["local_year", "local_month", "local_date"], how="left"
        )
        df["month_days_current"] = df["local_month_days_current"]
        df["month_days_total"] = df["local_month_days_total"]
        df["year_days_current"] = df["local_year_days_current"]
        df["year_days_total"] = df["local_year_days_total"]
        df.drop(
            columns=[
                "local_year",
                "local_month",
                "local_date",
                "local_datetime",
                "local_month_days_current",
                "local_month_days_total",
                "local_year_days_current",
                "local_year_days_total",
            ],
            inplace=True,
        )
        return df
