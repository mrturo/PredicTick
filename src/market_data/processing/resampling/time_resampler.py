"""Module for resampling time series data from higher to lower frequency intervals.

This module defines the TimeResampler class, which provides static methods to resample OHLCV
(Open, High, Low, Close, Volume, Adj Close) financial time series data from a higher-frequency
interval to a lower-frequency interval using consistent datetime-based aggregation windows.
"""

from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.api.types import is_list_like  # type: ignore

from src.market_data.utils.intervals.interval import IntervalConverter


# pylint: disable=too-few-public-methods
class TimeResampler:
    """Resamples data to lower-frequency intervals using consistent datetime-based aggregation."""

    @staticmethod
    def by_ratio(
        from_df: pd.DataFrame,
        from_interval: str,
        to_interval: str,
        date_time: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Aggregate a DataFrame of OHLCV candles to a lower-frequency interval."""
        df = from_df.sort_values("datetime").copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        from_minutes = IntervalConverter.to_minutes(from_interval)
        to_minutes = IntervalConverter.to_minutes(to_interval)
        if to_minutes == from_minutes:
            return df
        if to_minutes % from_minutes != 0:
            raise ValueError(
                f"To interval ({to_interval}) must be divisible by from interval ({from_interval})."
            )
        window_delta = timedelta(minutes=to_minutes)
        windows = TimeResampler._generate_windows(df, window_delta, date_time)
        aggregated = [
            TimeResampler._aggregate_window(df, start, start + window_delta)
            for start in windows
        ]
        to_df = pd.DataFrame([row for row in aggregated if row is not None])
        return to_df

    @staticmethod
    def _to_utc_datetime_series(
        obj: Union[
            None, pd.Series, pd.Index, Iterable, datetime, int, float, np.number, str
        ],
    ) -> pd.Series:
        """Helper: convert *obj* to a tz-aware Series[datetime64[ns, UTC]] with NaTs removed.

        Returns an **empty** Series if the input is None or all values are invalid.
        """
        if obj is None:
            return pd.Series(dtype="datetime64[ns, UTC]")
        # Wrap scalars in a list so pd.to_datetime always returns a Series/Index
        if not is_list_like(obj) or isinstance(obj, (str, bytes)):
            obj = [obj]
        dt = pd.to_datetime(obj, utc=True, errors="coerce")
        # Ensure we end up with a Series for uniform processing
        if isinstance(dt, pd.DatetimeIndex):
            dt = dt.to_series(index=range(len(dt)))
        return dt.dropna()

    @staticmethod
    def _normalize_window_delta(
        delta: Union[timedelta, np.timedelta64, int, float],
    ) -> timedelta:
        """Ensures *delta* is a plain ``datetime.timedelta``.

        * int/float is interpreted as **seconds**.
        * np.timedelta64 is converted via ``pd.Timedelta`` for better precision.
        """
        if isinstance(delta, timedelta):
            return delta
        if isinstance(delta, np.timedelta64):
            return pd.Timedelta(delta).to_pytimedelta()
        if isinstance(delta, (int, float, np.integer, np.floating)):
            return timedelta(seconds=float(delta))
        raise TypeError(f"Unsupported window_delta type: {type(delta)}")

    @staticmethod
    def _generate_windows(
        df: pd.DataFrame,
        window_delta: Union[timedelta, np.timedelta64, int, float],
        anchor: Optional[
            Union[pd.Series, pd.Index, Iterable, datetime, int, float, np.number, str]
        ],
    ) -> list[pd.Timestamp]:
        """Build a *non-empty* list of window-start timestamps."""
        window_delta = TimeResampler._normalize_window_delta(window_delta)
        dt_col = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dropna()
        if dt_col.empty:
            raise ValueError("'datetime' column contains only NaT values")
        start_time: pd.Timestamp = dt_col.iloc[0]
        end_time: pd.Timestamp = dt_col.iloc[-1]
        anchor_series = TimeResampler._to_utc_datetime_series(anchor)
        if not anchor_series.empty:
            anchor_time: pd.Timestamp = anchor_series.iloc[-1]
            diff_seconds = (anchor_time - start_time).total_seconds()
            delta_seconds = window_delta.total_seconds()
            offset_seconds = diff_seconds % delta_seconds
            first_window = (
                start_time + timedelta(seconds=offset_seconds)
                if offset_seconds
                else start_time
            )
            windows: List[pd.Timestamp] = list(
                pd.date_range(
                    start=first_window,
                    end=anchor_time - window_delta,
                    freq=window_delta,
                )
            )
            if not windows:
                windows = [first_window]
        else:
            windows = list(
                pd.date_range(start=start_time, end=end_time, freq=window_delta)
            )
            if not windows:
                windows = [start_time]
        return windows

    @staticmethod
    def _aggregate_window(
        df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
    ) -> Optional[dict]:
        """Aggregate a time window into a single OHLCV bar."""
        block = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        if block.empty:
            return None
        return {
            "datetime": block["datetime"].iloc[0],
            "open": block["open"].iloc[0],
            "high": block["high"].max(),
            "low": block["low"].min(),
            "close": block["close"].iloc[-1],
            "volume": block["volume"].sum(),
            "adj_close": block["adj_close"].iloc[-1],
        }
