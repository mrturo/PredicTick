"""Utilities for applying temporal cutoff filters to financial time series data."""

from typing import Optional

import pandas as pd  # type: ignore


def apply_cutoff_filters(
    df: pd.DataFrame,
    cutoff_from: Optional[pd.Timestamp],
    cutoff_to: Optional[pd.Timestamp],
    timezone: str,
    cutoff_minutes: int,
) -> pd.DataFrame:
    """Apply date/time filtering to a DataFrame based on cutoff constraints.

    Filters include:
    - Converting to timezone-aware datetimes.
    - Filtering between optional start and end timestamps.
    - Removing rows too close to the present based on `cutoff_minutes`.
    """
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    df = df.tz_convert(timezone).sort_index()
    df["date"] = df.index.to_series().dt.date
    if cutoff_from is not None:
        df = df[df.index > cutoff_from]
    if cutoff_to is not None:
        df = df[df.index <= cutoff_to]
    now = pd.Timestamp.now(tz=timezone)
    df = df[
        df.index.to_series().apply(
            lambda idx: (now - idx).total_seconds() > cutoff_minutes * 60
        )
    ]
    df.reset_index(inplace=True)
    return df
