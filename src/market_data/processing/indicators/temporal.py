"""Temporal‑context features for machine‑learning pipelines.

The functions in this module create **calendar‑based flags** and **continuous
fractions of time** that often prove valuable in intraday and daily models.
They are completely data‑driven and require only a timestamp column in the
input *DataFrame*.
"""

from __future__ import annotations

import datetime as _dt
from typing import Final, Set, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.utils.io.logger import Logger

__all__: Final[list[str]] = [
    "compute_temporal_event_feature",
    "compute_weekday",
    "compute_weekend",
    "compute_workday",
    "compute_time_fractions",
]


def _compute_time_month(
    date_times: pd.Series,
) -> Tuple[pd.Series, pd.Series]:  # noqa: D401
    """Return month‑elapsed fraction and *Timedelta* total for each sample."""
    month_start = date_times.dt.tz_localize(None).dt.to_period("M").dt.start_time
    next_month_start = month_start + pd.offsets.MonthBegin(1)
    month_start = month_start.dt.tz_localize(date_times.dt.tz)
    next_month_start = next_month_start.dt.tz_localize(date_times.dt.tz)
    month_elapsed = (date_times - month_start).dt.total_seconds()
    month_total = next_month_start - month_start
    month_total_seconds = month_total.dt.total_seconds()
    return month_elapsed / month_total_seconds, month_total


def _compute_time_year(
    date_times: pd.Series,
) -> Tuple[pd.Series, pd.Series]:  # noqa: D401
    """Return year‑elapsed fraction and *Timedelta* total for each sample."""
    year_start = date_times.dt.tz_localize(None).dt.to_period("Y").dt.start_time
    next_year_start = year_start + pd.offsets.YearBegin(1)
    year_start = year_start.dt.tz_localize(date_times.dt.tz)
    next_year_start = next_year_start.dt.tz_localize(date_times.dt.tz)
    year_elapsed = (date_times - year_start).dt.total_seconds()
    year_total = next_year_start - year_start
    year_total_seconds = year_total.dt.total_seconds()
    return year_elapsed / year_total_seconds, year_total


def compute_temporal_event_feature(
    df: pd.DataFrame,
    event_dates: Set[_dt.date],
    is_raw: bool = False,
) -> dict[str, pd.Series]:
    """Add decaying proximity features for a set of *event_dates*.

    Three vectors are generated:
    * is → 1.0 (or *True*) on the event date.
    * is_pre → linear decay *before* the event within a ±window.
    * is_post → linear decay *after* the event within the same window.

    The decay follows ``(window - k + 1) / window`` where *k* is the distance
    in *days* to the nearest event date.
    """
    window: int = 5  # noqa: WPS432 – centralised here; could be parameterised
    try:
        dates = df["datetime"].dt.date
        # Exact match
        is_event = dates.isin(event_dates).astype("float32")
        pre_decay = pd.Series(0.0, index=df.index, dtype="float32")
        post_decay = pd.Series(0.0, index=df.index, dtype="float32")
        for offset in range(1, window + 1):
            weight: float = (window - offset + 1) / window
            pre_mask = (dates + _dt.timedelta(offset)).isin(event_dates)
            pre_decay = pre_decay.where(~pre_mask, weight)
            post_mask = (dates - _dt.timedelta(offset)).isin(event_dates)
            post_decay = post_decay.where(~post_mask, weight)
        if is_raw:
            return {
                "is_pre": pre_decay.astype("float32") * 100,
                "is": is_event == 1,
                "is_post": post_decay.astype("float32") * 100,
            }
        return {
            "is_pre": pre_decay,
            "is": is_event,
            "is_post": post_decay,
        }
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_temporal_event_feature] failure: {exc}")
        na_float = pd.Series(np.nan, index=df.index, dtype="float32")
        na_bool = pd.Series(pd.NA, index=df.index, dtype="boolean")
        return {
            "is_pre": na_bool if is_raw else na_float,
            "is": na_bool if is_raw else na_float,
            "is_post": na_bool if is_raw else na_float,
        }


def compute_weekday(date_times: pd.Series, is_raw: bool = False) -> pd.Series:
    """Flag Monday–Friday."""
    try:
        dt = pd.to_datetime(date_times)
        is_weekday = dt.dt.weekday < 5
        if is_raw:
            return is_weekday.astype("boolean")
        return is_weekday.astype("float32")
    except Exception as exc:  # noqa: BLE001 # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_weekday] failure: {exc}")
        return (
            pd.Series(pd.NA, dtype="boolean")
            if is_raw
            else pd.Series(np.nan, dtype="float32")
        )


def compute_weekend(date_times: pd.Series, is_raw: bool = False) -> pd.Series:
    """Flag Saturday–Sunday."""
    try:
        dt = pd.to_datetime(date_times)
        is_weekend = dt.dt.weekday > 4
        if is_raw:
            return is_weekend.astype("boolean")
        return is_weekend.astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_weekend] failure: {exc}")
        return (
            pd.Series(pd.NA, dtype="boolean")
            if is_raw
            else pd.Series(np.nan, dtype="float32")
        )


def compute_workday(
    in_is_weekend: pd.Series,
    in_is_holiday: pd.Series,
    is_raw: bool = False,
) -> pd.Series:
    """Compute (¬weekend) ∧ (¬holiday)."""
    try:
        if is_raw:
            is_weekend = in_is_weekend.astype("boolean")
            is_holiday = in_is_holiday.astype("boolean")
        else:
            is_weekend = in_is_weekend.astype("float32") == 1
            is_holiday = in_is_holiday.astype("float32") == 1
        is_workday = (~is_weekend) & (~is_holiday)
        return is_workday.astype("boolean" if is_raw else "float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_workday] failure: {exc}")
        return (
            pd.Series(pd.NA, dtype="boolean")
            if is_raw
            else pd.Series(np.nan, dtype="float32")
        )


def compute_time_fractions(
    df: pd.DataFrame, is_raw: bool = False
) -> dict[str, pd.Series]:
    """Normalised fractions of day, week, month and year for each timestamp."""
    try:
        date_times = df["datetime"]
        dt = date_times.dt
        # Day fraction
        day_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        time_of_day = day_seconds / 86_400.0
        # Week fraction
        weekday_seconds = dt.weekday * 86_400 + day_seconds
        time_of_week = weekday_seconds / (7 * 86_400)
        # Month fraction
        time_of_month, month_total = _compute_time_month(date_times)
        # Year fraction
        time_of_year, year_total = _compute_time_year(date_times)
        if is_raw:
            return {
                "time_of_day": (time_of_day * 24).astype(np.float32),
                "time_of_week": (time_of_week * 7 + 1).astype(np.float32),
                "time_of_month": (
                    time_of_month * month_total.dt.days.astype(float) + 1
                ).astype(np.float32),
                "time_of_year": (
                    time_of_year * year_total.dt.days.astype(float) + 1
                ).astype(np.float32),
            }
        return {
            "time_of_day": time_of_day.astype(np.float32),
            "time_of_week": time_of_week.astype(np.float32),
            "time_of_month": time_of_month.astype(np.float32),
            "time_of_year": time_of_year.astype(np.float32),
        }
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_time_fractions] failure: {exc}")
        na_float = pd.Series(np.nan, index=df.index, dtype="float32")
        return {
            "time_of_day": na_float,
            "time_of_week": na_float,
            "time_of_month": na_float,
            "time_of_year": na_float,
        }
