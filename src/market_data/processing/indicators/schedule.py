"""Market-time schedule helpers and phase flags.

This module converts an **ad-hoc calendar** – typically provided by exchange
APIs or manual CSVs – into fast, vectorised flags that tell whether each row
falls inside *pre-market*, *regular* or *post-market* trading hours.

Unlike classical approaches that resample the data, the algorithm keeps the
original granularity (seconds, minutes, etc.) and attaches four boolean /
float32 columns that ML pipelines can consume directly:

* is_market_day: trading session exists *and* it is a workday.
* is_pre_market_time: timestamp is before the official *time_from*.
* is_market_time: timestamp in the [time_from, time_to] window.
* is_post_market_time: timestamp after *time_to* on valid session days.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, Final, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.market_data.utils.intervals.interval_validator import \
    IntervalValidator
from src.utils.io.logger import Logger

__all__: Final[list[str]] = [
    "compute_market_time",
]


def _is_intraday_interval(interval: str) -> bool:  # noqa: D401
    """Return *True* if *interval* denotes minutes or hours granularity."""
    match = IntervalValidator.PATTERN.fullmatch(interval.strip())
    if not match:  # noqa: WPS504
        raise ValueError(f"Invalid interval: '{interval}'")
    unit = match.group(1).lower()
    return unit in ("min", "m", "hour", "h")


def _floor_time_to_hour(time_str: str) -> str:  # noqa: D401
    """Floor an *HH:MM* string to the previous whole hour."""
    if not isinstance(time_str, str) or ":" not in time_str:
        return time_str
    hour = int(time_str.split(":")[0])
    return f"{hour:02d}:00"


def _build_schedule_df(
    market_time: Sequence[Dict[str, str]],
    floor: bool,
) -> pd.DataFrame:  # noqa: D401
    """Normalise *market_time* into a typed *DataFrame*."""
    df = pd.DataFrame(market_time)
    df["date_from"] = pd.to_datetime(df["date_from"]).dt.date
    df["date_to"] = pd.to_datetime(df["date_to"]).dt.date
    if floor:
        df["time_from"] = df["time_from"].map(_floor_time_to_hour)
        df["time_to"] = df["time_to"].map(_floor_time_to_hour)
    return df


def _broadcast_schedule(
    dates: pd.Series, sched: pd.DataFrame
) -> pd.Series:  # noqa: D401
    """Map each *date* to its corresponding schedule dict (vectorised)."""
    out = pd.Series(pd.NA, index=dates.index, dtype="object")
    for _, row in sched.iterrows():
        mask = (dates >= row.date_from) & (dates <= row.date_to)
        if mask.any():
            out.loc[mask] = [
                {
                    "date_from": row.date_from,
                    "date_to": row.date_to,
                    "time_from": row.time_from,
                    "time_to": row.time_to,
                }
            ] * int(mask.sum())
    return out


def _determine_time_handling(interval: str) -> tuple[bool, bool]:
    """Determine if intraday precision is needed and if times should be floored."""
    intraday = _is_intraday_interval(interval)
    match = IntervalValidator.PATTERN.fullmatch(interval.strip())
    unit = match.group(1).lower() if match is not None else None
    floor_times = (not intraday) or unit in {"hour", "h"}
    return intraday, floor_times


def _extract_schedule_time(s: pd.Series, key: str) -> pd.Series:
    """Extract a time object from schedule dictionaries."""
    return s.apply(
        lambda d: (
            _dt.datetime.strptime(d[key], "%H:%M").time()
            if isinstance(d, dict)
            else None
        )
    )


def _assign_market_flags_intraday(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Assign intraday market-phase flags based on current and schedule times."""
    t_from = _extract_schedule_time(df["schedule"], "time_from")
    t_to = _extract_schedule_time(df["schedule"], "time_to")
    current_t = df["datetime"].dt.time
    is_mkt = (current_t >= t_from) & (current_t <= t_to)
    is_pre = (current_t < t_from) & t_from.notna()
    is_post = (current_t > t_to) & t_to.notna()
    return is_pre, is_mkt, is_post


def _safe_cast_flags(
    is_raw: bool,
    is_workday: pd.Series,
    is_pre: pd.Series,
    is_mkt: pd.Series,
    is_post: pd.Series,
) -> Dict[str, pd.Series]:
    """Cast phase flags to the appropriate dtype."""
    cast = (
        (lambda s: s.astype("boolean")) if is_raw else (lambda s: s.astype("float32"))
    )
    return {
        "is_market_day": cast((is_pre | is_mkt | is_post) & is_workday),
        "is_pre_market_time": cast(is_pre),
        "is_market_time": cast(is_mkt),
        "is_post_market_time": cast(is_post),
    }


def _fallback_flags(idx: pd.Index, is_raw: bool) -> Dict[str, pd.Series]:
    """Return a dict of NA/NaN series when market-time computation fails."""
    na_bool = pd.Series(pd.NA, index=idx, dtype="boolean")
    na_float = pd.Series(np.nan, index=idx, dtype="float32")
    base = {
        "is_market_day": na_bool,
        "is_pre_market_time": na_bool,
        "is_market_time": na_bool,
        "is_post_market_time": na_bool,
    }
    return base if is_raw else {k: na_float for k in base}


def compute_market_time(
    df: pd.DataFrame,
    market_time: Sequence[Dict[str, str]],
    interval: str,
    is_raw: bool,
    is_workday: pd.Series,
) -> Tuple[Dict[str, pd.Series], Optional[pd.Series]]:
    """
    Build boolean flags identifying pre-market, market and post-market phases.

    for each row in *df* and return them together with the deduplicated
    schedule.
    """
    try:
        work_df = df.copy()
        intraday, floor = _determine_time_handling(interval)
        sched = _build_schedule_df(market_time, floor)
        work_df["schedule"] = _broadcast_schedule(work_df["datetime"].dt.date, sched)
        if intraday:
            pre, mkt, post = _assign_market_flags_intraday(work_df)
        else:
            pre = post = pd.Series(False, index=work_df.index)
            mkt = pd.Series(True, index=work_df.index)
        flags = _safe_cast_flags(is_raw, is_workday, pre, mkt, post)
        return flags, work_df["schedule"].drop_duplicates(ignore_index=True)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_market_time] failure: {exc}")
        return _fallback_flags(df.index, is_raw), None
