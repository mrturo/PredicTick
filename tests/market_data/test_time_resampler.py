"""Unit tests for the TimeResampler class located in the ``market_data`` package.

These tests validate correct aggregation of OHLCV data across different scenarios,
including:

* Standard 1‑minute → 5‑minute resampling with single and multiple windows.
* Handling of identical ``from`` / ``to`` intervals (should be a no‑op).
* Enforcement of divisibility between intervals, raising ``ValueError`` otherwise.
* Optional anchor‑based alignment of aggregation windows.

All tests avoid direct ``assert`` statements as per project guidelines. Instead they
leverage helper functions that raise ``AssertionError`` when expectations are not
met, or use ``pytest.raises`` to check error handling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pandas as pd  # type: ignore
import pytest  # type: ignore

# Project import (module path may vary depending on packaging structure)
from market_data.transform.time_resampler import TimeResampler  # type: ignore

###############################################################################
# Helper utilities
###############################################################################


def _check_eq(actual: Any, expected: Any, msg: str | None = None) -> None:
    """Raise :class:`AssertionError` if *actual* differs from *expected*.

    A tiny replacement for direct ``assert`` statements that keeps the style
    consistent across the test‑suite. All comparisons ultimately resolve to a
    boolean equality check.
    """

    if actual != expected:
        raise AssertionError(msg or f"Expected {expected}, got {actual}")


def _build_ohlcv_frame(
    *,
    start: str | datetime,
    periods: int,
    freq: str = "1min",
    base_price: float = 1.0,
) -> pd.DataFrame:
    """Construct a simple synthetic OHLCV DataFrame suitable for resampling.

    Parameters
    ----------
    start
        The starting timestamp (UTC).
    periods
        Number of rows to generate.
    freq
        Pandas‑style frequency string (default ``"1min"``).
    base_price
        The opening price of the first bar; subsequent bars increment by 1.
    """

    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    data: Dict[str, list[float]] = {
        "datetime": index,
        "open": [base_price + i for i in range(periods)],
    }
    # For simplicity: high = open + 0.5, low = open – 0.5, close = open
    data["high"] = [x + 0.5 for x in data["open"]]
    data["low"] = [x - 0.5 for x in data["open"]]
    data["close"] = data["open"].copy()
    data["volume"] = [100] * periods
    data["adj_close"] = data["close"].copy()

    return pd.DataFrame(data)


###############################################################################
# Test cases
###############################################################################


def test_by_ratio_single_window() -> None:
    """Resample five 1‑minute bars into a single 5‑minute bar."""

    df = _build_ohlcv_frame(start="2025-01-01 09:30", periods=5)

    result = TimeResampler.by_ratio(df, "1m", "5m")

    # Expect exactly one aggregated bar
    _check_eq(len(result), 1, "Expected a single aggregated bar")

    agg_row = result.iloc[0]

    _check_eq(agg_row["datetime"], df["datetime"].iloc[0])
    _check_eq(agg_row["open"], df["open"].iloc[0])
    _check_eq(agg_row["high"], max(df["high"]))
    _check_eq(agg_row["low"], min(df["low"]))
    _check_eq(agg_row["close"], df["close"].iloc[-1])
    _check_eq(agg_row["volume"], sum(df["volume"]))
    _check_eq(agg_row["adj_close"], df["adj_close"].iloc[-1])


def test_by_ratio_multiple_windows() -> None:
    """Ensure multiple 5‑minute windows are produced for ten 1‑minute bars."""

    df = _build_ohlcv_frame(start="2025-01-01 09:30", periods=10)

    result = TimeResampler.by_ratio(df, "1m", "5m")

    _check_eq(len(result), 2, "Expected two aggregated bars (09:30 & 09:35)")

    # Validate the first aggregated window (09:30 → 09:35)
    first = result.iloc[0]
    window1 = df.iloc[:5]
    _check_eq(first["datetime"], window1["datetime"].iloc[0])
    _check_eq(first["open"], window1["open"].iloc[0])
    _check_eq(first["high"], max(window1["high"]))
    _check_eq(first["low"], min(window1["low"]))
    _check_eq(first["close"], window1["close"].iloc[-1])
    _check_eq(first["volume"], sum(window1["volume"]))

    # Validate the second aggregated window (09:35 → 09:40)
    second = result.iloc[1]
    window2 = df.iloc[5:]
    _check_eq(second["datetime"], window2["datetime"].iloc[0])
    _check_eq(second["open"], window2["open"].iloc[0])
    _check_eq(second["high"], max(window2["high"]))
    _check_eq(second["low"], min(window2["low"]))
    _check_eq(second["close"], window2["close"].iloc[-1])
    _check_eq(second["volume"], sum(window2["volume"]))


def test_by_ratio_no_op_same_interval() -> None:
    """Calling *by_ratio* with identical intervals should return the original frame."""

    df = _build_ohlcv_frame(start="2025-01-01 09:30", periods=3)

    # The function should *not* mutate in‑place; verify identity via equality, not id.
    res = TimeResampler.by_ratio(df, "1m", "1m")

    # Pandas provides a robust frame comparator that already raises AssertionError
    pd.testing.assert_frame_equal(df.reset_index(drop=True), res.reset_index(drop=True))


def test_by_ratio_invalid_divisibility() -> None:
    """Non‑divisible intervals must raise ``ValueError``."""

    df = _build_ohlcv_frame(start="2025-01-01 09:30", periods=2, freq="5min")

    with pytest.raises(ValueError, match="must be divisible"):
        TimeResampler.by_ratio(df, "5m", "7m")


def test_by_ratio_anchor_alignment() -> None:
    """Windows should align relative to the provided anchor series."""

    df = _build_ohlcv_frame(start="2025-01-01 09:30", periods=10)

    # Use the *df*'s own datetime column as anchor; the last value is 09:39.
    anchor = df["datetime"]

    result = TimeResampler.by_ratio(df, "1m", "5m", datetime=anchor)

    # With anchor at 09:39, the sole window starts at 09:34.
    _check_eq(len(result), 1, "Expected a single anchored window")

    anchored_row = result.iloc[0]
    _check_eq(anchored_row["datetime"].isoformat(), "2025-01-01T09:34:00+00:00")

    # Validate that only the rows 09:34‑09:38 are aggregated.
    sub = df[df["datetime"] >= pd.Timestamp("2025-01-01 09:34", tz="UTC")]
    _check_eq(anchored_row["open"], sub["open"].iloc[0])
    _check_eq(anchored_row["close"], sub["close"].iloc[4])
    _check_eq(anchored_row["volume"], sum(sub["volume"].iloc[:5]))


def test_aggregate_window_empty_block_returns_none() -> None:
    """Directly verify that ``_aggregate_window`` returns *None* for an empty slice."""

    df = _build_ohlcv_frame(start="2025-01-01 09:30", periods=2)

    start = pd.Timestamp("2025-01-01 09:40", tz="UTC")
    end = start + pd.Timedelta(minutes=5)

    res = TimeResampler._aggregate_window(df, start, end)

    _check_eq(res, None, "Expected None when no rows fall inside the window")
