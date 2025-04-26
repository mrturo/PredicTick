"""Trend and momentum technical indicators plus shared helpers.

This module centralises the *time‑aware* helpers used across the indicators
sub‑package so that other files can simply and avoid code duplication.
Indicators implemented here focus on *trend* and
*volatility* concepts: ADX, ATR, RSI, MACD, Bollinger %B, Williams %R.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import ta  # type: ignore

from src.utils.io.logger import Logger

__all__: Final[list[str]] = [
    "_infer_bar_seconds",
    "_records_per_session",
    "compute_adx_14d",
    "compute_atr",
    "compute_atr_14d",
    "compute_rsi",
    "compute_stoch_rsi",
    "compute_macd",
    "compute_bollinger_pct_b",
    "compute_williams_r",
    "compute_open_close_result",
]


def _infer_bar_seconds(idx: pd.DatetimeIndex) -> int:  # noqa: D401
    """Infer modal bar length (seconds) from a *DatetimeIndex*."""
    if not isinstance(idx, pd.DatetimeIndex):  # noqa: TRY003
        raise TypeError("Index must be DatetimeIndex to infer bar size")
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:  # noqa: WPS504
        raise ValueError("Need at least two timestamps to infer bar size")
    bar_delta = diffs.mode().iloc[0]
    if not hasattr(bar_delta, "total_seconds"):
        raise TypeError("Time differences are not timedelta‑like; cannot infer size")
    return int(bar_delta.total_seconds())


def _records_per_session(idx: pd.DatetimeIndex) -> int:  # noqa: D401
    """Return median bars required to span one trading session."""
    sec_per_bar = _infer_bar_seconds(idx)
    daily_span_sec = (
        pd.Series(idx)
        .groupby(idx.date)
        .agg(lambda s: (s.max() - s.min()).total_seconds() + sec_per_bar)
    )
    return int(math.ceil(daily_span_sec.median() / sec_per_bar))


def compute_adx_14d(
    date_time: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """14‑session Average Directional Index (ADX)."""
    df = (
        pd.DataFrame({"datetime": date_time, "high": high, "low": low, "close": close})
        .set_index("datetime")
        .sort_index()
    )
    try:
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        if (df.index[-1] - df.index[0]).days < 14:
            raise ValueError("requires ≥ 14 complete sessions")
        bars_day = _records_per_session(df.index)
        window = 14 * bars_day
        adx = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=window
        ).adx()
        adx = adx.astype("float32")
        adx.index = high.index
        return adx
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[ADX‑14d] failure: {exc}")
        return pd.Series(np.nan, index=high.index, dtype="float32")


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Average True Range (ATR) over *window* bars (intraday‑agnostic)."""
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    )
    return tr.max(axis=1).rolling(window).mean().astype("float32")


def compute_atr_14d(
    date_time: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """14‑session ATR built on :pymeth:`ta.volatility.AverageTrueRange`."""
    df = (
        pd.DataFrame({"datetime": date_time, "high": high, "low": low, "close": close})
        .set_index("datetime")
        .sort_index()
    )
    try:
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        if (df.index[-1] - df.index[0]).days < 14:
            raise ValueError("requires ≥ 14 complete sessions")
        bars_day = _records_per_session(df.index)
        window = 14 * bars_day
        atr = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=window
        ).average_true_range()
        atr = atr.astype("float32")
        atr.index = high.index
        return atr
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[ATR‑14d] failure: {exc}")
        return pd.Series(np.nan, index=high.index, dtype="float32")


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index (RSI) 0–100."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).astype("float32")


def compute_stoch_rsi(close: pd.Series, window: int) -> pd.Series:
    """Stochastic RSI (0–1‑scaled RSI)."""
    rsi = compute_rsi(close, window)
    min_rsi = rsi.rolling(window, min_periods=1).min()
    max_rsi = rsi.rolling(window, min_periods=1).max()
    return ((rsi - min_rsi) / (max_rsi - min_rsi)).astype("float32")


def compute_macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """Moving Average Convergence/Divergence (MACD)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line.astype("float32"),
            "signal": signal_line.astype("float32"),
            "histogram": hist.astype("float32"),
        }
    )


def compute_bollinger_pct_b(
    close: pd.Series, window: int = 20, window_dev: float = 2.0
) -> pd.Series:
    """Bollinger Band %%B indicator using *ta.volatility.BollingerBands*."""
    return (
        ta.volatility.BollingerBands(
            close=close, window=window, window_dev=window_dev
        ).bollinger_pband()
    ).astype("float32")


def compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Williams %%R oscillator (‑100 to 0 range)."""
    high_max = high.rolling(window=window).max()
    low_min = low.rolling(window=window).min()
    williams_r = -100 * ((high_max - close) / (high_max - low_min))
    return williams_r.astype("float32")


def compute_open_close_result(
    open_: pd.Series, close: pd.Series, is_raw: bool
) -> pd.Series:
    """Return numeric or categorical direction from open vs close comparison."""
    try:
        if is_raw:
            conditions = [close > open_, close < open_]
            choices = ["UP", "DOWN"]
            result = np.select(conditions, choices, default="NEUTRAL")
            return pd.Series(result, index=open_.index, dtype="category")
        result = pd.Series(
            np.where(close > open_, 1.0, np.where(close < open_, 0.0, 0.5)),
            index=open_.index,
            dtype="float32",
        )
        return result
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[OpenCloseResult] failure: {exc}")
        return pd.Series(np.nan, index=open_.index, dtype="float32")
