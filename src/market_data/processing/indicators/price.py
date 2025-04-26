"""Price‑based technical indicators for quantitative and algorithmic trading.

All helpers in this module are **pure**, *vectorised* and return ``float32``
results for memory efficiency. When the available history is insufficient or
an unexpected failure occurs, the functions gracefully return ``NaN`` values.
"""

from __future__ import annotations

from typing import Final

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.utils.io.logger import Logger

__all__: Final[list[str]] = [
    "compute_intraday_return",
    "compute_price_change",
    "compute_range",
    "compute_volatility",
    "compute_bb_width",
    "compute_typical_price",
    "compute_average_price",
    "compute_price_derivative",
    "compute_smoothed_derivative",
    "compute_return",
    "compute_overnight_return",
]


def compute_intraday_return(close: pd.Series, open_: pd.Series) -> pd.Series:
    """Percentage change from *open* to *close*."""
    try:
        return ((close / open_) - 1).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_intraday_return] failure: {exc}")
        return pd.Series(np.nan, index=close.index, dtype="float32")


def compute_price_change(close: pd.Series, open_: pd.Series) -> pd.Series:
    """Raw price change between *open* and *close* prices."""
    try:
        return (close - open_).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_price_change] failure: {exc}")
        return pd.Series(np.nan, index=close.index, dtype="float32")


def compute_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """High‑low range per bar (`high - low`)."""
    try:
        return (high - low).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_range] failure: {exc}")
        return pd.Series(np.nan, index=high.index, dtype="float32")


def compute_volatility(high: pd.Series, low: pd.Series, open_: pd.Series) -> pd.Series:
    """Intrabar volatility defined as ``(high - low) / open``.

    The division safely handles *0* in *open* by converting them to *NA* before the
    calculation.
    """
    try:
        safe_open = open_.replace(0, pd.NA)
        return ((high - low) / safe_open).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_volatility] failure: {exc}")
        return pd.Series(np.nan, index=high.index, dtype="float32")


def compute_bb_width(close: pd.Series, window: int) -> pd.Series:
    """Bollinger Band *width* as ``max(window) - min(window)``.

    This differs from the classical percentage *B*; it simply measures the
    absolute spread inside the window.
    """
    try:
        return (
            close.rolling(window).apply(lambda x: x.max() - x.min(), raw=True)
        ).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_bb_width] failure: {exc}")
        return pd.Series(np.nan, index=close.index, dtype="float32")


def compute_typical_price(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Typical price as the mean of *high*, *low* and *close*."""
    try:
        return ((high + low + close) / 3.0).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_typical_price] failure: {exc}")
        return pd.Series(np.nan, index=high.index, dtype="float32")


def compute_average_price(high: pd.Series, low: pd.Series) -> pd.Series:
    """Mid‑price between *high* and *low*: ``(high + low) / 2``."""
    try:
        return ((high + low) / 2.0).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_average_price] failure: {exc}")
        return pd.Series(np.nan, index=high.index, dtype="float32")


def compute_price_derivative(close: pd.Series) -> pd.Series:
    """First‑order discrete derivative: ``close.diff()``."""
    try:
        return close.diff().astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_price_derivative] failure: {exc}")
        return pd.Series(np.nan, index=close.index, dtype="float32")


def compute_smoothed_derivative(close: pd.Series, window: int = 5) -> pd.Series:
    """Smoothed derivative of *close* (mean of first diff over a rolling window)."""
    try:
        return close.diff().rolling(window).mean().astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_smoothed_derivative] failure: {exc}")
        return pd.Series(np.nan, index=close.index, dtype="float32")


def compute_return(close: pd.Series) -> pd.Series:
    """Simple percentage returns of *close* prices (``close.pct_change()``)."""
    try:
        return close.pct_change(fill_method=None).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_return] failure: {exc}")
        return pd.Series(np.nan, index=close.index, dtype="float32")


def compute_overnight_return(open_: pd.Series) -> pd.Series:
    """Over‑night percentage return of *open* prices.

    First value is set to 0 to avoid a leading *NaN* in most downstream models.
    """
    try:
        return open_.pct_change(fill_method=None).fillna(0).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_overnight_return] failure: {exc}")
        return pd.Series(np.nan, index=open_.index, dtype="float32")
