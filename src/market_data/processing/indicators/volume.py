"""Volume‑based technical indicators.

All computations return ``float32`` series to minimise memory footprint and
behave gracefully when the historical window is insufficient: a *NaN* vector
is returned while the incident is logged via :pyclass:`utils.io.logger.Logger`.

This module reuses time‑aware helpers from :pymod:`.trend` to avoid code
duplication.  No additional third‑party dependencies are required beyond *ta*.
"""

from __future__ import annotations

from typing import Final

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import ta  # type: ignore

from src.utils.io.logger import Logger

from .trend import _records_per_session  # re‑exported helper (no public API)

__all__: Final[list[str]] = [
    "compute_volume_rvol_20d",
    "compute_relative_volume",
    "compute_volume_change",
    "compute_obv",
]


def compute_volume_rvol_20d(date_time: pd.Series, volume: pd.Series) -> pd.Series:
    """20‑session Relative Volume (RVOL)."""
    df = pd.DataFrame({"datetime": date_time, "volume": volume}).set_index("datetime")
    try:
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        if (df.index[-1] - df.index[0]).days < 20:
            raise ValueError("insufficient history (< 20 days)")
        bars_day = _records_per_session(df.index)
        window = 20 * bars_day
        rvol = volume / volume.rolling(window, min_periods=window).mean()
        return rvol.astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[RVOL‑20d] failure: {exc}")
        return pd.Series(np.nan, index=volume.index, dtype="float32")


def compute_relative_volume(volume: pd.Series, window: int) -> pd.Series:
    """Relative volume for an arbitrary window in *bars*."""
    try:
        return (volume / volume.rolling(window, min_periods=1).mean()).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_relative_volume] failure: {exc}")
        return pd.Series(np.nan, index=volume.index, dtype="float32")


def compute_volume_change(volume: pd.Series) -> pd.Series:
    """Percentage change of *volume* (``volume.pct_change()``)."""
    try:
        return volume.pct_change(fill_method=None).astype("float32")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_volume_change] failure: {exc}")
        return pd.Series(np.nan, index=volume.index, dtype="float32")


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On‑Balance Volume (OBV) via :pyclass:`ta.volume.OnBalanceVolumeIndicator`."""
    try:
        return (
            ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
            .on_balance_volume()
            .astype("float32")
        )
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_obv] failure: {exc}")
        return pd.Series(np.nan, index=close.index, dtype="float32")
