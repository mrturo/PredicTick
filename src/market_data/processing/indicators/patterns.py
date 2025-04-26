"""Single- and multi-candle pattern detection utilities.

This module encapsulates the pattern-recognition logic so it can evolve
independently of the rest of the indicators sub-package.  Two public helpers
are exposed:

* :pyfunc:`compute_candle_pattern` – evaluates each bar *individually*.
* :pyfunc:`compute_multi_candle_pattern` – evaluates rolling 3-bar windows
  (classic formations such as Morning Star, Three White Soldiers, etc.).
"""

from __future__ import annotations

from typing import Final, List, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.api.types import is_numeric_dtype  # type: ignore

from src.market_data.processing.candles.candle import Candle
from src.market_data.processing.candles.multi_candle_pattern import \
    MultiCandlePattern
from src.utils.io.logger import Logger

__all__: Final[list[str]] = [
    "compute_candle_pattern",
    "compute_multi_candle_pattern",
]


def _coerce_numeric(ohlc: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
    """Ensure the OHLC frame is numeric and *float32*-typed."""
    out = ohlc.copy()
    for col in out.columns:
        if not is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].astype("float32", copy=False)
    return out


def compute_candle_pattern(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    output_as_name: bool = True,
) -> Optional[Union[pd.Series, None]]:
    """Detect single-bar candlestick patterns."""
    try:
        ohlc = pd.DataFrame(
            {
                "open": open_.astype("float32"),
                "high": high.astype("float32"),
                "low": low.astype("float32"),
                "close": close.astype("float32"),
            }
        )
        if output_as_name:
            series = ohlc.apply(
                lambda r: Candle(r.open, r.high, r.low, r.close).detect_pattern(),
                axis=1,
            ).astype("category")
        else:
            series = ohlc.apply(
                lambda r: Candle(r.open, r.high, r.low, r.close).score(), axis=1
            ).astype("float32")
        return series
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_candle_pattern] failure: {exc}")
        if output_as_name:
            return pd.Series(pd.NA, index=open_.index, dtype="category")
        return pd.Series(np.nan, index=open_.index, dtype="float32")


def compute_multi_candle_pattern(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    output_as_name: bool = True,
) -> Optional[pd.Series]:
    """Detect patterns that span **three** consecutive candles.

    The first two rows are necessarily *NaN*/*<NA>* because a minimum of three
    bars is needed to evaluate the window.
    """
    try:
        ohlc = _coerce_numeric(
            pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
        )
        n_rows = len(ohlc)
        if n_rows < 3:  # noqa: WPS507
            raise ValueError("insufficient history (< 3 bars)")
        results: List[Union[str, np.float32, pd.NA]] = []
        for i in range(n_rows):
            if i < 2:
                results.append(pd.NA if output_as_name else np.nan)
                continue
            window = ohlc.iloc[i - 2 : i + 1]
            candles = [
                Candle(r.open, r.high, r.low, r.close)
                for r in window.itertuples(index=False)
            ]
            if output_as_name:
                pattern = MultiCandlePattern.detect_pattern(candles)
                results.append(pattern if pattern is not None else pd.NA)
            else:
                score = MultiCandlePattern.score(candles)
                results.append(np.float32(score))
        dtype = "category" if output_as_name else "float32"
        return pd.Series(results, index=open_.index, dtype=dtype)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        Logger.warning(f"[compute_multi_candle_pattern] failure: {exc}")
        if output_as_name:
            return pd.Series(pd.NA, index=open_.index, dtype="category")
        return pd.Series(np.nan, index=open_.index, dtype="float32")
