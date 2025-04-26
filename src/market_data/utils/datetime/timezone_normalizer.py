"""Utility for normalizing :class:`pandas.DataFrame` indices to Coordinated Universal Time (UTC).

using a configurable market timezone.

This module exposes the :class:`TimezoneNormalizer` helper which ensures that every timestamp
consumed by downstream models is expressed in UTC.  The default market timezone is loaded from
:class:`utils.config.parameters.ParameterLoader` and can be overridden per call.
"""

from typing import Optional, cast

import pandas as pd  # type: ignore
import pytz  # type: ignore
from pandas import DatetimeIndex  # type: ignore

from src.utils.config.parameters import ParameterLoader
from src.utils.io.logger import Logger


# pylint: disable=too-few-public-methods
class TimezoneNormalizer:
    """Static helpers to convert *naïve* or *localized* timestamps to UTC.

    The class reads the default market timezone from the global configuration once at import
    time, avoiding repeated I/O.  All methods are stateless and thread‑safe.
    """

    _PARAMS = ParameterLoader()
    _MARKET_TZ = _PARAMS.get("market_tz")

    @staticmethod
    def localize_to_market_time(
        df: pd.DataFrame, market_tz: Optional[str] = None
    ) -> pd.DataFrame:
        """Return *df* with its index converted to UTC."""
        if df.empty:
            return df
        effective_tz = market_tz or TimezoneNormalizer._MARKET_TZ
        try:
            effective_tz_obj = pytz.timezone(effective_tz)
        except Exception as exc:
            Logger.error(f"Invalid timezone: {effective_tz}. Exception: {exc}")
            raise
        idx = cast(DatetimeIndex, df.index)
        if idx.tz is None:
            df.index = idx.tz_localize(effective_tz_obj)
        else:
            df.index = idx
        return df.tz_convert("UTC")
