"""
Utility functions for timezone normalization in financial market data processing.

Includes tools for localizing naive or timezone-aware DataFrame timestamps to UTC
based on a given market timezone. Designed to support consistent datetime handling
throughout the market data ingestion and training pipeline.
"""

import pandas as pd
import pytz

from utils.logger import Logger
from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class TimezoneNormalizer:
    """Normalizes DataFrame timestamps to UTC using the configured market timezone."""

    _PARAMS = ParameterLoader()
    _MARKET_TZ = _PARAMS.get("market_tz")

    @staticmethod
    def localize_to_market_time(
        df: pd.DataFrame, market_tz: str = None
    ) -> pd.DataFrame:
        """
        Converts naive or localized timestamps to UTC based on market timezone.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DateTimeIndex to normalize.
        market_tz : str, optional
            Timezone name (e.g., "America/New_York"). If None, uses global config.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by UTC timestamps.
        """
        if df.empty:
            return df

        effective_tz = market_tz or TimezoneNormalizer._MARKET_TZ
        try:
            effective_tz_obj = pytz.timezone(effective_tz)
        except Exception as exc:
            Logger.error(f"Invalid timezone: {effective_tz}. Exception: {exc}")
            raise

        # Localize if naive; otherwise, just convert to UTC.
        if df.index.tz is None:
            df.index = df.index.tz_localize(effective_tz_obj)

        return df.tz_convert("UTC")
