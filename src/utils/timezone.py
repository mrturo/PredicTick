"""
Utility functions for timezone normalization in financial market data processing.

Includes tools for localizing naive or timezone-aware DataFrame timestamps to UTC
based on a given market timezone. Designed to support consistent datetime handling
throughout the market data ingestion and training pipeline.
"""

import pandas as pd
import pytz

from utils.logger import Logger


def localize_to_market_time(df: pd.DataFrame, market_tz_str: str):
    """Converts naive or localized timestamps to UTC based on market timezone."""
    if df.empty:
        Logger.warning("Received empty DataFrame for localization.")
        return df

    market_tz = pytz.timezone(market_tz_str)

    if df.index.tzinfo is None or df.index.tz is None:
        df.index = df.index.tz_localize(market_tz)
        Logger.info("Localized naive timestamps to New York Eastern Time.")

    return df.tz_convert("UTC")
