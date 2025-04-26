"""Interval module for managing market data timeframes.

Provides static access to raw and enriched data intervals with built-in fallback resolution.
Also enforces validation to ensure raw interval granularity aligns with enriched interval rules.
"""

from typing import Optional

from src.market_data.utils.intervals.interval_converter import \
    IntervalConverter
from src.market_data.utils.intervals.interval_provider import IntervalProvider


class Interval:
    """Public API for accessing and validating market data intervals.

    Provides raw and enriched data intervals and ensures correct hierarchical relation.
    """

    @staticmethod
    def market_raw_data() -> Optional[str]:
        """Get the configured interval for raw market data with fallback resolution."""
        return IntervalProvider.resolve("market_raw_data", "market_enriched_data")

    @staticmethod
    def market_enriched_data() -> Optional[str]:
        """Get the configured interval for enriched market data with fallback resolution."""
        return IntervalProvider.resolve("market_enriched_data", "market_raw_data")

    @staticmethod
    def validate_market_interval_hierarchy() -> None:
        """Ensure raw interval is shorter and a divisor of the enriched interval."""
        raw: Optional[str] = Interval.market_raw_data()
        enriched: Optional[str] = Interval.market_enriched_data()
        raw_minutes = IntervalConverter.to_minutes(raw)
        enriched_minutes = IntervalConverter.to_minutes(enriched)
        if raw_minutes > enriched_minutes:
            raise ValueError(
                f"Raw interval ({raw}) must not be greater than enriched interval ({enriched})."
            )
        if enriched_minutes % raw_minutes != 0:
            raise ValueError(
                f"Enriched interval ({enriched}) must be divisible by raw interval ({raw})."
            )
