"""IntervalProvider fetches, validates, and simplifies config intervals.

This module exposes `IntervalProvider`, a helper that retrieves, validates, and
simplifies interval strings from the central parameter map. Validation is
performed via `IntervalValidator`, while simplification is delegated to
`IntervalConverter`.
"""

from typing import Optional

from src.market_data.utils.intervals.interval_converter import \
    IntervalConverter
from src.market_data.utils.intervals.interval_validator import \
    IntervalValidator
from src.utils.config.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class IntervalProvider:
    """Load and validate interval strings from configuration.

    If the primary interval is absent or invalid, an optional fallback is used.
    """

    _PARAMS = ParameterLoader()
    _INTERVAL = _PARAMS.get("interval")

    @staticmethod
    def _get_interval_from_key(key: str) -> Optional[str]:
        """Return a validated interval for the given config key."""
        interval: Optional[str] = IntervalProvider._INTERVAL.get(key)
        if interval is None or len(interval.strip()) == 0:
            return None
        interval = interval.strip()
        if not IntervalValidator.is_valid(interval):
            raise ValueError(f"Invalid Interval for '{key}': {interval}")
        return interval

    @staticmethod
    def _get_interval(key: str) -> Optional[str]:
        try:
            interval = IntervalProvider._get_interval_from_key(key)
            if interval is not None:
                return IntervalConverter.simplify(interval)
            return None
        except ValueError:
            return None

    @staticmethod
    def resolve(primary_key: str, fallback_key: str) -> Optional[str]:
        """Return the first valid interval from *primary_key* or *fallback_key*."""
        interval = IntervalProvider._get_interval(primary_key)
        if interval is None:
            interval = IntervalProvider._get_interval(fallback_key)
            if interval is None:
                raise ValueError(f"Undefined interval: {primary_key}")
        return interval
