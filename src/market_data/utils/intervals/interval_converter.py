"""Interval conversion utilities for market data.

This module defines `IntervalConverter`, a helper class that converts, simplifies
and compares textual interval strings validated by `IntervalValidator`.
"""

from math import gcd
from typing import Dict, Optional, Union

from src.market_data.utils.intervals.interval_validator import \
    IntervalValidator


class IntervalConverter:
    """Convert textual time intervals into canonical numeric or string forms.

    All methods assume the input already satisfies `IntervalValidator.is_valid`.
    """

    _UNIT_TO_MINUTES = {
        "min": 1,
        "m": 1,
        "hour": 60,
        "h": 60,
        "day": 1440,
        "d": 1440,
        "week": 10080,
        "wk": 10080,
        "month": 43200,
        "mo": 43200,
        "year": 525600,
        "y": 525600,
    }
    _SIMPLIFICATION_ORDER = [
        ("year", 525600),
        ("month", 43200),
        ("week", 10080),
        ("day", 1440),
        ("hour", 60),
        ("min", 1),
    ]
    _SUFFIX_VARIANTS = {
        "min": "m",
        "hour": "h",
        "day": "d",
        "week": "wk",
        "month": "mo",
        "year": "y",
    }
    _UNIT_TO_PANDAS_FREQ = {
        "min": "min",
        "m": "min",
        "hour": "H",
        "h": "H",
        "day": "D",
        "d": "D",
        "week": "D",
        "wk": "D",
        "month": "D",
        "mo": "D",
        "year": "D",
        "y": "D",
    }

    @staticmethod
    def _extract_suffix(interval: Optional[str]) -> Optional[str]:
        """Return the unit suffix (e.g., 'm', 'h') from a validated *interval*."""
        if not IntervalConverter._has_content(interval):
            return None
        IntervalConverter._validate_format(interval)
        return IntervalConverter._parse_suffix(interval)

    @staticmethod
    def _has_content(interval: Optional[str]) -> bool:
        """Check if the interval has non-empty content."""
        return interval is not None and len(interval.strip()) > 0

    @staticmethod
    def _validate_format(interval: Optional[str]) -> None:
        """Validate that the interval is in correct format."""
        if not IntervalValidator.is_valid(interval or ""):
            raise ValueError(f"Invalid interval format: {interval}")

    @staticmethod
    def _parse_suffix(interval: Optional[str]) -> str:
        """Parse and return the suffix from a valid interval string."""
        match = IntervalValidator.PATTERN.fullmatch(interval or "")
        if match is None:
            raise ValueError(f"Unable to parse interval: {interval}")
        return match.group(1)

    @staticmethod
    def to_minutes(
        interval: Optional[str], extracted_suffix: Optional[str] = None
    ) -> int:
        """Return total minutes represented by *interval*."""
        if extracted_suffix is None:
            extracted_suffix = IntervalConverter._extract_suffix(interval)
        if interval is None or extracted_suffix is None:
            return 0
        number = int(interval[: -len(extracted_suffix)])
        try:
            return number * IntervalConverter._UNIT_TO_MINUTES[extracted_suffix]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported unit in interval: {extracted_suffix}"
            ) from exc

    @staticmethod
    def simplify(interval: Optional[str]) -> Optional[str]:
        """Return the shortest textual representation of *interval*."""
        input_suffix = IntervalConverter._extract_suffix(interval)
        total_minutes = IntervalConverter.to_minutes(interval, input_suffix)
        use_full = input_suffix in IntervalConverter._SUFFIX_VARIANTS
        for full_unit, minutes in IntervalConverter._SIMPLIFICATION_ORDER:
            if total_minutes % minutes == 0:
                value = total_minutes // minutes
                suffix = (
                    full_unit
                    if use_full
                    else IntervalConverter._SUFFIX_VARIANTS[full_unit]
                )
                return f"{value}{suffix}"
        return None

    @staticmethod
    def get_ratio(
        interval_a: Optional[str], interval_b: Optional[str]
    ) -> Dict[str, Union[str, int, float]]:
        """Return gcd-reduced ratio between *interval_a* and *interval_b*."""
        minutes_a = IntervalConverter.to_minutes(interval_a)
        minutes_b = IntervalConverter.to_minutes(interval_b)
        divisor = gcd(minutes_a, minutes_b)
        antecedent = minutes_a // divisor
        consequent = minutes_b // divisor
        return {
            "label": f"{antecedent}:{consequent}",
            "antecedent": antecedent,
            "consequent": consequent,
            "value": minutes_a / minutes_b,
        }

    @staticmethod
    def to_pandas_floor_freq(interval: Optional[str]) -> Optional[str]:
        """Return the pandas offset alias used by Series.dt.floor for *interval*."""
        input_suffix = IntervalConverter._extract_suffix(interval)
        if input_suffix is None or len(input_suffix.strip()) == 0:
            raise ValueError("Invalid interval format")
        try:
            return (
                (IntervalConverter._UNIT_TO_PANDAS_FREQ[input_suffix]).strip().lower()
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported unit: {input_suffix}") from exc
