"""
Module for managing and validating time-based intervals used in market data processing.

This module provides functionality to:
- Validate interval strings against an accepted format.
- Load intervals from configuration parameters.
- Resolve primary and fallback interval values with validation and error handling.

Classes:
    IntervalValidator: Provides validation for interval string formats.
    IntervalProvider: Retrieves and validates interval values from configuration.
    Interval: Public interface exposing validated interval values for specific use cases.
"""

import re
from math import gcd
from typing import Dict, Optional, Union

from utils.parameters import ParameterLoader


class IntervalValidator:  # pylint: disable=too-few-public-methods
    """
    Validates time interval strings against a predefined pattern.

    Accepted suffixes include full and abbreviated time units like:
    min, hour, day, week, month, year, m, h, d, wk, mo, y.
    """

    PATTERN = re.compile(r"^\d+(min|hour|day|week|month|year|m|h|d|wk|mo|y)$")

    @classmethod
    def is_valid(cls, interval: str) -> bool:
        """
        Checks whether the given interval string matches the accepted pattern and.

        starts with an integer greater than zero.

        Args:
            interval (str): The interval string to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if len(interval.strip()) == 0:
            return False
        match = cls.PATTERN.fullmatch(interval.strip())
        if not match:
            return False
        number_part = interval.strip()[: -len(match.group(1))]
        return number_part.isdigit() and int(number_part) > 0


class IntervalConverter:  # pylint: disable=too-few-public-methods
    """
    Converts valid interval strings into their equivalent number of minutes.

    Supports the same suffixes as IntervalValidator.
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
        "min": "min",  # minute bars → set seconds to 00
        "m": "min",
        "hour": "H",  # hourly bars → set minutes and seconds to 00
        "h": "H",
        "day": "D",  # daily/weekly/monthly/yearly → set time to 00:00:00
        "d": "D",
        "week": "D",
        "wk": "D",
        "month": "D",
        "mo": "D",
        "year": "D",
        "y": "D",
    }

    @staticmethod
    def to_minutes(interval: Optional[str]) -> int:
        """
        Converts a valid interval string into total minutes.

        Args:
            interval (str): Interval string (e.g., '2h', '3d').

        Returns:
            int: Total number of minutes.

        Raises:
            ValueError: If the interval is invalid or has an unsupported unit.
        """
        if interval is None or len(interval.strip()) == 0:
            return 0
        if not IntervalValidator.is_valid(interval):
            raise ValueError(f"Invalid interval format: {interval}")

        interval = interval.strip()
        match = IntervalValidator.PATTERN.fullmatch(interval)
        number = int(interval[: -len(match.group(1))])
        unit = match.group(1)

        try:
            return number * IntervalConverter._UNIT_TO_MINUTES[unit]
        except KeyError as exc:
            raise ValueError(f"Unsupported unit in interval: {unit}") from exc

    @staticmethod
    def simplify(interval: str) -> Optional[str]:
        """
        Simplifies a valid interval to its most compact form.

        Keeps the input unit's naming style: abbreviated or full.

        Args:
            interval (str): The interval string to simplify.

        Returns:
            str: A simplified equivalent interval string.

        Raises:
            ValueError: If the interval is invalid.
        """
        interval = interval.strip()
        if not IntervalValidator.is_valid(interval):
            raise ValueError(f"Invalid interval format: {interval}")

        match = IntervalValidator.PATTERN.fullmatch(interval)
        input_suffix = match.group(1)
        total_minutes = IntervalConverter.to_minutes(interval)

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
        return None  # pragma: no cover

    @staticmethod
    def get_ratio(
        interval_a: str, interval_b: str
    ) -> Dict[str, Union[str, int, float]]:
        """
        Computes the simplified ratio between two valid interval strings.

        Args:
            interval_a (str): First interval (e.g., '30m'). Represents the antecedent.
            interval_b (str): Second interval (e.g., '1h'). Represents the consequent.

        Returns:
            Dict[str, Union[str, int, float]]: A dictionary with:
                - 'label' (str): The simplified ratio in 'X:Y' format.
                - 'antecedent' (int): The simplified numerator.
                - 'consequent' (int): The simplified denominator.
                - 'value' (float): The decimal value of the ratio (antecedent / consequent).

        Raises:
            ValueError: If any interval is invalid or resolves to zero minutes.
        """
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
    def to_pandas_floor_freq(interval: str) -> str:
        """
        Convert an *interval* string into the `freq` keyword expected by.

        :py-meth:`pandas.Series.dt.floor`.

        Examples
        --------
        >>> IntervalConverter.to_pandas_floor_freq("30m")
        'min'
        >>> IntervalConverter.to_pandas_floor_freq("1h")
        'H'
        >>> IntervalConverter.to_pandas_floor_freq("1wk")
        'D'

        Parameters
        ----------
        interval : str
            A validated interval such as ``'30m'``, ``'2h'``, ``'1wk'``.

        Returns
        -------
        str
            The pandas frequency alias.

        Raises
        ------
        ValueError
            If *interval* is not valid or its unit is unsupported.
        """
        if not IntervalValidator.is_valid(interval):
            raise ValueError(f"Invalid interval format: {interval}")

        suffix = IntervalValidator.PATTERN.fullmatch(interval).group(1)
        try:
            return (IntervalConverter._UNIT_TO_PANDAS_FREQ[suffix]).strip().lower()
        except KeyError as exc:
            raise ValueError(f"Unsupported unit: {suffix}") from exc


class IntervalProvider:  # pylint: disable=too-few-public-methods
    """
    Loads interval values from configuration and ensures they are valid.

    If a primary interval is missing or invalid, a fallback is optionally attempted.
    """

    def __init__(self):
        """Initializes the IntervalProvider by loading interval parameters."""
        self._params = ParameterLoader().get("interval")

    def _get_interval_from_key(self, key: str) -> Optional[str]:
        """
        Retrieves and validates an interval from the parameter map by key.

        Args:
            key (str): The parameter key to retrieve.

        Returns:
            Optional[str]: The validated interval string or None if not found.

        Raises:
            ValueError: If the interval exists but is invalid.
        """
        interval: str = self._params.get(key)
        if interval:
            interval = interval.strip()
            if not IntervalValidator.is_valid(interval):
                raise ValueError(f"Invalid Interval for '{key}': {interval}")
            return interval
        return None

    def resolve(self, primary_key: str, fallback_key: str) -> str:
        """
        Resolves a valid interval value using a primary key, and optionally a fallback.

        Args:
            primary_key (str): Primary parameter key.
            fallback_key (str): Fallback parameter key if the primary is missing or invalid.

        Returns:
            str: A valid interval string.

        Raises:
            ValueError: If neither key yields a valid interval.
        """
        interval = self._get_interval_from_key(primary_key)
        if interval is not None:
            return IntervalConverter.simplify(interval)

        try:
            interval = self._get_interval_from_key(fallback_key)
        except ValueError:
            interval = None

        if interval is not None:
            return IntervalConverter.simplify(interval)

        raise ValueError(f"Undefined interval: {primary_key}")


class Interval:
    """
    Public API exposing predefined interval values for market data processing.

    Provides access to raw and enriched market data intervals with fallback resolution.
    """

    _provider = IntervalProvider()

    @staticmethod
    def market_raw_data() -> str:
        """
        Retrieves the interval for raw market data.

        Returns:
            str: A valid interval string for raw market data.

        Raises:
            ValueError: If no valid interval is found.
        """
        return Interval._provider.resolve("market_raw_data", "market_enriched_data")

    @staticmethod
    def market_enriched_data() -> str:
        """
        Retrieves the interval for enriched market data.

        Returns:
            str: A valid interval string for enriched market data.

        Raises:
            ValueError: If no valid interval is found.
        """
        return Interval._provider.resolve("market_enriched_data", "market_raw_data")

    @staticmethod
    def validate_market_interval_hierarchy() -> None:
        """
        Validates that the raw interval is less than or equal to the enriched interval in minutes,.

        and that the enriched interval is divisible by the raw interval.

        Raises:
            ValueError: If either condition is not met.
        """
        raw: str = Interval.market_raw_data()
        enriched: str = Interval.market_enriched_data()

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
