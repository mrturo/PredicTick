"""Provides validation for time interval strings using predefined suffix patterns.

This module defines the IntervalValidator class, which verifies whether a string
matches accepted interval formats. Valid suffixes include both full and abbreviated
units such as: min, hour, day, week, month, year, m, h, d, wk, mo, y.
"""

import re


# pylint: disable=too-few-public-methods
class IntervalValidator:
    """Validates time interval strings using a regular expression pattern.

    Acceptable suffixes include: min, hour, day, week, month, year, m, h, d, wk, mo, y.
    """

    PATTERN = re.compile(r"^\d+(min|hour|day|week|month|year|m|h|d|wk|mo|y)$")

    @classmethod
    def is_valid(cls, interval: str) -> bool:
        """Checks if the interval string matches the expected format and is greater than zero."""
        if len(interval.strip()) == 0:
            return False
        match = cls.PATTERN.fullmatch(interval.strip())
        if not match:
            return False
        number_part = interval.strip()[: -len(match.group(1))]
        return number_part.isdigit() and int(number_part) > 0
