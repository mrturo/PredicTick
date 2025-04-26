"""Typed, validated, and fully-encapsulated representation of a trading session.

This class ensures the open and close times are properly formatted and logically
consistent. At least one of them must be non-null, and valid hour-minute strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


# pylint: disable=too-few-public-methods
@dataclass
class Hours:
    """Container for a single session's open and/or close time.

    * open: Optional opening time in "HH:MM" format.
    * close: Optional closing time in "HH:MM" format.
    """

    __slots__ = ("_open", "_close")

    def __init__(self, open_time: Optional[str], close_time: Optional[str]) -> None:
        """Initialize a trading session with optional open/close times."""
        validated_open = self._validate_time(open_time, "open")
        validated_close = self._validate_time(close_time, "close")
        if validated_open and validated_close:
            if validated_close != "00:00" and validated_open > validated_close:
                raise ValueError("`open` must be <= `close`, unless `close` == '00:00'")
        self._open = validated_open
        self._close = validated_close

    @property
    def open(self) -> Optional[str]:
        """Return the session opening time or None."""
        return self._open

    @property
    def close(self) -> Optional[str]:
        """Return the session closing time or None."""
        return self._close

    def _validate_time(self, value: Optional[str], field: str) -> Optional[str]:
        """Validate the time string or return None."""
        if value is None:
            return "00:00"
        if not isinstance(value, str):
            raise TypeError(f"`{field}` must be a string or None")
        if len(value.strip()) == 0:
            return "00:00"
        if not re.fullmatch(r"\d{2}:\d{2}", value):
            raise ValueError(f"`{field}` must be in 'HH:MM' format")
        hours, minutes = map(int, value.split(":"))
        if not (0 <= hours <= 23 and 0 <= minutes <= 59):
            raise ValueError(f"`{field}` must be a valid time between 00:00 and 23:59")
        return value

    def to_json(self) -> Any:
        """Object to JSON."""
        return {"open": self.open, "close": self.close}
