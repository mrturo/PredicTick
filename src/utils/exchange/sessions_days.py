"""Typed, validated, and fully-encapsulated representation of an exchange’s.

weekly trading schedule.

The class hides internal state behind properties that enforce type integrity
while exposing handy helpers such as :meth:`is_trading_day` and
:meth:`open_days`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List


# pylint: disable=too-many-instance-attributes
@dataclass
class SessionsDays:
    """Container holding the weekly trading availability for an exchange.

    Each attribute is a boolean flag indicating whether the exchange is
    open on the corresponding weekday (Monday = 0 … Sunday = 6).
    """

    __slots__ = (
        "_weekdays",
        "_monday",
        "_tuesday",
        "_wednesday",
        "_thursday",
        "_friday",
        "_saturday",
        "_sunday",
    )

    def __init__(self, days: Dict[str, bool], weekdays: List[str]) -> None:
        """Create a :class:`SessionsDays` from a *day → bool* mapping."""
        if not isinstance(days, dict):
            raise TypeError("`days` must be `Dict[str, bool]`")
        missing = [d for d in weekdays if d not in days]
        if missing:
            raise ValueError(f"Missing keys in `days`: {', '.join(missing)}")
        for key, val in days.items():
            if key.lower() not in weekdays:
                raise ValueError(f"Unexpected key in `days`: '{key}'")
            if not isinstance(val, bool):
                raise TypeError(
                    f"Value for '{key}' must be bool, got {type(val).__name__}"
                )
        self._weekdays: List[str] = weekdays
        self.monday = days["monday"]
        self.tuesday = days["tuesday"]
        self.wednesday = days["wednesday"]
        self.thursday = days["thursday"]
        self.friday = days["friday"]
        self.saturday = days["saturday"]
        self.sunday = days["sunday"]

    @property
    def monday(self) -> bool:  # noqa: D401
        """Monday trading flag."""
        return self._monday

    @monday.setter
    def monday(self, value: bool) -> None:
        self._validate_bool(value, "monday")
        self._monday = value

    @property
    def tuesday(self) -> bool:  # noqa: D401
        """Tuesday trading flag."""
        return self._tuesday

    @tuesday.setter
    def tuesday(self, value: bool) -> None:
        self._validate_bool(value, "tuesday")
        self._tuesday = value

    @property
    def wednesday(self) -> bool:  # noqa: D401
        """Wednesday trading flag."""
        return self._wednesday

    @wednesday.setter
    def wednesday(self, value: bool) -> None:
        self._validate_bool(value, "wednesday")
        self._wednesday = value

    @property
    def thursday(self) -> bool:  # noqa: D401
        """Thursday trading flag."""
        return self._thursday

    @thursday.setter
    def thursday(self, value: bool) -> None:
        self._validate_bool(value, "thursday")
        self._thursday = value

    @property
    def friday(self) -> bool:  # noqa: D401
        """Friday trading flag."""
        return self._friday

    @friday.setter
    def friday(self, value: bool) -> None:
        self._validate_bool(value, "friday")
        self._friday = value

    @property
    def saturday(self) -> bool:  # noqa: D401
        """Saturday trading flag."""
        return self._saturday

    @saturday.setter
    def saturday(self, value: bool) -> None:
        self._validate_bool(value, "saturday")
        self._saturday = value

    @property
    def sunday(self) -> bool:  # noqa: D401
        """Sunday trading flag."""
        return self._sunday

    @sunday.setter
    def sunday(self, value: bool) -> None:
        self._validate_bool(value, "sunday")
        self._sunday = value

    def is_trading_day(self, value: date | datetime) -> bool:
        """Return ``True`` if *value* falls on an enabled trading weekday."""
        if isinstance(value, datetime):
            value = value.date()
        weekday_idx = value.weekday()
        return [
            self.monday,
            self.tuesday,
            self.wednesday,
            self.thursday,
            self.friday,
            self.saturday,
            self.sunday,
        ][weekday_idx]

    def open_days(self) -> List[str]:
        """Return the names of all weekdays where the exchange is open."""
        flags = (
            self.monday,
            self.tuesday,
            self.wednesday,
            self.thursday,
            self.friday,
            self.saturday,
            self.sunday,
        )
        return [day.lower() for day, flag in zip(self._weekdays, flags) if flag]

    def any_open(self) -> bool:
        """Fast check: is the exchange open at least one day per week?"""
        return any(self)

    def all_true(self) -> bool:
        """Return ``True`` if **all** weekday flags are ``True``."""
        return all(self)

    def all_false(self) -> bool:
        """Return ``True`` if **all** weekday flags are ``False``."""
        return not any(self)

    def to_json(self) -> Dict[str, bool]:
        """Return a JSON-serialisable mapping of weekday flags."""
        return {
            "monday": self.monday,
            "tuesday": self.tuesday,
            "wednesday": self.wednesday,
            "thursday": self.thursday,
            "friday": self.friday,
            "saturday": self.saturday,
            "sunday": self.sunday,
        }

    def __iter__(self):
        """Iterate over the seven weekday flags in calendar order."""
        yield from (
            self.monday,
            self.tuesday,
            self.wednesday,
            self.thursday,
            self.friday,
            self.saturday,
            self.sunday,
        )

    @staticmethod
    def _validate_bool(value: bool, field_name: str) -> None:
        """Ensure that every day flag is a plain :class:`bool`."""
        if not isinstance(value, bool):
            raise TypeError(f"'{field_name}' must be bool, got {type(value).__name__}")
