"""Typed, validated, and fully-encapsulated representation of a market’s trading.

sessions.  The class exposes *properties* that enforce type integrity and
timezone validity while cleanly hiding internal state behind conventional
getter/setter syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.utils.exchange.hours import Hours


# pylint: disable=too-many-instance-attributes
@dataclass
class SessionsHours:
    """Container holding the trading-day segmentation for an exchange.

    * pre_market: Optional hours for the pre-market auction or dark-pool phase.
    * regular: Mandatory *continuous* regular session hours.
    * post_market: Optional hours for the after-hours phase.
    * timezone: IANA timezone string (e.g. ``"America/New_York"``) used to interpret
       all :class:`utils.exchange.hours.Hours` objects.
    """

    __slots__ = ("_pre_market", "_regular", "_post_market", "_timezone")

    def __init__(
        self,
        pre_market: Optional[Hours],
        regular: Hours,
        post_market: Optional[Hours],
        timezone: str,
    ) -> None:
        """Initialize the trading-session segmentation for an exchange."""
        self.pre_market = pre_market
        self.post_market = post_market
        self.regular = regular
        self.timezone = timezone

    @property
    def pre_market(self) -> Optional[Hours]:
        """Return the pre-market :class:`utils.exchange.hours.Hours`, or ``None``."""
        return self._pre_market

    @pre_market.setter
    def pre_market(self, value: Optional[Hours]) -> None:
        """Validate and assign the pre-market session."""
        if value is not None and not isinstance(value, Hours):
            raise TypeError("`pre_market` must be `Hours | None`")
        self._pre_market = value

    @property
    def regular(self) -> Hours:
        """Return the regular trading-session :class:`utils.exchange.hours.Hours`."""
        return self._regular

    @regular.setter
    def regular(self, value: Hours) -> None:
        """Validate and assign the regular trading session.

        Enforce alignment with pre_market and post_market if defined.
        """
        if not isinstance(value, Hours):
            raise TypeError("`regular` must be an instance of `Hours`")
        if self.pre_market and self.pre_market.close != value.open:
            raise ValueError("`pre_market.close` must match `regular.open`")
        if self.post_market and self.post_market.open != value.close:
            raise ValueError("`post_market.open` must match `regular.close`")
        self._regular = value

    @property
    def post_market(self) -> Optional[Hours]:
        """Return the post-market :class:`utils.exchange.hours.Hours`, or ``None``."""
        return self._post_market

    @post_market.setter
    def post_market(self, value: Optional[Hours]) -> None:
        """Validate and assign the post-market session."""
        if value is not None and not isinstance(value, Hours):
            raise TypeError("`post_market` must be `Hours | None`")
        self._post_market = value

    @property
    def timezone(self) -> str:
        """Return the IANA timezone string associated with these sessions."""
        return self._timezone

    @timezone.setter
    def timezone(self, value: str) -> None:
        """Validate and assign the IANA timezone identifier."""
        try:
            value = value.strip()
            if len(value) == 0:
                raise ValueError("is empty")
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Invalid timezone: '{value}'") from exc
        self._timezone = value

    def to_utc(self) -> Optional[SessionsHours]:
        """Return a deep-copied ``SessionsHours`` expressed in UTC.

        * Preserves absolute instants: 09:30-16:00 America/New_York
          → 14:30-21:00 UTC (winter) or 13:30-20:00 UTC (summer).
        * If the ``Hours`` class exposes ``shift_timezone(src, dst)``, that method is
          preferred; otherwise, conversion falls back to ``datetime`` objects.
        * If already in UTC, the object itself is returned to avoid unnecessary copies.
        """
        if self.timezone == "UTC":
            return self
        src_tz, dst_tz = ZoneInfo(self.timezone), ZoneInfo("UTC")

        def _as_time(value: Optional[str | time]) -> time:
            """Coerce ISO-formatted string or ``time`` into ``time``."""
            return value if isinstance(value, time) else time.fromisoformat(value or "")

        def _shift(h: Optional[Hours]) -> Optional[Hours]:
            if h is None:
                return None
            if hasattr(h, "shift_timezone"):
                return h.shift_timezone(src_tz, dst_tz)  # type: ignore[attr-defined]
            base = date.today()
            open_dt = datetime.combine(base, _as_time(h.open), src_tz).astimezone(
                dst_tz
            )
            close_dt = datetime.combine(base, _as_time(h.close), src_tz).astimezone(
                dst_tz
            )
            return Hours(open_dt.strftime("%H:%M"), close_dt.strftime("%H:%M"))

        shift_regular = _shift(self.regular)
        if shift_regular is None:
            return None
        return SessionsHours(
            pre_market=_shift(self.pre_market),
            regular=shift_regular,
            post_market=_shift(self.post_market),
            timezone="UTC",
        )

    def to_json(self) -> Any:
        """Object to JSON."""
        return {
            "pre_market": (
                None if self.pre_market is None else self.pre_market.to_json()
            ),
            "post_market": (
                None if self.post_market is None else self.post_market.to_json()
            ),
            "regular": self.regular.to_json(),
            "timezone": self.timezone,
        }
