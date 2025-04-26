"""Unit tests for the SessionsHours class in utils.exchange.sessions_hours.

This module verifies correct validation, encapsulation, and timezone integrity
for trading session definitions, including optional and mandatory segments.
"""

import json
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest  # type: ignore

from src.utils.exchange.hours import Hours
from src.utils.exchange.sessions_hours import SessionsHours


def test_valid_sessions_hours():
    """Valid instantiation with all fields provided."""
    post = Hours("16:00", "18:00")
    pre = Hours("08:00", "09:30")
    reg = Hours("09:30", "16:00")
    tz = "America/New_York"
    sh = SessionsHours(pre, reg, post, tz)
    if sh.pre_market != pre:
        raise AssertionError("Expected correct pre-market hours")
    if sh.regular != reg:
        raise AssertionError("Expected correct regular hours")
    if sh.post_market != post:
        raise AssertionError("Expected correct post-market hours")
    if sh.timezone != tz:
        raise AssertionError("Expected correct timezone")
    received_str: str = json.dumps(sh.to_json(), ensure_ascii=False)
    expected_str: str = (
        '{"pre_market": {"open": "08:00", "close": "09:30"}, "post_market": {"open": "16:00", '
        + '"close": "18:00"}, "regular": {"open": "09:30", "close": "16:00"}, "timezone": '
        + '"America/New_York"}'
    )
    if received_str != expected_str:
        raise AssertionError(
            f"Expected json to be: {expected_str}. Received was: {received_str}"
        )


def test_sessions_hours_without_pre_and_post():
    """Valid instantiation with only regular session and timezone."""
    reg = Hours("09:30", "16:00")
    tz = "Europe/London"
    sh = SessionsHours(None, reg, None, tz)
    if sh.pre_market is not None:
        raise AssertionError("Expected pre_market to be None")
    if sh.post_market is not None:
        raise AssertionError("Expected post_market to be None")


def test_invalid_regular_type():
    """Raise error when regular is not a Hours instance."""
    with pytest.raises(TypeError, match="`regular` must be an instance of `Hours`"):
        SessionsHours(None, "09:30", None, "America/New_York")  # type: ignore


def test_invalid_pre_market_type():
    """Raise error when pre_market is not Hours or None."""
    with pytest.raises(TypeError, match="`pre_market` must be `Hours | None`"):
        SessionsHours(123, Hours("09:30", "16:00"), None, "America/New_York")  # type: ignore


def test_invalid_post_market_type():
    """Raise error when post_market is not Hours or None."""
    with pytest.raises(TypeError, match="`post_market` must be `Hours | None`"):
        SessionsHours(None, Hours("09:30", "16:00"), "16:00", "America/New_York")  # type: ignore


def test_invalid_timezone_empty():
    """Raise error when timezone is an empty string."""
    with pytest.raises(ValueError, match="is empty"):
        SessionsHours(None, Hours("09:30", "16:00"), None, "")


def test_invalid_timezone_format():
    """Raise error when timezone is invalid or unknown."""
    with pytest.raises(ValueError, match="Invalid timezone"):
        SessionsHours(None, Hours("09:30", "16:00"), None, "Mars/OlympusMons")


def test_regular_mismatch_with_pre_market():
    """Raise error when regular.open != pre_market.close."""
    pre = Hours("08:00", "09:00")
    reg = Hours("09:30", "16:00")  # ← mismatch with pre.close
    with pytest.raises(
        ValueError, match="`pre_market.close` must match `regular.open`"
    ):
        SessionsHours(pre, reg, None, "America/New_York")


def test_regular_mismatch_with_post_market():
    """Raise error when regular.close != post_market.open."""
    reg = Hours("09:30", "16:00")
    post = Hours("17:00", "18:00")  # ← mismatch with reg.close
    with pytest.raises(
        ValueError, match="`post_market.open` must match `regular.close`"
    ):
        SessionsHours(None, reg, post, "America/New_York")


def test_to_utc_identity_if_already_utc():
    """Should return the same instance if timezone is already UTC."""
    reg = Hours("09:30", "16:00")
    sh = SessionsHours(None, reg, None, "UTC")
    result = sh.to_utc()
    if result is not sh:
        raise AssertionError("Expected the same instance when timezone is UTC")


def test_to_utc_conversion_basic():
    """Should convert hours to UTC for a known timezone (e.g. NY)."""
    tz = "America/New_York"
    reg = Hours("09:30", "16:00")
    pre = Hours("08:00", "09:30")
    post = Hours("16:00", "18:00")
    sh = SessionsHours(pre, reg, post, tz)
    converted = sh.to_utc()
    if converted is None:
        raise AssertionError("Expected non-None result from to_utc")
    if converted.timezone != "UTC":
        raise AssertionError("Expected timezone to be 'UTC'")
    expected_open = (
        datetime.combine(date.today(), time(9, 30), ZoneInfo("America/New_York"))
        .astimezone(ZoneInfo("UTC"))
        .replace(second=0, microsecond=0)
        .time()
    )
    converted_open = time.fromisoformat(converted.regular.open or "")
    if converted_open != expected_open:
        raise AssertionError(
            f"Incorrect regular.open after conversion to UTC: "
            f"{converted_open} != {expected_open}"
        )


def test_to_utc_without_shift_timezone_method(monkeypatch):
    """Test fallback behavior when Hours lacks shift_timezone method."""
    reg: Hours = Hours("09:30", "16:00")
    sh = SessionsHours(None, reg, None, "Europe/London")
    monkeypatch.setattr(sh, "_regular", reg)
    monkeypatch.setattr(sh, "_pre_market", None)
    monkeypatch.setattr(sh, "_post_market", None)
    result = sh.to_utc()
    if result is None or result.timezone != "UTC":
        raise AssertionError("Fallback conversion to UTC failed")


def test_to_utc_with_shift_timezone_method():
    """Test that `shift_timezone` is used if available in Hours."""

    class MockHours(Hours):
        """Lightweight stub of :class:`Hours` used solely for this test.

        The class annotates the session label each time a timezone conversion
        occurs so that we can assert the delegation logic inside
        :py:meth:`SessionsHours.to_utc`.
        """

        def __init__(self, label: str):  # pylint: disable=super-init-not-called
            """Create the stub with a descriptive *label*."""
            self.label = label

        def shift_timezone(self, src, dst):
            """Return a new instance indicating the shift that was applied."""
            return MockHours(f"{self.label} shifted from {src} to {dst}")

        @property
        def open(self):
            """Return the synthetic open time."""
            if "pre" in self.label:
                return "08:00"
            if "reg" in self.label:
                return "09:30"
            return "16:00"

        @property
        def close(self):
            """Return the synthetic close time."""
            if "pre" in self.label:
                return "09:30"
            if "reg" in self.label:
                return "16:00"
            return "18:00"

    pre = MockHours("pre")
    reg = MockHours("reg")
    post = MockHours("post")
    sh = SessionsHours(pre, reg, post, "Europe/London")
    result = sh.to_utc()
    if result is None:
        raise AssertionError("Result is None")
    for h in (result.pre_market, result.regular, result.post_market):
        if not isinstance(h, MockHours):
            raise AssertionError("Expected instance of MockHours")
        if "shifted from" not in h.label:
            raise AssertionError("Expected label to indicate timezone shift")


def test_to_utc_returns_none_when_shift_fails(monkeypatch):
    """`to_utc` should return None if `shift_timezone` unexpectedly fails.
    We create a custom `Hours` double that implements `shift_timezone`
    but deliberately returns `None` to trigger the early exit guarded by:
        if shift_regular is None:
            return None
    """

    class FailingShiftHours(Hours):
        """A stand-in `Hours` whose timezone shift always fails."""

        def __init__(self) -> None:  # pylint: disable=super-init-not-called
            self._open = "09:30"
            self._close = "16:00"

        def shift_timezone(self, _src, _dst):
            """Return a new instance indicating the shift that was applied."""
            return None  # ← forces `to_utc` to return None

        # Properties accessed by `SessionsHours`
        @property
        def open(self):
            return self._open

        @property
        def close(self):
            return self._close

    # Create a valid SessionsHours instance and patch its `_regular` segment.
    regular_ok = Hours("09:30", "16:00")
    sh = SessionsHours(None, regular_ok, None, "Europe/London")
    monkeypatch.setattr(sh, "_regular", FailingShiftHours())
    result = sh.to_utc()
    if result is not None:
        raise AssertionError("Expected None when shift_timezone fails")
