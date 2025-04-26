"""Unit tests for the StockExchange class in utils.exchange.stock_exchange.

This module verifies correct validation of input parameters and encapsulation
of trading session metadata using a stubbed SessionsHours instance.
"""

# pylint: disable=protected-access

import json
from typing import Dict, List

import pytest  # type: ignore

from src.utils.exchange.hours import Hours
from src.utils.exchange.sessions_days import SessionsDays
from src.utils.exchange.sessions_hours import SessionsHours
from src.utils.exchange.stock_exchange import ExchangeConfig, StockExchange

weekdays = [
    "tuesday",
    "wednesday",
    "thursday",
    "sunday",
    "monday",
    "saturday",
    "friday",
]

valid_days = {
    "friday": True,
    "monday": True,
    "saturday": False,
    "sunday": False,
    "thursday": False,
    "tuesday": False,
    "wednesday": True,
}


def test_valid_stock_exchange():
    """Test creation of a valid StockExchange instance."""
    pre = Hours("08:00", "09:30")
    reg = Hours("09:30", "16:00")
    post = Hours("16:00", "18:00")
    tz = "America/New_York"
    sd = SessionsDays(valid_days, weekdays)
    sh = SessionsHours(pre, reg, post, tz)
    exchange = StockExchange(
        ExchangeConfig(
            code="NMS",
            country="US",
            sessions_days=sd,
            sessions_hours=sh,
            weekdays=weekdays,
        )
    )
    if exchange.code != "NMS":
        raise AssertionError("Expected code to be 'NMS'")
    if exchange.country != "US":
        raise AssertionError("Expected country to be 'US'")
    if exchange.sessions_hours is not sh:
        raise AssertionError("Expected sessions_hours to match input instance")
    received_str: str = json.dumps(exchange.to_json(), ensure_ascii=False)
    expected_str: str = (
        '{"code": "NMS", "country": "US", "sessions_days": {"monday": true, "tuesday": false, '
        + '"wednesday": true, "thursday": false, "friday": true, "saturday": false, "sunday": '
        + 'false}, "sessions_hours": {"pre_market": {"open": "08:00", "close": "09:30"}, '
        + '"post_market": {"open": "16:00", "close": "18:00"}, "regular": {"open": "09:30", '
        + '"close": "16:00"}, "timezone": "America/New_York"}}'
    )
    if received_str != expected_str:
        raise AssertionError(
            f"Expected json to be: {expected_str}. Received was: {received_str}"
        )


def test_stock_exchange_invalid_code():
    """Raise error for empty or non-string code."""
    sh = SessionsHours(
        Hours("00:00", "00:00"), Hours("00:00", "00:00"), Hours("00:00", "00:00"), "UTC"
    )
    sd = SessionsDays(valid_days, weekdays)
    with pytest.raises(ValueError, match="`code` must be a non-empty string"):
        StockExchange(
            ExchangeConfig(
                code="",
                country="USA",
                sessions_days=sd,
                sessions_hours=sh,
                weekdays=weekdays,
            )
        )
    with pytest.raises(ValueError, match="`code` must be a non-empty string"):
        StockExchange(
            ExchangeConfig(
                code=123,  # type: ignore
                country="USA",
                sessions_days=sd,
                sessions_hours=sh,
                weekdays=weekdays,
            )
        )


def test_get_validated_exchanges_none() -> None:
    """Raise when *exchanges* is ``None``."""
    with pytest.raises(ValueError, match="Parameter 'exchanges' is not defined"):
        StockExchange._get_validated_exchanges(None)  # type: ignore[arg-type]


def test_get_validated_exchanges_not_list() -> None:
    """Raise when *exchanges* is not a list instance."""
    with pytest.raises(ValueError, match="Parameter 'exchanges' is invalid"):
        StockExchange._get_validated_exchanges("not a list")  # type: ignore[arg-type]


def test_get_validated_exchanges_empty() -> None:
    """Raise when *exchanges* is an empty list."""
    with pytest.raises(ValueError, match="Parameter 'exchanges' is empty"):
        StockExchange._get_validated_exchanges([])


def test_get_validated_exchanges_only_none() -> None:
    """Raise when *exchanges* contains only ``None`` values."""
    with pytest.raises(
        ValueError, match="Parameter 'exchanges' contains only None values"
    ):
        StockExchange._get_validated_exchanges([None, None])  # type: ignore[list-item]


def test_get_validated_exchanges_valid() -> None:
    """Return cleaned list when at least one non-``None`` entry is present."""
    sample = [{"code": "NMS"}, None, {"code": "JPX"}]
    cleaned = StockExchange._get_validated_exchanges(sample)
    expected = [{"code": "NMS"}, {"code": "JPX"}]
    if cleaned != expected:
        raise AssertionError(
            "Expected non‑None items to be preserved in original order without alteration"
        )


def _make_exchange(
    code: str = "NMS",
    *,
    timezone: object = "America/New_York",
    sessions: object = None,
    country: object = "USA",
) -> Dict[str, object]:
    """Create a minimal exchange record for testing."""
    if sessions is None:
        sessions = {
            "regular": {
                "open": "09:30",
                "close": "16:00",
            }
        }
    return {
        "code": code,
        "timezone": timezone,
        "sessions_hours": sessions,
        "country": country,
    }


def test_find_default_duplicated_code() -> None:
    """Raise when *exchanges* contains duplicated codes."""
    data: List[dict] = [_make_exchange("NMS"), _make_exchange("NMS")]
    with pytest.raises(
        ValueError, match="Parameter 'exchanges' has duplicated items: NMS"
    ):
        StockExchange._find_validated_exchange_default("NMS", data)


def test_find_default_code_not_defined() -> None:
    """Raise when the requested code is not present in *exchanges*."""
    data = [_make_exchange("JPX")]
    with pytest.raises(
        ValueError, match="'NMS' is not defined in parameter 'exchanges'"
    ):
        StockExchange._find_validated_exchange_default("NMS", data)


@pytest.mark.parametrize(
    "timezone, expected",
    [
        (None, "Timezone of 'NMS' is not defined"),
        (123, "Timezone of 'NMS' is invalid"),
    ],
)
def test_find_default_invalid_timezone(timezone: object, expected: str) -> None:
    """Raise when *timezone* is ``None`` or not a ``str``."""
    data = [_make_exchange(timezone=timezone)]
    with pytest.raises(ValueError, match=expected):
        StockExchange._find_validated_exchange_default("NMS", data)


@pytest.mark.parametrize(
    "country, expected",
    [
        (None, "Country of 'NMS' is not defined"),
        (123, "Country of 'NMS' is invalid"),
    ],
)
def test_find_default_invalid_country(country: object, expected: str) -> None:
    """Raise when *country* is ``None`` or not a ``str``."""
    data = [_make_exchange(country=country)]
    with pytest.raises(ValueError, match=expected):
        StockExchange._find_validated_exchange_default("NMS", data)


def test_get_validated_exchange_default_code_none() -> None:
    """Raise when *exchange_default* is ``None``."""
    with pytest.raises(
        ValueError,
        match=r"Parameter 'exchange_default' is not defined\.",
    ):
        StockExchange._get_validated_exchange_default_code(None)  # type: ignore[arg-type]


def test_get_validated_exchange_default_code_not_str() -> None:
    """Raise when *exchange_default* is not a ``str``."""
    with pytest.raises(
        ValueError,
        match=r"Parameter 'exchange_default' is invalid: 123\.",
    ):
        StockExchange._get_validated_exchange_default_code(123)  # type: ignore[arg-type]


def test_get_validated_exchange_default_code_empty() -> None:
    """Raise when *exchange_default* is an empty or whitespace-only string."""
    with pytest.raises(
        ValueError,
        match=r"Parameter 'exchange_default' is empty\.",
    ):
        StockExchange._get_validated_exchange_default_code("   ")


def test_get_validated_exchange_default_code_valid() -> None:
    """Return uppercase trimmed code for a valid input string."""
    code = StockExchange._get_validated_exchange_default_code(" NMS ")
    if code != "NMS":
        raise AssertionError("Expected upper-case trimmed code 'NMS'")


@pytest.mark.parametrize(
    "segment, expected",
    [
        (
            {"open": 9, "close": "16:00"},  # open no-str
            r"Regular open hour of 'NMS' is invalid: '9'",
        ),
        (
            {"open": "09:30", "close": 1600},  # close no-str
            r"Regular close hour of 'NMS' is invalid: '1600'",
        ),
    ],
)
def test_extract_hours_invalid_open_close(segment: dict, expected: str) -> None:
    """Raise when open/close are not strings."""
    sessions = {"regular": segment}
    with pytest.raises(ValueError, match=expected):
        StockExchange._extract_hours(sessions, "regular", "NMS")


def test_extract_hours_missing_segment_returns_none() -> None:
    """If the key does not exist, the function must return ``None`` without raising an exception."""
    result = StockExchange._extract_hours(valid_days, "pre_market", "NMS")
    if result is not None:
        raise AssertionError("Expected None when the segment is not defined")


def test_from_parameter_missing_regular_raises() -> None:
    """Omit the *regular* segment ⇒ ValueError indicating the deficiency."""
    sessions: dict = {
        "pre_market": {"open": "07:00", "close": "09:29"},
        "post_market": {"open": "16:01", "close": "20:00"},
    }
    exch_list = [_make_exchange("NMS", sessions=sessions)]
    with pytest.raises(
        ValueError,
        match=r"The regular session of 'NMS' is not defined",
    ) as _exc_info:
        StockExchange.from_parameter("NMS", exch_list, weekdays)


def test_from_parameter_invalid_exchanges_type() -> None:
    """Passing a non‑list for *exchanges* propagates the underlying validation error."""
    with pytest.raises(
        ValueError,
        match=r"Parameter 'exchanges' is invalid",
    ):
        # type: ignore[arg-type]
        StockExchange.from_parameter("NMS", "not a list", weekdays)


def test_from_parameter_invalid_default_code() -> None:
    """Empty/whitespace default code is rejected early in the pipeline."""
    exch_list = [_make_exchange("NMS")]
    with pytest.raises(
        ValueError,
        match=r"Parameter 'exchange_default' is empty\.",
    ):
        StockExchange.from_parameter("   ", exch_list, weekdays)  # type: ignore[arg-type]


def test_find_default_sessions_hours_none() -> None:
    """Raise when *sessions_hours* is None."""
    exchange = _make_exchange()
    exchange["sessions_hours"] = None
    with pytest.raises(
        ValueError,
        match="The sessions hours of NMS are not defined",
    ):
        StockExchange._find_validated_exchange_default("NMS", [exchange])


def test_find_default_sessions_hours_not_dict() -> None:
    """Raise when *sessions_hours* is not a dict."""
    data = [_make_exchange(sessions="not-a-dict")]
    with pytest.raises(
        ValueError,
        match="The sessions hours of NMS are invalid: 'not-a-dict'",
    ):
        StockExchange._find_validated_exchange_default("NMS", data)


def test_stock_exchange_invalid_sessions_days() -> None:
    """Raise when sessions_days is not an instance of SessionsDays."""
    sh = SessionsHours(
        Hours("00:00", "00:00"), Hours("00:00", "00:00"), Hours("00:00", "00:00"), "UTC"
    )
    with pytest.raises(
        TypeError, match="`sessions_days` must be an instance of SessionsDays"
    ):
        StockExchange(
            ExchangeConfig(
                code="NMS",
                country="USA",
                sessions_days="invalid",  # type: ignore[arg-type]
                sessions_hours=sh,
                weekdays=weekdays,
            )
        )


def test_stock_exchange_invalid_sessions_hours() -> None:
    """Raise when sessions_hours is not an instance of SessionsHours."""
    sd = SessionsDays(valid_days, weekdays)
    with pytest.raises(
        TypeError, match="`sessions_hours` must be an instance of SessionsHours"
    ):
        StockExchange(
            ExchangeConfig(
                code="NMS",
                country="USA",
                sessions_days=sd,
                sessions_hours="invalid",  # type: ignore[arg-type]
                weekdays=weekdays,
            )
        )


def test_to_utc_returns_none_when_sessions_hours_is_none() -> None:
    """Return None when `sessions_hours.to_utc()` is None."""

    # pylint: disable=missing-class-docstring,super-init-not-called
    class StubSessionsHours(SessionsHours):
        """Stubbed subclass to simulate `to_utc()` returning None."""

        def __init__(self) -> None:
            self._pre_market = Hours("08:00", "09:30")
            self._regular = Hours("09:30", "16:00")
            self._post_market = Hours("16:00", "18:00")
            self._timezone = "America/New_York"

        def to_utc(self):
            return None

    # pylint: enable=missing-class-docstring,super-init-not-called
    sd = SessionsDays(valid_days, weekdays)
    sh = StubSessionsHours()
    exchange = StockExchange(
        ExchangeConfig(
            code="NMS",
            country="USA",
            sessions_days=sd,
            sessions_hours=sh,
            weekdays=weekdays,
        )
    )
    if exchange.to_utc() is not None:
        raise AssertionError(
            "Expected to_utc() to return None when sessions_hours.to_utc() is None"
        )
