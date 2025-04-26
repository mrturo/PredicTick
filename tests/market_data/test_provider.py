"""Unit tests for the Yahoo Finance Provider interface."""

# pylint: disable=protected-access

from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd  # type: ignore
import pytest  # type: ignore

from market_data.provider import (
    PriceDataConfig,
    Provider,
    TickerMetadata,
    _convert_value,
    _extract_real_type,
    _parse_bool,
    _parse_dict,
    _parse_list,
    _safe_eval,
    parse_value,
    snake_to_camel,
)

tm = TickerMetadata.from_dict


def test_snake_to_camel_conversion():
    """Test conversion from snake_case to camelCase."""

    result_1 = snake_to_camel("fifty_two_week_low")
    if result_1 != "fiftyTwoWeekLow":
        raise AssertionError("Expected 'fiftyTwoWeekLow', got: " + result_1)

    result_2 = snake_to_camel("eps")
    if result_2 != "eps":
        raise AssertionError("Expected 'eps', got: " + result_2)

    result_3 = snake_to_camel("market_cap")
    if result_3 != "marketCap":
        raise AssertionError("Expected 'marketCap', got: " + result_3)


def test_price_data_config_defaults():
    """Test default values in PriceDataConfig."""
    config = PriceDataConfig(symbols="AAPL")
    if config.symbols != "AAPL":
        raise AssertionError("Expected symbols to be 'AAPL'")
    if config.interval != "1d":
        raise AssertionError("Expected interval to be '1d'")
    if config.group_by != "ticker":
        raise AssertionError("Expected group_by to be 'ticker'")
    if config.auto_adjust is not True:
        raise AssertionError("Expected auto_adjust to be True")


def test_ticker_metadata_from_dict_partial():
    """Test TickerMetadata.from_dict handles partial data correctly."""
    input_data = {
        "longName": "Apple Inc.",
        "industry": "Technology",
        "marketCap": 1000000000,
        "symbol": "AAPL",
    }
    metadata = TickerMetadata.from_dict(input_data)

    if metadata.long_name != "Apple Inc.":
        raise AssertionError("Expected long_name to be 'Apple Inc.'")
    if metadata.industry != "Technology":
        raise AssertionError("Expected industry to be 'Technology'")
    if metadata.market_cap != 1000000000:
        raise AssertionError("Expected market_cap to be 1000000000")
    if metadata.symbol != "AAPL":
        raise AssertionError("Expected symbol to be 'AAPL'")
    if metadata.city is not None:
        raise AssertionError("Expected city to be None")


@patch("market_data.provider.yf.Ticker")
def test_provider_get_metadata(mock_yf_ticker):
    """Test get_metadata returns TickerMetadata from mocked ticker.info."""
    mock_info = {
        "longName": "Mock Corp",
        "marketCap": 999999999,
        "industry": "Mock Industry",
        "symbol": "MOCK",
    }
    mock_ticker = MagicMock()
    mock_ticker.info = mock_info
    mock_yf_ticker.return_value = mock_ticker

    provider = Provider()
    result = provider.get_metadata("MOCK")

    if not isinstance(result, TickerMetadata):
        raise AssertionError("Expected result to be instance of TickerMetadata")
    if result.long_name != "Mock Corp":
        raise AssertionError("Expected long_name to be 'Mock Corp'")
    if result.market_cap != 999999999:
        raise AssertionError("Expected market_cap to be 999999999")
    if result.industry != "Mock Industry":
        raise AssertionError("Expected industry to be 'Mock Industry'")


@patch("market_data.provider.yf.download")
def test_provider_get_price_data(mock_yf_download):
    """Test get_price_data calls yfinance.download with expected parameters."""
    mock_df = pd.DataFrame({"Open": [100], "Close": [105]})
    mock_yf_download.return_value = mock_df

    config = PriceDataConfig(
        symbols=["AAPL", "MSFT"],
        start="2023-01-01",
        end="2023-01-10",
        interval="1d",
        proxy="http://proxy",
    )

    provider = Provider()
    result = provider.get_price_data(config)

    if not isinstance(result, pd.DataFrame):
        raise AssertionError("Expected result to be a DataFrame")
    if "Open" not in result.columns:
        raise AssertionError("Expected column 'Open' in result")
    if "Close" not in result.columns:
        raise AssertionError("Expected column 'Close' in result")
    if not mock_yf_download.called:
        raise AssertionError("Expected yfinance.download to be called")

    _args, kwargs = mock_yf_download.call_args
    if kwargs["tickers"] != ["AAPL", "MSFT"]:
        raise AssertionError(
            f"Expected tickers to be ['AAPL', 'MSFT'], got {kwargs['tickers']}"
        )
    if kwargs["interval"] != "1d":
        raise AssertionError(f"Expected interval to be '1d', got {kwargs['interval']}")
    if kwargs["auto_adjust"] is not True:
        raise AssertionError("Expected auto_adjust to be True")


def test_safe_eval_valid_cases():
    """Test _safe_eval returns valid parsed data when input is safe and matches expected type."""
    result = _safe_eval("[1, 2, 3]", list)
    if result != [1, 2, 3]:
        raise AssertionError(f"Expected [1, 2, 3], got {result}")
    result = _safe_eval("{'a': 1}", dict)
    if result != {"a": 1}:
        raise AssertionError(f"Expected {{'a': 1}}, got {result}")


def test_safe_eval_invalid_cases():
    """Test _safe_eval returns None for unsafe or mismatched input strings."""
    if _safe_eval("not_a_list", list) is not None:
        raise AssertionError("Expected None for invalid list eval")
    if _safe_eval("123", dict) is not None:
        raise AssertionError("Expected None for invalid dict eval")
    if _safe_eval("__import__('os').system('rm -rf /')", dict) is not None:
        raise AssertionError("Expected None for dangerous code eval")


def test_convert_value_types():
    """Test _convert_value handles conversion from string to various data types correctly."""
    if _convert_value("123", int) != 123:
        raise AssertionError("Expected 123")
    if _convert_value("123.45", float) != 123.45:
        raise AssertionError("Expected 123.45")
    if _convert_value("true", bool) is not True:
        raise AssertionError("Expected True for 'true'")
    if _convert_value("[1, 2]", list) != [1, 2]:
        raise AssertionError("Expected [1, 2]")
    if _convert_value("{'x': 5}", dict) != {"x": 5}:
        raise AssertionError("Expected {'x': 5}")
    if _convert_value("some string", str) != "some string":
        raise AssertionError("Expected 'some string'")


def test_parse_bool_various():
    """Test _parse_bool returns correct boolean values for various input representations."""
    for val, expected in [
        (True, True),
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        (0, False),
        (1, True),
    ]:
        result = _parse_bool(val)
        if result is not expected:
            raise AssertionError(f"Expected {expected} for {val}, got {result}")


def test_parse_list_various():
    """Test _parse_list handles valid list input and raises ValueError on invalid data."""
    result = _parse_list([1, 2])
    if result != [1, 2]:
        raise AssertionError(f"Expected [1, 2], got {result}")
    result = _parse_list("[3, 4]")
    if result != [3, 4]:
        raise AssertionError(f"Expected [3, 4], got {result}")
    with pytest.raises(ValueError):
        _parse_list(123)


def test_parse_dict_various():
    """Test _parse_dict handles valid dictionary input and raises ValueError on invalid data."""
    result = _parse_dict({"a": 1})
    if result != {"a": 1}:
        raise AssertionError(f"Expected {{'a': 1}}, got {result}")
    result = _parse_dict("{'b': 2}")
    if result != {"b": 2}:
        raise AssertionError(f"Expected {{'b': 2}}, got {result}")
    with pytest.raises(ValueError):
        _parse_dict([1, 2])


def test_parse_value_invalid_conversion():
    """Ensure parse_value returns None on failed conversion."""
    result = parse_value("not_a_number", int)
    if result is not None:
        raise AssertionError("Expected None for invalid int conversion")

    result = parse_value("not_a_dict", dict)
    if result is not None:
        raise AssertionError("Expected None for invalid dict conversion")


def test_extract_real_type_optional_union():
    """Ensure _extract_real_type returns actual type from Optional."""
    typ = Optional[int]
    result = _extract_real_type(typ)
    if result is not int:
        raise AssertionError(f"Expected int, got {result}")


def test_from_dict_skips_non_init_fields():
    """Ensure from_dict skips fields where f.init is False or name is _field_mapping."""

    class Dummy:  # pylint: disable=R0903
        """Callable dummy class used to verify field skipping in from_dict logic."""

        def __init__(self):
            self.called = False

        def __call__(self, *args, **kwargs):
            self.called = True

    original_parse_value = parse_value
    dummy = Dummy()

    try:
        globals()["parse_value"] = dummy

        class DummyMeta(TickerMetadata):  # pylint: disable=R0903
            """Subclass of TickerMetadata used to test from_dict ignores special/non-init fields."""

            _field_mapping: dict = None  # type: ignore

        _ = DummyMeta.from_dict({})
    finally:
        globals()["parse_value"] = original_parse_value
