"""Unit tests for the Yahoo Finance Provider interface."""

# pylint: disable=protected-access

from unittest.mock import MagicMock, patch

import pandas as pd

from market_data.provider import (
    PriceDataConfig,
    Provider,
    TickerMetadata,
    snake_to_camel,
)


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
