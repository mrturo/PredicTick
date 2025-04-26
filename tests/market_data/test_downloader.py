"""
Unit tests for the Downloader class in the market_data module.

These tests cover core methods such as get_price_data, get_metadata,
is_symbol_available, and get_historical_prices. The tests use mocking
to isolate behavior from external data providers and ensure deterministic behavior.
"""

# pylint: disable=protected-access

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from market_data.downloader import Downloader
from market_data.provider import TickerMetadata


@pytest.fixture(name="ticker_fixture")
def sample_ticker():
    """
    Provide a sample TickerMetadata instance for use in metadata tests.

    Returns:
        TickerMetadata: A populated instance representing a financial instrument.
    """
    return TickerMetadata(
        symbol="AAPL",
        display_name="Apple Inc.",
        short_name="Apple",
        long_name="Apple Incorporated",
        type_disp="Equity",
        quote_type="EQ",
        sector="Technology",
        sector_key="Tech",
        sector_disp="Tech Display",
        industry="Consumer Electronics",
        industry_key="Electronics",
        industry_disp="Electronics Display",
        currency="USD",
        financial_currency="USD",
        exchange="NASDAQ",
    )


@patch("market_data.downloader.Downloader._PROVIDER")
@patch("utils.output_suppressor.OutputSuppressor.suppress")
@patch("utils.timezone_normalizer.TimezoneNormalizer.localize_to_market_time")
def test_get_price_data_success(mock_localize, _mock_suppress, mock_provider):
    """
    Test that get_price_data returns a valid non-empty DataFrame
    when the provider supplies valid price data.
    """
    df_mock = pd.DataFrame(
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
        data={"close": [150, 152]},
    )
    df_mock.index.name = "datetime"
    mock_provider.get_price_data.return_value = df_mock
    mock_localize.return_value = df_mock

    dl = Downloader(block_days=1, retries=1, sleep_seconds=0)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    result = dl.get_price_data("AAPL", start, end, "1d")

    if result.empty:
        raise AssertionError("Expected non-empty DataFrame")
    if not isinstance(result, pd.DataFrame):
        raise AssertionError("Result should be a DataFrame")


@patch("market_data.downloader.Downloader._PROVIDER")
def test_get_metadata_success(mock_provider, ticker_fixture):
    """
    Test that get_metadata correctly extracts and returns
    metadata fields from a mocked TickerMetadata instance.
    """
    mock_provider.get_metadata.return_value = ticker_fixture

    dl = Downloader(block_days=1, retries=1, sleep_seconds=0)
    result = dl.get_metadata("AAPL")

    if result["name"] != "Apple Inc.":
        raise AssertionError("Incorrect name")
    if result["sector"] != "Tech":
        raise AssertionError("Incorrect sector fallback logic")
    if result["currency"] != "USD":
        raise AssertionError("Incorrect currency fallback logic")


@patch("market_data.downloader.Downloader.get_price_data")
def test_is_symbol_available_true(mock_get_price):
    """
    Test that is_symbol_available returns True
    when recent price data is available for a symbol.
    """
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        index=pd.date_range(
            start=now - timedelta(days=1), periods=1, freq="1h", tz="UTC"
        ),
        data={"close": [150.0]},
    )
    mock_get_price.return_value = df

    dl = Downloader(block_days=1, retries=1, sleep_seconds=0)
    result = dl.is_symbol_available("AAPL")

    if not result:
        raise AssertionError("Expected symbol to be available")


@patch("market_data.downloader.Downloader.get_price_data")
def test_get_historical_prices_single_block(mock_get_price):
    """
    Test get_historical_prices returns non-empty data when the period fits within a single block.
    """
    now = datetime.now(timezone.utc)
    sample_data = pd.DataFrame(
        index=pd.date_range(now - timedelta(days=1), periods=2, freq="h"),
        data={
            "adj_close": [150, 151],
            "close": [150, 151],
            "high": [151, 152],
            "low": [149, 150],
            "open": [150, 150],
            "volume": [1000, 1200],
        },
    )
    mock_get_price.return_value = sample_data

    dl = Downloader(block_days=2, retries=1, sleep_seconds=0)
    result = dl.get_historical_prices("AAPL", now - timedelta(days=1))

    if result.empty:
        raise AssertionError("Expected historical prices not to be empty")
    if "datetime" not in result.columns:
        raise AssertionError("Expected 'datetime' column in results")


@patch("market_data.downloader.Downloader._PROVIDER")
@patch("utils.output_suppressor.OutputSuppressor.suppress")
def test_get_price_data_exception_logged(_mock_suppress, mock_provider):
    """
    Test that get_price_data handles exceptions and logs an error, returning an empty DataFrame.
    """
    mock_provider.get_price_data.side_effect = ValueError("API error")

    dl = Downloader(block_days=1, retries=1, sleep_seconds=0)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    result = dl.get_price_data("AAPL", start, end, "1d")

    if not result.empty:
        raise AssertionError("Expected empty DataFrame on exception")


@patch("market_data.downloader.Downloader._PROVIDER")
def test_get_metadata_exception_logged(mock_provider):
    """
    Test that get_metadata handles exceptions and returns default metadata.
    """
    mock_provider.get_metadata.side_effect = KeyError("Metadata not found")

    dl = Downloader(block_days=1, retries=1, sleep_seconds=0)
    result = dl.get_metadata("INVALID")

    expected = {
        "name": None,
        "type": None,
        "sector": None,
        "industry": None,
        "currency": "USD",
        "exchange": None,
    }
    if result != expected:
        raise AssertionError(f"Expected default metadata but got {result}")


@patch("market_data.downloader.Downloader.get_price_data")
def test_is_symbol_available_handles_exception(mock_get_price):
    """
    Test that is_symbol_available handles exceptions gracefully and returns False.
    """
    mock_get_price.side_effect = ValueError("Data error")

    dl = Downloader(block_days=1, retries=1, sleep_seconds=0)
    result = dl.is_symbol_available("ERRSYM")

    if result:
        raise AssertionError("Expected availability check to return False on error")


@patch("market_data.downloader.Downloader.get_price_data")
def test_get_historical_prices_no_data(mock_get_price):
    """
    Test get_historical_prices returns an empty DataFrame when no data is fetched.
    """
    mock_get_price.return_value = pd.DataFrame()

    dl = Downloader(block_days=1, retries=1, sleep_seconds=0)
    result = dl.get_historical_prices("AAPL", None)

    if not result.empty:
        raise AssertionError("Expected empty DataFrame when no data fetched")


@patch("market_data.downloader.Logger.warning")
@patch("market_data.downloader.Downloader._PROVIDER")
def test_get_metadata_no_metadata_logs_warning(mock_provider, mock_warning):
    """
    Test that get_metadata logs a warning and returns default metadata
    when no metadata is found for the symbol.
    """
    mock_provider.get_metadata.return_value = None  # Simula que no hay metadata

    dl = Downloader(block_days=1, retries=1, sleep_seconds=0)
    result = dl.get_metadata("UNKNOWN")

    expected = {
        "name": None,
        "type": None,
        "sector": None,
        "industry": None,
        "currency": dl._DEFAULT_CURRENCY,
        "exchange": None,
    }

    # Verifica que se retorne el metadata por defecto
    if result != expected:
        raise AssertionError(f"Expected default metadata but got {result}")

    # Verifica que Logger.warning fue llamado correctamente
    if not mock_warning.called:
        raise AssertionError(
            "Expected Logger.warning to be called when metadata is missing"
        )

    args, _ = mock_warning.call_args
    if "No metadata found for UNKNOWN" not in args[0]:
        raise AssertionError("Logger.warning message does not match expected output")
