"""
Unit tests for the SymbolProcessor class.

These tests validate the functionality of the symbol-level market data update pipeline,
including validation checks, enrichment logic, and symbol persistence. Covered scenarios:

- Validation of symbols using metadata availability.
- Logic to skip symbols based on existing and invalid data.
- Processing of individual symbols with and without new incremental data.
- Full batch processing of multiple symbols with tracking of update statistics.

Mocking is used extensively to isolate external dependencies such as Downloader, RawData,
Logger, Enrichment, and ScheduleBuilder to ensure deterministic behavior.
"""

# pylint: disable=protected-access

from unittest.mock import patch

import pandas as pd  # type: ignore
import pytest  # type: ignore

from market_data.symbol_processor import SymbolProcessor


@pytest.fixture(name="sample_entry")
def fixture_sample_entry():
    """
    Provide a sample symbol entry including metadata and historical prices.

    for use in single symbol processing tests.

    Returns:
        dict: Symbol metadata and historical price data.
    """
    return {
        "name": "Test Corp",
        "type": "Equity",
        "sector": "Technology",
        "industry": "Software",
        "currency": "USD",
        "exchange": "NASDAQ",
        "historical_prices": [
            {"datetime": "2024-01-01T00:00:00Z", "close": 100.0},
            {"datetime": "2024-01-02T00:00:00Z", "close": 101.0},
        ],
    }


@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_metadata")
def test_is_valid_symbol_success(mock_get_metadata):
    """Verify that _is_valid_symbol returns True when the metadata includes a valid name."""
    mock_get_metadata.return_value = {"name": "Valid Corp"}
    result = SymbolProcessor._is_valid_symbol("VALID")
    if result is not True:
        raise AssertionError("Expected valid symbol to return True")


@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_metadata")
def test_is_valid_symbol_invalid(mock_get_metadata):
    """Verify that _is_valid_symbol returns False when metadata retrieval raises an error."""
    mock_get_metadata.side_effect = KeyError("Missing")
    result = SymbolProcessor._is_valid_symbol("INVALID")
    if result is not False:
        raise AssertionError("Expected invalid symbol to return False")


@patch("market_data.symbol_processor.Logger.warning")
def test_should_skip_symbol_previously_invalid(mock_warn):
    """Ensure that _should_skip_symbol logs and skips a symbol already in the invalid set."""
    symbol = "FAIL"
    invalids = {symbol}
    result = SymbolProcessor._should_skip_symbol(symbol, {}, invalids)
    if not result:
        raise AssertionError("Expected symbol to be skipped if marked as invalid")
    if not any(symbol in str(call.args[0]) for call in mock_warn.call_args_list):
        raise AssertionError("Expected warning log for previously invalid symbol")


@patch("market_data.symbol_processor.SymbolProcessor._is_valid_symbol")
def test_should_skip_symbol_invalid(mock_valid):
    """
    Test that _should_skip_symbol returns True and tracks the symbol as invalid.

    when historical data is missing and metadata validation fails.
    """
    mock_valid.return_value = False
    invalids = set()
    result = SymbolProcessor._should_skip_symbol("FAKE", {}, invalids)
    if not result or "FAKE" not in invalids:
        raise AssertionError("Expected FAKE to be marked and skipped")


def test_should_skip_symbol_valid():
    """Verify that _should_skip_symbol returns False when historical_prices are present."""
    invalids = set()
    result = SymbolProcessor._should_skip_symbol(
        "REAL", {"historical_prices": [1]}, invalids
    )
    if result:
        raise AssertionError("Expected symbol with prices to not be skipped")


@patch("market_data.symbol_processor.SymbolProcessor._process_single_symbol")
@patch("market_data.symbol_processor.SymbolProcessor._should_skip_symbol")
@patch("market_data.symbol_processor.RawData.get_symbol")
@patch("market_data.symbol_processor.RawData.load")
@patch("market_data.symbol_processor.RawData.save")
def test_process_symbols_success(
    _mock_save, _mock_load, mock_get, mock_skip, mock_proc
):
    """
    Validate that process_symbols correctly updates statistics when one symbol.

    is successfully processed.
    """
    mock_get.return_value = {"historical_prices": [1]}
    mock_skip.return_value = False
    mock_proc.return_value = ("updated", "Test")

    result = SymbolProcessor.process_symbols(["TEST"], set())
    if result["updated"] != 1:
        raise AssertionError("Expected one updated symbol")


def test_process_symbols_skips_invalid_symbol():
    """
    Verify that process_symbols correctly skips a symbol when no historical data is present.

    and the symbol is determined to be invalid through metadata validation.

    This test ensures the symbol is counted as 'skipped' and added to the invalid_symbols set.
    """
    symbol = "INVALID"
    symbols = [symbol]
    invalid_symbols = set()

    with patch("market_data.symbol_processor.RawData.get_symbol", return_value=None):
        with patch.object(SymbolProcessor, "_is_valid_symbol", return_value=False):
            with patch("market_data.symbol_processor.RawData.load"), patch(
                "market_data.symbol_processor.RawData.save"
            ):
                counts = SymbolProcessor.process_symbols(symbols, invalid_symbols)

    if counts["skipped"] != 1:
        raise AssertionError(f"Expected 1 skipped symbol, got {counts['skipped']}")
    if symbol not in counts["invalid_symbols"]:
        raise AssertionError(f"Symbol {symbol} should be in invalid_symbols set")


@patch("market_data.symbol_processor.SymbolProcessor._process_single_symbol")
@patch(
    "market_data.symbol_processor.SymbolProcessor._should_skip_symbol",
    return_value=False,
)
@patch(
    "market_data.symbol_processor.RawData.get_symbol",
    return_value={"historical_prices": [1]},
)
@patch("market_data.symbol_processor.RawData.load")
@patch("market_data.symbol_processor.RawData.save")
@patch("market_data.symbol_processor.Logger.error")
def test_process_symbols_handles_exception(
    mock_logger, _mock_save, _mock_load, _mock_get, _mock_skip, mock_process
):
    """
    Ensure that process_symbols increments 'failed' when _process_single_symbol raises
    an exception.
    """
    mock_process.side_effect = RuntimeError("Unexpected error")
    result = SymbolProcessor.process_symbols(["BROKEN"], set())

    if result["failed"] != 1:
        raise AssertionError("Expected one failed symbol due to exception")
    if not any("BROKEN" in str(call.args[0]) for call in mock_logger.call_args_list):
        raise AssertionError("Expected error log for failed symbol")


@patch("market_data.symbol_processor.RawData.set_symbol")
@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_historical_prices")
def test_process_single_symbol_no_entry(mock_prices, _mock_set):
    """Should return no_new if no existing entry is provided and no new data is downloaded."""
    mock_prices.return_value = pd.DataFrame()
    result = SymbolProcessor._process_single_symbol("ABC", None)
    if result != ("no_new", ""):
        raise AssertionError("Expected result to be ('no_new', '')")


@patch("market_data.symbol_processor.RawData.set_symbol")
@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_historical_prices")
def test_process_single_symbol_existing_no_new_data(
    mock_prices, _mock_set, sample_entry
):
    """Should return no_new if all downloaded data already exists."""
    mock_prices.return_value = pd.DataFrame(sample_entry["historical_prices"])
    result = SymbolProcessor._process_single_symbol("XYZ", sample_entry)
    expected = ("no_new", sample_entry["name"])
    if result != expected:
        raise AssertionError(f"Expected result to be {expected}, got {result}")


@patch("market_data.symbol_processor.Logger.debug")
@patch("market_data.symbol_processor.RawData.set_symbol")
@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_metadata")
@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_historical_prices")
def test_process_single_symbol_fetch_metadata(
    mock_prices, mock_metadata, _mock_set, _mock_log
):
    """Should fetch metadata if missing and return updated."""
    entry = {
        "historical_prices": [
            {"datetime": "2024-01-01T00:00:00Z", "close": 100.0},
        ]
    }
    mock_prices.return_value = pd.DataFrame(
        [{"datetime": "2024-01-03T00:00:00Z", "close": 102.0}]
    )
    mock_metadata.return_value = {"name": "New Corp"}
    result = SymbolProcessor._process_single_symbol("NEW", entry)
    if result != ("updated", "New Corp"):
        raise AssertionError("Expected result to be ('updated', 'New Corp')")


@patch("market_data.symbol_processor.RawData.set_symbol")
@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_metadata")
@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_historical_prices")
def test_process_single_symbol_existing_empty_df(
    mock_prices, _mock_metadata, _mock_set
):
    """Should handle case where existing_df is empty."""
    entry = {"name": "Stub", "historical_prices": []}
    mock_prices.return_value = pd.DataFrame(
        [{"datetime": "2024-01-05T00:00:00Z", "close": 105.0}]
    )
    result = SymbolProcessor._process_single_symbol("EMPTY", entry)
    if result != ("updated", "Stub"):
        raise AssertionError("Expected result to be ('updated', 'Stub')")


@patch("market_data.symbol_processor.RawData.set_symbol")
@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_historical_prices")
def test_process_single_symbol_invalid_last_dt(mock_prices, _mock_set):
    """Ensure fallback to None if datetime is not parseable."""
    entry = {
        "name": "Invalid",
        "historical_prices": [
            {"datetime": None, "close": 100},
            {"datetime": "INVALID", "close": 101},
        ],
    }
    mock_prices.return_value = pd.DataFrame(
        [{"datetime": "2024-01-06T00:00:00Z", "close": 106.0}]
    )
    result = SymbolProcessor._process_single_symbol("BADTIME", entry)
    if result != ("updated", "Invalid"):
        raise AssertionError("Expected result to be ('updated', 'Invalid')")


@patch("market_data.symbol_processor.RawData.set_symbol")
@patch("market_data.symbol_processor.SymbolProcessor._DOWNLOADER.get_historical_prices")
def test_process_single_symbol_not_series_but_valid_timestamp(mock_prices, _mock_set):
    """Test else branch: datetime column is not a Series but a valid Timestamp."""
    entry = {
        "name": "Solo",
        "historical_prices": [
            {"datetime": "2024-01-01T00:00:00Z", "close": 100.0},
        ],
    }
    df = pd.DataFrame([{"datetime": "2024-01-02T00:00:00Z", "close": 101.0}])
    mock_prices.return_value = df
    result = SymbolProcessor._process_single_symbol("SOLO", entry)
    if result != ("updated", "Solo"):
        raise AssertionError("Expected result to be ('updated', 'Solo')")
