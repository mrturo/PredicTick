"""
Unit tests for the Summarizer class.

This module validates the summarization logic used for logging symbol update stats and
for exporting symbol metadata and historical price data into CSV files. It covers:

- Logging behavior of `print_sumary`.
- File output validation for `export_symbol_summary_to_csv`.
- Detailed data export checks for `export_symbol_detailed_to_csv`.

Mocks are used to isolate filesystem and Gateway/Google Drive interactions,
ensuring deterministic behavior and file outputs are verified via temporary paths.
"""

# pylint: disable=protected-access

import time
from unittest.mock import patch

import pandas as pd

from market_data.summarizer import Summarizer


def test_print_summary_logs(monkeypatch):
    """Test that print_summary logs expected values."""
    symbols = ["AAPL", "MSFT", "GOOG"]
    processed = {"updated": 2, "skipped": 0, "no_new": 1, "failed": 0}
    start_time = time.time() - 65

    logs = []

    def mock_debug(msg):
        logs.append(msg)

    def mock_error(msg):
        logs.append(msg)

    monkeypatch.setattr("utils.logger.Logger.debug", mock_debug)
    monkeypatch.setattr("utils.logger.Logger.error", mock_error)

    Summarizer.print_sumary(symbols, processed, start_time)

    expected = [
        "Summary",
        "  * Symbols processed: 3",
        "  * Symbols updated: 2",
        "  * Symbols skipped: 0",
        "  * Symbols with no new data: 1",
    ]
    for e in expected:
        if not any(e in log for log in logs):
            raise AssertionError(f"Missing log entry: {e}")


@patch("market_data.gateway.Gateway.get_symbols")
@patch("market_data.gateway.Gateway.load")
@patch("utils.google_drive_manager.GoogleDriveManager.upload_file")
def test_export_symbol_summary_to_csv(
    _mock_upload, _mock_load, mock_get_symbols, tmp_path, symbol_metadata
):
    """Test export_symbol_summary_to_csv writes correct CSV."""
    sample_data = {
        "AAPL": {
            **symbol_metadata,
            "historical_prices": [
                {"datetime": "2023-01-01T00:00:00Z"},
                {"datetime": "2023-01-02T00:00:00Z"},
            ],
            "schedule": {
                "monday": {
                    "all_day": False,
                    "min_open": "09:30:00",
                    "max_close": "16:00:00",
                },
            },
        }
    }

    mock_get_symbols.return_value = sample_data
    mock_summary_path = tmp_path / "summary.csv"

    Summarizer._MARKETDATA_SUMMARY_FILEPATH = str(mock_summary_path)
    Summarizer.export_symbol_summary_to_csv()

    df = pd.read_csv(mock_summary_path)
    if df.shape[0] != 1:
        raise AssertionError("Expected 1 row in summary CSV")
    if "symbol" not in df.columns:
        raise AssertionError("Missing 'symbol' column in summary CSV")


@patch("market_data.gateway.Gateway.get_symbols")
@patch("market_data.gateway.Gateway.load")
@patch("utils.google_drive_manager.GoogleDriveManager.upload_file")
def test_export_symbol_detailed_to_csv(
    _mock_upload, _mock_load, mock_get_symbols, tmp_path
):
    """Test export_symbol_detailed_to_csv writes detailed price data."""
    sample_data = {
        "AAPL": {
            "historical_prices": [
                {"datetime": "2023-01-01T00:00:00Z", "close": 150.0},
                {"datetime": "2023-01-02T00:00:00Z", "close": 152.0},
            ]
        }
    }

    mock_get_symbols.return_value = sample_data
    mock_detailed_path = tmp_path / "detailed.csv"

    Summarizer._MARKETDATA_DETAILED_FILEPATH = str(mock_detailed_path)
    Summarizer.export_symbol_detailed_to_csv()

    df = pd.read_csv(mock_detailed_path)
    if df.shape[0] != 2:
        raise AssertionError("Expected 2 rows in detailed CSV")
    if "symbol" not in df.columns:
        raise AssertionError("Missing 'symbol' column in detailed CSV")


def test_print_summary_logs_with_failures(monkeypatch):
    """Test that print_summary logs error when failures are present."""
    symbols = ["AAPL", "MSFT"]
    processed = {"updated": 1, "skipped": 0, "no_new": 0, "failed": 1}
    start_time = time.time() - 45

    logs = []

    def mock_debug(msg):
        logs.append(("debug", msg))

    def mock_error(msg):
        logs.append(("error", msg))

    monkeypatch.setattr("utils.logger.Logger.debug", mock_debug)
    monkeypatch.setattr("utils.logger.Logger.error", mock_error)

    Summarizer.print_sumary(symbols, processed, start_time)

    error_logs = [log for level, log in logs if level == "error"]
    if not any("Symbols failed: 1" in msg for msg in error_logs):
        raise AssertionError("Expected error log for failed symbols")
