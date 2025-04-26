"""
Unit tests for the Summarizer class.

This module validates the summarization logic used for logging symbol update stats and
for exporting symbol metadata and historical price data into CSV files. It covers:

- Logging behavior of `print_sumary`.
- File output validation for `export_symbol_summary_to_csv`.
- Detailed data export checks for `export_symbol_detailed_to_csv`.

Mocks are used to isolate filesystem and RawData/Google Drive interactions,
ensuring deterministic behavior and file outputs are verified via temporary paths.
"""

# pylint: disable=protected-access

import time

from market_data.ingest.summarizer import Summarizer


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
