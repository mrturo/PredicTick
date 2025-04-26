"""Unit tests for Summarizer.print_sumary."""

# pylint: disable=protected-access

import unittest
from unittest.mock import call, patch

# Import target with loose typing to avoid mypy/pyright issues in user projects
from src.market_data.ingestion.summarizers.summarizer import \
    Summarizer  # type: ignore


class TestSummarizer(unittest.TestCase):
    """Full coverage tests for Summarizer.print_sumary logging behavior."""

    def setUp(self):
        """Neutralize external dependencies initialized as class attrs."""
        # Avoid any side effects from instantiated dependencies
        Summarizer._GOOGLE_DRIVE = object()  # noqa: SLF001
        Summarizer._WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]  # noqa: SLF001

    @patch("time.time", return_value=1000 + 125)  # start_time=1000 → 2m 5s
    @patch("src.utils.io.logger.Logger.error")
    @patch("src.utils.io.logger.Logger.debug")
    def test_print_summary_logs_expected_lines_without_failures(
        self, mock_debug, mock_error, _mock_time
    ):
        """Should log summary lines via Logger.debug and never call Logger.error."""
        symbols = ["AAA", "BBB", "CCC"]
        processed = {"updated": 2, "skipped": 1, "no_new": 0, "failed": 0}
        start_time = 1000

        Summarizer.print_sumary(symbols, processed, start_time)

        # Validate debug calls content and order
        expected_debug_calls = [
            call("Summary"),
            call("  * Symbols processed: 3"),
            call("  * Symbols updated: 2"),
            call("  * Symbols skipped: 1"),
            call("  * Symbols with no new data: 0"),
            call("  Total time: 2m 5s"),
        ]
        self.assertEqual(mock_debug.call_args_list, expected_debug_calls)
        mock_error.assert_not_called()

    @patch("time.time", return_value=5000 + 61)  # start_time=5000 → 1m 1s
    @patch("src.utils.io.logger.Logger.error")
    @patch("src.utils.io.logger.Logger.debug")
    def test_print_summary_logs_error_when_failed_positive(
        self, mock_debug, mock_error, _mock_time
    ):
        """Should log an error line when there are failed symbols (>0)."""
        symbols = ["X", "Y"]
        processed = {"updated": 0, "skipped": 1, "no_new": 1, "failed": 3}
        start_time = 5000

        Summarizer.print_sumary(symbols, processed, start_time)

        # Last debug call must be the total time
        self.assertEqual(mock_debug.call_args_list[-1], call("  Total time: 1m 1s"))
        # Error should be called exactly once with failed count
        mock_error.assert_called_once_with("  * Symbols failed: 3")

    @patch("time.time", return_value=42)  # start_time=42 → 0m 0s
    @patch("src.utils.io.logger.Logger.error")
    @patch("src.utils.io.logger.Logger.debug")
    def test_print_summary_handles_empty_input_and_zero_durations(
        self, mock_debug, mock_error, _mock_time
    ):
        """Should handle empty symbol list and zero counts without errors."""
        symbols = []
        processed = {"updated": 0, "skipped": 0, "no_new": 0, "failed": 0}
        start_time = 42

        Summarizer.print_sumary(symbols, processed, start_time)

        # Ensure processed count reflects empty list and time shows 0m 0s
        self.assertIn(call("  * Symbols processed: 0"), mock_debug.call_args_list)
        self.assertIn(call("  Total time: 0m 0s"), mock_debug.call_args_list)
        mock_error.assert_not_called()
