"""Unit tests for the custom Logger utility."""

import re

import pytest

from utils.logger import Logger


def check_eq(actual, expected, message=""):
    """Raise AssertionError if actual != expected, with optional message."""
    if actual != expected:
        raise AssertionError(message or f"Expected {expected}, got {actual}")


@pytest.mark.parametrize(
    "method,symbol",
    [
        (Logger.info, "ℹ️  INFO "),
        (Logger.warning, "⚠️  WARN "),
        (Logger.error, "❌ ERROR"),
        (Logger.success, "✅ SUCC "),
        (Logger.debug, "🐞DEBUG "),
        (Logger.simple, "        "),
    ],
)
def test_logger_output_format(method, symbol, capsys):
    """Validate log output format includes timestamp, level symbol, and message."""
    message = "  test message"
    method(message)
    captured = capsys.readouterr().out.strip()
    timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    check_eq(
        re.match(
            rf"^{timestamp_pattern} \| {re.escape(symbol)} \|   test message$", captured
        )
        is not None,
        True,
        f"Log output did not match expected pattern. Got: {captured}",
    )


def test_logger_separator_default_length(capsys):
    """Test default separator line length is 100 asterisks."""
    Logger.separator()
    captured = capsys.readouterr().out.strip()
    check_eq(captured, "*" * 100)


def test_logger_separator_custom_length(capsys):
    """Test custom separator length is respected."""
    Logger.separator(50)
    captured = capsys.readouterr().out.strip()
    check_eq(captured, "*" * 50)


def test_format_message_output():
    """Test the internal message formatting logic."""
    result = Logger.format_message("TEST", "  indented message")
    check_eq(result.endswith("| TEST |   indented message"), True)


def test_check_eq_function():
    """Test that check_eq passes for equal values."""
    check_eq(1, 1)


def test_check_eq_fail():
    """Test that check_eq raises AssertionError for different values."""
    with pytest.raises(AssertionError, match="Expected 2, got 1"):
        check_eq(1, 2)
