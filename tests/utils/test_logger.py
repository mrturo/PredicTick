"""Unit tests for the custom Logger utility."""

import pytest

from utils.logger import Logger


def check_eq(actual, expected, message=""):
    """Raise AssertionError if actual != expected, with optional message."""
    if actual != expected:
        raise AssertionError(message or f"Expected {expected}, got {actual}")


@pytest.mark.parametrize(
    "method,symbol",
    [
        (Logger.info, "🔵 INFO "),
        (Logger.warning, "🟡 WARN "),
        (Logger.error, "🔴 ERROR"),
        (Logger.success, "🟢 SUCC "),
        (Logger.debug, "⚫ DEBUG"),
    ],
)
def test_logger_output_format(method, symbol):
    """Validate the formatted log message structure."""
    message = "  test message"
    Logger.logger = None  # Reset to force re-initialization
    formatted = method(message)

    expected = f"{symbol} |   test message"
    check_eq(formatted, expected, f"Got: '{formatted}', Expected: '{expected}'")


def test_logger_separator_default_length():
    """Test default separator line length is 100 asterisks."""
    formatted = Logger.separator()
    check_eq(formatted.endswith("*" * 100), True)


def test_logger_separator_custom_length():
    """Test custom separator length is respected."""
    formatted = Logger.separator(50)
    check_eq(formatted.endswith("*" * 50), True)


def test_format_message_output():
    """Test the internal message formatting logic."""
    result = Logger.log("TEST", "  indented message")
    check_eq(result.endswith("TEST |   indented message"), True)


def test_check_eq_function():
    """Test that check_eq passes for equal values."""
    check_eq(1, 1)


def test_check_eq_fail():
    """Test that check_eq raises AssertionError for different values."""
    with pytest.raises(AssertionError, match="Expected 2, got 1"):
        check_eq(1, 2)
