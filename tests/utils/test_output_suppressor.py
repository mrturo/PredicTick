"""Unit tests for the OutputSuppressor utility."""

import io
import sys

import pytest

from utils.logger import Logger
from utils.output_suppressor import OutputSuppressor


def test_suppress_stdout_and_stderr(monkeypatch):
    """Test that stdout and stderr are suppressed within the context."""
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_stdout)
    monkeypatch.setattr(sys, "stderr", captured_stderr)

    with OutputSuppressor.suppress():
        Logger.warning("This should not be seen")

    out = captured_stdout.getvalue()
    err = captured_stderr.getvalue()

    if "This should not be seen" in out:
        pytest.fail("Unexpected stdout output found.")
    if "Error message" in err:
        pytest.fail("Unexpected stderr output found.")


def test_run_with_suppression_returns_function_result():
    """Test that run_with_suppression returns the function's result."""

    def noisy_function(x, y):
        return x + y

    result = OutputSuppressor.run_with_suppression(noisy_function, 3, 4)
    if result != 7:
        pytest.fail(f"Expected 7 but got {result}")


def test_run_with_suppression_suppresses_output(monkeypatch):
    """Test that run_with_suppression suppresses printed output."""
    captured_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_stdout)

    def noisy_function():
        Logger.debug("noisy print")

    OutputSuppressor.run_with_suppression(noisy_function)

    out = captured_stdout.getvalue()
    if "noisy print" in out:
        pytest.fail("Unexpected output was not suppressed.")
