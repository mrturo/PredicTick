"""Unit tests for the OutputSuppressor utility."""

import io
import sys

import pytest  # type: ignore

from src.utils.io.logger import Logger
from src.utils.io.output_suppressor import OutputSuppressor


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

    result, _streams = OutputSuppressor.run_with_suppression(noisy_function, 3, 4)
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


def test_suppress_capture_captures_stdout_and_stderr():
    """Debe devolver los buffers y contener lo impreso."""
    with OutputSuppressor.suppress(capture=True) as (out_buf, err_buf):
        print("visible_in_out")
        sys.stderr.write("visible_in_err\n")
    out_val = out_buf.getvalue() if out_buf is not None else ""
    err_val = err_buf.getvalue() if err_buf is not None else ""
    if "visible_in_out" not in out_val:
        pytest.fail("stdout not captured as expected.")
    if "visible_in_err" not in err_val:
        pytest.fail("stderr not captured as expected.")


def test_run_with_suppression_capture_true():
    """run_with_suppression con capture=True regresa resultado y streams."""

    def noisy():
        print("foo")
        sys.stderr.write("bar\n")
        return 123

    result, (out_buf, err_buf) = OutputSuppressor.run_with_suppression(
        noisy, capture=True
    )
    if result != 123:
        pytest.fail(f"Expected 123 but got {result}")
    out_val = out_buf.getvalue() if out_buf is not None else ""
    err_val = err_buf.getvalue() if err_buf is not None else ""
    if "foo" not in out_val:
        pytest.fail("stdout not captured as expected.")
    if "bar" not in err_val:
        pytest.fail("stderr not captured as expected.")
