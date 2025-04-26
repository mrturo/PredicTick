"""Utility to suppress stdout and stderr output for noisy operations."""

import contextlib
import io


class OutputSuppressor:
    """Context manager to suppress stdout and stderr."""

    @staticmethod
    @contextlib.contextmanager
    def suppress():
        """Suppress stdout and stderr temporarily."""
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            yield

    @staticmethod
    def run_with_suppression(func, *args, **kwargs):
        """Run a function while suppressing stdout and stderr."""
        with OutputSuppressor.suppress():
            return func(*args, **kwargs)
