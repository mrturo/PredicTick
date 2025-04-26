"""Utility to (optionally) suppress and/or capture stdout / stderr."""

import contextlib
import io
from collections.abc import Callable
from typing import Any, Iterator, Optional, Tuple


class OutputSuppressor:
    """Context manager to silence stdout/stderr and, if requested, capture them."""

    @staticmethod
    @contextlib.contextmanager
    def suppress(
        capture: bool = False,
    ) -> Iterator[Tuple[Optional[io.StringIO], Optional[io.StringIO]]]:
        """Suppress stdout and stderr temporarily (optionally capturando)."""
        if capture:
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(
                buf_err
            ):
                yield buf_out, buf_err  # ─── uso dentro del with
            # Al salir del contexto rebobinamos para que el caller lea desde el inicio
            buf_out.seek(0)
            buf_err.seek(0)
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                yield None, None

    @staticmethod
    def run_with_suppression(
        func: Callable[..., Any],
        *args: Any,
        capture: bool = False,
        **kwargs: Any,
    ) -> Tuple[Any, Tuple[Optional[io.StringIO], Optional[io.StringIO]]]:
        """Executes *func* while suppressing (and optionally capturing) the output."""
        with OutputSuppressor.suppress(capture=capture) as streams:
            result = func(*args, **kwargs)
        return result, streams
