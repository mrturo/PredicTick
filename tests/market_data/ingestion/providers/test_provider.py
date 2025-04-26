"""Unit tests for Provider: Yahoo Finance-based market data access."""

import io
import os
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd  # type: ignore

from src.market_data.ingestion.providers.provider import \
    Provider  # type: ignore


class _DummyCtx:
    """Helper context manager to mimic OutputSuppressor.suppress()."""

    def __init__(self, out_text: str = "", err_text: Optional[str] = ""):
        self._out = io.StringIO(out_text)
        self._err = None if err_text is None else io.StringIO(err_text)

    def __enter__(self):
        return self._out, self._err

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class TestProvider(unittest.TestCase):
    """Full coverage tests for Provider behavior."""

    # -------------------- get_metadata --------------------
    @patch("yfinance.Ticker")
    @patch.object(
        # Ensure we don't depend on actual TickerMetadata implementation
        target=__import__(
            "src.market_data.ingestion.providers.ticker_metadata",
            fromlist=["TickerMetadata"],
        ).TickerMetadata,
        attribute="from_dict",
        autospec=True,
    )
    def test_get_metadata_uses_yfinance_and_factory(self, mock_from_dict, mock_ticker):
        """Should fetch ticker.info and build TickerMetadata via from_dict()."""
        mock_instance = MagicMock()
        mock_instance.info = {"symbol": "AAPL", "exchange": "NASDAQ"}
        mock_ticker.return_value = mock_instance
        sentinel_meta = object()
        mock_from_dict.return_value = sentinel_meta
        out = Provider().get_metadata("AAPL")
        mock_ticker.assert_called_once_with("AAPL")
        mock_from_dict.assert_called_once_with({"symbol": "AAPL", "exchange": "NASDAQ"})
        self.assertIs(out, sentinel_meta)

    # -------------------- get_price_data --------------------
    @patch("yfinance.download")
    @patch(
        "src.utils.io.output_suppressor.OutputSuppressor.suppress",
        return_value=_DummyCtx(out_text="", err_text=""),
    )
    def test_get_price_data_single_symbol_sets_proxy_and_group_by_column(
        self, _mock_suppress, mock_download
    ):
        """When symbols is str, group_by must be 'column' and proxy env vars set."""
        df = pd.DataFrame(
            {"Close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        mock_download.return_value = df
        config = SimpleNamespace(
            symbols="AAPL",
            start="2024-01-01",
            end="2024-01-31",
            interval="1d",
            group_by="ticker",  # should be ignored → 'column'
            auto_adjust=True,
            prepost=False,
            threads=True,
            progress=False,
            proxy="http://proxy.local:8080",
        )
        with patch.dict(os.environ, {}, clear=True):
            out = Provider().get_price_data(config)
            self.assertIs(out, df)
            self.assertEqual(os.environ["HTTP_PROXY"], "http://proxy.local:8080")
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://proxy.local:8080")
        mock_download.assert_called_once_with(
            tickers="AAPL",
            start="2024-01-01",
            end="2024-01-31",
            interval="1d",
            group_by="column",  # forced for single symbol
            auto_adjust=True,
            prepost=False,
            threads=True,
            progress=False,
        )

    @patch("yfinance.download")
    @patch(
        "src.utils.io.output_suppressor.OutputSuppressor.suppress",
        return_value=_DummyCtx(out_text="", err_text="Some error message\n"),
    )
    def test_get_price_data_raises_value_error_on_stderr(
        self, _mock_suppress, mock_download
    ):
        """Non-empty stderr from yfinance must raise ValueError with stripped message."""
        mock_download.return_value = pd.DataFrame()
        config = SimpleNamespace(
            symbols=["AAPL", "MSFT"],
            start=None,
            end=None,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            prepost=True,
            threads=False,
            progress=True,
            proxy=None,
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, r"^Some error message$"):
                Provider().get_price_data(config)
            self.assertNotIn("HTTP_PROXY", os.environ)
            self.assertNotIn("HTTPS_PROXY", os.environ)
        mock_download.assert_called_once_with(
            tickers=["AAPL", "MSFT"],
            start=None,
            end=None,
            interval="1d",
            group_by="ticker",  # preserved for multi-symbol
            auto_adjust=False,
            prepost=True,
            threads=False,
            progress=True,
        )

    @patch("yfinance.download")
    @patch(
        "src.utils.io.output_suppressor.OutputSuppressor.suppress",
        return_value=_DummyCtx(out_text="ignored", err_text=None),
    )
    def test_get_price_data_handles_none_stderr_object(
        self, _mock_suppress, mock_download
    ):
        """If stderr buffer is None, should not raise and return the DataFrame."""
        df = pd.DataFrame(
            {"Close": [10.0]}, index=pd.date_range("2024-01-01", periods=1)
        )
        mock_download.return_value = df
        config = SimpleNamespace(
            symbols="MSFT",
            start="2024-01-01",
            end="2024-01-10",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            prepost=False,
            threads=True,
            progress=False,
            proxy=None,
        )
        out = Provider().get_price_data(config)
        self.assertIs(out, df)
        mock_download.assert_called_once_with(
            tickers="MSFT",
            start="2024-01-01",
            end="2024-01-10",
            interval="1d",
            group_by="column",
            auto_adjust=True,
            prepost=False,
            threads=True,
            progress=False,
        )
