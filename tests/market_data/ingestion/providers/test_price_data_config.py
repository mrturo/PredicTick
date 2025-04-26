"""Unit tests for the PriceDataConfig dataclass."""

import unittest
from typing import List

from src.market_data.ingestion.providers.price_data_config import \
    PriceDataConfig


class TestPriceDataConfig(unittest.TestCase):
    """Test suite validating construction and behavior of PriceDataConfig."""

    def test_minimal_initialization_uses_defaults(self):
        """When only symbols is provided, all other fields should use defaults."""
        cfg = PriceDataConfig(symbols="AAPL")

        self.assertEqual(cfg.symbols, "AAPL")
        self.assertIsNone(cfg.start)
        self.assertIsNone(cfg.end)
        self.assertEqual(cfg.interval, "1d")
        self.assertEqual(cfg.group_by, "ticker")
        self.assertTrue(cfg.auto_adjust)
        self.assertFalse(cfg.prepost)
        self.assertTrue(cfg.threads)
        self.assertIsNone(cfg.proxy)
        self.assertTrue(cfg.progress)

    def test_symbols_accepts_string(self):
        """symbols should accept a single ticker as string."""
        cfg = PriceDataConfig(symbols="MSFT")
        self.assertIsInstance(cfg.symbols, str)
        self.assertEqual(cfg.symbols, "MSFT")

    def test_symbols_accepts_list_of_strings(self):
        """symbols should accept a list of tickers."""
        tickers: List[str] = ["AAPL", "GOOG", "TSLA"]
        cfg = PriceDataConfig(symbols=tickers)
        self.assertIsInstance(cfg.symbols, list)
        self.assertListEqual(cfg.symbols, tickers)

    def test_full_initialization_overrides_defaults(self):
        """All fields should be set to provided values."""
        cfg = PriceDataConfig(
            symbols=["AAPL", "AMZN"],
            start="2020-01-01",
            end="2020-12-31",
            interval="1h",
            group_by="column",
            auto_adjust=False,
            prepost=True,
            threads=False,
            proxy="http://localhost:8080",
            progress=False,
        )
        self.assertListEqual(cfg.symbols, ["AAPL", "AMZN"])
        self.assertEqual(cfg.start, "2020-01-01")
        self.assertEqual(cfg.end, "2020-12-31")
        self.assertEqual(cfg.interval, "1h")
        self.assertEqual(cfg.group_by, "column")
        self.assertFalse(cfg.auto_adjust)
        self.assertTrue(cfg.prepost)
        self.assertFalse(cfg.threads)
        self.assertEqual(cfg.proxy, "http://localhost:8080")
        self.assertFalse(cfg.progress)

    def test_mutability_of_fields(self):
        """Dataclass fields should be mutable (not frozen)."""
        cfg = PriceDataConfig(symbols="AAPL")
        cfg.symbols = ["AAPL", "MSFT"]  # type: ignore[assignment]
        cfg.interval = "1wk"
        cfg.auto_adjust = False
        self.assertIsInstance(cfg.symbols, list)
        self.assertEqual(cfg.interval, "1wk")
        self.assertFalse(cfg.auto_adjust)

    def test_type_hints_match_union_for_symbols(self):
        """Runtime check: symbols should be either str or list[str]."""
        cfg_str = PriceDataConfig(symbols="IBM")
        cfg_list = PriceDataConfig(symbols=["IBM", "ORCL"])
        self.assertIsInstance(cfg_str.symbols, (str, list))
        self.assertIsInstance(cfg_list.symbols, (str, list))

        # Explicit union-style guard
        def _is_valid_symbols(value: object) -> bool:
            return isinstance(value, str) or (
                isinstance(value, list) and all(isinstance(v, str) for v in value)
            )

        self.assertTrue(_is_valid_symbols(cfg_str.symbols))
        self.assertTrue(_is_valid_symbols(cfg_list.symbols))

    def test_repr_contains_field_names(self):
        """Dataclasses provide a helpful repr including field names/values."""
        cfg = PriceDataConfig(symbols="AAPL", interval="1d")
        text = repr(cfg)
        self.assertIn("PriceDataConfig", text)
        for field in (
            "symbols",
            "start",
            "end",
            "interval",
            "group_by",
            "auto_adjust",
            "prepost",
            "threads",
            "proxy",
            "progress",
        ):
            self.assertIn(field, text)
