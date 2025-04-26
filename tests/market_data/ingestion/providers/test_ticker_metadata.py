"""Unit tests for the TickerMetadata dataclass."""

import unittest
from typing import Any, Dict

# Adjust this import to your real project path if needed.
from src.market_data.ingestion.providers.ticker_metadata import TickerMetadata


class TestTickerMetadata(unittest.TestCase):
    """Test suite for construction and parsing behavior of TickerMetadata."""

    def test_from_dict_parses_basic_types(self):
        """from_dict should convert basic types (int, float, bool, str)."""
        raw: Dict[str, Any] = {
            "symbol": "AAPL",
            "fullTimeEmployees": "100",
            "previousClose": "123.45",
            "tradeable": "true",
            "triggerable": 0,
            "website": "https://example.com",
        }
        meta = TickerMetadata.from_dict(raw)

        self.assertEqual(meta.symbol, "AAPL")
        self.assertIsInstance(meta.full_time_employees, int)
        self.assertEqual(meta.full_time_employees, 100)
        self.assertIsInstance(meta.previous_close, float)
        self.assertAlmostEqual(meta.previous_close or 0.0, 123.45, places=6)
        self.assertIs(meta.tradeable, True)
        self.assertIs(meta.triggerable, False)
        self.assertEqual(meta.website, "https://example.com")

    def test_from_dict_parses_collections_from_strings(self):
        """from_dict should parse list and dict from strings via literal_eval."""
        raw: Dict[str, Any] = {
            "companyOfficers": "['CEO', 'CFO']",
            "corporateActions": "{'split': '2:1'}",
        }
        meta = TickerMetadata.from_dict(raw)
        self.assertIsInstance(meta.company_officers, list)
        self.assertListEqual(meta.company_officers or [], ["CEO", "CFO"])
        self.assertIsInstance(meta.corporate_actions, dict)
        self.assertEqual((meta.corporate_actions or {}).get("split"), "2:1")

    def test_from_dict_handles_empty_and_none_as_none(self):
        """Empty strings or None should become None."""
        raw: Dict[str, Any] = {
            "website": "",
            "industry": "   ",
            "fax": None,
        }
        meta = TickerMetadata.from_dict(raw)
        self.assertIsNone(meta.website)
        self.assertIsNone(meta.industry)
        self.assertIsNone(meta.fax)

    def test_from_dict_invalid_numbers_yield_none(self):
        """Non-numeric strings for numeric fields should yield None."""
        raw: Dict[str, Any] = {
            "volume": "abc",
            "marketCap": "12.34.56",
            "regularMarketTime": "1_700_000_000x",
        }
        meta = TickerMetadata.from_dict(raw)
        self.assertIsNone(meta.volume)
        self.assertIsNone(meta.market_cap)
        self.assertIsNone(meta.regular_market_time)

    def test_from_dict_invalid_collections_yield_none(self):
        """Invalid strings for list/dict should yield None."""
        raw: Dict[str, Any] = {
            "executiveTeam": "not-a-list",
            "corporateActions": "not-a-dict",
        }
        meta = TickerMetadata.from_dict(raw)
        self.assertIsNone(meta.executive_team)
        self.assertIsNone(meta.corporate_actions)

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys should be ignored."""
        raw: Dict[str, Any] = {"unknownKey": "value", "symbol": "MSFT"}
        meta = TickerMetadata.from_dict(raw)
        self.assertEqual(meta.symbol, "MSFT")
        self.assertFalse(hasattr(meta, "unknownKey"))

    def test_camel_case_mapping_is_applied(self):
        """camelCase keys should map to snake_case fields."""
        raw: Dict[str, Any] = {
            "regularMarketPreviousClose": "321.0",
            "regularMarketOpen": "320",
            "fiftyTwoWeekHigh": "400.5",
            "fiftyTwoWeekLow": "250.25",
            "twoHundredDayAverageChangePercent": "0.1234",
            "fiveYearAvgDividendYield": "1.23",
        }
        meta = TickerMetadata.from_dict(raw)
        self.assertAlmostEqual(
            meta.regular_market_previous_close or 0.0, 321.0, places=6
        )
        self.assertEqual(meta.regular_market_open, 320.0)
        self.assertEqual(meta.fifty_two_week_high, 400.5)
        self.assertEqual(meta.fifty_two_week_low, 250.25)
        self.assertEqual(meta.two_hundred_day_average_change_percent, 0.1234)
        self.assertEqual(meta.five_year_avg_dividend_yield, 1.23)

    def test_boolean_parsing_variants(self):
        """Different boolean representations should parse correctly."""
        raw_true: Dict[str, Any] = {
            "cryptoTradeable": "Yes",
            "hasPrePostOld": "1",
            "esgPopulated": True,
        }
        raw_false: Dict[str, Any] = {
            "cryptoTradeable": "no",
            "hasPrePostOld": "0",
            "esgPopulated": 0,
        }
        meta_true = TickerMetadata.from_dict(raw_true)
        meta_false = TickerMetadata.from_dict(raw_false)
        self.assertIs(meta_true.crypto_tradeable, True)
        self.assertIs(meta_true.has_pre_post_old, True)
        self.assertIs(meta_true.esg_populated, True)
        self.assertIs(meta_false.crypto_tradeable, False)
        self.assertIs(meta_false.has_pre_post_old, False)
        self.assertIs(meta_false.esg_populated, False)

    def test_repr_contains_class_and_selected_fields(self):
        """repr should include class name and selected fields."""
        raw: Dict[str, Any] = {
            "symbol": "TSLA",
            "shortName": "Tesla",
            "currentPrice": "250.0",
        }
        meta = TickerMetadata.from_dict(raw)
        text = repr(meta)
        self.assertIn("TickerMetadata", text)
        self.assertIn("symbol='TSLA'", text)
        self.assertIn("short_name='Tesla'", text)
        self.assertIn("current_price=250.0", text)

    def test_missing_keys_default_to_none(self):
        """Missing keys should initialize to None."""
        meta = TickerMetadata.from_dict({})
        self.assertIsNone(meta.symbol)
        self.assertIsNone(meta.long_name)
        self.assertIsNone(meta.volume)
        self.assertIsNone(meta.company_officers)
        self.assertIsNone(meta.corporate_actions)

    def test_numeric_strings_with_spaces_are_parsed(self):
        """Numeric strings with spaces should be parsed correctly."""
        raw: Dict[str, Any] = {
            "marketCap": "  1500000000  ",
            "targetMeanPrice": "  300.75 ",
        }
        meta = TickerMetadata.from_dict(raw)
        self.assertEqual(meta.market_cap, 1_500_000_000.0)
        self.assertEqual(meta.target_mean_price, 300.75)

    def test_day_range_and_ranges_remain_strings(self):
        """Fields that are typed as str should remain strings."""
        raw: Dict[str, Any] = {
            "regularMarketDayRange": "120.3 - 125.8",
            "fiftyTwoWeekRange": "100.0 - 200.0",
        }
        meta = TickerMetadata.from_dict(raw)
        self.assertEqual(meta.regular_market_day_range, "120.3 - 125.8")
        self.assertEqual(meta.fifty_two_week_range, "100.0 - 200.0")
