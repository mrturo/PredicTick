"""Unit tests for TickerMetadata dataclass utilities and construction."""

# pylint: disable=protected-access

import unittest
from typing import Optional, Union, get_args, get_origin

from src.market_data.ingestion.providers.ticker_metadata import TickerMetadata


class TestTickerMetadata(unittest.TestCase):
    """Full coverage for TickerMetadata."""

    # ---------- Low-level utilities ----------
    def test_snake_to_camel(self):
        """Should correctly convert snake_case strings to camelCase."""
        self.assertEqual(
            TickerMetadata._snake_to_camel("regular_market_price"),
            "regularMarketPrice",
        )
        self.assertEqual(TickerMetadata._snake_to_camel("a"), "a")
        self.assertEqual(TickerMetadata._snake_to_camel("two_words"), "twoWords")

    def test_safe_eval_success_and_type_match(self):
        """Should evaluate safe strings and return value if type matches."""
        self.assertEqual(TickerMetadata._safe_eval("[1, 2, 3]", list), [1, 2, 3])
        self.assertEqual(TickerMetadata._safe_eval("{'a': 1}", dict), {"a": 1})

    def test_safe_eval_wrong_type_or_invalid(self):
        """Should return None if evaluated type mismatches or string is invalid."""
        self.assertIsNone(TickerMetadata._safe_eval("[1,2]", dict))
        self.assertIsNone(TickerMetadata._safe_eval("not a literal", list))

    def test_extract_real_type_from_optional_union(self):
        """Should extract the actual type from Optional or return the same type."""
        opt_int = Optional[int]
        origin = get_origin(opt_int)
        self.assertEqual(origin, Union)
        self.assertIn(type(None), get_args(opt_int))
        self.assertIs(TickerMetadata._extract_real_type(opt_int), int)
        self.assertIs(TickerMetadata._extract_real_type(float), float)

    def test_parse_bool(self):
        """Should correctly parse boolean values from various formats."""
        self.assertTrue(TickerMetadata._parse_bool(True))
        self.assertTrue(TickerMetadata._parse_bool("true"))
        self.assertTrue(TickerMetadata._parse_bool("TrUe"))
        self.assertTrue(TickerMetadata._parse_bool("1"))
        self.assertTrue(TickerMetadata._parse_bool("yes"))
        self.assertFalse(TickerMetadata._parse_bool(False))
        self.assertFalse(TickerMetadata._parse_bool("false"))
        self.assertFalse(TickerMetadata._parse_bool("0"))
        self.assertFalse(TickerMetadata._parse_bool(""))
        self.assertFalse(TickerMetadata._parse_bool(0))
        self.assertTrue(TickerMetadata._parse_bool(2))

    def test_parse_list_direct_and_via_parse_value(self):
        """Should parse valid lists and handle invalid formats appropriately."""
        # Valid inputs
        self.assertEqual(TickerMetadata._parse_list([1, 2]), [1, 2])
        self.assertEqual(TickerMetadata._parse_list("[1, 2]"), [1, 2])
        # Invalid string → safe_eval returns None (no exception here)
        self.assertIsNone(TickerMetadata._parse_list("not a list"))
        # Non list/str → ValueError
        with self.assertRaises(ValueError):
            TickerMetadata._parse_list(123)  # type: ignore[arg-type]
        # Public path swallows errors and returns None
        self.assertIsNone(TickerMetadata._parse_value("not a list", Optional[list]))

    def test_parse_dict_direct_and_via_parse_value(self):
        """Should parse valid dicts and handle invalid formats appropriately."""
        # Valid inputs
        self.assertEqual(TickerMetadata._parse_dict({"a": 1}), {"a": 1})
        self.assertEqual(TickerMetadata._parse_dict("{'a': 1}"), {"a": 1})
        # Invalid string → safe_eval returns None (no exception here)
        self.assertIsNone(TickerMetadata._parse_dict("not a dict"))
        # Non dict/str → ValueError
        with self.assertRaises(ValueError):
            TickerMetadata._parse_dict(123)  # type: ignore[arg-type]
        # Public path swallows errors and returns None
        self.assertIsNone(TickerMetadata._parse_value("not a dict", Optional[dict]))

    def test_convert_value_branches(self):
        """Should cover all type conversion branches in _convert_value."""
        self.assertEqual(TickerMetadata._convert_value("10", int), 10)
        self.assertEqual(TickerMetadata._convert_value("3.14", float), 3.14)
        self.assertTrue(TickerMetadata._convert_value("true", bool))
        self.assertEqual(TickerMetadata._convert_value("[1]", list), [1])
        self.assertEqual(TickerMetadata._convert_value("{'k': 1}", dict), {"k": 1})
        self.assertEqual(TickerMetadata._convert_value("text", str), "text")

    def test_parse_value_none_and_empty_string(self):
        """Should return None if value is None or empty string."""
        self.assertIsNone(TickerMetadata._parse_value(None, Optional[int]))
        self.assertIsNone(TickerMetadata._parse_value("", Optional[float]))
        self.assertIsNone(TickerMetadata._parse_value("   ", Optional[float]))

    def test_parse_value_successful_conversions_and_error_swallow(self):
        """Should convert valid values and return None if conversion fails."""
        self.assertEqual(TickerMetadata._parse_value("42", Optional[int]), 42)
        self.assertAlmostEqual(TickerMetadata._parse_value("1.5", Optional[float]), 1.5)
        self.assertTrue(TickerMetadata._parse_value("yes", Optional[bool]))
        self.assertEqual(TickerMetadata._parse_value("[1,2]", Optional[list]), [1, 2])
        self.assertEqual(
            TickerMetadata._parse_value("{'a':2}", Optional[dict]),
            {"a": 2},
        )
        # int("NaN") raises ValueError internally → _parse_value returns None
        self.assertIsNone(TickerMetadata._parse_value("NaN", Optional[int]))

    # ---------- High-level constructor ----------
    def test_from_dict_maps_and_converts_all_supported_types(self):
        """Should map camelCase to snake_case; values stay as-is due to string annotations."""
        data = {
            "address1": "1 Infinite Loop",
            "city": "Cupertino",
            "fullTimeEmployees": "100000",
            "regularMarketPrice": "123.45",
            "tradeable": "true",
            "companyOfficers": "['CEO', 'CFO']",
            "corporateActions": "{'dividend': True}",
            "regularMarketDayRange": "120.0 - 125.0",
            "shortName": "AAPL",
            "symbol": "AAPL",
            "marketCap": "3000000000000",
            "averageAnalystRating": "Buy",
            "website": "",
            "unknownField": "ignore me",
        }
        tm = TickerMetadata.from_dict(data)
        # Mapped string fields
        self.assertEqual(tm.address1, "1 Infinite Loop")
        self.assertEqual(tm.city, "Cupertino")
        self.assertEqual(tm.short_name, "AAPL")
        self.assertEqual(tm.symbol, "AAPL")
        self.assertEqual(tm.average_analyst_rating, "Buy")
        self.assertEqual(tm.regular_market_day_range, "120.0 - 125.0")
        # Because dataclass annotations are strings (from __future__), types are not converted
        self.assertEqual(tm.full_time_employees, 100000)
        self.assertEqual(tm.regular_market_price, 123.45)
        self.assertEqual(tm.tradeable, True)
        self.assertEqual(tm.company_officers, ["CEO", "CFO"])
        self.assertEqual(tm.corporate_actions, {"dividend": True})
        self.assertEqual(tm.market_cap, 3000000000000)
        # Empty string handled early → None
        self.assertIsNone(tm.website)

    def test_from_dict_missing_keys_default_to_none(self):
        """Should return None for fields not present in the input dictionary."""
        tm = TickerMetadata.from_dict({})
        self.assertIsNone(tm.address1)
        self.assertIsNone(tm.volume)
        self.assertIsNone(tm.quote_type)
        self.assertIsNone(tm.triggerable)

    def test_repr_contains_class_name(self):
        """__repr__ should contain class name and important field values."""
        tm = TickerMetadata.from_dict({"shortName": "ACME"})
        text = repr(tm)
        self.assertIn("TickerMetadata", text)
        self.assertIn("short_name='ACME'", text)
