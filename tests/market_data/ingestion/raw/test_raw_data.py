"""Unit tests for RawData: symbol metadata and historical prices management."""

# pylint: disable=protected-access

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.market_data.ingestion.raw.raw_data import RawData  # type: ignore


class TestRawData(unittest.TestCase):
    """Full coverage tests for RawData behavior."""

    def setUp(self):
        """Reset RawData static state and inject deterministic parameters."""
        # Static state reset
        RawData.set_id(None)
        RawData.set_interval(None)
        RawData.set_last_check(None)
        RawData.set_last_updated(None)
        RawData.set_latest_price_date(None)
        RawData.set_stale_symbols([])
        RawData.set_symbols({})
        # Secure, isolated temp path for this test case
        # pylint: disable-next=consider-using-with
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.raw_path = str(Path(self._tmpdir.name) / "market_raw.json")
        # Deterministic params
        RawData._RAW_MARKETDATA_FILEPATH = self.raw_path  # noqa: SLF001
        RawData._STALE_DAYS_THRESHOLD = 2  # noqa: SLF001

        class FakeRepo:
            """Fake in-memory symbol repository for testing."""

            def __init__(self):
                """Initialize with empty invalid symbol set."""
                self._invalid = set()

            def get_invalid_symbols(self):
                """Return the current set of invalid symbols."""
                return set(self._invalid)

            def set_invalid_symbols(self, new_set):
                """Replace invalid symbols with a new set."""
                self._invalid = set(new_set)

        RawData._SYMBOL_REPO = FakeRepo()  # noqa: SLF001

    # -------------------- Simple getters/setters --------------------
    def test_id_get_set_and_new_id(self):
        """Should set/get id and generate a new random id."""
        self.assertIsNone(RawData.get_id())
        RawData.set_id("ABC123")
        self.assertEqual(RawData.get_id(), "ABC123")
        RawData.set_new_id()
        nid = RawData.get_id()
        self.assertIsInstance(nid, str)
        self.assertEqual(len(nid), 10)
        self.assertTrue(nid.isalnum())

    def test_interval_get_set(self):
        """Should set and get interval."""
        self.assertIsNone(RawData.get_interval())
        RawData.set_interval("1d")
        self.assertEqual(RawData.get_interval(), "1d")

    def test_timestamps_get_set(self):
        """Should set/get last_check, last_updated, latest_price_date."""
        ts1 = pd.Timestamp("2024-01-01T00:00:00Z")
        ts2 = pd.Timestamp("2024-01-02T00:00:00Z")
        ts3 = pd.Timestamp("2024-01-03T00:00:00Z")
        RawData.set_last_check(ts1)
        RawData.set_last_updated(ts2)
        RawData.set_latest_price_date(ts3)
        self.assertEqual(RawData.get_last_check(), ts1)
        self.assertEqual(RawData.get_last_updated(), ts2)
        self.assertEqual(RawData.get_latest_price_date(), ts3)

    def test_stale_symbols_and_symbols_get_set(self):
        """Should set/get stale symbols and symbol dictionary."""
        RawData.set_stale_symbols(["AAPL"])
        self.assertEqual(RawData.get_stale_symbols(), ["AAPL"])
        RawData.set_symbols({"AAPL": {"symbol": "AAPL"}})
        self.assertIn("AAPL", RawData.get_symbols())

    def test_set_and_get_single_symbol_structure(self):
        """Should build symbol entry structure from metadata and price list."""
        hist = [{"datetime": "2024-01-01T00:00:00Z", "close": 1.0}]
        meta = {
            "currency": "USD",
            "exchange": "NASDAQ",
            "industry": "Tech",
            "name": "Apple",
            "sector": "Technology",
            "type": "EQUITY",
        }
        RawData.set_symbol("AAPL", hist, meta)
        entry = RawData.get_symbol("AAPL")
        self.assertEqual(
            entry,
            {
                "currency": "USD",
                "exchange": "NASDAQ",
                "historical_prices": hist,
                "industry": "Tech",
                "name": "Apple",
                "sector": "Technology",
                "symbol": "AAPL",
                "type": "EQUITY",
            },
        )

    # -------------------- Utility helpers --------------------
    def test_get_filepath(self):
        """Should return default for None/empty, and the provided path otherwise."""
        self.assertEqual(
            RawData._get_filepath(None, "/default"), "/default"
        )  # noqa: SLF001
        self.assertEqual(
            RawData._get_filepath("   ", "/default"), "/default"
        )  # noqa: SLF001
        self.assertEqual(
            RawData._get_filepath("/custom", "/default"), "/custom"
        )  # noqa: SLF001

    def test_normalize_historical_prices_with_interpolation(self):
        """Should normalize datetimes and interpolate zero volume values."""
        symbols = [
            {
                "symbol": "AAA",
                "historical_prices": [
                    {"datetime": "2024-01-01T00:00:00Z", "volume": 100},
                    {"datetime": "2024-01-02T00:00:00Z", "volume": 0},
                    {"datetime": "2024-01-03T00:00:00Z", "volume": 300},
                ],
            }
        ]
        out = RawData.normalize_historical_prices(symbols)
        rows = out["AAA"]["historical_prices"]
        self.assertIsInstance(rows[0]["datetime"], pd.Timestamp)
        self.assertEqual(int(rows[1]["volume"]), 200)

    def test_normalize_historical_prices_without_volume_column(self):
        """Should normalize datetimes even if volume column does not exist."""
        symbols = [
            {
                "symbol": "BBB",
                "historical_prices": [
                    {"datetime": "2024-01-01T00:00:00Z", "close": 10.0},
                ],
            }
        ]
        out = RawData.normalize_historical_prices(symbols)
        rows = out["BBB"]["historical_prices"]
        self.assertIsInstance(rows[0]["datetime"], pd.Timestamp)
        self.assertIn("close", rows[0])

    # -------------------- JSON manager integration --------------------
    @patch("src.utils.io.json_manager.JsonManager.exists", return_value=True)
    def test_exist_uses_default_path(self, mock_exists):
        """Should call JsonManager.exists with the default filepath when None is passed."""
        self.assertTrue(RawData.exist())
        mock_exists.assert_called_once_with(self.raw_path)

    @patch("src.utils.io.json_manager.JsonManager.exists", return_value=False)
    def test_exist_with_custom_path(self, mock_exists):
        """Should call JsonManager.exists with a custom filepath when provided."""
        self.assertFalse(RawData.exist("/custom/path.json"))
        mock_exists.assert_called_once_with("/custom/path.json")

    # -------------------- load() scenarios --------------------
    @patch("src.utils.io.json_manager.JsonManager.exists", return_value=False)
    @patch("pandas.Timestamp.now")
    def test_load_when_file_absent_generates_id_and_uses_now(
        self, mock_now, _mock_exists
    ):
        """When file is absent, should generate id, set last_updated=now, leave interval None."""
        fixed_now = pd.Timestamp("2024-01-10T12:00:00Z")
        mock_now.return_value = fixed_now
        # Provide some invalids beforehand; load() should preserve same set via set_invalid_symbols
        RawData._SYMBOL_REPO.set_invalid_symbols({"BAD"})  # noqa: SLF001
        out = RawData.load()
        self.assertIsInstance(out["id"], str)
        self.assertEqual(out["last_check"], fixed_now)
        self.assertEqual(out["last_updated"], fixed_now)
        self.assertIsNone(out["interval"])
        self.assertEqual(out["symbols"], {})
        self.assertEqual(out["filepath"], self.raw_path)
        # No stale symbols for empty dataset
        self.assertEqual(RawData.get_stale_symbols(), [])

    @patch(
        "src.market_data.utils.intervals.interval.Interval.market_raw_data",
        return_value="1d",
    )
    @patch("src.utils.io.json_manager.JsonManager.load")
    @patch("src.utils.io.json_manager.JsonManager.exists", return_value=True)
    @patch("pandas.Timestamp.now")
    def test_load_when_file_present_parses_fields_and_filters_invalids(
        self, mock_now, _mock_exists, mock_load, _mock_interval
    ):
        """When file is present, should parse fields, set interval, normalize and filter invalids"""
        fixed_now = pd.Timestamp("2024-01-10T12:00:00Z")
        mock_now.return_value = fixed_now
        # Two symbols; one will be filtered as invalid
        raw_payload = {
            "id": "  RAW123  ",
            "last_updated": "2024-01-09T10:00:00Z",
            "last_check": "2024-01-10T11:00:00Z",
            "symbols": [
                {
                    "symbol": "KEEP",
                    "historical_prices": [
                        {"datetime": "2024-01-08T00:00:00Z", "volume": 0, "close": 1.0},
                        {
                            "datetime": "2024-01-09T00:00:00Z",
                            "volume": 100,
                            "close": 2.0,
                        },
                    ],
                },
                {
                    "symbol": "DROP",
                    "historical_prices": [
                        {
                            "datetime": "2023-12-31T00:00:00Z",
                            "volume": 50,
                            "close": 9.0,
                        },
                    ],
                },
            ],
        }
        mock_load.return_value = raw_payload
        RawData._SYMBOL_REPO.set_invalid_symbols({"DROP"})  # noqa: SLF001
        out = RawData.load()
        # id is trimmed and set; last_updated parsed from payload
        self.assertEqual(out["id"], "RAW123")
        self.assertEqual(out["last_check"], pd.Timestamp("2024-01-10T11:00:00Z"))
        self.assertEqual(out["last_updated"], pd.Timestamp("2024-01-09T10:00:00Z"))
        self.assertEqual(out["interval"], "1d")
        # Only KEEP remains after filtering invalids
        symbols = out["symbols"]
        self.assertListEqual(sorted(symbols.keys()), ["KEEP"])
        keep_rows = symbols["KEEP"]["historical_prices"]
        # Volume zero should be interpolated/bfilled → becomes 100 at first row
        self.assertEqual(int(keep_rows[0]["volume"]), 100)
        self.assertEqual(out["filepath"], self.raw_path)

    @patch(
        "src.market_data.utils.intervals.interval.Interval.market_raw_data",
        return_value="1d",
    )
    @patch("src.utils.io.json_manager.JsonManager.load")
    @patch("src.utils.io.json_manager.JsonManager.exists", return_value=True)
    @patch("pandas.Timestamp.now")
    def test_load_when_missing_keys_defaults_applied_and_new_id_created(
        self, mock_now, _mock_exists, mock_load, _mock_interval
    ):
        """Missing keys should default gracefully; empty id triggers new random id."""
        fixed_now = pd.Timestamp("2024-01-10T12:00:00Z")
        mock_now.return_value = fixed_now
        # Deliberately missing keys and empty id
        raw_payload = {
            "id": "   ",
            "symbols": [],
        }
        mock_load.return_value = raw_payload
        RawData._SYMBOL_REPO.set_invalid_symbols(set())  # noqa: SLF001
        out = RawData.load()
        # New id generated, last_updated set to now when id was empty
        self.assertIsInstance(out["id"], str)
        self.assertEqual(out["last_check"], fixed_now)
        self.assertEqual(out["last_updated"], fixed_now)
        self.assertEqual(out["interval"], "1d")
        self.assertEqual(out["symbols"], {})

    # -------------------- save() --------------------
    @patch("src.utils.io.json_manager.JsonManager.save")
    def test_save_writes_expected_payload(self, mock_save):
        """Should serialize current state and call JsonManager.save with payload."""
        # Prepare state
        RawData.set_id("ID1")
        RawData.set_interval("1d")
        RawData.set_last_updated(pd.Timestamp("2024-01-09T10:00:00Z"))
        RawData.set_symbols(
            {
                "X": {
                    "symbol": "X",
                    "historical_prices": [{"datetime": "2024-01-01T00:00:00Z"}],
                },
                "Y": {
                    "symbol": "Y",
                    "historical_prices": [{"datetime": "2024-01-02T00:00:00Z"}],
                },
            }
        )
        result = RawData.save()
        # Validate returned structure
        self.assertEqual(result["id"], "ID1")
        self.assertEqual(result["last_updated"], pd.Timestamp("2024-01-09T10:00:00Z"))
        self.assertEqual(result["interval"], "1d")
        # last_check is an ISO string; ensure parseable
        self.assertIsInstance(pd.to_datetime(result["last_check"]), pd.Timestamp)
        # Symbols serialized as list of dicts (values of the symbols map)
        self.assertEqual({e["symbol"] for e in result["symbols"]}, {"X", "Y"})
        # Ensure save was called with same payload and default path
        args, kwargs = mock_save.call_args
        self.assertEqual(args[1], self.raw_path)  # filepath
        self.assertEqual(args[0]["id"], "ID1")
        self.assertEqual({e["symbol"] for e in args[0]["symbols"]}, {"X", "Y"})
        self.assertEqual(kwargs, {})

    # -------------------- stale detection --------------------
    def test_detect_stale_symbols_logic(self):
        """Should mark symbols stale when their latest date is older than threshold."""
        # Prepare symbols: OLD is stale, FRESH is within threshold
        RawData.set_symbols(
            {
                "OLD": {
                    "historical_prices": [
                        {"datetime": "2024-01-01T00:00:00Z"},
                        {"datetime": "2024-01-02T00:00:00Z"},
                    ]
                },
                "FRESH": {
                    "historical_prices": [
                        {"datetime": "2024-01-08T00:00:00Z"},
                        {"datetime": "2024-01-10T00:00:00Z"},
                    ]
                },
                "EMPTY": {"historical_prices": []},
            }
        )
        latest = pd.Timestamp("2024-01-12T00:00:00Z")
        stale, updated_invalids = RawData._detect_stale_symbols(  # noqa: SLF001
            latest, {"BAD"}
        )
        self.assertCountEqual(stale, ["OLD", "EMPTY"])
        self.assertTrue({"OLD", "EMPTY"}.issubset(updated_invalids))
        self.assertIn("BAD", updated_invalids)

    @patch.object(
        RawData, "_detect_stale_symbols", return_value=(["AAA"], {"AAA", "ZZZ"})
    )
    @patch("pandas.Timestamp.now")
    def test_update_stale_symbols_uses_detection_and_updates_repo(
        self, mock_now, mock_detect
    ):
        """Should update stale list and propagate invalids to symbol repo."""
        RawData._SYMBOL_REPO.set_invalid_symbols({"ZZZ"})  # noqa: SLF001
        RawData.set_latest_price_date(pd.Timestamp("2024-02-01T00:00:00Z"))
        # Call method under test
        RawData._update_stale_symbols()  # noqa: SLF001
        # Stale list updated
        self.assertEqual(RawData.get_stale_symbols(), ["AAA"])
        # Repo updated with returned set
        self.assertEqual(
            RawData._SYMBOL_REPO.get_invalid_symbols(), {"AAA", "ZZZ"}
        )  # noqa: SLF001
        # Timestamp.now should not be needed because latest_price_date was set
        mock_now.assert_not_called()
        mock_detect.assert_called_once()

    @patch(
        "src.market_data.utils.intervals.interval.Interval.market_raw_data",
        return_value="1d",
    )
    @patch("src.utils.io.json_manager.JsonManager.load")
    @patch("src.utils.io.json_manager.JsonManager.exists", return_value=True)
    @patch("pandas.Timestamp.now")
    def test_load_with_non_str_id_and_non_list_symbols(
        self, mock_now, _mock_exists, mock_load, _mock_interval
    ):
        """Non-string id and non-list symbols should default to new id and empty symbols map."""
        fixed_now = pd.Timestamp("2024-01-10T12:00:00Z")
        mock_now.return_value = fixed_now
        # id no string y symbols no lista
        raw_payload = {
            "id": 12345,
            "last_check": "2024-01-09T10:00:00Z",
            "symbols": "not-a-list",
        }
        mock_load.return_value = raw_payload
        RawData._SYMBOL_REPO.set_invalid_symbols(set())  # noqa: SLF001
        out = RawData.load()
        self.assertIsInstance(out["id"], str)
        self.assertEqual(out["last_check"], pd.Timestamp("2024-01-09T10:00:00Z"))
        self.assertEqual(out["last_updated"], fixed_now)
        self.assertEqual(out["interval"], "1d")
        self.assertEqual(out["symbols"], {})

    @patch(
        "src.market_data.utils.intervals.interval.Interval.market_raw_data",
        return_value="1d",
    )
    @patch("src.utils.io.json_manager.JsonManager.load")
    @patch("src.utils.io.json_manager.JsonManager.exists", return_value=True)
    @patch("pandas.Timestamp.now")
    def test_load_keyerror_on_id_and_symbols_defaults_applied(
        self, mock_now, _mock_exists, mock_load, _mock_interval
    ):
        """Missing 'id' and 'symbols' keys should default to new id and empty symbols map."""
        fixed_now = pd.Timestamp("2024-01-10T12:00:00Z")
        mock_now.return_value = fixed_now
        raw_payload = {
            "last_check": "2024-01-10T11:00:00Z",
            "last_updated": "2024-01-09T10:00:00Z",
        }
        mock_load.return_value = raw_payload
        RawData._SYMBOL_REPO.set_invalid_symbols(set())  # noqa: SLF001
        out = RawData.load()
        self.assertIsInstance(out["id"], str)
        self.assertEqual(out["last_check"], pd.Timestamp("2024-01-10T11:00:00Z"))
        self.assertEqual(out["last_updated"], fixed_now)
        self.assertEqual(out["interval"], "1d")
        self.assertEqual(out["symbols"], {})
        self.assertEqual(out["filepath"], self.raw_path)

    @patch(
        "src.market_data.utils.intervals.interval.Interval.market_raw_data",
        return_value="1d",
    )
    @patch("src.utils.io.json_manager.JsonManager.load")
    @patch("src.utils.io.json_manager.JsonManager.exists", return_value=True)
    @patch("pandas.Timestamp.now")
    def test_load_typeerror_when_raw_data_is_non_mapping(
        self, mock_now, _mock_exists, mock_load, _mock_interval
    ):
        """Non-mapping raw_data should default all fields and initialize empty symbols map."""
        fixed_now = pd.Timestamp("2024-01-10T12:00:00Z")
        mock_now.return_value = fixed_now
        mock_load.return_value = 42
        RawData._SYMBOL_REPO.set_invalid_symbols(set())  # noqa: SLF001
        out = RawData.load()
        self.assertIsInstance(out["id"], str)
        self.assertEqual(out["last_check"], fixed_now)
        self.assertEqual(out["last_updated"], fixed_now)
        self.assertEqual(out["interval"], "1d")
        self.assertEqual(out["symbols"], {})
        self.assertEqual(out["filepath"], self.raw_path)
