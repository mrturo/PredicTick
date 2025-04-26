"""Unit tests for the FileManager class in the updater module."""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from updater.file_manager import FileManager


def test_last_update_when_never_set_returns_none():
    """Test that last_update returns None when not initialized."""
    result = FileManager.last_update()
    if result is not None:
        pytest.fail("Expected last_update to return None when never set")


def test_update_and_find_symbol():
    """Test update_symbol followed by find_symbol returns correct data."""
    historical = [{"Datetime": "2024-01-01T00:00:00Z", "Close": 100.0}]
    metadata = {
        "Name": "Test Corp",
        "Type": "Equity",
        "Sector": "Tech",
        "Currency": "USD",
        "Exchange": "NASDAQ",
    }

    FileManager.update_symbol("TEST", historical, metadata)
    symbol_data = FileManager.find_symbol("TEST")

    if symbol_data["symbol"] != "TEST":
        pytest.fail("Symbol should match")
    if symbol_data["name"] != "Test Corp":
        pytest.fail("Name should match")
    if symbol_data["historical_prices"] != historical:
        pytest.fail("Historical prices should match")


def test_load_valid_file():
    """Test FileManager.load loads valid JSON file properly."""
    sample_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "stocks": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "type": "Equity",
                "sector": "Tech",
                "currency": "USD",
                "exchange": "NASDAQ",
                "historical_prices": [
                    {"Datetime": "2023-01-01T00:00:00Z", "Close": 150.0}
                ],
            }
        ],
    }

    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        json.dump(sample_data, tf)
        tf.seek(0)
        stocks = FileManager.load(tf.name)

    if not isinstance(stocks, list):
        pytest.fail("Expected list of stocks")
    if stocks[0]["symbol"] != "AAPL":
        pytest.fail("Expected symbol AAPL")
    if not isinstance(stocks[0]["historical_prices"], list):
        pytest.fail("Expected historical prices list")
    if "Datetime" not in stocks[0]["historical_prices"][0]:
        pytest.fail("Datetime key missing")
    os.remove(tf.name)


def test_load_invalid_json():
    """Test FileManager.load raises JSONDecodeError on invalid JSON."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write("{invalid json")
        tf.seek(0)

    with pytest.raises(json.JSONDecodeError):
        FileManager.load(tf.name)

    os.remove(tf.name)


def test_load_missing_file():
    """Test FileManager.load returns empty list when file does not exist."""
    result = FileManager.load("nonexistent_file.json")
    if result != []:
        pytest.fail("Expected empty list for missing file")


def test_save_creates_file_and_content(tmp_path):
    """Test FileManager.save writes content to disk correctly."""
    file_path = tmp_path / "market_data.json"
    prices = [{"Datetime": pd.Timestamp("2024-01-01T00:00:00Z"), "Close": 100}]
    metadata = {
        "Name": "Test Corp",
        "Type": "Equity",
        "Sector": "Tech",
        "Currency": "USD",
        "Exchange": "NASDAQ",
    }

    FileManager.update_symbol("TEST", prices, metadata)
    FileManager.save(str(file_path))

    with open(file_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    if "stocks" not in saved:
        pytest.fail("Saved data should contain 'stocks'")
    if saved["stocks"][0]["symbol"] != "TEST":
        pytest.fail("Expected symbol TEST in saved data")


def test_load_with_timestamp_object():
    """Test FileManager.load with last_updated as timestamp string."""
    last_updated = pd.Timestamp("2023-12-31T00:00:00Z", tz="UTC")
    sample_data = {
        "last_updated": last_updated.isoformat(),
        "stocks": [
            {
                "symbol": "GOOG",
                "name": "Alphabet Inc.",
                "type": "Equity",
                "sector": "Tech",
                "currency": "USD",
                "exchange": "NASDAQ",
                "historical_prices": [
                    {"Datetime": "2023-01-01T00:00:00Z", "Close": 2500.0}
                ],
            }
        ],
    }

    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        json.dump(sample_data, tf)
        tf.seek(0)
        stocks = FileManager.load(tf.name)

    if stocks[0]["symbol"] != "GOOG":
        pytest.fail("Expected symbol GOOG")
    if not isinstance(FileManager.last_update(), pd.Timestamp):
        pytest.fail("Expected last_update to return pd.Timestamp")
    os.remove(tf.name)


def test_load_raises_generic_exception():
    """Test FileManager.load raises exception for unexpected error."""

    def mock_open(*args, **kwargs):
        raise OSError("Simulated file access error")

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open):
            with pytest.raises(OSError):
                FileManager.load("dummy_path.json")
