"""Unit tests for the JsonManager utility module.

These tests verify JSON file operations including saving, loading,
deleting, and handling error scenarios such as missing files and
serialization failures."""

from datetime import datetime
from unittest.mock import patch

import pandas as pd  # type: ignore
import pytest  # type: ignore

from src.utils.io.json_manager import JsonManager


@pytest.fixture
def sample_data():
    """Fixture that provides a sample dictionary containing serializable types.

    including datetime and pandas.Timestamp for testing JSON operations."""
    return {
        "int": 1,
        "float": 3.14,
        "string": "hello",
        "datetime": datetime(2023, 1, 1, 12, 0),
        "timestamp": pd.Timestamp("2023-01-01T12:00:00"),
    }


# pylint: disable=redefined-outer-name
def test_save_and_load_json(tmp_path, sample_data):
    """Test saving and loading a JSON file with various serializable data types.

    Verifies that the file is created, and that data can be read back correctly."""
    filepath = tmp_path / "test.json"
    result = JsonManager.save(sample_data, str(filepath))
    if result is not True:
        raise AssertionError("Expected save to return True")
    if not filepath.exists():
        raise AssertionError("Expected file to exist after saving")
    if JsonManager.load(str(filepath)) is None:
        raise AssertionError("Expected load to return non-None result")


def test_load_file_not_found(tmp_path):
    """Test loading a non-existent JSON file.

    Verifies that the method returns None when the file does not exist."""
    filepath = tmp_path / "not_exists.json"
    result = JsonManager.load(str(filepath))
    if result is not None:
        raise AssertionError("Expected None for non-existent file")


def test_save_invalid_data(tmp_path):
    """Test saving a dictionary with a non-serializable object.

    Ensures that the method handles serialization errors and returns False."""
    filepath = tmp_path / "bad.json"

    # pylint: disable=too-few-public-methods
    class NotSerializable:
        """Dummy class representing a non-serializable object."""

    data = {"obj": NotSerializable()}
    result = JsonManager.save(data, str(filepath))
    if result is not False:
        raise AssertionError("Expected save to return False for unserializable object")


def test_delete_file(tmp_path):
    """Test deleting an existing JSON file.

    Confirms the file is deleted and the function returns True."""
    filepath = tmp_path / "delete_me.json"
    filepath.write_text("{}")
    result = JsonManager.delete(str(filepath))
    if result is not True:
        raise AssertionError("Expected delete to return True")
    if filepath.exists():
        raise AssertionError("Expected file to be deleted")


def test_delete_file_not_found(tmp_path):
    """Test deleting a file that does not exist.

    Verifies that the method returns False when attempting to delete a missing file."""
    filepath = tmp_path / "missing.json"
    result = JsonManager.delete(str(filepath))
    if result is not False:
        raise AssertionError("Expected delete to return False for missing file")


def test_load_json_decode_error(tmp_path):
    """Test loading a malformed JSON file.

    Ensures that a JSONDecodeError is caught and the method returns None."""
    filepath = tmp_path / "malformed.json"
    filepath.write_text("{ invalid json ")
    result = JsonManager.load(str(filepath))
    if result is not None:
        raise AssertionError("Expected load to return None for malformed JSON")


# pylint: disable=unused-argument
def test_delete_file_oserror(tmp_path, capfd):
    """Test deletion of a file when an OSError is raised.

    Simulates a permission error and ensures the method returns None."""
    filepath = tmp_path / "locked.json"
    filepath.write_text("{}")
    with patch("os.remove", side_effect=OSError("Permission denied")):
        result = JsonManager.delete(str(filepath))
        if result is not False:
            raise AssertionError(
                "Expected delete to return False when OSError is raised"
            )


@pytest.mark.parametrize("invalid_path", [None, "", "   "])
def test_jsonmanager_load_with_empty_path(monkeypatch, caplog, invalid_path):
    """Test JsonManager.load returns None and logs an error when filepath is empty or None."""
    caplog.clear()
    result = JsonManager.load(invalid_path)
    if result is not None:
        raise AssertionError("Expected None when filepath is invalid.")
    if "filepath is empty" not in caplog.text:
        raise AssertionError("Expected error log for empty filepath.")


@pytest.mark.parametrize("invalid_path", [None, "", "   "])
def test_jsonmanager_save_with_empty_path(caplog, sample_data, invalid_path):
    """Test JsonManager.save returns False and logs an error when filepath is empty or None."""
    caplog.clear()
    result = JsonManager.save(sample_data, invalid_path)
    if result is not False:
        raise AssertionError("Expected False when filepath is invalid.")
    if "filepath is empty" not in caplog.text:
        raise AssertionError("Expected error log for empty filepath.")
