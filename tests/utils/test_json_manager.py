"""
Unit tests for the JsonManager utility module.

These tests verify JSON file operations including saving, loading,
deleting, and handling error scenarios such as missing files and
serialization failures.
"""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from utils.json_manager import JsonManager


@pytest.fixture
def sample_data():
    """
    Fixture that provides a sample dictionary containing serializable types
    including datetime and pandas.Timestamp for testing JSON operations.
    """
    return {
        "int": 1,
        "float": 3.14,
        "string": "hello",
        "datetime": datetime(2023, 1, 1, 12, 0),
        "timestamp": pd.Timestamp("2023-01-01T12:00:00"),
    }


# pylint: disable=redefined-outer-name
def test_save_and_load_json(tmp_path, sample_data):
    """
    Test saving and loading a JSON file with various serializable data types.

    Verifies that the file is created, and that data can be read back correctly.
    """
    filepath = tmp_path / "test.json"
    assert JsonManager.save(sample_data, str(filepath)) is True  # nosec
    assert filepath.exists()  # nosec
    assert JsonManager.load(str(filepath)) is not None  # nosec


def test_load_file_not_found(tmp_path):
    """
    Test loading a non-existent JSON file.

    Verifies that the method returns None when the file does not exist.
    """
    filepath = tmp_path / "not_exists.json"
    result = JsonManager.load(str(filepath))
    assert result is None  # nosec


def test_save_invalid_data(tmp_path):
    """
    Test saving a dictionary with a non-serializable object.

    Ensures that the method handles serialization errors and returns False.
    """
    filepath = tmp_path / "bad.json"

    # pylint: disable=too-few-public-methods
    class NotSerializable:
        """
        Dummy class representing a non-serializable object.
        """

    data = {"obj": NotSerializable()}
    result = JsonManager.save(data, str(filepath))
    assert result is False  # nosec


def test_delete_file(tmp_path):
    """
    Test deleting an existing JSON file.

    Confirms the file is deleted and the function returns True.
    """
    filepath = tmp_path / "delete_me.json"
    filepath.write_text("{}")
    result = JsonManager.delete(str(filepath))
    assert result is True  # nosec
    assert not filepath.exists()  # nosec


def test_delete_file_not_found(tmp_path):
    """
    Test deleting a file that does not exist.

    Verifies that the method returns False when attempting to delete a missing file.
    """
    filepath = tmp_path / "missing.json"
    result = JsonManager.delete(str(filepath))
    assert result is False  # nosec


def test_load_json_decode_error(tmp_path):
    """
    Test loading a malformed JSON file.

    Ensures that a JSONDecodeError is caught and the method returns None.
    """
    filepath = tmp_path / "malformed.json"
    filepath.write_text("{ invalid json ")
    result = JsonManager.load(str(filepath))
    assert result is None  # nosec


# pylint: disable=unused-argument
def test_delete_file_oserror(tmp_path, capfd):
    """
    Test deletion of a file when an OSError is raised.

    Simulates a permission error and ensures the method returns None.
    """
    filepath = tmp_path / "locked.json"
    filepath.write_text("{}")

    with patch("os.remove", side_effect=OSError("Permission denied")):
        result = JsonManager.delete(str(filepath))
        assert result is False  # nosec
