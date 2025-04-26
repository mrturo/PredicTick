"""
Unit tests for the Updater class in the market_data module.

These tests verify the correctness and robustness of the Updater class, which
coordinates updates, synchronization, and validation of historical market data.

Covered scenarios include:
- Valid and invalid JSON structure validation.
- Symbol update execution with mocked dependencies.
- Synchronization logic between local and Google Drive sources.
- Full update orchestration with retry logic and post-validation flow.
"""

# pylint: disable=protected-access

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from market_data.updater import Updater


def test_is_valid_marketdata_valid():
    """
    Test that a valid marketdata dictionary structure is accepted.

    Asserts True is returned when the structure includes a non-empty 'symbols'
    list containing both 'symbol' and 'historical_prices' keys.
    """
    valid_data = {
        "symbols": [
            {
                "symbol": "AAPL",
                "historical_prices": [{"datetime": "2024-01-01", "close": 150.0}],
            }
        ]
    }
    result = Updater.is_valid_marketdata(valid_data)
    if not result:
        raise AssertionError("Expected valid marketdata structure to return True")


def test_is_valid_marketdata_invalid():
    """
    Test that an invalid marketdata dictionary structure is rejected.

    Asserts False is returned when 'symbols' is missing or malformed.
    """
    invalid_data = {"not_symbols": []}
    result = Updater.is_valid_marketdata(invalid_data)
    if result:
        raise AssertionError("Expected invalid marketdata structure to return False")


@patch("market_data.updater.Updater._PARAMS")
@patch("market_data.updater.SymbolProcessor.process_symbols")
@patch("market_data.updater.Summarizer.print_sumary")
@patch("market_data.updater.Logger.separator")
def test_perform_single_update(_mock_sep, _mock_summary, mock_process, mock_params):
    """
    Test a single symbol update call returns expected update statistics.

    Verifies that SymbolProcessor and logger components interact as expected
    and return an update summary with correct counts.
    """
    mock_repo = MagicMock()
    mock_repo.get_invalid_symbols.return_value = []
    mock_params.symbol_repo = mock_repo
    mock_process.return_value = {"updated": 1, "invalid_symbols": []}

    result = Updater._perform_single_update(["AAPL"], 0.0)
    if result["updated"] != 1:
        raise AssertionError("Expected 1 updated symbol")


@patch("market_data.updater.Logger.warning")
def test_load_valid_marketdata_invalid(_mock_warn):
    """
    Test that malformed 'symbols' list causes marketdata to be ignored.

    Ensures that validation fails when 'symbols' is not a list and logs a warning.
    """
    invalid_data = {"symbols": "not_a_list"}
    result, ts = Updater._load_valid_marketdata(invalid_data)
    if result is not None or ts is not None:
        raise AssertionError(
            "Expected result and timestamp to be None for invalid data"
        )


@patch("market_data.updater.JsonManager.load")
@patch("market_data.updater.JsonManager.save")
@patch("market_data.updater.Logger.info")
@patch("market_data.updater.Logger.warning")
@patch("market_data.updater.GoogleDriveManager.download_file")
@patch("market_data.updater.JsonManager.delete")
def test_synchronize_marketdata_only_local(
    _mock_delete, mock_download, _mock_warn, mock_info, _mock_save, mock_load
):
    """
    Test synchronization logic when only a valid local JSON is present.

    Simulates a scenario where Google Drive does not provide a newer file,
    and the Updater retains the local version.
    """
    valid_data = {
        "last_updated": "2024-01-01T00:00:00Z",
        "symbols": [
            {
                "symbol": "AAPL",
                "historical_prices": [{"datetime": "2024-01-01", "close": 150.0}],
            }
        ],
    }
    mock_download.return_value = False
    mock_load.return_value = valid_data

    Updater._synchronize_marketdata_with_drive("mock_path.json")

    expected_msgs = [call.args[0] for call in mock_info.call_args_list]
    if not any("Using local JSON" in msg for msg in expected_msgs):
        raise AssertionError("Expected info log about using local JSON")


@patch("market_data.updater.Updater._perform_single_update")
@patch("market_data.updater.Gateway.get_latest_price_date")
@patch("market_data.updater.Gateway.get_stale_symbols")
@patch("market_data.updater.Validator.validate_market_data")
@patch("market_data.updater.Updater._GOOGLE_DRIVE.upload_file")
@patch("market_data.updater.Updater._synchronize_marketdata_with_drive")
def test_update_data_with_retry(
    _mock_sync, mock_upload, mock_validate, mock_stale, mock_latest, mock_update
):
    """
    Test full update flow execution with retry and validation logic.

    Ensures that if updates complete without changes, and validation passes,
    then the result is uploaded to Google Drive.
    """
    mock_update.return_value = {"updated": 0}
    mock_validate.return_value = True
    mock_stale.return_value = []
    mock_latest.return_value = pd.Timestamp("2024-01-01T00:00:00Z")

    Updater.update_data(in_max_retries=1, filepath="mock_path.json")

    if not mock_upload.called:
        raise AssertionError(
            "Expected upload_file to be called after successful validation"
        )


@pytest.mark.parametrize(
    "input_data, expected_ts_type, expected_data_none",
    [
        (None, type(None), True),
        ("invalid_format", type(None), True),
        ({"symbols": []}, type(None), True),
        (
            {"symbols": [{"symbol": "AAPL", "historical_prices": [1, 2]}]},
            pd.Timestamp,
            False,
        ),
        (
            {
                "last_updated": "2024-01-01T00:00:00Z",
                "symbols": [{"symbol": "AAPL", "historical_prices": [1]}],
            },
            pd.Timestamp,
            False,
        ),
    ],
)
def test_load_valid_marketdata_cases(
    input_data, expected_ts_type, expected_data_none, monkeypatch
):
    """
    Test coverage for the _load_valid_marketdata method in the Updater class.

    Verifies behavior for:
    - None input
    - Invalid format input (non-dict)
    - Valid format but empty symbols
    - Valid format with populated symbols
    - Valid input with explicit 'last_updated' field
    """
    monkeypatch.setattr(
        Updater,
        "is_valid_marketdata",
        lambda data: isinstance(data, dict)
        and "symbols" in data
        and isinstance(data["symbols"], list)
        and all("symbol" in s and "historical_prices" in s for s in data["symbols"]),
    )

    result_data, result_ts = Updater._load_valid_marketdata(input_data)

    if expected_data_none:
        if result_data is not None:
            raise AssertionError("Expected result_data to be None")
    else:
        if result_data is None:
            raise AssertionError("Expected result_data to not be None")

    if not isinstance(result_ts, expected_ts_type):
        raise AssertionError(
            f"Expected result_ts type {expected_ts_type}, got {type(result_ts)}"
        )


@pytest.fixture
def mock_dependencies():
    """
    Fixture to patch all dependencies required for the update_data method.

    This includes mocks for:
    - Synchronization with Google Drive.
    - Performing single symbol updates.
    - Retrieving latest price date and stale symbols from Gateway.
    - Validating updated market data.
    - Uploading updated file to Google Drive.

    Returns:
        Tuple[MagicMock, MagicMock]: Mocks for _perform_single_update and upload_file.
    """
    with patch("market_data.updater.Updater._synchronize_marketdata_with_drive"), patch(
        "market_data.updater.Updater._perform_single_update"
    ) as mock_update, patch(
        "market_data.updater.Gateway.get_latest_price_date",
        return_value=pd.Timestamp.now(),
    ), patch(
        "market_data.updater.Gateway.get_stale_symbols", return_value=[]
    ), patch(
        "market_data.updater.Validator.validate_market_data", return_value=True
    ), patch(
        "market_data.updater.Updater._GOOGLE_DRIVE.upload_file"
    ) as mock_upload:

        mock_update.return_value = {"updated": 1, "invalid_symbols": []}
        yield mock_update, mock_upload


# pylint: disable=redefined-outer-name
def test_update_data_with_default_filepath_and_retries(mock_dependencies):
    """
    Test update_data execution with default filepath and retry logic.

    Verifies that:
    - The update loop executes the expected number of retries.
    - The update function is called the correct number of times.
    - The upload to Google Drive is triggered once after successful validation.
    """
    mock_update, mock_upload = mock_dependencies

    Updater.update_data(in_max_retries=2, filepath=" ")

    if mock_update.call_count != 2:
        raise AssertionError(
            f"Expected 2 update attempts but got: {mock_update.call_count}"
        )

    if mock_upload.call_count != 1:
        raise AssertionError("Expected upload_file to be called once.")


@patch("market_data.updater.JsonManager.save")
@patch("market_data.updater.Logger.info")
def test_sync_drive_newer_than_local(mock_info, mock_save):
    """
    Test synchronization behavior when Google Drive file is newer than local.

    Simulates a case where the downloaded Drive JSON has a later timestamp than
    the local file, and validates that:
    - The Drive version is saved locally.
    - An informative log is generated about using the newer Drive file.
    """
    local = {
        "last_updated": "2024-01-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": [1]}],
    }
    drive = {
        "last_updated": "2024-02-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": [1]}],
    }

    with patch(
        "market_data.updater.JsonManager.load", side_effect=[local, drive]
    ), patch(
        "market_data.updater.GoogleDriveManager.download_file", return_value=True
    ), patch(
        "market_data.updater.JsonManager.delete", return_value=True
    ):
        Updater._synchronize_marketdata_with_drive("mock_path.json")

    mock_save.assert_called_with(drive, "mock_path.json")
    if not any(
        "Google Drive JSON (newer)" in str(c.args) for c in mock_info.call_args_list
    ):
        raise AssertionError("Expected log about newer Google Drive JSON")


@patch("market_data.updater.JsonManager.save")
@patch("market_data.updater.Logger.info")
def test_sync_local_newer_than_drive(mock_info, mock_save):
    """
    Test synchronization behavior when local JSON is newer than Drive version.

    Ensures that:
    - The local version is preserved without being overwritten.
    - The system logs a message indicating that local data is more recent.
    """
    local = {
        "last_updated": "2024-03-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": [1]}],
    }
    drive = {
        "last_updated": "2024-02-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": [1]}],
    }

    with patch(
        "market_data.updater.JsonManager.load", side_effect=[local, drive]
    ), patch(
        "market_data.updater.GoogleDriveManager.download_file", return_value=True
    ), patch(
        "market_data.updater.JsonManager.delete", return_value=True
    ):
        Updater._synchronize_marketdata_with_drive("mock_path.json")

    mock_save.assert_not_called()
    if not any("local JSON (newer)" in str(c.args) for c in mock_info.call_args_list):
        raise AssertionError("Expected log about newer local JSON")


@patch("market_data.updater.JsonManager.save")
@patch("market_data.updater.Logger.info")
def test_sync_equal_timestamps(mock_info, mock_save):
    """
    Test synchronization logic when local and Drive timestamps are equal.

    Verifies that:
    - No update occurs to either file.
    - The system logs that the timestamps are equal, indicating no preference.
    """
    ts = "2024-02-01T00:00:00Z"
    local = {
        "last_updated": ts,
        "symbols": [{"symbol": "AAPL", "historical_prices": [1]}],
    }
    drive = {
        "last_updated": ts,
        "symbols": [{"symbol": "AAPL", "historical_prices": [1]}],
    }

    with patch(
        "market_data.updater.JsonManager.load", side_effect=[local, drive]
    ), patch(
        "market_data.updater.GoogleDriveManager.download_file", return_value=True
    ), patch(
        "market_data.updater.JsonManager.delete", return_value=True
    ):
        Updater._synchronize_marketdata_with_drive("mock_path.json")

    mock_save.assert_not_called()
    if not any("Timestamps are equal" in str(c.args) for c in mock_info.call_args_list):
        raise AssertionError("Expected log about equal timestamps")


@patch("market_data.updater.JsonManager.save")
@patch("market_data.updater.Logger.info")
def test_sync_drive_only_available(mock_info, mock_save):
    """
    Test synchronization logic when only a valid Google Drive JSON is available.

    Ensures that:
    - The Drive file is saved as the local file.
    - The correct informational log is emitted indicating Drive JSON usage.
    """
    drive = {
        "last_updated": "2024-02-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": [1]}],
    }

    with patch(
        "market_data.updater.JsonManager.load", side_effect=[None, drive]
    ), patch(
        "market_data.updater.GoogleDriveManager.download_file", return_value=True
    ), patch(
        "market_data.updater.JsonManager.delete", return_value=True
    ):
        Updater._synchronize_marketdata_with_drive("mock_path.json")

    mock_save.assert_called_with(drive, "mock_path.json")
    if not any(
        "Drive JSON (only available)" in str(c.args) for c in mock_info.call_args_list
    ):
        raise AssertionError(
            "Expected log about drive JSON being the only valid option"
        )


@patch("market_data.updater.JsonManager.save")
@patch("market_data.updater.Logger.warning")
def test_sync_none_valid(mock_warn, mock_save):
    """
    Test behavior when neither local nor Drive JSON files are valid or available.

    Confirms that:
    - A fallback structure with empty symbols and a timestamp is saved locally.
    - A warning is logged to indicate that no valid JSON data could be found.
    """
    with patch(
        "market_data.updater.JsonManager.load", side_effect=Exception("fail")
    ), patch(
        "market_data.updater.GoogleDriveManager.download_file", return_value=False
    ):
        Updater._synchronize_marketdata_with_drive("mock_path.json")

    args, _ = mock_save.call_args
    if "symbols" not in args[0] or args[0]["symbols"] != []:
        raise AssertionError("Expected empty 'symbols' list in saved JSON")
    if "last_updated" not in args[0]:
        raise AssertionError("Expected 'last_updated' in saved JSON")
    if not any("No valid JSON" in str(c.args) for c in mock_warn.call_args_list):
        raise AssertionError("Expected warning about no valid JSON found")


@pytest.fixture
def dummy_filepath():
    """
    Fixture for test filepath used across synchronization tests.

    Returns:
        str: Dummy filename for JSON storage.
    """
    return "test_marketdata.json"


@patch("utils.json_manager.JsonManager.load")
@patch("utils.json_manager.JsonManager.save")
@patch("utils.google_drive_manager.GoogleDriveManager.download_file")
@patch("utils.json_manager.JsonManager.delete")
@patch("utils.logger.Logger.info")
@patch("utils.logger.Logger.warning")
@patch("utils.logger.Logger.debug")
def test_both_sources_invalid(
    _mock_debug,
    mock_warning,
    _mock_info,
    _mock_delete,
    mock_download,
    mock_save,
    mock_load,
    dummy_filepath,
):
    """
    Test when both local and Google Drive sources are invalid.

    Verifies that:
    - Logger.warning is triggered.
    - A new fallback JSON is saved locally.
    """
    mock_load.side_effect = Exception("fail")
    mock_download.return_value = False

    Updater._synchronize_marketdata_with_drive(dummy_filepath)

    if not mock_warning.called:
        raise AssertionError("Expected Logger.warning to be called")
    if not mock_save.called:
        raise AssertionError("Expected JsonManager.save to be called")


@patch("utils.json_manager.JsonManager.load")
@patch("utils.json_manager.JsonManager.save")
@patch("utils.google_drive_manager.GoogleDriveManager.download_file")
@patch("utils.json_manager.JsonManager.delete")
@patch("utils.logger.Logger.info")
def test_drive_newer_than_local(
    mock_info, _mock_delete, mock_download, mock_save, mock_load, dummy_filepath
):
    """
    Test that newer Drive JSON replaces older local JSON.

    Ensures:
    - Drive version is selected.
    - Logger.info reports 'Google Drive JSON'.
    """
    local = {
        "last_updated": "2024-01-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": []}],
    }
    drive = {
        "last_updated": "2025-01-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": []}],
    }

    def load_side_effect(path):
        return drive if path.endswith(".tmp") else local

    mock_load.side_effect = load_side_effect
    mock_download.return_value = True

    Updater._synchronize_marketdata_with_drive(dummy_filepath)

    if not mock_save.called:
        raise AssertionError("Expected JsonManager.save to be called")
    if not any(
        "Google Drive JSON" in str(call.args[0]) for call in mock_info.call_args_list
    ):
        raise AssertionError("Expected info log mentioning 'Google Drive JSON'")


@patch("utils.json_manager.JsonManager.load")
@patch("utils.json_manager.JsonManager.save")
@patch("utils.google_drive_manager.GoogleDriveManager.download_file")
@patch("utils.logger.Logger.info")
def test_local_newer_than_drive(
    mock_info, mock_download, mock_save, mock_load, dummy_filepath
):
    """
    Test that newer local JSON is kept over Drive JSON.

    Ensures:
    - No overwrite occurs.
    - Logger reports 'local JSON'.
    """
    local = {
        "last_updated": "2025-01-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": []}],
    }
    drive = {
        "last_updated": "2024-01-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": []}],
    }

    def load_side_effect(path):
        return drive if path.endswith(".tmp") else local

    mock_load.side_effect = load_side_effect
    mock_download.return_value = True

    Updater._synchronize_marketdata_with_drive(dummy_filepath)

    if mock_save.called:
        raise AssertionError("Expected JsonManager.save to NOT be called")
    if not any("local JSON" in str(call.args[0]) for call in mock_info.call_args_list):
        raise AssertionError("Expected info log mentioning 'local JSON'")


@patch("utils.json_manager.JsonManager.load")
@patch("utils.logger.Logger.info")
def test_equal_timestamps_uses_local(mock_info, mock_load, dummy_filepath):
    """
    Test when both timestamps are equal.

    Ensures that local JSON is retained and a log is emitted.
    """
    data = {
        "last_updated": "2025-01-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": []}],
    }

    mock_load.side_effect = lambda path: data
    with patch(
        "utils.google_drive_manager.GoogleDriveManager.download_file", return_value=True
    ):
        Updater._synchronize_marketdata_with_drive(dummy_filepath)

    if not any(
        "Timestamps are equal" in str(call.args[0]) for call in mock_info.call_args_list
    ):
        raise AssertionError("Expected info log mentioning 'Timestamps are equal'")


@patch("utils.json_manager.JsonManager.load")
@patch("utils.logger.Logger.info")
def test_drive_only_available(mock_info, mock_load, dummy_filepath):
    """
    Test when only Google Drive JSON is valid.

    Ensures it is saved locally and logs the usage.
    """
    drive = {
        "last_updated": "2025-01-01T00:00:00Z",
        "symbols": [{"symbol": "AAPL", "historical_prices": []}],
    }

    def load_side_effect(path):
        if path.endswith(".tmp"):
            return drive
        raise RuntimeError("no local file")

    mock_load.side_effect = load_side_effect

    with patch(
        "utils.google_drive_manager.GoogleDriveManager.download_file", return_value=True
    ), patch("utils.json_manager.JsonManager.save") as mock_save:
        Updater._synchronize_marketdata_with_drive(dummy_filepath)
        if not mock_save.called:
            raise AssertionError("Expected JsonManager.save to be called")
        if not any(
            "Google Drive JSON (only available)" in str(call.args[0])
            for call in mock_info.call_args_list
        ):
            raise AssertionError(
                "Expected log about 'Google Drive JSON (only available)'"
            )


def test_filepath_none_uses_default(monkeypatch):
    """
    Test that _synchronize_marketdata_with_drive uses default path when filepath is None.

    Ensures that the internal default filepath is used in case of a None input.
    """
    called = {}

    def fake_load(path):
        called["path"] = path
        raise RuntimeError("no file")  # Reemplazo de Exception

    monkeypatch.setattr("utils.json_manager.JsonManager.load", fake_load)
    monkeypatch.setattr(
        "utils.google_drive_manager.GoogleDriveManager.download_file",
        lambda *a, **kw: False,
    )
    monkeypatch.setattr("utils.json_manager.JsonManager.save", lambda *a, **kw: None)

    Updater._synchronize_marketdata_with_drive(None)
    if called["path"] != Updater._MARKETDATA_FILEPATH:
        raise AssertionError("Expected default filepath to be used")


def test_drive_load_exception(monkeypatch):
    """
    Test behavior when loading Google Drive JSON fails with an exception.

    Simulates an exception during temporary file loading from Drive and
    checks for fallback behavior.
    """

    def fake_load(path):
        if path.endswith(".tmp"):
            raise RuntimeError("Drive load fail")  # Reemplazo de Exception
        return {
            "last_updated": "2024-01-01T00:00:00Z",
            "symbols": [{"symbol": "AAPL", "historical_prices": []}],
        }

    monkeypatch.setattr("utils.json_manager.JsonManager.load", fake_load)
    monkeypatch.setattr(
        "utils.google_drive_manager.GoogleDriveManager.download_file",
        lambda *a, **kw: True,
    )
    monkeypatch.setattr("utils.json_manager.JsonManager.delete", lambda *a, **kw: True)
    monkeypatch.setattr("utils.logger.Logger.info", lambda *a, **kw: None)
    monkeypatch.setattr("utils.logger.Logger.warning", lambda *a, **kw: None)
    monkeypatch.setattr("utils.json_manager.JsonManager.save", lambda *a, **kw: None)

    Updater._synchronize_marketdata_with_drive("dummy.json")
