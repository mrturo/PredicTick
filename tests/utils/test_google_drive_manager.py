"""Unit tests for the GoogleDriveManager utility module."""

# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


import os
from datetime import datetime
from unittest.mock import MagicMock, mock_open, patch

import pytest

from utils.google_drive_manager import GoogleDriveManager
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_TEST_MARKETDATA_FILEPATH = _PARAMS.get("test_marketdata_filepath")


@pytest.fixture
def fixture_mock_params():
    """Mock the ParameterLoader to provide test configuration values."""
    with patch("src.utils.google_drive_manager.ParameterLoader") as mock_loader:
        instance = mock_loader.return_value
        instance.get.side_effect = lambda key: {
            "gcp_credentials_filepath": "credentials.json",
            "gcp_token_filepath": "token.json",
            "gdrive_folder_id": "folder123",
        }[key]
        yield instance


@pytest.fixture
def fixture_mock_auth_service():
    """Mock authentication and Google Drive service."""
    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", mock_open(read_data="{}")
    ), patch(
        "src.utils.google_drive_manager.Credentials.from_authorized_user_file"
    ) as mock_creds, patch(
        "src.utils.google_drive_manager.build"
    ) as mock_build:

        mock_creds.return_value.valid = True
        mock_service = MagicMock()
        mock_service.files.return_value.list.return_value.execute.return_value = {
            "files": []
        }
        mock_service.files.return_value.create.return_value.execute.return_value = {
            "id": "file123"
        }
        mock_service.files.return_value.update.return_value.execute.return_value = {
            "id": "file123"
        }
        mock_build.return_value = mock_service

        yield mock_service


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_upload_new_file(fixture_mock_auth_service):
    """Test uploading a new file to Google Drive."""
    with patch("mimetypes.guess_type", return_value=("application/json", None)), patch(
        "src.utils.google_drive_manager.MediaFileUpload"
    ):
        manager = GoogleDriveManager(
            service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
        )
        manager.upload_file(_TEST_MARKETDATA_FILEPATH)

    fixture_mock_auth_service.files.return_value.create.assert_called()


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_update_existing_file(fixture_mock_auth_service):
    """Test updating an existing file in Google Drive."""
    fixture_mock_auth_service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "existing_id"}]
    }

    with patch("mimetypes.guess_type", return_value=("application/json", None)), patch(
        "src.utils.google_drive_manager.MediaFileUpload"
    ):
        manager = GoogleDriveManager(
            service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
        )
        manager.upload_file(_TEST_MARKETDATA_FILEPATH)

    fixture_mock_auth_service.files.return_value.update.assert_called()


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_upload_with_backup(fixture_mock_auth_service):
    """Test uploading a file with backup creation to Google Drive."""
    fixture_mock_auth_service.files.return_value.list.return_value.execute.return_value = {
        "files": []
    }
    fixture_mock_auth_service.files.return_value.create.return_value.execute.side_effect = [
        {"id": "main_file"},
        {"id": "backup_file"},
    ]

    with patch("mimetypes.guess_type", return_value=("application/json", None)), patch(
        "src.utils.google_drive_manager.MediaFileUpload"
    ), patch("src.utils.google_drive_manager.datetime.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 5, 18, 17, 30, 45)

        manager = GoogleDriveManager(
            service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
        )
        manager.upload_file(_TEST_MARKETDATA_FILEPATH, backup=True)

    create_calls = fixture_mock_auth_service.files.return_value.create.call_args_list

    expected_name = os.path.basename(_TEST_MARKETDATA_FILEPATH)

    if not any(call.kwargs["body"]["name"] == expected_name for call in create_calls):
        pytest.fail(f"Expected '{expected_name}' upload not found")

    if not any(
        call.kwargs["body"]["name"] == "market_data_20250518173045.json"
        for call in create_calls
    ):
        pytest.fail("Expected backup file upload not found")


def test_authenticate_refresh_token_flow():
    """Test the authentication flow using refresh token."""
    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", mock_open(read_data="{}")
    ), patch(
        "src.utils.google_drive_manager.Credentials.from_authorized_user_file"
    ) as mock_creds, patch(
        "src.utils.google_drive_manager.Request"
    ), patch(
        "src.utils.google_drive_manager.build"
    ) as mock_build:

        mock_cred_instance = MagicMock()
        mock_cred_instance.valid = False
        mock_cred_instance.expired = True
        mock_cred_instance.refresh_token = "token"  # nosec
        mock_creds.return_value = mock_cred_instance
        mock_service = MagicMock()
        mock_build.return_value = mock_service

        GoogleDriveManager(
            service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
        )
        mock_cred_instance.refresh.assert_called()


def test_authenticate_installed_app_flow():
    """Test the authentication flow using OAuth InstalledAppFlow."""
    with patch("os.path.exists", return_value=False), patch(
        "src.utils.google_drive_manager.InstalledAppFlow.from_client_secrets_file"
    ) as mock_flow, patch("src.utils.google_drive_manager.build") as mock_build, patch(
        "builtins.open", mock_open()
    ) as mock_file:

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.to_json.return_value = '{"mock": "token"}'
        mock_flow.return_value.run_local_server.return_value = mock_creds
        mock_service = MagicMock()
        mock_build.return_value = mock_service

        GoogleDriveManager(
            service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
        )

        mock_flow.assert_called()
        mock_file().write.assert_called_with('{"mock": "token"}')


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_file_exists_true(fixture_mock_auth_service):
    """Test file_exists returns True when the file is found in Google Drive."""
    fixture_mock_auth_service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "file123"}]
    }

    manager = GoogleDriveManager(
        service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
    )
    result = manager.file_exists(_TEST_MARKETDATA_FILEPATH)
    if not result:
        pytest.fail("Expected file to exist in Google Drive.")


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_file_exists_false(fixture_mock_auth_service):
    """Test file_exists returns False when the file is not found in Google Drive."""
    fixture_mock_auth_service.files.return_value.list.return_value.execute.return_value = {
        "files": []
    }

    manager = GoogleDriveManager(
        service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
    )
    result = manager.file_exists(_TEST_MARKETDATA_FILEPATH)
    if result:
        pytest.fail("Expected file not to exist in Google Drive.")


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_download_file_success(fixture_mock_auth_service):
    """Test download_file correctly writes file when found."""
    fixture_mock_auth_service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "file123"}]
    }

    mock_request = MagicMock()
    mock_request.uri = "mock://file"
    mock_request.http = MagicMock()
    mock_request.http.request.return_value = (MagicMock(status=200), b"content")

    with patch(
        "utils.google_drive_manager.io.FileIO", return_value=MagicMock()
    ) as _, patch("utils.google_drive_manager.MediaIoBaseDownload") as mock_downloader:

        mock_downloader_instance = MagicMock()
        mock_downloader_instance.next_chunk.side_effect = [(None, True)]
        mock_downloader.return_value = mock_downloader_instance

        fixture_mock_auth_service.files.return_value.get_media.return_value = (
            mock_request
        )

        manager = GoogleDriveManager(
            service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
        )
        manager.download_file(_TEST_MARKETDATA_FILEPATH)

        mock_downloader.assert_called_once()
        args, _ = mock_downloader.call_args
        if args[1] != mock_request:
            pytest.fail("MediaIoBaseDownload did not receive expected request object.")
        mock_downloader_instance.next_chunk.assert_called()


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_download_file_not_found(fixture_mock_auth_service):
    """Test download_file handles missing file in Google Drive gracefully."""
    fixture_mock_auth_service.files.return_value.list.return_value.execute.return_value = {
        "files": []
    }

    manager = GoogleDriveManager(
        service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
    )
    with patch("src.utils.google_drive_manager.Logger.error") as mock_log_error:
        manager.download_file("nonexistent_file.json")
        mock_log_error.assert_called_once_with(
            "File 'nonexistent_file.json' not found in Google Drive."
        )


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_download_file_raises_exception(fixture_mock_auth_service):
    """Test download_file returns False when an exception is raised during download."""

    # Forzar excepción al intentar obtener el media file
    fixture_mock_auth_service.files.return_value.get_media.side_effect = RuntimeError(
        "Mocked download error"
    )

    manager = GoogleDriveManager(
        service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
    )
    result = manager.download_file(_TEST_MARKETDATA_FILEPATH)

    if result is not False:
        pytest.fail("Expected download_file to return False.")


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_download_file_exception_handling(fixture_mock_auth_service):
    """Test that download_file handles exceptions gracefully and returns False."""
    # Configurar el mock para que get_media lance una excepción
    fixture_mock_auth_service.files.return_value.get_media.side_effect = Exception(
        "Simulated download error"
    )

    manager = GoogleDriveManager(
        service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
    )
    result = manager.download_file(_TEST_MARKETDATA_FILEPATH)

    if result is not False:
        pytest.fail("Expected download_file to return False when an exception occurs.")


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_download_file_exception_during_file_write(fixture_mock_auth_service):
    """Test download_file returns False if an exception is raised during file write."""
    fixture_mock_auth_service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "file123"}]
    }

    manager = GoogleDriveManager(
        service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
    )

    with patch(
        "utils.google_drive_manager.io.FileIO",
        side_effect=IOError("Simulated write error"),
    ):
        result = manager.download_file(_TEST_MARKETDATA_FILEPATH)

    if result is not False:
        pytest.fail("Expected download_file to return False.")


def test_file_exists_service_none():
    """Test file_exists returns False when service is None and logs warning."""
    with patch.object(GoogleDriveManager, "_authenticate", return_value=None):
        manager = GoogleDriveManager(service=None, drive_folder_id="drive_folder_fafe")
        with patch("src.utils.google_drive_manager.Logger.warning") as mock_log:
            result = manager.file_exists("dummy.json")
            mock_log.assert_called_once()
            if (
                "File existence check in Google Drive skipped"
                not in mock_log.call_args[0][0]
            ):
                pytest.fail("Expected warning about unavailable Google Drive service.")
            if result is not False:
                pytest.fail(
                    "Expected file_exists to return False when service is None."
                )


def test_download_file_service_none():
    """Test download_file returns False when service is None and logs warning if file is missing."""
    with patch.object(GoogleDriveManager, "_authenticate", return_value=None):
        manager = GoogleDriveManager(service=None, drive_folder_id="drive_folder_fafe")
        with patch("src.utils.google_drive_manager.Logger.warning") as mock_log, patch(
            "os.path.exists", return_value=True
        ):
            result = manager.download_file("dummy.json")
            mock_log.assert_called_once()
            if "Download from Google Drive skipped" not in mock_log.call_args[0][0]:
                pytest.fail("Expected warning about unavailable Google Drive service.")
            if result is not False:
                pytest.fail(
                    "Expected download_file to return False when service is None."
                )


def test_upload_file_raises_exception_logs_error():
    """Test that upload_file logs an error when an exception occurs."""
    with patch(
        "src.utils.google_drive_manager.MediaFileUpload",
        side_effect=TypeError("Simulated error"),
    ), patch("src.utils.google_drive_manager.Logger.error") as mock_log, patch(
        "mimetypes.guess_type", return_value=("application/json", None)
    ):
        manager = GoogleDriveManager(
            service=MagicMock(), drive_folder_id="drive_folder_fafe"
        )
        manager.upload_file("dummy.json")

        mock_log.assert_called_once()
        if "Error uploading file" not in mock_log.call_args[0][0]:
            pytest.fail("Expected error log about upload_file failure.")


def test_authenticate_invalid_token_file_triggers_refresh():
    """Test that invalid token triggers re-authentication and token is written."""
    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", mock_open(read_data="{}")
    ) as mock_file_open, patch(
        "src.utils.google_drive_manager.Credentials.from_authorized_user_file"
    ) as mock_creds_loader, patch(
        "src.utils.google_drive_manager.InstalledAppFlow.from_client_secrets_file"
    ) as mock_flow:

        mock_invalid_creds = MagicMock()
        mock_invalid_creds.valid = False
        mock_invalid_creds.expired = False
        mock_invalid_creds.refresh_token = None
        mock_creds_loader.return_value = mock_invalid_creds

        mock_valid_creds = MagicMock()
        mock_valid_creds.valid = True
        mock_valid_creds.to_json.return_value = '{"mock": "token"}'
        mock_flow.return_value.run_local_server.return_value = mock_valid_creds

        GoogleDriveManager(service=MagicMock(), drive_folder_id="drive_folder_fafe")

        mock_flow.return_value.run_local_server.assert_called_once()
        mock_file_open().write.assert_called_once_with('{"mock": "token"}')


def test_upload_file_service_none():
    """Test upload_file returns False when service is None and logs warning if file is missing."""
    with patch.object(GoogleDriveManager, "_authenticate", return_value=None):
        manager = GoogleDriveManager(service=None, drive_folder_id="drive_folder_fafe")
        with patch("src.utils.google_drive_manager.Logger.warning") as mock_log, patch(
            "os.path.exists", return_value=True
        ):
            manager.upload_file("dummy.json")
            mock_log.assert_called_once()
            if "Upload to Google Drive skipped" not in mock_log.call_args[0][0]:
                pytest.fail("Expected warning about unavailable Google Drive service.")


def test_upload_file_service_none_logs_warning():
    """Test upload_file logs warning when service is None."""
    with patch.object(GoogleDriveManager, "_authenticate", return_value=None):
        manager = GoogleDriveManager(service=None, drive_folder_id="drive_folder_fafe")
        with patch("src.utils.google_drive_manager.Logger.warning") as mock_log:
            manager.upload_file("dummy.json")
            mock_log.assert_called_once()
            if "Upload to Google Drive skipped" not in mock_log.call_args[0][0]:
                pytest.fail("Expected warning about unavailable service.")


@pytest.mark.usefixtures("fixture_mock_auth_service")
def test_download_file_generic_exception(fixture_mock_auth_service):
    """Test download_file returns False on unexpected exceptions."""
    fixture_mock_auth_service.files.return_value.get_media.side_effect = Exception(
        "Unexpected"
    )

    manager = GoogleDriveManager(
        service=fixture_mock_auth_service, drive_folder_id="drive_folder_fafe"
    )
    result = manager.download_file(_TEST_MARKETDATA_FILEPATH)

    if result is not False:
        pytest.fail("Expected download_file to return False on exception.")


def test_file_exists_service_none_logs_warning():
    """Test file_exists logs a warning when service is None and returns False."""
    with patch.object(GoogleDriveManager, "_authenticate", return_value=None):
        manager = GoogleDriveManager(service=None, drive_folder_id="drive_folder_fafe")
        with patch("src.utils.google_drive_manager.Logger.warning") as mock_log:
            result = manager.file_exists("dummy.json")
            mock_log.assert_called_once()
            if (
                "File existence check in Google Drive skipped"
                not in mock_log.call_args[0][0]
            ):
                pytest.fail(
                    "Expected warning about file_exists with unavailable service."
                )
            if result is not False:
                pytest.fail(
                    "Expected file_exists to return False when service is None."
                )


def test_download_file_service_none_logs_warning():
    """Test download_file logs a warning when service is None and returns False."""
    with patch.object(GoogleDriveManager, "_authenticate", return_value=None):
        manager = GoogleDriveManager(service=None, drive_folder_id="drive_folder_fafe")
        with patch("src.utils.google_drive_manager.Logger.warning") as mock_log:
            result = manager.download_file("dummy.json", download_path="dummy.json")
            mock_log.assert_called_once()
            if "Download from Google Drive skipped" not in mock_log.call_args[0][0]:
                pytest.fail(
                    "Expected warning about download_file with unavailable service."
                )
            if result is not False:
                pytest.fail(
                    "Expected download_file to return False when service is None."
                )


def test_authenticate_raises_exception_logs_warning():
    """Test that _authenticate logs a warning and returns None if an exception occurs."""
    with patch(
        "src.utils.google_drive_manager.os.path.exists",
        side_effect=RuntimeError("Mocked failure"),
    ), patch("src.utils.google_drive_manager.Logger.warning") as mock_log:

        manager = GoogleDriveManager(service=None, drive_folder_id="drive_folder_fafe")
        if manager._creds is not None:
            raise AssertionError(
                "Expected credentials to be None after exception in _authenticate."
            )

        mock_log.assert_called_once()
        if "Google Drive authentication failed" not in mock_log.call_args[0][0]:
            raise AssertionError("Expected warning about authentication failure.")


@pytest.mark.parametrize(
    "local_path,drive_path,expected",
    [
        (
            "/tests/data/test_file.json",
            None,
            ("/tests/data/test_file.json", "test_file.json"),
        ),
        ("./data/sample.csv", "uploaded.csv", ("./data/sample.csv", "uploaded.csv")),
        (
            "folder\\document.txt",
            "new_doc.txt",
            ("folder\\document.txt", "new_doc.txt"),
        ),
        (
            "/tests/data/test_file.json",
            "",
            ("/tests/data/test_file.json", "test_file.json"),
        ),
        (
            "/tests/data/test_file.json",
            "   ",
            ("/tests/data/test_file.json", "test_file.json"),
        ),
    ],
)
def test_validate_filename_valid_inputs(local_path, drive_path, expected):
    """Test _validate_filename returns the correct tuple for valid inputs."""
    result = GoogleDriveManager._validate_filename(local_path, drive_path)
    if result != expected:
        raise AssertionError(f"Expected {expected} but got {result}")


@pytest.mark.parametrize(
    "local_path",
    [
        None,
        "",
        "   ",
        "invalid/",
        "invalid\\",
    ],
)
def test_validate_filename_invalid_local_path_structure(local_path):
    """Test raises ValueError when local_path is empty, invalid, or ends with a separator."""
    with pytest.raises(ValueError) as exc:
        GoogleDriveManager._validate_filename(local_path)
    if "Local filepath" not in str(exc.value):
        raise AssertionError("Expected ValueError about local filepath")


@pytest.mark.parametrize(
    "drive_path",
    [
        "folder/.file",
        "namewithoutextension",
    ],
)
def test_validate_filename_invalid_drive_path_structure(drive_path):
    """Test raises ValueError when drive_path is malformed or lacks a valid file extension."""
    local_path = "/tests/data/valid_name.txt"
    with pytest.raises(ValueError) as exc:
        GoogleDriveManager._validate_filename(local_path, drive_path)
    if "Drive filepath" not in str(exc.value):
        raise AssertionError("Expected ValueError about drive filepath")


def test_validate_filename_hidden_drive_file_rejected():
    """Test _validate_filename raises ValueError if drive_path is a hidden file."""
    with pytest.raises(ValueError) as exc:
        GoogleDriveManager._validate_filename(
            "/tests/data/valid_file.txt", ".hiddenfile"
        )
    if "Drive filepath" not in str(exc.value):
        raise AssertionError("Expected error about hidden drive filepath")


def test_validate_filename_path_object_raises_exception():
    """Test _validate_filename handles exception raised by Path constructor."""
    with patch(
        "utils.google_drive_manager.Path", side_effect=RuntimeError("Simulated failure")
    ):
        with pytest.raises(ValueError) as exc:
            GoogleDriveManager._validate_filename("/some/valid/file.txt")
        if "Local filepath is invalid" not in str(exc.value):
            raise AssertionError(
                "Expected ValueError about invalid local filepath due to Path error"
            )


def test_validate_filename_missing_extension_raises():
    """Test that a local file without extension and with valid path raises expected ValueError."""
    with pytest.raises(ValueError) as exc:
        GoogleDriveManager._validate_filename("/some/valid/valid_name_without_suffix")
    if "Local filepath does not appear to be a valid file path" not in str(exc.value):
        raise AssertionError(
            f"Expected ValueError about missing extension in local filepath but got: {exc.value}"
        )
