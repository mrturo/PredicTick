"""Module for handling Google Drive file uploads and downloads."""

import datetime
import io
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from utils.logger import Logger
from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods,no-member
class GoogleDriveManager:
    """Handles authentication and file upload and download to Google Drive."""

    _SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    _PARAMS = ParameterLoader()
    _GCP_CREDENTIALS_FILEPATH = _PARAMS.get("gcp_credentials_filepath")
    _GCP_TOKEN_FILEPATH = _PARAMS.get("gcp_token_filepath")
    _DRIVE_FOLDER_ID = _PARAMS.get("gdrive_folder_id")

    def __init__(
        self, service: Optional[Resource] = None, drive_folder_id: str = None
    ) -> None:
        self._creds = self._authenticate()
        self._service: Optional[Resource] = service or (
            build("drive", "v3", credentials=self._creds) if self._creds else None
        )

        self._drive_folder_id = (
            GoogleDriveManager._DRIVE_FOLDER_ID
            if (not drive_folder_id or len(drive_folder_id.strip()) == 0)
            else drive_folder_id.strip()
        )

    def _authenticate(self) -> Credentials:
        """Authenticate using OAuth2 credentials and return a valid Credentials object."""
        creds = None
        try:
            if os.path.exists(GoogleDriveManager._GCP_TOKEN_FILEPATH):
                creds = Credentials.from_authorized_user_file(
                    GoogleDriveManager._GCP_TOKEN_FILEPATH, GoogleDriveManager._SCOPES
                )

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        GoogleDriveManager._GCP_CREDENTIALS_FILEPATH,
                        GoogleDriveManager._SCOPES,
                    )
                    creds = flow.run_local_server(port=0)
                with open(
                    GoogleDriveManager._GCP_TOKEN_FILEPATH, "w", encoding="utf-8"
                ) as token:
                    token.write(creds.to_json())
        except Exception as ex:  # pylint: disable=broad-exception-caught
            Logger.warning(f"Google Drive authentication failed: {str(ex)}")
            creds = None

        return creds

    @staticmethod
    def _validate_filename(local_filepath: str, drive_filename: str = None) -> Any:
        if local_filepath is None or len(local_filepath.strip()) == 0:
            raise ValueError("Local filepath is empty")
        local_filepath = local_filepath.strip()
        if local_filepath.endswith(("/", "\\")):
            raise ValueError(
                f"Local filepath ends with a directory separator: {local_filepath}"
            )

        try:
            path_obj = Path(local_filepath)
            base_name = path_obj.name
        except Exception as e:
            raise ValueError(f"Local filepath is invalid: {local_filepath}") from e

        if not base_name or not path_obj.suffix:
            raise ValueError(
                f"Local filepath does not appear to be a valid file path: {local_filepath}"
            )

        if drive_filename is None or len(drive_filename.strip()) == 0:
            drive_filename = base_name
        else:
            drive_filename = drive_filename.strip()
            path_obj = Path(drive_filename)
            if not path_obj.name or not path_obj.suffix:
                raise ValueError(
                    f"Drive filepath does not appear to be a valid file path: {drive_filename}"
                )

        return local_filepath, drive_filename

    def upload_file(self, filepath: str, backup: bool = False) -> None:
        """Upload a file to Google Drive. Optionally create a timestamped backup."""
        try:

            if self._service is None:
                Logger.warning(
                    f"Upload to Google Drive skipped '{filepath}': "
                    f"Google Drive service is not available."
                )
                return
            filepath, drive_filename = GoogleDriveManager._validate_filename(filepath)
            mime_type, _ = mimetypes.guess_type(filepath)

            file_id = self._get_existing_file_id(drive_filename)
            if file_id:
                media = MediaFileUpload(filepath, mimetype=mime_type)
                updated_file = (
                    self._service.files()
                    .update(fileId=file_id, media_body=media)
                    .execute()
                )
                Logger.debug(
                    f"File overwritten '{drive_filename}' successfully to Google Drive"
                    f" with ID: {updated_file.get('id')}"
                )
            else:
                file_metadata = {"name": drive_filename}
                if self._drive_folder_id:
                    file_metadata["parents"] = [self._drive_folder_id]
                media = MediaFileUpload(filepath, mimetype=mime_type)
                created_file = (
                    self._service.files()
                    .create(body=file_metadata, media_body=media, fields="id")
                    .execute()
                )
                Logger.debug(
                    f"File uploaded '{drive_filename}' successfully to Google Drive"
                    f" with ID: {created_file.get('id')}"
                )

            if backup:
                self._create_backup(filepath, drive_filename, mime_type)
        except Exception as ex:  # pylint: disable=broad-exception-caught
            Logger.error(f"Error uploading file '{filepath}': {str(ex)}")

    def _get_existing_file_id(self, filename: str) -> Optional[str]:
        """Retrieve the ID of a file with a given name in the target Google Drive folder."""
        query = f"name = '{filename}' and trashed = false"
        if self._drive_folder_id:
            query += f" and '{self._drive_folder_id}' in parents"

        results = (
            self._service.files().list(q=query, fields="files(id, name)").execute()
        )
        files = results.get("files", [])
        return files[0]["id"] if files else None

    def _create_backup(
        self, filepath: str, drive_filename: str, mime_type: str
    ) -> None:
        """Create a timestamped backup of the uploaded file in Google Drive."""
        timestamp = datetime.datetime.now().strftime("_%Y%m%d%H%M%S")
        base_name, ext = os.path.splitext(drive_filename)
        backup_name = f"{base_name}{timestamp}{ext}"

        backup_metadata = {"name": backup_name}
        if self._drive_folder_id:
            backup_metadata["parents"] = [self._drive_folder_id]
        backup_media = MediaFileUpload(filepath, mimetype=mime_type)

        backup_file = (
            self._service.files()
            .create(body=backup_metadata, media_body=backup_media, fields="id")
            .execute()
        )
        Logger.debug(
            f"Backup file created: {backup_name} with ID: {backup_file.get('id')}"
        )

    def file_exists(self, filepath: str) -> bool:
        """Check if a file with the given name exists in the target Google Drive folder."""
        if self._service is None:
            Logger.warning(
                "File existence check in Google Drive skipped: "
                "Google Drive service is not available."
            )
            return False

        filename = os.path.basename(filepath)
        base_name, ext = os.path.splitext(filename)
        query = f"name = '{base_name}{ext}' and trashed = false"
        if self._drive_folder_id:
            query += f" and '{self._drive_folder_id}' in parents"

        results = (
            self._service.files()
            .list(q=query, fields="files(id)", pageSize=1)
            .execute()
        )
        files = results.get("files", [])
        return bool(files)

    def download_file(self, filepath: str, download_path: str = None) -> bool:
        """Download a file from Google Drive by its name to the specified destination path."""
        try:
            if download_path is None or len(download_path.strip()) == 0:
                download_path = filepath

            filename = os.path.basename(filepath)
            if self._service is None:
                Logger.warning(
                    f"Download from Google Drive skipped '{filename}': "
                    f"Google Drive service is not available."
                )
                return False

            file_id = self._get_existing_file_id(filename)
            if not file_id:
                Logger.error(f"File '{filename}' not found in Google Drive.")
                return False

            os.makedirs(os.path.dirname(download_path) or ".", exist_ok=True)
            request = self._service.files().get_media(fileId=file_id)
            fh = io.FileIO(download_path, mode="wb")
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            Logger.debug(
                f"File downloaded successfully from Google Drive to '{download_path}'"
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return False
        return True
