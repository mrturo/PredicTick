"""
Marketdata synchronization utilities for local storage and Google Drive.

This module provides functionality to ensure consistency between a local
marketdata JSON file and its corresponding version stored on Google Drive.
It includes validation of JSON structures, conflict resolution based on
timestamps, and persistence of the most up-to-date data.

Main features:
    - Validate local marketdata JSON schema.
    - Load and compare versions from local filesystem and Google Drive.
    - Resolve conflicts by preserving the most recent data.
    - Ensure a consistent JSON file is always available locally.
"""

from typing import Any, Optional

import pandas as pd

from src.market_data.utils.storage.google_drive_manager import \
    GoogleDriveManager
from src.market_data.utils.storage.market_data_version import MarketDataVersion
from src.utils.io.json_manager import JsonManager
from src.utils.io.logger import Logger


class MarketDataSyncManager:
    """
    Manager for synchronizing marketdata JSON files between local storage and Google Drive.

    This class provides static methods to validate, load, and resolve conflicts
    between different versions of marketdata. The synchronization process ensures
    that the local file always contains the most recent and valid dataset.

    Responsibilities:
        - Validate the structure of marketdata JSON files.
        - Load and sanitize local and Google Drive marketdata.
        - Handle temporary files and cleanup after Drive downloads.
        - Resolve version conflicts using timestamps and file identifiers.
        - Guarantee local persistence of the latest valid marketdata.
    """

    _GOOGLE_DRIVE = GoogleDriveManager()

    @staticmethod
    def is_valid_marketdata(data: dict) -> bool:
        """Validate the structure of a marketdata dictionary.

        This method checks whether the input data conforms to the expected schema
        used for storing marketdata, including presence of a symbol list with
        required keys in each entry.
        """
        return (
            isinstance(data, dict)
            and "symbols" in data
            and isinstance(data["symbols"], list)
            and all("symbol" in s and "historical_prices" in s for s in data["symbols"])
        )

    @staticmethod
    def _load_valid_marketdata(data: Any) -> Any:
        """Load marketdata JSON and return data and timestamp only if symbols are non-empty."""
        file_id: Optional[str] = None
        symbols: Optional[list[Any]] = None
        ts: Optional[pd.Timestamp] = None
        if data is not None:
            if not MarketDataSyncManager.is_valid_marketdata(data):
                Logger.warning("Local marketdata is invalid. Ignoring file.")
                data = None
                file_id = None
                ts = None
            else:
                file_id = data.get("id", None)
                symbols = data.get("symbols", []) if isinstance(data, dict) else []
                ts = (
                    pd.to_datetime(data.get("last_updated"), utc=True)
                    if data and "last_updated" in data
                    else pd.Timestamp.min.tz_localize("UTC")
                )
                if len(symbols) == 0:
                    data = None
                    file_id = None
                    ts = None
        return data, ts, file_id

    @staticmethod
    def _load_local_marketdata(
        filepath: str,
    ) -> tuple[Any, Optional[pd.Timestamp], Optional[str]]:
        try:
            data = JsonManager.load(filepath)
            return MarketDataSyncManager._load_valid_marketdata(data)
        except Exception:  # pylint: disable=broad-exception-caught
            return None, None, None

    @staticmethod
    def _load_drive_marketdata(
        filepath: str,
    ) -> tuple[Any, Optional[pd.Timestamp], Optional[str]]:
        try:
            temp_path = f"{filepath}.tmp"
            if MarketDataSyncManager._GOOGLE_DRIVE.download_file(filepath, temp_path):
                data = JsonManager.load(temp_path)
                result = MarketDataSyncManager._load_valid_marketdata(data)
                Logger.debug(
                    "Marketdata JSON successfully retrieved from Google Drive file."
                )
                if JsonManager.delete(temp_path):
                    Logger.debug(f"Temporary file '{temp_path}' deleted successfully.")
                return result
        except Exception as e:  # pylint: disable=broad-exception-caught
            Logger.warning(f"Failed to load marketdata from Drive: {e}")
        return None, None, None

    @staticmethod
    def _resolve_data_conflict(
        local_version: MarketDataVersion,
        drive_version: MarketDataVersion,
        filepath: str,
    ) -> None:
        local_data, local_ts, local_id = (
            local_version.data,
            local_version.timestamp,
            local_version.file_id,
        )
        drive_data, drive_ts, drive_id = (
            drive_version.data,
            drive_version.timestamp,
            drive_version.file_id,
        )
        if (
            local_data is not None
            and local_ts is not None
            and drive_data is not None
            and drive_ts is not None
        ):
            if local_id and drive_id and local_id == drive_id:
                Logger.info("Using local data JSON (It's same to Google Drive JSON).")
            elif drive_ts > local_ts:
                Logger.info("Using Google Drive data JSON (newer).")
                JsonManager.save(drive_data, filepath)
            elif drive_ts < local_ts:
                Logger.info("Using local JSON (newer).")
            else:
                Logger.info("Timestamps are equal. Retaining local JSON by default.")
        elif drive_data is not None and drive_ts is not None:
            Logger.info("Using Google Drive JSON (only available).")
            JsonManager.save(drive_data, filepath)
        elif local_data is not None and local_ts is not None:
            Logger.info("Using local JSON (only available).")
        else:
            Logger.warning(
                "No valid JSON found in either local or Drive sources. A new marketdata "
                "file will be created from scratch, and all symbol data will be downloaded."
            )
            now = pd.Timestamp.now(tz="UTC").isoformat()
            JsonManager.save(
                {"id": None, "last_updated": now, "last_check": now, "symbols": []},
                filepath,
            )

    @staticmethod
    def synchronize_marketdata_with_drive(filepath: Optional[str] = None) -> Any:
        """Synchronize local and Drive JSONs by preserving the most recent version locally."""
        filepath = None if not filepath or not filepath.strip() else filepath.strip()
        if not filepath:
            raise ValueError("Filepath is empty")
        local_data = MarketDataVersion(
            *MarketDataSyncManager._load_local_marketdata(filepath)
        )
        drive_data = MarketDataVersion(
            *MarketDataSyncManager._load_drive_marketdata(filepath)
        )
        MarketDataSyncManager._resolve_data_conflict(local_data, drive_data, filepath)
        local_data = MarketDataVersion(
            *MarketDataSyncManager._load_local_marketdata(filepath)
        )
        if (
            local_data is not None
            and local_data.data is None
            and local_data.timestamp is None
            and local_data.file_id is None
        ):
            local_data = None
        if (
            drive_data is not None
            and drive_data.data is None
            and drive_data.timestamp is None
            and drive_data.file_id is None
        ):
            drive_data = None
        return {
            "local_data": local_data is not None,
            "drive_data": drive_data is not None,
        }
