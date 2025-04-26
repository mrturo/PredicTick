"""Module for managing JSON configurations and data files."""

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd

from utils.logger import Logger


class JsonManager:
    """Class for handling JSON file operations."""

    @staticmethod
    def exists(filepath: str) -> bool:
        """Check if a file exists at the given path."""
        return os.path.exists(filepath)

    @staticmethod
    def load(filepath: str) -> Any:
        """Load JSON data from a file.

        Args:
            filepath (str): Path to the JSON file.

        Returns:
            Any: Parsed data or None if error occurs.
        """
        if not JsonManager.exists(filepath):
            Logger.warning(f"File not found: {filepath}")
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, TypeError, json.JSONDecodeError) as e:
            Logger.error(f"Error loading JSON file {filepath}: {e}")
            return None

    @staticmethod
    def save(data: Any, filepath: str) -> bool:
        """Save data to a JSON file.

        Args:
            data (Any): Data to be serialized.
            filepath (str): Destination file path.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:

            def custom_serializer(obj):
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                raise TypeError(
                    f"Object of type {type(obj).__name__} is not JSON serializable"
                )

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, default=custom_serializer)
            return True
        except (OSError, TypeError) as e:
            Logger.error(f"Error saving JSON file {filepath}: {e}")
            return False

    @staticmethod
    def delete(filepath: str) -> bool:
        """Delete a JSON file.

        Args:
            filepath (str): File path to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        if not JsonManager.exists(filepath):
            Logger.warning(f"File to delete not found: {filepath}")
            return False
        try:
            os.remove(filepath)
            return True
        except OSError as e:
            Logger.error(f"Error deleting JSON file {filepath}: {e}")
            return False
