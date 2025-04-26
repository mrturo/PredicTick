"""Handles loading, updating, and saving of structured market data to disk."""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from utils.logger import Logger
from utils.parameters import ParameterLoader

# Global parameter instance
parameters = ParameterLoader()


class FileManager:
    """Handles reading and writing of market data files with symbol metadata."""

    _index: Dict[str, dict] = {}
    _last_update: Optional[pd.Timestamp] = None

    @staticmethod
    def last_update() -> pd.Timestamp:
        """Get last update of data market."""
        return FileManager._last_update

    @staticmethod
    def load(file_path: Optional[str] = None) -> List[dict]:
        """Loads market data from JSON file."""
        file_path = file_path or parameters["marketdata_filepath"]

        if not os.path.exists(file_path):
            Logger.error("Market data file does not exist or has wrong format.")
            FileManager._index = {}
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            Logger.error(f"JSON decode error: {e.msg} (line {e.lineno}, col {e.colno})")
            Logger.error(f"Consider validating or recreating the file: {file_path}")
            raise
        except Exception as e:
            Logger.error(f"Failed to load JSON file: {e}")
            raise

        stocks = data.get("stocks", [])

        FileManager._last_update = (
            pd.to_datetime(data["last_updated"])
            if isinstance(data.get("last_updated"), str)
            else data.get("last_updated", pd.Timestamp.now(tz="UTC"))
        )

        for entry in stocks:
            df = pd.DataFrame(entry["historical_prices"])
            df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
            entry["historical_prices"] = df.to_dict(orient="records")

        FileManager._index = {entry["symbol"]: entry for entry in stocks}
        return stocks

    @staticmethod
    def find_symbol(symbol: str):
        """Finds a symbol entry in loaded market data."""
        return FileManager._index.get(symbol)

    @staticmethod
    def update_symbol(symbol: str, historical_prices: List[dict], metadata: dict):
        """Updates or adds a symbol entry with new historical prices and metadata."""
        FileManager._index[symbol] = {
            "symbol": symbol,
            "name": metadata.get("Name"),
            "type": metadata.get("Type"),
            "sector": metadata.get("Sector"),
            "currency": metadata.get("Currency"),
            "exchange": metadata.get("Exchange"),
            "historical_prices": historical_prices,
        }

    @staticmethod
    def save(file_path: Optional[str] = None) -> None:
        """Saves all loaded market data back to the JSON file."""
        file_path = file_path or parameters["marketdata_filepath"]
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "stocks": list(FileManager._index.values()),
        }

        def convert_timestamps(obj):
            """Recursively convert Timestamps to ISO strings."""
            if isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_timestamps(i) for i in obj]
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        json_safe_data = convert_timestamps(data)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(json_safe_data, file, indent=2)
        Logger.success(f"Market data saved successfully: {file_path}")
