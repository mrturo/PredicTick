"""Market raw data for managing symbol metadata and historical prices."""

import secrets
import string
from datetime import datetime, timezone
from sqlite3.dbapi2 import Timestamp
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore

from src.market_data.utils.intervals.interval import Interval
from src.utils.config.parameters import ParameterLoader
from src.utils.io.json_manager import JsonManager


class RawData:
    """Centralized access point for managing symbol raw data."""

    _PARAMS = ParameterLoader()
    _RAW_MARKETDATA_FILEPATH = _PARAMS.get("raw_marketdata_filepath")
    _STALE_DAYS_THRESHOLD = _PARAMS.get("stale_days_threshold")
    _SYMBOL_REPO = _PARAMS.symbol_repo

    _id: Optional[str] = None
    _interval: Optional[str] = None
    _last_check: Optional[pd.Timestamp] = None
    _last_updated: Optional[pd.Timestamp] = None
    _latest_price_date: Optional[pd.Timestamp] = None
    _stale_symbols: List[str] = []
    _symbols: Dict[str, dict] = {}

    @staticmethod
    def get_id() -> Optional[str]:
        """Return id."""
        return RawData._id

    @staticmethod
    def set_id(new_id: Optional[str]) -> None:
        """Manually set an id."""
        RawData._id = new_id

    @staticmethod
    def set_new_id() -> None:
        """Set a new random id."""
        chars = string.ascii_letters + string.digits
        new_id = "".join(secrets.choice(chars) for _ in range(10))
        RawData._id = new_id

    @staticmethod
    def get_interval() -> Optional[str]:
        """Retrieve the currently set interval value for raw data."""
        return RawData._interval

    @staticmethod
    def set_interval(interval: Optional[str]) -> None:
        """Set the interval value to be used for raw data operations."""
        RawData._interval = interval

    @staticmethod
    def get_last_check() -> Optional[pd.Timestamp]:
        """Return the last check timestamp."""
        return RawData._last_check

    @staticmethod
    def set_last_check(last_check: Optional[pd.Timestamp]) -> None:
        """Set the last check timestamp."""
        RawData._last_check = last_check

    @staticmethod
    def get_last_updated() -> Optional[pd.Timestamp]:
        """Return the last update timestamp."""
        return RawData._last_updated

    @staticmethod
    def set_last_updated(last_updated: Optional[pd.Timestamp]) -> None:
        """Set the last update timestamp."""
        RawData._last_updated = last_updated

    @staticmethod
    def get_latest_price_date() -> Optional[pd.Timestamp]:
        """Return the latest price timestamp among all symbols."""
        return RawData._latest_price_date

    @staticmethod
    def set_latest_price_date(latest_price_date: Optional[pd.Timestamp]) -> None:
        """Manually set the latest price date."""
        RawData._latest_price_date = latest_price_date

    @staticmethod
    def get_stale_symbols() -> List[str]:
        """Return the list of stale symbols."""
        return RawData._stale_symbols

    @staticmethod
    def set_stale_symbols(stale_symbols: List[str]) -> None:
        """Manually set stale symbols."""
        RawData._stale_symbols = stale_symbols

    @staticmethod
    def get_symbols() -> Dict[str, dict]:
        """Return all loaded symbols."""
        return RawData._symbols

    @staticmethod
    def set_symbols(symbols: Dict[str, dict]) -> None:
        """Set the dictionary of symbols."""
        RawData._symbols = symbols

    @staticmethod
    def get_symbol(symbol: str):
        """Retrieve metadata for a specific symbol."""
        return RawData._symbols.get(symbol)

    @staticmethod
    def set_symbol(symbol: str, historical_prices: List[dict], metadata: dict):
        """Set symbol metadata and historical prices."""
        RawData._symbols[symbol] = {
            "currency": metadata.get("currency"),
            "exchange": metadata.get("exchange"),
            "historical_prices": historical_prices,
            "industry": metadata.get("industry"),
            "name": metadata.get("name"),
            "sector": metadata.get("sector"),
            "symbol": symbol,
            "type": metadata.get("type"),
        }

    @staticmethod
    def _get_filepath(filepath: Optional[str], default=None) -> Optional[str]:
        if filepath is None or len(filepath.strip()) == 0:
            return default
        return filepath

    @staticmethod
    def normalize_historical_prices(
        symbols: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Normalizes the historical price data for a list of symbol entries.

        Each entry must contain a 'symbol' and 'historical_prices' key.
        Timestamps in 'historical_prices' are converted to UTC-aware datetimes.
        Interpolates zero volume values if present.
        """
        normalized = {}
        for entry in symbols:
            symbol = entry.get("symbol")
            df = pd.DataFrame(entry["historical_prices"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            # Interpolate or forward-fill volume = 0
            if "volume" in df.columns:
                zero_volume_mask = df["volume"] == 0
                if zero_volume_mask.any():
                    df["volume"] = df["volume"].replace(0, pd.NA)
                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                    df["volume"] = df["volume"].infer_objects()
                    df["volume"] = df["volume"].interpolate(
                        method="linear", limit_direction="forward"
                    )
                    df["volume"] = df["volume"].bfill()
                    df["volume"] = df["volume"].infer_objects()
            entry["historical_prices"] = df.to_dict(orient="records")
            normalized[symbol] = entry
        return normalized

    @staticmethod
    def exist(filepath: Optional[str] = None) -> bool:
        """Check if market raw data file exists at the specified or default path."""
        local_filepath: Optional[str] = RawData._get_filepath(
            filepath, RawData._RAW_MARKETDATA_FILEPATH
        )
        return JsonManager.exists(local_filepath)

    @staticmethod
    def load(filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load market raw data from disk and initialize internal structures."""
        local_filepath: Optional[str] = RawData._get_filepath(
            filepath, RawData._RAW_MARKETDATA_FILEPATH
        )
        raw_data = (
            JsonManager.load(local_filepath)
            if JsonManager.exists(local_filepath) is True
            else None
        )
        raw_last_updated: Optional[pd.Timestamp] = None
        raw_last_check: Optional[pd.Timestamp] = None
        raw_id: Optional[str] = None
        raw_symbols: list = []
        if raw_data is not None:
            try:
                raw_id = raw_data["id"]
            except (KeyError, TypeError):
                raw_id = None
            if not isinstance(raw_id, str):
                raw_id = None
            try:
                raw_last_updated = raw_data["last_updated"]
            except (KeyError, TypeError):
                raw_last_updated = None
            try:
                raw_last_check = raw_data["last_check"]
            except (KeyError, TypeError):
                raw_last_check = None
            try:
                raw_symbols = raw_data["symbols"]
            except (KeyError, TypeError):
                raw_symbols = []
            if not isinstance(raw_symbols, list):
                raw_symbols = []
        now: Timestamp = pd.Timestamp.now(tz="UTC")
        RawData.set_last_check(
            pd.to_datetime(raw_last_check)
            if raw_last_check is not None and isinstance(raw_last_check, str)
            else now
        )
        if raw_id is None or len(raw_id.strip()) == 0:
            RawData.set_new_id()
            RawData.set_last_updated(now)
        else:
            RawData.set_id(raw_id.strip())
            RawData.set_last_updated(
                pd.to_datetime(raw_last_updated)
                if raw_last_updated is not None and isinstance(raw_last_updated, str)
                else RawData._last_check
            )
        if raw_data is not None:
            RawData._interval = Interval.market_raw_data()
        enriched = RawData.normalize_historical_prices(raw_symbols)
        RawData.set_symbols(
            {
                sym: data
                for sym, data in enriched.items()
                if sym not in RawData._SYMBOL_REPO.get_invalid_symbols()
            }
        )
        RawData._update_stale_symbols()
        return {
            "id": RawData.get_id(),
            "last_updated": RawData.get_last_updated(),
            "last_check": RawData.get_last_check(),
            "interval": RawData.get_interval(),
            "symbols": RawData.get_symbols(),
            "filepath": local_filepath,
        }

    @staticmethod
    def save(filepath: Optional[str] = None) -> dict[str, Any]:
        """Persist current symbol data to disk."""
        result = {
            "id": RawData.get_id(),
            "last_updated": RawData.get_last_updated(),
            "last_check": datetime.now(timezone.utc).isoformat(),
            "interval": RawData.get_interval(),
            "symbols": list(RawData.get_symbols().values()),
        }
        local_filepath = RawData._get_filepath(
            filepath, RawData._RAW_MARKETDATA_FILEPATH
        )
        JsonManager.save(result, local_filepath)
        return result

    @staticmethod
    def _update_stale_symbols() -> None:
        """Identify and mark symbols with outdated price data."""
        latest_date = RawData.get_latest_price_date() or pd.Timestamp.now(tz="UTC")
        current_invalids = set(RawData._SYMBOL_REPO.get_invalid_symbols())
        stale, updated_invalids = RawData._detect_stale_symbols(
            latest_date, current_invalids
        )
        RawData._stale_symbols = stale
        RawData._SYMBOL_REPO.set_invalid_symbols(updated_invalids)

    @staticmethod
    def _detect_stale_symbols(
        latest_date: pd.Timestamp, invalid_symbols: set
    ) -> tuple[list, set]:
        """Detect stale symbols and return updated lists (for internal testing)."""
        stale = []
        for symbol, entry in RawData.get_symbols().items():
            datetimes = [
                pd.to_datetime(p["datetime"], utc=True, errors="coerce")
                for p in entry.get("historical_prices", [])
            ]
            symbol_latest = max(filter(pd.notnull, datetimes), default=None)
            if (
                not symbol_latest
                or (latest_date - symbol_latest).days > RawData._STALE_DAYS_THRESHOLD
            ):
                stale.append(symbol)
                invalid_symbols.add(symbol)
        return stale, invalid_symbols
