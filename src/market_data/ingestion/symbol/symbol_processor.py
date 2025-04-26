"""Handles symbol-level market data update logic."""

import time
from typing import Any, List, Optional

import pandas as pd  # type: ignore

from src.market_data.ingestion.downloaders.downloader import Downloader
from src.market_data.ingestion.raw.raw_data import RawData
from src.market_data.utils.intervals.interval import Interval
from src.utils.config.parameters import ParameterLoader
from src.utils.io.logger import Logger


# pylint: disable=too-few-public-methods
class SymbolProcessor:
    """Class for orchestrating raw symbol-level market data updates.

    This includes validation, and incremental downloading via RawData.
    """

    _PARAMS = ParameterLoader()
    _BLOCK_DAYS = _PARAMS.get("block_days")
    _DOWNLOAD_RETRIES = _PARAMS.get("download_retries")
    _RETRY_SLEEP_SECONDS = _PARAMS.get("retry_sleep_seconds")
    _RAW_DATA_INTERVAL = Interval.market_raw_data()
    _EXCHANGE_DEFAULT = _PARAMS.get("exchange_default")

    _DOWNLOADER = Downloader(_BLOCK_DAYS, _DOWNLOAD_RETRIES, _RETRY_SLEEP_SECONDS)

    @staticmethod
    def _is_valid_symbol(symbol: str) -> bool:
        """Check if the given symbol has valid metadata indicating it is active and tradable."""
        try:
            info = SymbolProcessor._DOWNLOADER.get_metadata(symbol)
            return info is not None and info.get("name") is not None
        except (KeyError, ValueError):
            return False

    @staticmethod
    def _should_skip_symbol(
        symbol: str, entry: Optional[dict], invalid_symbols: set
    ) -> bool:
        """Check if a symbol should be skipped due to missing or invalid data."""
        if not (entry and entry.get("historical_prices")):
            if symbol in invalid_symbols:
                Logger.warning(f"    Skipping {symbol}: previously marked as invalid.")
                return True
            if not SymbolProcessor._is_valid_symbol(symbol):
                Logger.warning(f"    Skipping {symbol}: appears invalid or delisted.")
                invalid_symbols.add(symbol)
                return True
        return False

    @staticmethod
    def _process_single_symbol(
        symbol: str, exchange_id: str, entry: Optional[dict]
    ) -> tuple[str, str, Optional[str], Optional[int]]:
        """Process historical price update for a single symbol."""
        existing_df = (
            pd.DataFrame(entry["historical_prices"]) if entry else pd.DataFrame()
        )
        existing_metadata: dict[str, Any] = (
            {
                k: entry.get(k)
                for k in ["name", "type", "sector", "industry", "currency", "exchange"]
            }
            if entry
            else {}
        )
        if not existing_df.empty and "datetime" in existing_df.columns:
            last_dt_ts = pd.to_datetime(
                existing_df["datetime"], utc=True, errors="coerce"
            )
            if isinstance(last_dt_ts, pd.Series):
                last_dt = (
                    last_dt_ts.dropna().max()
                    if isinstance(last_dt_ts, pd.Series)
                    else (last_dt_ts if pd.notnull(last_dt_ts) else None)  # type: ignore
                )
        else:
            last_dt = None
        if not (isinstance(last_dt, pd.Timestamp) or last_dt is None):
            last_dt = None
        incremental, remaining, seconds = (
            SymbolProcessor._DOWNLOADER.get_historical_prices(
                symbol, exchange_id, last_dt
            )
        )
        if not incremental.empty and not existing_df.empty:
            incremental = incremental[
                ~incremental["datetime"].isin(existing_df["datetime"])
            ].reset_index(drop=True)
        if incremental.empty:
            updated_entry: dict[str, Any] = {
                "historical_prices": existing_df.to_dict(orient="records")
            }
            RawData.set_symbol(
                symbol, updated_entry["historical_prices"], existing_metadata
            )
            return "no_new", existing_metadata.get("name", ""), remaining, seconds
        combined_df = (
            pd.concat([incremental, existing_df], axis=0)
            .drop_duplicates("datetime")
            .sort_values("datetime")
        )
        combined_df["datetime"] = pd.to_datetime(
            combined_df["datetime"], utc=True, errors="coerce"
        )
        metadata: Optional[dict[str, Any]] = existing_metadata
        if not existing_metadata.get("name"):
            metadata = SymbolProcessor._DOWNLOADER.get_metadata(symbol)
            Logger.debug(f"    Metadata fetched for {symbol}")
        metadata = metadata if metadata is not None else {}
        updated_entry: dict[str, Any] = {
            "historical_prices": combined_df.to_dict(orient="records")
        }
        RawData.set_symbol(symbol, updated_entry["historical_prices"], metadata)
        Logger.debug(f"    Updated data for {symbol} with {len(incremental)} new rows.")
        return "updated", metadata.get("name", ""), remaining, seconds

    @staticmethod
    def _resolve_exchange_id(entry: Optional[dict[str, Any]]) -> str:  # noqa: D401
        """Return a valid *uppercase* exchange identifier for *entry*.

        The logic lives in a dedicated helper to minimise the cognitive load
        inside :py:meth:`process_symbols` and to prevent pylint ``R0914`` by
        reducing the number of required local variables.
        """
        exchange: Optional[str] = None
        if entry is not None:
            raw_exchange = entry.get("exchange")
            if isinstance(raw_exchange, str) and raw_exchange.strip():
                exchange = raw_exchange
        if exchange is None:
            default = SymbolProcessor._EXCHANGE_DEFAULT
            if isinstance(default, str) and default.strip():
                exchange = default
            else:
                raise ValueError("Exchange symbol and default exchange not found")
        return exchange.strip().upper()

    @staticmethod
    def process_symbols(
        symbols: List[str], invalid_symbols: set[str]
    ) -> dict[str, Any]:
        """Batch‑process *symbols* updating raw data and feature stores.

        This refactor fixes ``R0914: too‑many‑locals`` by:

        * Extracting exchange resolution to :py:meth:`_resolve_exchange_id`.
        * Collapsing the *remaining* artefacts into a single tuple variable.
        * Avoiding throw‑away aliases and redundant temporaries.
        """
        counts: dict[str, Any] = {
            "updated": 0,
            "skipped": 0,
            "no_new": 0,
            "failed": 0,
            "invalid_symbols": [],
        }
        RawData.load()
        invalid_symbols_set = set(invalid_symbols)
        best_remaining: tuple[Optional[str], Optional[int]] = (None, None)
        for idx, symbol in enumerate(symbols, start=1):
            Logger.info(f" * Processing symbol ({idx}/{len(symbols)}): {symbol}")
            entry = RawData.get_symbol(symbol)
            start_ts = time.time()
            if SymbolProcessor._should_skip_symbol(symbol, entry, invalid_symbols_set):
                counts["skipped"] += 1
                continue
            try:
                exchange_id = SymbolProcessor._resolve_exchange_id(entry)
                result, name, rem_str, rem_int = SymbolProcessor._process_single_symbol(
                    symbol,
                    exchange_id,
                    entry,
                )
                if (
                    rem_str is not None
                    and rem_int is not None
                    and (best_remaining[1] is None or rem_int < best_remaining[1])
                ):
                    best_remaining = (rem_str.strip(), rem_int)
                counts[result] += 1
                Logger.success(
                    f"    Completed {symbol} ({name.strip()}) in {(time.time() - start_ts):.2f}s",
                )
            except Exception as err:  # pylint: disable=broad-exception-caught
                Logger.error(f"  Failed to update {symbol}: {err}")
                counts["failed"] += 1
        RawData.set_interval(SymbolProcessor._RAW_DATA_INTERVAL)
        if counts["updated"]:
            RawData.set_new_id()
            RawData.set_last_updated(pd.Timestamp.now(tz="UTC"))
        RawData.save()
        counts["invalid_symbols"] = invalid_symbols_set
        counts["remaining_str"], counts["remaining_int"] = best_remaining
        return counts
