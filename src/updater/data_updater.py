"""Main orchestrator for updating and validating historical market data."""

import json
import os
import time
from datetime import timezone
from typing import List

import pandas as pd
import yfinance as yf

from updater.file_manager import FileManager
from updater.market_data_downloader import MarketDataDownloader
from updater.market_data_service import MarketDataService
from utils.logger import Logger
from utils.parameters import ParameterLoader

# Global parameter instance
parameters = ParameterLoader()


def load_invalid_symbols() -> set:
    """Loads list of invalid symbols."""
    if os.path.exists(parameters["symbols_invalid_filepath"]):
        with open(
            parameters["symbols_invalid_filepath"], "r", encoding="utf-8"
        ) as file:
            return set(json.load(file))
    return set()


def save_invalid_symbols(invalid_symbols: set):
    """Saves invalid symbols set to disk."""
    with open(parameters["symbols_invalid_filepath"], "w", encoding="utf-8") as file:
        json.dump(sorted(list(invalid_symbols)), file, indent=2)


def is_valid_symbol(symbol: str) -> bool:
    """Checks if the symbol is valid and has market data."""
    try:
        info = yf.Ticker(symbol).info
        return "shortName" in info and info.get("regularMarketPrice") is not None
    except (KeyError, ValueError):
        return False


def fetch_symbol_metadata(symbol: str) -> dict:
    """Fetches symbol metadata using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        quote_type = info.get("quoteType", "").lower()
        metadata = {
            "Name": info.get("shortName", symbol),
            "Type": (
                "ETF"
                if "etf" in quote_type
                else "Crypto" if "crypto" in quote_type else "Stock"
            ),
            "Sector": info.get("sector", "Unknown"),
            "Currency": info.get("currency", "USD"),
            "Exchange": info.get("exchange", "Unknown"),
        }
        Logger.success(f"  Metadata fetched for {symbol}")
        return metadata
    except (KeyError, ValueError) as error:
        Logger.error(f"  Failed to fetch metadata for {symbol}: {error}")
        return {
            "Name": symbol,
            "Type": "Unknown",
            "Sector": "Unknown",
            "Currency": "USD",
            "Exchange": "Unknown",
        }


def enrich_historical_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches historical prices with computed features."""
    if df.empty or any(col not in df.columns for col in ["Close", "Open", "Volume"]):
        Logger.warning("  Cannot enrich historical prices: essential columns missing.")
        return df

    df = df.sort_values("Datetime")
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = (df["High"] - df["Low"]) / df["Open"].replace(0, pd.NA)
    df["PriceChange"] = df["Close"] - df["Open"]
    df["VolumeChange"] = df["Volume"].pct_change()
    df["TypicalPrice"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["AveragePrice"] = (df["High"] + df["Low"]) / 2
    df["IsGreenCandle"] = (df["Close"] > df["Open"]).astype(int)
    df["Range"] = df["High"] - df["Low"]
    df["RelativeVolume"] = (
        df["Volume"]
        / df["Volume"].rolling(window=parameters["volume_window"], min_periods=1).mean()
    )
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    Logger.success("  Enriched historical prices with additional metadata.")
    return df


def print_sumary(symbols, procesed_symbols, start_time) -> None:
    """
    Print a summary report of the symbol update process.

    Args:
        symbols (List[str]): List of all symbols processed.
        procesed_symbols (dict): Dictionary with update counts including keys:
                                 'updated', 'skipped', 'no_new', 'failed'.
        start_time (float): Timestamp indicating when the process started.
    """
    updated = procesed_symbols["updated"]
    skipped = procesed_symbols["skipped"]
    no_new = procesed_symbols["no_new"]
    failed = procesed_symbols["failed"]

    Logger.success("Summary")
    Logger.simple(f"  * Symbols processed: {len(symbols)}")
    Logger.simple(f"  * Symbols updated: {updated}")
    Logger.simple(f"  * Symbols skipped: {skipped}")
    Logger.simple(f"  * Symbols with no new data: {no_new}")
    if failed > 0:
        Logger.simple(f"  * Symbols failed: {failed}")
    total_time = time.time() - start_time
    Logger.simple(f"  Total time: {int(total_time // 60)}m {int(total_time % 60)}s")


def process_symbols(symbols, invalid_symbols, service):
    """
    Download and update historical market data for a list of symbols.

    Args:
        symbols (List[str]): List of ticker symbols to process.
        invalid_symbols (set): Set of previously invalid or skipped symbols.
        service (MarketDataService): Service instance to manage downloading and enrichment.

    Returns:
        dict: Summary statistics including updated, skipped, no_new, failed counts,
              and the final set of invalid symbols.
    """
    counts = {
        "updated": 0,
        "skipped": 0,
        "no_new": 0,
        "failed": 0,
    }
    FileManager.load()
    for idx, symbol in enumerate(symbols, start=1):
        symbol_start_time = time.time()
        Logger.info(
            f"Processing symbol ({idx}/{len(symbols)} → "
            f"{(((idx-1)*100)/len(symbols)):.2f}%): {symbol}"
        )
        entry = FileManager.find_symbol(symbol)
        if not (entry and entry.get("historical_prices")):
            if symbol in invalid_symbols:
                Logger.warning(f"  Skipping {symbol}: previously marked as invalid.")
                counts["skipped"] = counts["skipped"] + 1
                continue
            if not is_valid_symbol(symbol):
                Logger.warning(f"  Skipping {symbol}: appears invalid or delisted.")
                invalid_symbols.add(symbol)
                counts["skipped"] = counts["skipped"] + 1
                continue

        try:
            existing_df = (
                pd.DataFrame(entry["historical_prices"]) if entry else pd.DataFrame()
            )
            existing_metadata = (
                {
                    k: entry.get(k)
                    for k in ["Name", "Type", "Sector", "Currency", "Exchange"]
                }
                if entry
                else {}
            )

            if existing_df.empty or "Datetime" not in existing_df.columns:
                Logger.info(
                    f"  No existing data for {symbol}. Proceeding with full download."
                )
                last_dt = None
            else:
                existing_df["Datetime"] = pd.to_datetime(
                    existing_df["Datetime"], utc=True, errors="coerce"
                )
                last_dt = existing_df["Datetime"].max()

            incremental = service.get_incremental_data(symbol, last_dt)

            if not incremental.empty and not existing_df.empty:
                incremental = incremental[
                    ~incremental["Datetime"].isin(existing_df["Datetime"])
                ]

            if not incremental.empty:
                combined_df = (
                    pd.concat([incremental, existing_df])
                    .drop_duplicates(subset=["Datetime"], keep="first")
                    .sort_values("Datetime")
                )
                combined_df.reset_index(drop=True, inplace=True)
                combined_df = enrich_historical_prices(combined_df)
                combined_df["Datetime"] = combined_df["Datetime"].dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                FileManager.update_symbol(
                    symbol,
                    combined_df.to_dict(orient="records"),
                    (
                        fetch_symbol_metadata(symbol)
                        if not existing_metadata.get("Name")
                        else existing_metadata
                    ),
                )
                Logger.success(
                    f"  Updated data for {symbol} with {len(incremental)} new rows."
                )
                counts["updated"] = counts["updated"] + 1
            else:
                if last_dt:
                    last_dt_str = pd.to_datetime(last_dt).astimezone(timezone.utc)
                    last_dt_str = last_dt_str.strftime("%Y-%m-%d %H:%M:%S UTC")
                    Logger.warning(
                        f"  No new data found for {symbol}. "
                        f"Last local datetime: {last_dt_str}"
                    )
                counts["no_new"] = counts["no_new"] + 1

        except (ValueError, IOError) as error:
            Logger.error(f"  Failed to update {symbol}: {error}")
            counts["failed"] = counts["failed"] + 1
            continue

        Logger.success(
            f"  Completed {symbol} in {(time.time() - symbol_start_time):.2f} seconds."
        )
    FileManager.save()
    return {
        "updated": counts["updated"],
        "skipped": counts["skipped"],
        "no_new": counts["no_new"],
        "failed": counts["failed"],
        "invalid_symbols": invalid_symbols,
    }


def update_data():
    """Updates all symbols' historical market data and enriches them."""
    start_time = time.time()
    symbols = sorted(parameters["training_symbols"])
    procesed_symbols = process_symbols(
        symbols,
        load_invalid_symbols(),
        MarketDataService(
            parameters["interval"],
            parameters["block_days"],
            MarketDataDownloader(
                parameters["download_retries"], parameters["retry_sleep_seconds"]
            ),
        ),
    )
    save_invalid_symbols(procesed_symbols["invalid_symbols"])
    print_sumary(
        symbols,
        procesed_symbols,
        start_time,
    )


def validate_market_data():
    """Validates integrity of the market data JSON file."""
    Logger.success("Validation Summary")

    raw_data = FileManager.load()

    # Compatibilidad con formato nuevo
    if isinstance(raw_data, dict) and "stocks" in raw_data:
        raw_data = raw_data["stocks"]

    if not isinstance(raw_data, list):
        Logger.error(
            "Invalid data format for validation. Expected list of symbol entries."
        )
        return

    total_symbols = len(raw_data)
    all_symbols_in_file = {
        entry.get("symbol") for entry in raw_data if "Symbol" in entry
    }

    invalid_symbols = load_invalid_symbols()
    invalid_symbols_in_file = all_symbols_in_file.intersection(invalid_symbols)
    valid_expected_symbols = total_symbols - len(invalid_symbols_in_file)

    Logger.simple(f"  * Total symbols in file: {total_symbols}")
    if valid_expected_symbols != total_symbols:
        Logger.info(f"  * Expected valid symbols: {valid_expected_symbols}")

    success_symbols, failed_symbols, issues = validate_symbols(
        raw_data, parameters["required_market_columns"]
    )

    Logger.simple(f"  * Symbols passed: {success_symbols}")
    Logger.simple(f"  * Symbols failed: {failed_symbols}")

    if issues:
        Logger.warning("Issues encountered:")
        for symbol, error in issues.items():
            Logger.simple(f"   - {symbol}: {error}")


def load_market_data(file_path: str):
    """Loads market data from JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_symbols(data: List[dict], required_columns: List[str]):
    """Validates all symbols for structure, columns, and chronological order."""
    success_count = 0
    fail_count = 0
    issues = {}

    for entry in data:
        symbol = entry.get("symbol", "<UNKNOWN>")
        historical_prices = entry.get("historical_prices", [])

        try:
            df = pd.DataFrame(historical_prices)
            if df.empty:
                raise ValueError("Historical prices are empty.")

            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")

            df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
            df = df.sort_values("Datetime")

            if not df["Datetime"].is_monotonic_increasing:
                raise ValueError("Datetime column is not sorted.")

            success_count += 1

        except (ValueError, KeyError, TypeError) as error:
            fail_count += 1
            issues[symbol] = str(error)

    return success_count, fail_count, issues


if __name__ == "__main__":
    update_data()
    Logger.separator()
    validate_market_data()
