"""This module defines the EnrichedData class which processes, enriches,.

and manages historical market data with technical indicators,
temporal features, and event annotations. It supports loading raw
market data, computing derived features, resampling time intervals,
and formatting enriched outputs for use in downstream ML pipelines.

Dependencies include pandas, numpy, and internal utilities for
calendar management, logging, JSON operations, interval handling,
and technical indicators.
"""

import datetime
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.market_data.ingestion.raw.raw_data import RawData
from src.market_data.processing.enrichment.indicator_builder import \
    IndicatorBuilder
from src.market_data.processing.enrichment.market_context import MarketContext
from src.market_data.processing.resampling.time_resampler import TimeResampler
from src.market_data.utils.intervals.interval import (Interval,
                                                      IntervalConverter)
from src.market_data.utils.storage.market_data_sync_manager import \
    MarketDataSyncManager
from src.utils.config.parameters import ParameterLoader
from src.utils.exchange.calendar_manager import CalendarManager
from src.utils.io.json_manager import JsonManager
from src.utils.io.logger import Logger

TradingBounds = Dict[str, Dict[datetime.date, Tuple[datetime.time, datetime.time]]]
DailyBounds = Dict[datetime.date, Tuple[datetime.time, datetime.time]]
DayBlocks = List[Dict[str, Any]]
DayObject = Dict[str, DayBlocks]
WeekObject = Dict[str, DayObject]
MarketRanges = Dict[str, Dict[str, Union[str, WeekObject]]]


class EnrichedData:
    """EnrichedData class for transforming raw historical market data.

    into enriched datasets with technical indicators and temporal features.

    Provides functionality to load, process, enrich, and save market data.
    Includes utilities for scaling, resampling, and formatting enriched data.
    """

    _PARAMS = ParameterLoader()
    _ATR_WINDOW = _PARAMS.get("atr_window")
    _BOLLINGER_BAND_METHOD = _PARAMS.get("bollinger_band_method")
    _BOLLINGER_WINDOW = _PARAMS.get("bollinger_window")
    _ENRICHED_MARKETDATA_FILEPATH = _PARAMS.get("enriched_marketdata_filepath")
    _MACD_FAST = _PARAMS.get("macd_fast")
    _MACD_SIGNAL = _PARAMS.get("macd_signal")
    _MACD_SLOW = _PARAMS.get("macd_slow")
    _OBV_FILL_METHOD = _PARAMS.get("obv_fill_method")
    _REQUIRED_MARKET_ENRICHED_COLUMNS: list[str] = list(
        dict.fromkeys(
            (_PARAMS.get("required_market_raw_columns") or [])
            + (_PARAMS.get("required_market_enriched_columns") or [])
        )
    )
    _RSI_WINDOW = _PARAMS.get("rsi_window_backtest")
    _STOCH_RSI_MIN_PERIODS = _PARAMS.get("stoch_rsi_min_periods")
    _STOCH_RSI_WINDOW = _PARAMS.get("stoch_rsi_window")
    _WEEKDAYS: List[str] = _PARAMS.get("weekdays")
    _WILLIAMS_R_WINDOW = _PARAMS.get("williams_r_window")

    _ENRICHED_DATA_INTERVAL = Interval.market_enriched_data()
    _RAW_DATA_INTERVAL = Interval.market_raw_data()

    _id: Optional[str] = None
    _interval: Optional[str] = None
    _last_updated: Optional[pd.Timestamp] = None
    _ranges: Any = {}
    _symbols: Dict[str, dict] = {}

    @staticmethod
    def _get_filepath(
        filepath: Optional[str], default: Optional[str] = None
    ) -> Optional[str]:
        """Return a cleaned filepath, falling back to a default if provided."""
        if filepath is None or len(filepath.strip()) == 0:
            return default.strip() if default is not None else None
        return filepath.strip()

    @staticmethod
    def get_id() -> Optional[str]:
        """Get the current enriched data identifier."""
        return EnrichedData._id

    @staticmethod
    def set_id(file_id: Optional[str]) -> None:
        """Set the enriched data identifier."""
        EnrichedData._id = file_id

    @staticmethod
    def get_interval() -> Optional[str]:
        """Get the current data interval used for enrichment."""
        return EnrichedData._interval

    @staticmethod
    def set_interval(interval: Optional[str]) -> None:
        """Set the data interval used for enrichment."""
        EnrichedData._interval = interval

    @staticmethod
    def get_last_updated() -> Optional[pd.Timestamp]:
        """Retrieve the last updated timestamp of enriched data."""
        return EnrichedData._last_updated

    @staticmethod
    def set_last_updated(last_updated: Optional[pd.Timestamp]) -> None:
        """Set the timestamp for when data was last enriched."""
        EnrichedData._last_updated = last_updated

    @staticmethod
    def get_ranges() -> Any:
        """Get the scaling ranges used for price and volume features."""
        return EnrichedData._ranges

    @staticmethod
    def set_ranges(ranges: Any) -> None:
        """Set the scaling ranges for price and volume features."""
        EnrichedData._ranges = ranges

    @staticmethod
    def get_market_time() -> Optional[List]:
        """Get the market_time."""
        return EnrichedData._market_time

    @staticmethod
    def set_market_time(market_time: Optional[List]) -> None:
        """Set the market_time."""
        EnrichedData._market_time = market_time

    @staticmethod
    def get_symbol(symbol: str) -> Optional[dict]:
        """Retrieve enriched data for a specific symbol."""
        return EnrichedData._symbols.get(symbol)

    @staticmethod
    def get_symbols() -> Dict[str, dict]:
        """Get the full dictionary of enriched symbols."""
        return EnrichedData._symbols

    @staticmethod
    def set_symbols(symbols: Dict[str, dict]) -> None:
        """Replace the full symbol dictionary with new enriched data."""
        EnrichedData._symbols = symbols

    @staticmethod
    def _convert_records(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Normalize numeric types in records to float or int as appropriate."""
        return [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, np.float32, np.float64))
                    else (
                        int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v
                    )
                )
                for k, v in row.items()
            }
            for row in records
        ]

    @staticmethod
    def _scale_column(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
        """Scale a numeric column to [0, 1] using provided min and max values."""
        return (
            (series - min_val) / (max_val - min_val)
            if min_val is not None and max_val not in (None, min_val)
            else pd.Series(0.0, index=series.index, dtype="float32")
        )

    @staticmethod
    def _day_name(d: datetime.date) -> str:
        """Get the weekday name for a given date."""
        return EnrichedData._WEEKDAYS[d.weekday()]

    @staticmethod
    def _fmt(t: datetime.date) -> str:
        """Format a datetime object as 'HH:MM'."""
        return t.strftime("%H:%M")

    @staticmethod
    def _group_date_ranges(dates: List[datetime.date]) -> List[Tuple[str, str]]:
        """Group contiguous date ranges allowing up to 8-day gaps."""
        if not dates:
            return []
        dates = sorted(dates)
        start = prev = dates[0]
        ranges: list[tuple[str, str]] = []
        for d in dates[1:]:
            if (d - prev).days <= 8:
                prev = d
                continue
            ranges.append((start.isoformat(), prev.isoformat()))
            start = prev = d
        ranges.append((start.isoformat(), prev.isoformat()))
        return ranges

    @staticmethod
    def _merge_intervals(
        intervals: list[tuple[datetime.date, datetime.date]],
    ) -> list[tuple[datetime.date, datetime.date]]:
        """Merge overlapping or close intervals within an 8-day threshold."""
        if not intervals:
            return []
        intervals.sort(key=lambda p: p[0])
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            _prev_start, prev_end = merged[-1]
            if (start - prev_end).days <= 8:
                merged[-1][1] = max(prev_end, end)
            else:
                merged.append([start, end])
        return [(s, e) for s, e in merged]  # pylint: disable=unnecessary-comprehension

    @staticmethod
    def _floor_datetimes(
        df: pd.DataFrame,
        interval: str,
        *,
        column: str = "datetime",
    ) -> pd.DataFrame:
        """Floor datetime column values to match the specified interval frequency."""
        freq = IntervalConverter.to_pandas_floor_freq(interval)
        df[column] = pd.to_datetime(df[column]).dt.floor(freq)
        return df

    @staticmethod
    def _get_market_time(symbols_data: Dict[str, dict]) -> Any:
        """Compute daily trading bounds from historical price records."""
        bounds: TradingBounds = defaultdict(dict)
        for symbol, data in symbols_data.items():
            daily_bounds: Dict[datetime.date, Tuple[datetime.time, datetime.time]] = {}
            for hist in data.get("historical_prices", []):
                dt = pd.to_datetime(hist["datetime"], utc=True)
                d, t = dt.date(), dt.time()
                if d not in daily_bounds:
                    daily_bounds[d] = (t, t)
                else:
                    lo, hi = daily_bounds[d]
                    daily_bounds[d] = (min(lo, t), max(hi, t))
            bounds[symbol] = dict(sorted(daily_bounds.items()))
        return bounds

    @staticmethod
    def _build_symbol_summary(by_day: dict) -> list[dict[str, str]]:
        """Construct a summary of recurrent trading hours for each weekday."""
        buckets: defaultdict[
            tuple[str, str], list[tuple[datetime.date, datetime.date]]
        ] = defaultdict(list)
        for day_info in by_day.values():
            for item in day_info["summary"]:
                tf, tt = item["time_from"], item["time_to"]
                df = pd.to_datetime(item["date_from"]).date()
                dt = pd.to_datetime(item["date_to"]).date()
                buckets[(tf, tt)].append((df, dt))
        global_summary: list[dict[str, str]] = []
        for (tf, tt), intervals in buckets.items():
            ref_item = next(
                itm
                for itm in by_day.values()
                for itm in itm["summary"]
                if itm["time_from"] == tf and itm["time_to"] == tt
            )
            for start, end in EnrichedData._merge_intervals(intervals):
                global_summary.append(
                    {
                        "time_from": tf,
                        "time_to": tt,
                        "date_from": start.isoformat(),
                        "date_to": end.isoformat(),
                        "all_day": ref_item["all_day"],
                        "hours": ref_item["hours"],
                    }
                )
        global_summary.sort(key=lambda x: x["date_from"])
        return global_summary

    @staticmethod
    def _consolidate_market_times_summary(
        market_times_by_symbol: MarketRanges,
    ) -> List[Dict[str, Any]]:
        """Aggregate recurring market hours across multiple symbols."""
        buckets: Dict[Tuple[str, str, float], Dict[str, Any]] = defaultdict(
            lambda: {"ranges": [], "symbols": set()}
        )
        for symbol, data in market_times_by_symbol.items():
            for item in data["market_times"]["summary"]:
                if str(item["all_day"]).lower() == "true" or item["all_day"] is True:
                    continue
                key = (
                    item["time_from"],
                    item["time_to"],
                    float(item["hours"]),
                )
                d_from = pd.to_datetime(item["date_from"]).date()
                d_to = pd.to_datetime(item["date_to"]).date()
                buckets[key]["ranges"].append((d_from, d_to))
                buckets[key]["symbols"].add(symbol)
        consolidated: List[Dict[str, Any]] = []
        for (tf, tt, _hrs), info in buckets.items():
            for start, end in EnrichedData._merge_intervals(info["ranges"]):
                consolidated.append(
                    {
                        "time_from": tf,
                        "time_to": tt,
                        "date_from": start.isoformat(),
                        "date_to": end.isoformat(),
                    }
                )
        consolidated.sort(
            key=lambda x: (
                x["date_from"],
                x["time_from"],
                x["time_to"],
            )
        )
        return consolidated

    @staticmethod
    def _group_by_day_time_blocks(
        daily_bounds: DailyBounds, interval_minutes: int
    ) -> Dict[str, Dict[Tuple[datetime.time, datetime.time], List[datetime.date]]]:
        """Group daily trading time bounds by weekday and time intervals."""
        interval_delta = datetime.timedelta(minutes=interval_minutes)
        tmp: Dict[
            str, Dict[Tuple[datetime.time, datetime.time], List[datetime.date]]
        ] = defaultdict(lambda: defaultdict(list))
        for d, (lo, hi) in daily_bounds.items():
            hi_end = (
                datetime.datetime.combine(datetime.date.min, hi) + interval_delta
            ).time()
            tmp[EnrichedData._day_name(d)][(lo, hi_end)].append(d)
        return tmp

    @staticmethod
    def _generate_day_structure(
        grouped: Dict[
            str, Dict[Tuple[datetime.time, datetime.time], List[datetime.date]]
        ],
    ) -> Dict[str, Dict[str, Any]]:
        """Build detailed and summary structures of recurring time blocks per weekday."""

        def _build_blocks(
            combos: Dict[Tuple[datetime.time, datetime.time], List[datetime.date]],
        ) -> List[Dict[str, Any]]:
            blocks = []
            for (lo, hi_end), dates in combos.items():
                secs = (
                    datetime.datetime.combine(datetime.date.min, hi_end)
                    - datetime.datetime.combine(datetime.date.min, lo)
                ).total_seconds()
                if secs <= 0:
                    secs += 86400
                hours = round(secs / 3600, 2)
                blocks.append(
                    {
                        "from": EnrichedData._fmt(lo),
                        "to": EnrichedData._fmt(hi_end),
                        "hours": hours,
                        "all_day": hours == 24,
                        "dates": len(set(dates)),
                    }
                )
            return blocks

        def _mark_recurrent(blocks: List[Dict[str, Any]], total_dates: int) -> None:
            max_hours = max((b["hours"] for b in blocks), default=0.0)
            threshold = 0.2 * total_dates
            for b in blocks:
                b["recurrent"] = (b["hours"] == max_hours) and (b["dates"] >= threshold)

        def _build_summary(
            blocks: List[Dict[str, Any]],
            combos: Dict[Tuple[datetime.time, datetime.time], List[datetime.date]],
        ) -> List[Dict[str, Any]]:
            summary = []
            for b in blocks:
                if not b["recurrent"]:
                    continue
                key = (
                    datetime.datetime.strptime(b["from"], "%H:%M").time(),
                    datetime.datetime.strptime(b["to"], "%H:%M").time(),
                )
                ranges = EnrichedData._group_date_ranges(combos[key])
                for dr_from, dr_to in ranges:
                    summary.append(
                        {
                            "time_from": b["from"],
                            "time_to": b["to"],
                            "date_from": dr_from,
                            "date_to": dr_to,
                            "all_day": b["all_day"],
                            "hours": b["hours"],
                        }
                    )
            summary.sort(key=lambda r: r["date_from"])
            return summary

        def _merge_summary(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            merged = []
            for r in summary:
                if not merged or (r["time_from"], r["time_to"]) != (
                    merged[-1]["time_from"],
                    merged[-1]["time_to"],
                ):
                    merged.append(r)
                else:
                    merged[-1]["date_to"] = r["date_to"]
            return merged

        day_struct = {}
        for day, combos in grouped.items():
            blocks = _build_blocks(combos)
            total_dates = sum(b["dates"] for b in blocks)
            _mark_recurrent(blocks, total_dates)
            summary = _build_summary(blocks, combos)
            merged = _merge_summary(summary)
            day_struct[day] = {"summary": merged, "details": blocks}
        return day_struct

    @staticmethod
    def _assemble_market_times(
        symbol_daily_bounds: Dict[str, DailyBounds],
    ) -> Tuple[List[Dict[str, Any]], MarketRanges]:
        """Compute and structure trading time summaries for each symbol and overall."""
        interval_minutes = IntervalConverter.to_minutes(EnrichedData._RAW_DATA_INTERVAL)
        market_times_by_symbol: MarketRanges = {}
        for symbol, daily_bounds in symbol_daily_bounds.items():
            grouped = EnrichedData._group_by_day_time_blocks(
                daily_bounds, interval_minutes
            )
            day_struct = EnrichedData._generate_day_structure(grouped)
            summary = EnrichedData._build_symbol_summary(day_struct)
            market_times_by_symbol[symbol] = {
                "symbol": symbol,
                "market_times": {"summary": summary, "byDay": day_struct},
            }
        consolidated = EnrichedData._consolidate_market_times_summary(
            market_times_by_symbol
        )
        return consolidated, market_times_by_symbol

    @staticmethod
    def _build_market_times(
        symbol_daily_bounds: Dict[str, DailyBounds],
    ) -> Tuple[List[Dict[str, Any]], MarketRanges]:
        """Generate structured and summarized market times for all symbols."""
        return EnrichedData._assemble_market_times(symbol_daily_bounds)

    @staticmethod
    def _filter_prices_from_global_min(
        symbols: Dict[str, dict],
    ) -> Optional[Dict[str, dict]]:
        """Filter historical prices from the latest shared start date across symbols."""
        min_datetimes = [
            min(pd.to_datetime([row["datetime"] for row in data["historical_prices"]]))
            for data in symbols.values()
            if data.get("historical_prices")
        ]
        if not min_datetimes:
            return None
        global_start = max(min_datetimes)
        for _symbol, data in symbols.items():
            if "historical_prices" not in data:
                continue
            filtered = [
                row
                for row in data["historical_prices"]
                if pd.to_datetime(row["datetime"]) >= global_start
            ]
            data["historical_prices"] = filtered
        return symbols

    @staticmethod
    def _make_consecutive_market_time(
        records: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Adjust adjacent date pairs so that they are consecutive, with no gaps or overlaps.

        If the gap between the `date_to` of record *i* and the `date_from` of
        record *i + 1* is greater than one day, shift both boundaries toward the
        mid-point.
        """
        for idx in range(len(records) - 1):
            d_to = pd.to_datetime(records[idx]["date_to"]).date()
            d_from_next = pd.to_datetime(records[idx + 1]["date_from"]).date()
            diff_days = (d_from_next - d_to).days
            if diff_days <= 1:  # ya son consecutivos
                continue
            midpoint = d_to + datetime.timedelta(days=diff_days // 2)
            records[idx]["date_to"] = midpoint.isoformat()
            records[idx + 1]["date_from"] = (
                midpoint + datetime.timedelta(days=1)
            ).isoformat()
        return records

    @staticmethod
    def _format_symbol_output(key: str, value: Any, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare the final dictionary output for a single symbol with raw and scaled features."""
        df = df[df["is_market_time"].astype(bool)].reset_index(drop=True)
        df = df[df["is_workday"].astype(bool)].reset_index(drop=True)
        base_cols = [
            "datetime",
            "open",
            "low",
            "high",
            "close",
            "adj_close",
            "volume",
            "adx_14d",
            "atr",
            "atr_14d",
            "average_price",
            "bb_width",
            "bollinger_pct_b",
            "intraday_return",
            "macd",
            "obv",
            "overnight_return",
            "price_change",
            "price_derivative",
            "range",
            "relative_volume",
            "return",
            "rsi",
            "smoothed_derivative",
            "stoch_rsi",
            "typical_price",
            "volatility",
            "volume_change",
            "volume_rvol_20d",
            "williams_r",
            "candle_pattern",
            "multi_candle_pattern",
            "is_pre_fed_event",
            "is_fed_event",
            "is_post_fed_event",
            "is_pre_holiday",
            "is_holiday",
            "is_post_holiday",
            "time_of_day",
            "time_of_month",
            "open_close_result",
            "time_of_week",
            "time_of_year",
        ]
        raw_cols = [
            col
            for col in df.columns
            if col in base_cols or col in value.get("features", [])
        ]
        raw_no_datetime = [col for col in raw_cols if col != "datetime"]
        raw_records = EnrichedData._convert_records(
            df[raw_no_datetime].to_dict(orient="records")  # type: ignore
        )
        df["raw"] = raw_records
        formatted = pd.DataFrame(
            {
                "datetime": pd.to_datetime(df["datetime"]).astype(str),
                **{
                    (
                        col.replace("scaled_", "") if col.startswith("scaled_") else col
                    ): df[col]
                    for col in ["raw"]
                    + list(df.columns[df.columns.str.startswith("scaled_")])
                },
            }
        )
        return {
            "symbol": value.get("symbol", key),
            "name": value.get("name", ""),
            "type": value.get("type", ""),
            "sector": value.get("sector", ""),
            "industry": value.get("industry", ""),
            "currency": value.get("currency", ""),
            "exchange": value.get("exchange", ""),
            "historical_prices": EnrichedData._convert_records(
                formatted.to_dict(orient="records")  # type: ignore
            ),
        }

    @staticmethod
    def _compute_features(
        resampled_df: pd.DataFrame,
        ranges: Dict[str, float],
        market_context: MarketContext,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Compute scaled values and enrich raw data with indicators and filters."""
        Logger.debug("     Scaling price and volume features.")
        enriched_df = resampled_df.copy()
        price_cols = ["open", "low", "high", "close", "adj_close"]
        for col in price_cols:
            enriched_df[f"scaled_{col}"] = EnrichedData._scale_column(
                enriched_df[col], ranges["min_price"], ranges["max_price"]
            )
        enriched_df["scaled_volume"] = np.zeros(len(enriched_df), dtype="float32")
        positive_mask = enriched_df["volume"] > 0
        enriched_df.loc[positive_mask, "scaled_volume"] = EnrichedData._scale_column(
            enriched_df.loc[positive_mask, "volume"],
            ranges["min_volume"],
            ranges["max_volume"],
        ).astype("float32")
        Logger.debug("     Adding raw and scaled indicators.")
        enriched_df, market_time = IndicatorBuilder.add_indicators(
            enriched_df, market_context
        )
        enriched_df, _ = IndicatorBuilder.add_indicators(
            enriched_df, market_context, prefix="scaled_"
        )
        Logger.debug("     Cleaning incomplete rows from enriched DataFrame.")
        always_keep = {
            "volume",
            "obv",
            "relative_volume",
            "volume_change",
            "volume_rvol_20d",
        }
        columns_to_check = [c for c in enriched_df.columns if c not in always_keep]
        columns_with_data = [
            c for c in columns_to_check if not enriched_df[c].isna().all()
        ]
        if not columns_with_data:
            Logger.warning(
                "     All feature columns contain NaNs. Returning empty DataFrame."
            )
            return pd.DataFrame(), None
        last_nan_idx = (
            enriched_df[columns_with_data]
            .isna()
            .any(axis=1)
            .pipe(lambda s: s[s].index.max())
        )
        if pd.notna(last_nan_idx):
            enriched_df = enriched_df.loc[last_nan_idx + 1 :]
        enriched_df = enriched_df.dropna(subset=columns_with_data)
        return enriched_df, market_time

    @staticmethod
    def _process_symbol(
        key: str,
        value: Any,
        ranges: Dict[str, float],
        interval: Dict[str, str],
        market_context: MarketContext,
    ) -> Tuple[Dict[str, Any], Optional[pd.Series]]:
        """Run the full enrichment pipeline on a single symbol and return output dict."""
        Logger.debug(f"     Preparing DataFrame for symbol: {key}")
        raw_df = pd.DataFrame(value["historical_prices"])
        interval_raw_data = interval["raw_data"]
        interval_enriched_data = interval["enriched_data"]
        resampled_ratio_df: pd.DataFrame = pd.DataFrame()
        raw_df = EnrichedData._floor_datetimes(raw_df, interval_enriched_data)
        if interval_raw_data != interval_enriched_data:
            ratio = IntervalConverter.get_ratio(
                interval_raw_data, interval_enriched_data
            )
            label = ratio["label"]
            Logger.debug(f"     Resampling data for {key}, interval ratio: {label}")
            resampled_ratio_df = TimeResampler.by_ratio(
                raw_df, interval_raw_data, interval_enriched_data
            )
        else:
            resampled_ratio_df = raw_df.copy()
        df, market_time = EnrichedData._compute_features(
            resampled_ratio_df, ranges, market_context
        )
        Logger.success(f"     Feature computation completed for: {key}")
        return EnrichedData._format_symbol_output(key, value, df), market_time

    @staticmethod
    def exist(filepath: Optional[str] = None) -> bool:
        """Check if enriched market data file exists at the specified or default path."""
        local_filepath: Optional[str] = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        return JsonManager.exists(local_filepath)

    @staticmethod
    def load(filepath: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load enriched market data from a JSON file and populate class attributes."""
        local_filepath: Optional[str] = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        enriched_data = (
            JsonManager.load(local_filepath)
            if JsonManager.exists(local_filepath) is True
            else None
        )
        if enriched_data is None:
            return None
        file_id: Optional[str] = None
        interval: Optional[str] = None
        last_updated: Optional[pd.Timestamp] = None
        ranges: Any = {}
        symbols: list = []
        if enriched_data is not None:
            file_id = enriched_data.get("id")
            interval = enriched_data.get("interval")
            last_updated = enriched_data.get("last_updated")
            ranges = enriched_data.get("ranges")
            symbols = enriched_data.get("symbols")
        symbols_result: Dict[str, dict] = dict(
            RawData.normalize_historical_prices(symbols)
        )
        EnrichedData._id = file_id
        EnrichedData._interval = interval
        EnrichedData._last_updated = last_updated
        EnrichedData._ranges = ranges
        EnrichedData._symbols = symbols_result
        allowed_keys = EnrichedData._REQUIRED_MARKET_ENRICHED_COLUMNS
        for symbol_data in symbols_result.values():
            if "historical_prices" not in symbol_data:
                continue
            symbol_data["historical_prices"] = [
                {k: v for k, v in row.items() if k in allowed_keys}
                for row in symbol_data["historical_prices"]
            ]
        return {
            "id": EnrichedData.get_id(),
            "last_updated": EnrichedData.get_last_updated(),
            "interval": EnrichedData.get_interval(),
            "ranges": EnrichedData.get_ranges(),
            "symbols": symbols_result,
            "filepath": local_filepath,
        }

    @staticmethod
    def save(filepath: Optional[str] = None) -> Dict[str, Any]:
        """Persist the current enriched data to a JSON file."""
        last_updated = (
            pd.Timestamp.now(tz="UTC")
            if EnrichedData.get_last_updated() is None
            else EnrichedData.get_last_updated()
        )
        result = {
            "id": EnrichedData.get_id(),
            "last_updated": last_updated,
            "interval": EnrichedData.get_interval(),
            "ranges": EnrichedData.get_ranges(),
            "market_time": EnrichedData.get_market_time(),
            "symbols": list(EnrichedData._symbols.values()),
        }
        local_filepath = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        JsonManager.save(result, local_filepath)
        return result

    @staticmethod
    def _prepare_generation_context(
        filepath: Optional[str],
    ) -> tuple[
        Optional[str],
        List[datetime.date],
        List[datetime.date],
        List[Dict[str, Any]],
        Dict[str, dict],
    ]:
        local_filepath = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        if local_filepath and JsonManager.exists(local_filepath):
            Logger.warning(f"Removing existing enriched data at: {local_filepath}")
            JsonManager.delete(local_filepath)
        RawData.load()
        symbols_data = RawData.get_symbols()
        EnrichedData.set_id(RawData.get_id())
        EnrichedData.set_interval(Interval.market_enriched_data())
        EnrichedData.set_last_updated(RawData.get_last_updated())
        _, us_holidays, fed_events = CalendarManager.build_market_calendars()
        Logger.debug("US holidays and FED events loaded.")
        market_time_consolidated, _ = EnrichedData._build_market_times(
            EnrichedData._get_market_time(symbols_data)
        )
        return (
            local_filepath,
            us_holidays,
            fed_events,
            market_time_consolidated,
            symbols_data,
        )

    @staticmethod
    def _compute_feature_ranges(symbols_data: Dict[str, dict]) -> Dict[str, float]:
        df_all = pd.concat(
            [pd.DataFrame(s["historical_prices"]) for s in symbols_data.values()],
            ignore_index=True,
        )
        return {
            "min_price": float(df_all["low"].min()),
            "max_price": float(df_all["high"].max()),
            "min_volume": float(df_all["volume"].min()),
            "max_volume": float(df_all["volume"].max()),
        }

    @staticmethod
    def _safe_date_from(x):
        if isinstance(x, dict) and "date_from" in x:
            return x["date_from"]
        return pd.NaT

    @staticmethod
    def _to_iso(obj: Any) -> Any:
        """Convert a *date-like* object datetime to an ISO-8601 string."""
        if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):
            return obj.isoformat()
        return obj

    @staticmethod
    def _enrich_symbols(
        symbols_data: Dict[str, dict],
        ranges: Dict[str, float],
        interval: Dict[str, str],
        market_context: MarketContext,
    ) -> Tuple[Dict[str, dict], Optional[List]]:
        result = {}
        market_time_json: list[dict] = []
        market_time: Optional[pd.Series] = None
        for symbol_key, symbol_value in symbols_data.items():
            Logger.info(f"  * Enriching symbol: {symbol_key}")
            result[symbol_key], local_market_time = EnrichedData._process_symbol(
                symbol_key,
                symbol_value,
                ranges,
                interval,
                market_context,
            )
            market_time = (
                local_market_time
                if local_market_time is not None
                and len(local_market_time) > 0
                and (market_time is None or len(market_time) == 0)
                else market_time
            )
        if market_time is not None:
            market_time = market_time.sort_values(
                key=lambda s: pd.to_datetime(s.apply(EnrichedData._safe_date_from)),
                ignore_index=True,
            )
            valid_market_time = market_time.dropna().loc[
                market_time.apply(lambda x: isinstance(x, Mapping))
            ]
            market_time_json = [
                {
                    k: (EnrichedData._to_iso(v) if v is not pd.NA else None)
                    for k, v in rec.items()
                }
                for rec in valid_market_time.tolist()
            ]
            market_time_json = [
                rec
                for rec in market_time_json
                if rec.get("time_from") != "00:00" and rec.get("time_to") != "00:00"
            ]
            market_time_json = EnrichedData._make_consecutive_market_time(
                market_time_json
            )
        return result, market_time_json

    @staticmethod
    def generate(filepath: Optional[str] = None) -> Dict[str, Any]:
        """Trigger full enrichment generation pipeline from raw data and save output."""
        filepath = filepath or EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        MarketDataSyncManager.synchronize_marketdata_with_drive(filepath)
        Logger.separator()
        RawData.load()
        EnrichedData.load()
        raw_id = RawData.get_id()
        enriched_id = EnrichedData.get_id()
        if raw_id == enriched_id:
            Logger.info("Raw and enriched data IDs match. Skipping enrichment.")
            return EnrichedData.get_symbols()
        (
            _local_filepath,
            us_holidays,
            fed_events,
            market_time_consolidated,
            symbols_data,
        ) = EnrichedData._prepare_generation_context(filepath)
        ranges = EnrichedData._compute_feature_ranges(symbols_data)
        Logger.debug("Price and volume ranges computed.")
        interval = {
            "raw_data": EnrichedData._RAW_DATA_INTERVAL or "",
            "enriched_data": EnrichedData._ENRICHED_DATA_INTERVAL or "",
        }
        Logger.info("Generating enriched market data from raw inputs:")
        context = MarketContext(us_holidays, fed_events, market_time_consolidated)
        EnrichedData._symbols, market_time = EnrichedData._enrich_symbols(
            symbols_data,
            ranges,
            interval,
            context,
        )
        EnrichedData.set_market_time(market_time)
        EnrichedData.set_ranges(
            {
                "price": {"min": ranges["min_price"], "max": ranges["max_price"]},
                "volume": {"min": ranges["min_volume"], "max": ranges["max_volume"]},
            }
        )
        filtered_symbols = EnrichedData._filter_prices_from_global_min(
            EnrichedData.get_symbols()
        )
        if filtered_symbols:
            Logger.debug("Historical prices filtered from global min date.")
            EnrichedData.set_symbols(filtered_symbols)
        for symbol_data in EnrichedData._symbols.values():
            if "historical_prices" not in symbol_data:
                continue
            symbol_data["historical_prices"] = [
                {
                    k: v
                    for k, v in row.items()
                    if k
                    in list(
                        dict.fromkeys(
                            (EnrichedData._REQUIRED_MARKET_ENRICHED_COLUMNS or [])
                            + (["raw"])
                        )
                    )
                }
                for row in symbol_data["historical_prices"]
            ]
        Logger.success("Enriched market data generation completed.")
        return EnrichedData.save(filepath)

    @staticmethod
    def get_indicator_parameters() -> Dict[str, Any]:
        """Return current configuration values for key technical indicators."""
        return {
            "rsi_window": EnrichedData._RSI_WINDOW,
            "macd_fast": EnrichedData._MACD_FAST,
            "macd_slow": EnrichedData._MACD_SLOW,
            "macd_signal": EnrichedData._MACD_SIGNAL,
            "bollinger_window": EnrichedData._BOLLINGER_WINDOW,
            "bollinger_band_method": EnrichedData._BOLLINGER_BAND_METHOD,
            "stoch_rsi_window": EnrichedData._STOCH_RSI_WINDOW,
            "stoch_rsi_min_periods": EnrichedData._STOCH_RSI_MIN_PERIODS,
            "obv_fill_method": EnrichedData._OBV_FILL_METHOD,
            "atr_window": EnrichedData._ATR_WINDOW,
            "williams_r_window": EnrichedData._WILLIAMS_R_WINDOW,
        }
