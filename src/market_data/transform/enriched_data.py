import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from market_data.ingest.raw_data import RawData
from market_data.transform.technical_indicators import TechnicalIndicators
from market_data.transform.time_resampler import TimeResampler
from market_data.utils.interval import Interval, IntervalConverter
from utils.calendar_manager import CalendarManager
from utils.json_manager import JsonManager
from utils.logger import Logger
from utils.parameters import ParameterLoader

TradingBounds = Dict[str, Dict[datetime.date, Tuple[datetime.time, datetime.time]]]
DailyBounds = Dict[datetime.date, Tuple[datetime.time, datetime.time]]

DayBlocks = List[Dict[str, Any]]
DayObject = Dict[str, DayBlocks]
WeekObject = Dict[str, DayObject]
MarketRanges = Dict[str, Dict[str, Union[str, WeekObject]]]


class EnrichedData:

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
    _VOLUME_WINDOW = _PARAMS.get("volume_window")
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
        if filepath is None or len(filepath.strip()) == 0:
            return default.strip() if default is not None else None
        return filepath.strip()

    @staticmethod
    def get_id() -> Optional[str]:
        return EnrichedData._id

    @staticmethod
    def set_id(file_id: Optional[str]) -> None:
        EnrichedData._id = file_id

    @staticmethod
    def get_interval() -> Optional[str]:
        return EnrichedData._interval

    @staticmethod
    def set_interval(interval: Optional[str]) -> None:
        EnrichedData._interval = interval

    @staticmethod
    def get_last_updated() -> Optional[pd.Timestamp]:
        return EnrichedData._last_updated

    @staticmethod
    def set_last_updated(last_updated: Optional[pd.Timestamp]) -> None:
        EnrichedData._last_updated = last_updated

    @staticmethod
    def get_ranges() -> Any:
        return EnrichedData._ranges

    @staticmethod
    def set_ranges(ranges: Any) -> None:
        EnrichedData._ranges = ranges

    @staticmethod
    def get_symbol(symbol: str) -> Optional[dict]:
        return EnrichedData._symbols.get(symbol)

    @staticmethod
    def get_symbols() -> Dict[str, dict]:
        return EnrichedData._symbols

    @staticmethod
    def set_symbols(symbols: Dict[str, dict]) -> None:
        EnrichedData._symbols = symbols

    @staticmethod
    def _convert_records(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
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
        return (
            (series - min_val) / (max_val - min_val)
            if min_val is not None and max_val not in (None, min_val)
            else pd.Series(0.0, index=series.index, dtype="float32")
        )

    @staticmethod
    def _day_name(d: datetime.date) -> str:
        return EnrichedData._WEEKDAYS[d.weekday()]

    @staticmethod
    def _fmt(t: datetime.date) -> str:
        return t.strftime("%H:%M")

    @staticmethod
    def _group_date_ranges(dates: List[datetime.date]) -> List[Tuple[str, str]]:
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
        if not intervals:
            return []
        intervals.sort(key=lambda p: p[0])
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if (start - prev_end).days <= 8:
                merged[-1][1] = max(prev_end, end)
            else:
                merged.append([start, end])
        return [(s, e) for s, e in merged]

    @staticmethod
    def _floor_datetimes(
        df: pd.DataFrame,
        interval: str,
        *,
        column: str = "datetime",
    ) -> pd.DataFrame:
        freq = IntervalConverter.to_pandas_floor_freq(interval)
        df[column] = pd.to_datetime(df[column]).dt.floor(freq)
        return df

    @staticmethod
    def _get_market_time(symbols_data: Dict[str, dict]) -> Any:
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
            merged_intervals = EnrichedData._merge_intervals(info["ranges"])
            for start, end in merged_intervals:
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
    def _build_market_times(
        symbol_daily_bounds: Dict[str, DailyBounds],
    ) -> Tuple[List[Dict[str, Any]], MarketRanges]:
        interval_minutes = IntervalConverter.to_minutes(EnrichedData._RAW_DATA_INTERVAL)
        interval_delta = datetime.timedelta(minutes=interval_minutes)
        market_times_by_symbol: MarketRanges = {}
        for symbol, daily_bounds in symbol_daily_bounds.items():
            tmp: Dict[
                str, Dict[Tuple[datetime.time, datetime.time], List[datetime.date]]
            ] = defaultdict(lambda: defaultdict(list))
            for d, (lo, hi) in daily_bounds.items():
                hi_end = (
                    datetime.datetime.combine(datetime.date.min, hi) + interval_delta
                ).time()
                tmp[EnrichedData._day_name(d)][(lo, hi_end)].append(d)
            day_struct: Dict[str, Dict[str, Any]] = {}
            for day, combos in tmp.items():
                blocks: DayBlocks = []
                for (lo, hi_end), dates in combos.items():
                    secs = (
                        datetime.datetime.combine(datetime.date.min, hi_end)
                        - datetime.datetime.combine(datetime.date.min, lo)
                    ).total_seconds()
                    if secs <= 0:
                        secs += 86_400
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
                total_dates = sum(b["dates"] for b in blocks)
                max_hours = max(b["hours"] for b in blocks) if blocks else 0.0
                threshold = 0.2 * total_dates
                for b in blocks:
                    b["recurrent"] = (b["hours"] == max_hours) and (
                        b["dates"] >= threshold
                    )
                summary: list[dict[str, str]] = []
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
                merged: list[dict[str, str]] = []
                for r in summary:
                    if not merged or (r["time_from"], r["time_to"]) != (
                        merged[-1]["time_from"],
                        merged[-1]["time_to"],
                    ):
                        merged.append(r)
                    else:
                        merged[-1]["date_to"] = r["date_to"]
                summary = merged
                day_struct[day] = {
                    "summary": summary,
                    "details": blocks,
                }
            symbol_summary = EnrichedData._build_symbol_summary(day_struct)
            market_times_by_symbol[symbol] = {
                "symbol": symbol,
                "market_times": {"summary": symbol_summary, "byDay": day_struct},
            }
        return (
            EnrichedData._consolidate_market_times_summary(market_times_by_symbol),
            market_times_by_symbol,
        )

    @staticmethod
    def _filter_prices_from_global_min(
        symbols: Dict[str, dict],
    ) -> Optional[Dict[str, dict]]:
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
    def _format_symbol_output(key: str, value: Any, df: pd.DataFrame) -> Dict[str, Any]:
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
            "internal_multi_candle_pattern",
            "is_pre_fed_event",
            "is_fed_event",
            "is_post_fed_event",
            "is_pre_holiday",
            "is_holiday",
            "is_post_holiday",
            "time_of_day",
            "time_of_month",
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
    def _add_indicators(
        raw_df: pd.DataFrame,
        enriched_df: pd.DataFrame,
        us_holidays: List[datetime.date],
        fed_events: List[datetime.date],
        market_time: Any,
        prefix: str = "",
    ) -> pd.DataFrame:
        prefix = prefix.strip()
        is_raw: bool = len(prefix) == 0

        def prefixed(col: str) -> str:
            return f"{prefix}{col}" if prefix else col

        enriched_df[prefixed("adx_14d")] = TechnicalIndicators.compute_adx_14d(
            enriched_df["datetime"],
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
        )
        enriched_df[prefixed("atr")] = TechnicalIndicators.compute_atr(
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
            EnrichedData._ATR_WINDOW,
        )
        enriched_df[prefixed("atr_14d")] = TechnicalIndicators.compute_atr_14d(
            enriched_df["datetime"],
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
        )
        enriched_df[prefixed("average_price")] = (
            TechnicalIndicators.compute_average_price(
                enriched_df[prefixed("high")], enriched_df[prefixed("low")]
            )
        )
        enriched_df[prefixed("bollinger_pct_b")] = (
            TechnicalIndicators.compute_bollinger_pct_b(enriched_df[prefixed("close")])
        )
        if EnrichedData._BOLLINGER_BAND_METHOD == "max-min":
            enriched_df[prefixed("bb_width")] = TechnicalIndicators.compute_bb_width(
                enriched_df[prefixed("close")], EnrichedData._BOLLINGER_WINDOW
            )
        enriched_df[prefixed("intraday_return")] = (
            TechnicalIndicators.compute_intraday_return(
                enriched_df[prefixed("close")], enriched_df[prefixed("open")]
            )
        )
        enriched_df[prefixed("macd")] = TechnicalIndicators.compute_macd(
            enriched_df[prefixed("close")],
            EnrichedData._MACD_FAST,
            EnrichedData._MACD_SLOW,
            EnrichedData._MACD_SIGNAL,
        )["histogram"]
        if "volume" in enriched_df.columns:
            enriched_df[prefixed("obv")] = TechnicalIndicators.compute_obv(
                enriched_df[prefixed("close")], enriched_df[prefixed("volume")]
            )
        enriched_df[prefixed("overnight_return")] = (
            TechnicalIndicators.compute_overnight_return(enriched_df[prefixed("open")])
        )
        enriched_df[prefixed("price_change")] = (
            TechnicalIndicators.compute_price_change(
                enriched_df[prefixed("close")], enriched_df[prefixed("open")]
            )
        )
        enriched_df[prefixed("price_derivative")] = (
            TechnicalIndicators.compute_price_derivative(enriched_df[prefixed("close")])
        )
        enriched_df[prefixed("range")] = TechnicalIndicators.compute_range(
            enriched_df[prefixed("high")], enriched_df[prefixed("low")]
        )
        if "volume" in enriched_df.columns:
            enriched_df[prefixed("relative_volume")] = (
                TechnicalIndicators.compute_relative_volume(
                    enriched_df[prefixed("volume")], EnrichedData._VOLUME_WINDOW
                )
            )
        enriched_df[prefixed("return")] = TechnicalIndicators.compute_return(
            enriched_df[prefixed("close")]
        )
        enriched_df[prefixed("rsi")] = TechnicalIndicators.compute_rsi(
            enriched_df[prefixed("close")], EnrichedData._RSI_WINDOW
        )
        enriched_df[prefixed("smoothed_derivative")] = (
            TechnicalIndicators.compute_smoothed_derivative(
                enriched_df[prefixed("close")]
            )
        )
        enriched_df[prefixed("stoch_rsi")] = TechnicalIndicators.compute_stoch_rsi(
            enriched_df[prefixed("close")], EnrichedData._STOCH_RSI_WINDOW
        )
        enriched_df[prefixed("typical_price")] = (
            TechnicalIndicators.compute_typical_price(
                enriched_df[prefixed("high")],
                enriched_df[prefixed("low")],
                enriched_df[prefixed("close")],
            )
        )
        enriched_df[prefixed("volatility")] = TechnicalIndicators.compute_volatility(
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("open")],
        )
        if "volume" in enriched_df.columns:
            enriched_df[prefixed("volume_change")] = (
                TechnicalIndicators.compute_volume_change(
                    enriched_df[prefixed("volume")]
                )
            )
        if "volume" in enriched_df.columns:
            enriched_df[prefixed("volume_rvol_20d")] = (
                TechnicalIndicators.compute_volume_rvol_20d(
                    enriched_df["datetime"], enriched_df[prefixed("volume")]
                )
            )
        enriched_df[prefixed("williams_r")] = TechnicalIndicators.compute_williams_r(
            enriched_df[prefixed("high")],
            enriched_df[prefixed("low")],
            enriched_df[prefixed("close")],
            EnrichedData._WILLIAMS_R_WINDOW,
        )
        enriched_df[prefixed("candle_pattern")] = (
            TechnicalIndicators.compute_candle_pattern(
                enriched_df[prefixed("open")],
                enriched_df[prefixed("high")],
                enriched_df[prefixed("low")],
                enriched_df[prefixed("close")],
                is_raw,
            )
        )
        enriched_df[prefixed("multi_candle_pattern")] = (
            TechnicalIndicators.compute_multi_candle_pattern(
                enriched_df[prefixed("open")],
                enriched_df[prefixed("high")],
                enriched_df[prefixed("low")],
                enriched_df[prefixed("close")],
                is_raw,
            )
        )
        enriched_df[prefixed("internal_multi_candle_pattern")] = (
            TechnicalIndicators.compute_internal_multi_candle_pattern(
                raw_df,
                enriched_df,
                is_raw,
            )
        )

        enriched_df[prefixed("is_market_day")] = False
        enriched_df[prefixed("is_pre_market_time")] = False
        enriched_df[prefixed("is_market_time")] = False
        enriched_df[prefixed("is_post_market_time")] = False

        features_fed_event = TechnicalIndicators.compute_temporal_event_feature(
            df=enriched_df,
            event_dates=set(fed_events),
            is_raw=is_raw,
        )
        enriched_df[prefixed("is_pre_fed_event")] = features_fed_event["is_pre"]
        enriched_df[prefixed("is_fed_event")] = features_fed_event["is"]
        enriched_df[prefixed("is_post_fed_event")] = features_fed_event["is_post"]

        features_holiday = TechnicalIndicators.compute_temporal_event_feature(
            df=enriched_df,
            event_dates=set(us_holidays),
            is_raw=is_raw,
        )
        enriched_df[prefixed("is_pre_holiday")] = features_holiday["is_pre"]
        enriched_df[prefixed("is_holiday")] = features_holiday["is"]
        enriched_df[prefixed("is_post_holiday")] = features_holiday["is_post"]

        enriched_df[prefixed("is_weekday")] = TechnicalIndicators.compute_weekday(
            enriched_df["datetime"], is_raw
        )
        enriched_df[prefixed("is_weekend")] = TechnicalIndicators.compute_weekend(
            enriched_df["datetime"], is_raw
        )
        enriched_df[prefixed("is_workday")] = TechnicalIndicators.compute_workday(
            enriched_df[prefixed("is_weekend")],
            enriched_df[prefixed("is_holiday")],
            is_raw,
        )
        features_time_fractions = TechnicalIndicators.compute_time_fractions(
            df=enriched_df,
            is_raw=is_raw,
        )
        enriched_df[prefixed("time_of_day")] = features_time_fractions["time_of_day"]
        enriched_df[prefixed("time_of_week")] = features_time_fractions["time_of_week"]
        enriched_df[prefixed("time_of_month")] = features_time_fractions[
            "time_of_month"
        ]
        enriched_df[prefixed("time_of_year")] = features_time_fractions["time_of_year"]

        features_market_time = TechnicalIndicators.compute_market_time(
            df=enriched_df,
            market_time=market_time,
            interval=EnrichedData._ENRICHED_DATA_INTERVAL,
            is_raw=is_raw,
            is_workday=enriched_df[prefixed("is_workday")],
        )
        enriched_df[prefixed("is_market_day")] = features_market_time["is_market_day"]
        enriched_df[prefixed("is_pre_market_time")] = features_market_time[
            "is_pre_market_time"
        ]
        enriched_df[prefixed("is_market_time")] = features_market_time["is_market_time"]
        enriched_df[prefixed("is_post_market_time")] = features_market_time[
            "is_post_market_time"
        ]
        return enriched_df

    @staticmethod
    def _compute_features(
        raw_df: pd.DataFrame,
        resampled_df: pd.DataFrame,
        ranges: Dict[str, float],
        us_holidays: List[datetime.date],
        fed_events: List[datetime.date],
        market_time: Any,
    ) -> pd.DataFrame:
        Logger.debug("     Scaling price and volume features.")
        enriched_df = resampled_df.copy()
        price_cols = ["open", "low", "high", "close", "adj_close"]
        for col in price_cols:
            enriched_df[f"scaled_{col}"] = EnrichedData._scale_column(
                enriched_df[col], ranges["min_price"], ranges["max_price"]
            )
        enriched_df["scaled_volume"] = EnrichedData._scale_column(
            enriched_df["volume"], ranges["min_volume"], ranges["max_volume"]
        )
        Logger.debug("     Adding raw and scaled indicators.")
        enriched_df = EnrichedData._add_indicators(
            raw_df, enriched_df, us_holidays, fed_events, market_time
        )
        enriched_df = EnrichedData._add_indicators(
            raw_df, enriched_df, us_holidays, fed_events, market_time, prefix="scaled_"
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
            return enriched_df.iloc[0:0]
        last_nan_idx = (
            enriched_df[columns_with_data]
            .isna()
            .any(axis=1)
            .pipe(lambda s: s[s].index.max())
        )
        if pd.notna(last_nan_idx):
            enriched_df = enriched_df.loc[last_nan_idx + 1 :]
        enriched_df = enriched_df.dropna(subset=columns_with_data)
        return enriched_df

    @staticmethod
    def _process_symbol(
        key: str,
        value: Any,
        ranges: Dict[str, float],
        interval: Dict[str, str],
        us_holidays: List[datetime.date],
        fed_events: List[datetime.date],
        market_time: Any,
    ) -> Dict[str, Any]:
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
        df = EnrichedData._compute_features(
            raw_df, resampled_ratio_df, ranges, us_holidays, fed_events, market_time
        )
        Logger.success(f"     Feature computation completed for: {key}")
        return EnrichedData._format_symbol_output(key, value, df)

    @staticmethod
    def load(filepath: Optional[str] = None) -> Dict[str, Any]:
        local_filepath: Optional[str] = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        enriched_data = JsonManager.load(local_filepath)
        file_id: Optional[str] = None
        interval: Optional[str] = None
        last_updated: Optional[pd.Timestamp] = None
        ranges: Any = {}
        symbols: list = []
        if enriched_data is not None:
            file_id = enriched_data["id"]
            interval = enriched_data["interval"]
            last_updated = enriched_data["last_updated"]
            ranges = enriched_data["ranges"]
            symbols = enriched_data["symbols"]
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
        }

    @staticmethod
    def save(filepath: Optional[str] = None) -> Dict[str, Any]:
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
            "symbols": list(EnrichedData._symbols.values()),
        }
        local_filepath = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        JsonManager.save(result, local_filepath)
        return result

    @staticmethod
    def generate(filepath: Optional[str] = None) -> Dict[str, Any]:
        local_filepath = EnrichedData._get_filepath(
            filepath, EnrichedData._ENRICHED_MARKETDATA_FILEPATH
        )
        if local_filepath and JsonManager.exists(local_filepath):
            Logger.warning(f"Removing existing enriched data at: {local_filepath}")
            JsonManager.delete(local_filepath)
        interval = Interval.market_enriched_data()
        RawData.load()
        symbols_data = RawData.get_symbols()
        df_all = pd.concat(
            [pd.DataFrame(s["historical_prices"]) for s in symbols_data.values()],
            ignore_index=True,
        )
        min_price = float(df_all["low"].min())
        max_price = float(df_all["high"].max())
        min_volume = float(df_all["volume"].min())
        max_volume = float(df_all["volume"].max())
        Logger.debug("Price and volume ranges computed.")
        EnrichedData._symbols = {}
        EnrichedData.set_id(RawData.get_id())
        EnrichedData.set_interval(interval)
        EnrichedData.set_last_updated(RawData.get_last_update())
        _, us_holidays, fed_events = CalendarManager.build_market_calendars()
        Logger.debug("US holidays and FED events loaded.")
        symbol_daily_bounds = EnrichedData._get_market_time(symbols_data)
        market_time_consolidated, _market_time_by_symbol = (
            EnrichedData._build_market_times(symbol_daily_bounds)
        )
        Logger.info("Generating enriched market data from raw inputs:")
        for symbol_key, symbol_value in symbols_data.items():
            Logger.info(f"  * Enriching symbol: {symbol_key}")
            EnrichedData._symbols[symbol_key] = EnrichedData._process_symbol(
                symbol_key,
                symbol_value,
                {
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_volume": min_volume,
                    "max_volume": max_volume,
                },
                {
                    "raw_data": EnrichedData._RAW_DATA_INTERVAL,
                    "enriched_data": EnrichedData._ENRICHED_DATA_INTERVAL,
                },
                us_holidays,
                fed_events,
                market_time_consolidated,
            )
        EnrichedData.set_ranges(
            {
                "price": {"min": min_price, "max": max_price},
                "volume": {"min": min_volume, "max": max_volume},
            }
        )
        filtered_symbols = EnrichedData._filter_prices_from_global_min(
            EnrichedData.get_symbols()
        )
        if filtered_symbols:
            Logger.debug("Historical prices filtered from global min date.")
            EnrichedData.set_symbols(filtered_symbols)
        allowed_keys: list[str] = list(
            dict.fromkeys(
                (EnrichedData._REQUIRED_MARKET_ENRICHED_COLUMNS or []) + (["raw"])
            )
        )
        for symbol_data in EnrichedData._symbols.values():
            if "historical_prices" not in symbol_data:
                continue
            symbol_data["historical_prices"] = [
                {k: v for k, v in row.items() if k in allowed_keys}
                for row in symbol_data["historical_prices"]
            ]
        Logger.success("Enriched market data generation completed.")
        return EnrichedData.save(filepath)

    @staticmethod
    def get_indicator_parameters() -> Dict[str, Any]:
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
