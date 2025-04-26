"""Market data gateway for managing symbol metadata and historical prices."""

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd  # type: ignore

from market_data.schedule_builder import ScheduleBuilder
from utils.calendar_manager import CalendarManager
from utils.json_manager import JsonManager
from utils.logger import Logger
from utils.parameters import ParameterLoader


class Gateway:
    """Centralized access point for managing symbol data and market schedules."""

    _PARAMS = ParameterLoader()
    _STALE_DAYS_THRESHOLD = _PARAMS.get("stale_days_threshold")
    _SYMBOL_REPO = _PARAMS.symbol_repo
    _WEEKDAYS = _PARAMS.get("weekdays")
    _MARKETDATA_FILEPATH = _PARAMS.get("marketdata_filepath")

    _symbols: Dict[str, dict] = {}
    _last_updated: Optional[pd.Timestamp] = None
    _latest_price_date: Optional[pd.Timestamp] = None
    _stale_symbols: List[str] = []
    _global_schedule: Dict[str, dict] = {}
    _market_time: List[Dict[str, str]] = []

    @staticmethod
    def _get_filepath(filepath: Optional[str], default=None) -> Optional[str]:
        if filepath is None or len(filepath.strip()) == 0:
            return default
        return filepath

    @staticmethod
    def get_symbols() -> Dict[str, dict]:
        """Return all loaded symbols."""
        return Gateway._symbols

    @staticmethod
    def set_symbols(symbols: Dict[str, dict]) -> None:
        """Set the dictionary of symbols."""
        Gateway._symbols = symbols

    @staticmethod
    def get_market_time() -> List[Dict[str, str]]:
        """Get market time symbols."""
        return Gateway._market_time

    @staticmethod
    def set_market_time(market_time: List[Dict[str, str]]) -> None:
        """Set market time symbols."""
        Gateway._market_time = market_time

    @staticmethod
    def get_symbol(symbol: str):
        """Retrieve metadata for a specific symbol."""
        return Gateway._symbols.get(symbol)

    @staticmethod
    def set_symbol(symbol: str, historical_prices: List[dict], metadata: dict):
        """Set symbol metadata and historical prices."""
        Gateway._symbols[symbol] = {
            "symbol": symbol,
            "name": metadata.get("name"),
            "type": metadata.get("type"),
            "sector": metadata.get("sector"),
            "industry": metadata.get("industry"),
            "currency": metadata.get("currency"),
            "exchange": metadata.get("exchange"),
            "schedule": metadata.get("schedule"),
            "historical_prices": historical_prices,
        }

    @staticmethod
    def get_last_update() -> Optional[pd.Timestamp]:
        """Return the last update timestamp."""
        return Gateway._last_updated

    @staticmethod
    def get_latest_price_date() -> Optional[pd.Timestamp]:
        """Return the latest price timestamp among all symbols."""
        return Gateway._latest_price_date

    @staticmethod
    def set_latest_price_date(latest_price_date: Optional[pd.Timestamp]) -> None:
        """Manually set the latest price date."""
        Gateway._latest_price_date = latest_price_date

    @staticmethod
    def get_stale_symbols() -> List[str]:
        """Return the list of stale symbols."""
        return Gateway._stale_symbols

    @staticmethod
    def set_stale_symbols(stale_symbols: List[str]) -> None:
        """Manually set stale symbols."""
        Gateway._stale_symbols = stale_symbols

    @staticmethod
    def get_global_schedule() -> Dict[str, dict]:
        """Return the globalschedule of symbols."""
        return Gateway._global_schedule

    @staticmethod
    def invalid_symbols() -> set:
        """Return the set of currently invalid symbols."""
        return set(Gateway._SYMBOL_REPO.get_invalid_symbols())

    @staticmethod
    def latest_price_date() -> Optional[pd.Timestamp]:
        """Alias to get the latest price date."""
        return Gateway._latest_price_date

    @staticmethod
    def load(filepath: Optional[str] = None) -> Dict:
        """Load market data from disk and initialize internal structures."""
        local_filepath: Optional[str] = Gateway._get_filepath(
            filepath, Gateway._MARKETDATA_FILEPATH
        )
        raw_data = JsonManager.load(local_filepath)
        raw_last_updated: Optional[pd.Timestamp] = None
        symbols: list = []

        if raw_data is not None:
            raw_last_updated = raw_data["last_updated"]
            symbols = raw_data["symbols"]

        Gateway._last_updated = (
            pd.to_datetime(raw_last_updated)
            if raw_last_updated is not None and isinstance(raw_last_updated, str)
            else pd.Timestamp.now(tz="UTC")
        )

        enriched = {}
        for entry in symbols:
            symbol = entry.get("symbol")
            df = pd.DataFrame(entry["historical_prices"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            entry["historical_prices"] = df.to_dict(orient="records")
            enriched[symbol] = entry

        Gateway._symbols = {
            sym: data
            for sym, data in enriched.items()
            if sym not in Gateway._SYMBOL_REPO.get_invalid_symbols()
        }
        Gateway._update_latest_price_date()
        Gateway._update_stale_symbols()
        Gateway._update_global_schedule()
        Gateway._extract_market_time_ranges()
        Gateway._annotate_market_times()
        return {
            "last_updated": Gateway._last_updated,
            "schedule": Gateway._global_schedule,
            "market_time": Gateway._market_time,
            "symbols": Gateway._symbols,
        }

    @staticmethod
    def save(filepath: Optional[str] = None) -> dict[str, Any]:
        """Persist current symbol data to disk."""
        result = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "schedule": Gateway._global_schedule,
            "market_time": Gateway._market_time,
            "symbols": list(Gateway._symbols.values()),
        }
        local_filepath = Gateway._get_filepath(filepath, Gateway._MARKETDATA_FILEPATH)
        JsonManager.save(result, local_filepath)
        return result

    @staticmethod
    def _update_latest_price_date() -> None:
        """Update the latest price date across all symbols."""
        max_date = None
        for entry in Gateway._symbols.values():
            datetimes = [
                pd.to_datetime(p["datetime"], utc=True, errors="coerce")
                for p in entry.get("historical_prices", [])
            ]
            entry_max = max(filter(pd.notnull, datetimes), default=None)
            if entry_max:
                max_date = max(max_date or entry_max, entry_max)
        Gateway._latest_price_date = max_date

    @staticmethod
    def _update_stale_symbols() -> None:
        """Identify and mark symbols with outdated price data."""
        latest_date = Gateway._latest_price_date or pd.Timestamp.now(tz="UTC")
        current_invalids = set(Gateway._SYMBOL_REPO.get_invalid_symbols())

        stale, updated_invalids = Gateway._detect_stale_symbols(
            latest_date, current_invalids
        )

        Gateway._stale_symbols = stale
        Gateway._SYMBOL_REPO.set_invalid_symbols(updated_invalids)

    @staticmethod
    def _detect_stale_symbols(
        latest_date: pd.Timestamp, invalid_symbols: set
    ) -> tuple[list, set]:
        """Detect stale symbols and return updated lists (for internal testing)."""
        stale = []
        for symbol, entry in Gateway._symbols.items():
            datetimes = [
                pd.to_datetime(p["datetime"], utc=True, errors="coerce")
                for p in entry.get("historical_prices", [])
            ]
            symbol_latest = max(filter(pd.notnull, datetimes), default=None)
            if (
                not symbol_latest
                or (latest_date - symbol_latest).days > Gateway._STALE_DAYS_THRESHOLD
            ):
                stale.append(symbol)
                invalid_symbols.add(symbol)
        return stale, invalid_symbols

    @staticmethod
    def _update_global_schedule() -> None:
        """Construct a combined trading schedule from all symbols."""

        def collect_variants(symbols: List[dict], weekdays: List[str]) -> dict:
            variants = {day.lower(): {} for day in weekdays}
            for entry in symbols:
                symbol = entry.get("symbol")
                schedule = entry.get("schedule", {})
                for day in weekdays:
                    sch = schedule.get(day.lower(), {})
                    key = (sch.get("min_open"), sch.get("max_close"))
                    if key[0] and key[1]:
                        variants[day.lower()].setdefault(key, []).append(symbol)
            return variants

        def build_day_schedule(variants: dict) -> dict:
            opens, closes, schedules = [], [], []
            for (open_str, close_str), symbols in variants.items():
                open_dt = datetime.strptime(open_str, "%H:%M:%S")
                close_dt = datetime.strptime(close_str, "%H:%M:%S")
                if close_dt <= open_dt:
                    close_dt += timedelta(days=1)
                opens.append(open_dt)
                closes.append(close_dt)
                all_day = (
                    open_str == close_str and open_str == "00:00:00"
                    if open_str and close_str
                    else None
                )
                open_hours = ScheduleBuilder.calc_open_hours(
                    open_str, close_str, all_day
                )
                schedules.append(
                    {
                        "all_day": all_day,
                        "open": open_str,
                        "close": close_str,
                        "symbols": sorted(symbols),
                        "open_hours": open_hours,
                    }
                )

            min_open = min(opens).strftime("%H:%M:%S") if opens else None
            max_close = max(closes).time().strftime("%H:%M:%S") if closes else None

            return ScheduleBuilder.build_schedule_sumary(
                min_open, max_close, schedules or [{"all_day": None}]
            )

        variants_by_day = collect_variants(
            list(Gateway._symbols.values()), Gateway._WEEKDAYS
        )
        Gateway._global_schedule = {
            day.lower(): build_day_schedule(variants_by_day[day.lower()])
            for day in Gateway._WEEKDAYS
        }

    @staticmethod
    def _merge_intervals(intervals: List[Dict[str, str]]) -> List[Dict[str, str]]:
        unique = {(i["from"], i["to"], i["open"], i["close"]): i for i in intervals}
        sorted_intervals = sorted(unique.values(), key=lambda x: x["from"])

        grouped = []
        for interval in sorted_intervals:
            if not grouped:
                grouped.append(interval)
            else:
                last = grouped[-1]
                if (
                    interval["open"] == last["open"]
                    and interval["close"] == last["close"]
                    and datetime.strptime(interval["from"], "%Y-%m-%d")
                    <= datetime.strptime(last["to"], "%Y-%m-%d") + timedelta(days=1)
                ):
                    last["to"] = max(last["to"], interval["to"])
                else:
                    grouped.append(interval)
        return grouped

    @staticmethod
    def _extract_market_time_ranges() -> None:
        """Extracts market_time ranges from symbol schedules by weekday."""

        def _collect_all_intervals() -> List[Dict[str, str]]:
            intervals = []

            def process_entry(entry: dict, date_group: list) -> None:
                if entry.get("market_time") and date_group:
                    intervals.append(
                        {
                            "from": date_group[0],
                            "to": date_group[-1],
                            "open": entry["open"],
                            "close": entry["close"],
                        }
                    )

            for symbol in Gateway._symbols.values():
                schedule = symbol.get("schedule", {})
                for day_sched in schedule.values():
                    for entry in day_sched.get("schedules", []):
                        for date_group in entry.get("dates", []):
                            process_entry(entry, date_group)

            return intervals

        def _smooth_edges(intervals: List[Dict[str, str]]) -> None:
            for i in range(len(intervals) - 1):
                current = intervals[i]
                next_ = intervals[i + 1]
                if "to" in current and "from" in next_:
                    start = datetime.strptime(current["to"], "%Y-%m-%d")
                    end = datetime.strptime(next_["from"], "%Y-%m-%d")
                    mid = start + (end - start) // 2
                    current["to"] = mid.strftime("%Y-%m-%d")
                    next_["from"] = (mid + timedelta(days=1)).strftime("%Y-%m-%d")

        def _cleanup_edges(intervals: List[Dict[str, str]]) -> None:
            if intervals:
                intervals[0].pop("from", None)
                intervals[-1].pop("to", None)

        all_intervals = _collect_all_intervals()
        merged = Gateway._merge_intervals(all_intervals)
        _smooth_edges(merged)
        _cleanup_edges(merged)
        Gateway._market_time = merged

    @staticmethod
    def get_range_for_day(
        local_dt: datetime,
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Returns the market open and close time range for a given local date.

        Iterates through the `_market_time` intervals and identifies which one applies
        to the specified `local_dt`, considering optional 'from' and 'to' date bounds.
        If a matching interval is found, it computes the corresponding UTC `datetime`
        objects for market open and close.

        If the close time is less than or equal to the open time, the method assumes the
        market session extends to the next day (overnight session).

        Args:
            local_dt (datetime): The local date and time for which to find the market range.

        Returns:
            Optional[Tuple[datetime, datetime]]: A tuple `(open_dt, close_dt)` in UTC if a
            valid interval is found, otherwise `None`.
        """
        for rng in Gateway._market_time:
            from_dt = (
                datetime.strptime(rng["from"], "%Y-%m-%d") if "from" in rng else None
            )
            to_dt = datetime.strptime(rng["to"], "%Y-%m-%d") if "to" in rng else None
            if (from_dt is None or local_dt.date() >= from_dt.date()) and (
                to_dt is None or local_dt.date() <= to_dt.date()
            ):
                open_t = datetime.strptime(rng["open"], "%H:%M:%S").time()
                close_t = datetime.strptime(rng["close"], "%H:%M:%S").time()
                open_dt = datetime.combine(local_dt.date(), open_t).replace(
                    tzinfo=timezone.utc
                )
                close_dt = datetime.combine(local_dt.date(), close_t).replace(
                    tzinfo=timezone.utc
                )
                if close_dt <= open_dt:
                    close_dt += timedelta(days=1)
                return open_dt, close_dt
        return None

    @staticmethod
    def set_temporal_event_feature(
        record: dict,
        dt_date: date,
        event_dates: set[date],
        window: int = 5,
        prefix: Literal["holiday", "fed_event"] = "holiday",
    ) -> dict:
        """
        Annotates a price record with decaying proximity features for temporal events like holidays.

        or Fed events.

        Parameters:
            record (dict): Historical price record.
            dt_date (date): Date of the record.
            event_dates (set[date]): Set of relevant event dates.
            window (int): Number of days before/after the event to consider.
            prefix (str): Prefix for generated feature keys (e.g., 'holiday' or 'fed_event').

        Returns:
            dict: Updated record with keys:
                  - f"is_{prefix}"
                  - f"is_pre_{prefix}"
                  - f"is_post_{prefix}"
        """
        is_event: bool = dt_date in event_dates
        pre_decay: float = 0.0
        post_decay: float = 0.0

        if is_event is False:
            for i in range(1, window + 1):
                pre_day = dt_date + timedelta(days=i)
                post_day = dt_date - timedelta(days=i)
                if pre_day in event_dates:
                    pre_decay = max(pre_decay, (window - i + 1) / window)
                if post_day in event_dates:
                    post_decay = max(post_decay, (window - i + 1) / window)

        record[f"is_{prefix}"] = 1 if is_event else 0
        record[f"is_pre_{prefix}"] = pre_decay
        record[f"is_post_{prefix}"] = post_decay
        return record

    @staticmethod
    def _annotate_market_times() -> None:
        """
        Annotates each historical price record with pre_market_time, market_time, post_market_time.

        Returns:
            None
        """
        _, us_holidays, fed_event = CalendarManager.build_market_calendars()

        total_records = 0
        for symbol_key, symbol in Gateway._symbols.items():
            if "historical_prices" not in symbol:
                Logger.warning(f"Symbol {symbol_key} missing historical_prices.")
                continue

            for record in symbol["historical_prices"]:
                total_records += 1

                dt = pd.to_datetime(record["datetime"])
                is_weekday: bool = dt.weekday() < 5

                record["date"] = dt.strftime("%Y-%m-%d")
                record["hour"] = (
                    dt.hour
                    + dt.minute / 60
                    + dt.second / 3600
                    + dt.microsecond / 3_600_000_000
                )

                record["is_pre_market_time"] = 0
                record["is_market_time"] = 0
                record["is_post_market_time"] = 0
                record["is_market_day"] = 0

                record = Gateway.set_temporal_event_feature(
                    record=record,
                    dt_date=dt.date(),
                    event_dates=set(us_holidays),
                    window=5,
                    prefix="holiday",
                )
                record = Gateway.set_temporal_event_feature(
                    record=record,
                    dt_date=dt.date(),
                    event_dates=set(fed_event),
                    window=5,
                    prefix="fed_event",
                )

                if not is_weekday or record["is_holiday"] == 1:
                    continue

                rng = Gateway.get_range_for_day(dt)
                if rng is None:
                    Logger.debug(
                        f"No market range for datetime {dt} in symbol {symbol_key}."
                    )
                    continue

                open_dt, close_dt = rng
                if dt < open_dt:
                    record["is_pre_market_time"] = 1
                elif open_dt <= dt < close_dt:
                    record["is_market_time"] = 1
                else:
                    record["is_post_market_time"] = 1
                record["is_market_day"] = (
                    1
                    if (
                        (record["is_pre_market_time"] == 1)
                        or (record["is_market_time"] == 1)
                        or (record["is_post_market_time"] == 1)
                    )
                    else 0
                )
