"""Provides an interface to fetch historical price data and metadata using Yahoo Finance."""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Union, get_args, get_origin


@dataclass
class TickerMetadata:  # pylint: disable=too-many-instance-attributes
    """Generic representation of ticker metadata."""

    address1: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    country: Optional[str] = None
    phone: Optional[str] = None
    fax: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    industry_key: Optional[str] = None
    industry_disp: Optional[str] = None
    sector: Optional[str] = None
    sector_key: Optional[str] = None
    sector_disp: Optional[str] = None
    long_business_summary: Optional[str] = None
    full_time_employees: Optional[int] = None
    company_officers: Optional[list] = None
    audit_risk: Optional[int] = None
    board_risk: Optional[int] = None
    compensation_risk: Optional[int] = None
    share_holder_rights_risk: Optional[int] = None
    overall_risk: Optional[int] = None
    governance_epoch_date: Optional[int] = None
    compensation_as_of_epoch_date: Optional[int] = None
    ir_website: Optional[str] = None
    executive_team: Optional[list] = None
    max_age: Optional[int] = None
    price_hint: Optional[int] = None
    previous_close: Optional[float] = None
    open: Optional[float] = None
    day_low: Optional[float] = None
    day_high: Optional[float] = None
    regular_market_previous_close: Optional[float] = None
    regular_market_open: Optional[float] = None
    regular_market_day_low: Optional[float] = None
    regular_market_day_high: Optional[float] = None
    ex_dividend_date: Optional[int] = None
    payout_ratio: Optional[float] = None
    five_year_avg_dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    volume: Optional[int] = None
    regular_market_volume: Optional[int] = None
    average_volume: Optional[int] = None
    average_volume10days: Optional[int] = None
    average_daily_volume10_day: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    market_cap: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    price_to_sales_trailing12_months: Optional[float] = None
    fifty_day_average: Optional[float] = None
    two_hundred_day_average: Optional[float] = None
    trailing_annual_dividend_rate: Optional[float] = None
    trailing_annual_dividend_yield: Optional[float] = None
    currency: Optional[str] = None
    tradeable: Optional[bool] = None
    enterprise_value: Optional[float] = None
    profit_margins: Optional[float] = None
    float_shares: Optional[int] = None
    shares_outstanding: Optional[int] = None
    shares_short: Optional[int] = None
    shares_short_prior_month: Optional[int] = None
    shares_short_previous_month_date: Optional[int] = None
    date_short_interest: Optional[int] = None
    shares_percent_shares_out: Optional[float] = None
    held_percent_insiders: Optional[float] = None
    held_percent_institutions: Optional[float] = None
    short_ratio: Optional[float] = None
    short_percent_of_float: Optional[float] = None
    implied_shares_outstanding: Optional[int] = None
    book_value: Optional[float] = None
    price_to_book: Optional[float] = None
    last_fiscal_year_end: Optional[int] = None
    next_fiscal_year_end: Optional[int] = None
    most_recent_quarter: Optional[int] = None
    earnings_quarterly_growth: Optional[float] = None
    net_income_to_common: Optional[float] = None
    trailing_eps: Optional[float] = None
    forward_eps: Optional[float] = None
    last_split_factor: Optional[str] = None
    last_split_date: Optional[int] = None
    enterprise_to_revenue: Optional[float] = None
    enterprise_to_ebitda: Optional[float] = None
    fifty_two_week_change: Optional[float] = None
    sand_p52_week_change: Optional[float] = None
    last_dividend_value: Optional[float] = None
    last_dividend_date: Optional[int] = None
    quote_type: Optional[str] = None
    current_price: Optional[float] = None
    target_high_price: Optional[float] = None
    target_low_price: Optional[float] = None
    target_mean_price: Optional[float] = None
    target_median_price: Optional[float] = None
    recommendation_mean: Optional[float] = None
    recommendation_key: Optional[str] = None
    number_of_analyst_opinions: Optional[int] = None
    total_cash: Optional[float] = None
    total_cash_per_share: Optional[float] = None
    ebitda: Optional[float] = None
    total_debt: Optional[float] = None
    quick_ratio: Optional[float] = None
    current_ratio: Optional[float] = None
    total_revenue: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_per_share: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_equity: Optional[float] = None
    gross_profits: Optional[float] = None
    free_cashflow: Optional[float] = None
    operating_cashflow: Optional[float] = None
    earnings_growth: Optional[float] = None
    revenue_growth: Optional[float] = None
    gross_margins: Optional[float] = None
    ebitda_margins: Optional[float] = None
    operating_margins: Optional[float] = None
    financial_currency: Optional[str] = None
    symbol: Optional[str] = None
    language: Optional[str] = None
    region: Optional[str] = None
    type_disp: Optional[str] = None
    quote_source_name: Optional[str] = None
    triggerable: Optional[bool] = None
    custom_price_alert_confidence: Optional[str] = None
    market_state: Optional[str] = None
    corporate_actions: Optional[dict] = None
    post_market_time: Optional[int] = None
    regular_market_time: Optional[int] = None
    exchange: Optional[str] = None
    message_board_id: Optional[str] = None
    exchange_timezone_name: Optional[str] = None
    exchange_timezone_short_name: Optional[str] = None
    gmt_off_set_milliseconds: Optional[int] = None
    market: Optional[str] = None
    esg_populated: Optional[bool] = None
    post_market_change_percent: Optional[float] = None
    post_market_price: Optional[float] = None
    post_market_change: Optional[float] = None
    regular_market_change: Optional[float] = None
    regular_market_day_range: Optional[str] = None
    full_exchange_name: Optional[str] = None
    average_daily_volume3_month: Optional[int] = None
    fifty_two_week_low_change: Optional[float] = None
    fifty_two_week_low_change_percent: Optional[float] = None
    fifty_two_week_range: Optional[str] = None
    fifty_two_week_high_change: Optional[float] = None
    fifty_two_week_high_change_percent: Optional[float] = None
    fifty_two_week_change_percent: Optional[float] = None
    regular_market_change_percent: Optional[float] = None
    regular_market_price: Optional[float] = None
    short_name: Optional[str] = None
    long_name: Optional[str] = None
    crypto_tradeable: Optional[bool] = None
    has_pre_post_old: Optional[bool] = None
    first_trade_date_milliseconds: Optional[int] = None
    earnings_timestamp: Optional[int] = None
    earnings_timestamp_start: Optional[int] = None
    earnings_timestamp_end: Optional[int] = None
    earnings_call_timestamp_start: Optional[int] = None
    earnings_call_timestamp_end: Optional[int] = None
    is_earnings_date_estimate: Optional[bool] = None
    eps_trailing_twelve_months: Optional[float] = None
    eps_forward: Optional[float] = None
    eps_current_year: Optional[float] = None
    price_eps_current_year: Optional[float] = None
    fifty_day_average_change: Optional[float] = None
    fifty_day_average_change_percent: Optional[float] = None
    two_hundred_day_average_change: Optional[float] = None
    two_hundred_day_average_change_percent: Optional[float] = None
    source_interval: Optional[int] = None
    exchange_data_delayed_by: Optional[int] = None
    average_analyst_rating: Optional[str] = None
    display_name: Optional[str] = None
    trailing_peg_ratio: Optional[float] = None

    @staticmethod
    def _snake_to_camel(s: str) -> str:
        """Converts snake_case to camelCase."""
        parts = s.split("_")
        return parts[0] + "".join(word.capitalize() for word in parts[1:])

    @staticmethod
    def _safe_eval(val: str, expected_type: type) -> Any:
        """Safely evaluates a string and checks if it matches the expected type."""
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, expected_type):
                return parsed
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        return None

    @staticmethod
    def _parse_value(val: Any, typ: Any) -> Any:
        """Parses and converts the value to the appropriate type."""
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return None
        typ = TickerMetadata._extract_real_type(typ)
        try:
            return TickerMetadata._convert_value(val, typ)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    @staticmethod
    def _extract_real_type(typ: Any) -> Any:
        """Handles Optional types, extracting the actual type."""
        origin = get_origin(typ)
        args = get_args(typ)
        if origin is Union and type(None) in args:
            return next(a for a in args if a is not type(None))
        return typ

    @staticmethod
    def _convert_value(val: Any, typ: Any) -> Any:
        """Converts the value to the specified type."""
        if typ is int:
            return int(val)
        if typ is float:
            return float(val)
        if typ is bool:
            return TickerMetadata._parse_bool(val)
        if typ is list:
            return TickerMetadata._parse_list(val)
        if typ is dict:
            return TickerMetadata._parse_dict(val)
        return val

    @staticmethod
    def _parse_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    @staticmethod
    def _parse_list(val: Any) -> list:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return TickerMetadata._safe_eval(val, list)
        raise ValueError("Invalid list format")

    @staticmethod
    def _parse_dict(val: Any) -> dict:
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            return TickerMetadata._safe_eval(val, dict)
        raise ValueError("Invalid dict format")

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> TickerMetadata:
        """Builds TickerMetadata from a dictionary using camelCase → snake_case conversion."""
        filtered_data = {}
        for f in fields(TickerMetadata):
            camel_name = TickerMetadata._snake_to_camel(f.name)
            val = data.get(camel_name, None)
            filtered_data[f.name] = TickerMetadata._parse_value(val, f.type)
        return TickerMetadata(**filtered_data)
