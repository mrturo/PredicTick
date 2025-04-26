"""Central configuration manager.

This module handles the loading and initialization of both static and dynamic parameters,
including symbol lists, cutoff dates, model configuration values, and paths to key artifacts.
It integrates with SymbolRepository and supports centralized access to runtime configuration
through dictionary-style and method-based access.
"""

import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd  # type: ignore
from dateutil.relativedelta import relativedelta  # type: ignore
from dotenv import load_dotenv

from src.utils.config.path_utils import PathUtils
from src.utils.config.symbols import SymbolRepository
from src.utils.exchange.stock_exchange import StockExchange


class ParameterLoader:
    """Centralized configuration manager for all pipeline parameters."""

    _BACKTESTING_BASEPATH = PathUtils.build("data/backtesting")
    _CONF_MATRIX_PLOT_FILEPATH = "confusion_matrix_percentage.png"
    _ENRICHED_MARKETDATA_FILEPATH = PathUtils.build("data/market_enriched_data.json")
    _EVALUATION_REPORT_BASEPATH = PathUtils.build("data/evaluation/")
    _EVENT_DATES_FILEPATH = PathUtils.build("config/event_dates.json")
    _F1_SCORE_PLOT_FILEPATH = "f1_score_by_class.png"
    _GCP_BASEPATH = PathUtils.build("config/gcp/")
    _GCP_CREDENTIALS_FILEPATH = "credentials.json"  # nosec
    _GCP_TOKEN_FILEPATH = "token.json"  # nosec
    _MARKETDATA_DETAILED_FILEPATH = PathUtils.build("data/market_data_detailed.csv")
    _MARKETDATA_SUMMARY_FILEPATH = PathUtils.build("data/market_data_summary.csv")
    _MODEL_FILEPATH = PathUtils.build("data/model/model.pkl")
    _OPTUNA_FILEPATH = PathUtils.build("data/model/optuna_study.pkl")
    _RAW_MARKETDATA_FILEPATH = PathUtils.build("data/market_raw_data.json")
    _SCALER_FILEPATH = PathUtils.build("data/model/scaler.pkl")
    _SYMBOLS_FILEPATH = PathUtils.build("config/symbols.json")
    _SYMBOLS_INVALID_FILEPATH = PathUtils.build("config/symbols_invalid.json")
    _TEST_BACKTESTING_BASEPATH = "tests/data/backtesting"
    _TEST_ENRICHED_MARKETDATA_FILEPATH = PathUtils.build(
        "data/market_enriched_data.json"
    )
    _TEST_EVALUATION_REPORT_BASEPATH = "tests/data/evaluation/"
    _TEST_MARKETDATA_DETAILED_FILEPATH = "tests/data/market_data_detailed.csv"
    _TEST_MARKETDATA_SUMMARY_FILEPATH = "tests/data/market_data_summary.csv"
    _TEST_MODEL_FILEPATH = "tests/data/model/model.pkl"
    _TEST_OPTUNA_FILEPATH = "tests/data/model/optuna_study.pkl"
    _TEST_RAW_MARKETDATA_FILEPATH = "tests/data/market_raw_data.json"
    _TEST_SCALER_FILEPATH = "tests/data/model/scaler.pkl"

    _ENV_FILEPATH = ".env"

    def __init__(self, last_updated: Optional[pd.Timestamp] = None):
        self.env_filepath = Path(ParameterLoader._ENV_FILEPATH)
        load_dotenv(dotenv_path=self.env_filepath)
        self.symbol_repo = SymbolRepository(
            ParameterLoader._SYMBOLS_FILEPATH, ParameterLoader._SYMBOLS_INVALID_FILEPATH
        )
        self.last_update = (
            pd.Timestamp.utcnow() if last_updated is None else last_updated
        )
        self._parameters: Dict[str, Any] = self._initialize_parameters()
        exchange_default: Any = self.get("exchange_default")
        exchanges: Any = self.get("exchanges")
        weekdays: Any = self.get("weekdays")
        stock_exchange_default: Optional[StockExchange] = StockExchange.from_parameter(
            exchange_default, exchanges, weekdays
        ).to_utc()
        if stock_exchange_default is None:
            raise ValueError("Stock exchange not found")
        self._stock_exchange: StockExchange = stock_exchange_default

    def _initialize_parameters(self) -> Dict[str, Any]:
        """Initializes the parameters dictionary by merging static JSON and dynamic values."""
        round_last_update = self.last_update.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        one_year_ago_last_update: pd.Timestamp = pd.Timestamp(
            round_last_update - relativedelta(years=1) + timedelta(days=1)
        )
        dynamic_params = {
            "all_symbols": self.symbol_repo.get_all_symbols(),
            "correlative_symbols": self.symbol_repo.get_correlative_symbols(),
            "cutoff_date": one_year_ago_last_update.strftime("%Y-%m-%d"),
            "training_symbols": self.symbol_repo.get_training_symbols(),
        }
        prediction_groups_names = self.symbol_repo.get_all_prediction_group_name()
        for group_name in prediction_groups_names:
            dynamic_params[group_name] = self.symbol_repo.get_prediction_group_symbols(
                group_name
            )
        constant_params = {
            "atr_window": 14,
            "availability_days_window": 7,
            "block_days": 240,
            "bollinger_band_method": "max-min",
            "bollinger_window": 20,
            "candle_metrics": {
                "doji_body_threshold": 0.1,
                "doji_max_lower_shadow": 0.1,
                "doji_max_upper_shadow": 0.1,
                "hammer_body_threshold": 0.3,
                "hammer_shadow_ratio": 2.0,
                "lower_shadow_max_ratio": 0.3,
                "shooting_star_body_threshold": 0.3,
                "shooting_star_shadow_ratio": 2.0,
                "upper_shadow_max_ratio": 0.3,
            },
            "candle_multiple_score": {
                "bearish_engulfing": 0.0,
                "evening_star": 0.05,
                "dark_cloud_cover": 0.1,
                "three_black_crows": 0.15,
                "tweezer_top": 0.2,
                "tweezer_bottom": 0.8,
                "piercing_line": 0.9,
                "morning_star": 0.95,
                "three_white_soldiers": 0.98,
                "bullish_engulfing": 1.0,
            },
            "candle_simple_score": {
                "shooting_star_bearish": 0.0,
                "gravestone_doji": 0.05,
                "hammer_bearish": 0.1,
                "marubozu_bearish": 0.15,
                "long_upper_shadow": 0.18,
                "hanging_man": 0.2,
                "shooting_star_bullish": 0.25,
                "bearish": 0.3,
                "spinning_top_bearish": 0.4,
                "doji": 0.5,
                "spinning_top_bullish": 0.6,
                "bullish": 0.7,
                "inverted_hammer": 0.75,
                "long_lower_shadow": 0.8,
                "dragonfly_doji": 0.85,
                "marubozu_bullish": 0.9,
                "hammer_bullish": 1.0,
            },
            "confusion_matrix_figsize": [6, 5],
            "cutoff_minutes": 75,
            "dashboard_footer_caption": "Created with ❤️ using Streamlit and Optuna.",
            "dashboard_layout": "wide",
            "dashboard_page_title": "Optuna Dashboard",
            "date_format": "%Y-%m-%d",
            "default_currency": "USD",
            "download_retries": 3,
            "down_threshold": 0.4,
            "f1_score_figsize": [6, 4],
            "f1_score_plot_title": "F1-Score per Class",
            "features": [
                "atr",
                "bb_width",
                "is_fed_event",
                "is_holiday",
                "is_post_fed_event",
                "is_post_holiday",
                "is_pre_fed_event",
                "is_pre_holiday",
                "macd",
                "obv",
                "price_derivative",
                "rsi",
                "smoothed_derivative",
                "stoch_rsi",
                "time_of_day",
                "time_of_month",
                "time_of_week",
                "time_of_year",
                "open_close_result",
                "volume",
                "williams_r",
            ],
            "force_data_enrichment": False,
            "historical_days_fallback": {
                "1m": 7,
                "2m": 59,
                "5m": 59,
                "15m": 59,
                "30m": 59,
                "60m": 59,
                "90m": 59,
                "1h": 729,
                "1d": 3000,
                "5d": 3000,
                "1wk": 3000,
                "1mo": 3000,
                "3mo": 3000,
            },
            "historical_window_days": 365,
            "holiday_country": "US",
            "ingester_retries": 5,
            "interval": {"market_enriched_data": "1d", "market_raw_data": "1h"},
            "macd_fast": 12,
            "macd_signal": 9,
            "macd_slow": 26,
            "market_tz": "America/New_York",
            "n_splits": 5,
            "n_trials": 5,
            "obv_fill_method": 0,
            "prediction_workers": 8,
            "required_market_enriched_columns": [
                "williams_r",
                "volume_rvol_20d",
                "volume_change",
                "volatility",
                "typical_price",
                "time_of_year",
                "open_close_result",
                "time_of_week",
                "time_of_month",
                "time_of_day",
                "stoch_rsi",
                "smoothed_derivative",
                "rsi",
                "return",
                "relative_volume",
                "range",
                "price_derivative",
                "price_change",
                "overnight_return",
                "obv",
                "multi_candle_pattern",
                "macd",
                "is_pre_holiday",
                "is_pre_fed_event",
                "is_post_holiday",
                "is_post_fed_event",
                "is_holiday",
                "is_fed_event",
                "intraday_return",
                "candle_pattern",
                "bollinger_pct_b",
                "bb_width",
                "average_price",
                "atr_14d",
                "atr",
                "adx_14d",
            ],
            "required_market_raw_columns": [
                "datetime",
                "adj_close",
                "close",
                "high",
                "low",
                "open",
                "volume",
            ],
            "retry_sleep_seconds": 1,
            "rsi_window": 6,
            "rsi_window_backtest": 6,
            "stale_days_threshold": 7,
            "stoch_rsi_min_periods": 1,
            "stoch_rsi_window": 14,
            "exchanges": [
                {
                    "code": "NGM",
                    "country": "US",
                    "timezone": "America/New_York",
                    "sessions_days": {
                        "monday": True,
                        "tuesday": True,
                        "wednesday": True,
                        "thursday": True,
                        "friday": True,
                        "saturday": False,
                        "sunday": False,
                    },
                    "sessions_hours": {
                        "pre_market": {"open": "04:00", "close": "09:30"},
                        "regular": {"open": "09:30", "close": "16:00"},
                        "post_market": {"open": "16:00", "close": "20:00"},
                    },
                },
                {
                    "code": "NMS",
                    "country": "US",
                    "timezone": "America/New_York",
                    "sessions_days": {
                        "monday": True,
                        "tuesday": True,
                        "wednesday": True,
                        "thursday": True,
                        "friday": True,
                        "saturday": False,
                        "sunday": False,
                    },
                    "sessions_hours": {
                        "pre_market": {"open": "04:00", "close": "09:30"},
                        "regular": {"open": "09:30", "close": "16:00"},
                        "post_market": {"open": "16:00", "close": "20:00"},
                    },
                },
                {
                    "code": "PCX",
                    "country": "US",
                    "timezone": "America/New_York",
                    "sessions_days": {
                        "monday": True,
                        "tuesday": True,
                        "wednesday": True,
                        "thursday": True,
                        "friday": True,
                        "saturday": False,
                        "sunday": False,
                    },
                    "sessions_hours": {
                        "pre_market": {"open": "04:00", "close": "09:30"},
                        "regular": {"open": "09:30", "close": "16:00"},
                        "post_market": {"open": "16:00", "close": "20:00"},
                    },
                },
                {
                    "code": "CCC",
                    "country": "US",
                    "timezone": "UTC",
                    "sessions_days": {
                        "monday": True,
                        "tuesday": True,
                        "wednesday": True,
                        "thursday": True,
                        "friday": True,
                        "saturday": True,
                        "sunday": True,
                    },
                    "sessions_hours": {
                        "regular": {"open": "00:00", "close": "00:00"},
                    },
                },
                {
                    "code": "CCY",
                    "country": "US",
                    "timezone": "America/New_York",
                    "sessions_days": {
                        "monday": True,
                        "tuesday": True,
                        "wednesday": True,
                        "thursday": True,
                        "friday": True,
                        "saturday": False,
                        "sunday": False,
                    },
                    "sessions_hours": {
                        "regular": {"open": "00:00", "close": "00:00"},
                    },
                },
            ],
            "exchange_code_map": {
                "NMS": "NYSE",
                "PCX": "NYSE",
                "NGM": "NYSE",
                "CCC": "24/7",
                "CCY": "24/5",
            },
            "exchange_default": "NMS",
            "up_threshold": 0.4,
            "volume_window": 20,
            "weekdays": [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ],
            "williams_r_window": 14,
            "xgb_params": {
                "eval_metric": "mlogloss",
                "num_class": 3,
                "objective": "multi:softprob",
                "tree_method": "hist",
                "use_label_encoder": False,
                "verbosity": 0,
            },
        }
        vulnerable_params = {
            "gdrive_folder_id": os.getenv("GDRIVE_FOLDER_ID"),
        }
        path_params = {
            "backtesting_basepath": self._BACKTESTING_BASEPATH,
            "conf_matrix_plot_filepath": self._CONF_MATRIX_PLOT_FILEPATH,
            "enriched_marketdata_filepath": self._ENRICHED_MARKETDATA_FILEPATH,
            "evaluation_report_basepath": self._EVALUATION_REPORT_BASEPATH,
            "event_dates_filepath": self._EVENT_DATES_FILEPATH,
            "f1_score_plot_filepath": self._F1_SCORE_PLOT_FILEPATH,
            "gcp_credentials_filepath": f"{self._GCP_BASEPATH}{self._GCP_CREDENTIALS_FILEPATH}",
            "gcp_token_filepath": f"{self._GCP_BASEPATH}{self._GCP_TOKEN_FILEPATH}",
            "marketdata_detailed_filepath": self._MARKETDATA_DETAILED_FILEPATH,
            "marketdata_summary_filepath": self._MARKETDATA_SUMMARY_FILEPATH,
            "model_filepath": self._MODEL_FILEPATH,
            "optuna_filepath": self._OPTUNA_FILEPATH,
            "raw_marketdata_filepath": self._RAW_MARKETDATA_FILEPATH,
            "scaler_filepath": self._SCALER_FILEPATH,
            "test_backtesting_basepath": self._TEST_BACKTESTING_BASEPATH,
            "test_enriched_marketdata_filepath": self._TEST_ENRICHED_MARKETDATA_FILEPATH,
            "test_evaluation_report_basepath": self._TEST_EVALUATION_REPORT_BASEPATH,
            "test_marketdata_detailed_filepath": self._TEST_MARKETDATA_DETAILED_FILEPATH,
            "test_marketdata_summary_filepath": self._TEST_MARKETDATA_SUMMARY_FILEPATH,
            "test_model_filepath": self._TEST_MODEL_FILEPATH,
            "test_optuna_filepath": self._TEST_OPTUNA_FILEPATH,
            "test_raw_marketdata_filepath": self._TEST_RAW_MARKETDATA_FILEPATH,
            "test_scaler_filepath": self._TEST_SCALER_FILEPATH,
        }
        return {**vulnerable_params, **dynamic_params, **constant_params, **path_params}

    def get_all(self) -> Any:
        """Return all parameter."""
        return self._parameters

    def get(self, key: str, default: Any = None) -> Any:
        """Return parameter value if exists, else None."""
        try:
            return self._parameters[key]
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to parameters."""
        return self._parameters[key]

    def exchange_default(self) -> StockExchange:
        """Return default stock exchange."""
        return self._stock_exchange

    def exchange(self, exchange_id: str) -> StockExchange:
        """Return a specific stock exchange."""
        exchanges: Any = self.get("exchanges")
        weekdays: Any = self.get("weekdays")
        stock_exchange: Optional[StockExchange] = StockExchange.from_parameter(
            exchange_id, exchanges, weekdays
        ).to_utc()
        if stock_exchange is None:
            raise ValueError("Stock exchange not found")
        return stock_exchange
