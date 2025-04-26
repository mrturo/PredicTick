"""
ParameterLoader — Central configuration manager.

This module handles the loading and initialization of both static and dynamic parameters,
including symbol lists, cutoff dates, model configuration values, and paths to key artifacts.
It integrates with SymbolRepository and supports centralized access to runtime configuration
through dictionary-style and method-based access.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

from utils.symbols import SymbolRepository


class ParameterLoader:
    """Centralized configuration manager for all pipeline parameters."""

    _BACKTESTING_BASEPATH = "data/backtesting"
    _CONF_MATRIX_PLOT_FILEPATH = "confusion_matrix_percentage.png"
    _EVALUATION_REPORT_BASEPATH = "data/evaluation/"
    _F1_SCORE_PLOT_FILEPATH = "f1_score_by_class.png"
    _GCP_BASEPATH = "config/gcp/"
    _GCP_CREDENTIALS_FILEPATH = "credentials.json"  # nosec
    _GCP_TOKEN_FILEPATH = "token.json"  # nosec
    _MARKETDATA_DETAILED_FILEPATH = "data/market_data_detailed.csv"
    _MARKETDATA_FILEPATH = "data/market_data.json"
    _MARKETDATA_SUMMARY_FILEPATH = "data/market_data_summary.csv"
    _MODEL_FILEPATH = "data/model/model.pkl"
    _OPTUNA_FILEPATH = "data/model/optuna_study.pkl"
    _SCALER_FILEPATH = "data/model/scaler.pkl"
    _SYMBOLS_FILEPATH = "config/symbols.json"
    _SYMBOLS_INVALID_FILEPATH = "config/symbols_invalid.json"
    _TEST_BACKTESTING_BASEPATH = "tests/data/backtesting"
    _TEST_EVALUATION_REPORT_BASEPATH = "tests/data/evaluation/"
    _TEST_MARKETDATA_DETAILED_FILEPATH = "tests/data/market_data_detailed.csv"
    _TEST_MARKETDATA_FILEPATH = "tests/data/market_data.json"
    _TEST_MARKETDATA_SUMMARY_FILEPATH = "tests/data/market_data_summary.csv"
    _TEST_MODEL_FILEPATH = "tests/data/model/model.pkl"
    _TEST_OPTUNA_FILEPATH = "tests/data/model/optuna_study.pkl"
    _TEST_SCALER_FILEPATH = "tests/data/model/scaler.pkl"

    _ENV_FILEPATH = ".env"

    def __init__(self, last_updated: pd.Timestamp = None):
        self.env_filepath = Path(ParameterLoader._ENV_FILEPATH)
        load_dotenv(dotenv_path=self.env_filepath)
        self.symbol_repo = SymbolRepository(
            ParameterLoader._SYMBOLS_FILEPATH, ParameterLoader._SYMBOLS_INVALID_FILEPATH
        )
        self.last_update = (
            datetime.now(timezone.utc) if last_updated is None else last_updated
        )
        self._parameters: Dict[str, Any] = self._initialize_parameters()

    def _initialize_parameters(self) -> Dict[str, Any]:
        """Initializes the parameters dictionary by merging static JSON and dynamic values."""
        round_last_update = self.last_update.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        one_year_ago_last_update = (
            round_last_update - relativedelta(years=1) - timedelta(days=1)
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
            "block_days": 60,
            "bollinger_band_method": "max-min",
            "bollinger_window": 20,
            "classification_periods_ahead": 1,
            "classification_threshold": 0.005,
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
                "day_of_week",
                "is_before_holiday",
                "is_fed_day",
                "is_holiday",
                "is_post_market",
                "is_pre_market",
                "macd",
                "momentum_3h",
                "obv",
                "return_1h",
                "rsi",
                "stoch_rsi",
                "volatility_3h",
                "volume",
                "williams_r",
            ],
            "fed_event_days": [
                "2024-06-12",
                "2024-07-31",
                "2024-09-18",
                "2024-11-06",
                "2024-12-18",
            ],
            "historical_days_fallback": 729,
            "historical_window_days": 365,
            "holiday_country": "US",
            "interval": "1h",
            "macd_fast": 12,
            "macd_signal": 9,
            "macd_slow": 26,
            "market_calendar_name": "NYSE",
            "market_tz": "America/New_York",
            "n_splits": 5,
            "n_trials": 5,
            "obv_fill_method": 0,
            "prediction_workers": 8,
            "required_market_columns": [
                "adj_close",
                "adx_14",
                "atr_14",
                "average_price",
                "candle",
                "close",
                "datetime",
                "high",
                "intraday_return",
                "low",
                "month_days_current",
                "month_days_total",
                "obv",
                "open",
                "overnight_return",
                "pattern_bearish_engulfing",
                "pattern_bullish_engulfing",
                "pct_day",
                "pct_from_52w_high",
                "pct_month",
                "pct_week",
                "pct_year",
                "price_change",
                "range",
                "records_day_current",
                "records_day_total",
                "relative_volume",
                "typical_price",
                "volatility",
                "volume",
                "volume_rvol_20d",
                "weekdays_current",
                "weekdays_total",
                "year_days_current",
                "year_days_total",
            ],
            "retry_sleep_seconds": 1,
            "rsi_window": 6,
            "rsi_window_backtest": 6,
            "stale_days_threshold": 7,
            "start_hour_today_check": 13,
            "stoch_rsi_min_periods": 1,
            "stoch_rsi_window": 14,
            "target_labels": ["↓ Down", "→ Neutral", "↑ Up"],
            "threshold": 0.002,
            "updater_retries": 5,
            "up_threshold": 0.4,
            "volume_window": 20,
            "weekdays": [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            "williams_r_window": 14,
            "pre_market_end": 13,
            "post_market_start": 22,
            "xgb_params": {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "use_label_encoder": False,
                "tree_method": "hist",
                "verbosity": 0,
            },
        }

        vulnerable_params = {
            "gdrive_folder_id": os.getenv("GDRIVE_FOLDER_ID"),
        }

        path_params = {
            "backtesting_basepath": self._BACKTESTING_BASEPATH,
            "conf_matrix_plot_filepath": self._CONF_MATRIX_PLOT_FILEPATH,
            "evaluation_report_basepath": self._EVALUATION_REPORT_BASEPATH,
            "f1_score_plot_filepath": self._F1_SCORE_PLOT_FILEPATH,
            "gcp_credentials_filepath": f"{self._GCP_BASEPATH}{self._GCP_CREDENTIALS_FILEPATH}",
            "gcp_token_filepath": f"{self._GCP_BASEPATH}{self._GCP_TOKEN_FILEPATH}",
            "marketdata_detailed_filepath": self._MARKETDATA_DETAILED_FILEPATH,
            "marketdata_filepath": self._MARKETDATA_FILEPATH,
            "marketdata_summary_filepath": self._MARKETDATA_SUMMARY_FILEPATH,
            "model_filepath": self._MODEL_FILEPATH,
            "optuna_filepath": self._OPTUNA_FILEPATH,
            "scaler_filepath": self._SCALER_FILEPATH,
            "test_backtesting_basepath": self._TEST_BACKTESTING_BASEPATH,
            "test_evaluation_report_basepath": self._TEST_EVALUATION_REPORT_BASEPATH,
            "test_marketdata_detailed_filepath": self._TEST_MARKETDATA_DETAILED_FILEPATH,
            "test_marketdata_filepath": self._TEST_MARKETDATA_FILEPATH,
            "test_marketdata_summary_filepath": self._TEST_MARKETDATA_SUMMARY_FILEPATH,
            "test_model_filepath": self._TEST_MODEL_FILEPATH,
            "test_optuna_filepath": self._TEST_OPTUNA_FILEPATH,
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
