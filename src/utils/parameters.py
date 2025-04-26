"""
ParameterLoader — Central configuration manager.

This module handles the loading and initialization of both static and dynamic parameters,
including symbol lists, cutoff dates, model configuration values, and paths to key artifacts.
It integrates with SymbolRepository and supports centralized access to runtime configuration
through dictionary-style and method-based access.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd
from dateutil.relativedelta import relativedelta

from utils.logger import Logger
from utils.symbols import SymbolRepository


class ParameterLoader:
    """Centralized configuration manager for all pipeline parameters."""

    def __init__(self, last_updated: pd.Timestamp = None):
        self.symbol_repo = SymbolRepository(
            mercado_pago_path=os.path.join("config/symbols/symbols_mercado_pago.json"),
            training_path=os.path.join("config/symbols/symbols_training.json"),
            xtb_path=os.path.join("config/symbols/symbols_xtb.json"),
            correlative_path=os.path.join("config/symbols/symbols_correlative.json"),
        )
        self.last_update = (
            datetime.now(timezone.utc) if last_updated is None else last_updated
        )
        self._parameters: Dict[str, Any] = self._initialize_parameters()

    def _initialize_parameters(self):
        """Initializes the parameters dictionary by merging static JSON and dynamic values."""
        config_path = os.path.join("config", "config_parameters.json")
        if not os.path.exists(config_path):
            Logger.error(f"Configuration file not found at: {config_path}")
            raise FileNotFoundError(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            static_params = json.load(f)

        round_last_update = self.last_update.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        one_year_ago_last_update = (
            round_last_update - relativedelta(years=1) - timedelta(days=1)
        )

        dynamic_params = {
            "cutoff_date": one_year_ago_last_update.strftime("%Y-%m-%d"),
            "training_symbols": self.symbol_repo.load_training_symbols(),
            "mercado_pago_symbols": self.symbol_repo.load_mercado_pago_symbols(),
            "xtb_symbols": self.symbol_repo.load_xtb_symbols(),
            "correlative_symbols": self.symbol_repo.load_correlative_symbols(),
        }

        return {**static_params, **dynamic_params}

    def get(self, key: str):
        """Return parameter value if exists, else None."""
        return self._parameters.get(key)

    def __getitem__(self, key: str):
        """Allow dict-style access to parameters."""
        return self._parameters[key]
