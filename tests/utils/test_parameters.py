"""Unit tests for the ParameterLoader configuration manager."""

from unittest.mock import MagicMock, patch

import pytest

from utils.parameters import ParameterLoader


@pytest.fixture
def mock_env(monkeypatch):
    """Fixture para establecer variables de entorno simuladas."""
    env_vars = {
        "ATR_WINDOW": "14",
        "BACKTESTING_OUTPUT_BASEPATH": "backtests/",
        "BLOCK_DAYS": "5",
        "BOLLINGER_BAND_METHOD": "std",
        "BOLLINGER_WINDOW": "20",
        "CUTOFF_MINUTES": "60",
        "DASHBOARD_FOOTER_CAPTION": "Footer",
        "DASHBOARD_LAYOUT": "wide",
        "DASHBOARD_PAGE_TITLE": "Market Dashboard",
        "DOWNLOAD_RETRIES": "3",
        "UPDATER_RETRIES": "2",
        "DOWN_THRESHOLD": "0.02",
        "EVALUATION_REPORT_BASEPATH": "reports/",
        "HISTORICAL_DAYS_FALLBACK": "30",
        "HISTORICAL_WINDOW_DAYS": "365",
        "INTERVAL": "1h",
        "MACD_FAST": "12",
        "MACD_SIGNAL": "9",
        "MACD_SLOW": "26",
        "MARKET_TZ": "UTC",
        "N_SPLITS": "5",
        "N_TRIALS": "50",
        "OBV_FILL_METHOD": "0",
        "PREDICTION_WORKERS": "4",
        "RETRY_SLEEP_SECONDS": "10",
        "RSI_WINDOW": "14",
        "RSI_WINDOW_BACKTEST": "14",
        "START_HOUR_TODAY_CHECK": "9",
        "STOCH_RSI_MIN_PERIODS": "14",
        "STOCH_RSI_WINDOW": "14",
        "THRESHOLD": "0.01",
        "UP_THRESHOLD": "0.02",
        "VOLUME_WINDOW": "20",
        "WILLIAMS_R_WINDOW": "14",
        "MARKET_CALENDAR_NAME": "NYSE",
        "HOLIDAY_COUNTRY": "US",
        "DATE_FORMAT": "%Y-%m-%d",
        "CLASSIFICATION_PERIODS_AHEAD": "1",
        "CLASSIFICATION_THRESHOLD": "0.01",
        "F1_SCORE_PLOT_TITLE": "F1 Score by Class",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


# pylint: disable=redefined-outer-name, unused-argument
@patch("src.utils.parameters.SymbolRepository")
def test_parameter_loader_get_method(mock_repo_class):
    """Prueba el método get de ParameterLoader."""
    mock_repo_class.return_value = MagicMock()
    loader = ParameterLoader()
    result = loader.get("non_existent_key")
    if result is not None:
        raise AssertionError("Expected None for missing key")

    result_with_default = loader.get("non_existent_key", default="default_value")
    if result_with_default != "default_value":
        raise AssertionError("Expected default_value for missing key with default")


# pylint: disable=redefined-outer-name, unused-argument
@patch("src.utils.parameters.SymbolRepository")
def test_parameter_loader_getitem_method(mock_repo_class):
    """Prueba el acceso tipo diccionario de ParameterLoader."""
    mock_repo_class.return_value = MagicMock()
    loader = ParameterLoader()

    if loader["atr_window"] != 14:
        raise AssertionError("atr_window should be 14")
    if loader["bollinger_band_method"] != "max-min":
        raise AssertionError("bollinger_band_method should be max-min")


# pylint: disable=redefined-outer-name, unused-argument
@pytest.mark.usefixtures("mock_env")
@patch("src.utils.parameters.SymbolRepository")
def test_parameter_loader_get_all_method(mock_repo_class):
    """Prueba el método get_all de ParameterLoader."""
    mock_repo_class.return_value = MagicMock()
    loader = ParameterLoader()
    all_params = loader.get_all()

    if not isinstance(all_params, dict):
        raise AssertionError("Expected all_params to be a dict")
    if "atr_window" not in all_params:
        raise AssertionError("Expected atr_window key in parameters")
    if "cutoff_date" not in all_params:
        raise AssertionError("Expected cutoff_date key in parameters")
    if "conf_matrix_plot_filepath" not in all_params:
        raise AssertionError("Expected conf_matrix_plot_filepath key in parameters")
