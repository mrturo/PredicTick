"""Unit tests for the ParameterLoader configuration manager."""

from unittest.mock import MagicMock, patch

import pytest  # type: ignore

from src.utils.config.parameters import ParameterLoader
from src.utils.exchange.stock_exchange import StockExchange


@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to set up mock environment variables."""
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
        "INGESTER_RETRIES": "2",
        "DOWN_THRESHOLD": "0.02",
        "EVALUATION_REPORT_BASEPATH": "reports/",
        "HISTORICAL_DAYS_FALLBACK": "30",
        "HISTORICAL_WINDOW_DAYS": "365",
        "INTERVAL": "1m",
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
        "STOCH_RSI_MIN_PERIODS": "14",
        "STOCH_RSI_WINDOW": "14",
        "UP_THRESHOLD": "0.02",
        "VOLUME_WINDOW": "20",
        "WILLIAMS_R_WINDOW": "14",
        "EXCHANGE_DEFAULT": "NMS",
        "HOLIDAY_COUNTRY": "US",
        "DATE_FORMAT": "%Y-%m-%d",
        "F1_SCORE_PLOT_TITLE": "F1 Score by Class",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


# pylint: disable=redefined-outer-name, unused-argument
@patch("src.utils.config.parameters.SymbolRepository")
def test_parameter_loader_get_method(mock_repo_class):
    """Test the get method of ParameterLoader."""
    mock_repo_class.return_value = MagicMock()
    loader = ParameterLoader()
    result = loader.get("non_existent_key")
    if result is not None:
        raise AssertionError("Expected None for missing key")
    result_with_default = loader.get("non_existent_key", default="default_value")
    if result_with_default != "default_value":
        raise AssertionError("Expected default_value for missing key with default")


# pylint: disable=redefined-outer-name, unused-argument
@patch("src.utils.config.parameters.SymbolRepository")
def test_parameter_loader_getitem_method(mock_repo_class):
    """Test dictionary-style access of ParameterLoader."""
    mock_repo_class.return_value = MagicMock()
    loader = ParameterLoader()
    if loader["atr_window"] != 14:
        raise AssertionError("atr_window should be 14")
    if loader["bollinger_band_method"] != "max-min":
        raise AssertionError("bollinger_band_method should be max-min")


# pylint: disable=redefined-outer-name, unused-argument
@pytest.mark.usefixtures("mock_env")
@patch("src.utils.config.parameters.SymbolRepository")
def test_parameter_loader_get_all_method(mock_repo_class):
    """Test the get_all method of ParameterLoader."""
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


# pylint: disable=redefined-outer-name, unused-argument
@patch("src.utils.config.parameters.SymbolRepository")
def test_parameter_loader_exchange_default_method(mock_repo_class):
    """Test the exchange method of ParameterLoader."""
    mock_repo_class.return_value = MagicMock()
    loader = ParameterLoader()
    exchange = loader.exchange_default()
    if not isinstance(exchange, StockExchange):
        raise AssertionError("Expected exchange to be an instance of StockExchange")
    if exchange.code != "NMS":
        raise AssertionError("Expected exchange code to be 'NMS'")
    if exchange.country != "US":
        raise AssertionError("Expected exchange country to be 'US'")
    if exchange.sessions_hours is None:
        raise AssertionError("Expected sessions_hours to be defined")


@patch("src.utils.config.parameters.SymbolRepository")
@patch("src.utils.config.parameters.StockExchange.from_parameter")
def test_parameter_loader_raises_if_stock_exchange_not_found(
    mock_from_parameter, mock_repo_class
):
    """Test that ValueError is raised if default exchange is not found."""
    mock_repo_class.return_value = MagicMock()
    mock_from_parameter.return_value.to_utc.return_value = None
    with pytest.raises(ValueError, match="Stock exchange not found"):
        ParameterLoader()


# pylint: disable=redefined-outer-name, unused-argument
@patch("src.utils.config.parameters.SymbolRepository")
def test_parameter_loader_exchange_method_success(mock_repo_class):
    """ParameterLoader.exchange should return the requested StockExchange instance."""
    mock_repo_class.return_value = MagicMock()
    loader = ParameterLoader()
    exchange = loader.exchange(
        "PCX"
    )  # uses the static config defined in ParameterLoader
    if not isinstance(exchange, StockExchange):
        raise AssertionError("Expected a StockExchange instance")
    if exchange.code != "PCX":
        raise AssertionError("Exchange code should match the requested ID")
    if exchange.country != "US":
        raise AssertionError("Exchange country should be 'US'")
    if exchange.sessions_hours is None:
        raise AssertionError("sessions_hours must be populated for the exchange")


# pylint: disable=redefined-outer-name, unused-argument
@patch("src.utils.config.parameters.SymbolRepository")
def test_parameter_loader_exchange_method_raises_for_invalid_code(mock_repo_class):
    """ParameterLoader.exchange should raise ValueError when the ID is not found."""
    mock_repo_class.return_value = MagicMock()
    loader = ParameterLoader()
    # Ensure the inner StockExchange.from_parameter call returns an object whose
    # to_utc() returns None, mimicking a lookup failure.
    with patch(
        "src.utils.config.parameters.StockExchange.from_parameter"
    ) as mock_from_parameter:
        mock_from_parameter.return_value.to_utc.return_value = None
        with pytest.raises(ValueError, match="Stock exchange not found"):
            loader.exchange("INVALID")
