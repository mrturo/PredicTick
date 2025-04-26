"""Unit tests for the ParameterLoader class and configuration logic."""

import json
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from utils.parameters import ParameterLoader


@pytest.fixture
def config_file(tmp_path):
    """Creates a temporary configuration file fixture."""
    config_data = {"model_name": "xgboost_v1", "learning_rate": 0.1}
    path = tmp_path / "config_parameters.json"
    path.write_text(json.dumps(config_data))
    return path


@patch("utils.parameters.SymbolRepository")
@pytest.mark.parametrize(
    "key, expected, should_fail",
    [
        ("model_name", "xgboost_v1", False),
        ("training_symbols", ["AAPL"], False),
        ("mercado_pago_symbols", ["MP"], False),
        ("xtb_symbols", ["EURUSD"], False),
        ("correlative_symbols", ["BTC"], False),
        ("model_name", "WRONG", True),
    ],
)
def test_parameter_loader_initialization(mock_repo, key, expected, should_fail):
    """Test initialization and retrieval of parameters using .get and [] methods."""
    mock_data = json.dumps({"model_name": "xgboost_v1", "learning_rate": 0.1})
    m_open = mock_open(read_data=mock_data)

    with patch("utils.parameters.os.path.exists", return_value=True), patch(
        "utils.parameters.open", m_open
    ):
        mock_repo.return_value.load_training_symbols.return_value = ["AAPL"]
        mock_repo.return_value.load_mercado_pago_symbols.return_value = ["MP"]
        mock_repo.return_value.load_xtb_symbols.return_value = ["EURUSD"]
        mock_repo.return_value.load_correlative_symbols.return_value = ["BTC"]

        loader = ParameterLoader(last_updated=pd.Timestamp("2024-01-01"))
        value = loader.get(key)

        if should_fail:
            if value == expected:
                pytest.fail(f"{key} unexpectedly matched {expected}")
        else:
            if value != expected:
                pytest.fail(f"{key} mismatch: got {value}")

        if not isinstance(loader.get("cutoff_date"), str):
            pytest.fail("cutoff_date should be string")

        try:
            val = loader["training_symbols"]
            if val != ["AAPL"]:
                pytest.fail("unexpected value from __getitem__")
        except KeyError as ex:
            pytest.fail(f"getitem failed unexpectedly: {ex}")


@patch("utils.parameters.os.path.exists", return_value=False)
@patch("utils.parameters.Logger.error")
def test_config_file_not_found(mock_logger, _):
    """Test that missing config file raises FileNotFoundError and logs error."""
    with pytest.raises(FileNotFoundError):
        ParameterLoader()
    if not mock_logger.called:
        pytest.fail("Logger.error was not called")
    if mock_logger.call_args is None:
        pytest.fail("Logger.error call_args is None")


@patch("utils.parameters.SymbolRepository")
@pytest.mark.parametrize(
    "key, expected, should_fail",
    [
        ("param", "value", False),
        ("mercado_pago_symbols", ["MP"], False),
        ("xtb_symbols", "WRONG", True),
    ],
)
def test_parameter_access_methods(mock_repo, key, expected, should_fail):
    """Test hybrid use of get() and __getitem__ for config values."""
    config_data = {"param": "value"}
    m_open = mock_open(read_data=json.dumps(config_data))

    with patch("utils.parameters.os.path.exists", return_value=True), patch(
        "utils.parameters.open", m_open
    ):
        mock_repo.return_value.load_training_symbols.return_value = ["AAPL"]
        mock_repo.return_value.load_mercado_pago_symbols.return_value = ["MP"]
        mock_repo.return_value.load_xtb_symbols.return_value = ["EURUSD"]
        mock_repo.return_value.load_correlative_symbols.return_value = ["BTC"]

        loader = ParameterLoader(last_updated=pd.Timestamp("2024-01-01"))
        value = (
            loader.get(key)
            if key not in loader.__dict__.get("_parameters", {})
            else loader[key]
        )

        if should_fail:
            if value == expected:
                pytest.fail(f"Expected failure for key={key}, got match")
        else:
            if value != expected:
                pytest.fail(f"{key} mismatch: got {value}")


@patch("utils.parameters.SymbolRepository")
def test_parameter_missing_keys(mock_repo):
    """Test retrieval of missing keys returns None or raises KeyError."""
    m_open = mock_open(read_data=json.dumps({"some_key": "some_value"}))

    with patch("utils.parameters.os.path.exists", return_value=True), patch(
        "utils.parameters.open", m_open
    ):
        mock_repo.return_value.load_training_symbols.return_value = []
        mock_repo.return_value.load_mercado_pago_symbols.return_value = []
        mock_repo.return_value.load_xtb_symbols.return_value = []
        mock_repo.return_value.load_correlative_symbols.return_value = []

        loader = ParameterLoader(last_updated=pd.Timestamp("2024-01-01"))

        if loader.get("non_existent") is not None:
            pytest.fail("Expected None for missing key")

        with pytest.raises(KeyError):
            _ = loader["non_existent"]
