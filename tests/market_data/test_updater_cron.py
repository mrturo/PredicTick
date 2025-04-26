"""
Unit tests for the UpdaterCron scheduler class.

These tests validate the behavior of the run_conditional_market_update method,
ensuring it correctly triggers updates only when the appropriate time threshold is met,
and logs the appropriate messages.

Dependencies are mocked to isolate and control behavior.
"""

# pylint: disable=protected-access

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from market_data.updater_cron import UpdaterCron


@patch("market_data.updater.Updater.update_data")
@patch("market_data.gateway.Gateway.get_last_update", return_value=None)
@patch("market_data.gateway.Gateway.load")
@patch("utils.logger.Logger.warning")
def test_update_triggers_when_no_previous_update(
    mock_warning, _mock_load, _mock_last, mock_update
):
    """
    Ensure update_data is triggered when no last update exists.
    """
    tries = UpdaterCron.run_conditional_market_update(1)
    if tries != 1:
        raise AssertionError("Expected tries to be reset to 1")
    if not mock_update.called:
        raise AssertionError("Expected update_data to be called")
    if not mock_warning.called:
        raise AssertionError("Expected warning to be logged")


@patch("market_data.updater.Updater.update_data")
@patch("market_data.gateway.Gateway.get_last_update")
@patch("market_data.gateway.Gateway.load")
@patch("utils.logger.Logger.info")
def test_no_update_when_within_hour(mock_info, _mock_load, mock_last, mock_update):
    """
    Ensure update_data is not called when last update was less than 1 hour ago.
    """
    mock_last.return_value = datetime.now(timezone.utc) - timedelta(minutes=30)
    tries = UpdaterCron.run_conditional_market_update(1)
    if tries != 2:
        raise AssertionError("Expected tries to increment")
    if mock_update.called:
        raise AssertionError("Expected update_data not to be called")
    if not mock_info.called:
        raise AssertionError("Expected info log for waiting time")


@patch("market_data.updater.Updater.update_data")
@patch("market_data.gateway.Gateway.get_last_update")
@patch("market_data.gateway.Gateway.load")
@patch("utils.logger.Logger.info")
def test_update_triggered_after_one_hour(mock_info, _mock_load, mock_last, mock_update):
    """
    Ensure update_data is called when more than 1 hour has passed since last update.
    """
    mock_last.return_value = datetime.now(timezone.utc) - timedelta(hours=2)
    tries = UpdaterCron.run_conditional_market_update(3)
    if tries != 1:
        raise AssertionError("Expected tries to reset to 1")
    if not mock_update.called:
        raise AssertionError("Expected update_data to be called")
    if not mock_info.called:
        raise AssertionError("Expected info log to be called for update start")


@patch("market_data.updater.Updater.update_data")
@patch("market_data.gateway.Gateway.get_last_update")
@patch("market_data.gateway.Gateway.load")
def test_tries_increment_without_logging_second_time(
    _mock_load, mock_last, mock_update
):
    """
    Ensure tries increments without logging again when called repeatedly within the hour.
    """
    mock_last.return_value = datetime.now(timezone.utc) - timedelta(minutes=45)
    tries = 2
    updated_tries = UpdaterCron.run_conditional_market_update(tries)
    if updated_tries != 3:
        raise AssertionError("Expected tries to increment to 3")
    if mock_update.called:
        raise AssertionError("update_data should not be called again within 1 hour")
