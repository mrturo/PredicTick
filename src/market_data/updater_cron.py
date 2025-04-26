"""Scheduler to run conditional market update every hour."""

import time
from datetime import datetime, timedelta, timezone

from market_data.gateway import Gateway
from market_data.updater import Updater
from utils.logger import Logger


# pylint: disable=too-few-public-methods
class UpdaterCron:
    """Periodically triggers market data updates based on time elapsed since last update."""

    @staticmethod
    def run_conditional_market_update(tries: int) -> int:
        """Triggers market data update only if at least one hour's passed since the last update."""
        Gateway.load()
        last_update = Gateway.get_last_update()

        if last_update is None:
            Logger.warning(
                "No previous update timestamp found. Proceeding with update."
            )
            Updater.update_data()
            tries = 1
            return tries

        now = datetime.now(timezone.utc)
        elapsed = now - last_update

        if elapsed < timedelta(hours=1):
            if tries == 1:
                remaining = timedelta(hours=1) - elapsed
                mins, secs = divmod(int(remaining.total_seconds()), 60)
                last_str = last_update.strftime("%Y-%m-%d %H:%M:%S UTC")
                Logger.info(
                    f"Last update was at {last_str}. "
                    f"Next eligible update in {mins}m {secs}s."
                )
            tries += 1
        else:
            Logger.info("Starting market data update...")
            Updater.update_data()
            tries = 1

        return tries


if __name__ == "__main__":
    TRIES = 1
    while True:
        try:
            TRIES = UpdaterCron.run_conditional_market_update(TRIES)
        except RuntimeError as error:
            Logger.error(f"Error during scheduled execution: {error}")
        time.sleep(60)
