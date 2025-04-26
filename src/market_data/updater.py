"""Market data refresh and enrichment coordinator.

This module orchestrates a streamlined ETL workflow that:

1. Ingests raw market data via :py:meth:`Ingester.ingest_data`.
2. Checks data freshness with :py:meth:`Validator.validate_data`.
3. Uploads files to Google Drive through :class:`GoogleDriveManager`.
4. Enriches market-symbol information by calling :meth:`Updater.enrich_data`.

All steps emit structured events through the centralized logger defined in
``utils.logger.Logger``.

The module can be imported as a library (exposing :class:`Updater`) or executed
directly (``python -m updater``). When run as a script it performs the complete
workflow described above.
"""

from market_data.ingestion.pipelines.ingester import Ingester
from market_data.ingestion.raw.raw_data import RawData
from market_data.processing.enrichment.enriched_data import EnrichedData
from market_data.utils.storage.google_drive_manager import GoogleDriveManager
from market_data.utils.validation.validator import Validator
from utils.logger import Logger


# pylint: disable=too-few-public-methods
class Updater:
    """Facade for the market-data enrichment stage.

    The class is intentionally *stateless*; it exposes a single static method
    that wraps enriched-data generation with consistent logging and error
    handling.

    Because no instances are created, the class is safe to use in multi-process
    or multi-threaded environments.
    """

    @staticmethod
    def enrich_data() -> None:
        """Enrich and persist market symbol data with logging and error handling."""
        try:
            EnrichedData.generate()
        except Exception as e:  # pylint: disable=broad-exception-caught
            Logger.error(f"Symbol data enrichment failed: {e}")


if __name__ == "__main__":
    updated, filepath, found_data = Ingester.ingest_data()
    if Validator.validate_data(RawData.get_symbols(), True):
        RawData.save()
        if updated is True or found_data["drive_data"] is False:
            google_drive = GoogleDriveManager()
            google_drive.upload_file(filepath)
        else:
            Logger.warning(
                f"Upload to Google Drive skipped '{filepath}': "
                f"No changes detected in market data."
            )
            Logger.separator()
        Updater.enrich_data()
        Logger.separator()
        Validator.validate_data(EnrichedData.get_symbols(), False)
