"""Market data refresh and enrichment coordinator.

This module orchestrates a streamlined ETL workflow that:

1. Ingests raw market data via :py:meth:`Ingester.ingest_raw_data`.
2. Checks data freshness with :py:meth:`Validator.validate_data`.
3. Uploads files to Google Drive through :class:`GoogleDriveManager`.
4. Enriches market-symbol information by calling :meth:`Updater.enrich_data`.

All steps emit structured events through the centralized logger defined in
``utils.io.logger.Logger``.

The module can be imported as a library (exposing :class:`Updater`) or executed
directly (``python -m updater``). When run as a script it performs the complete
workflow described above.
"""

from typing import Any, Dict, Optional

from src.market_data.ingestion.pipelines.ingester import Ingester
from src.market_data.ingestion.raw.raw_data import RawData
from src.market_data.processing.enrichment.enriched_data import EnrichedData
from src.market_data.utils.storage.google_drive_manager import \
    GoogleDriveManager
from src.market_data.utils.validation.validator import Validator
from src.utils.config.parameters import ParameterLoader
from src.utils.io.logger import Logger


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
            Logger.error(f"Symbols data enrichment failed: {e}")
        if EnrichedData.exist() is True:
            Logger.separator()
            Validator.validate_data(EnrichedData.get_symbols(), False)


_PARAMS = ParameterLoader()
_FORCE_DATA_ENRICHMENT: Any = _PARAMS.get("force_data_enrichment") if _PARAMS else None
_INTERVAL: Optional[Dict[str, Any]] = _PARAMS.get("interval") if _PARAMS else None


def _force_data_enrichment() -> bool:
    """Load and validate the 'force_data_enrichment' parameter from configuration."""
    result: Any = _FORCE_DATA_ENRICHMENT
    if result is None:
        result = False
    elif not isinstance(result, bool):
        raise TypeError("force_data_enrichment property must be of type bool or None")
    return result


def _can_enriched_data(file_path: Optional[str]) -> bool:
    file_path = file_path.strip() if file_path else ""
    _enriched_data_id = (EnrichedData.get_id() or "").strip()
    _raw_data_id = (RawData.get_id() or "").strip()
    _file_enriched_interval = (EnrichedData.get_interval() or "").strip()
    _property_enriched_interval = (
        _INTERVAL.get("market_enriched_data") if _INTERVAL else ""
    )
    _enriched_symbols: Optional[Dict[str, dict]] = EnrichedData.get_symbols()
    _raw_symbols: Optional[Dict[str, dict]] = RawData.get_symbols()
    conditions_met = True
    if _force_data_enrichment() is True:
        Logger.debug("Forcing data enrichment...")
        conditions_met = False
    elif not _enriched_data_id:
        Logger.debug("No enriched data found...")
        conditions_met = False
    elif not _raw_data_id or _enriched_data_id != _raw_data_id:
        Logger.debug(
            f"Enriched data ID does not match raw data ID... "
            f"({_enriched_data_id} != {_raw_data_id})"
        )
        conditions_met = False
    elif not _file_enriched_interval:
        Logger.debug("No enriched interval found...")
        conditions_met = False
    elif (
        not _property_enriched_interval
        or _file_enriched_interval != _property_enriched_interval
    ):
        Logger.debug(
            f"Enriched interval does not match raw interval... "
            f"({_file_enriched_interval} != {_property_enriched_interval})"
        )
        conditions_met = False
    elif not _enriched_symbols:
        Logger.debug("No enriched symbols found...")
        conditions_met = False
    elif not _raw_symbols or len(_enriched_symbols) != len(_raw_symbols):
        Logger.debug(
            f"Enriched symbols do not match raw symbols... "
            f"({len(_enriched_symbols)} != {len(_raw_symbols) if _raw_symbols else 'None'})"
        )
        conditions_met = False
    elif Validator.validate_data(_enriched_symbols, False, True) is False:
        Logger.debug("Validation of enriched symbols against raw symbols failed...")
        conditions_met = False
    return conditions_met


if __name__ == "__main__":
    updated, filepath, found_raw_data = Ingester.ingest_raw_data()
    Logger.separator()
    RawData.load()
    raw_symbols: Dict[str, dict] = RawData.get_symbols()
    if Validator.validate_data(raw_symbols, True):
        google_drive = GoogleDriveManager()
        Logger.separator()
        RawData.save()
        if updated is True or found_raw_data["drive_data"] is False:
            google_drive.upload_file(filepath)
        else:
            Logger.warning(
                f"Upload to Google Drive skipped '{filepath}': "
                f"No changes detected in market data."
            )
        Logger.separator()
        _enriche_data: Dict[str, Any] = EnrichedData.load()
        _enriched_filepath: Optional[str] = (
            _enriche_data["filepath"].strip()
            if _enriche_data is not None
            and isinstance(_enriche_data["filepath"], str)
            and len(_enriche_data["filepath"].strip()) > 0
            else _PARAMS.get("enriched_marketdata_filepath")
        )
        if updated is True or _can_enriched_data(_enriched_filepath) is False:
            Updater.enrich_data()
        else:
            if _enriched_filepath is not None and len(_enriched_filepath.strip()) > 0:
                enriched_filepath = (  # pylint: disable=invalid-name
                    f" '{_enriched_filepath}'"
                )
            Logger.separator()
            Logger.warning(
                f"Data enrichment skipped {_enriched_filepath}: "
                "No changes detected in market data "
                "and enriched data already exists."
            )
            Logger.separator()
        google_drive.upload_file(_enriched_filepath)
        Logger.separator()
