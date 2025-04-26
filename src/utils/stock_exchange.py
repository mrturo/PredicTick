"""Typed and validated representation of a stock exchange and its session hours."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from utils.hours import Hours
from utils.sessions_hours import SessionsHours


@dataclass
class StockExchange:
    """Container for an exchange's identity and session segmentation.

    * code: Exchange ticker code (e.g., 'NYSE', 'JPX').
    * country: Country or region where the exchange operates.
    * sessions_hours: Object containing timezone-aware trading session definitions.
    """

    __slots__ = ("_code", "_country", "_sessions_hours")

    def __init__(self, code: str, country: str, sessions_hours: SessionsHours) -> None:
        self._code = self._validate_str(code, "code")
        self._country = self._validate_str(country, "country")
        if not isinstance(sessions_hours, SessionsHours):
            raise TypeError("`sessions_hours` must be an instance of SessionsHours")
        self._sessions_hours = sessions_hours

    @staticmethod
    def _validate_str(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{field_name}` must be a non-empty string")
        return value.strip().upper()

    @property
    def code(self) -> str:
        """Return the exchange code."""
        return self._code

    @property
    def country(self) -> str:
        """Return the exchange country."""
        return self._country

    @property
    def sessions_hours(self) -> SessionsHours:
        """Return the trading sessions for this exchange."""
        return self._sessions_hours

    def to_utc(self) -> Optional[StockExchange]:
        """Return a deep-copied with ``sessions_hours`` expressed in UTC."""
        utc_sessions_hours = self.sessions_hours.to_utc()
        return (
            None
            if utc_sessions_hours is None
            else StockExchange(self.code, self.country, utc_sessions_hours)
        )

    def to_json(self) -> Any:
        """Object to JSON."""
        return {
            "code": self.code,
            "country": self.country,
            "sessions_hours": (
                None if self.sessions_hours is None else self.sessions_hours.to_json()
            ),
        }

    @staticmethod
    def _get_validated_exchanges(exchanges: Any) -> List[Any]:
        if exchanges is None:
            raise ValueError("Parameter 'exchanges' is not defined")
        if not isinstance(exchanges, List):
            raise ValueError(f"Parameter 'exchanges' is invalid: {exchanges}")
        if len(exchanges) == 0:
            raise ValueError("Parameter 'exchanges' is empty")
        cleaned: List[Any] = [e for e in exchanges if e is not None]
        if len(cleaned) == 0:
            raise ValueError("Parameter 'exchanges' contains only None values")
        return cleaned

    @staticmethod
    def _find_validated_exchange_default(code: str, exchanges: List[Any]) -> Any:
        exchange_default: Any = None
        for exchange in exchanges:
            tmp_code = exchange["code"]
            if tmp_code is not None and isinstance(tmp_code, str):
                tmp_code = tmp_code.strip().upper()
                if tmp_code == code:
                    if exchange_default is not None:
                        raise ValueError(
                            f"Parameter 'exchanges' has duplicated items: {code}"
                        )
                    exchange_default = exchange
        if exchange_default is None:
            raise ValueError(f"'{code}' is not defined in parameter 'exchanges'")
        timezone = exchange_default["timezone"]
        if timezone is None:
            raise ValueError(f"Timezone of '{code}' is not defined")
        if not isinstance(timezone, str):
            raise ValueError(f"Timezone of '{code}' is invalid: '{timezone}'")
        sessions = exchange_default["sessions_hours"]
        if sessions is None:
            raise ValueError(f"The sessions hours of {code} are not defined")
        if not isinstance(sessions, dict):
            raise ValueError(f"The sessions hours of {code} are invalid: '{sessions}'")
        country = exchange_default["country"]
        if country is None:
            raise ValueError(f"Country of '{code}' is not defined")
        if not isinstance(country, str):
            raise ValueError(f"Country of '{code}' is invalid: '{country}'")
        return exchange_default

    @staticmethod
    def _get_validated_exchange_default_code(exchange_default: Any) -> str:
        if exchange_default is None:
            raise ValueError("Parameter 'exchange_default' is not defined.")
        if not isinstance(exchange_default, str):
            raise ValueError(
                f"Parameter 'exchange_default' is invalid: {exchange_default}."
            )
        exchange_default = exchange_default.strip().upper()
        if len(exchange_default) == 0:
            raise ValueError("Parameter 'exchange_default' is empty.")
        return exchange_default

    @staticmethod
    def _extract_hours(sessions: dict, key: str, exchange_code: str) -> Optional[Hours]:
        """Extracts and validates the trading hours for a specific session type."""
        segment = sessions.get(key)
        if segment is None:
            return None
        open_hour = segment.get("open")
        close_hour = segment.get("close")
        if open_hour is not None and not isinstance(open_hour, str):
            raise ValueError(
                f"{key.capitalize()} open hour of '{exchange_code}' is invalid: '{open_hour}'"
            )
        if close_hour is not None and not isinstance(close_hour, str):
            raise ValueError(
                f"{key.capitalize()} close hour of '{exchange_code}' is invalid: '{close_hour}'"
            )
        return Hours(open_hour, close_hour)

    @staticmethod
    def from_parameter(exchange_default: Any, exchanges: Any) -> StockExchange:
        """Initializes and validates the default stock exchange configuration.

        Retrieves the default exchange code and list of exchanges from parameters,
        matches and validates the selected exchange, and constructs a StockExchange
        instance with properly validated session hours and timezone.
        """
        exchange_default_code: str = StockExchange._get_validated_exchange_default_code(
            exchange_default
        )
        validated_exchanges: List[Any] = StockExchange._get_validated_exchanges(
            exchanges
        )
        tmp_exchange: Any = StockExchange._find_validated_exchange_default(
            exchange_default_code, validated_exchanges
        )
        sessions = tmp_exchange["sessions_hours"]
        regular = StockExchange._extract_hours(
            sessions, "regular", exchange_default_code
        )
        if regular is None:
            raise ValueError(
                f"The regular session of '{exchange_default_code}' is not defined"
            )
        pre_market = StockExchange._extract_hours(
            sessions, "pre_market", exchange_default_code
        )
        post_market = StockExchange._extract_hours(
            sessions, "post_market", exchange_default_code
        )
        return StockExchange(
            exchange_default_code,
            tmp_exchange["country"],
            SessionsHours(pre_market, regular, post_market, tmp_exchange["timezone"]),
        )
