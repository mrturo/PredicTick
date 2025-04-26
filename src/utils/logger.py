"""
Logger — Structured logging utility.

Provides standardized, timestamped output for info, warnings, errors, successes, and debug messages.
"""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


class Logger:
    """Advanced logger with daily file rotation and formatted console output."""

    log_basepath = "logs"
    log_filepath = "app.log"
    _max_backup = 7
    logger = None

    @staticmethod
    def _initialize():
        """Initializes the logger configuration with console and file handlers."""
        if Logger.logger is not None:
            return

        Logger.logger = logging.getLogger("AppLogger")
        Logger.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        Logger.logger.addHandler(console_handler)

        is_pytest = "PYTEST_CURRENT_TEST" in os.environ or any(
            "pytest" in arg for arg in sys.argv
        )
        if not is_pytest:  # pragma: no cover
            os.makedirs(Logger.log_basepath, exist_ok=True)
            log_filepath = os.path.join(Logger.log_basepath, Logger.log_filepath)
            file_handler = TimedRotatingFileHandler(
                log_filepath,
                when="midnight",
                backupCount=Logger._max_backup,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            Logger.logger.addHandler(file_handler)

    @staticmethod
    def log(symbol: str, message: str) -> str:
        """Formats the log message with a timestamp and indentation."""
        Logger._initialize()
        leading_spaces = len(message) - len(message.lstrip(" "))
        prefix = f"{symbol} | " if symbol.strip() else ""
        formatted = f"{prefix}{' ' * leading_spaces}{message.lstrip(' ')}"
        Logger.logger.info(formatted)
        return formatted

    @staticmethod
    def debug(message: str) -> str:
        """Logs a debug message with timestamp."""
        return Logger.log("⚫ DEBUG", message)

    @staticmethod
    def error(message: str) -> str:
        """Logs an error message with timestamp."""
        return Logger.log("🔴 ERROR", message)

    @staticmethod
    def info(message: str) -> str:
        """Logs an informational message with timestamp."""
        return Logger.log("🔵 INFO ", message)

    @staticmethod
    def separator(length: int = 100) -> str:
        """Prints a separator line with a configurable number of asterisks."""
        line = "*" * length
        return Logger.log("", line)

    @staticmethod
    def success(message: str) -> str:
        """Logs a success message with timestamp."""
        return Logger.log("🟢 SUCC ", message)

    @staticmethod
    def warning(message: str) -> str:
        """Logs a warning message with timestamp."""
        return Logger.log("🟡 WARN ", message)


# if __name__ == "__main__":
#    Logger.debug("MESSAGE")
#    Logger.error("MESSAGE")
#    Logger.info("MESSAGE")
#    Logger.success("MESSAGE")
#    Logger.warning("MESSAGE")
