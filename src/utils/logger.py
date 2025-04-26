"""
Logger — Structured logging utility.

Provides standardized, timestamped output for info, warnings, errors, successes, and debug messages.
"""

from datetime import datetime


class Logger:
    """Simple logger for consistent messaging with indentation preservation and timestamp."""

    @staticmethod
    def format_message(symbol: str, message: str):
        """Public wrapper for testing internal formatting logic."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        leading_spaces = len(message) - len(message.lstrip(" "))
        return f"{timestamp} | {symbol} | {' ' * leading_spaces}{message.lstrip(' ')}"

    @staticmethod
    def debug(message: str):
        """Logs a debug message with timestamp."""
        print(Logger.format_message("🐞DEBUG ", message))

    @staticmethod
    def error(message: str):
        """Logs an error message with timestamp."""
        print(Logger.format_message("❌ ERROR", message))

    @staticmethod
    def info(message: str):
        """Logs an informational message with timestamp."""
        print(Logger.format_message("ℹ️  INFO ", message))

    @staticmethod
    def separator(length: int = 100):
        """Prints a separator line with a configurable number of asterisks."""
        print("*" * length)

    @staticmethod
    def simple(message: str):
        """Logs a plain message without tag."""
        print(Logger.format_message("        ", message))

    @staticmethod
    def success(message: str):
        """Logs a success message with timestamp."""
        print(Logger.format_message("✅ SUCC ", message))

    @staticmethod
    def warning(message: str):
        """Logs a warning message with timestamp."""
        print(Logger.format_message("⚠️  WARN ", message))


# if __name__ == "__main__":
#     Logger.debug("MESSAGE")
#    Logger.error("MESSAGE")
#    Logger.info("MESSAGE")
#    Logger.simple("MESSAGE")
#    Logger.success("MESSAGE")
#    Logger.warning("MESSAGE")
