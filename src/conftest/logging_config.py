"""
ConfTest Structured Logging Subsystem.

Provides unified, formatted logging with ISO-8601 timestamps,
log levels, module names, and optional structured JSON output.
"""

import logging
import sys
from typing import Optional
from conftest.config import settings


class ConsoleFormatter(logging.Formatter):
    """Clean, human-readable terminal formatter with level colors for developer clarity."""

    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: BLUE + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure root and application loggers.

    Args:
        level: Optional log level override (e.g. 'DEBUG', 'INFO', 'WARNING').
    """
    log_level_str = (level or settings.log_level).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate log outputs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    # Suppress verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Helper function to get a named logger.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured logging.Logger instance.
    """
    return logging.getLogger(name)


# Initialize logging immediately on import
configure_logging()
