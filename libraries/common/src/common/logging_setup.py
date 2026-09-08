"""
Application-wide logging setup shared by all services.
"""

import logging

import gen3logging

from common.config import DEBUG, GEN3_JSON_LOGS, VERBOSE_INTERNAL_LOGS

# Loggers that are chatty enough to drown out application logs at their own default
# levels, so they get pinned to `warning` unless VERBOSE_INTERNAL_LOGS is on.
INTERNAL_LOGGERS = frozenset(
    {
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "httpcore",
        "httpx",
        "asyncio",
    }
)


def configure_logging() -> None:
    """
    Route the root logger and noisy third-party loggers through gen3logging.

    Must run *after* uvicorn installs its own logging configuration, otherwise uvicorn
    overwrites these handlers. Uvicorn configures logging in `Config.__init__` and only
    imports the application afterwards in `Config.load`, so calling this from the app
    factory satisfies the ordering.
    """
    _remove_handlers(logging.getLogger())
    gen3logging.get_logger(None, log_level="debug" if DEBUG else "info", json_logs=GEN3_JSON_LOGS)

    internal_log_level = "debug" if VERBOSE_INTERNAL_LOGS else "warning"

    for logger_name in sorted(INTERNAL_LOGGERS):
        _remove_handlers(logging.getLogger(logger_name))
        gen3logging.get_logger(logger_name, log_level=internal_log_level, json_logs=GEN3_JSON_LOGS)


def _remove_handlers(logger: logging.Logger) -> None:
    """
    Remove every handler attached to the given logger.

    Args:
        logger (logging.Logger): The logger to strip handlers from.
    """
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
