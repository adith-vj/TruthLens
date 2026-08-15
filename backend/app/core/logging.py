"""
core/logging.py — Structured logging configuration for the verification pipeline.

Configures the root logger and exposes a dedicated child logger for this
service under the 'truthlens.verify' namespace, keeping it isolated from
the 'truthlens' logger used by the existing backend/main.py.

Usage:
    from app.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Claim received", extra={"claim_length": len(text)})

Call configure_logging() once at application startup (in the FastAPI lifespan
context). Subsequent calls are idempotent.
"""

import logging
import sys

from app.core.config import settings

# Namespace for all loggers in this service.
# Child loggers created via get_logger(__name__) will inherit this handler.
SERVICE_LOGGER_NAME = "truthlens.verify"

_configured = False


def configure_logging() -> None:
    """
    Configure logging for the verification service.

    - Sets the log level from settings.LOG_LEVEL.
    - Outputs to stdout with ISO 8601 timestamps.
    - Idempotent: safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure the service root logger. Child loggers inherit this handler.
    service_logger = logging.getLogger(SERVICE_LOGGER_NAME)
    service_logger.setLevel(log_level)

    # Avoid adding duplicate handlers if called more than once.
    if not service_logger.handlers:
        service_logger.addHandler(handler)

    # Prevent log records from propagating to the root logger and being
    # printed twice (uvicorn also attaches a root handler).
    service_logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'truthlens.verify' namespace.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A Logger instance named 'truthlens.verify.<name>'.
    """
    # Strip the 'app.' prefix from module names for cleaner log output.
    short_name = name.removeprefix("app.")
    return logging.getLogger(f"{SERVICE_LOGGER_NAME}.{short_name}")
