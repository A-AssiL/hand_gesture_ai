"""Centralized logging configuration.

Provides a single application logger (``hand_gesture_ai``) with a console
handler and an optional rotating file handler. Modules should obtain child
loggers via :func:`get_logger` so all records share the same configuration.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import LoggingConfig

ROOT_LOGGER_NAME = "hand_gesture_ai"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def setup_logging(config: "LoggingConfig") -> logging.Logger:
    """Configure and return the application root logger.

    Safe to call multiple times; handlers are only attached once.

    Args:
        config: The logging configuration section.

    Returns:
        The configured ``hand_gesture_ai`` logger.
    """
    global _configured
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if _configured:
        return logger

    level = getattr(logging, str(config.level).upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if config.log_to_file:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / config.log_file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _configured = True
    logger.debug("Logging initialized at level %s", config.level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, e.g. ``hand_gesture_ai.camera``."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
