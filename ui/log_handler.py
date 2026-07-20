"""A logging handler that forwards records to a Qt signal.

This lets the on-screen Logs panel mirror everything written to the console and
log file, without the panel needing to know about the logging internals.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal


class QtLogEmitter(QObject):
    """Small ``QObject`` that owns the log signal (handlers can't be QObjects)."""

    message = Signal(str)


class QtLogHandler(logging.Handler):
    """Emit formatted log records through a Qt signal for the UI."""

    def __init__(self) -> None:
        super().__init__()
        self.emitter = QtLogEmitter()
        self.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - stdlib API
        try:
            self.emitter.message.emit(self.format(record))
        except Exception:  # pragma: no cover - never let logging crash the app
            self.handleError(record)
