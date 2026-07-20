"""Application entry point for the AI Hand Gesture Recognition system.

This module wires together the configuration, logging, and the PySide6 GUI.
Run it with:

    python app.py

Phase 1 scope: project setup, GUI shell, live webcam, and real-time
MediaPipe hand tracking. Later phases add finger analysis, gesture
recognition, orientation/measurements, dataset recording, and model training.
"""
from __future__ import annotations

import sys

from config.settings import load_config
from utils.logger import setup_logging


def main() -> int:
    """Bootstrap and run the Qt application.

    Returns:
        The process exit code returned by the Qt event loop.
    """
    config = load_config()
    logger = setup_logging(config.logging)
    logger.info("Starting %s v%s", config.app.name, config.app.version)

    # Imported lazily so that configuration/logging failures surface before we
    # pull in the (heavier) Qt + OpenCV + MediaPipe stack.
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(config.app.name)
    app.setApplicationDisplayName(config.app.name)

    window = MainWindow(config)
    window.show()

    exit_code = app.exec()
    logger.info("Application closed (exit code=%s)", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
