"""The main application window and its control logic.

Layout::

    +-----------------------------------------------------+
    |  [ Camera controls ]                                |
    | +---------------------------+  +------------------+  |
    | |                          |  | Statistics       |  |
    | |        Video feed        |  | Hand Information  |  |
    | |                          |  | Finger States    |  |
    | |                          |  | Detected Gesture  |  |
    | +---------------------------+  | Logs             |  |
    |  [ Action buttons ]           +------------------+  |
    +-----------------------------------------------------+

Phase 1 wires up: Start/Stop camera, camera switching, live FPS, hand info,
and Save Screenshot. Buttons for later phases (Capture Dataset, Train Model,
Load Model, Record Video) are present but guarded and log a friendly message.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import cv2
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.log_handler import QtLogHandler
from ui.panels import (
    FingerStatePanel,
    GesturePanel,
    HandInfoPanel,
    LogPanel,
    StatisticsPanel,
)
from ui.theme import DARK_STYLESHEET
from ui.video_thread import VideoThread
from ui.video_widget import VideoWidget
from utils.logger import ROOT_LOGGER_NAME, get_logger
from vision.landmarks import HandResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import Config

logger = get_logger("main_window")


class MainWindow(QMainWindow):
    """Top-level window orchestrating the video thread and info panels."""

    def __init__(self, config: "Config") -> None:
        super().__init__()
        self._config = config
        self._thread: Optional[VideoThread] = None
        self._last_frame = None

        self.setWindowTitle(config.app.name)
        self.resize(1280, 760)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self._attach_log_panel()
        self._populate_cameras()
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)

        # ---- Left column: camera controls + video + action buttons ----
        left = QVBoxLayout()

        controls = QHBoxLayout()
        self._start_btn = QPushButton("Start Camera")
        self._start_btn.clicked.connect(self.start_camera)
        self._stop_btn = QPushButton("Stop Camera")
        self._stop_btn.setObjectName("StopButton")
        self._stop_btn.clicked.connect(self.stop_camera)
        self._stop_btn.setEnabled(False)
        self._camera_combo = QComboBox()
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        controls.addWidget(self._start_btn)
        controls.addWidget(self._stop_btn)
        controls.addWidget(QLabel("Camera:"))
        controls.addWidget(self._camera_combo, 1)
        left.addLayout(controls)

        self._video = VideoWidget()
        left.addWidget(self._video, 1)

        actions = QGridLayout()
        self._screenshot_btn = QPushButton("Save Screenshot")
        self._screenshot_btn.clicked.connect(self.save_screenshot)
        self._capture_btn = QPushButton("Capture Dataset")
        self._capture_btn.clicked.connect(lambda: self._coming_soon("Dataset recording", 2))
        self._train_btn = QPushButton("Train Model")
        self._train_btn.clicked.connect(lambda: self._coming_soon("Model training", 3))
        self._load_btn = QPushButton("Load Model")
        self._load_btn.clicked.connect(lambda: self._coming_soon("Model loading", 3))
        self._record_btn = QPushButton("Record Video")
        self._record_btn.clicked.connect(lambda: self._coming_soon("Video recording", 4))
        self._settings_btn = QPushButton("Settings")
        self._settings_btn.clicked.connect(lambda: self._coming_soon("Settings dialog", 4))
        actions.addWidget(self._screenshot_btn, 0, 0)
        actions.addWidget(self._capture_btn, 0, 1)
        actions.addWidget(self._train_btn, 0, 2)
        actions.addWidget(self._load_btn, 1, 0)
        actions.addWidget(self._record_btn, 1, 1)
        actions.addWidget(self._settings_btn, 1, 2)
        left.addLayout(actions)

        # ---- Right column: information panels ----
        right = QVBoxLayout()
        self._stats_panel = StatisticsPanel()
        self._hand_panel = HandInfoPanel()
        self._finger_panel = FingerStatePanel()
        self._gesture_panel = GesturePanel()
        self._log_panel = LogPanel()
        right.addWidget(self._stats_panel)
        right.addWidget(self._hand_panel)
        right.addWidget(self._finger_panel)
        right.addWidget(self._gesture_panel)
        right.addWidget(self._log_panel, 1)

        root.addLayout(left, 3)
        root.addLayout(right, 2)
        self.setCentralWidget(central)

        self._stats_panel.update_resolution(
            self._config.camera.width, self._config.camera.height
        )

    def _attach_log_panel(self) -> None:
        handler = QtLogHandler()
        handler.setLevel(logging.INFO)
        handler.emitter.message.connect(self._log_panel.append)
        logging.getLogger(ROOT_LOGGER_NAME).addHandler(handler)
        self._log_handler = handler

    def _populate_cameras(self) -> None:
        probe = VideoThread(self._config)
        try:
            cameras = probe.available_cameras()
        finally:
            del probe
        if not cameras:
            cameras = [self._config.camera.index]
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        for idx in cameras:
            self._camera_combo.addItem(f"Camera {idx}", idx)
        self._camera_combo.blockSignals(False)

    # -------------------------------------------------------------- actions
    def start_camera(self) -> None:
        """Start the capture thread and begin streaming."""
        if self._thread is not None:
            return
        self._thread = VideoThread(self._config)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.hands_updated.connect(self._on_hands)
        self._thread.fps_updated.connect(self._stats_panel.update_fps)
        self._thread.error.connect(self._on_error)
        index = self._camera_combo.currentData()
        if index is not None:
            self._thread.switch_camera(int(index))
        self._thread.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self.statusBar().showMessage("Camera started")
        logger.info("Camera started")

    def stop_camera(self) -> None:
        """Stop the capture thread and reset the UI."""
        if self._thread is None:
            return
        self._thread.stop()
        self._thread = None
        self._last_frame = None
        self._video.clear_frame()
        self._stats_panel.update_fps(0.0)
        self._stats_panel.update_hands(0)
        self._hand_panel.update_hands([])
        self._finger_panel.reset()
        self._gesture_panel.reset()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.statusBar().showMessage("Camera stopped")
        logger.info("Camera stopped")

    def save_screenshot(self) -> None:
        """Save the current annotated frame to a PNG file."""
        if self._last_frame is None:
            QMessageBox.information(self, "Screenshot", "Start the camera first.")
            return
        captures = Path(self._config.logging.log_dir).parent / "assets" / "captures"
        captures.mkdir(parents=True, exist_ok=True)
        default = str(captures / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", default, "PNG (*.png)")
        if path:
            cv2.imwrite(path, self._last_frame)
            self.statusBar().showMessage(f"Screenshot saved: {path}")
            logger.info("Screenshot saved: %s", path)

    # ---------------------------------------------------------------- slots
    def _on_frame(self, frame) -> None:
        self._last_frame = frame
        self._video.show_frame(frame)

    def _on_hands(self, hands: List[HandResult]) -> None:
        self._stats_panel.update_hands(len(hands))
        self._hand_panel.update_hands(hands)

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Camera Error", message)
        logger.error("Camera error: %s", message)
        self.stop_camera()

    def _on_camera_changed(self, _combo_index: int) -> None:
        index = self._camera_combo.currentData()
        if self._thread is not None and index is not None:
            self._thread.switch_camera(int(index))
            logger.info("Switched to camera %s", index)

    def _coming_soon(self, feature: str, phase: int) -> None:
        message = f"{feature} arrives in Phase {phase}."
        self.statusBar().showMessage(message)
        logger.info(message)
        QMessageBox.information(self, "Coming soon", message)

    # --------------------------------------------------------------- events
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Ensure the capture thread is stopped when the window closes."""
        self.stop_camera()
        super().closeEvent(event)
