"""Webcam capture abstraction built on OpenCV ``VideoCapture``.

Encapsulates opening/closing devices, applying resolution/FPS settings,
optional horizontal mirroring, and enumerating available cameras.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import cv2

from utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from config.settings import CameraConfig

logger = get_logger("camera")


class Camera:
    """Manage a single OpenCV webcam device."""

    def __init__(self, config: "CameraConfig") -> None:
        self._config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._index = config.index

    @property
    def index(self) -> int:
        """The currently selected camera index."""
        return self._index

    def open(self, index: Optional[int] = None) -> None:
        """Open the webcam at ``index`` (or the configured default).

        Raises:
            RuntimeError: If the device cannot be opened.
        """
        if index is not None:
            self._index = index
        self.release()
        logger.info("Opening camera %s", self._index)
        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            logger.error("Failed to open camera %s", self._index)
            raise RuntimeError(f"Unable to open camera {self._index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        self._capture = capture
        logger.info(
            "Camera %s opened (%sx%s @ %s fps)",
            self._index,
            self._config.width,
            self._config.height,
            self._config.fps,
        )

    def read(self) -> "Optional[np.ndarray]":
        """Grab a single frame, applying horizontal flip if configured.

        Returns:
            The BGR frame, or ``None`` if no frame is available.
        """
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        if self._config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    def set_resolution(self, width: int, height: int) -> None:
        """Change the capture resolution on the fly."""
        if self._capture is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info("Camera resolution set to %sx%s", width, height)

    def is_open(self) -> bool:
        """Return whether the device is currently open."""
        return self._capture is not None and self._capture.isOpened()

    def release(self) -> None:
        """Release the device if open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Camera released")

    def list_available(self) -> List[int]:
        """Probe camera indices and return those that can be opened."""
        available: List[int] = []
        for i in range(self._config.max_scan_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
            cap.release()
        logger.info("Available cameras: %s", available)
        return available
