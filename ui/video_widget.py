"""Widget that renders BGR frames coming from the capture thread."""
from __future__ import annotations

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    """A ``QLabel`` specialized for displaying OpenCV frames."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("VideoLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("Camera stopped\n\nPress \u201cStart Camera\u201d to begin")

    def show_frame(self, frame) -> None:
        """Convert a BGR numpy frame to a scaled pixmap and display it."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def clear_frame(self) -> None:
        """Reset to the idle placeholder message."""
        self.clear()
        self.setText("Camera stopped\n\nPress \u201cStart Camera\u201d to begin")
