"""Reusable information panels shown alongside the video feed.

Each panel is a self-contained ``QGroupBox`` with a small ``update_*`` API. In
Phase 1 the Hand Information and Statistics panels are fully live; the Finger
States and Detected Gesture panels are present but display placeholders until
their analyzers land in later phases.
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from vision.landmarks import HandResult

_FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")


class StatisticsPanel(QGroupBox):
    """Live performance & detection statistics."""

    def __init__(self) -> None:
        super().__init__("Statistics")
        layout = QVBoxLayout(self)
        self._fps = QLabel("FPS: 0.0")
        self._hands = QLabel("Hands detected: 0")
        self._resolution = QLabel("Resolution: -")
        for label in (self._fps, self._hands, self._resolution):
            layout.addWidget(label)
        layout.addStretch(1)

    def update_fps(self, fps: float) -> None:
        self._fps.setText(f"FPS: {fps:.1f}")

    def update_hands(self, count: int) -> None:
        self._hands.setText(f"Hands detected: {count}")

    def update_resolution(self, width: int, height: int) -> None:
        self._resolution.setText(f"Resolution: {width} x {height}")


class HandInfoPanel(QGroupBox):
    """Handedness, confidence, and landmark counts for detected hands."""

    def __init__(self) -> None:
        super().__init__("Hand Information")
        layout = QVBoxLayout(self)
        self._body = QLabel("No hand detected")
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body.setWordWrap(True)
        self._body.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._body)
        layout.addStretch(1)

    def update_hands(self, hands: List[HandResult]) -> None:
        if not hands:
            self._body.setText("No hand detected")
            return
        blocks = []
        for i, hand in enumerate(hands, start=1):
            blocks.append(
                f"<b>Hand {i}: {hand.handedness}</b><br>"
                f"Confidence: {hand.score * 100:.0f}%<br>"
                f"Landmarks: {len(hand.landmarks)} / 21"
            )
        self._body.setText("<br><br>".join(blocks))


class FingerStatePanel(QGroupBox):
    """Per-finger extended/folded states (populated in Phase 2)."""

    def __init__(self) -> None:
        super().__init__("Finger States")
        layout = QVBoxLayout(self)
        self._labels = {}
        for name in _FINGER_NAMES:
            label = QLabel(f"{name}: \u2014")
            self._labels[name] = label
            layout.addWidget(label)
        layout.addStretch(1)

    def update_states(self, states: dict) -> None:
        """Update finger labels from a ``{finger: 'Extended'|'Folded'}`` mapping."""
        for name in _FINGER_NAMES:
            value = states.get(name, "\u2014")
            self._labels[name].setText(f"{name}: {value}")

    def reset(self) -> None:
        for name in _FINGER_NAMES:
            self._labels[name].setText(f"{name}: \u2014")


class GesturePanel(QGroupBox):
    """Detected gesture + confidence (populated in Phase 3)."""

    def __init__(self) -> None:
        super().__init__("Detected Gesture")
        layout = QVBoxLayout(self)
        self._gesture = QLabel("\u2014")
        self._gesture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gesture.setStyleSheet("font-size: 20px; font-weight: 700; color: #8ab4f8;")
        self._confidence = QLabel("Confidence: \u2014")
        self._confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._gesture)
        layout.addWidget(self._confidence)
        layout.addStretch(1)

    def update_gesture(self, name: str, confidence: float) -> None:
        self._gesture.setText(name)
        self._confidence.setText(f"Confidence: {confidence * 100:.0f}%")

    def reset(self) -> None:
        self._gesture.setText("\u2014")
        self._confidence.setText("Confidence: \u2014")


class LogPanel(QGroupBox):
    """Read-only view mirroring the application log."""

    def __init__(self, max_lines: int = 500) -> None:
        super().__init__("Logs")
        layout = QVBoxLayout(self)
        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(max_lines)
        layout.addWidget(self._view)

    def append(self, message: str) -> None:
        self._view.appendPlainText(message)
