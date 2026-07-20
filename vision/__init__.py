"""Computer-vision layer: hand tracking, landmark models, and drawing.

The landmark data models are dependency-light and imported eagerly. The
OpenCV/MediaPipe-backed ``HandTracker`` and ``LandmarkPainter`` are exposed
lazily so importing this package never forces those heavy dependencies.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from vision.analysis import (
    FINGER_NAMES,
    FingerStates,
    HandAnalysis,
    HandAnalyzer,
    HandMeasurements,
    HandOrientation,
)
from vision.landmarks import HAND_CONNECTIONS, HandLandmark, HandResult, Landmark

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vision.drawing import LandmarkPainter, draw_fps
    from vision.hand_tracker import HandTracker

__all__ = [
    "HAND_CONNECTIONS",
    "HandLandmark",
    "HandResult",
    "Landmark",
    "FINGER_NAMES",
    "FingerStates",
    "HandAnalysis",
    "HandAnalyzer",
    "HandMeasurements",
    "HandOrientation",
    "HandTracker",
    "LandmarkPainter",
    "draw_fps",
]


def __getattr__(name: str):
    """Lazily resolve OpenCV/MediaPipe-backed exports (PEP 562)."""
    if name == "HandTracker":
        from vision.hand_tracker import HandTracker

        return HandTracker
    if name in {"LandmarkPainter", "draw_fps"}:
        from vision import drawing

        return getattr(drawing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
