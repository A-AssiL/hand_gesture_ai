"""Data structures for MediaPipe hand landmarks.

These classes are framework-agnostic (no OpenCV / MediaPipe imports) so they
can be reused by the UI, measurement, and ML layers and unit-tested in
isolation. MediaPipe returns 21 landmarks per hand; see :class:`HandLandmark`
for the canonical indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple


class HandLandmark(IntEnum):
    """Canonical MediaPipe Hands landmark indices (0-20)."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Edges connecting landmarks to render the hand skeleton.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # palm base
)


@dataclass
class Landmark:
    """A single normalized 3D landmark.

    Coordinates are normalized to the frame: ``x`` and ``y`` are in [0, 1]
    relative to the image width/height, and ``z`` is a relative depth (smaller
    is closer to the camera).
    """

    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return ``(x, y, z)``."""
        return (self.x, self.y, self.z)

    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Return integer pixel coordinates for the given frame size."""
        return int(self.x * width), int(self.y * height)


@dataclass
class HandResult:
    """A detected hand: handedness, confidence, and its 21 landmarks."""

    handedness: str  # "Left" or "Right"
    score: float     # detection/handedness confidence in [0, 1]
    landmarks: List[Landmark]

    def landmark(self, index: HandLandmark | int) -> Landmark:
        """Return the landmark at the given index."""
        return self.landmarks[int(index)]

    def pixel(self, index: HandLandmark | int, width: int, height: int) -> Tuple[int, int]:
        """Return pixel coordinates of a landmark for a frame of given size."""
        return self.landmark(index).to_pixel(width, height)

    def bounding_box(
        self, width: int, height: int, padding: int = 20
    ) -> Tuple[int, int, int, int]:
        """Return the pixel bounding box ``(x_min, y_min, x_max, y_max)``.

        The box is padded and clamped to the frame boundaries.
        """
        xs = [lm.x for lm in self.landmarks]
        ys = [lm.y for lm in self.landmarks]
        x_min = max(int(min(xs) * width) - padding, 0)
        y_min = max(int(min(ys) * height) - padding, 0)
        x_max = min(int(max(xs) * width) + padding, width)
        y_max = min(int(max(ys) * height) + padding, height)
        return x_min, y_min, x_max, y_max
