"""Rendering overlays (skeleton, landmarks, bounding box, IDs) onto frames.

The :class:`LandmarkPainter` draws directly on the BGR frame using OpenCV. It
is driven entirely by :class:`~config.settings.DisplayConfig` so overlays can
be toggled from the configuration file or the UI at runtime.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Tuple

import cv2

from vision.landmarks import HAND_CONNECTIONS, HandResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import DisplayConfig

# BGR colors
_COLOR_LANDMARK: Tuple[int, int, int] = (0, 255, 0)
_COLOR_CONNECTION: Tuple[int, int, int] = (255, 200, 0)
_COLOR_BBOX: Tuple[int, int, int] = (0, 165, 255)
_COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)
_COLOR_LABEL_BG: Tuple[int, int, int] = (0, 0, 0)


class LandmarkPainter:
    """Draw hand overlays according to a :class:`DisplayConfig`."""

    def __init__(self, config: "DisplayConfig") -> None:
        self._config = config

    def draw(self, frame, hands: Iterable[HandResult]):
        """Draw every enabled overlay for all hands. Mutates and returns frame."""
        height, width = frame.shape[:2]
        for hand in hands:
            if self._config.draw_bounding_box:
                self._draw_bounding_box(frame, hand, width, height)
            if self._config.draw_skeleton:
                self._draw_skeleton(frame, hand, width, height)
            if self._config.draw_landmarks:
                self._draw_landmarks(frame, hand, width, height)
        return frame

    def _draw_skeleton(self, frame, hand: HandResult, width: int, height: int) -> None:
        for start, end in HAND_CONNECTIONS:
            p1 = hand.pixel(start, width, height)
            p2 = hand.pixel(end, width, height)
            cv2.line(frame, p1, p2, _COLOR_CONNECTION, self._config.connection_thickness)

    def _draw_landmarks(self, frame, hand: HandResult, width: int, height: int) -> None:
        radius = self._config.landmark_radius
        for idx, landmark in enumerate(hand.landmarks):
            point = landmark.to_pixel(width, height)
            cv2.circle(frame, point, radius, _COLOR_LANDMARK, -1)
            if self._config.draw_landmark_ids:
                cv2.putText(
                    frame,
                    str(idx),
                    (point[0] + 4, point[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    _COLOR_TEXT,
                    1,
                    cv2.LINE_AA,
                )

    def _draw_bounding_box(self, frame, hand: HandResult, width: int, height: int) -> None:
        x_min, y_min, x_max, y_max = hand.bounding_box(width, height)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), _COLOR_BBOX, 2)
        label = f"{hand.handedness} {hand.score * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (x_min, y_min - th - 8),
            (x_min + tw + 6, y_min),
            _COLOR_LABEL_BG,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x_min + 3, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )


def draw_fps(frame, fps: float) -> None:
    """Draw an FPS counter in the top-left corner of the frame."""
    text = f"FPS: {fps:5.1f}"
    cv2.putText(
        frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA
    )
