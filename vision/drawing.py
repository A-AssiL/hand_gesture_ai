"""Rendering overlays (skeleton, landmarks, bounding box, IDs) onto frames.

The :class:`LandmarkPainter` draws directly on the BGR frame using OpenCV. It
is driven entirely by :class:`~config.settings.DisplayConfig` so overlays can
be toggled from the configuration file or the UI at runtime.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Tuple

import cv2

from vision.landmarks import HAND_CONNECTIONS, HandResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import DisplayConfig
    from gestures.base import GestureResult
    from vision.analysis import HandAnalysis

# BGR colors
_COLOR_LANDMARK: Tuple[int, int, int] = (0, 255, 0)
_COLOR_CONNECTION: Tuple[int, int, int] = (255, 200, 0)
_COLOR_BBOX: Tuple[int, int, int] = (0, 165, 255)
_COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)
_COLOR_LABEL_BG: Tuple[int, int, int] = (0, 0, 0)
_COLOR_ANALYSIS: Tuple[int, int, int] = (0, 255, 255)
_COLOR_GESTURE: Tuple[int, int, int] = (0, 220, 120)


class LandmarkPainter:
    """Draw hand overlays according to a :class:`DisplayConfig`."""

    def __init__(self, config: "DisplayConfig") -> None:
        self._config = config

    def draw(
        self,
        frame,
        hands: Iterable[HandResult],
        analyses: "Optional[Sequence[HandAnalysis]]" = None,
        gestures: "Optional[Sequence[GestureResult]]" = None,
    ):
        """Draw every enabled overlay for all hands. Mutates and returns frame.

        Args:
            frame: BGR image to draw on (mutated in place).
            hands: Detected hands to render.
            analyses: Optional per-hand analysis (same order/length as ``hands``);
                when present, a compact summary is drawn beneath each box.
            gestures: Optional per-hand gesture results; when present, the
                recognized gesture name is drawn above each box.
        """
        height, width = frame.shape[:2]
        hand_list = list(hands)
        analysis_list = list(analyses) if analyses else []
        gesture_list = list(gestures) if gestures else []
        for i, hand in enumerate(hand_list):
            if self._config.draw_bounding_box:
                self._draw_bounding_box(frame, hand, width, height)
            if self._config.draw_skeleton:
                self._draw_skeleton(frame, hand, width, height)
            if self._config.draw_landmarks:
                self._draw_landmarks(frame, hand, width, height)
            if i < len(analysis_list):
                self._draw_analysis(frame, hand, analysis_list[i], width, height)
            if i < len(gesture_list):
                self._draw_gesture(frame, hand, gesture_list[i], width, height)
        return frame

    def _draw_gesture(
        self, frame, hand: HandResult, gesture, width: int, height: int
    ) -> None:
        """Draw the recognized gesture name above the hand's bounding box."""
        x_min, y_min, _, _ = hand.bounding_box(width, height)
        label = f"{gesture.name} {gesture.confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(y_min - 26, th + 4)
        cv2.rectangle(
            frame, (x_min, y - th - 6), (x_min + tw + 8, y + 4), _COLOR_LABEL_BG, -1
        )
        cv2.putText(
            frame,
            label,
            (x_min + 4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _COLOR_GESTURE,
            2,
            cv2.LINE_AA,
        )

    def _draw_analysis(
        self, frame, hand: HandResult, analysis: "HandAnalysis", width: int, height: int
    ) -> None:
        """Draw a compact analysis summary beneath the hand's bounding box."""
        _, _, x_max, y_max = hand.bounding_box(width, height)
        x_min = hand.bounding_box(width, height)[0]
        text = (
            f"{analysis.finger_states.count_extended}/5 up | "
            f"{analysis.orientation.facing} | {analysis.orientation.pointing}"
        )
        cv2.putText(
            frame,
            text,
            (x_min, min(y_max + 18, height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _COLOR_ANALYSIS,
            1,
            cv2.LINE_AA,
        )

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
