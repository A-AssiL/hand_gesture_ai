"""Thin wrapper around MediaPipe Hands.
​
Converts MediaPipe's raw output into the framework-agnostic
:class:`~vision.landmarks.HandResult` objects used throughout the app.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import cv2

from utils.logger import get_logger
from vision.landmarks import HandResult, Landmark

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import HandTrackingConfig

logger = get_logger("hand_tracker")


def _load_mediapipe_hands():
    """Return MediaPipe's ``hands`` solutions module, robustly.

    MediaPipe exposes the classic graph API as ``mediapipe.solutions.hands``.
    On some installs the ``solutions`` attribute is lazily loaded and can be
    missing (incompatible Python version, numpy 2.x, or a broken/partial
    install). We try the attribute first, then the explicit submodule path,
    and finally raise a clear, actionable error.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - install guidance
        raise ImportError(
            "MediaPipe is not installed. Install it with:\n"
            "    pip install \"mediapipe>=0.10.14\""
        ) from exc

    # Preferred: the public attribute.
    hands_module = getattr(getattr(mp, "solutions", None), "hands", None)
    if hands_module is not None:
        return hands_module

    # Fallback: import the concrete submodule directly.
    try:
        from mediapipe.python.solutions import hands as hands_module  # type: ignore
        return hands_module
    except Exception as exc:  # noqa: BLE001 - surface a helpful message
        raise ImportError(
            "MediaPipe is installed but its 'solutions' API failed to load "
            "(module 'mediapipe' has no attribute 'solutions').\n"
            "This is an environment issue, not an application bug. Please check:\n"
            "  1. Python version: use 3.10-3.12. MediaPipe does not yet ship "
            "wheels for every newer version (e.g. 3.13). Run 'python --version'.\n"
            "  2. NumPy version: MediaPipe needs NumPy < 2.0. Run "
            "'pip install \"numpy<2\"'.\n"
            "  3. Reinstall cleanly: "
            "'pip install --force-reinstall --no-cache-dir \"mediapipe>=0.10.14\"'.\n"
            "  4. Make sure no local file or folder named 'mediapipe.py' / "
            "'mediapipe' shadows the installed package."
        ) from exc


class HandTracker:
    """Detect and track up to ``max_num_hands`` hands in a BGR frame.

    The tracker owns a single MediaPipe ``Hands`` graph. Call :meth:`close`
    (or use it as a context manager) to release native resources.
    """

    def __init__(self, config: "HandTrackingConfig") -> None:
        self._config = config
        self._mp_hands = _load_mediapipe_hands()
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_num_hands,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        logger.info(
            "HandTracker initialized (max_num_hands=%s, complexity=%s)",
            config.max_num_hands,
            config.model_complexity,
        )

    def process(self, frame_bgr) -> List[HandResult]:
        """Run detection on a single BGR frame.

        Args:
            frame_bgr: An OpenCV BGR image (``numpy.ndarray``).

        Returns:
            A list of :class:`HandResult`, one per detected hand (may be empty).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # perf: allow MediaPipe to work zero-copy
        results = self._hands.process(rgb)

        hands: List[HandResult] = []
        if not results.multi_hand_landmarks:
            return hands

        handedness_list = results.multi_handedness or []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label, score = "Unknown", 0.0
            if idx < len(handedness_list):
                classification = handedness_list[idx].classification[0]
                label = classification.label
                score = float(classification.score)
            landmarks = [Landmark(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            hands.append(HandResult(handedness=label, score=score, landmarks=landmarks))
        return hands

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._hands.close()
        logger.info("HandTracker closed")

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
