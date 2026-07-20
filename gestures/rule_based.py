"""Rule-based gesture recognizer.

A deterministic, dependency-light recognizer built on top of the Phase 2
:class:`~vision.analysis.HandAnalyzer`. It needs no training and no ML
libraries, so gesture names appear live the moment the app starts. It also
serves as a sensible fallback whenever a trained model is unavailable.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from gestures.base import GestureRecognizer, GestureResult
from gestures import labels as L
from utils.geometry import euclidean_distance
from vision.analysis import HandAnalyzer
from vision.landmarks import HandLandmark as LM
from vision.landmarks import HandResult

_EPS = 1e-6


class RuleBasedRecognizer(GestureRecognizer):
    """Classify a hand from its finger states, orientation, and pinch distance.

    Args:
        analysis_config: Optional analysis configuration forwarded to the
            internal :class:`HandAnalyzer` (finger straight/curl thresholds).
        pinch_ratio: Thumb-tip/index-tip distance (as a fraction of hand size)
            below which the thumb and index are considered touching.
    """

    def __init__(self, analysis_config=None, pinch_ratio: float = 0.35) -> None:
        self._analyzer = HandAnalyzer(analysis_config)
        self._pinch_ratio = pinch_ratio

    @property
    def labels(self) -> List[str]:
        return list(L.RULE_BASED_GESTURES)

    def is_ready(self) -> bool:
        return True

    def predict(self, hand: HandResult) -> GestureResult:
        start = time.perf_counter()
        name, confidence = self._classify(hand)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return GestureResult(
            name=name,
            confidence=confidence,
            probabilities={name: confidence},
            inference_ms=elapsed_ms,
        )

    # -- internals ---------------------------------------------------------
    def _pinch_distance(self, hand: HandResult) -> float:
        scale = euclidean_distance(
            hand.landmark(LM.WRIST).as_tuple(),
            hand.landmark(LM.MIDDLE_MCP).as_tuple(),
        )
        scale = scale if scale > _EPS else _EPS
        tip_gap = euclidean_distance(
            hand.landmark(LM.THUMB_TIP).as_tuple(),
            hand.landmark(LM.INDEX_TIP).as_tuple(),
        )
        return tip_gap / scale

    def _classify(self, hand: HandResult) -> Tuple[str, float]:
        analysis = self._analyzer.analyze(hand)
        fs = analysis.finger_states.states
        thumb = fs["Thumb"]
        index = fs["Index"]
        middle = fs["Middle"]
        ring = fs["Ring"]
        pinky = fs["Pinky"]
        count = analysis.finger_states.count_extended
        pointing = analysis.orientation.pointing
        pinch = self._pinch_distance(hand)

        # A fully folded or fully open hand is unambiguous; resolve it first so
        # a closed fist (whose thumb/index tips are near each other) is never
        # mistaken for a pinch.
        if count == 0:
            return L.FIST, 0.95
        if count == 5:
            return L.OPEN_HAND, 0.95

        # Thumb + index touching: OK (other three up) or a bare Pinch.
        if pinch < self._pinch_ratio:
            if middle and ring and pinky and not index:
                return L.OK, 0.90
            if index and not middle and not ring and not pinky:
                return L.PINCH, 0.85

        only = lambda *fingers: all(fingers) and count == sum(bool(f) for f in fingers)

        if thumb and only(thumb):
            return (L.THUMBS_DOWN, 0.90) if pointing == "Down" else (L.THUMBS_UP, 0.90)
        if index and not middle and not ring and not pinky and not thumb:
            return L.POINTING, 0.90
        if index and middle and not ring and not pinky and not thumb:
            return L.PEACE, 0.90
        if thumb and pinky and not index and not middle and not ring:
            return L.CALL_ME, 0.90
        if index and middle and ring and not pinky and not thumb:
            return L.THREE, 0.85
        if index and middle and ring and pinky and not thumb:
            return L.FOUR, 0.85
        return L.UNKNOWN, 0.30


__all__ = ["RuleBasedRecognizer"]
