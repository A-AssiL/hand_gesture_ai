"""The frame-processing pipeline that ties vision components together.

:class:`FrameProcessor` is deliberately UI-agnostic: it takes a raw BGR frame
and returns an annotated frame plus the structured detection results. This keeps
the Qt layer thin and makes the pipeline easy to unit-test or reuse headless.

Phase 1 pipeline::

    frame -> HandTracker -> LandmarkPainter -> annotated frame + HandResult[]

Later phases plug additional analyzers (finger states, orientation,
measurements, gesture classifier) into :meth:`FrameProcessor.process` without
changing its public contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from gestures.base import GestureRecognizer, GestureResult
from gestures.rule_based import RuleBasedRecognizer
from utils.logger import get_logger
from vision.analysis import HandAnalysis, HandAnalyzer
from vision.drawing import LandmarkPainter
from vision.hand_tracker import HandTracker
from vision.landmarks import HandResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from config.settings import Config

logger = get_logger("engine")


@dataclass
class ProcessResult:
    """Structured output of a single processing step."""

    frame: "np.ndarray"                       # annotated BGR frame
    hands: List[HandResult] = field(default_factory=list)
    analyses: List[HandAnalysis] = field(default_factory=list)  # one per hand
    gestures: List[GestureResult] = field(default_factory=list)  # one per hand


class FrameProcessor:
    """Run the per-frame vision pipeline.

    Args:
        config: The full application configuration.
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self._tracker = HandTracker(config.hand_tracking)
        self._painter = LandmarkPainter(config.display)
        self._analyzer = HandAnalyzer(config.analysis)
        self._analysis_enabled = config.analysis.enabled
        self._gesture_enabled = config.gesture.enabled
        self._recognizer: GestureRecognizer = self._build_recognizer()
        self._last_hand_count = 0

    def _build_recognizer(self) -> GestureRecognizer:
        """Create the configured recognizer, falling back to the rule-based one.

        A trained Random Forest is used only when the backend is selected *and*
        its model file exists; otherwise the always-available rule-based
        recognizer is returned so gestures work with zero setup.
        """
        cfg = self._config.gesture
        rule_based = RuleBasedRecognizer(self._config.analysis)
        if cfg.backend == "random_forest" and cfg.model_path:
            from pathlib import Path

            if Path(cfg.model_path).exists():
                try:
                    from gestures.ml_classifier import RandomForestRecognizer

                    model = RandomForestRecognizer(
                        cfg.model_path, min_confidence=cfg.min_confidence
                    )
                    if model.is_ready():
                        logger.info("Gesture backend: Random Forest (%s)", cfg.model_path)
                        return model
                except Exception as exc:  # pragma: no cover - depends on env
                    logger.error("Falling back to rule-based recognizer: %s", exc)
        logger.info("Gesture backend: rule-based")
        return rule_based

    def set_recognizer(self, recognizer: GestureRecognizer) -> None:
        """Swap the active recognizer at runtime (e.g. after training/loading)."""
        self._recognizer = recognizer
        self._gesture_enabled = True

    def load_model(self, model_path: str) -> bool:
        """Load a trained model and make it the active recognizer."""
        from gestures.ml_classifier import RandomForestRecognizer

        model = RandomForestRecognizer(
            model_path, min_confidence=self._config.gesture.min_confidence
        )
        if model.is_ready():
            self.set_recognizer(model)
            return True
        return False

    def use_rule_based(self) -> None:
        """Revert to the dependency-free rule-based recognizer."""
        self.set_recognizer(RuleBasedRecognizer(self._config.analysis))

    def process(self, frame) -> ProcessResult:
        """Detect hands in ``frame`` and draw overlays.

        Args:
            frame: A raw BGR frame from the camera.

        Returns:
            A :class:`ProcessResult` with the annotated frame and detections.
        """
        hands = self._tracker.process(frame)
        if len(hands) != self._last_hand_count:
            logger.info("Hands detected: %s", len(hands))
            self._last_hand_count = len(hands)
        analyses = (
            [self._analyzer.analyze(hand) for hand in hands]
            if self._analysis_enabled
            else []
        )
        gestures = (
            [self._recognizer.predict(hand) for hand in hands]
            if self._gesture_enabled
            else []
        )
        annotated = self._painter.draw(frame, hands, analyses, gestures)
        return ProcessResult(
            frame=annotated, hands=hands, analyses=analyses, gestures=gestures
        )

    def close(self) -> None:
        """Release owned resources."""
        self._tracker.close()
