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

from utils.logger import get_logger
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
    # Reserved for later phases (populated once analyzers are added):
    # descriptions, gestures, orientations, measurements ...


class FrameProcessor:
    """Run the per-frame vision pipeline.

    Args:
        config: The full application configuration.
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self._tracker = HandTracker(config.hand_tracking)
        self._painter = LandmarkPainter(config.display)
        self._last_hand_count = 0

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
        annotated = self._painter.draw(frame, hands)
        return ProcessResult(frame=annotated, hands=hands)

    def close(self) -> None:
        """Release owned resources."""
        self._tracker.close()
