"""Abstract interfaces for gesture recognition.

Defining these contracts now lets Phase 1 stay decoupled from any specific
recognition strategy. Later phases provide concrete implementations:

* ``RuleBasedRecognizer`` - finger-state heuristics (Phase 2/3)
* ``RandomForestRecognizer`` - scikit-learn on landmark features (Phase 3)
* ``TorchRecognizer`` - neural network classifier (future)

All implementations return a :class:`GestureResult`, so the UI and pipeline
never need to know which backend produced the prediction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from vision.landmarks import HandResult


@dataclass
class GestureResult:
    """The outcome of classifying a single hand."""

    name: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    inference_ms: float = 0.0


class GestureRecognizer(ABC):
    """Common interface every gesture backend must implement."""

    @abstractmethod
    def predict(self, hand: HandResult) -> GestureResult:
        """Classify a single hand into a gesture."""
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """Return the ordered list of gesture labels this backend can output."""
        raise NotImplementedError

    def is_ready(self) -> bool:
        """Return whether the recognizer is loaded and ready to predict."""
        return True
