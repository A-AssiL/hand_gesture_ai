"""Machine-learning gesture recognizer (scikit-learn Random Forest).

The heavy dependencies (scikit-learn, joblib) are imported lazily inside the
methods that need them, so importing this module never fails even in an
environment where they are not installed. This keeps the app importable and
lets the rule-based recognizer act as a fallback.

A saved model is a joblib "bundle" dict with keys:
    ``model``       - the fitted estimator (exposes ``predict_proba``)
    ``labels``      - ordered class labels
    ``feature_dim`` - expected feature-vector length
    ``metrics``     - optional training-metrics dict
    ``created_at``  - ISO timestamp
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from gestures.base import GestureRecognizer, GestureResult
from gestures.features import FEATURE_DIM, extract_features
from gestures.labels import UNKNOWN
from utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestRecognizer(GestureRecognizer):
    """Predict gestures with a trained scikit-learn classifier.

    Args:
        model_path: Optional path to a joblib model bundle to load immediately.
        min_confidence: Predictions below this probability are reported as
            ``"Unknown"`` (0 disables the threshold).
    """

    def __init__(
        self, model_path: Optional[str] = None, min_confidence: float = 0.0
    ) -> None:
        self._model = None
        self._labels: List[str] = []
        self._feature_dim = FEATURE_DIM
        self._min_confidence = float(min_confidence)
        self._metrics: dict = {}
        if model_path:
            self.load(model_path)

    # -- lifecycle ---------------------------------------------------------
    def load(self, model_path: str) -> bool:
        """Load a joblib model bundle. Returns True on success."""
        path = Path(model_path)
        if not path.exists():
            logger.warning("Model file not found: %s", path)
            return False
        try:
            import joblib  # lazy: only needed when actually loading a model

            bundle = joblib.load(path)
            self._model = bundle["model"]
            self._labels = list(bundle.get("labels", []))
            self._feature_dim = int(bundle.get("feature_dim", FEATURE_DIM))
            self._metrics = dict(bundle.get("metrics", {}))
            logger.info(
                "Loaded gesture model '%s' (%d classes)", path.name, len(self._labels)
            )
            return True
        except Exception as exc:  # pragma: no cover - depends on env/model file
            logger.error("Failed to load model %s: %s", path, exc)
            self._model = None
            return False

    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)

    # -- inference ---------------------------------------------------------
    def predict(self, hand: HandResult) -> GestureResult:  # type: ignore[name-defined]
        if not self.is_ready():
            return GestureResult(name=UNKNOWN, confidence=0.0)
        start = time.perf_counter()
        features = extract_features(hand).reshape(1, -1)
        proba = self._model.predict_proba(features)[0]
        classes = [str(c) for c in self._model.classes_]
        probabilities = {c: float(p) for c, p in zip(classes, proba)}
        best = int(np.argmax(proba))
        name = classes[best]
        confidence = float(proba[best])
        if confidence < self._min_confidence:
            name = UNKNOWN
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return GestureResult(
            name=name,
            confidence=confidence,
            probabilities=probabilities,
            inference_ms=elapsed_ms,
        )


# Imported here (not at top) purely for the type hint above; avoids an unused
# import lint warning while keeping the runtime dependency graph minimal.
from vision.landmarks import HandResult  # noqa: E402

__all__ = ["RandomForestRecognizer"]
