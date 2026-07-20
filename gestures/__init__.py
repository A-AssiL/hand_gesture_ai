"""Gesture-recognition layer.

Exposes the stable interfaces plus the dependency-light building blocks (feature
extraction, canonical labels, and the rule-based recognizer). The ML recognizer
is exported lazily via :func:`__getattr__` so importing this package never
requires scikit-learn/joblib to be installed.
"""
from gestures import labels
from gestures.base import GestureRecognizer, GestureResult
from gestures.features import FEATURE_DIM, FEATURE_NAMES, extract_batch, extract_features
from gestures.rule_based import RuleBasedRecognizer

__all__ = [
    "GestureResult",
    "GestureRecognizer",
    "RuleBasedRecognizer",
    "RandomForestRecognizer",
    "extract_features",
    "extract_batch",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "labels",
]


def __getattr__(name: str):
    """Lazily import the ML recognizer only when first accessed."""
    if name == "RandomForestRecognizer":
        from gestures.ml_classifier import RandomForestRecognizer

        return RandomForestRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
