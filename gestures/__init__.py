"""Gesture-recognition layer.

Phase 1 ships only the abstract interfaces so the rest of the app can depend on
stable contracts. Concrete analyzers (finger states, orientation, rule-based
and ML classifiers) are implemented in later phases.
"""
from gestures.base import GestureResult, GestureRecognizer

__all__ = ["GestureResult", "GestureRecognizer"]
