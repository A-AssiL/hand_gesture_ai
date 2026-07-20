"""Canonical gesture label vocabulary.

A single source of truth for gesture names so the rule-based recognizer, the
ML classifier, the trainer, and the UI all agree on spelling and ordering.
"""
from __future__ import annotations

from typing import List

# Sentinel returned when no known gesture matches.
UNKNOWN = "Unknown"

# Individual named constants (handy for imports and to avoid typos).
FIST = "Fist"
OPEN_HAND = "Open Hand"
PEACE = "Peace"
THUMBS_UP = "Thumbs Up"
THUMBS_DOWN = "Thumbs Down"
OK = "OK"
PINCH = "Pinch"
POINTING = "Pointing"
CALL_ME = "Call Me"
THREE = "Three"
FOUR = "Four"

# Gestures the dependency-light rule-based recognizer can output.
RULE_BASED_GESTURES: List[str] = [
    FIST,
    OPEN_HAND,
    PEACE,
    THUMBS_UP,
    THUMBS_DOWN,
    OK,
    PINCH,
    POINTING,
    CALL_ME,
    THREE,
    FOUR,
]

# Gestures the bundled synthetic dataset generates, i.e. the default classes a
# freshly trained Random Forest model will learn. (Pinch is intentionally left
# to the rule-based path and the real Phase 4 dataset recorder.)
SYNTHETIC_GESTURES: List[str] = [
    FIST,
    OPEN_HAND,
    PEACE,
    THUMBS_UP,
    THUMBS_DOWN,
    OK,
    POINTING,
    CALL_ME,
    THREE,
    FOUR,
]

# Common aliases from the specification, mapped onto the canonical labels above.
# Kept for documentation and for a future gesture-mapping/config layer.
ALIASES = {
    "Rock": FIST,
    "Paper": OPEN_HAND,
    "Scissors": PEACE,
    "Victory": PEACE,
    "Stop": OPEN_HAND,
    "Five": OPEN_HAND,
    "One": POINTING,
    "Two": PEACE,
}

__all__ = [
    "UNKNOWN",
    "FIST",
    "OPEN_HAND",
    "PEACE",
    "THUMBS_UP",
    "THUMBS_DOWN",
    "OK",
    "PINCH",
    "POINTING",
    "CALL_ME",
    "THREE",
    "FOUR",
    "RULE_BASED_GESTURES",
    "SYNTHETIC_GESTURES",
    "ALIASES",
]
