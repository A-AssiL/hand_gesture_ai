"""Tests for the rule-based recognizer using synthetic hands (numpy only)."""
from __future__ import annotations

import pytest

from gestures.rule_based import RuleBasedRecognizer
from training.synthetic import make_gesture_hand


@pytest.mark.parametrize(
    "gesture",
    ["Fist", "Open Hand", "Peace", "Thumbs Up", "Pointing", "Call Me", "Three", "Four"],
)
def test_recognizes_core_gestures(gesture: str) -> None:
    recognizer = RuleBasedRecognizer()
    hand = make_gesture_hand(gesture, jitter=0.0)
    result = recognizer.predict(hand)
    assert result.name == gesture, f"expected {gesture}, got {result.name}"


def test_recognizer_is_ready_without_training() -> None:
    assert RuleBasedRecognizer().is_ready() is True


def test_result_has_probabilities() -> None:
    recognizer = RuleBasedRecognizer()
    result = recognizer.predict(make_gesture_hand("Fist", jitter=0.0))
    assert 0.0 <= result.confidence <= 1.0
    assert result.name in result.probabilities
    assert result.inference_ms >= 0.0
