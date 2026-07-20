"""Tests for the gesture feature extractor (numpy only, no ML libs)."""
from __future__ import annotations

import math

import numpy as np

from gestures.features import FEATURE_DIM, FEATURE_NAMES, extract_batch, extract_features
from training.synthetic import make_gesture_hand
from vision.landmarks import HandLandmark as LM


def test_feature_dimension() -> None:
    assert FEATURE_DIM == 44
    assert len(FEATURE_NAMES) == 44
    assert FEATURE_NAMES[-2:] == ["orient_sin", "orient_cos"]


def test_wrist_maps_to_origin() -> None:
    hand = make_gesture_hand("Open Hand", jitter=0.0)
    feats = extract_features(hand)
    wrist_x = feats[2 * int(LM.WRIST)]
    wrist_y = feats[2 * int(LM.WRIST) + 1]
    assert abs(wrist_x) < 1e-9
    assert abs(wrist_y) < 1e-9


def test_rotation_invariant_shape() -> None:
    # Same gesture at two rotations must yield identical shape features.
    from training.synthetic import make_hand, _PATTERNS

    a = make_hand(_PATTERNS["Peace"], rotation_deg=0.0, jitter=0.0)
    b = make_hand(_PATTERNS["Peace"], rotation_deg=40.0, jitter=0.0)
    fa = extract_features(a)[:42]
    fb = extract_features(b)[:42]
    assert np.allclose(fa, fb, atol=1e-6)


def test_orientation_features_differ() -> None:
    # Thumbs Up vs Thumbs Down share a shape but differ in absolute orientation.
    up = extract_features(make_gesture_hand("Thumbs Up", jitter=0.0))
    down = extract_features(make_gesture_hand("Thumbs Down", jitter=0.0))
    assert not np.allclose(up[-2:], down[-2:], atol=1e-3)


def test_determinism() -> None:
    hand = make_gesture_hand("Fist", jitter=0.0)
    assert np.allclose(extract_features(hand), extract_features(hand))


def test_extract_batch_shape() -> None:
    hands = [make_gesture_hand("Fist", jitter=0.0), make_gesture_hand("Peace", jitter=0.0)]
    batch = extract_batch(hands)
    assert batch.shape == (2, FEATURE_DIM)
    assert extract_batch([]).shape == (0, FEATURE_DIM)
