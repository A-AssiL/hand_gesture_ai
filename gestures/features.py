"""Landmark -> feature-vector conversion for ML gesture classification.

The feature vector is designed to be robust for a small-data classifier such as
a Random Forest:

* **Translation invariant** - all points are expressed relative to the wrist.
* **Scale invariant** - divided by the wrist -> middle-finger-MCP distance, so
  the hand's distance from the camera does not matter.
* **Rotation normalized (shape)** - the 42 shape features are rotated so the
  wrist -> middle-MCP axis is canonical, meaning the *shape* of a gesture reads
  the same at any in-plane rotation.
* **Handedness normalized** - left hands are mirrored onto the right-hand frame
  so the same gesture maps to the same region of feature space.
* **Absolute orientation** - two extra features (sin/cos of the raw hand angle)
  are appended so orientation-dependent gestures (Thumbs Up vs Thumbs Down)
  remain separable even though the shape is rotation-normalized.

This module is dependency-light (numpy only) so it can be unit-tested without
OpenCV, MediaPipe, scikit-learn, or Qt.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from vision.landmarks import HandLandmark as LM
from vision.landmarks import HandResult

_EPS = 1e-6


def _feature_names() -> List[str]:
    names: List[str] = []
    for i in range(21):
        names.append(f"x{i}")
        names.append(f"y{i}")
    names.append("orient_sin")
    names.append("orient_cos")
    return names


FEATURE_NAMES: List[str] = _feature_names()
FEATURE_DIM: int = len(FEATURE_NAMES)  # 44


def extract_features(hand: HandResult) -> np.ndarray:
    """Return a normalized ``float`` feature vector of length :data:`FEATURE_DIM`.

    Args:
        hand: A detected hand with its 21 landmarks.

    Returns:
        A 1-D numpy array combining 42 rotation/scale/translation-normalized
        landmark coordinates followed by ``[orient_sin, orient_cos]``.
    """
    pts = np.array([[lm.x, lm.y] for lm in hand.landmarks], dtype=float)  # (21, 2)
    wrist = pts[int(LM.WRIST)].copy()
    pts = pts - wrist

    ref = pts[int(LM.MIDDLE_MCP)]
    scale = float(math.hypot(ref[0], ref[1]))
    if scale < _EPS:
        scale = _EPS

    angle = math.atan2(ref[1], ref[0])  # absolute orientation of the hand
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)

    norm = (pts @ rot.T) / scale  # rotation + scale normalized shape
    if hand.handedness == "Left":
        norm[:, 1] = -norm[:, 1]  # mirror onto the right-hand frame

    shape = norm.reshape(-1)  # 42 features
    orientation = np.array([math.sin(angle), math.cos(angle)], dtype=float)
    return np.concatenate([shape, orientation]).astype(float)


def extract_batch(hands: List[HandResult]) -> np.ndarray:
    """Vectorize a list of hands into a ``(n_hands, FEATURE_DIM)`` array."""
    if not hands:
        return np.empty((0, FEATURE_DIM), dtype=float)
    return np.array([extract_features(h) for h in hands], dtype=float)


__all__ = ["FEATURE_NAMES", "FEATURE_DIM", "extract_features", "extract_batch"]
