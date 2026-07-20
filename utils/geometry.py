"""Geometry helpers used by vision/measurement modules.

These are pure functions operating on ``(x, y)`` or ``(x, y, z)`` tuples /
numpy arrays. They are intentionally dependency-light (numpy only) so they can
be unit-tested without OpenCV, MediaPipe, or Qt installed.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Point = Sequence[float]


def euclidean_distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two points of equal dimension."""
    pa = np.asarray(a, dtype=float)
    pb = np.asarray(b, dtype=float)
    return float(np.linalg.norm(pa - pb))


def angle_between(a: Point, b: Point, c: Point) -> float:
    """Return the angle (in degrees) at vertex ``b`` formed by a-b-c."""
    pa = np.asarray(a, dtype=float)
    pb = np.asarray(b, dtype=float)
    pc = np.asarray(c, dtype=float)
    v1 = pa - pb
    v2 = pc - pb
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def vector_angle_deg(a: Point, b: Point) -> float:
    """Return the orientation (degrees, 0-360) of the vector from ``a`` to ``b``."""
    pa = np.asarray(a, dtype=float)
    pb = np.asarray(b, dtype=float)
    dx, dy = (pb - pa)[:2]
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360.0


def normalize(values: Sequence[float]) -> np.ndarray:
    """Min-max normalize a 1D sequence into the [0, 1] range."""
    arr = np.asarray(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo == 0:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
