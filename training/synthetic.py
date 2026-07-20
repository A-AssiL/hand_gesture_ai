"""Procedural synthetic hand generator + dataset builder.

Until the Phase 4 dataset recorder lets you capture real samples from your
webcam, this module bootstraps a labelled dataset by *constructing* archetypal
hand poses (with random jitter) for each core gesture. That makes the whole ML
pipeline - train, evaluate, save, load, predict - demonstrable out of the box.

The poses are deliberately simple; a model trained on synthetic data mainly
proves the pipeline works. Re-train on recorded data (Phase 4) for real-world
accuracy.

Run as a script:
    python -m training.synthetic --out datasets/synthetic.csv --samples 80
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gestures.features import extract_features
from gestures.labels import SYNTHETIC_GESTURES
from vision.landmarks import HandResult, Landmark

# Base upright right-hand geometry (normalized image coordinates, y grows down).
_WRIST: Tuple[float, float] = (0.50, 0.90)
_MCP: Dict[str, Tuple[float, float]] = {
    "Index": (0.44, 0.62),
    "Middle": (0.50, 0.61),
    "Ring": (0.56, 0.62),
    "Pinky": (0.61, 0.64),
}
_DIR: Dict[str, Tuple[float, float]] = {
    "Index": (-0.03, -1.0),
    "Middle": (0.0, -1.0),
    "Ring": (0.03, -1.0),
    "Pinky": (0.07, -1.0),
}
_THUMB_CMC: Tuple[float, float] = (0.41, 0.83)
_THUMB_MCP: Tuple[float, float] = (0.36, 0.77)


def _unit(v: Tuple[float, float]) -> Tuple[float, float]:
    n = math.hypot(v[0], v[1]) or 1.0
    return (v[0] / n, v[1] / n)


def _finger(name: str, extended: bool) -> List[Tuple[float, float]]:
    mcp = _MCP[name]
    d = _unit(_DIR[name])
    if extended:
        offs = (0.11, 0.17, 0.23)
        pip = (mcp[0] + d[0] * offs[0], mcp[1] + d[1] * offs[0])
        dip = (mcp[0] + d[0] * offs[1], mcp[1] + d[1] * offs[1])
        tip = (mcp[0] + d[0] * offs[2], mcp[1] + d[1] * offs[2])
    else:
        pip = (mcp[0] + d[0] * 0.05, mcp[1] + d[1] * 0.05)
        dip = (mcp[0] + d[0] * 0.01, mcp[1] + d[1] * 0.01)
        tip = (mcp[0] - d[0] * 0.03, mcp[1] - d[1] * 0.03)
    return [mcp, pip, dip, tip]


def _thumb(extended: bool) -> List[Tuple[float, float]]:
    cmc, mcp = _THUMB_CMC, _THUMB_MCP
    if extended:
        d = _unit((-1.0, -0.5))
        ip = (mcp[0] + d[0] * 0.07, mcp[1] + d[1] * 0.07)
        tip = (mcp[0] + d[0] * 0.13, mcp[1] + d[1] * 0.13)
    else:
        ip = (0.43, 0.75)
        tip = (0.49, 0.73)
    return [cmc, mcp, ip, tip]


def make_hand(
    extended: Dict[str, bool],
    rotation_deg: float = 0.0,
    handedness: str = "Right",
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    ok_circle: bool = False,
) -> HandResult:
    """Construct a synthetic :class:`HandResult` from a finger-extension spec."""
    rng = rng if rng is not None else np.random.default_rng()
    thumb = _thumb(extended["Thumb"])
    index = _finger("Index", extended["Index"])
    middle = _finger("Middle", extended["Middle"])
    ring = _finger("Ring", extended["Ring"])
    pinky = _finger("Pinky", extended["Pinky"])

    if ok_circle:
        # Thumb tip and index tip meet to form the "OK" ring.
        index[3] = (0.42, 0.66)
        index[2] = (0.43, 0.70)
        thumb[2] = (0.40, 0.72)
        thumb[3] = (0.42, 0.67)

    points: List[Tuple[float, float]] = [
        _WRIST,
        thumb[0], thumb[1], thumb[2], thumb[3],
        index[0], index[1], index[2], index[3],
        middle[0], middle[1], middle[2], middle[3],
        ring[0], ring[1], ring[2], ring[3],
        pinky[0], pinky[1], pinky[2], pinky[3],
    ]
    arr = np.array(points, dtype=float)

    if rotation_deg:
        theta = math.radians(rotation_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
        origin = np.array(_WRIST, dtype=float)
        arr = (arr - origin) @ rot.T + origin

    if handedness == "Left":
        arr[:, 0] = 2.0 * _WRIST[0] - arr[:, 0]

    if jitter:
        arr = arr + rng.normal(0.0, jitter, arr.shape)

    landmarks = [Landmark(x=float(p[0]), y=float(p[1]), z=0.0) for p in arr]
    return HandResult(handedness=handedness, score=0.99, landmarks=landmarks)


_PATTERNS: Dict[str, Dict[str, bool]] = {
    "Fist": dict(Thumb=False, Index=False, Middle=False, Ring=False, Pinky=False),
    "Open Hand": dict(Thumb=True, Index=True, Middle=True, Ring=True, Pinky=True),
    "Peace": dict(Thumb=False, Index=True, Middle=True, Ring=False, Pinky=False),
    "Thumbs Up": dict(Thumb=True, Index=False, Middle=False, Ring=False, Pinky=False),
    "Thumbs Down": dict(Thumb=True, Index=False, Middle=False, Ring=False, Pinky=False),
    "OK": dict(Thumb=False, Index=False, Middle=True, Ring=True, Pinky=True),
    "Pointing": dict(Thumb=False, Index=True, Middle=False, Ring=False, Pinky=False),
    "Call Me": dict(Thumb=True, Index=False, Middle=False, Ring=False, Pinky=True),
    "Three": dict(Thumb=False, Index=True, Middle=True, Ring=True, Pinky=False),
    "Four": dict(Thumb=False, Index=True, Middle=True, Ring=True, Pinky=True),
}


def make_gesture_hand(
    name: str,
    handedness: str = "Right",
    jitter: float = 0.012,
    rng: Optional[np.random.Generator] = None,
) -> HandResult:
    """Build a synthetic hand for a named gesture."""
    pattern = _PATTERNS[name]
    rotation = 180.0 if name == "Thumbs Down" else 0.0
    return make_hand(
        pattern,
        rotation_deg=rotation,
        handedness=handedness,
        jitter=jitter,
        rng=rng,
        ok_circle=(name == "OK"),
    )


def generate_dataset(
    path: str | Path,
    samples_per_class: int = 80,
    jitter: float = 0.012,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Generate a labelled synthetic dataset and save it to ``path``."""
    rng = np.random.default_rng(seed)
    rows: List[np.ndarray] = []
    labels: List[str] = []
    for name in SYNTHETIC_GESTURES:
        for i in range(samples_per_class):
            hand = make_gesture_hand(
                name,
                handedness=("Left" if i % 2 else "Right"),
                jitter=jitter,
                rng=rng,
            )
            rows.append(extract_features(hand))
            labels.append(name)
    X = np.array(rows, dtype=float)

    from training.dataset import save_dataset

    save_dataset(path, X, labels)
    return X, labels


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic gesture dataset")
    parser.add_argument("--out", default="datasets/synthetic.csv", help="output CSV path")
    parser.add_argument("--samples", type=int, default=80, help="samples per gesture")
    parser.add_argument("--jitter", type=float, default=0.012, help="gaussian jitter std")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _, labels = generate_dataset(args.out, args.samples, args.jitter, args.seed)
    print(f"Wrote {len(labels)} samples across {len(set(labels))} gestures to {args.out}")


if __name__ == "__main__":
    _main()


__all__ = ["make_hand", "make_gesture_hand", "generate_dataset"]
