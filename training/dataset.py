"""Dataset IO for gesture feature vectors.

Samples are stored as a simple CSV: one row per sample with the 44 feature
columns followed by a ``label`` column. Using the stdlib ``csv`` module keeps
this loader dependency-light (numpy only) so it can be unit-tested anywhere.
The Phase 4 dataset recorder will append to the very same format.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from gestures.features import FEATURE_NAMES

_HEADER = FEATURE_NAMES + ["label"]


def save_dataset(path: str | Path, X: Sequence[Sequence[float]], y: Sequence[str]) -> None:
    """Write a full dataset (overwriting any existing file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(X, dtype=float)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        for row, label in zip(arr, y):
            writer.writerow([f"{v:.6f}" for v in row] + [label])


def append_samples(path: str | Path, X: Sequence[Sequence[float]], y: Sequence[str]) -> None:
    """Append samples, writing the header first if the file is new/empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_content = path.exists() and path.stat().st_size > 0
    arr = np.asarray(X, dtype=float)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not has_content:
            writer.writerow(_HEADER)
        for row, label in zip(arr, y):
            writer.writerow([f"{v:.6f}" for v in row] + [label])


def load_dataset(path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """Load a dataset CSV into ``(X, y)``."""
    path = Path(path)
    features: List[List[float]] = []
    labels: List[str] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            *values, label = row
            features.append([float(v) for v in values])
            labels.append(label)
    return np.array(features, dtype=float), labels


def count_by_label(path: str | Path) -> Dict[str, int]:
    """Return a ``{label: sample_count}`` mapping for a dataset file."""
    _, labels = load_dataset(path)
    return dict(Counter(labels))


__all__ = ["save_dataset", "append_samples", "load_dataset", "count_by_label"]
