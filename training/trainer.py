"""Train and evaluate a gesture classifier, reporting full metrics.

Implements Feature 11: a model trainer that reports accuracy, precision,
recall, F1, and a confusion matrix, then saves a portable model bundle that
:class:`~gestures.ml_classifier.RandomForestRecognizer` can load.

scikit-learn / joblib are imported lazily inside methods so this module can be
imported (and its dataclasses used) even where they are not installed. On a
machine with the project's requirements installed, run:

    python -m training.trainer --synthetic --out models/gesture_rf.joblib
    python -m training.trainer --dataset datasets/my_data.csv --out models/gesture_rf.joblib
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gestures.features import FEATURE_DIM
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingReport:
    """Structured training/evaluation results."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    labels: List[str]
    confusion_matrix: List[List[int]]
    per_class: Dict[str, Dict[str, float]]
    n_train: int
    n_test: int
    n_features: int
    model_path: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def summary(self) -> str:
        """Return a short human-readable metrics summary."""
        return (
            f"Accuracy   : {self.accuracy * 100:.1f}%\n"
            f"Precision  : {self.precision_macro * 100:.1f}% (macro)\n"
            f"Recall     : {self.recall_macro * 100:.1f}% (macro)\n"
            f"F1-score   : {self.f1_macro * 100:.1f}% (macro)\n"
            f"Classes    : {len(self.labels)}\n"
            f"Train/Test : {self.n_train}/{self.n_test} samples"
        )

    def confusion_text(self) -> str:
        """Render the confusion matrix as a fixed-width text table."""
        width = max((len(name) for name in self.labels), default=4)
        width = max(width, 4)
        header = " " * (width + 2) + " ".join(f"{i:>4}" for i in range(len(self.labels)))
        lines = [header]
        for i, (name, row) in enumerate(zip(self.labels, self.confusion_matrix)):
            cells = " ".join(f"{v:>4}" for v in row)
            lines.append(f"{i:>2} {name:<{width}} {cells}")
        return "\n".join(lines)

    def save_report(self, path: str | Path) -> None:
        """Write the report as JSON plus a companion ``.txt`` summary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)
        txt = path.with_suffix(".txt")
        with txt.open("w", encoding="utf-8") as handle:
            handle.write(self.summary() + "\n\nConfusion matrix:\n")
            handle.write(self.confusion_text() + "\n")


class GestureModelTrainer:
    """Train a Random Forest gesture classifier and evaluate it.

    Args:
        n_estimators: Number of trees in the forest.
        test_size: Fraction of the dataset held out for evaluation.
        random_state: Seed for reproducible splits/forests.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state

    def train(self, X: np.ndarray, y: List[str]) -> Tuple[object, TrainingReport]:
        """Fit and evaluate a classifier. Returns ``(model, report)``."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import train_test_split

        X = np.asarray(X, dtype=float)
        y = list(y)
        labels = sorted(set(y))
        stratify = y if len(labels) > 1 and len(y) >= 2 * len(labels) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_ms = (time.perf_counter() - start) * 1000.0
        logger.info("Trained RandomForest on %d samples in %.0f ms", len(X_train), train_ms)

        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average=None, zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
        per_class = {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        }

        report = TrainingReport(
            accuracy=accuracy,
            precision_macro=float(p_macro),
            recall_macro=float(r_macro),
            f1_macro=float(f1_macro),
            labels=labels,
            confusion_matrix=cm,
            per_class=per_class,
            n_train=len(X_train),
            n_test=len(X_test),
            n_features=X.shape[1] if X.ndim == 2 else FEATURE_DIM,
        )
        return model, report

    def save_model(
        self,
        model: object,
        labels: List[str],
        path: str | Path,
        metrics: Optional[dict] = None,
    ) -> None:
        """Persist a model bundle with joblib."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": model,
            "labels": list(labels),
            "feature_dim": FEATURE_DIM,
            "metrics": metrics or {},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        joblib.dump(bundle, path)
        logger.info("Saved gesture model to %s", path)

    def train_from_dataset(
        self, dataset_path: str | Path, model_out: str | Path
    ) -> TrainingReport:
        """Load a dataset CSV, train, evaluate, and save the model + report."""
        from training.dataset import load_dataset

        X, y = load_dataset(dataset_path)
        if len(y) == 0:
            raise ValueError(f"Dataset '{dataset_path}' is empty")
        model, report = self.train(X, y)
        self.save_model(model, report.labels, model_out, metrics=report_metrics(report))
        report.model_path = str(model_out)
        report.save_report(Path(model_out).with_suffix(".report.json"))
        return report


def report_metrics(report: TrainingReport) -> dict:
    """Extract the scalar metrics worth embedding in the saved model bundle."""
    return {
        "accuracy": report.accuracy,
        "precision_macro": report.precision_macro,
        "recall_macro": report.recall_macro,
        "f1_macro": report.f1_macro,
        "n_train": report.n_train,
        "n_test": report.n_test,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train a gesture classifier")
    parser.add_argument("--dataset", help="path to a dataset CSV")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="generate a synthetic dataset first (ignored if --dataset exists)",
    )
    parser.add_argument("--out", default="models/gesture_rf.joblib", help="model output path")
    parser.add_argument("--samples", type=int, default=80, help="synthetic samples/class")
    parser.add_argument("--estimators", type=int, default=200)
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path or not Path(dataset_path).exists():
        if args.synthetic or not dataset_path:
            from training.synthetic import generate_dataset

            dataset_path = dataset_path or "datasets/synthetic.csv"
            print(f"Generating synthetic dataset at {dataset_path} ...")
            generate_dataset(dataset_path, samples_per_class=args.samples)
        else:
            raise SystemExit(f"Dataset not found: {dataset_path}")

    trainer = GestureModelTrainer(n_estimators=args.estimators)
    report = trainer.train_from_dataset(dataset_path, args.out)
    print(report.summary())
    print("\nConfusion matrix:")
    print(report.confusion_text())
    print(f"\nModel saved to {args.out}")


if __name__ == "__main__":
    _main()


__all__ = ["TrainingReport", "GestureModelTrainer", "report_metrics"]
