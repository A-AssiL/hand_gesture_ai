"""Typed configuration objects and a YAML-backed loader.

The application reads a single ``config.yaml`` file and maps each section onto a
strongly-typed :class:`dataclass`. Unknown keys are ignored and missing keys
fall back to sensible defaults, so the app still runs even with a partial
configuration file.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"

T = TypeVar("T")


@dataclass
class AppConfig:
    """General application metadata."""

    name: str = "AI Hand Gesture Recognition"
    version: str = "0.1.0"
    theme: str = "dark"


@dataclass
class CameraConfig:
    """Webcam capture settings."""

    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_horizontal: bool = True
    max_scan_index: int = 5


@dataclass
class HandTrackingConfig:
    """MediaPipe Hands parameters."""

    max_num_hands: int = 2
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


@dataclass
class DisplayConfig:
    """Overlay/rendering options for the live video feed."""

    draw_landmarks: bool = True
    draw_skeleton: bool = True
    draw_bounding_box: bool = True
    draw_landmark_ids: bool = True
    show_fps: bool = True
    landmark_radius: int = 4
    connection_thickness: int = 2


@dataclass
class AnalysisConfig:
    """Hand-analysis (finger states, orientation, measurements) options."""

    enabled: bool = True
    # A finger with a PIP/IP joint angle at or above this (degrees) counts as
    # fully straight; at or below ``finger_curled_deg`` it counts as fully
    # curled. Values in between are interpolated for the openness metric.
    finger_straight_deg: float = 160.0
    finger_curled_deg: float = 30.0
    # Angle (degrees) between the index and pinky directions treated as 100%
    # spread when normalizing the spread metric.
    spread_max_deg: float = 50.0
    # Draw a small analysis overlay (extended-finger count + palm facing)
    # beneath each hand's bounding box.
    draw_overlay: bool = True


@dataclass
class GestureConfig:
    """Gesture-recognition options."""

    enabled: bool = True
    # "rule_based" (no training required) or "random_forest" (trained model).
    backend: str = "rule_based"
    # Path to the trained model bundle used when backend == "random_forest".
    model_path: str = "models/gesture_rf.joblib"
    # ML predictions below this probability are reported as "Unknown".
    min_confidence: float = 0.35
    # How many class probabilities to show in the Detected Gesture panel.
    top_k: int = 3


@dataclass
class LoggingConfig:
    """Logging configuration (console + rotating file handler)."""

    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    log_file: str = "app.log"
    max_bytes: int = 1_048_576
    backup_count: int = 3


@dataclass
class Config:
    """Root configuration aggregating every section."""

    app: AppConfig = field(default_factory=AppConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _build(cls: Type[T], data: Any) -> T:
    """Instantiate a dataclass from a mapping, ignoring unknown keys."""
    if not isinstance(data, dict):
        return cls()  # type: ignore[call-arg]
    valid = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in valid}
    return cls(**filtered)  # type: ignore[arg-type]


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file. Defaults to the packaged
            ``config/config.yaml``.

    Returns:
        A fully-populated :class:`Config` instance. Missing files or sections
        fall back to defaults.
    """
    path = Path(path)
    data: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    return Config(
        app=_build(AppConfig, data.get("app")),
        camera=_build(CameraConfig, data.get("camera")),
        hand_tracking=_build(HandTrackingConfig, data.get("hand_tracking")),
        display=_build(DisplayConfig, data.get("display")),
        analysis=_build(AnalysisConfig, data.get("analysis")),
        gesture=_build(GestureConfig, data.get("gesture")),
        logging=_build(LoggingConfig, data.get("logging")),
    )


def to_dict(config: Config) -> Dict[str, Any]:
    """Serialize a :class:`Config` back into a plain dictionary."""
    return dataclasses.asdict(config)
