# AI Hand Gesture Recognition and Analysis

A production-quality, real-time **hand tracking, gesture recognition, and hand
analysis** desktop application written in Python. It captures your webcam feed,
detects one or two hands with **MediaPipe**, draws the 21-point hand skeleton,
analyzes finger states / orientation / measurements, and classifies the pose
into a named gesture — either with a **zero-setup rule-based recognizer** or a
trainable **scikit-learn Random Forest** model. Everything is presented in a
polished dark-themed **PySide6 (Qt)** GUI.

> **Reading this as an AI/LLM?** This README is intentionally exhaustive. It
> describes the full architecture, every module, the data flow, the config
> schema, the ML feature design, and the extension points, so you can reason
> about or modify the project without reading every source file first. Jump to
> [Section 6 — Project structure](#6-project-structure-every-file-explained) and
> [Section 8 — Data flow](#8-data-flow-camera--screen) for the mental model.

---

## Table of contents

1. [What this app does](#1-what-this-app-does)
2. [Feature checklist](#2-feature-checklist)
3. [Screenshots / UI layout](#3-ui-layout)
4. [Quick start](#4-quick-start)
5. [Requirements & environment](#5-requirements--environment)
6. [Project structure (every file explained)](#6-project-structure-every-file-explained)
7. [Architecture & design principles](#7-architecture--design-principles)
8. [Data flow (camera → screen)](#8-data-flow-camera--screen)
9. [The vision pipeline in detail](#9-the-vision-pipeline-in-detail)
10. [Hand analysis explained](#10-hand-analysis-explained)
11. [Gesture recognition explained](#11-gesture-recognition-explained)
12. [The machine-learning pipeline](#12-the-machine-learning-pipeline)
13. [Configuration reference](#13-configuration-reference)
14. [Usage guide (buttons & workflows)](#14-usage-guide-buttons--workflows)
15. [Command-line tools](#15-command-line-tools)
16. [Testing](#16-testing)
17. [Logging](#17-logging)
18. [Troubleshooting](#18-troubleshooting)
19. [Development roadmap](#19-development-roadmap)
20. [Extending the project](#20-extending-the-project)
21. [Glossary](#21-glossary)
22. [License](#22-license)

---

## 1. What this app does

The application runs a continuous loop:

1. Grab a frame from the webcam.
2. Detect hands and their 21 landmarks (MediaPipe Hands).
3. Analyze each hand: which fingers are extended, palm orientation, geometric
   measurements, and a plain-English description.
4. Recognize the gesture (rule-based or ML).
5. Draw all overlays on the frame and update the info panels — at 30–60 FPS.

It is built **incrementally in phases** (see the [roadmap](#19-development-roadmap)).
Phases 1–3 are complete: project foundation + live tracking, hand analysis, and
gesture recognition + ML.

---

## 2. Feature checklist

The original specification defined 18 feature areas. Status by phase:

| # | Feature | Status | Where |
|---|---------|--------|-------|
| 1 | Live webcam (30–60 FPS), adjustable resolution, FPS counter, camera switching | ✅ | `core/camera.py`, `core/fps_counter.py`, `ui/video_thread.py` |
| 2 | Hand detection (L/R/both) + skeleton, landmarks, bounding box | ✅ | `vision/hand_tracker.py`, `vision/drawing.py` |
| 3 | 21-landmark extraction (x/y/z) + landmark IDs | ✅ | `vision/landmarks.py` |
| 4 | Finger state analysis (extended/folded) | ✅ | `vision/analysis.py` |
| 5 | Human-readable hand description | ✅ | `vision/analysis.py` |
| 6 | Gesture recognition (rule-based, many gestures) | ✅ | `gestures/rule_based.py` |
| 7 | Hand orientation (facing, pointing, rotation/tilt) | ✅ | `vision/analysis.py` |
| 8 | Measurements (angles, distances, palm size, openness, grip %, spread %) | ✅ | `vision/analysis.py` |
| 9 | Modular AI classifier (landmarks → features → Random Forest) | ✅ | `gestures/features.py`, `gestures/ml_classifier.py` |
| 10 | Dataset recorder | ⏳ Phase 4 | (planned) |
| 11 | Model trainer with accuracy/precision/recall/F1/confusion matrix | ✅ | `training/trainer.py` |
| 12 | Real-time prediction (gesture, confidence, probabilities, timings) | ✅ | `core/engine.py`, `ui/panels.py` |
| 13 | Dark-theme UI panels + control buttons | ✅ (screenshot done; recorder/settings ⏳ Phase 4) | `ui/` |
| 14 | Logging to file | ✅ | `utils/logger.py` |
| 15 | Performance (30–60 FPS, GPU if available) | ⏳ Phase 4 tuning | `ui/video_thread.py` |
| 16 | Documentation | ✅ (this README + `docs/`) | `docs/` |
| 17 | Code quality (type hints, docstrings, OOP, SOLID, error handling) | ✅ | entire codebase |
| 18 | Future expansion modules | 🔭 designed for | `gestures/base.py` interface |

**Currently recognized gestures** (rule-based, no training needed): `Fist`,
`Open Hand`, `Peace`, `Thumbs Up`, `Thumbs Down`, `OK`, `Pinch`, `Pointing`,
`Call Me`, `Three`, `Four`, plus `Unknown` for anything unclear. The ML model
currently trains on 10 of these classes (see [Section 12](#12-the-machine-learning-pipeline)).

---

## 3. UI layout

```
+-----------------------------------------------------------+
| [Start Camera] [Stop Camera]   Camera: [ dropdown v ]     |
| +-----------------------------+  +---------------------+   |
| |                             |  | Statistics          |   |
| |                             |  |  FPS / #hands / res |   |
| |        LIVE VIDEO           |  +---------------------+   |
| |  (skeleton + landmarks +    |  | Hand Information    |   |
| |   bbox + gesture label)     |  |  handedness, angles |   |
| |                             |  +---------------------+   |
| |                             |  | Finger States       |   |
| |                             |  |  Thumb..Pinky       |   |
| +-----------------------------+  +---------------------+   |
| [Save Screenshot][Capture Dataset][Train Model]           |
| [Load Model][Record Video][Settings]                      |
|                                  | Detected Gesture    |   |
|                                  |  NAME + conf + top-k|   |
|                                  +---------------------+   |
|                                  | Logs                |   |
|                                  +---------------------+   |
| Status bar: Ready / messages                              |
+-----------------------------------------------------------+
```

The **right column** holds five live panels: Statistics, Hand Information,
Finger States, Detected Gesture, and Logs. The **left column** holds camera
controls, the video feed, and the action buttons.

---

## 4. Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.10–3.12 recommended)
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# macOS/Linux:
# source .venv/bin/activate && pip install --upgrade pip

# 2. Install dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Run the app
.\.venv\Scripts\python.exe app.py
```

On macOS/Linux the run command is simply `python app.py` once the venv is active.

> **Why call the venv Python directly?** On Windows, `Activate.ps1` can be
> blocked by the PowerShell execution policy. Invoking
> `.\.venv\Scripts\python.exe app.py` sidesteps activation entirely and always
> uses the right interpreter.

When the window opens: pick a camera, click **Start Camera**, and show your hand.
Gesture names appear immediately (rule-based). Click **Train Model** to build
and switch to a Random Forest classifier.

---

## 5. Requirements & environment

### Python
- **Python 3.10 – 3.12** is strongly recommended.
- MediaPipe does not ship wheels for every newer version (e.g. 3.13) and
  requires **NumPy < 2.0**, which is why versions are pinned.

### Dependencies (`requirements.txt`)

| Package | Version pin | Purpose |
|---------|-------------|---------|
| `PySide6` | `>=6.6.0` | Qt-based GUI framework |
| `opencv-python` | `>=4.9.0` | Camera capture, image ops, drawing |
| `mediapipe` | `>=0.10.9,<0.10.30` | Hand detection & 21-landmark tracking |
| `numpy` | `>=1.26,<2.0` | Numeric arrays (MediaPipe needs <2.0) |
| `PyYAML` | `>=6.0.1` | Load `config/config.yaml` |
| `scikit-learn` | `>=1.4.0` | Random Forest classifier + metrics |
| `pandas` | `>=2.2.0` | Data handling for datasets |
| `joblib` | `>=1.3.0` | Serialize/deserialize trained models |

Commented-out **future** extras: `torch`, `xgboost`, `matplotlib`, `seaborn`.

> ⚠️ **Pin note:** mediapipe `0.10.35` restructured/dropped the classic
> `mediapipe.solutions.hands` API and breaks tracking. Stay in the pinned range
> (`0.10.21` is known-good). Also install `opencv-python`, **not** `cv2`
> (`cv2` is the import name, not the pip package).

### Hardware
- A webcam.
- CPU is enough for real-time tracking. GPU acceleration is a Phase 4 item.

---

## 6. Project structure (every file explained)

```
hand_gesture_ai/
├── app.py                     # Entry point: load config, set up logging, launch GUI
├── requirements.txt           # Pinned dependencies
├── README.md                  # This file
├── LICENSE                    # MIT license
├── .gitignore
│
├── config/                    # Typed configuration + YAML
│   ├── __init__.py
│   ├── config.yaml            # All runtime settings (edit this)
│   └── settings.py            # Dataclasses + YAML loader (source of truth for schema)
│
├── core/                      # Camera + frame pipeline (framework-light)
│   ├── __init__.py
│   ├── camera.py              # OpenCV VideoCapture wrapper, resolution, camera scan
│   ├── engine.py              # FrameProcessor: ties tracker+analyzer+recognizer+painter
│   └── fps_counter.py         # Rolling FPS measurement
│
├── vision/                    # Computer-vision layer
│   ├── __init__.py
│   ├── landmarks.py           # HandLandmark enum, Landmark & HandResult dataclasses
│   ├── hand_tracker.py        # MediaPipe Hands wrapper -> HandResult objects
│   ├── analysis.py            # HandAnalyzer: finger states, orientation, measurements
│   └── drawing.py             # LandmarkPainter: skeleton/landmarks/bbox/labels overlays
│
├── gestures/                  # Gesture recognition layer
│   ├── __init__.py            # Exports; lazily exposes the ML recognizer
│   ├── base.py                # GestureResult + GestureRecognizer ABC (the contract)
│   ├── labels.py              # Canonical gesture-name constants & lists
│   ├── features.py            # Landmark -> 44-dim feature vector (numpy only)
│   ├── rule_based.py          # RuleBasedRecognizer (no training required)
│   └── ml_classifier.py       # RandomForestRecognizer (scikit-learn, lazy imports)
│
├── training/                  # Dataset + model training tooling
│   ├── __init__.py
│   ├── dataset.py             # CSV save/load/append/count for feature datasets
│   ├── synthetic.py           # Procedural hand generator to bootstrap training
│   └── trainer.py             # GestureModelTrainer + TrainingReport (metrics/CM)
│
├── ui/                        # PySide6 GUI
│   ├── __init__.py
│   ├── main_window.py         # MainWindow: layout, buttons, slots, orchestration
│   ├── panels.py              # Statistics/HandInfo/FingerState/Gesture/Log panels
│   ├── video_thread.py        # QThread capture loop; emits Qt signals
│   ├── video_widget.py        # Widget that paints frames (numpy BGR -> QImage)
│   ├── theme.py               # Dark stylesheet
│   └── log_handler.py         # Bridges Python logging -> the Logs panel (Qt signal)
│
├── utils/                     # Shared helpers
│   ├── __init__.py
│   ├── geometry.py            # euclidean_distance, angle_between, vector_angle_deg, normalize
│   └── logger.py              # setup_logging / get_logger (console + rotating file)
│
├── models/                    # Saved model bundles (.joblib) land here
│   ├── __init__.py
│   └── .gitkeep
├── datasets/                  # Recorded/synthetic datasets (.csv) land here
│   └── .gitkeep
├── assets/                    # Screenshots, captures, static assets
│   └── .gitkeep
│
├── tests/                     # Unit tests (pytest-style; also runnable manually)
│   ├── __init__.py
│   ├── conftest_note.md
│   ├── test_config.py         # Config loading & defaults
│   ├── test_geometry.py       # Geometry helpers
│   ├── test_fps_counter.py    # FPS counter math
│   ├── test_analysis.py       # HandAnalyzer finger states/orientation
│   ├── test_features.py       # Feature vector shape & invariances
│   └── test_rule_based.py     # Rule-based recognizer on synthetic hands
│
└── docs/                      # Extra documentation
    ├── ARCHITECTURE.md        # Architecture deep-dive
    ├── API.md                 # Module/class API reference
    └── ROADMAP.md             # Phase-by-phase plan and status
```

### Key module responsibilities

- **`app.py`** — Loads config, configures logging, then lazily imports Qt so any
  config/logging error surfaces before the heavy GUI stack loads. Launches
  `MainWindow` and runs the Qt event loop.
- **`config/settings.py`** — The **schema source of truth**. Each YAML section
  maps to a dataclass: `AppConfig`, `CameraConfig`, `HandTrackingConfig`,
  `DisplayConfig`, `AnalysisConfig`, `GestureConfig`, `LoggingConfig`, all
  aggregated in `Config`. `load_config()` reads YAML, ignores unknown keys, and
  falls back to defaults for missing keys, so a partial file still runs.
- **`core/camera.py`** — Wraps `cv2.VideoCapture`: opens by index, sets
  resolution/FPS, optionally mirrors the frame, and can scan for available
  camera indices.
- **`core/engine.py`** — `FrameProcessor` is the **UI-agnostic heart** of the
  app. `process(frame)` returns a `ProcessResult(frame, hands, analyses,
  gestures)`. It builds the recognizer from config (rule-based or a trained RF
  if a model file exists) and exposes `load_model()`, `set_recognizer()`,
  `use_rule_based()` for runtime swapping.
- **`vision/landmarks.py`** — Framework-agnostic data types (no cv2/mediapipe):
  `HandLandmark` (IntEnum 0–20), `Landmark(x, y, z)` with `to_pixel()`, and
  `HandResult(handedness, score, landmarks)` with `landmark()`, `pixel()`,
  `bounding_box()`. These flow through the whole app so every layer is testable
  in isolation.
- **`vision/hand_tracker.py`** — Wraps MediaPipe Hands, converts its raw output
  into `HandResult`s, and contains a robust loader with actionable error
  messages for common environment problems.
- **`vision/analysis.py`** — `HandAnalyzer.analyze(hand)` returns a
  `HandAnalysis` (finger states, orientation, measurements, description).
  Orientation-tolerant heuristics (distance-based, not raw-axis).
- **`vision/drawing.py`** — `LandmarkPainter.draw(frame, hands, analyses,
  gestures)` renders skeleton, landmarks (+IDs), bounding box, an analysis
  summary line, and the gesture label. `draw_fps()` overlays the FPS counter.
- **`gestures/base.py`** — The abstraction every backend implements:
  `predict(hand) -> GestureResult`, a `labels` property, and `is_ready()`.
- **`gestures/features.py`** — Converts a `HandResult` into a **44-dimensional**
  normalized feature vector (see [Section 12](#12-the-machine-learning-pipeline)).
- **`gestures/rule_based.py`** — Deterministic recognizer using finger states +
  orientation + a pinch metric. Always ready; needs no ML libraries.
- **`gestures/ml_classifier.py`** — `RandomForestRecognizer` loads a joblib
  bundle and predicts with class probabilities. sklearn/joblib are imported
  **lazily** so importing the module never fails without them.
- **`training/`** — `dataset.py` (CSV IO), `synthetic.py` (procedural hands to
  bootstrap a dataset before the Phase 4 recorder exists), `trainer.py`
  (train/evaluate/report/save).
- **`ui/`** — `video_thread.py` runs capture off the GUI thread and emits
  signals (`frame_ready`, `hands_updated`, `analysis_updated`, `gesture_updated`,
  `fps_updated`, `error`). `main_window.py` connects those signals to panel
  updates and wires the buttons.
- **`utils/`** — `geometry.py` (pure-math helpers) and `logger.py`
  (console + rotating file handler, plus a UI handler bridge).

---

## 7. Architecture & design principles

The codebase follows clean, layered, SOLID-oriented design:

- **Separation of concerns / layering.** `vision` (perception) → `gestures`
  (interpretation) → `core` (orchestration) → `ui` (presentation). Lower layers
  never import upper layers.
- **Framework-agnostic core.** `vision/landmarks.py`, `vision/analysis.py`,
  `gestures/features.py`, `gestures/rule_based.py`, and `utils/geometry.py`
  depend only on the stdlib + numpy. This makes them **unit-testable without a
  camera, Qt, MediaPipe, or scikit-learn**.
- **Dependency inversion.** The UI and pipeline depend on the
  `GestureRecognizer` abstraction, not on concrete recognizers. Swapping
  rule-based ↔ Random Forest ↔ a future neural net is a one-line change.
- **Lazy heavy imports.** Qt/OpenCV/MediaPipe (in `app.py`) and sklearn/joblib
  (in `ml_classifier.py`/`trainer.py`) are imported only when needed, so
  configuration errors surface early and the app remains importable in minimal
  environments.
- **Configuration over code.** Behavior is driven by `config/config.yaml`
  mapped to typed dataclasses; unknown keys are ignored and missing keys use
  defaults.
- **Responsiveness.** Capture + inference run in a `QThread`; the GUI thread
  only paints and updates labels, keeping the UI smooth at 30–60 FPS.
- **Type hints & docstrings everywhere**, with defensive error handling around
  the camera, model loading, and per-frame processing.

---

## 8. Data flow (camera → screen)

```
        ┌────────────┐   BGR frame   ┌──────────────────────────────┐
Webcam ─▶│  Camera    │──────────────▶│        FrameProcessor        │
        │ (core)     │               │            (core)            │
        └────────────┘               │                              │
                                      │  HandTracker  → hands        │
                                      │  HandAnalyzer → analyses     │
                                      │  Recognizer   → gestures     │
                                      │  LandmarkPainter → annotated │
                                      └──────────────┬───────────────┘
                                                     │ ProcessResult
                                                     ▼
                                   ┌──────────────────────────────────┐
                                   │            VideoThread            │
                                   │  (QThread; runs the loop)         │
                                   │  emits Qt signals:                │
                                   │   frame_ready / hands_updated /   │
                                   │   analysis_updated /              │
                                   │   gesture_updated / fps_updated   │
                                   └──────────────┬───────────────────┘
                                                  │ signals (thread-safe)
                                                  ▼
                                   ┌──────────────────────────────────┐
                                   │            MainWindow             │
                                   │  VideoWidget  ← frame_ready       │
                                   │  Statistics   ← fps/hands         │
                                   │  HandInfo     ← analysis          │
                                   │  FingerStates ← analysis          │
                                   │  Gesture      ← gesture           │
                                   │  Logs         ← logging handler   │
                                   └──────────────────────────────────┘
```

The key contract is **`ProcessResult`** (in `core/engine.py`):

```python
@dataclass
class ProcessResult:
    frame: np.ndarray                 # annotated BGR frame
    hands: List[HandResult]           # raw detections
    analyses: List[HandAnalysis]      # one per hand (if analysis enabled)
    gestures: List[GestureResult]     # one per hand (if gesture enabled)
```

---

## 9. The vision pipeline in detail

### 9.1 Landmarks (`vision/landmarks.py`)
MediaPipe returns **21 landmarks per hand**. Indices are defined by the
`HandLandmark` IntEnum:

```
0  WRIST
1  THUMB_CMC   2 THUMB_MCP  3 THUMB_IP   4 THUMB_TIP
5  INDEX_MCP   6 INDEX_PIP  7 INDEX_DIP  8 INDEX_TIP
9  MIDDLE_MCP 10 MIDDLE_PIP 11 MIDDLE_DIP 12 MIDDLE_TIP
13 RING_MCP   14 RING_PIP  15 RING_DIP  16 RING_TIP
17 PINKY_MCP  18 PINKY_PIP 19 PINKY_DIP 20 PINKY_TIP
```

Each `Landmark` has normalized `x, y` in `[0,1]` (fraction of frame width/height)
and a relative depth `z` (smaller = closer). `HAND_CONNECTIONS` lists the edges
used to draw the skeleton. `HandResult.bounding_box()` computes a padded, clamped
pixel box from the landmarks.

### 9.2 Tracking (`vision/hand_tracker.py`)
`HandTracker` owns a single MediaPipe `Hands` graph configured from
`HandTrackingConfig` (max hands, model complexity, detection/tracking
confidence). `process(frame_bgr)`:
1. Converts BGR→RGB (MediaPipe expects RGB).
2. Marks the array read-only for a zero-copy speedup.
3. Runs detection and converts each result into a `HandResult` with handedness
   label + confidence and 21 `Landmark`s.

`_load_mediapipe_hands()` robustly locates the `solutions.hands` API and raises
clear, actionable errors (Python version, NumPy 2.x, broken install, shadowing
file) when it can't.

### 9.3 Drawing (`vision/drawing.py`)
`LandmarkPainter.draw(frame, hands, analyses, gestures)` mutates and returns the
frame, drawing (each toggleable via `DisplayConfig`):
- **Bounding box** with a `handedness score%` label.
- **Skeleton** lines from `HAND_CONNECTIONS`.
- **Landmarks** as dots, optionally with their numeric IDs.
- **Analysis summary** beneath the box (extended count, facing, pointing).
- **Gesture label** above the box (name + confidence), in a distinct color.

---

## 10. Hand analysis explained

`vision/analysis.py`'s `HandAnalyzer.analyze(hand)` produces a `HandAnalysis`:

- **`FingerStates`** (feature 4) — per-finger extended/folded.
  - Four fingers: extended when the **tip is farther from the wrist than the
    PIP** joint (orientation-independent).
  - Thumb: extended when its **tip is farther from the opposite (pinky) MCP
    than the IP** joint.
  - Exposes `extended`, `folded`, `count_extended`, and `as_labels()`.
- **`HandOrientation`** (feature 7) —
  - `facing`: `"Palm"` or `"Back"`, from the sign of the 2D cross product of
    wrist→index and wrist→pinky vectors (disambiguated by handedness).
  - `pointing`: `"Up"|"Down"|"Left"|"Right"` from the wrist→middle-MCP axis.
  - `rotation_deg` (0–360) and `tilt_deg` of the knuckle line.
- **`HandMeasurements`** (feature 8) — `palm_width`, `palm_height`,
  `hand_length` (normalized units), plus ratios in `[0,1]`: `openness`, `grip`,
  `spread`; and dicts `finger_angles` (PIP/IP joint angles in degrees) and
  `tip_distances` (tip→wrist / hand scale).
- **`description`** (feature 5) — a human-readable sentence summarizing the
  above (e.g. "Right palm facing camera, pointing up, 2 fingers extended").

All measurements are normalized by a **hand scale** = distance(wrist,
middle-MCP), so results are robust to how far the hand is from the camera.

---

## 11. Gesture recognition explained

### 11.1 The contract (`gestures/base.py`)
```python
@dataclass
class GestureResult:
    name: str
    confidence: float
    probabilities: Dict[str, float] = {}
    inference_ms: float = 0.0

class GestureRecognizer(ABC):
    def predict(self, hand: HandResult) -> GestureResult: ...
    @property
    def labels(self) -> List[str]: ...
    def is_ready(self) -> bool: ...   # default True
```

### 11.2 Rule-based recognizer (`gestures/rule_based.py`)
Needs **no training** and no ML libraries, so gestures appear the instant the
app starts. It uses the `HandAnalyzer` internally, plus a **pinch metric**
(distance(thumb-tip, index-tip) / hand scale). Decision order:
1. `count == 0` → **Fist**; `count == 5` → **Open Hand** (resolved first so a
   closed fist is never mistaken for a pinch).
2. If thumb & index are touching: **OK** (other three extended) or **Pinch**
   (only thumb + index).
3. Single-/multi-finger patterns → **Thumbs Up/Down** (thumb only; direction
   from orientation), **Pointing** (index only), **Peace** (index+middle),
   **Call Me** (thumb+pinky), **Three** (index+middle+ring), **Four**
   (index+middle+ring+pinky).
4. Otherwise → **Unknown** (confidence 0.30).

### 11.3 ML recognizer (`gestures/ml_classifier.py`)
`RandomForestRecognizer` loads a joblib **bundle** and predicts with
`predict_proba`, returning the top class, its confidence, the full probability
dict, and inference time. Predictions below `min_confidence` are reported as
`Unknown`. sklearn/joblib are imported lazily.

### 11.4 Backend selection (`core/engine.py`)
On startup `FrameProcessor._build_recognizer()`:
- Uses **Random Forest** only if `gesture.backend == "random_forest"` **and**
  the model file at `gesture.model_path` exists and loads.
- Otherwise falls back to the always-available **rule-based** recognizer.

At runtime, **Train Model** / **Load Model** swap the live recognizer via
`VideoThread.set_model()` → `FrameProcessor.load_model()`.

---

## 12. The machine-learning pipeline

### 12.1 Feature vector (`gestures/features.py`) — **44 dimensions**
`extract_features(hand)` builds a vector engineered for a small-data classifier:

1. Take the 21 `(x, y)` points.
2. **Translate** so the wrist is the origin (position-invariant).
3. **Rotate** so the wrist→middle-MCP axis is canonical → the *shape* reads the
   same at any in-plane rotation.
4. **Scale** by the wrist→middle-MCP distance (distance-invariant).
5. **Mirror** left hands onto the right-hand frame (handedness-invariant).
6. Flatten to **42** shape features, then append **2** absolute-orientation
   features `[sin(angle), cos(angle)]` so orientation-dependent gestures
   (Thumbs Up vs Thumbs Down) stay separable despite rotation-normalization.

Column names: `x0,y0,…,x20,y20,orient_sin,orient_cos`. `extract_batch(hands)`
vectorizes a list into an `(n, 44)` array. Dependency: **numpy only**.

### 12.2 Datasets (`training/dataset.py`)
Datasets are simple CSVs: 44 feature columns + a `label` column.
- `save_dataset(path, X, y)` — write (overwrite).
- `append_samples(path, X, y)` — append (writes header if new).
- `load_dataset(path) -> (X, y)`.
- `count_by_label(path) -> {label: count}`.

The Phase 4 dataset recorder will append real captured samples to this exact
format.

### 12.3 Synthetic bootstrap (`training/synthetic.py`)
Until the recorder exists, this **procedurally constructs** archetypal hand
poses (with jitter) for 10 gestures so the entire ML pipeline is demonstrable
out of the box:
- `make_hand(extended, rotation_deg, handedness, jitter, rng, ok_circle)` builds
  a `HandResult` from a finger-extension spec.
- `make_gesture_hand(name, …)` maps a gesture name to its pose (e.g. Thumbs
  Down = 180° rotation, OK = thumb/index ring).
- `generate_dataset(path, samples_per_class=80, jitter=0.012, seed=42)` writes a
  labelled CSV. Classes: `Fist, Open Hand, Peace, Thumbs Up, Thumbs Down, OK,
  Pointing, Call Me, Three, Four`.

> ⚠️ Synthetic data proves the pipeline works; **re-train on recorded data**
> (Phase 4) for real-world accuracy.

### 12.4 Trainer (`training/trainer.py`)
- `GestureModelTrainer.train(X, y)` — stratified train/test split, fits a
  `RandomForestClassifier`, and computes **accuracy, macro precision/recall/F1,
  per-class metrics, and a confusion matrix**.
- `TrainingReport` — dataclass holding all metrics with `summary()`,
  `confusion_text()`, and `save_report()` (JSON + `.txt`).
- `save_model(model, labels, path, metrics)` — writes a joblib bundle
  `{model, labels, feature_dim, metrics, created_at}`.
- `train_from_dataset(dataset_path, model_out)` — load → train → save model +
  report.
- sklearn/joblib are imported lazily inside methods.

---

## 13. Configuration reference

All settings live in **`config/config.yaml`** and are validated by
`config/settings.py`. Sections and defaults:

```yaml
app:
  name: "AI Hand Gesture Recognition"
  version: "0.1.0"
  theme: "dark"                 # only 'dark' implemented

camera:
  index: 0                      # default webcam index
  width: 1280                   # capture width
  height: 720                   # capture height
  fps: 30                       # target FPS (30-60)
  flip_horizontal: true         # mirror so it feels natural
  max_scan_index: 5             # how many indices to probe when listing cameras

hand_tracking:
  max_num_hands: 2              # detect up to two hands
  model_complexity: 1          # 0 (fast) or 1 (accurate)
  min_detection_confidence: 0.6
  min_tracking_confidence: 0.5

display:
  draw_landmarks: true
  draw_skeleton: true
  draw_bounding_box: true
  draw_landmark_ids: true
  show_fps: true
  landmark_radius: 4
  connection_thickness: 2

analysis:
  enabled: true
  finger_straight_deg: 160.0    # PIP/IP angle >= this => finger straight
  finger_curled_deg: 30.0       # PIP/IP angle <= this => finger curled
  spread_max_deg: 50.0          # index<->pinky angle mapped to 100% spread
  draw_overlay: true            # extended-count + facing on the frame

gesture:
  enabled: true
  backend: "rule_based"         # rule_based (no training) | random_forest (trained)
  model_path: "models/gesture_rf.joblib"
  min_confidence: 0.35          # ML predictions below this => "Unknown"
  top_k: 3                      # probability rows shown in the gesture panel

logging:
  level: "INFO"                 # DEBUG | INFO | WARNING | ERROR | CRITICAL
  log_to_file: true
  log_dir: "logs"
  log_file: "app.log"
  max_bytes: 1048576            # 1 MB per file before rotation
  backup_count: 3
```

Missing keys fall back to the dataclass defaults, and unknown keys are ignored —
so you can keep a partial config and the app still runs.

---

## 14. Usage guide (buttons & workflows)

| Button | What it does |
|--------|--------------|
| **Start Camera** | Opens the selected camera and begins streaming + inference. |
| **Stop Camera** | Stops the capture thread and resets panels. |
| **Camera dropdown** | Switch between detected camera indices (live). |
| **Save Screenshot** | Saves the current annotated frame as PNG (file dialog; defaults under `assets/captures/`). |
| **Capture Dataset** | ⏳ Phase 4 — record labelled samples. |
| **Train Model** | Generates a synthetic dataset (if none), trains a Random Forest, shows metrics, saves `models/gesture_rf.joblib`, and switches the live recognizer to it. Needs scikit-learn + joblib. |
| **Load Model** | Pick a `.joblib` bundle and use it live. |
| **Record Video** | ⏳ Phase 4. |
| **Settings** | ⏳ Phase 4 (edit `config/config.yaml` for now). |

**Typical workflows**

- *Just see gestures:* Start Camera → show your hand. Rule-based labels appear
  instantly; no setup.
- *Use ML:* Train Model (bootstraps synthetic data, trains, switches to RF) —
  or Load Model to use a previously trained bundle.
- *Tune tracking:* edit `hand_tracking` / `display` in `config.yaml` and restart.

---

## 15. Command-line tools

Both training modules are runnable as scripts (with the venv active / using the
venv Python):

```bash
# Generate a synthetic dataset
python -m training.synthetic --out datasets/synthetic.csv --samples 80

# Train from a dataset (or synthesize first) and save a model + report
python -m training.trainer --synthetic --out models/gesture_rf.joblib
python -m training.trainer --dataset datasets/my_data.csv --out models/gesture_rf.joblib
```

The trainer prints the metric summary and confusion matrix and writes a
`*.report.json` / `*.report.txt` next to the model.

---

## 16. Testing

Unit tests live in `tests/` and target the dependency-light layers so they run
without a camera, Qt, MediaPipe, or scikit-learn:

- `test_config.py` — YAML loading, defaults, unknown-key tolerance.
- `test_geometry.py` — distance/angle helpers.
- `test_fps_counter.py` — FPS math.
- `test_analysis.py` — finger states & orientation on constructed hands.
- `test_features.py` — feature dimension (44), wrist→origin, rotation
  invariance, orientation separability, determinism, batch shape.
- `test_rule_based.py` — rule-based recognizer classifies synthetic gestures.

Run with pytest (recommended):

```bash
pytest -q
```

If `pytest` isn't installed, the numpy-only tests can also be driven by
importing each `tests.test_*` module and calling its `test_*` functions.

---

## 17. Logging

`utils/logger.py` configures a root logger with a **console handler** and a
**rotating file handler** (`logs/app.log`, size-based rotation). `get_logger()`
returns namespaced child loggers used across modules. The GUI **Logs** panel
mirrors log records live via `ui/log_handler.py`, which emits a Qt signal from a
logging handler so log lines appear in the UI thread-safely.

---

## 18. Troubleshooting

| Symptom | Cause & fix |
|---------|-------------|
| `module 'mediapipe' has no attribute 'solutions'` / `No module named 'mediapipe.python'` | Broken/incompatible mediapipe. Use Python 3.10–3.12, `numpy<2`, and pin `mediapipe>=0.10.9,<0.10.30` (0.10.21 works). Reinstall with `--force-reinstall --no-cache-dir`. |
| `No module named 'cv2'` after `pip install cv2` | The pip package is **`opencv-python`**, not `cv2`. |
| PowerShell: running scripts is disabled (`Activate.ps1`) | Don't activate; call the interpreter directly: `.\.venv\Scripts\python.exe app.py`. |
| Training button warns about scikit-learn | `pip install scikit-learn joblib` (already in `requirements.txt`). |
| Camera won't open / black frame | Try another `camera.index`, close other apps using the webcam, check OS camera permissions. |
| Low FPS | Set `hand_tracking.model_complexity: 0`, lower `camera.width/height`, or `max_num_hands: 1`. |
| Analysis/gesture panels look empty in the terminal | They update in the **GUI panels**, not the terminal. |

---

## 19. Development roadmap

- **✅ Phase 1 — Foundation:** scaffold, typed config, logging, live webcam,
  hand detection + skeleton/landmarks/bbox, GUI shell (features 1, 2, 3, 13
  partial, 14).
- **✅ Phase 2 — Hand analysis:** finger states, human-readable description,
  orientation, measurements (features 4, 5, 7, 8).
- **✅ Phase 3 — Gesture recognition & ML:** feature extraction, rule-based
  recognizer, Random Forest classifier, real-time prediction with
  probabilities/timings, trainer with full metrics, synthetic dataset generator
  (features 6, 9, 11, 12).
- **⏳ Phase 4 — Data & tooling:** dataset recorder, custom gestures, video
  recording, Settings dialog, performance/GPU tuning (features 10, 13 remainder,
  15).
- **⏳ Phase 5 — Documentation & polish:** expanded API docs, class diagrams,
  more test coverage (features 16, 17).
- **🔭 Future (feature 18):** dynamic/sequence gestures, sign-language
  recognition, tracking history, multi-person, PyTorch/XGBoost/Transformer/YOLO
  backends, mouse/keyboard control, and more — all enabled by the
  `GestureRecognizer` interface and pluggable `FrameProcessor`.

See `docs/ROADMAP.md` for the living version.

---

## 20. Extending the project

### Add a new gesture backend (e.g. PyTorch)
1. Subclass `GestureRecognizer` in `gestures/`, implement `predict()`, `labels`,
   and `is_ready()`, returning a `GestureResult`.
2. Reuse `gestures/features.extract_features()` (or define your own).
3. Register it in `FrameProcessor._build_recognizer()` (add a `backend` value)
   and add matching keys to `GestureConfig` + `config.yaml`.

### Add a new rule-based gesture
Add a constant to `gestures/labels.py` and a branch in
`RuleBasedRecognizer._classify()`.

### Train on your own data (before the Phase 4 recorder)
Produce a CSV in the `training/dataset.py` format (44 features + `label`) and run
`python -m training.trainer --dataset <your.csv> --out models/gesture_rf.joblib`.

### Change what's drawn
Edit `vision/drawing.py` (`LandmarkPainter`) and toggle via `DisplayConfig`.

---

## 21. Glossary

- **Landmark** — one of 21 keypoints on a hand (x, y normalized, z relative
  depth).
- **Handedness** — whether MediaPipe classifies a hand as "Left" or "Right".
- **MCP / PIP / DIP / IP / TIP** — finger joints (Metacarpophalangeal /
  Proximal / Distal Interphalangeal / Interphalangeal (thumb) / fingertip).
- **Hand scale** — reference length = distance(wrist, middle-MCP); used to make
  measurements distance-invariant.
- **Openness / grip / spread** — normalized `[0,1]` measures of how open the
  hand is, how curled (grip), and how far fingers are spread.
- **Feature vector** — the 44-number representation of a hand fed to the ML
  model.
- **Model bundle** — the joblib dict `{model, labels, feature_dim, metrics,
  created_at}` saved by the trainer and read by the ML recognizer.
- **Rule-based vs ML** — deterministic heuristics vs a trained statistical
  classifier; both implement the same `GestureRecognizer` interface.

---

## 22. License

Released under the **MIT License** — see the [`LICENSE`](LICENSE) file.

---

*Built incrementally with a clean, testable, and extensible architecture.
Contributions and new recognizer backends are welcome via the
`GestureRecognizer` interface.*
