# ✋ AI Hand Gesture Recognition and Analysis

A production-quality desktop application for **real-time hand tracking, gesture
recognition, and hand analysis** using a webcam. Built with **PySide6**,
**OpenCV**, and **MediaPipe Hands**, with a modular architecture designed so new
AI models (scikit-learn today, PyTorch/XGBoost/Transformers tomorrow) can be
added without touching the UI or capture pipeline.

> **Status: Phase 1 complete** — project scaffold, dark-theme GUI, live webcam,
> and real-time hand tracking (skeleton, landmarks, bounding box, landmark IDs).
> Later phases add finger analysis, gesture recognition, orientation,
> measurements, dataset recording, and model training. See
> [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## ✨ Features

### Available now (Phase 1)
- **Live webcam** with adjustable resolution, target 30–60 FPS, live FPS counter, and camera switching.
- **Hand detection** for left, right, or both hands simultaneously.
- **Overlays**: hand skeleton, all 21 landmarks, per-hand bounding box, and landmark IDs.
- **21-landmark extraction** with normalized `x`, `y`, `z` per point.
- **Modern dark-theme UI** with Statistics, Hand Information, Finger States, Detected Gesture, and Logs panels.
- **Logging** to console and a rotating file (`logs/app.log`), mirrored in the UI.
- **Save Screenshot** of the annotated feed.

### Planned (see roadmap)
- Finger state analysis, human-readable hand description, and hand orientation.
- Rule-based + Random Forest gesture recognition (20+ gestures) with real-time confidence.
- Geometric measurements (angles, spread, grip %, palm size, openness).
- Dataset recorder and full training pipeline with metrics.
- PyTorch / XGBoost / Transformer / YOLO backends and many future modules.

---

## 🧱 Project structure

```
hand_gesture_ai/
├── app.py                # Application entry point
├── requirements.txt
├── README.md
├── LICENSE
├── config/               # YAML config + typed settings loader
│   ├── config.yaml
│   └── settings.py
├── core/                 # Camera, FPS counter, processing engine
│   ├── camera.py
│   ├── fps_counter.py
│   └── engine.py
├── vision/               # MediaPipe tracking, landmark models, drawing
│   ├── hand_tracker.py
│   ├── landmarks.py
│   └── drawing.py
├── gestures/             # Recognizer interfaces (implementations in later phases)
│   └── base.py
├── ui/                   # PySide6 GUI (window, panels, video thread, theme)
│   ├── main_window.py
│   ├── panels.py
│   ├── video_thread.py
│   ├── video_widget.py
│   ├── log_handler.py
│   └── theme.py
├── models/               # Trained model artifacts (Phase 3)
├── training/             # Training pipeline (Phase 3)
├── datasets/             # Recorded gesture samples (Phase 5)
├── utils/                # Logging + geometry helpers
│   ├── logger.py
│   └── geometry.py
├── assets/               # Screenshots, recordings, icons
├── tests/                # Unit tests
└── docs/                 # Architecture, roadmap, API notes
```

---

## 🚀 Installation

> Requires **Python 3.12+** and a working webcam.

```bash
# 1. Clone / extract the project, then create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note on MediaPipe:** MediaPipe currently supports Python 3.9–3.12. If you
> are on 3.13+, use a 3.12 environment for full compatibility.

---

## ▶️ Usage

```bash
.\.venv\Scripts\python.exe app.py
```

1. Pick a camera from the **Camera** dropdown.
2. Click **Start Camera** to begin streaming and tracking.
3. Move one or two hands into frame — skeleton, landmarks, IDs, and bounding
   boxes render live, and the panels update in real time.
4. Click **Save Screenshot** to capture the annotated frame.
5. Click **Stop Camera** to end the session.

Buttons for later phases (Capture Dataset, Train Model, Load Model, Record
Video, Settings) are present and will show a short "coming soon" note.

---

## ⚙️ Configuration

All runtime behavior is controlled by [`config/config.yaml`](config/config.yaml):
camera resolution/FPS, MediaPipe confidence thresholds, which overlays to draw,
and logging. Values are validated by `config/settings.py` and fall back to safe
defaults.

---

## 🧪 Tests

The dependency-light unit tests run without a camera, Qt, or MediaPipe:

```bash
python -m unittest discover -s tests -v
```

---

## 📚 Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — architecture & data flow.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — phased delivery plan mapped to the 18 feature areas.

## 📄 License

Released under the [MIT License](LICENSE).
