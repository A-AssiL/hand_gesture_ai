# Architecture

## Design principles

The application follows a **layered, SOLID-oriented** design. Each layer depends
only on the layer(s) below it and communicates through small, typed contracts:

```
         +---------------------------------------------------+
  UI     |  ui/ : MainWindow, panels, VideoThread, theme     |
         +-------------------------+-------------------------+
                                   | Qt signals (frames, hands, fps)
         +-------------------------v-------------------------+
  Core   |  core/ : Camera, FPSCounter, FrameProcessor       |
         +-------------------------+-------------------------+
                                   | BGR frames / ProcessResult
         +-------------------------v-------------------------+
 Vision  |  vision/ : HandTracker, LandmarkPainter, models   |
         +-------------------------+-------------------------+
                                   | HandResult (framework-agnostic)
         +-------------------------v-------------------------+
  AI     |  gestures/ : GestureRecognizer interface (+ impls) |
         +---------------------------------------------------+

  Cross-cutting: config/ (typed YAML settings), utils/ (logging, geometry)
```

### Why this shape?

- **Single Responsibility** — capture, tracking, drawing, and presentation are
  separate classes in separate modules.
- **Dependency Inversion** — the UI and pipeline depend on the abstract
  `GestureRecognizer` interface, not on any specific ML backend, so Random
  Forest / PyTorch / XGBoost implementations are drop-in.
- **Open/Closed** — new per-frame analyzers (finger states, orientation,
  measurements) plug into `FrameProcessor.process()` and enrich `ProcessResult`
  without changing its callers.
- **Testability** — `vision.landmarks`, `utils.geometry`, and `core.fps_counter`
  have no heavy dependencies and are unit-tested directly.

## Data flow (Phase 1)

1. `VideoThread` (a `QThread`) runs the capture loop off the GUI thread.
2. `Camera.read()` grabs a BGR frame (optionally mirrored).
3. `FrameProcessor.process(frame)`:
   - `HandTracker.process()` runs MediaPipe and returns `HandResult[]`.
   - `LandmarkPainter.draw()` overlays skeleton/landmarks/bbox/IDs.
4. The thread emits `frame_ready`, `hands_updated`, and `fps_updated`.
5. `MainWindow` renders the frame in `VideoWidget` and refreshes the panels.

## Threading model

All OpenCV/MediaPipe work happens on the worker thread; only lightweight Qt
widget updates run on the main thread, received via queued signal/slot
connections. This keeps the UI responsive at 30–60 FPS.

## Key classes (Phase 1)

| Class | Module | Responsibility |
|-------|--------|----------------|
| `Config` & sections | `config/settings.py` | Typed, validated settings loaded from YAML |
| `Camera` | `core/camera.py` | Open/close webcam, resolution, enumeration |
| `FPSCounter` | `core/fps_counter.py` | Rolling-average FPS |
| `FrameProcessor` | `core/engine.py` | Orchestrates the per-frame pipeline |
| `HandTracker` | `vision/hand_tracker.py` | MediaPipe wrapper → `HandResult` |
| `HandResult`, `Landmark` | `vision/landmarks.py` | Framework-agnostic detection model |
| `LandmarkPainter` | `vision/drawing.py` | Draw overlays on frames |
| `GestureRecognizer` | `gestures/base.py` | Abstract classifier contract |
| `VideoThread` | `ui/video_thread.py` | Threaded capture + processing |
| `MainWindow` | `ui/main_window.py` | Window, controls, panels, wiring |

## Extension points

- **New analyzer:** add a class in `vision/` or `gestures/`, call it in
  `FrameProcessor.process()`, add fields to `ProcessResult`, and surface them in
  a panel.
- **New ML backend:** implement `GestureRecognizer` in `gestures/` and select it
  from configuration.
- **New panel:** add a `QGroupBox` in `ui/panels.py` and wire a slot in
  `MainWindow`.
