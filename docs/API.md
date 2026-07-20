# API notes (Phase 1)

This document summarizes the public surface delivered in Phase 1. Full API
documentation with class diagrams is expanded in Phase 5.

## `config.settings`
- `load_config(path=DEFAULT_CONFIG_PATH) -> Config` — load & validate YAML.
- `Config` — aggregates `AppConfig`, `CameraConfig`, `HandTrackingConfig`,
  `DisplayConfig`, `LoggingConfig` (all dataclasses).
- `to_dict(config) -> dict` — serialize back to a plain mapping.

## `utils.logger`
- `setup_logging(logging_config) -> logging.Logger` — configure root logger.
- `get_logger(name) -> logging.Logger` — namespaced child logger.

## `utils.geometry`
- `euclidean_distance(a, b) -> float`
- `angle_between(a, b, c) -> float` — angle at vertex `b`, in degrees.
- `vector_angle_deg(a, b) -> float` — orientation of vector a→b (0–360°).
- `normalize(values) -> np.ndarray` — min-max normalization.

## `core`
- `Camera(config.camera)` — `open(index=None)`, `read()`, `set_resolution(w, h)`,
  `is_open()`, `release()`, `list_available()`.
- `FPSCounter(window=30, clock=perf_counter)` — `tick()`, `fps`, `reset()`.
- `FrameProcessor(config)` — `process(frame) -> ProcessResult`, `close()`.
- `ProcessResult` — `frame`, `hands: List[HandResult]`.

## `vision`
- `HandTracker(config.hand_tracking)` — `process(frame_bgr) -> List[HandResult]`,
  `close()`; usable as a context manager.
- `HandResult` — `handedness`, `score`, `landmarks`, `landmark(i)`,
  `pixel(i, w, h)`, `bounding_box(w, h, padding=20)`.
- `Landmark` — `x`, `y`, `z`, `as_tuple()`, `to_pixel(w, h)`.
- `HandLandmark` — IntEnum of the 21 landmark indices.
- `LandmarkPainter(config.display)` — `draw(frame, hands) -> frame`.
- `draw_fps(frame, fps)` — overlay an FPS counter.

## `gestures`
- `GestureRecognizer` (ABC) — `predict(hand) -> GestureResult`, `labels`,
  `is_ready()`.
- `GestureResult` — `name`, `confidence`, `probabilities`, `inference_ms`.

## `ui`
- `MainWindow(config)` — main window; `start_camera()`, `stop_camera()`,
  `save_screenshot()`.
- `VideoThread(config)` — `QThread` emitting `frame_ready`, `hands_updated`,
  `fps_updated`, `error`; `stop()`, `switch_camera(index)`, `available_cameras()`.
- `VideoWidget` — `show_frame(frame)`, `clear_frame()`.
- Panels: `StatisticsPanel`, `HandInfoPanel`, `FingerStatePanel`,
  `GesturePanel`, `LogPanel`.
