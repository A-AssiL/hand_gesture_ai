# Roadmap

Incremental delivery plan. Each phase is independently runnable and builds on
the previous one. The 18 feature areas from the specification are mapped to the
phase that delivers them.

## ✅ Phase 1 — Foundation (this delivery)
- Project scaffold & professional folder structure
- Typed YAML configuration (`config/`)
- Logging (console + rotating file, mirrored in UI)
- Live webcam: adjustable resolution, FPS counter, camera switching *(Feature 1)*
- Hand detection: 1–2 hands, skeleton, landmarks, bounding box *(Feature 2)*
- 21-landmark extraction with x/y/z and landmark IDs *(Feature 3)*
- Dark-theme GUI shell with all panels & buttons *(Feature 13, partial)*
- Application logging events *(Feature 14)*

## ⏳ Phase 2 — Hand analysis
- Finger state analysis: extended/folded per finger *(Feature 4)*
- Human-readable hand description *(Feature 5)*
- Hand orientation: facing, direction, rotation & wrist angles *(Feature 7)*
- Measurements: angles, distances, palm size, openness, grip %, spread % *(Feature 8)*

## ⏳ Phase 3 — Gesture recognition & ML
- Feature extraction from landmarks
- Rule-based recognizer for core gestures
- Random Forest classifier (scikit-learn) *(Features 6, 9)*
- Real-time prediction: gesture, confidence, probabilities, timings *(Feature 12)*
- Model trainer with accuracy/precision/recall/F1/confusion matrix *(Feature 11)*

## ⏳ Phase 4 — Data & tooling
- Dataset recorder (N samples per gesture into `datasets/`) *(Feature 10)*
- Custom gesture support
- Screenshot (done) + video recording, Settings dialog *(Feature 13, remainder)*
- Performance tuning & optional GPU acceleration *(Feature 15)*

## ⏳ Phase 5 — Documentation & polish
- Expanded API docs, class diagrams *(Feature 16)*
- Test coverage expansion *(Feature 17)*

## 🔭 Future expansion *(Feature 18)*
Sign-language recognition, dynamic gestures, tracking history, multi-person
detection, voice assistant, mouse/keyboard control, VR/AR, robotic-hand control,
gesture macros, custom AI models, and cloud model updates — all enabled by the
`GestureRecognizer` interface and the pluggable `FrameProcessor` pipeline.
