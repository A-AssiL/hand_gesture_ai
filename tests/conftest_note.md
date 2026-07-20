# Running tests

The Phase 1 unit tests are intentionally dependency-light (standard library +
numpy + PyYAML) so they run without a camera, Qt, or MediaPipe:

```bash
cd hand_gesture_ai
python -m unittest discover -s tests -v
```

Tests that would require OpenCV/MediaPipe/PySide6 (camera, tracker, UI) are
covered by manual/integration runs of `python app.py` and will gain automated
coverage in Phase 5.
