"""Background capture/processing thread.

Running the camera loop off the GUI thread keeps the interface responsive and
lets us hit 30-60 FPS. The thread emits Qt signals with the annotated frame,
the structured detections, and the current FPS so the UI can update reactively.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from PySide6.QtCore import QThread, Signal

from core.camera import Camera
from core.engine import FrameProcessor
from core.fps_counter import FPSCounter
from utils.logger import get_logger
from vision.landmarks import HandResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import Config

logger = get_logger("video_thread")


class VideoThread(QThread):
    """Continuously capture frames, process them, and emit results."""

    frame_ready = Signal(object)   # annotated BGR numpy frame
    hands_updated = Signal(object)  # List[HandResult]
    analysis_updated = Signal(object)  # List[HandAnalysis]
    fps_updated = Signal(float)
    error = Signal(str)

    def __init__(self, config: "Config") -> None:
        super().__init__()
        self._config = config
        self._camera = Camera(config.camera)
        self._processor = FrameProcessor(config)
        self._fps = FPSCounter()
        self._running = False
        self._camera_index = config.camera.index

    def run(self) -> None:  # noqa: D401 - Qt entry point
        """Thread entry point: open the camera and loop until stopped."""
        try:
            self._camera.open(self._camera_index)
        except RuntimeError as exc:
            logger.error("Camera open failed: %s", exc)
            self.error.emit(str(exc))
            return

        self._running = True
        self._fps.reset()
        while self._running:
            frame = self._camera.read()
            if frame is None:
                continue
            try:
                result = self._processor.process(frame)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Frame processing error")
                self.error.emit(str(exc))
                continue
            self._fps.tick()
            self.frame_ready.emit(result.frame)
            self.hands_updated.emit(result.hands)
            self.analysis_updated.emit(result.analyses)
            self.fps_updated.emit(self._fps.fps)

        self._camera.release()

    def stop(self) -> None:
        """Signal the loop to stop, wait for it, and free resources."""
        self._running = False
        self.wait(2000)
        self._processor.close()
        logger.info("VideoThread stopped")

    def switch_camera(self, index: int) -> None:
        """Switch to a different camera index (takes effect on next open)."""
        self._camera_index = index
        if self._running:
            try:
                self._camera.open(index)
            except RuntimeError as exc:
                self.error.emit(str(exc))

    def available_cameras(self) -> List[int]:
        """Return the list of openable camera indices."""
        return self._camera.list_available()
