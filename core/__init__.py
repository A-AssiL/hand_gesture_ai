"""Core runtime: camera capture, FPS tracking, and the processing pipeline.

Heavy, optional dependencies (OpenCV/MediaPipe via ``camera``/``engine``) are
imported lazily so that lightweight utilities such as :class:`FPSCounter` can be
used (and unit-tested) without those packages installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.fps_counter import FPSCounter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.camera import Camera
    from core.engine import FrameProcessor, ProcessResult

__all__ = ["Camera", "FPSCounter", "FrameProcessor", "ProcessResult"]


def __getattr__(name: str):
    """Lazily resolve heavy submodule exports on first access (PEP 562)."""
    if name == "Camera":
        from core.camera import Camera

        return Camera
    if name in {"FrameProcessor", "ProcessResult"}:
        from core import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
