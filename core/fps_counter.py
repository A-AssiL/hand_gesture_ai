"""A small rolling-average FPS counter."""
from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque


class FPSCounter:
    """Compute frames-per-second over a sliding window of timestamps.

    Args:
        window: Number of recent frames to average over.
        clock: Time source (injectable for testing). Defaults to
            :func:`time.perf_counter`.
    """

    def __init__(self, window: int = 30, clock: Callable[[], float] = time.perf_counter) -> None:
        self._timestamps: Deque[float] = deque(maxlen=max(2, window))
        self._clock = clock

    def tick(self) -> None:
        """Record that a frame has just been produced."""
        self._timestamps.append(self._clock())

    @property
    def fps(self) -> float:
        """Return the current average FPS (0.0 until enough samples exist)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()
