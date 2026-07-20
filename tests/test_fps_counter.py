"""Tests for the rolling FPS counter."""
import unittest

from core.fps_counter import FPSCounter


class FPSCounterTest(unittest.TestCase):
    def test_zero_before_two_samples(self) -> None:
        counter = FPSCounter()
        self.assertEqual(counter.fps, 0.0)
        counter.tick()
        self.assertEqual(counter.fps, 0.0)

    def test_computes_fps_from_fake_clock(self) -> None:
        times = iter([0.0, 0.1, 0.2, 0.3])  # 10 FPS spacing
        counter = FPSCounter(window=10, clock=lambda: next(times))
        for _ in range(4):
            counter.tick()
        # 3 intervals over 0.3s -> 10 FPS
        self.assertAlmostEqual(counter.fps, 10.0, places=5)

    def test_reset_clears_samples(self) -> None:
        counter = FPSCounter()
        counter.tick()
        counter.tick()
        counter.reset()
        self.assertEqual(counter.fps, 0.0)


if __name__ == "__main__":
    unittest.main()
