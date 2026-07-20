"""Tests for geometry helpers."""
import math
import unittest

from utils.geometry import angle_between, euclidean_distance, normalize, vector_angle_deg


class GeometryTest(unittest.TestCase):
    def test_euclidean_distance(self) -> None:
        self.assertAlmostEqual(euclidean_distance((0, 0), (3, 4)), 5.0)

    def test_right_angle(self) -> None:
        self.assertAlmostEqual(angle_between((1, 0), (0, 0), (0, 1)), 90.0, places=5)

    def test_straight_angle(self) -> None:
        self.assertAlmostEqual(angle_between((-1, 0), (0, 0), (1, 0)), 180.0, places=5)

    def test_vector_angle(self) -> None:
        self.assertAlmostEqual(vector_angle_deg((0, 0), (1, 0)), 0.0, places=5)
        self.assertAlmostEqual(vector_angle_deg((0, 0), (0, 1)), 90.0, places=5)

    def test_normalize_range(self) -> None:
        result = normalize([0, 5, 10])
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[-1], 1.0)

    def test_normalize_constant(self) -> None:
        result = normalize([4, 4, 4])
        self.assertTrue(all(v == 0.0 for v in result))


if __name__ == "__main__":
    unittest.main()
