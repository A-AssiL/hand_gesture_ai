"""Unit tests for the dependency-light hand analysis module (Phase 2).

These build synthetic :class:`HandResult` objects from explicit landmark
coordinates, so they run without OpenCV, MediaPipe, or a camera.
"""
from __future__ import annotations

from typing import List, Tuple

from vision.analysis import HandAnalyzer
from vision.landmarks import HandResult, Landmark


def _hand(coords: List[Tuple[float, float]], handedness: str = "Right") -> HandResult:
    """Build a HandResult from 21 (x, y) pairs (z defaults to 0)."""
    assert len(coords) == 21, "expected 21 landmarks"
    landmarks = [Landmark(x=x, y=y, z=0.0) for x, y in coords]
    return HandResult(landmarks=landmarks, handedness=handedness, score=0.99)


# An open right hand, fingers pointing up (image y grows downward).
_OPEN_HAND: List[Tuple[float, float]] = [
    (0.50, 0.95),  # 0  wrist
    (0.42, 0.88),  # 1  thumb cmc
    (0.36, 0.82),  # 2  thumb mcp
    (0.31, 0.77),  # 3  thumb ip
    (0.26, 0.73),  # 4  thumb tip
    (0.46, 0.70),  # 5  index mcp
    (0.45, 0.58),  # 6  index pip
    (0.44, 0.50),  # 7  index dip
    (0.44, 0.42),  # 8  index tip
    (0.52, 0.69),  # 9  middle mcp
    (0.52, 0.56),  # 10 middle pip
    (0.52, 0.47),  # 11 middle dip
    (0.52, 0.39),  # 12 middle tip
    (0.58, 0.70),  # 13 ring mcp
    (0.59, 0.58),  # 14 ring pip
    (0.60, 0.50),  # 15 ring dip
    (0.61, 0.43),  # 16 ring tip
    (0.63, 0.73),  # 17 pinky mcp
    (0.65, 0.63),  # 18 pinky pip
    (0.66, 0.57),  # 19 pinky dip
    (0.67, 0.52),  # 20 pinky tip
]

# A closed fist: finger tips curled back toward the palm.
_FIST: List[Tuple[float, float]] = [
    (0.50, 0.95),  # 0  wrist
    (0.42, 0.88),  # 1  thumb cmc
    (0.38, 0.83),  # 2  thumb mcp
    (0.40, 0.80),  # 3  thumb ip
    (0.45, 0.80),  # 4  thumb tip (across palm)
    (0.46, 0.72),  # 5  index mcp
    (0.45, 0.63),  # 6  index pip
    (0.46, 0.67),  # 7  index dip
    (0.47, 0.71),  # 8  index tip (curled)
    (0.52, 0.71),  # 9  middle mcp
    (0.52, 0.61),  # 10 middle pip
    (0.52, 0.66),  # 11 middle dip
    (0.52, 0.71),  # 12 middle tip (curled)
    (0.58, 0.72),  # 13 ring mcp
    (0.59, 0.62),  # 14 ring pip
    (0.58, 0.67),  # 15 ring dip
    (0.57, 0.72),  # 16 ring tip (curled)
    (0.63, 0.74),  # 17 pinky mcp
    (0.65, 0.66),  # 18 pinky pip
    (0.63, 0.70),  # 19 pinky dip
    (0.62, 0.74),  # 20 pinky tip (curled)
]


def test_open_hand_all_fingers_extended() -> None:
    analysis = HandAnalyzer().analyze(_hand(_OPEN_HAND))
    assert analysis.finger_states.count_extended == 5
    assert analysis.finger_states.states["Thumb"] is True
    assert analysis.orientation.pointing == "Up"
    # A fully open hand should read as highly open / low grip.
    assert analysis.measurements.openness > 0.6
    assert analysis.measurements.grip < 0.4


def test_fist_no_fingers_extended() -> None:
    analysis = HandAnalyzer().analyze(_hand(_FIST))
    assert analysis.finger_states.count_extended == 0
    assert analysis.finger_states.states["Thumb"] is False
    assert analysis.measurements.grip > 0.5


def test_labels_and_description() -> None:
    analysis = HandAnalyzer().analyze(_hand(_OPEN_HAND))
    labels = analysis.finger_states.as_labels()
    assert set(labels) == {"Thumb", "Index", "Middle", "Ring", "Pinky"}
    assert all(v == "Extended" for v in labels.values())
    assert "Right hand" in analysis.description
    assert "extended" in analysis.description


def test_handedness_affects_facing() -> None:
    right = HandAnalyzer().analyze(_hand(_OPEN_HAND, handedness="Right"))
    left = HandAnalyzer().analyze(_hand(_OPEN_HAND, handedness="Left"))
    # Same geometry with opposite handedness flips the palm/back reading.
    assert right.orientation.facing != left.orientation.facing
