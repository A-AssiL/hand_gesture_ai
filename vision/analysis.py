"""Hand analysis: finger states, orientation, measurements, and description.

This module turns the raw 21-landmark :class:`~vision.landmarks.HandResult`
into higher-level, human-meaningful information (Phase 2 features 4, 5, 7, 8):

* :class:`FingerStates`   - per-finger extended / folded booleans (feature 4)
* :class:`HandOrientation` - palm facing, pointing direction, rotation (feature 7)
* :class:`HandMeasurements` - palm size, openness, grip, spread, angles (feature 8)
* :class:`HandAnalysis`    - the aggregate, including a human-readable
  ``description`` string (feature 5)

Everything here is deliberately dependency-light (numpy only, via
:mod:`utils.geometry`) so it can be unit-tested without OpenCV, MediaPipe, or
Qt installed. The heuristics are orientation-tolerant: finger extension is
decided by comparing joint-to-reference distances rather than raw image axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from utils.geometry import angle_between, euclidean_distance, vector_angle_deg
from vision.landmarks import HandLandmark as LM
from vision.landmarks import HandResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import AnalysisConfig

# Ordered finger names used across the app/UI.
FINGER_NAMES: Tuple[str, ...] = ("Thumb", "Index", "Middle", "Ring", "Pinky")

# (mcp, pip, tip) landmark indices per finger. For the thumb we use
# (mcp, ip, tip) since it has no PIP joint.
_FINGER_JOINTS: Dict[str, Tuple[int, int, int]] = {
    "Thumb": (LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP),
    "Index": (LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_TIP),
    "Middle": (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP),
    "Ring": (LM.RING_MCP, LM.RING_PIP, LM.RING_TIP),
    "Pinky": (LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_TIP),
}


def _clamp01(value: float) -> float:
    """Clamp a value into the ``[0, 1]`` range."""
    return max(0.0, min(1.0, value))


@dataclass
class FingerStates:
    """Per-finger extended/folded flags (feature 4)."""

    states: Dict[str, bool]  # finger name -> is_extended

    @property
    def extended(self) -> List[str]:
        """Names of extended fingers, in canonical order."""
        return [name for name in FINGER_NAMES if self.states.get(name)]

    @property
    def folded(self) -> List[str]:
        """Names of folded fingers, in canonical order."""
        return [name for name in FINGER_NAMES if not self.states.get(name)]

    @property
    def count_extended(self) -> int:
        """Number of extended fingers (0-5)."""
        return sum(1 for name in FINGER_NAMES if self.states.get(name))

    def as_labels(self) -> Dict[str, str]:
        """Return ``{finger: 'Extended'|'Folded'}`` for display."""
        return {
            name: ("Extended" if self.states.get(name) else "Folded")
            for name in FINGER_NAMES
        }


@dataclass
class HandOrientation:
    """Coarse hand orientation in the image plane (feature 7)."""

    facing: str        # "Palm" or "Back" (which side faces the camera)
    pointing: str      # "Up" | "Down" | "Left" | "Right"
    rotation_deg: float  # in-plane rotation of the wrist->middle-MCP axis (0-360)
    tilt_deg: float      # tilt of the knuckle line vs. horizontal (-90..90)


@dataclass
class HandMeasurements:
    """Geometric measurements of the hand (feature 8).

    Sizes are expressed in normalized image units (0-1). ``openness``,
    ``grip`` and ``spread`` are ratios in ``[0, 1]``.
    """

    palm_width: float
    palm_height: float
    hand_length: float
    openness: float
    grip: float
    spread: float
    finger_angles: Dict[str, float] = field(default_factory=dict)   # PIP/IP angle, deg
    tip_distances: Dict[str, float] = field(default_factory=dict)   # tip->wrist / scale


@dataclass
class HandAnalysis:
    """Aggregate analysis for a single detected hand (features 4/5/7/8)."""

    handedness: str
    score: float
    finger_states: FingerStates
    orientation: HandOrientation
    measurements: HandMeasurements
    description: str = ""


class HandAnalyzer:
    """Compute a :class:`HandAnalysis` from a :class:`HandResult`.

    Args:
        config: Optional :class:`~config.settings.AnalysisConfig`. When omitted,
            built-in defaults are used, so the analyzer is usable standalone in
            tests.
    """

    def __init__(self, config: Optional["AnalysisConfig"] = None) -> None:
        self._straight_deg = getattr(config, "finger_straight_deg", 160.0)
        self._curled_deg = getattr(config, "finger_curled_deg", 30.0)
        self._spread_max_deg = getattr(config, "spread_max_deg", 50.0)

    # ------------------------------------------------------------------ API
    def analyze(self, hand: HandResult) -> HandAnalysis:
        """Run the full analysis pipeline for one hand."""
        finger_states = self._finger_states(hand)
        orientation = self._orientation(hand)
        measurements = self._measurements(hand, finger_states)
        description = self._describe(hand, finger_states, orientation, measurements)
        return HandAnalysis(
            handedness=hand.handedness,
            score=hand.score,
            finger_states=finger_states,
            orientation=orientation,
            measurements=measurements,
            description=description,
        )

    # -------------------------------------------------------------- helpers
    def _hand_scale(self, hand: HandResult) -> float:
        """Reference length (wrist -> middle MCP); never zero."""
        scale = euclidean_distance(
            hand.landmark(LM.WRIST).as_tuple(),
            hand.landmark(LM.MIDDLE_MCP).as_tuple(),
        )
        return scale if scale > 1e-6 else 1e-6

    def _finger_states(self, hand: HandResult) -> FingerStates:
        """Decide extended/folded for each finger.

        Four fingers: extended when the tip is farther from the wrist than the
        PIP joint. Thumb: extended when the tip is farther from the opposite
        (pinky) MCP than the IP joint. Both tests are orientation-independent.
        """
        wrist = hand.landmark(LM.WRIST).as_tuple()
        states: Dict[str, bool] = {}
        for name in ("Index", "Middle", "Ring", "Pinky"):
            _mcp, pip, tip = _FINGER_JOINTS[name]
            d_tip = euclidean_distance(wrist, hand.landmark(tip).as_tuple())
            d_pip = euclidean_distance(wrist, hand.landmark(pip).as_tuple())
            states[name] = d_tip > d_pip

        pinky_mcp = hand.landmark(LM.PINKY_MCP).as_tuple()
        d_thumb_tip = euclidean_distance(pinky_mcp, hand.landmark(LM.THUMB_TIP).as_tuple())
        d_thumb_ip = euclidean_distance(pinky_mcp, hand.landmark(LM.THUMB_IP).as_tuple())
        states["Thumb"] = d_thumb_tip > d_thumb_ip
        return FingerStates(states=states)

    def _orientation(self, hand: HandResult) -> HandOrientation:
        wrist = hand.landmark(LM.WRIST)
        index_mcp = hand.landmark(LM.INDEX_MCP)
        pinky_mcp = hand.landmark(LM.PINKY_MCP)
        middle_mcp = hand.landmark(LM.MIDDLE_MCP)

        # Palm vs. back: sign of the 2D cross product of the wrist->index and
        # wrist->pinky vectors, disambiguated by handedness.
        v_index = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)
        v_pinky = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y)
        cross = v_index[0] * v_pinky[1] - v_index[1] * v_pinky[0]
        oriented = cross if hand.handedness == "Right" else -cross
        facing = "Palm" if oriented > 0 else "Back"

        # Pointing direction from the overall hand axis (wrist -> middle MCP).
        # Image y grows downward, so negate dy to make "up" positive.
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        pointing = self._direction_label(dx, dy)
        rotation_deg = vector_angle_deg(wrist.as_tuple(), middle_mcp.as_tuple())

        # Tilt of the knuckle line (index MCP -> pinky MCP) vs. horizontal.
        import math

        tilt = math.degrees(
            math.atan2(pinky_mcp.y - index_mcp.y, pinky_mcp.x - index_mcp.x)
        )
        if tilt > 90:
            tilt -= 180
        elif tilt < -90:
            tilt += 180
        return HandOrientation(
            facing=facing,
            pointing=pointing,
            rotation_deg=rotation_deg,
            tilt_deg=tilt,
        )

    @staticmethod
    def _direction_label(dx: float, dy: float) -> str:
        """Map a 2D vector to Up/Down/Left/Right (image coordinates)."""
        import math

        # Flip dy so that "up" on screen is a positive angle.
        angle = math.degrees(math.atan2(-dy, dx))
        if -45 <= angle <= 45:
            return "Right"
        if 45 < angle <= 135:
            return "Up"
        if angle > 135 or angle < -135:
            return "Left"
        return "Down"

    def _measurements(
        self, hand: HandResult, finger_states: FingerStates
    ) -> HandMeasurements:
        scale = self._hand_scale(hand)
        wrist = hand.landmark(LM.WRIST).as_tuple()

        palm_width = euclidean_distance(
            hand.landmark(LM.INDEX_MCP).as_tuple(),
            hand.landmark(LM.PINKY_MCP).as_tuple(),
        )
        palm_height = euclidean_distance(
            wrist, hand.landmark(LM.MIDDLE_MCP).as_tuple()
        )
        hand_length = euclidean_distance(
            wrist, hand.landmark(LM.MIDDLE_TIP).as_tuple()
        )

        finger_angles: Dict[str, float] = {}
        straightness: List[float] = []
        tip_distances: Dict[str, float] = {}
        span = max(self._straight_deg - self._curled_deg, 1e-6)
        for name in FINGER_NAMES:
            mcp, pip, tip = _FINGER_JOINTS[name]
            angle = angle_between(
                hand.landmark(mcp).as_tuple(),
                hand.landmark(pip).as_tuple(),
                hand.landmark(tip).as_tuple(),
            )
            finger_angles[name] = angle
            straightness.append(_clamp01((angle - self._curled_deg) / span))
            tip_distances[name] = (
                euclidean_distance(wrist, hand.landmark(tip).as_tuple()) / scale
            )

        openness = sum(straightness) / len(straightness) if straightness else 0.0
        grip = 1.0 - openness
        spread = self._spread(hand)
        return HandMeasurements(
            palm_width=palm_width,
            palm_height=palm_height,
            hand_length=hand_length,
            openness=openness,
            grip=grip,
            spread=spread,
            finger_angles=finger_angles,
            tip_distances=tip_distances,
        )

    def _spread(self, hand: HandResult) -> float:
        """Fan angle between the index and pinky finger directions, normalized."""
        index_dir = (
            hand.landmark(LM.INDEX_TIP).x - hand.landmark(LM.INDEX_MCP).x,
            hand.landmark(LM.INDEX_TIP).y - hand.landmark(LM.INDEX_MCP).y,
        )
        pinky_dir = (
            hand.landmark(LM.PINKY_TIP).x - hand.landmark(LM.PINKY_MCP).x,
            hand.landmark(LM.PINKY_TIP).y - hand.landmark(LM.PINKY_MCP).y,
        )
        # Reuse angle_between with a shared origin at (0, 0).
        angle = angle_between(index_dir, (0.0, 0.0), pinky_dir)
        return _clamp01(angle / self._spread_max_deg)

    def _describe(
        self,
        hand: HandResult,
        finger_states: FingerStates,
        orientation: HandOrientation,
        measurements: HandMeasurements,
    ) -> str:
        """Build a natural-language description of the hand (feature 5)."""
        side = "palm" if orientation.facing == "Palm" else "back"
        count = finger_states.count_extended
        if count == 0:
            fingers = "all fingers folded"
        elif count == 5:
            fingers = "all fingers extended"
        else:
            names = ", ".join(finger_states.extended)
            fingers = f"{count}/5 fingers extended ({names})"
        return (
            f"{hand.handedness} hand, {side} facing the camera, "
            f"pointing {orientation.pointing.lower()}. {fingers}. "
            f"Openness {measurements.openness * 100:.0f}%, "
            f"grip {measurements.grip * 100:.0f}%, "
            f"spread {measurements.spread * 100:.0f}%."
        )
