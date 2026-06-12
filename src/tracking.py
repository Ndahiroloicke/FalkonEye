"""
Pan tracking controller for the MQTT servo camera.

Responsibilities:
- TRACKING: keep the locked face centered using a PID controller on the
  normalized horizontal error, with a pixel dead zone, error smoothing and a
  per-update rate limit (Issue #6).
- SEARCH: when the locked target is lost, run an autonomous, direction-aware,
  expanding sweep that keeps publishing servo waypoints until the target is
  reacquired (Issues #4, #5).

The controller never decides *who* to lock — that is the FaceTracker's job. It
only converts a target's position (or its absence) into servo angles.
"""

import time
from typing import List, Optional, Tuple, TYPE_CHECKING

from . import config
from .mqtt_camera_controller import MQTTCameraController

if TYPE_CHECKING:
    from .tracking_log import TrackingLogger


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class PanTracker:
    """Maps a locked face's horizontal position to smooth servo motion."""

    def __init__(
        self,
        mqtt: Optional[MQTTCameraController] = None,
        logger: Optional["TrackingLogger"] = None,
    ):
        self.mqtt = mqtt
        self.log = logger

        # PID state
        self.smoothed_error: Optional[float] = None
        self.prev_error: float = 0.0
        self.integral: float = 0.0
        self.target_angle: float = float(config.SERVO_CENTER_ANGLE)

        # Centering state
        self.frames_in_center = 0
        self.center_locked = False

        # Reacquisition memory (Issue #5)
        self.last_error_sign: int = 0          # -1 face was left, +1 face was right
        self.last_known_angle: float = float(config.SERVO_CENTER_ANGLE)

        # Search state (Issue #4)
        self.search_manual = False
        self._search_waypoints: List[int] = []
        self._search_index = 0
        self._last_search_step_time = 0.0

    # ------------------------------------------------------------------ utils
    def reset(self) -> None:
        self.smoothed_error = None
        self.prev_error = 0.0
        self.integral = 0.0
        self.frames_in_center = 0
        self.center_locked = False
        self.search_manual = False
        self._search_waypoints = []
        self._search_index = 0

    @property
    def current_angle(self) -> float:
        if self.mqtt:
            return float(self.mqtt.current_angle)
        return self.target_angle

    def normalized_error(self, face_center_x: float, frame_width: int) -> float:
        return (face_center_x - frame_width / 2.0) / (frame_width / 2.0)

    def _in_dead_zone(self, face_center_x: float, frame_width: int) -> bool:
        return abs(face_center_x - frame_width / 2.0) < config.CENTER_DEAD_ZONE

    def in_center_zone(self, error: float) -> bool:
        return abs(error) < config.CENTERING_TOLERANCE

    # --------------------------------------------------------------- tracking
    def track(self, face_center_x: float, frame_width: int) -> Tuple[str, Optional[int]]:
        """
        PID-control the servo toward centering the face.
        Returns (state_label, commanded_angle | None).
        """
        # Stop search the moment we have a target.
        self._search_waypoints = []
        self._search_index = 0

        raw_error = self.normalized_error(face_center_x, frame_width)

        # Remember exit direction for later reacquisition.
        if abs(raw_error) > config.CENTERING_TOLERANCE:
            self.last_error_sign = 1 if raw_error > 0 else -1
        self.last_known_angle = self.current_angle

        # Center bookkeeping for the "centered" state label.
        if self.in_center_zone(raw_error):
            self.frames_in_center += 1
            self.center_locked = self.frames_in_center >= config.FRAMES_TO_LOCK_CENTER
        else:
            self.frames_in_center = 0
            self.center_locked = False

        # Dead zone: do not chase tiny offsets (anti-jitter).
        if self._in_dead_zone(face_center_x, frame_width):
            self.prev_error = raw_error
            self.integral = 0.0
            label = "centered" if self.center_locked else "tracking"
            if self.log:
                reason = "face centered in dead zone" if self.center_locked else "face in dead zone"
                self.log.servo_hold(self.current_angle, reason)
            return (label, None)

        # Error smoothing (EMA) to suppress detection noise.
        a = config.SMOOTHING_FACTOR
        if self.smoothed_error is None:
            self.smoothed_error = raw_error
        else:
            self.smoothed_error = a * raw_error + (1.0 - a) * self.smoothed_error
        error = self.smoothed_error

        # PID terms.
        self.integral = _clamp(
            self.integral + error,
            -config.SERVO_PID_I_CLAMP / max(config.SERVO_PID_KI, 1e-6) if config.SERVO_PID_KI else 0.0,
            config.SERVO_PID_I_CLAMP / max(config.SERVO_PID_KI, 1e-6) if config.SERVO_PID_KI else 0.0,
        )
        derivative = error - self.prev_error
        self.prev_error = error

        delta = (
            config.SERVO_PID_KP * error
            + config.SERVO_PID_KI * self.integral
            + config.SERVO_PID_KD * derivative
        )
        delta *= config.SERVO_DIRECTION_SIGN

        # Rate limit (prevents violent servo jumps).
        delta = _clamp(delta, -config.SERVO_MAX_SPEED, config.SERVO_MAX_SPEED)

        new_angle = _clamp(
            self.current_angle + delta,
            config.SERVO_MIN_ANGLE,
            config.SERVO_MAX_ANGLE,
        )
        self.target_angle = new_angle

        commanded = None
        rounded = int(round(new_angle))
        from_angle = self.current_angle
        if not self.mqtt:
            if self.log:
                self.log.servo_hold(from_angle, "MQTT not connected")
            return ("tracking", None)

        if self.mqtt.move_to_angle(rounded):
            commanded = rounded
            if self.log:
                side = "right" if raw_error > 0 else "left"
                self.log.servo_move(from_angle, rounded, f"centering face ({side}, err={raw_error:+.2f})")
        elif self.log:
            if abs(rounded - from_angle) < 1:
                self.log.servo_hold(from_angle, "already at target angle")
            else:
                self.log.servo_hold(from_angle, "rate limited or command skipped")
        return ("tracking", commanded)

    # ----------------------------------------------------------------- search
    def _build_search_waypoints(self) -> List[int]:
        """Direction-aware, expanding sweep starting from the last known angle."""
        lo, hi = config.SERVO_MIN_ANGLE, config.SERVO_MAX_ANGLE
        step = config.SEARCH_SWEEP_STEP

        # Decide initial direction.
        if config.SEARCH_START_DIRECTION == "left":
            primary = -1
        elif config.SEARCH_START_DIRECTION == "right":
            primary = 1
        else:  # "last"
            primary = self.last_error_sign or 1
        primary *= config.SERVO_DIRECTION_SIGN

        waypoints: List[int] = []
        if config.SEARCH_EXPAND_ENABLED:
            base = int(round(_clamp(self.last_known_angle, lo, hi)))
            waypoints.append(base)
            amp = step
            # Expand outward, primary side first, until both bounds are covered.
            while True:
                first = int(round(_clamp(base + primary * amp, lo, hi)))
                second = int(round(_clamp(base - primary * amp, lo, hi)))
                waypoints.append(first)
                waypoints.append(second)
                if base + amp >= hi and base - amp <= lo:
                    break
                amp += step

        # Always finish with a full edge-to-edge sweep so coverage is complete.
        sweep = list(range(lo, hi + 1, step))
        if primary < 0:
            sweep = list(reversed(sweep))
        waypoints.extend(sweep)
        waypoints.extend(reversed(sweep))
        return waypoints

    def search(self) -> Tuple[str, Optional[int]]:
        """
        Advance the autonomous search sweep. Recognition keeps running in the
        main loop; this only drives the servo. Returns (state, commanded_angle).
        """
        self.smoothed_error = None
        self.integral = 0.0
        self.center_locked = False
        self.frames_in_center = 0

        if not self._search_waypoints:
            self._search_waypoints = self._build_search_waypoints()
            self._search_index = 0
            self._last_search_step_time = 0.0

        if not self.mqtt:
            if self.log:
                self.log.servo_hold(self.current_angle, "search paused — MQTT not connected")
            return ("searching", None)

        now = time.time()
        step_interval = config.SEARCH_SWEEP_STEP / max(config.SEARCH_SWEEP_SPEED, 1e-3)
        if now - self._last_search_step_time < step_interval:
            if self.log:
                self.log.servo_hold(self.current_angle, "search waiting for next sweep step")
            return ("searching", None)

        from_angle = self.current_angle
        angle = self._search_waypoints[self._search_index % len(self._search_waypoints)]
        step_num = self._search_index + 1
        self._search_index += 1
        self._last_search_step_time = now
        commanded = None
        if self.mqtt.move_to_angle(int(angle)):
            commanded = int(angle)
            if self.log:
                self.log.servo_search_step(from_angle, int(angle), step_num)
        elif self.log:
            self.log.servo_hold(from_angle, "search step skipped (rate limited)")
        return ("searching", commanded)

    # ----------------------------------------------------------------- manual
    def toggle_search(self) -> None:
        self.search_manual = not self.search_manual
        if self.search_manual:
            self._search_waypoints = []
            self._search_index = 0

    def force_center(self) -> None:
        self.reset()
        self.target_angle = float(config.SERVO_CENTER_ANGLE)
        if self.mqtt:
            self.mqtt.center()
