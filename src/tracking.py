"""
Pan tracking controller for the MQTT servo camera.

TRACKING: smooth centering — face offset maps to a gently filtered pan angle.
SEARCH:  continuous full-range sweep (0-180) until the speaker is reacquired.
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

        self._smooth_angle: float = float(config.SERVO_CENTER_ANGLE)
        self._last_published: int = config.SERVO_CENTER_ANGLE

        self.frames_in_center = 0
        self.center_locked = False

        self.last_error_sign: int = 0
        self.last_known_angle: float = float(config.SERVO_CENTER_ANGLE)

        self.search_manual = False
        self._search_angle: float = float(config.SERVO_CENTER_ANGLE)
        self._search_dir: int = 1
        self._last_search_step_time = 0.0
        self._search_initialized = False

    def reset(self) -> None:
        self._smooth_angle = float(self.current_angle)
        self._last_published = int(round(self._smooth_angle))
        self.frames_in_center = 0
        self.center_locked = False
        self.search_manual = False
        self._search_angle = float(self.current_angle)
        self._search_dir = 1
        self._last_search_step_time = 0.0
        self._search_initialized = False

    @property
    def current_angle(self) -> float:
        if self.mqtt:
            return float(self.mqtt.current_angle)
        return self._smooth_angle

    @property
    def target_angle(self) -> float:
        return self._smooth_angle

    def normalized_error(self, face_center_x: float, frame_width: int) -> float:
        return (face_center_x - frame_width / 2.0) / (frame_width / 2.0)

    def _in_dead_zone(self, face_center_x: float, frame_width: int) -> bool:
        return abs(face_center_x - frame_width / 2.0) < config.CENTER_DEAD_ZONE

    def in_center_zone(self, error: float) -> bool:
        return abs(error) < config.CENTERING_TOLERANCE

    def _publish_angle(self, angle: int, reason: str) -> Optional[int]:
        """Publish only when angle changed enough — prevents wiggly micro-commands."""
        if not self.mqtt:
            if self.log:
                self.log.servo_hold(self.current_angle, "MQTT not connected")
            return None

        from_angle = int(round(self.current_angle))
        if abs(angle - self._last_published) < config.SERVO_MIN_PUBLISH_DELTA:
            return None

        if self.mqtt.move_to_angle(angle):
            self._last_published = angle
            if self.log:
                self.log.servo_move(from_angle, angle, reason)
            return angle

        if self.log:
            self.log.servo_hold(from_angle, "waiting for servo")
        return None

    # --------------------------------------------------------------- tracking
    def track(self, face_center_x: float, frame_width: int) -> Tuple[str, Optional[int]]:
        """
        Smoothly center the face in frame.
        Face left of center -> rotate camera left; face right -> rotate right.
        """
        self._search_initialized = False
        raw_error = self.normalized_error(face_center_x, frame_width)

        if abs(raw_error) > config.CENTERING_TOLERANCE:
            self.last_error_sign = 1 if raw_error > 0 else -1
        self.last_known_angle = self.current_angle

        if self.in_center_zone(raw_error):
            self.frames_in_center += 1
            self.center_locked = self.frames_in_center >= config.FRAMES_TO_LOCK_CENTER
        else:
            self.frames_in_center = 0
            self.center_locked = False

        if self._in_dead_zone(face_center_x, frame_width):
            label = "centered" if self.center_locked else "tracking"
            if self.log:
                self.log.servo_hold(self.current_angle, "face centered")
            return (label, None)

        # Desired pan: face at +error (right of center) -> rotate to bring it center
        desired = config.SERVO_CENTER_ANGLE + (
            raw_error * config.SERVO_PAN_RANGE * config.SERVO_DIRECTION_SIGN
        )
        desired = _clamp(desired, config.SERVO_MIN_ANGLE, config.SERVO_MAX_ANGLE)

        # Smooth approach — move faster when face is far from center, slow when close
        err_mag = min(1.0, abs(raw_error))
        alpha = config.SERVO_SMOOTH_ALPHA + err_mag * (
            config.SERVO_SMOOTH_ALPHA_MAX - config.SERVO_SMOOTH_ALPHA
        )
        step = alpha * (desired - self._smooth_angle)
        step = _clamp(step, -config.SERVO_MAX_STEP_PER_FRAME, config.SERVO_MAX_STEP_PER_FRAME)
        self._smooth_angle = _clamp(
            self._smooth_angle + step,
            config.SERVO_MIN_ANGLE,
            config.SERVO_MAX_ANGLE,
        )

        rounded = int(round(self._smooth_angle))
        side = "right" if raw_error > 0 else "left"
        commanded = self._publish_angle(
            rounded, f"centering face ({side}, err={raw_error:+.2f})"
        )
        label = "centered" if self.center_locked else "tracking"
        return (label, commanded)

    # ----------------------------------------------------------------- search
    def search(self) -> Tuple[str, Optional[int]]:
        """
        Continuous full-range sweep (0 -> 180 -> 0 -> ...) until target found.
        """
        self.frames_in_center = 0
        self.center_locked = False

        if not self.mqtt:
            if self.log:
                self.log.servo_hold(self.current_angle, "search paused — no MQTT")
            return ("searching", None)

        now = time.time()
        if now - self._last_search_step_time < config.SEARCH_STEP_INTERVAL_SEC:
            return ("searching", None)

        self._last_search_step_time = now

        if not self._search_initialized:
            self._search_angle = float(self.current_angle)
            self._search_dir = self.last_error_sign * config.SERVO_DIRECTION_SIGN
            if self._search_dir == 0:
                self._search_dir = 1
            self._search_initialized = True

        step = config.SEARCH_SWEEP_STEP * self._search_dir
        next_angle = self._search_angle + step

        # Full-range scan: wrap 180 -> 0 -> 180 (continuous rotation feel)
        if next_angle > config.SERVO_MAX_ANGLE:
            next_angle = float(config.SERVO_MIN_ANGLE)
        elif next_angle < config.SERVO_MIN_ANGLE:
            next_angle = float(config.SERVO_MAX_ANGLE)
        self._search_angle = next_angle

        self._smooth_angle = self._search_angle
        target = int(round(self._search_angle))
        from_angle = int(round(self.current_angle))

        # Search always publishes (no min-delta gate) so sweep never gets stuck
        commanded = None
        if self.mqtt and self.mqtt.move_to_angle(target):
            self._last_published = target
            commanded = target
            if self.log:
                self.log.servo_move(from_angle, target, f"search sweep -> {target}")
        elif self.log:
            self.log.servo_hold(from_angle, "search waiting for servo")

        return ("searching", commanded)

    # ----------------------------------------------------------------- manual
    def toggle_search(self) -> None:
        self.search_manual = not self.search_manual
        if self.search_manual:
            self._search_angle = float(self.current_angle)
            self._last_search_step_time = 0.0
            self._search_initialized = False

    def force_center(self) -> None:
        self.reset()
        self._smooth_angle = float(config.SERVO_CENTER_ANGLE)
        if self.mqtt:
            self.mqtt.center()
