"""
Pan tracking controller for the MQTT servo camera.

TRACKING: strict centering — face near the edge gets fast, large corrections;
          face near the middle holds still (anti-wiggle dead zone).
SEARCH:   full 0-180 sweep with wrap when the speaker leaves the frame.
"""

import time
from typing import Optional, Tuple, TYPE_CHECKING

from . import config
from .mqtt_camera_controller import MQTTCameraController

if TYPE_CHECKING:
    from .tracking_log import TrackingLogger


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class PanTracker:
    """Maps a locked face's horizontal position to servo pan commands."""

    def __init__(
        self,
        mqtt: Optional[MQTTCameraController] = None,
        logger: Optional["TrackingLogger"] = None,
    ):
        self.mqtt = mqtt
        self.log = logger

        self._smooth_face_x: Optional[float] = None
        self._smooth_angle: float = float(config.SERVO_CENTER_ANGLE)
        self._last_published: int = config.SERVO_CENTER_ANGLE

        self.frames_in_center = 0
        self.center_locked = False

        self.last_error_sign: int = 0
        self.last_known_angle: float = float(config.SERVO_CENTER_ANGLE)

        self.search_manual = False
        self._search_dir: int = 1
        self._last_search_step_time = 0.0
        self._search_active = False

    def reset(self) -> None:
        self._smooth_face_x = None
        self._smooth_angle = float(self.current_angle)
        self._last_published = int(round(self._smooth_angle))
        self.frames_in_center = 0
        self.center_locked = False
        self.search_manual = False
        self._search_dir = 1
        self._last_search_step_time = 0.0
        self._search_active = False

    def begin_search(self) -> None:
        """Call once when entering SEARCH mode — clears stale motion state."""
        self._search_active = True
        self._last_search_step_time = 0.0
        self._smooth_face_x = None
        if self.mqtt:
            self.mqtt.sync_target_to_current()
        if self.last_error_sign != 0:
            self._search_dir = self.last_error_sign * config.SERVO_DIRECTION_SIGN
        elif self._search_dir == 0:
            self._search_dir = 1

    @property
    def current_angle(self) -> float:
        if self.mqtt:
            return float(self.mqtt.current_angle)
        return self._smooth_angle

    @property
    def target_angle(self) -> float:
        return self._smooth_angle

    def _ema_face_x(self, face_center_x: float, abs_error: float) -> float:
        """Smooth face X; less smoothing when the face is far from center."""
        if self._smooth_face_x is None:
            self._smooth_face_x = face_center_x
            return self._smooth_face_x

        if abs_error >= config.SERVO_EDGE_ERROR:
            alpha = 0.55
        elif abs_error >= config.SERVO_MID_ERROR:
            alpha = 0.35
        else:
            w = max(2, config.SMOOTHING_WINDOW)
            alpha = 2.0 / (w + 1.0)

        self._smooth_face_x = alpha * face_center_x + (1.0 - alpha) * self._smooth_face_x
        return self._smooth_face_x

    def normalized_error(self, face_center_x: float, frame_width: int) -> float:
        return (face_center_x - frame_width / 2.0) / (frame_width / 2.0)

    def _in_dead_zone(self, error: float, face_center_x: float, frame_width: int) -> bool:
        # Never hold still when the face is near the frame edge.
        if abs(error) >= config.SERVO_EDGE_ERROR:
            return False
        if abs(error) < config.SERVO_DEAD_ZONE_NORMALIZED:
            return True
        return abs(face_center_x - frame_width / 2.0) < config.CENTER_DEAD_ZONE

    def in_center_zone(self, error: float) -> bool:
        return abs(error) < config.CENTERING_TOLERANCE

    def _desired_angle(self, raw_error: float) -> float:
        """Map normalized face offset to a servo angle."""
        return _clamp(
            config.SERVO_CENTER_ANGLE
            + raw_error * config.SERVO_PAN_RANGE * config.SERVO_DIRECTION_SIGN,
            config.SERVO_MIN_ANGLE,
            config.SERVO_MAX_ANGLE,
        )

    def _update_filtered_target(self, raw_error: float) -> float:
        raw_desired = self._desired_angle(raw_error)
        abs_err = abs(raw_error)

        if abs_err >= config.SERVO_EDGE_ERROR:
            self._smooth_angle = raw_desired
        else:
            alpha = config.SERVO_SMOOTH_ALPHA
            if abs_err >= config.SERVO_MID_ERROR:
                alpha = config.SERVO_SMOOTH_ALPHA_MAX
            self._smooth_angle += alpha * (raw_desired - self._smooth_angle)

        return self._smooth_angle

    def _tracking_limits(self, abs_error: float) -> Tuple[int, int, bool]:
        """Return (max_step, min_delta, allow_retarget) for this error magnitude."""
        if abs_error >= config.SERVO_EDGE_ERROR:
            return config.SERVO_EDGE_MAX_STEP, config.SERVO_EDGE_MIN_PUBLISH_DELTA, True
        if abs_error >= config.SERVO_MID_ERROR:
            return config.SERVO_CMD_MAX_STEP, 2, True
        return (
            min(config.SERVO_CMD_MAX_STEP, config.SERVO_MAX_STEP_PER_FRAME),
            config.SERVO_MIN_PUBLISH_DELTA,
            False,
        )

    def _tracking_command(
        self,
        desired: int,
        raw_error: float,
        reason: str,
    ) -> Optional[int]:
        if not self.mqtt or not self.mqtt.is_connected:
            if self.log:
                self.log.servo_hold(self.current_angle, "MQTT not connected")
            return None

        from_angle = int(round(self.current_angle))
        diff = desired - from_angle
        abs_err = abs(raw_error)
        max_step, min_delta, allow_retarget = self._tracking_limits(abs_err)

        if abs(diff) < min_delta:
            return None

        if abs(diff) <= max_step:
            cmd = desired
        else:
            cmd = from_angle + (max_step if diff > 0 else -max_step)

        cmd = int(_clamp(cmd, config.SERVO_MIN_ANGLE, config.SERVO_MAX_ANGLE))
        if abs(cmd - from_angle) < min_delta:
            return None

        if not allow_retarget and not self.mqtt.ready_for_command():
            if self.log:
                self.log.servo_hold(from_angle, "servo moving")
            return None

        if self.mqtt.move_to_angle(cmd, allow_retarget=allow_retarget):
            self._last_published = cmd
            if self.log:
                self.log.servo_move(from_angle, cmd, reason)
            return cmd

        if self.log:
            self.log.servo_hold(from_angle, "command skipped")
        return None

    def _search_command(self, desired: int, reason: str) -> Optional[int]:
        if not self.mqtt or not self.mqtt.is_connected:
            if self.log:
                self.log.servo_hold(self.current_angle, "MQTT not connected")
            return None

        from_angle = int(round(self.current_angle))
        desired = int(_clamp(desired, config.SERVO_MIN_ANGLE, config.SERVO_MAX_ANGLE))
        if desired == from_angle:
            return None

        allow_stale = self.mqtt._motion_stale()
        if not self.mqtt.ready_for_command(allow_stale=allow_stale):
            if self.log:
                self.log.servo_hold(from_angle, "search — servo moving")
            return None

        if self.mqtt.move_to_angle(desired, force=True):
            self._last_published = desired
            self._smooth_angle = float(desired)
            if self.log:
                self.log.servo_move(from_angle, desired, reason)
            return desired

        if self.log:
            self.log.servo_hold(from_angle, "search command skipped")
        return None

    def _next_search_angle(self, from_angle: int) -> Tuple[int, str]:
        lo = config.SERVO_MIN_ANGLE
        hi = config.SERVO_MAX_ANGLE
        step = config.SEARCH_SWEEP_STEP

        if self._search_dir >= 0:
            if from_angle >= hi - 2:
                if config.SEARCH_WRAP_AT_END:
                    return lo, "search wrap 180° -> 0°"
                self._search_dir = -1
                return hi - step, "search reverse at max"
            nxt = min(from_angle + step, hi)
            return nxt, f"search sweep -> {nxt}"

        if from_angle <= lo + 2:
            if config.SEARCH_WRAP_AT_END:
                return hi, "search wrap 0° -> 180°"
            self._search_dir = 1
            return lo + step, "search reverse at min"
        nxt = max(from_angle - step, lo)
        return nxt, f"search sweep -> {nxt}"

    # --------------------------------------------------------------- tracking
    def track(self, face_center_x: float, frame_width: int) -> Tuple[str, Optional[int]]:
        """
        Keep the locked face in the middle of the frame.
        Face on the right -> pan camera to pull face toward center (sign from config).
        """
        raw_error = self.normalized_error(face_center_x, frame_width)
        smooth_x = self._ema_face_x(face_center_x, abs(raw_error))
        error = self.normalized_error(smooth_x, frame_width)

        if abs(error) > config.CENTERING_TOLERANCE:
            self.last_error_sign = 1 if error > 0 else -1
        self.last_known_angle = self.current_angle

        if self.in_center_zone(error):
            self.frames_in_center += 1
            self.center_locked = self.frames_in_center >= config.FRAMES_TO_LOCK_CENTER
        else:
            self.frames_in_center = 0
            self.center_locked = False

        if self._in_dead_zone(error, smooth_x, frame_width):
            label = "centered" if self.center_locked else "tracking"
            if self.log:
                self.log.servo_hold(self.current_angle, "face centered — holding")
            return (label, None)

        filtered = self._update_filtered_target(error)
        desired_i = int(round(filtered))

        side = "right" if error > 0 else "left"
        urgency = "strict" if abs(error) >= config.SERVO_EDGE_ERROR else "fine"
        commanded = self._tracking_command(
            desired_i,
            error,
            f"centering ({urgency}, {side}, err={error:+.2f})",
        )
        label = "centered" if self.center_locked else "tracking"
        return (label, commanded)

    # ----------------------------------------------------------------- search
    def search(self) -> Tuple[str, Optional[int]]:
        self.frames_in_center = 0
        self.center_locked = False

        if not self.mqtt or not self.mqtt.is_connected:
            if self.log:
                self.log.servo_hold(self.current_angle, "search paused — no MQTT")
            return ("searching", None)

        now = time.time()
        if now - self._last_search_step_time < config.SEARCH_STEP_INTERVAL_SEC:
            return ("searching", None)

        from_angle = int(round(self.current_angle))
        next_angle, reason = self._next_search_angle(from_angle)

        self._last_search_step_time = now
        commanded = self._search_command(next_angle, reason)
        return ("searching", commanded)

    def toggle_search(self) -> None:
        self.search_manual = not self.search_manual
        if self.search_manual:
            self._last_search_step_time = 0.0

    def force_center(self) -> None:
        self.reset()
        self._smooth_angle = float(config.SERVO_CENTER_ANGLE)
        if self.mqtt:
            self.mqtt.center()
