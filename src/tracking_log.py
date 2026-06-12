"""
Console logging for lock visibility and servo movement decisions.

Logs are throttled so the terminal stays readable:
- State changes (target lost / found / search started) always print immediately.
- Repeated "hold" or "still missing" messages print at most once per interval.
"""

import logging
import time
from typing import Optional

from . import config

_logger = logging.getLogger("facelocking.tracking")


def setup_tracking_logger() -> None:
    """Configure the tracking logger once at startup."""
    if _logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [TRACK] %(message)s", datefmt="%H:%M:%S")
    )
    _logger.addHandler(handler)
    level = logging.DEBUG if config.VERBOSE_LOGGING else logging.INFO
    _logger.setLevel(level if config.TRACKING_LOG_ENABLED else logging.WARNING)


class TrackingLogger:
    """State-change and servo-decision logger for the tracking loop."""

    def __init__(self):
        setup_tracking_logger()
        self._last_state: Optional[str] = None
        self._last_servo_reason: Optional[str] = None
        self._last_status_time = 0.0
        self._last_servo_log_time = 0.0

    def _should_repeat(self, now: float) -> bool:
        return (now - self._last_status_time) >= config.TRACKING_STATUS_INTERVAL_SEC

    # ------------------------------------------------------------------ target
    def target_visible(self, lock_name: str, track_id: int, center: tuple, servo_angle: float) -> None:
        now = time.time()
        if self._last_state in (None, "visible"):
            if self._last_state == "visible" and not self._should_repeat(now):
                return
            _logger.info(
                "TARGET IN FRAME: %s (track #%d) at (%.0f, %.0f) | servo %d°",
                lock_name, track_id, center[0], center[1], int(servo_angle),
            )
            self._last_status_time = now
            self._last_state = "visible"
            return

        _logger.info(
            "TARGET REACQUIRED: %s (track #%d) at (%.0f, %.0f) | resuming tracking",
            lock_name, track_id, center[0], center[1],
        )
        self._last_state = "visible"
        self._last_status_time = now
        self._last_servo_reason = None

    def target_lost(self, lock_name: str) -> None:
        if self._last_state == "lost_grace":
            return
        _logger.warning("TARGET OUT OF FRAME: %s left view — grace period started", lock_name)
        self._last_state = "lost_grace"
        self._last_servo_reason = None

    def target_still_missing(self, lock_name: str, lost_for: float) -> None:
        now = time.time()
        if not self._should_repeat(now):
            return
        remaining = max(0.0, config.LOST_TARGET_TIMEOUT - lost_for)
        _logger.warning(
            "TARGET STILL MISSING: %s (%.1fs missing, search in %.1fs)",
            lock_name, lost_for, remaining,
        )
        self._last_status_time = now
        self._last_state = "lost_grace"

    def search_started(self, lock_name: str, last_angle: float, direction: str) -> None:
        if self._last_state == "searching":
            return
        _logger.warning(
            "SEARCH MODE: %s out of frame — servo sweep started (last angle %d°, bias %s)",
            lock_name, int(last_angle), direction,
        )
        self._last_state = "searching"
        self._last_servo_reason = None

    def idle(self) -> None:
        if self._last_state == "idle":
            return
        self._last_state = "idle"
        self._last_servo_reason = None

    def lock_armed(self, lock_name: str) -> None:
        _logger.info(
            "Lock armed: watching for '%s' | search starts after %.1fs out of frame",
            lock_name, config.LOST_TARGET_TIMEOUT,
        )

    # ------------------------------------------------------------------ servo
    def servo_move(self, from_angle: float, to_angle: int, reason: str) -> None:
        key = f"move:{to_angle}:{reason}"
        now = time.time()
        if key == self._last_servo_reason and (now - self._last_servo_log_time) < config.TRACKING_STATUS_INTERVAL_SEC:
            return
        _logger.info(
            "SERVO MOVE: %d° → %d° (%s)",
            int(from_angle), to_angle, reason,
        )
        self._last_servo_reason = key
        self._last_servo_log_time = now

    def servo_hold(self, angle: float, reason: str) -> None:
        key = f"hold:{reason}"
        now = time.time()
        if key == self._last_servo_reason and (now - self._last_servo_log_time) < config.TRACKING_STATUS_INTERVAL_SEC:
            return
        _logger.info("SERVO HOLD: %d° (%s)", int(angle), reason)
        self._last_servo_reason = key
        self._last_servo_log_time = now

    def servo_search_step(self, from_angle: float, to_angle: int, step_index: int) -> None:
        _logger.info(
            "SERVO SEARCH: %d° → %d° (sweep step %d)",
            int(from_angle), to_angle, step_index,
        )
        self._last_servo_reason = f"search:{to_angle}"
        self._last_servo_log_time = time.time()

    def edge_recovery(self, from_angle: float, to_angle: int, side: str) -> None:
        _logger.info(
            "EDGE RECOVERY: %d° → %d° (face on %s — pulling toward center)",
            int(from_angle), to_angle, side,
        )
        self._last_servo_reason = f"recovery:{to_angle}"
        self._last_servo_log_time = time.time()

    def learned_pan_sign(self, side: str, new_sign: float) -> None:
        _logger.warning(
            "LEARNED PAN SIGN: flipped to %+.0f after repeated blocks with face on %s",
            new_sign, side,
        )
