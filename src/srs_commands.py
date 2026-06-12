"""
SRS tracking command vocabulary (BENAX single-speaker camera system).

Maps internal pan/lock state to assessor-facing motor status labels:
MOVED_LEFT, MOVED_RIGHT, CENTERED, STOPPED, OUT_OF_FRAME, SCAN.
"""

from enum import Enum
from typing import Optional, Tuple


class TrackingCommand(str, Enum):
    MOVED_LEFT = "MOVED_LEFT"
    MOVED_RIGHT = "MOVED_RIGHT"
    CENTERED = "CENTERED"
    STOPPED = "STOPPED"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    SCAN = "SCAN"
    IDLE = "IDLE"


def derive_command(
    system_state: str,
    pan_label: Optional[str],
    servo_angle: float,
    prev_servo_angle: float,
    commanded_angle: Optional[int],
    in_grace_period: bool = False,
) -> TrackingCommand:
    """
    Convert runtime state into an SRS motor-status command.

    system_state: IDLE | TRACKING | LOCKED | SEARCHING
    pan_label: centered | tracking | searching (from PanTracker.track)
    """
    if system_state == "IDLE":
        return TrackingCommand.IDLE

    if system_state == "SEARCHING":
        return TrackingCommand.SCAN

    if in_grace_period or (system_state == "TRACKING" and pan_label is None):
        return TrackingCommand.OUT_OF_FRAME

    if pan_label == "centered" or system_state == "LOCKED":
        return TrackingCommand.CENTERED

    if commanded_angle is None:
        return TrackingCommand.STOPPED

    delta = servo_angle - prev_servo_angle
    if delta > 0.5:
        return TrackingCommand.MOVED_RIGHT
    if delta < -0.5:
        return TrackingCommand.MOVED_LEFT
    return TrackingCommand.STOPPED


def motor_command_label(cmd: TrackingCommand) -> str:
    """Short motor command alias used in evidence logs (SRS: LEFT/RIGHT/STOP/SCAN)."""
    mapping = {
        TrackingCommand.MOVED_LEFT: "LEFT",
        TrackingCommand.MOVED_RIGHT: "RIGHT",
        TrackingCommand.CENTERED: "STOP",
        TrackingCommand.STOPPED: "STOP",
        TrackingCommand.OUT_OF_FRAME: "STOP",
        TrackingCommand.SCAN: "SCAN",
        TrackingCommand.IDLE: "STOP",
    }
    return mapping.get(cmd, "STOP")
