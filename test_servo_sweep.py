#!/usr/bin/env python3
"""
Servo sweep test — turn 0 -> 180 -> 0 via MQTT (no camera).

Use this to verify ESP8266 wiring and full range of motion.

  .venv\\Scripts\\python.exe test_servo_sweep.py
"""

import argparse
import sys
import time

from src import config
from src.mqtt_camera_controller import MQTTCameraController


def wait_for_angle(ctrl: MQTTCameraController, target: int, timeout_sec: float = 12.0) -> bool:
    """Block until the ESP reports the servo has reached target (within 2 deg)."""
    deadline = time.time() + timeout_sec
    last_print = 0.0
    while time.time() < deadline:
        current = int(ctrl.current_angle)
        if abs(current - target) <= 2 and not ctrl.is_moving():
            print(f"  OK: reached {current} deg")
            return True
        now = time.time()
        if now - last_print >= 0.5:
            print(f"  ... at {current} deg (target {target})")
            last_print = now
        time.sleep(0.05)
    print(f"  WARN: timeout at {int(ctrl.current_angle)} deg (wanted {target})")
    return False


def go_to(ctrl: MQTTCameraController, angle: int, label: str) -> bool:
    print(f"\n>> {label} -> {angle} deg")
    if not ctrl.move_to_angle(angle, force=True):
        print("  ERROR: MQTT publish failed")
        return False
    return wait_for_angle(ctrl, angle)


def main() -> bool:
    parser = argparse.ArgumentParser(description="Servo full sweep test (0-180-0)")
    parser.add_argument("--broker", type=str, default=None, help="MQTT broker IP")
    parser.add_argument("--port", type=int, default=None, help="MQTT broker port")
    parser.add_argument("--pause", type=float, default=1.0, help="Pause at each end (seconds)")
    parser.add_argument("--no-center", action="store_true", help="Skip return to center at end")
    args = parser.parse_args()

    lo = config.SERVO_MIN_ANGLE
    hi = config.SERVO_MAX_ANGLE
    center = config.SERVO_CENTER_ANGLE

    print("=" * 50)
    print("Servo sweep test: 0 -> 180 -> 0")
    print(f"Broker: {args.broker or config.MQTT_BROKER_HOST}:{args.port or config.MQTT_BROKER_PORT}")
    print("=" * 50)

    ctrl = MQTTCameraController(broker_host=args.broker, broker_port=args.port)
    if not ctrl.wait_for_connection(timeout_sec=5.0):
        print("ERROR: Cannot connect to MQTT broker")
        return False

    time.sleep(0.5)
    print(f"Starting angle: {int(ctrl.current_angle)} deg")

    steps = [
        (center, f"Center first ({center} deg)"),
        (lo, f"Full left ({lo} deg)"),
        (hi, f"Full right ({hi} deg)"),
        (lo, f"Back to left ({lo} deg)"),
    ]
    if not args.no_center:
        steps.append((center, f"Return to center ({center} deg)"))

    for angle, label in steps:
        if not go_to(ctrl, angle, label):
            ctrl.close()
            return False
        if angle in (lo, hi):
            time.sleep(args.pause)

    ctrl.close()
    print("\nOK: Sweep complete — servo should have moved 0 -> 180 -> 0")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
