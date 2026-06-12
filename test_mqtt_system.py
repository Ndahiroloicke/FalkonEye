#!/usr/bin/env python3
"""Test MQTT broker connectivity and servo commands."""

import argparse
import sys
import time

from src import config
from src.mqtt_camera_controller import MQTTCameraController


def test_mqtt(broker: str = None, port: int = None) -> bool:
    print("=" * 60)
    print(" MQTT System Test")
    print("=" * 60)

    ctrl = MQTTCameraController(broker_host=broker, broker_port=port)
    time.sleep(2)

    if not ctrl.is_connected:
        print("✗ FAILED: Cannot connect to MQTT broker")
        print(f"  Broker: {broker or config.MQTT_BROKER_HOST}:{port or config.MQTT_BROKER_PORT}")
        return False

    print("✓ Broker connected")

    print("\nTest 1: Center")
    ctrl.center()
    time.sleep(2)

    print("Test 2: Move left")
    ctrl.move_left()
    time.sleep(2)

    print("Test 3: Move right")
    ctrl.move_right()
    time.sleep(2)

    print("Test 4: Absolute angle 90")
    ctrl.move_to_angle(90)
    time.sleep(2)

    print("Test 5: Sweep positions")
    for angle in [30, 60, 90, 120, 90]:
        ctrl.move_to_angle(angle)
        time.sleep(1)

    ctrl.close()
    print("\n✓ MQTT system test complete")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    sys.exit(0 if test_mqtt(args.broker, args.port) else 1)
