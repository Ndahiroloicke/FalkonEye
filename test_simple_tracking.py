#!/usr/bin/env python3
"""Send basic MQTT servo commands (no camera). Use to verify ESP8266 wiring."""

import sys
import time

from src import config
from src.mqtt_camera_controller import MQTTCameraController


def main() -> bool:
    print("Simple MQTT servo test — left / right / center")
    print(f"Broker: {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}")

    ctrl = MQTTCameraController()
    time.sleep(2)
    if not ctrl.is_connected:
        print("✗ Cannot connect to MQTT broker")
        return False

    for label, action in [
        ("Center", ctrl.center),
        ("Left", ctrl.move_left),
        ("Right", ctrl.move_right),
        ("Center", ctrl.center),
    ]:
        print(f"→ {label}")
        action()
        time.sleep(2)

    ctrl.close()
    print("✓ Done")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
