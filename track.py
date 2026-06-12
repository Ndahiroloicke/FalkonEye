#!/usr/bin/env python3
"""Shortcut: face recognition + MQTT camera tracking (BENAX SRS demo)."""

import argparse
import sys

from src.recognize_with_tracking import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face lock + MQTT servo tracking (SRS demo)")
    parser.add_argument("--fullscreen", "-f", action="store_true")
    parser.add_argument("--no-mqtt", action="store_true")
    parser.add_argument("--broker", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--speaker", type=str, default=None, help="Enrolled speaker to lock")
    args = parser.parse_args()

    success = main(
        start_fullscreen=args.fullscreen,
        enable_mqtt=not args.no_mqtt,
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        speaker_lock=args.speaker,
    )
    sys.exit(0 if success else 1)
