#!/usr/bin/env python3
"""Standalone dashboard — MQTT servo stats without running track.py."""

import json
import signal
import sys
import time

import paho.mqtt.client as mqtt

from src import config
from src.dashboard_server import start_dashboard
from src.dashboard_state import reset_session, set_mode, stop, update


def main() -> int:
    reset_session()
    set_mode("mqtt-only")
    url = start_dashboard()
    print(f"Dashboard: {url}")
    print(f"Listening MQTT status on {config.MQTT_TOPIC_STATUS}")
    print("Press Ctrl+C to stop.")

    def on_message(_client, _userdata, msg):
        try:
            payload = msg.payload.decode()
            data = json.loads(payload) if payload.startswith("{") else {}
            angle = int(data.get("angle", 90))
            target = int(data.get("target", angle))
            moving = bool(data.get("moving", False))
            update(
                servo_angle=angle,
                servo_target=target,
                servo_moving=moving,
                mqtt_connected=True,
                mqtt_broker=f"{config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}",
                system_state="MQTT MONITOR",
            )
        except Exception:
            pass

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="FaceLocking_Dashboard")
    if config.MQTT_USERNAME and config.MQTT_PASSWORD:
        client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
    client.on_message = on_message

    def on_connect(c, _u, _f, reason_code, _props=None):
        if int(reason_code) == 0:
            c.subscribe(config.MQTT_TOPIC_STATUS, qos=config.MQTT_QOS)
            update(
                mqtt_connected=True,
                mqtt_broker=f"{config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}",
            )
            print("OK: MQTT connected")
        else:
            print(f"MQTT connect failed: {reason_code}")

    client.on_connect = on_connect
    client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, keepalive=config.MQTT_KEEPALIVE)
    client.loop_start()

    def _shutdown(*_args):
        stop()
        client.loop_stop()
        client.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
