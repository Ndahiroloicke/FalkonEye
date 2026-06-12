#!/usr/bin/env python3
"""
Standalone dashboard — shows MQTT servo stats without running track.py.

Usage:
  python -m src.dashboard
  python -m src.dashboard --port 8765
"""

import argparse
import json
import time

import paho.mqtt.client as mqtt

from . import config
from .dashboard_server import start_dashboard, stop_dashboard
from .dashboard_state import update


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceLocking live stats dashboard")
    parser.add_argument("--host", default=config.DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    parser.add_argument("--broker", default=config.MQTT_BROKER_HOST)
    parser.add_argument("--mqtt-port", type=int, default=config.MQTT_BROKER_PORT)
    args = parser.parse_args()

    update(
        mode="mqtt_only",
        session_started_at=time.time(),
        mqtt_broker=f"{args.broker}:{args.mqtt_port}",
        system_state="STANDBY",
    )

    def on_message(_client, _userdata, msg):
        if msg.topic != config.MQTT_TOPIC_STATUS:
            return
        try:
            payload = msg.payload.decode()
            data = json.loads(payload) if payload.startswith("{") else {}
            update(
                servo_angle=int(data.get("angle", 90)),
                servo_target=int(data.get("target", 90)),
                servo_moving=bool(data.get("moving", False)),
                mqtt_connected=True,
                mqtt_broker=f"{args.broker}:{args.mqtt_port}",
            )
        except Exception:
            pass

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="FaceLocking_Dashboard")
    client.on_connect = lambda c, _u, _f, rc, _p=None: (
        c.subscribe(config.MQTT_TOPIC_STATUS, qos=config.MQTT_QOS) if int(rc) == 0 else None
    )
    client.on_message = on_message

    url = start_dashboard(host=args.host, port=args.port)
    print(f"Dashboard: {url}")
    print(f"Listening to MQTT {args.broker}:{args.mqtt_port} topic {config.MQTT_TOPIC_STATUS}")
    print("Press Ctrl+C to stop.")

    try:
        client.connect(args.broker, args.mqtt_port, keepalive=config.MQTT_KEEPALIVE)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        client.disconnect()
        stop_dashboard()


if __name__ == "__main__":
    main()
