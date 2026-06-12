#!/usr/bin/env python3
"""Monitor MQTT camera tracking topics."""

import json
import sys

import paho.mqtt.client as mqtt

from src import config


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✓ Connected to broker")
        client.subscribe(config.MQTT_TOPIC_HORIZONTAL)
        client.subscribe(config.MQTT_TOPIC_COMMAND)
        client.subscribe(config.MQTT_TOPIC_STATUS)
        print(f"  Subscribed: {config.MQTT_TOPIC_HORIZONTAL}")
        print(f"  Subscribed: {config.MQTT_TOPIC_COMMAND}")
        print(f"  Subscribed: {config.MQTT_TOPIC_STATUS}")
    else:
        print(f"✗ Connect failed rc={rc}")


def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    print(f"[{msg.topic}] {payload}")
    if msg.topic == config.MQTT_TOPIC_STATUS and payload.startswith("{"):
        try:
            status = json.loads(payload)
            print(f"  angle={status.get('angle')} moving={status.get('moving')}")
        except json.JSONDecodeError:
            pass


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="FaceLocking_Debug")
    client.on_connect = on_connect
    client.on_message = on_message
    print(f"Connecting to {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}...")
    client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, 60)
    print("Listening (Ctrl+C to stop)...")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
