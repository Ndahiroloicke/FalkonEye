"""
MQTT camera controller — publishes pan commands to ESP8266 servo firmware.
"""

import json
import time
from typing import Optional

import paho.mqtt.client as mqtt

from . import config


class MQTTCameraController:
    """Control camera servo via MQTT."""

    def __init__(
        self,
        broker_host: str = None,
        broker_port: int = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.broker_host = broker_host or config.MQTT_BROKER_HOST
        self.broker_port = broker_port or config.MQTT_BROKER_PORT
        self.username = username if username is not None else config.MQTT_USERNAME
        self.password = password if password is not None else config.MQTT_PASSWORD

        self.topic_horizontal = config.MQTT_TOPIC_HORIZONTAL
        self.topic_command = config.MQTT_TOPIC_COMMAND
        self.topic_status = config.MQTT_TOPIC_STATUS
        self.topic_events = config.MQTT_TOPIC_EVENTS

        self.current_angle = config.SERVO_CENTER_ANGLE
        self.is_connected = False
        self.last_status: dict = {}
        self._last_publish_ms = 0.0

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="FaceLocking_Controller",
        )
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=config.MQTT_KEEPALIVE)
            self.client.loop_start()
            print(f"✓ MQTT connecting to {self.broker_host}:{self.broker_port}")
        except Exception as exc:
            print(f"✗ MQTT connection failed: {exc}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.is_connected = True
            client.subscribe(self.topic_status, qos=config.MQTT_QOS)
            print(f"✓ MQTT connected, subscribed to {self.topic_status}")
        else:
            print(f"✗ MQTT connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        self.is_connected = False
        if rc != 0:
            print(f"⚠ MQTT disconnected rc={rc}")

    def _on_message(self, client, userdata, msg):
        if msg.topic != self.topic_status:
            return
        try:
            payload = msg.payload.decode()
            if payload.startswith("{"):
                self.last_status = json.loads(payload)
            else:
                self.last_status = {"raw": payload}
            angle = self.last_status.get("angle")
            if angle is not None:
                self.current_angle = int(angle)
        except Exception:
            pass

    def _rate_limited(self) -> bool:
        now = time.time() * 1000.0
        if now - self._last_publish_ms < config.MQTT_MIN_COMMAND_INTERVAL_MS:
            return True
        self._last_publish_ms = now
        return False

    def _publish(self, topic: str, payload: str) -> bool:
        if not self.is_connected:
            return False
        if self._rate_limited():
            return False
        result = self.client.publish(topic, payload, qos=config.MQTT_QOS)
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def move_to_angle(self, angle: int) -> bool:
        angle = int(max(config.SERVO_MIN_ANGLE, min(config.SERVO_MAX_ANGLE, angle)))
        if abs(angle - self.current_angle) < 1:
            return False
        ok = self._publish(self.topic_horizontal, str(angle))
        if ok:
            self.current_angle = angle
        return ok

    def send_command(self, command: str) -> bool:
        return self._publish(self.topic_command, command)

    def move_left(self, step: int = None) -> bool:
        step = step or config.SERVO_STEP_SIZE
        return self.move_to_angle(self.current_angle - step)

    def move_right(self, step: int = None) -> bool:
        step = step or config.SERVO_STEP_SIZE
        return self.move_to_angle(self.current_angle + step)

    def center(self) -> bool:
        return self.move_to_angle(config.SERVO_CENTER_ANGLE)

    def publish_tracking_event(
        self,
        event: str,
        speaker_id: str = "",
        confidence: float = 0.0,
        servo_angle: int = 0,
        motor_command: str = "",
        details: str = "",
    ) -> bool:
        """Publish SRS-style tracking event JSON for assessor evidence."""
        payload = json.dumps(
            {
                "event": event,
                "speaker": speaker_id,
                "confidence": round(confidence, 4),
                "servo_angle": int(servo_angle),
                "motor_command": motor_command,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "details": details,
            }
        )
        return self._publish(self.topic_events, payload)

    def wait_for_connection(self, timeout_sec: float = 5.0) -> bool:
        """Block until connected or timeout (connect is async via loop_start)."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.is_connected:
                return True
            time.sleep(0.1)
        return self.is_connected

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
