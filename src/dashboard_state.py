"""Thread-safe live stats for the tracking dashboard."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "mode": "idle",
    "running": False,
    "updated_at": 0.0,
    "session_started": 0.0,
    "frame_number": 0,
    "system_state": "IDLE",
    "locked_speaker": None,
    "face_count": 0,
    "confidence": 0.0,
    "face_center_x": None,
    "face_center_y": None,
    "frame_width": 0,
    "frame_height": 0,
    "face_error": None,
    "track_fps": 0.0,
    "recog_fps": 0.0,
    "servo_angle": 90,
    "servo_target": 90,
    "servo_moving": False,
    "commanded_angle": None,
    "motor_command": "STOP",
    "mqtt_connected": False,
    "mqtt_broker": "",
    "lost_for_sec": 0.0,
    "threshold": 0.45,
    "angle_history": [],
}

_HISTORY_MAX = 120


def reset_session() -> None:
    with _lock:
        now = time.time()
        _state["session_started"] = now
        _state["frame_number"] = 0
        _state["angle_history"] = []
        _state["running"] = True
        _state["mode"] = "tracking"
        _state["updated_at"] = now


def set_mode(mode: str) -> None:
    with _lock:
        _state["mode"] = mode
        _state["running"] = True


def update(**fields: Any) -> None:
    with _lock:
        for key, value in fields.items():
            if key == "angle_history":
                continue
            _state[key] = value

        angle = fields.get("servo_angle", _state.get("servo_angle"))
        if angle is not None:
            history: List[float] = list(_state.get("angle_history") or [])
            history.append(float(angle))
            if len(history) > _HISTORY_MAX:
                history = history[-_HISTORY_MAX:]
            _state["angle_history"] = history

        _state["updated_at"] = time.time()


def snapshot() -> Dict[str, Any]:
    with _lock:
        data = deepcopy(_state)
    if data.get("session_started"):
        data["uptime_sec"] = round(time.time() - data["session_started"], 1)
    else:
        data["uptime_sec"] = 0.0
    data["stale"] = (time.time() - data.get("updated_at", 0)) > 3.0
    return data


def stop() -> None:
    with _lock:
        _state["running"] = False
        _state["mode"] = "stopped"
        _state["updated_at"] = time.time()
