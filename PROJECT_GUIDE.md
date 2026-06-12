# FaceLocking — Project Guide

CPU-first face recognition with optional ESP8266 MQTT pan tracking.

## Architecture

```
Webcam → Haar + MediaPipe landmarks → Align 112×112 → ArcFace embed → Match DB
       → [optional] Lock person → Activity log (blink/smile/movement)
       → [optional] PanTracker → MQTT → ESP8266 servo
```

## Entry points

| Command | Purpose |
|---------|---------|
| `python -m src.enroll` | Enroll faces into `data/db/` |
| `python -m src.recognize` | Live recognition + activity logging |
| `python track.py` | Recognition + MQTT servo tracking |
| `python -m src.view_activity_logs` | Review session CSV/JSON logs |

## MQTT stack

- **Python:** `mqtt_camera_controller.py`, `tracking.py`, `recognize_with_tracking.py`
- **Firmware:** `arduino/esp8266_camera_tracker/esp8266_camera_tracker.ino`
- **Topics:** `camera/track/horizontal`, `camera/track/command`, `camera/status`
- **Config:** `src/config.py` (broker IP, servo limits, search sweep)

## Shared recognition core

`recognition_core.py` holds database load, matching, camera open, and drawing — used by both `recognize.py` and `recognize_with_tracking.py`.

## Data layout

- `data/db/face_db.npz` — embeddings
- `data/enroll/<name>/` — enrollment crops
- `data/history/<name>_*` — activity logs per locked session
- `models/` — ArcFace ONNX + MediaPipe face landmarker

See [TEST_GUIDE.md](TEST_GUIDE.md) for validation procedures.
