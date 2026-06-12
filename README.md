# FaceLocking

Face recognition with optional MQTT pan tracking for an ESP8266 servo-mounted camera.

## Quick start

```bash
bash setup.sh
source .venv/bin/activate
python download_model.py
python -m src.enroll
python -m src.recognize          # recognition + activity logging
python track.py                  # recognition + MQTT servo tracking
```

## MQTT tracking stack

| Component | Role |
|-----------|------|
| `src/recognize_with_tracking.py` | Main tracking loop |
| `src/tracking.py` | Pan smoothing, dead zone, search sweep |
| `src/mqtt_camera_controller.py` | MQTT publish to ESP8266 |
| `arduino/esp8266_camera_tracker/` | Servo firmware |

**Test MQTT without camera:**

```bash
python test_mqtt_system.py
python debug_mqtt_tracking.py
```

**Tracking controls:** `q` quit · `l` unlock · `s` search · `c` center · `+/-` threshold

Configure broker and camera in `src/config.py`. Flash the Arduino sketch after updating WiFi credentials.

## Docs

- [TEST_GUIDE.md](TEST_GUIDE.md) — step-by-step testing
- [arduino/README.md](arduino/README.md) — wiring and firmware
