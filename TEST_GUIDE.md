# FaceLocking — Test Guide

Comprehensive testing procedures for the FaceLocking face recognition and MQTT camera tracking system.

**Prerequisites:** Python 3.9+, virtual environment activated, ArcFace model downloaded, at least one person enrolled.

---

## Environment Validation

### EV-01: Build / Dependencies Installed

**Preconditions:** Fresh clone or after `setup.sh`.

**Steps:**
1. `source .venv/bin/activate` (or Windows equivalent)
2. `pip list | grep -E "opencv|onnxruntime|mediapipe|paho-mqtt"`
3. `python verify.py`

**Expected Result:**
- All required packages installed at versions matching `requirements.txt`
- `verify.py` reports all core files present (note: does not check MQTT modules — see EV-01b)

### EV-01b: Extended Integrity Check

**Steps:**
```bash
python -c "
from pathlib import Path
root = Path('.')
for f in ['src/recognize_with_tracking.py','src/mqtt_camera_controller.py','src/activity_logger.py','track.py']:
    assert (root/f).exists(), f'Missing: {f}'
print('Extended check OK')
"
```

**Expected Result:** All files exist, no assertion error.

### EV-02: Directory Structure

**Steps:**
```bash
python -c "from src import config; config.ensure_dirs(); print('dirs OK')"
ls data/db data/enroll data/history models 2>/dev/null
```

**Expected Result:** `data/db/`, `data/enroll/`, `data/history/`, `models/` exist.

### EV-03: Model File Present

**Steps:**
```bash
ls -la models/embedder_arcface.onnx
python -c "from src.config import ARCFACE_MODEL_PATH; assert ARCFACE_MODEL_PATH.exists()"
```

**Expected Result:** ONNX model file exists (~170 MB). If missing, run `python download_model.py`.

### EV-04: Database Connectivity

**Steps:**
```bash
python -c "
import numpy as np
from src.config import DB_NPZ_PATH, DB_JSON_PATH
assert DB_NPZ_PATH.exists(), 'Run: python -m src.enroll'
data = np.load(str(DB_NPZ_PATH), allow_pickle=True)
print('Identities:', list(data.files))
print('JSON:', DB_JSON_PATH.read_text()[:200])
"
```

**Expected Result:** At least one identity in `.npz`; JSON metadata readable.

### EV-05: Configuration Validation

**Steps:**
```bash
python -c "
from src import config
assert config.EMBEDDING_DIM == 512
assert config.ALIGNMENT_OUTPUT_SIZE == (112, 112)
assert config.MQTT_BROKER_PORT == 1883
print('Config OK')
"
```

**Expected Result:** Constants match expected values.

### EV-06: Camera Available

**Steps:**
```bash
python -m src.camera
# Press q to exit after confirming live video
```

**Expected Result:** Live video window with FPS counter. No "Cannot open camera" error.

---

## Functional Testing

### FT-01: Haar Face Detection

| Field | Value |
|-------|-------|
| **Preconditions** | Camera working (EV-06) |
| **Steps** | `python -m src.detect` — stand in front of camera |
| **Expected** | Green bounding box around face(s); press q to quit |

### FT-02: 5-Point Landmarks

| Field | Value |
|-------|-------|
| **Preconditions** | FT-01 passes |
| **Steps** | `python -m src.landmarks` |
| **Expected** | 5 green dots: left eye, right eye, nose, left mouth, right mouth |

### FT-03: Face Alignment

| Field | Value |
|-------|-------|
| **Preconditions** | FT-02 passes |
| **Steps** | `python -m src.align` — press s to save one crop |
| **Expected** | Two windows: original frame + 112×112 aligned face; saved crop in `data/debug_aligned/` |

### FT-04: Embedding Extraction

| Field | Value |
|-------|-------|
| **Preconditions** | FT-03 passes |
| **Steps** | `python -m src.embed` — press p for stats |
| **Expected** | Embedding shape `(512,)`, L2 norm ≈ 1.0 |

### FT-05: Face Enrollment

| Field | Value |
|-------|-------|
| **Preconditions** | FT-04 passes |
| **Steps** | `python -m src.enroll` → enter test name → capture 5+ samples → press S |
| **Expected** | `data/db/face_db.npz` updated; `data/enroll/<name>/` contains JPG crops; console confirms save |

### FT-06: Threshold Evaluation

| Field | Value |
|-------|-------|
| **Preconditions** | 2+ people enrolled |
| **Steps** | `python -m src.evaluate` |
| **Expected** | Genuine distances < impostor distances; recommended threshold printed |

### FT-07: Live Recognition — Known Person

| Field | Value |
|-------|-------|
| **Preconditions** | Person enrolled, threshold tuned |
| **Steps** | `python -m src.recognize` → Enter for no lock → show face to camera |
| **Expected** | Enrolled name displayed; confidence/distance shown |

### FT-08: Live Recognition — Unknown Person

| Field | Value |
|-------|-------|
| **Preconditions** | FT-07 environment |
| **Steps** | Show face of person NOT in database |
| **Expected** | Red box, "Unknown" label |

### FT-09: Lock Mode

| Field | Value |
|-------|-------|
| **Preconditions** | 2+ people enrolled |
| **Steps** | `python -m src.recognize` → enter name/number to lock |
| **Expected** | Locked person: green box + landmarks + "(locked)"; others: name only |

### FT-10: Activity Logging — Blink

| Field | Value |
|-------|-------|
| **Preconditions** | FT-09 with lock active |
| **Steps** | Blink several times → press q to quit |
| **Expected** | "Blink!" on-screen; CSV in `data/history/<name>_*_activities.csv` contains `blink` rows |

### FT-11: Activity Logging — Smile

| Field | Value |
|-------|-------|
| **Preconditions** | Lock active |
| **Steps** | Smile → quit |
| **Expected** | `smile` entries in CSV; JSON summary has `smile` count > 0 |

### FT-12: Activity Logging — Movement

| Field | Value |
|-------|-------|
| **Preconditions** | Lock active |
| **Steps** | Move head left/right/up/down significantly |
| **Expected** | `move_left`, `move_right`, `move_up`, or `move_down` in CSV |

### FT-13: Activity Log Viewer

| Field | Value |
|-------|-------|
| **Preconditions** | At least one session log in `data/history/` |
| **Steps** | `python src/view_activity_logs.py` |
| **Expected** | Lists sessions; can view details and pattern analysis |

### FT-14: Fullscreen Mode

| Field | Value |
|-------|-------|
| **Preconditions** | Recognition running |
| **Steps** | `python -m src.recognize --fullscreen` or press f during session |
| **Expected** | Window fills screen; f toggles back |

### FT-15: Threshold Hot-Adjust

| Field | Value |
|-------|-------|
| **Preconditions** | Recognition running |
| **Steps** | Press + three times, then - three times |
| **Expected** | On-screen threshold value changes; acceptance behavior shifts |

### FT-16: Database Reload

| Field | Value |
|-------|-------|
| **Preconditions** | Recognition running |
| **Steps** | Enroll new person in another terminal → press R in recognition window |
| **Expected** | New identity recognized without restart |

### FT-17: MQTT Tracking — Basic

| Field | Value |
|-------|-------|
| **Preconditions** | MQTT broker running, ESP8266 connected, person enrolled |
| **Steps** | `python track.py` → lock to person → move left/right |
| **Expected** | Servo follows movement; MQTT status messages on `camera/status` |

### FT-18: MQTT Tracking — Search Mode

| Field | Value |
|-------|-------|
| **Preconditions** | FT-17 running with lock |
| **Steps** | Leave camera frame for ~2+ seconds |
| **Expected** | Search mode activates; servo sweeps 0°→180°→0° pattern; press s to toggle |

### FT-19: MQTT Tracking — No MQTT Fallback

| Field | Value |
|-------|-------|
| **Preconditions** | Person enrolled |
| **Steps** | `python track.py --no-mqtt` |
| **Expected** | Recognition works; no MQTT connection attempted; no servo movement |

### FT-20: track.py Shortcut

| Field | Value |
|-------|-------|
| **Preconditions** | Same as FT-17 |
| **Steps** | `python track.py -f` |
| **Expected** | Same as `recognize_with_tracking` with fullscreen |

---

## Integration Testing

### IT-01: Enrollment → Recognition Pipeline

**Steps:**
1. Delete `data/db/face_db.npz` and `face_db.json`
2. Enroll person A (15 samples)
3. `python -m src.recognize` — verify A recognized
4. Enroll person B
5. Press R in recognition — verify both A and B recognized

**Expected:** End-to-end data flow from enrollment crops → embeddings → live match.

### IT-02: Enrollment → Evaluate → Recognize

**Steps:**
1. Enroll 2 people with 15+ samples each
2. `python -m src.evaluate` — note recommended threshold
3. Set `DEFAULT_DISTANCE_THRESHOLD` in config.py
4. `python -m src.recognize` — verify low false accepts/rejects

**Expected:** Threshold from evaluate.py improves recognition accuracy.

### IT-03: Recognition → Activity Logger → Viewer

**Steps:**
1. Lock to person, perform blinks/smiles/movements, quit
2. `python src/view_activity_logs.py` — open latest session

**Expected:** CSV and JSON match on-screen activity counts.

### IT-04: Recognition → MQTT → ESP8266

**Steps:**
1. `python debug_mqtt_tracking.py` (terminal 1)
2. `python track.py` (terminal 2) — lock and move
3. Observe MQTT messages in terminal 1

**Expected:** `camera/track/horizontal` or `camera/track/command` published on movement; `camera/status` heartbeat every 1 s.

### IT-05: HaarMediaPipeFaceDetector → Aligner → Embedder Chain

**Steps:**
```bash
python -c "
import cv2, numpy as np
from src.haar_5pt import HaarMediaPipeFaceDetector
from src.align import FaceAligner
from src.embed import ArcFaceEmbedder

cap = cv2.VideoCapture(2)  # adjust index
det = HaarMediaPipeFaceDetector()
aligner = FaceAligner()
embedder = ArcFaceEmbedder()

for _ in range(30):
    ret, frame = cap.read()
    if not ret: break
    faces = det.detect(frame)
    if faces:
        aligned = aligner.align(frame, faces[0].landmarks_5pt)
        emb = embedder.embed(aligned)
        assert emb.shape == (512,)
        print('Chain OK, norm:', np.linalg.norm(emb))
        break
cap.release()
"
```

**Expected:** Single pipeline chain produces 512-dim embedding without error.

### IT-06: Two-Person Discrimination

**Steps:**
```bash
python test_two_person_recognition.py
```

**Expected:** Script reports sufficient distance between two enrolled identities.

---

## Regression Testing

Critical flows that must never break:

| ID | Flow | Command | Pass Criteria |
|----|------|---------|---------------|
| RG-01 | Camera opens | `python -m src.camera` | Video displays |
| RG-02 | Model loads | `python -m src.embed` | No ONNX error |
| RG-03 | DB loads | `python -m src.recognize` (quit immediately) | No DB error if enrolled |
| RG-04 | Enrollment saves | `python -m src.enroll` | NPZ + JSON updated |
| RG-05 | Multi-face detect | `python -m src.detect` with 2 people | 2 boxes |
| RG-06 | Lock + log | `python -m src.recognize` with lock | CSV created on quit |
| RG-07 | MQTT connect | `python test_mqtt_system.py` | Broker connection OK |
| RG-08 | track.py entry | `python track.py --no-mqtt` | Starts without crash |

---

## Edge Cases

### EC-01: Empty Database

**Steps:** Delete `face_db.npz`, run `python -m src.recognize`  
**Expected:** Clear error: "Database not found. Run enrollment first."

### EC-02: Enrollment — Minimum Samples

**Steps:** Enroll with only 1–2 samples, press S  
**Expected:** Warning or rejection if below `MIN_SAMPLES_TO_SAVE` (3).

### EC-03: Invalid Lock Name

**Steps:** `python -m src.recognize` → enter "nonexistent"  
**Expected:** Message "Unknown name"; proceeds with all identities.

### EC-04: Camera Index Wrong

**Steps:** Set `CAMERA_INDEX = 99` in config.py, run camera test  
**Expected:** Graceful error, not hang.

### EC-05: MQTT Broker Unreachable

**Steps:** Set invalid `MQTT_BROKER_HOST`, run `python track.py`  
**Expected:** Connection failure message; recognition may still run (check behavior).

### EC-06: No Face in Frame

**Steps:** Run recognition, cover camera  
**Expected:** No crash; no false recognition.

### EC-07: Extreme Lighting

**Steps:** Recognition in very dark / backlit conditions  
**Expected:** Degraded detection; no crash; may show Unknown.

### EC-08: Duplicate Enrollment Name

**Steps:** Enroll same name twice  
**Expected:** Existing samples loaded; new samples appended; mean embedding updated.

### EC-09: Corrupt NPZ

**Steps:** Truncate `face_db.npz` to 0 bytes, run recognize  
**Expected:** Error handled gracefully (not segfault).

### EC-10: EOF on Lock Prompt

**Steps:** `echo "" | python -m src.recognize`  
**Expected:** No lock selected; proceeds without hang.

---

## Security Testing

### SEC-01: Offline Operation

**Steps:** Disconnect network, run `python -m src.recognize`  
**Expected:** Full recognition works (no network calls except optional MQTT).

### SEC-02: Sensitive Data in Logs

**Steps:** Inspect `data/history/*.csv` and `face_db.json`  
**Expected:** Activity logs contain timestamps and coordinates, not raw face images. Embeddings in NPZ are mathematical vectors, not reversible photos.

### SEC-03: MQTT Without Authentication

**Steps:** Connect to broker without credentials  
**Expected:** Works on default config (documents risk — MQTT is unauthenticated).

### SEC-04: Hardcoded Credentials Exposure

**Steps:** `grep -r "password\|RCA" arduino/ src/config.py`  
**Expected:** Document finding; WiFi password in Arduino sketch — rotate before deployment.

### SEC-05: Enrollment Image Storage

**Steps:** Check `data/enroll/`  
**Expected:** Raw face crops stored if `SAVE_ENROLLMENT_CROPS = True` — PII risk documented.

### SEC-06: Input Validation — Enrollment Name

**Steps:** Enroll with name `../../etc/passwd` or empty string  
**Expected:** Empty rejected; path traversal does not write outside `data/enroll/`.

---

## Performance Testing

### PERF-01: Pipeline FPS

**Steps:** Run `python -m src.recognize`, observe on-screen FPS  
**Expected:** 8–10 FPS on modern CPU.

### PERF-02: Embedding Latency

**Steps:** `python -m src.embed`, note per-frame embedding time in console  
**Expected:** 8–12 FPS for embedding stage alone.

### PERF-03: Memory Usage

**Steps:** `python -m src.recognize` — monitor with `top` or `htop`  
**Expected:** ~300 MB RSS, stable over 5 minutes.

### PERF-04: Database Scale

**Steps:** With 6 enrolled identities, measure match loop time  
**Expected:** Near-instant matching (<1 ms per face) — 6 identities is trivial.

### PERF-05: MQTT Latency

**Steps:** `python test_mqtt_system.py` — note round-trip timing  
**Expected:** 10–50 ms publish latency on LAN.

### PERF-06: Search Mode Cycle Time

**Steps:** Trigger search, time full 12-position sweep  
**Expected:** ~24 seconds per cycle (12 × 2 s interval).

---

## Manual Test Checklist

### Environment
- [ ] Virtual environment activates
- [ ] All pip dependencies installed
- [ ] `models/embedder_arcface.onnx` exists
- [ ] `python verify.py` passes
- [ ] Camera opens (`python -m src.camera`)

### Pipeline Stages
- [ ] FT-01: Haar detection draws boxes
- [ ] FT-02: 5 landmarks visible
- [ ] FT-03: Alignment produces 112×112 crop
- [ ] FT-04: Embedding is 512-dim, norm ≈ 1.0

### Enrollment & Matching
- [ ] FT-05: New person enrolls successfully
- [ ] FT-06: Evaluate produces threshold recommendation
- [ ] FT-07: Known person recognized
- [ ] FT-08: Unknown person rejected
- [ ] FT-09: Lock mode works with multiple people

### Activity Logging
- [ ] FT-10: Blinks logged
- [ ] FT-11: Smiles logged
- [ ] FT-12: Movements logged
- [ ] FT-13: Log viewer displays sessions

### Recognition Controls
- [ ] FT-14: Fullscreen toggle works
- [ ] FT-15: +/- threshold adjustment works
- [ ] FT-16: R reloads database

### MQTT Tracking (if hardware available)
- [ ] MQTT broker running
- [ ] ESP8266 connected and publishing status
- [ ] FT-17: Servo follows locked person
- [ ] FT-18: Search mode sweeps on person lost
- [ ] FT-19: `--no-mqtt` works without broker

### Integration
- [ ] IT-01: Enroll → recognize pipeline
- [ ] IT-04: MQTT messages visible in debug tool
- [ ] IT-06: Two-person test passes

### Regression
- [ ] RG-01 through RG-08 all pass

---

## Release Verification Checklist

Before deploying to a lab or demo environment:

### Code & Config
- [ ] `requirements.txt` pinned versions installed
- [ ] `DEFAULT_DISTANCE_THRESHOLD` tuned via `evaluate.py`
- [ ] `CAMERA_INDEX` set for target machine
- [ ] `MQTT_BROKER_HOST` set to correct broker (not dev IP)
- [ ] Arduino WiFi/MQTT credentials updated for target network
- [ ] `SERVO_CENTER_ANGLE` aligned between Python and Arduino

### Data
- [ ] All required persons enrolled with 15+ samples
- [ ] `face_db.npz` backed up
- [ ] Enrollment images reviewed for quality
- [ ] No test/history logs committed to version control

### Functional
- [ ] Full validation pipeline passes (FT-01 through FT-04)
- [ ] Recognition accuracy verified with all enrolled users
- [ ] Lock + activity logging verified
- [ ] MQTT tracking verified (if used)
- [ ] Search mode verified

### Security
- [ ] Hardcoded credentials rotated for deployment network
- [ ] MQTT authentication enabled if on shared network
- [ ] `data/enroll/` access restricted on shared machines

### Documentation
- [ ] [PROJECT_GUIDE.md](PROJECT_GUIDE.md) reviewed by operator
- [ ] Operators know keyboard controls (q, r, l, f, +/-, s, c)

### Performance
- [ ] 8+ FPS achieved on target hardware
- [ ] No memory leaks over 30-minute session

---

## Automated Test Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `verify.py` | Project file integrity | `python verify.py` |
| `test_two_person_recognition.py` | Embedding discrimination | `python test_two_person_recognition.py` |
| `test_mqtt_system.py` | Full MQTT integration | `python test_mqtt_system.py` |
| `test_simple_tracking.py` | Servo commands only | `python test_simple_tracking.py` |
| `debug_mqtt_tracking.py` | MQTT message monitor | `python debug_mqtt_tracking.py` |
| `examples/activity_logging_example.py` | ActivityLogger demo | `python examples/activity_logging_example.py` |

---

## Known Test Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| No pytest/unittest suite | Regressions undetected | Manual checklist + regression table above |
| `verify.py` outdated | Missing MQTT module checks | Use EV-01b extended check |
| No CI pipeline | Broken builds undetected | Run RG-01–RG-08 before releases |
| Hardware tests require physical ESP8266 | MQTT untested in CI | Use `test_simple_tracking.py` on lab machine |
| No load test for 100+ identities | Scale unknown | Acceptable for current scope (<10 users) |
