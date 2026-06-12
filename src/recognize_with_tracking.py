"""
Live face recognition with MQTT pan tracking and autonomous lost-target search.

State machine
-------------
IDLE      : no identity selected to lock (recognition still runs).
TRACKING  : locked target visible, servo actively centering it.
LOCKED    : locked target visible and centered/stable.
SEARCHING : locked target missing beyond LOST_TARGET_TIMEOUT — autonomous,
            direction-aware servo sweep while recognition keeps hunting for the
            ORIGINAL locked identity. Never locks anyone else.

Performance & multi-face strategy
---------------------------------
Detection runs every frame (smooth boxes). ArcFace embedding is the expensive
step, so the FaceTracker caches recognition per persistent track ID and only
re-embeds on an interval (more often for the locked track). The lock is bound
to a track ID, which prevents identity/lock switching when other known faces
appear.
"""

import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from . import actions as action_module
from .activity_logger import ActivityLogger
from .align import FaceAligner
from .embed import ArcFaceEmbedder
from .haar_5pt import HaarMediaPipeFaceDetector
from .face_tracker import FaceTracker
from .mqtt_camera_controller import MQTTCameraController
from .recognition_core import (
    choose_lock_identity,
    draw_tracks,
    load_database,
    open_camera,
    recognize_face,
)
from .tracking import PanTracker
from .tracking_log import TrackingLogger
from .camera_utils import CameraStream
from .evidence_logger import EvidenceLogger
from .srs_commands import derive_command, motor_command_label
from . import dashboard_state
from .dashboard_server import start_dashboard, stop_dashboard


def _draw_debug_overlay(
    vis: np.ndarray,
    state: str,
    lock_name: Optional[str],
    servo_angle,
    face_count: int,
    recog_fps: float,
    track_fps: float,
    mqtt_ok: bool,
    threshold: float,
    lost_for: float,
) -> None:
    """On-screen diagnostics panel (Issue #8)."""
    lines = [
        f"State: {state}",
        f"Locked: {lock_name or '(none)'}",
        f"Servo: {servo_angle}",
        f"Faces: {face_count}",
        f"Recog FPS: {recog_fps:.1f}",
        f"Track FPS: {track_fps:.1f}",
        f"MQTT: {'OK' if mqtt_ok else '--'}",
        f"Thresh: {threshold:.2f}",
    ]
    if state == "SEARCHING":
        lines.append(f"Lost for: {lost_for:.1f}s")

    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 6
    line_h = 20
    panel_w = 210
    panel_h = line_h * len(lines) + pad
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

    state_color = {
        "LOCKED": config.COLOR_LOCKED,
        "TRACKING": (0, 255, 255),
        "SEARCHING": config.COLOR_LOST,
        "IDLE": (200, 200, 200),
    }.get(state, config.COLOR_HUD)

    y = pad + 14
    for i, t in enumerate(lines):
        color = state_color if i == 0 else config.COLOR_HUD
        cv2.putText(vis, t, (pad, y), font, 0.5, color, 1, cv2.LINE_AA)
        y += line_h


def _draw_search_banner(vis: np.ndarray, lock_name: str) -> None:
    text = f"SEARCHING FOR TARGET: {lock_name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    x = (vis.shape[1] - tw) // 2
    y = 40
    cv2.rectangle(vis, (x - 10, y - th - 8), (x + tw + 10, y + 8), config.COLOR_LOST, -1)
    cv2.putText(vis, text, (x, y), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def main(
    start_fullscreen: bool = False,
    enable_mqtt: bool = True,
    mqtt_broker: str = None,
    mqtt_port: int = None,
    speaker_lock: str = None,
    enable_dashboard: bool = True,
) -> bool:
    db = load_database()
    if not db:
        print("ERROR: No enrolled identities. Run: python -m src.enroll")
        return False

    print(f"✓ Loaded {len(db)} enrolled identities")

    detector = HaarMediaPipeFaceDetector(min_size=config.HAAR_MIN_SIZE)
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)

    names = sorted(db.keys())
    embeddings_matrix = np.stack([db[n].reshape(-1) for n in names], axis=0).astype(np.float32)

    lock_name: Optional[str] = speaker_lock or config.DEFAULT_SPEAKER_LOCK
    if lock_name and lock_name not in names:
        print(f"WARNING: Speaker '{lock_name}' not enrolled. Choose from: {names}")
        lock_name = None
    if not lock_name:
        lock_name = choose_lock_identity(names)
    if not lock_name:
        print("WARNING: No lock selected. Running recognition only (IDLE).")

    mqtt: Optional[MQTTCameraController] = None
    if enable_mqtt:
        mqtt = MQTTCameraController(broker_host=mqtt_broker, broker_port=mqtt_port)
        if not mqtt.wait_for_connection(timeout_sec=5.0):
            print("✗ MQTT NOT CONNECTED — servo will NOT move.")
            print(f"  Broker: {mqtt_broker or config.MQTT_BROKER_HOST}:{mqtt_port or config.MQTT_BROKER_PORT}")
            print("  Fix: run python test_mqtt_system.py  OR  python test_simple_tracking.py")
            print("  Check: broker IP reachable, ESP8266 on same WiFi, firmware flashed.")
        else:
            mqtt.center()
            print("✓ MQTT ready — servo centered to start")
    tracker = FaceTracker()
    tlog = TrackingLogger()
    pan = PanTracker(mqtt=mqtt, logger=tlog)

    activity_logger: Optional[ActivityLogger] = None
    evidence_logger: Optional[EvidenceLogger] = None
    if lock_name:
        activity_logger = ActivityLogger(lock_name, config.HISTORY_DIR)
        if config.EVIDENCE_LOG_ENABLED:
            evidence_logger = EvidenceLogger(lock_name, config.HISTORY_DIR)
            print(f"Evidence log: {evidence_logger.csv_path}")

    if enable_dashboard and config.DASHBOARD_ENABLED:
        dashboard_state.reset_session()
        dash_url = start_dashboard()
        print(f"Dashboard: {dash_url}")

    cam = open_camera()
    if cam is None:
        print("ERROR: Cannot open camera.")
        print("Run: python -m src.camera_utils to find the correct camera index.")
        return False

    threshold = config.RECOGNITION_THRESHOLD
    baseline_mouth_width = None
    mouth_width_samples: List[float] = []
    last_action_frame: Dict[str, int] = {}
    frame_idx = 0
    action_display: List[Tuple[str, int]] = []
    ACTION_DISPLAY_DURATION = 15

    lost_since: Optional[float] = None
    prev_locked_track_id: Optional[int] = None
    state = "IDLE"

    # FPS accounting (tracking = loop rate, recognition = embeddings/sec).
    t_loop = time.time()
    loop_count = 0
    track_fps = 0.0
    recog_events = 0
    t_recog = time.time()
    recog_fps = 0.0
    last_frame = None
    camera_warning_frames = 0
    prev_servo_angle = float(config.SERVO_CENTER_ANGLE)
    last_srs_command = None

    window_name = "Face Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, config.DISPLAY_WINDOW_WIDTH, config.DISPLAY_WINDOW_HEIGHT)
    if start_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nFace Recognition + MQTT Tracking + Search")
    print("Controls: q=quit  r=reload  l=unlock  k=lock  s=search  c=center  f=fullscreen  +/-=threshold")
    if config.TRACKING_LOG_ENABLED:
        print("Tracking logs: ON (set TRACKING_LOG_ENABLED=False in config.py to disable)")
        if lock_name:
            tlog.lock_armed(lock_name)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                if last_frame is None:
                    time.sleep(0.05)
                    continue
                frame = last_frame.copy()
                camera_warning_frames += 1
                if camera_warning_frames == 1:
                    print("⚠ Camera frame lost — retrying (tracking continues)...")
            else:
                last_frame = frame
                camera_warning_frames = 0

            frame_idx += 1
            loop_count += 1
            frame_w = frame.shape[1]

            # --- Detect + associate (every frame when faces/search active) ---
            search_active = (
                lock_name
                and lost_since is not None
                and (time.time() - lost_since) >= config.LOST_TARGET_TIMEOUT
            )
            tracking_active = (
                bool(tracker.visible_tracks())
                or tracker.locked_track_id is not None
                or bool(lock_name)
                or search_active
                or pan.search_manual
            )
            if tracking_active:
                run_detect = frame_idx % config.DETECT_EVERY_N_FRAMES_FACE == 0
            else:
                run_detect = frame_idx % config.DETECT_EVERY_N_FRAMES_IDLE == 0
            if run_detect:
                detections = detector.detect(frame)[: config.MAX_FACES]
                visible = tracker.update(detections, frame_idx, frame_w)
            else:
                visible = tracker.visible_tracks()
            faces_present = bool(visible)

            # --- Recognition pass (cached; locked first) -------------------
            locked = tracker.locked_track
            ordered = sorted(
                visible,
                key=lambda t: (t.track_id != tracker.locked_track_id, t.track_id),
            )
            budget = config.MAX_FACES
            for tr in ordered:
                if budget <= 0:
                    break
                is_lk = tr.track_id == tracker.locked_track_id
                if tr.needs_recognition(frame_idx, is_lk, faces_present=faces_present):
                    # Looser match while searching / re-acquiring the locked speaker
                    reacquiring_face = (
                        lock_name
                        and tracker.locked_track_id is None
                    )
                    match_threshold = (
                        config.FACE_MATCH_THRESHOLD
                        if (is_lk or reacquiring_face)
                        else threshold
                    )
                    name, dist, accepted = recognize_face(
                        frame, tr.landmarks, aligner, embedder,
                        embeddings_matrix, names, match_threshold,
                    )
                    tr.apply_recognition(name, dist, accepted, frame_idx)
                    recog_events += 1
                    budget -= 1

            # --- Lock acquisition / reacquisition --------------------------
            reacquiring = state == "SEARCHING" or lost_since is not None
            if lock_name and tracker.locked_track_id is None:
                candidate = tracker.acquire_lock(
                    lock_name, require_stable_frames=reacquiring
                )
                if candidate is not None:
                    pan.end_search()
                    print(f"OK: Locked onto {lock_name} (track #{candidate.track_id})")

            locked = tracker.locked_track
            follow = tracker.follow_track()
            if follow is None and lock_name:
                follow = tracker.visible_speaker(lock_name)

            pan_label: Optional[str] = None
            commanded_angle: Optional[int] = None
            in_grace_period = False

            # --- Servo control + state machine -----------------------------
            if not lock_name:
                state = "IDLE"
                lost_since = None
                prev_locked_track_id = None
                tlog.idle()
            elif follow is not None:
                track_name = follow.name if follow.accepted else lock_name
                if prev_locked_track_id is None and lost_since is not None:
                    tlog.target_visible(
                        lock_name, follow.track_id, follow.center, pan.current_angle,
                    )
                elif prev_locked_track_id != follow.track_id:
                    tlog.target_visible(
                        lock_name, follow.track_id, follow.center, pan.current_angle,
                    )
                lost_since = None
                prev_locked_track_id = follow.track_id
                pan.end_search()
                pan_label, commanded_angle = pan.track(follow.center[0], frame_w)
                if follow.accepted and follow.name == lock_name:
                    state = "LOCKED" if pan_label == "centered" else "TRACKING"
                else:
                    state = "TRACKING"
            else:
                if locked is not None and locked.name != lock_name:
                    tracker.release_lock()
                    locked = None
                # No visible face for the locked speaker — grace, then search.
                if lost_since is None:
                    lost_since = time.time()
                    prev_locked_track_id = None
                    tlog.target_lost(lock_name)
                lost_for = time.time() - lost_since
                if lost_for >= config.LOST_TARGET_TIMEOUT or pan.search_manual:
                    if state != "SEARCHING":
                        direction = "right" if pan.last_error_sign > 0 else "left" if pan.last_error_sign < 0 else "center"
                        tlog.search_started(lock_name, pan.last_known_angle, direction)
                        pan.begin_search()
                    state = "SEARCHING"
                    pan_label, commanded_angle = pan.search()
                else:
                    state = "TRACKING"  # brief grace period: hold position
                    in_grace_period = True
                    pan_label = None
                    tlog.target_still_missing(lock_name, lost_for)
                    tlog.servo_hold(pan.current_angle, "target out of frame — holding during grace period")

            # --- SRS evidence + MQTT tracking events ---------------------
            if lock_name:
                ev_track = follow if follow is not None else locked
                srs_cmd = derive_command(
                    state,
                    pan_label,
                    pan.current_angle,
                    prev_servo_angle,
                    commanded_angle,
                    in_grace_period=in_grace_period,
                )
                confidence = ev_track.confidence if ev_track else 0.0
                distance = ev_track.best_dist if ev_track else 1.0
                if evidence_logger:
                    evidence_logger.log_frame(
                        frame_number=frame_idx,
                        confidence=confidence,
                        distance=distance,
                        system_state=state,
                        tracking_command=srs_cmd,
                        servo_angle=pan.current_angle,
                        servo_target=commanded_angle,
                        face_center_x=ev_track.center[0] if ev_track else None,
                        face_center_y=ev_track.center[1] if ev_track else None,
                        frame_width=frame_w,
                        mqtt_connected=bool(mqtt and mqtt.is_connected),
                        details=f"visible_faces={len(visible)}",
                        force=config.EVIDENCE_LOG_EVERY_FRAME,
                    )
                if mqtt and mqtt.is_connected and srs_cmd.value != last_srs_command:
                    mqtt.publish_tracking_event(
                        event=srs_cmd.value,
                        speaker_id=lock_name,
                        confidence=confidence,
                        servo_angle=int(round(pan.current_angle)),
                        motor_command=motor_command_label(srs_cmd),
                        details=state,
                    )
                    last_srs_command = srs_cmd.value
                prev_servo_angle = pan.current_angle

            # --- Activity logging for the locked, visible target -----------
            if (
                lock_name and activity_logger and follow is not None
                and frame_idx % config.ACTION_DETECT_EVERY_N_FRAMES == 0
                and follow.full_landmarks
            ):
                detected_actions, baseline_mouth_width, mouth_width_samples = action_module.detect_smile_blink(
                    frame, baseline_mouth_width, mouth_width_samples,
                    last_action_frame, frame_idx,
                    cooldown_frames=config.LOCK_ACTION_COOLDOWN_FRAMES,
                    landmarks_list=follow.full_landmarks,
                )
                for act in detected_actions:
                    activity_logger.log_activity(act, frame_idx, follow.center)
                    action_display.append((act.capitalize() + "!", ACTION_DISPLAY_DURATION))
                for mv in activity_logger.detect_and_log_movement(follow.center, frame_idx):
                    action_display.append((mv.replace("_", " ").title() + "!", ACTION_DISPLAY_DURATION))

            action_display = [(label, n - 1) for label, n in action_display if n > 1]

            # --- FPS counters ----------------------------------------------
            now = time.time()
            if now - t_loop >= 1.0:
                track_fps = loop_count / (now - t_loop)
                loop_count = 0
                t_loop = now
            if now - t_recog >= 1.0:
                recog_fps = recog_events / (now - t_recog)
                recog_events = 0
                t_recog = now

            # --- Render ----------------------------------------------------
            vis = frame
            draw_tracks(vis, visible, tracker.locked_track_id, searching=(state == "SEARCHING"))
            if state == "SEARCHING" and lock_name:
                _draw_search_banner(vis, lock_name)

            servo_angle = f"{int(round(pan.current_angle))}" if mqtt else "-"
            _draw_debug_overlay(
                vis, state, lock_name, servo_angle, len(visible),
                recog_fps, track_fps, bool(mqtt and mqtt.is_connected),
                threshold, (time.time() - lost_since) if lost_since else 0.0,
            )

            y_action = vis.shape[0] - 16 * len(action_display) - 8
            for lbl, _ in action_display:
                cv2.putText(vis, lbl, (vis.shape[1] - 160, y_action),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                y_action += 18

            cv2.imshow(window_name, vis)

            if enable_dashboard and config.DASHBOARD_ENABLED:
                face_x = follow.center[0] if follow is not None else None
                face_y = follow.center[1] if follow is not None else None
                face_err = None
                if face_x is not None and frame_w:
                    face_err = (face_x - frame_w / 2.0) / (frame_w / 2.0)
                servo_target = int(mqtt.last_status.get("target", mqtt.target_angle)) if mqtt else int(round(pan.current_angle))
                dashboard_state.update(
                    frame_number=frame_idx,
                    system_state=state,
                    locked_speaker=lock_name,
                    face_count=len(visible),
                    confidence=follow.confidence if follow and follow.accepted else 0.0,
                    face_center_x=face_x,
                    face_center_y=face_y,
                    frame_width=frame_w,
                    frame_height=frame.shape[0],
                    face_error=face_err,
                    track_fps=track_fps,
                    recog_fps=recog_fps,
                    servo_angle=float(pan.current_angle),
                    servo_target=servo_target,
                    servo_moving=mqtt.is_moving() if mqtt else False,
                    commanded_angle=commanded_angle,
                    motor_command=motor_command_label(last_srs_command) if last_srs_command else "STOP",
                    mqtt_connected=bool(mqtt and mqtt.is_connected),
                    mqtt_broker=f"{mqtt.broker_host}:{mqtt.broker_port}" if mqtt else "",
                    lost_for_sec=(time.time() - lost_since) if lost_since else 0.0,
                    threshold=threshold,
                )

            # --- Keyboard --------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if not CameraStream.is_window_open(window_name):
                print("\nDisplay window closed — exiting.")
                break
            if key == ord("q"):
                break
            if key == ord("r"):
                db = load_database()
                names = sorted(db.keys())
                embeddings_matrix = np.stack([db[n].reshape(-1) for n in names], axis=0).astype(np.float32)
                if lock_name and lock_name not in names:
                    lock_name = None
                    tracker.release_lock()
                    pan.reset()
                print(f"✓ Reloaded {len(db)} identities")
            elif key == ord("l"):
                if activity_logger:
                    activity_logger.save_summary()
                    activity_logger = None
                lock_name = None
                tracker.release_lock()
                pan.reset()
                lost_since = None
                print("Lock cleared")
            elif key == ord("k"):
                new_lock = choose_lock_identity(names)
                if new_lock:
                    lock_name = new_lock
                    tracker.release_lock()
                    pan.reset()
                    if activity_logger is None:
                        activity_logger = ActivityLogger(lock_name, config.HISTORY_DIR)
                    if config.EVIDENCE_LOG_ENABLED:
                        if evidence_logger:
                            evidence_logger.close()
                        evidence_logger = EvidenceLogger(lock_name, config.HISTORY_DIR)
                    print(f"Lock target set to {lock_name}")
            elif key == ord("s"):
                pan.toggle_search()
                print(f"Manual search: {'ON' if pan.search_manual else 'OFF'}")
            elif key == ord("c"):
                pan.force_center()
                print("Camera centered")
            elif key == ord("f"):
                prop = cv2.WND_PROP_FULLSCREEN
                cur = cv2.getWindowProperty(window_name, prop)
                cv2.setWindowProperty(
                    window_name, prop,
                    cv2.WINDOW_FULLSCREEN if cur != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL,
                )
            elif key in (ord("+"), ord("=")):
                threshold = min(1.0, threshold + 0.01)
            elif key == ord("-"):
                threshold = max(0.0, threshold - 0.01)

    finally:
        if enable_dashboard and config.DASHBOARD_ENABLED:
            dashboard_state.stop()
            stop_dashboard()
        if activity_logger:
            activity_logger.save_summary()
        if evidence_logger:
            summary = evidence_logger.close()
            print(f"Evidence saved: {summary.get('evidence_csv')}")
        if mqtt:
            mqtt.close()
        detector.close()
        cam.release()
        cv2.destroyAllWindows()

    print("✓ Tracking ended.")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face recognition with MQTT camera tracking")
    parser.add_argument("--fullscreen", "-f", action="store_true")
    parser.add_argument("--no-mqtt", action="store_true", help="Disable MQTT servo control")
    parser.add_argument("--broker", type=str, default=None, help="MQTT broker IP")
    parser.add_argument("--port", type=int, default=None, help="MQTT broker port")
    parser.add_argument("--speaker", type=str, default=None, help="Pre-enrolled speaker to lock (SRS)")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable live web dashboard")
    args = parser.parse_args()

    ok = main(
        start_fullscreen=args.fullscreen,
        enable_mqtt=not args.no_mqtt,
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        speaker_lock=args.speaker,
        enable_dashboard=not args.no_dashboard,
    )
    sys.exit(0 if ok else 1)
