"""
Live face recognition module.
Real-time face matching against enrolled database.
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
from .recognition_core import (
    FaceResult,
    choose_lock_identity,
    draw_results,
    find_locked_result,
    load_database,
    open_camera,
    process_faces,
)


def main(start_fullscreen: bool = False):
    """Live recognition pipeline."""
    db = load_database()
    if not db:
        print("ERROR: No enrolled identities found. Run enrollment first.")
        return False

    print(f"✓ Loaded {len(db)} enrolled identities")

    detector = HaarMediaPipeFaceDetector(min_size=config.HAAR_MIN_SIZE)
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)

    names = sorted(db.keys())
    embeddings_matrix = np.stack([db[n].reshape(-1) for n in names], axis=0).astype(np.float32)

    lock_name: Optional[str] = choose_lock_identity(names)

    activity_logger: Optional[ActivityLogger] = None
    if lock_name:
        print(f"Lock: {lock_name}")
        activity_logger = ActivityLogger(lock_name, config.HISTORY_DIR)
    else:
        print("Lock: (none) – all enrolled identities shown by name")

    from .camera_utils import CameraStream

    cam = open_camera()
    if cam is None:
        print("ERROR: Cannot open camera after 3 attempts.")
        print("Run: python -m src.camera_utils to find the correct camera index.")
        return False

    threshold = config.DEFAULT_DISTANCE_THRESHOLD
    baseline_mouth_width: Optional[float] = None
    mouth_width_samples: List[float] = []
    last_action_frame: Dict[str, int] = {}
    frame_idx = 0
    action_display: List[Tuple[str, int]] = []
    ACTION_DISPLAY_DURATION = 15

    cached_results: List[FaceResult] = []
    hold_frames = 0
    lock_lost_frames = 0

    window_name = "Live Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, config.DISPLAY_WINDOW_WIDTH, config.DISPLAY_WINDOW_HEIGHT)
    if start_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nLive Recognition")
    print("Controls: q=quit  r=reload  l=clear lock  f=fullscreen  +/-=threshold")

    t0 = time.time()
    frame_count = 0
    fps = 0.0
    last_frame = None
    camera_warning_frames = 0

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
                    print("⚠ Camera frame lost — retrying (recognition continues)...")
            else:
                last_frame = frame
                camera_warning_frames = 0

            frame_idx += 1
            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()

            face_active = bool(cached_results) or hold_frames > 0
            process_every = (
                config.PROCESS_EVERY_N_FRAMES_FACE
                if face_active
                else config.PROCESS_EVERY_N_FRAMES_IDLE
            )
            run_inference = (
                frame_idx % process_every == 0
                or not cached_results
            )

            if run_inference:
                faces = detector.detect(frame)[: config.MAX_FACES_TO_PROCESS]
                if faces:
                    cached_results = process_faces(
                        frame, faces, aligner, embedder,
                        embeddings_matrix, names, threshold, lock_name,
                    )
                    hold_frames = config.ACCEPT_HOLD_FRAMES
                else:
                    cached_results = []
                    hold_frames = 0
            else:
                hold_frames -= 1
                if hold_frames <= 0 and not lock_name:
                    cached_results = []

            locked_result = find_locked_result(cached_results) if lock_name else None
            if lock_name and locked_result:
                if locked_result.best_dist > config.LOCK_RELEASE_DISTANCE:
                    lock_lost_frames += 1
                else:
                    lock_lost_frames = 0
            elif lock_name:
                lock_lost_frames += 1

            show_locked = (
                lock_name
                and locked_result
                and lock_lost_frames < config.LOCK_RELEASE_FRAMES
            )
            display_results = []
            for result in cached_results:
                if lock_name and result.is_locked_person and not show_locked:
                    display_results.append(
                        FaceResult(
                            face=result.face,
                            name=result.name,
                            display_name=f"{result.name} (lost)",
                            accepted=False,
                            best_dist=result.best_dist,
                            confidence=result.confidence,
                            is_locked_person=False,
                        )
                    )
                else:
                    display_results.append(result)

            detected_actions: List[str] = []
            run_actions = (
                show_locked
                and activity_logger
                and frame_idx % config.ACTION_DETECT_EVERY_N_FRAMES == 0
            )
            if run_actions and locked_result and locked_result.face.full_landmarks:
                detected_actions, baseline_mouth_width, mouth_width_samples = (
                    action_module.detect_smile_blink(
                        frame,
                        baseline_mouth_width,
                        mouth_width_samples,
                        last_action_frame,
                        frame_idx,
                        cooldown_frames=config.LOCK_ACTION_COOLDOWN_FRAMES,
                        landmarks_list=locked_result.face.full_landmarks,
                    )
                )
                face_center = locked_result.face_center
                for act in detected_actions:
                    activity_logger.log_activity(act, frame_idx, face_center)
                    action_display.append((act.capitalize() + "!", ACTION_DISPLAY_DURATION))
                for movement in activity_logger.detect_and_log_movement(face_center, frame_idx):
                    action_display.append(
                        (movement.replace("_", " ").title() + "!", ACTION_DISPLAY_DURATION)
                    )

            action_display = [(label, n - 1) for label, n in action_display if n > 1]

            vis = frame
            draw_results(vis, display_results)

            lock_status = f"Lock: {lock_name}" if lock_name else "Lock: (none)"
            lost_info = ""
            if lock_name:
                lost_info = f" | Lost: {lock_lost_frames}/{config.LOCK_RELEASE_FRAMES}"
            status_color = (0, 255, 0) if camera_warning_frames == 0 else (0, 165, 255)
            cam_note = "" if camera_warning_frames == 0 else " | CAM RECONNECTING"
            cv2.putText(
                vis, f"{lock_status}{lost_info} | Thresh: {threshold:.2f} | FPS: {fps:.1f}{cam_note}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2,
            )

            y_action = 54
            for label, _ in action_display:
                cv2.putText(vis, label, (10, y_action), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_action += 24

            if activity_logger:
                stats = activity_logger.get_statistics()
                y_stat = vis.shape[0] - 100
                cv2.putText(vis, f"Blinks: {stats['counts']['blink']}  Smiles: {stats['counts']['smile']}",
                            (10, y_stat), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow(window_name, vis)

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
                cached_results = []
                lock_lost_frames = 0
                if lock_name and lock_name not in names:
                    lock_name = None
                print(f"✓ Reloaded {len(db)} identities")
            elif key == ord("l"):
                if activity_logger:
                    activity_logger.save_summary()
                    activity_logger = None
                lock_name = None
                lock_lost_frames = 0
                print("Lock cleared")
            elif key == ord("f"):
                prop = cv2.WND_PROP_FULLSCREEN
                current = cv2.getWindowProperty(window_name, prop)
                fullscreen = current != cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(
                    window_name, prop,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            elif key in (ord("+"), ord("=")):
                threshold = min(1.0, threshold + 0.01)
            elif key == ord("-"):
                threshold = max(0.0, threshold - 0.01)

    finally:
        if activity_logger:
            activity_logger.save_summary()
        detector.close()
        cam.release()
        cv2.destroyAllWindows()

    print("✓ Recognition ended.")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live face recognition")
    parser.add_argument("--fullscreen", "-f", action="store_true")
    args = parser.parse_args()
    sys.exit(0 if main(start_fullscreen=args.fullscreen) else 1)
