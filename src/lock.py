"""
Face Locking module (Term-02 Week-04).
Locks onto one enrolled identity, tracks that face, and records actions
(moved left/right, blink, smile) to a history file.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

from . import config
from .haar_5pt import HaarMediaPipeFaceDetector, FaceDetection
from .align import FaceAligner
from .embed import ArcFaceEmbedder
from . import actions as action_module


def load_database():
    """Load enrolled face database."""
    if not config.DB_NPZ_PATH.exists():
        return {}
    data = np.load(str(config.DB_NPZ_PATH), allow_pickle=True)
    return {k: data[k].astype(np.float32) for k in data.files}


def cosine_distance(a, b):
    """Compute cosine distance between two vectors."""
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return 1.0 - float(np.dot(a, b))


def get_full_landmarks(frame):
    """Run MediaPipe Face Mesh on frame; return first face landmark list or None."""
    return action_module.get_face_mesh_landmarks(frame)


def detect_actions(frame, prev_center_x, center_x, prev_ear, prev_mouth_width,
                   baseline_mouth_width, frame_idx, last_action_frame):
    """Detect actions: face_moved_left, face_moved_right, eye_blink, smile."""
    actions = []
    H, W = frame.shape[:2]
    cooldown = config.LOCK_ACTION_COOLDOWN_FRAMES

    if prev_center_x is not None:
        dx = center_x - prev_center_x
        if dx <= -config.LOCK_MOVEMENT_THRESHOLD_PX:
            if frame_idx - last_action_frame.get("face_moved_left", -999) >= cooldown:
                actions.append(("face_moved_left", "face moved left"))
                last_action_frame["face_moved_left"] = frame_idx
        elif dx >= config.LOCK_MOVEMENT_THRESHOLD_PX:
            if frame_idx - last_action_frame.get("face_moved_right", -999) >= cooldown:
                actions.append(("face_moved_right", "face moved right"))
                last_action_frame["face_moved_right"] = frame_idx

    landmarks_list = get_full_landmarks(frame)
    if landmarks_list is None:
        return actions

    ear = action_module.compute_ear(landmarks_list, W, H)
    if ear < config.LOCK_EAR_BLINK_THRESHOLD:
        if frame_idx - last_action_frame.get("eye_blink", -999) >= cooldown:
            actions.append(("eye_blink", "eye blink"))
            last_action_frame["eye_blink"] = frame_idx

    mouth_width = action_module.compute_mouth_width(landmarks_list, W, H)
    if baseline_mouth_width is not None and baseline_mouth_width > 1.0:
        if mouth_width >= baseline_mouth_width * config.LOCK_SMILE_MOUTH_RATIO:
            if frame_idx - last_action_frame.get("smile", -999) >= cooldown:
                actions.append(("smile", "smile or laugh"))
                last_action_frame["smile"] = frame_idx

    return actions


def main():
    """Face Locking: select one identity, lock onto that face, track and record actions."""
    db = load_database()
    if not db:
        print("ERROR: No enrolled identities. Run: python -m src.enroll")
        return False

    names = sorted(db.keys())
    print("\nEnrolled identities:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    print("\nEnter the name of the identity to lock (exact match): ", end="")
    try:
        choice = input().strip()
    except EOFError:
        choice = names[0] if names else ""
    if not choice:
        choice = names[0] if names else ""
    if choice not in db:
        print(f"ERROR: '{choice}' not in database. Choose from: {names}")
        return False
    lock_identity = choice
    print("Will lock onto:", lock_identity)

    config.ensure_dirs()
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    detector = HaarMediaPipeFaceDetector(min_size=config.HAAR_MIN_SIZE)
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)
    embeddings_matrix = np.stack([db[n].reshape(-1) for n in names], axis=0)
    lock_idx = names.index(lock_identity)
    threshold = config.DEFAULT_DISTANCE_THRESHOLD

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return False

    locked = False
    fail_count = 0
    history_file = None
    history_path = None
    prev_center_x = None
    baseline_mouth_width = None
    mouth_width_samples = []
    last_action_frame = {}
    prev_ear = None
    prev_mouth_width = None
    frame_idx = 0

    print("\nFace Locking - When the selected face appears, system will lock. q=Quit")

    t0 = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()
            frame_count += 1

            vis = frame.copy()
            H, W = frame.shape[:2]
            faces = detector.detect(frame)

            if not locked:
                for face in faces:
                    aligned, _ = aligner.align(frame, face.landmarks)
                    query_emb, _ = embedder.embed(aligned)
                    dists = np.array([cosine_distance(query_emb, embeddings_matrix[i]) for i in range(len(names))])
                    best_idx = int(np.argmin(dists))
                    best_dist = dists[best_idx]
                    if best_idx == lock_idx and best_dist <= threshold:
                        locked = True
                        fail_count = 0
                        ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
                        safe_name = lock_identity.replace(" ", "_").lower()
                        history_path = config.HISTORY_DIR / (safe_name + "_history_" + ts + ".txt")
                        history_file = open(history_path, "w", encoding="utf-8")
                        history_file.write("# Face Lock history: " + lock_identity + "\n")
                        history_file.write("# Started: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
                        history_file.write("# Format: timestamp  action_type  description\n# ---\n")
                        history_file.flush()
                        prev_center_x = (face.x1 + face.x2) / 2.0
                        mouth_width_samples = []
                        baseline_mouth_width = None
                        print("LOCKED onto", lock_identity, "Recording to", history_path.name)
                        break

            else:
                matched_face = None
                best_dist = 1.0
                for face in faces:
                    aligned, _ = aligner.align(frame, face.landmarks)
                    query_emb, _ = embedder.embed(aligned)
                    dists = np.array([cosine_distance(query_emb, embeddings_matrix[i]) for i in range(len(names))])
                    idx = int(np.argmin(dists))
                    d = dists[idx]
                    if idx == lock_idx and d <= threshold:
                        matched_face = face
                        best_dist = d
                        fail_count = 0
                        break
                    if len(faces) == 1 and d < 0.5:
                        matched_face = face
                        best_dist = d
                        fail_count += 1
                        if fail_count <= 5:
                            break
                        matched_face = None
                        break

                if matched_face is None:
                    fail_count += 1
                    if fail_count >= config.LOCK_RELEASE_FRAMES:
                        locked = False
                        if history_file:
                            history_file.close()
                            history_file = None
                        print("Lock released (face not seen for", config.LOCK_RELEASE_FRAMES, "frames).")
                else:
                    center_x = (matched_face.x1 + matched_face.x2) / 2.0
                    action_list = detect_actions(
                        frame, prev_center_x, center_x, prev_ear, prev_mouth_width,
                        baseline_mouth_width, frame_idx, last_action_frame
                    )
                    ts = time.time()
                    for action_type, desc in action_list:
                        line = "%.2f  %s  %s\n" % (ts, action_type, desc)
                        if history_file:
                            history_file.write(line)
                            history_file.flush()

                    landmarks_list = get_full_landmarks(frame)
                    if landmarks_list:
                        mw = action_module.compute_mouth_width(landmarks_list, W, H)
                        prev_ear = action_module.compute_ear(landmarks_list, W, H)
                        mouth_width_samples.append(mw)
                        if len(mouth_width_samples) > 30:
                            mouth_width_samples.pop(0)
                        if baseline_mouth_width is None and len(mouth_width_samples) >= 15:
                            baseline_mouth_width = float(np.median(mouth_width_samples))
                        prev_mouth_width = mw
                    prev_center_x = center_x

                    cv2.rectangle(vis, (matched_face.x1, matched_face.y1), (matched_face.x2, matched_face.y2), (0, 255, 0), 4)
                    cv2.putText(vis, "LOCKED: " + lock_identity, (matched_face.x1, max(0, matched_face.y1 - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(vis, "dist=%.3f" % best_dist, (matched_face.x1, matched_face.y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            if locked:
                for face in faces:
                    cx = (face.x1 + face.x2) / 2.0
                    if prev_center_x is not None and abs(cx - prev_center_x) < 50:
                        continue
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 0, 255), 2)
                    cv2.putText(vis, "Other", (face.x1, face.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                for face in faces:
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                    cv2.putText(vis, "Looking for lock...", (face.x1, face.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            header = "Lock target: " + lock_identity + " | " + ("LOCKED" if locked else "Searching...") + " | FPS: %.1f" % fps
            cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "q=quit", (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Face Locking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if history_file:
            history_file.close()
        if history_path:
            print("History saved to:", history_path)

    print("Face Locking ended.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
