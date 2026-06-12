"""
Action detection module: smile, blink (and optional face movement).
Reuses landmarks from the face detector — no second MediaPipe pass.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import config
from .face_mesh import FaceMesh

_shared_mesh: Optional[FaceMesh] = None


def _ear_from_landmarks(landmarks_list: List[Any], indices: Tuple[int, ...], W: int, H: int) -> float:
    """Eye Aspect Ratio from 6 landmark indices (lower = more closed)."""
    pts = []
    for i in indices:
        lm = landmarks_list[i]
        pts.append((lm.x * W, lm.y * H))
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h < 1e-6:
        return 0.5
    return (v1 + v2) / (2.0 * h)


def _mouth_width_from_landmarks(
    landmarks_list: List[Any], left_idx: int, right_idx: int, W: int, H: int
) -> float:
    """Mouth width in pixels."""
    l = landmarks_list[left_idx]
    r = landmarks_list[right_idx]
    return float(np.hypot((r.x - l.x) * W, (r.y - l.y) * H))


def compute_ear(landmarks_list: List[Any], W: int, H: int) -> float:
    """Average Eye Aspect Ratio (both eyes). Lower = eyes more closed."""
    ear_left = _ear_from_landmarks(landmarks_list, config.LOCK_EAR_LEFT_INDICES, W, H)
    ear_right = _ear_from_landmarks(landmarks_list, config.LOCK_EAR_RIGHT_INDICES, W, H)
    return (ear_left + ear_right) / 2.0


def compute_mouth_width(landmarks_list: List[Any], W: int, H: int) -> float:
    """Mouth width in pixels."""
    return _mouth_width_from_landmarks(
        landmarks_list,
        config.LOCK_MOUTH_LEFT_INDEX,
        config.LOCK_MOUTH_RIGHT_INDEX,
        W,
        H,
    )


def get_face_mesh_landmarks(frame) -> Optional[List[Any]]:
    """Legacy helper for lock.py — prefer passing full_landmarks from the detector."""
    global _shared_mesh
    if _shared_mesh is None:
        _shared_mesh = FaceMesh(max_num_faces=1)
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _shared_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0].landmark


def detect_smile_blink_from_landmarks(
    landmarks_list: List[Any],
    frame_width: int,
    frame_height: int,
    baseline_mouth_width: Optional[float],
    mouth_width_samples: List[float],
    last_action_frame: Dict[str, int],
    frame_idx: int,
    cooldown_frames: int = 10,
) -> Tuple[List[str], Optional[float], List[float]]:
    """Detect blink/smile from landmarks already produced by the face detector."""
    actions: List[str] = []
    W, H = frame_width, frame_height

    ear = compute_ear(landmarks_list, W, H)
    mouth_width = compute_mouth_width(landmarks_list, W, H)

    if ear < config.LOCK_EAR_BLINK_THRESHOLD:
        if frame_idx - last_action_frame.get("blink", -999) >= cooldown_frames:
            actions.append("blink")
            last_action_frame["blink"] = frame_idx

    mouth_width_samples = list(mouth_width_samples) + [mouth_width]
    if len(mouth_width_samples) > 30:
        mouth_width_samples = mouth_width_samples[-30:]
    if baseline_mouth_width is None and len(mouth_width_samples) >= 15:
        baseline_mouth_width = float(np.median(mouth_width_samples))
    if baseline_mouth_width is not None and baseline_mouth_width > 1.0:
        if mouth_width >= baseline_mouth_width * config.LOCK_SMILE_MOUTH_RATIO:
            if frame_idx - last_action_frame.get("smile", -999) >= cooldown_frames:
                actions.append("smile")
                last_action_frame["smile"] = frame_idx

    return actions, baseline_mouth_width, mouth_width_samples


def detect_smile_blink(
    frame,
    baseline_mouth_width: Optional[float],
    mouth_width_samples: List[float],
    last_action_frame: Dict[str, int],
    frame_idx: int,
    cooldown_frames: int = 10,
    landmarks_list: Optional[List[Any]] = None,
) -> Tuple[List[str], Optional[float], List[float]]:
    """
    Detect blink and smile. Pass landmarks_list from FaceDetection.full_landmarks
    to avoid a second MediaPipe inference.
    """
    if landmarks_list is None:
        return [], baseline_mouth_width, mouth_width_samples

    H, W = frame.shape[:2]
    return detect_smile_blink_from_landmarks(
        landmarks_list,
        W,
        H,
        baseline_mouth_width,
        mouth_width_samples,
        last_action_frame,
        frame_idx,
        cooldown_frames,
    )
