"""
Shared face recognition utilities used by recognize.py and recognize_with_tracking.py.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .haar_5pt import FaceDetection
from .align import FaceAligner
from .embed import ArcFaceEmbedder


@dataclass
class FaceResult:
    face: FaceDetection
    name: str
    display_name: str
    accepted: bool
    best_dist: float
    confidence: float
    is_locked_person: bool

    @property
    def face_center(self) -> Tuple[float, float]:
        f = self.face
        return ((f.x1 + f.x2) / 2.0, (f.y1 + f.y2) / 2.0)


def load_database() -> Dict[str, np.ndarray]:
    if not config.DB_NPZ_PATH.exists():
        print("ERROR: Database not found. Run enrollment first.")
        return {}
    data = np.load(str(config.DB_NPZ_PATH), allow_pickle=True)
    return {k: data[k].astype(np.float32) for k in data.files}


def choose_lock_identity(names: list) -> Optional[str]:
    if not names:
        return None
    print("\nEnrolled identities:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    print("Lock/track one person? Enter number or name (or Enter for none): ", end="")
    try:
        raw = input().strip()
    except EOFError:
        return None
    if not raw:
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(names):
            return names[idx - 1]
        return None
    if raw in names:
        return raw
    low = raw.lower()
    for n in names:
        if n.lower() == low:
            return n
    print(f"Unknown name '{raw}'. Proceeding with all identities.")
    return None


def match_embedding(
    query_emb: np.ndarray,
    embeddings_matrix: np.ndarray,
    names: List[str],
    threshold: float,
    lock_name: Optional[str],
) -> Tuple[str, str, bool, float, float, bool]:
    dists = 1.0 - (embeddings_matrix @ query_emb.reshape(-1))
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    best_match_name = names[best_idx]

    if best_dist <= threshold:
        confidence = 1.0 - best_dist
        if lock_name and best_match_name == lock_name:
            return best_match_name, f"{best_match_name} (locked)", True, best_dist, confidence, True
        return best_match_name, best_match_name, True, best_dist, confidence, False

    return "Unknown", "Unknown", False, best_dist, 0.0, False


def open_camera():
    """Open a resilient camera stream (retries reads, auto-reconnect)."""
    from .camera_utils import CameraStream

    for attempt in range(3):
        cam = CameraStream()
        if cam.open():
            return cam
        if attempt < 2:
            print(f"Attempt {attempt + 1}/3: Camera not ready, retrying...")
            time.sleep(1)
    return None


def process_faces(
    frame: np.ndarray,
    faces: List[FaceDetection],
    aligner: FaceAligner,
    embedder: ArcFaceEmbedder,
    embeddings_matrix: np.ndarray,
    names: List[str],
    threshold: float,
    lock_name: Optional[str],
) -> List[FaceResult]:
    if not faces:
        return []

    aligned_faces = []
    for face in faces:
        aligned, _ = aligner.align(frame, face.landmarks)
        aligned_faces.append(aligned)

    query_embs = embedder.embed_batch(aligned_faces)
    dists_all = 1.0 - (query_embs @ embeddings_matrix.T)

    results: List[FaceResult] = []
    for i, face in enumerate(faces):
        best_idx = int(np.argmin(dists_all[i]))
        best_dist = float(dists_all[i, best_idx])
        best_match_name = names[best_idx]

        if best_dist <= threshold:
            confidence = 1.0 - best_dist
            if lock_name and best_match_name == lock_name:
                name, display_name, accepted, is_locked = (
                    best_match_name, f"{best_match_name} (locked)", True, True
                )
            else:
                name, display_name, accepted, is_locked = (
                    best_match_name, best_match_name, True, False
                )
        else:
            name, display_name, accepted, confidence, is_locked = (
                "Unknown", "Unknown", False, 0.0, False
            )
            best_dist = float(dists_all[i, best_idx])

        results.append(
            FaceResult(
                face=face,
                name=name,
                display_name=display_name,
                accepted=accepted,
                best_dist=best_dist,
                confidence=confidence if accepted else 0.0,
                is_locked_person=is_locked,
            )
        )
    return results


def recognize_face(
    frame: np.ndarray,
    landmarks: np.ndarray,
    aligner: FaceAligner,
    embedder: ArcFaceEmbedder,
    embeddings_matrix: np.ndarray,
    names: List[str],
    threshold: float,
) -> Tuple[str, float, bool]:
    """Align + embed + match a single face. Returns (name, best_dist, accepted)."""
    aligned, _ = aligner.align(frame, landmarks)
    query_emb, _ = embedder.embed(aligned)
    dists = 1.0 - (embeddings_matrix @ query_emb.reshape(-1))
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    if best_dist <= threshold:
        return names[best_idx], best_dist, True
    return "Unknown", best_dist, False


def find_locked_result(results: List[FaceResult]) -> Optional[FaceResult]:
    for result in results:
        if result.is_locked_person:
            return result
    return None


def draw_face_box(
    vis: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    lines: List[str],
    thickness: int = 2,
    confidence: Optional[float] = None,
) -> None:
    """
    Draw a face box with a filled, readable label block above it.
    Used everywhere so colors/labels stay consistent and never flicker.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    text_th = 2
    pad = 4
    sizes = [cv2.getTextSize(t, font, scale, text_th)[0] for t in lines]
    block_w = max(w for (w, _) in sizes) + pad * 2
    line_h = max(h for (_, h) in sizes) + pad
    block_h = line_h * len(lines) + pad

    by2 = y1
    by1 = y1 - block_h
    if by1 < 0:  # not enough room above — draw inside the box top
        by1 = y1
        by2 = y1 + block_h
    bx1 = x1
    bx2 = min(vis.shape[1] - 1, x1 + block_w)

    # Solid label background for contrast.
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, -1)
    text_color = (0, 0, 0) if color == config.COLOR_LOCKED else (255, 255, 255)
    ty = by1 + line_h
    for t in lines:
        cv2.putText(vis, t, (bx1 + pad, ty - pad), font, scale, text_color, text_th, cv2.LINE_AA)
        ty += line_h

    if confidence is not None:
        bar_w, bar_h = (x2 - x1), 5
        cv2.rectangle(vis, (x1, y2 + 2), (x1 + bar_w, y2 + 2 + bar_h), (60, 60, 60), -1)
        cv2.rectangle(vis, (x1, y2 + 2), (x1 + int(bar_w * confidence), y2 + 2 + bar_h), color, -1)


def draw_results(vis: np.ndarray, results: List[FaceResult], tracking_state: str = "") -> None:
    """4-color scheme (Issue #2) for the non-servo recognition view."""
    for result in results:
        face = result.face
        bbox = (face.x1, face.y1, face.x2, face.y2)
        conf_pct = int(round(result.confidence * 100))

        if result.is_locked_person:
            color = config.COLOR_LOCKED
            lines = [f"LOCKED: {result.name}", f"{conf_pct}%"]
        elif result.accepted:
            color = config.COLOR_KNOWN
            lines = [result.name, f"{conf_pct}%"]
        else:
            color = config.COLOR_UNKNOWN
            lines = ["Unknown"]

        draw_face_box(
            vis, bbox, color, lines,
            thickness=3 if result.is_locked_person else 2,
            confidence=result.confidence if (config.DISPLAY_CONFIDENCE and result.accepted) else None,
        )

        if result.is_locked_person:
            cx, cy = result.face_center
            mid_x = vis.shape[1] // 2
            cv2.line(vis, (mid_x, int(cy) - 20), (mid_x, int(cy) + 20), (255, 255, 0), 1)
            cv2.circle(vis, (int(cx), int(cy)), 4, (255, 255, 0), -1)


def draw_tracks(
    vis: np.ndarray,
    tracks,
    locked_track_id: Optional[int],
    searching: bool,
) -> None:
    """
    Draw tracked faces with stable per-track colors (Issue #2 + #3).

    - Unknown            -> RED   "Unknown"
    - Known, not locked  -> BLUE  "Name" + "conf%"
    - Locked & visible   -> GREEN "LOCKED: Name" + "conf%"
    - Locked but missing -> handled by caller via `searching` banner; here the
      locked track simply isn't in the visible list.
    """
    mid_x = vis.shape[1] // 2
    for tr in tracks:
        bbox = tr.bbox
        conf_pct = int(round(tr.confidence * 100))
        is_locked = tr.track_id == locked_track_id

        if is_locked:
            color = config.COLOR_LOCKED
            lines = [f"LOCKED: {tr.name}", f"{conf_pct}%"]
            thickness = 3
        elif tr.accepted:
            color = config.COLOR_KNOWN
            lines = [f"{tr.name}", f"{conf_pct}%"]
            thickness = 2
        else:
            color = config.COLOR_UNKNOWN
            lines = ["Unknown"]
            thickness = 2

        draw_face_box(
            vis, bbox, color, lines, thickness=thickness,
            confidence=tr.confidence if (config.DISPLAY_CONFIDENCE and tr.accepted) else None,
        )

        if is_locked:
            cx, cy = tr.center
            cv2.line(vis, (mid_x, int(cy) - 20), (mid_x, int(cy) + 20), (255, 255, 0), 1)
            cv2.circle(vis, (int(cx), int(cy)), 4, (255, 255, 0), -1)
