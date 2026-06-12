"""
Multi-face tracker with persistent IDs and recognition caching.

Why this design (Issue #3 — "which approach is best for this project"):
- Detections from Haar+MediaPipe are noisy frame-to-frame; running ArcFace on
  every face every frame is the main cost (Issue #1).
- A lightweight IoU + centroid tracker gives each face a STABLE track ID. We
  recognize a face once, cache the result on its track, and only re-embed
  periodically. This is cheaper and far more stable than re-identifying every
  frame, and prevents identity/lock switching.
- Full DeepSORT (appearance + Kalman) is overkill here: at <=5 webcam faces a
  greedy IoU/centroid associator with velocity prediction is robust, has no
  extra dependencies, and runs in microseconds. We *do* borrow the key
  DeepSORT idea — confirm identity by a short voting window — for stability.

The LOCK is bound to a track ID, not to "whichever face currently matches the
name". That is what prevents lock switching when a second known face appears.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from . import config
from .haar_5pt import FaceDetection


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1)) * max(1, (ay2 - ay1))
    area_b = max(1, (bx2 - bx1)) * max(1, (by2 - by1))
    return inter / float(area_a + area_b - inter)


@dataclass
class Track:
    """One tracked face with cached recognition state."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    landmarks: np.ndarray
    full_landmarks: Optional[List[Any]] = None
    score: float = 1.0

    # Cached recognition (stabilized over a short window)
    name: str = "Unknown"
    best_dist: float = 1.0
    confidence: float = 0.0
    accepted: bool = False

    # Lifecycle
    hits: int = 0
    misses: int = 0
    last_seen_frame: int = -1
    last_recognized_frame: int = -10_000
    velocity: Tuple[float, float] = (0.0, 0.0)

    _recent_names: Deque[str] = field(default_factory=lambda: deque(maxlen=config.RECOGNITION_STABILIZE_WINDOW))
    _recent_dists: Deque[float] = field(default_factory=lambda: deque(maxlen=config.RECOGNITION_STABILIZE_WINDOW))

    def predicted_bbox(self) -> Tuple[int, int, int, int]:
        vx, vy = self.velocity
        x1, y1, x2, y2 = self.bbox
        return (int(x1 + vx), int(y1 + vy), int(x2 + vx), int(y2 + vy))

    def update_geometry(self, det: FaceDetection, frame_idx: int) -> None:
        new_center = ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)
        self.velocity = (
            new_center[0] - self.center[0],
            new_center[1] - self.center[1],
        )
        self.bbox = (det.x1, det.y1, det.x2, det.y2)
        self.center = new_center
        self.landmarks = det.landmarks
        self.full_landmarks = det.full_landmarks
        self.score = det.score
        self.hits += 1
        self.misses = 0
        self.last_seen_frame = frame_idx

    def apply_recognition(self, name: str, best_dist: float, accepted: bool, frame_idx: int) -> None:
        """Fold a fresh recognition result into the stabilized cache."""
        self.last_recognized_frame = frame_idx
        self._recent_names.append(name if accepted else "Unknown")
        self._recent_dists.append(best_dist)

        # Majority vote over the window for the identity.
        votes: Dict[str, int] = {}
        for n in self._recent_names:
            votes[n] = votes.get(n, 0) + 1
        voted_name = max(votes, key=votes.get)
        accepted_dists = [d for n, d in zip(self._recent_names, self._recent_dists) if n == voted_name]

        self.name = voted_name
        self.accepted = voted_name != "Unknown"
        self.best_dist = float(np.median(accepted_dists)) if accepted_dists else best_dist
        self.confidence = max(0.0, 1.0 - self.best_dist)

    def needs_recognition(self, frame_idx: int, is_locked: bool, faces_present: bool = True) -> bool:
        age = frame_idx - self.last_recognized_frame
        if self.last_recognized_frame < 0:
            return True  # never recognized
        if faces_present:
            if is_locked:
                return age >= config.RECOGNITION_INTERVAL_LOCKED_FACE
            if not self.accepted:
                return age >= config.RECOGNITION_COOLDOWN_FRAMES_FACE
            return age >= config.RECOGNITION_INTERVAL_FACE
        if is_locked:
            return age >= config.RECOGNITION_INTERVAL_LOCKED
        if not self.accepted:
            return age >= config.RECOGNITION_COOLDOWN_FRAMES
        return age >= config.RECOGNITION_INTERVAL_IDLE


class FaceTracker:
    """Associates detections to persistent tracks and owns the lock binding."""

    def __init__(self, iou_threshold: float = 0.3, max_center_dist_ratio: float = 0.15):
        self.tracks: Dict[int, Track] = {}
        self.iou_threshold = iou_threshold
        self.max_center_dist_ratio = max_center_dist_ratio
        self._next_id = 1
        self.locked_track_id: Optional[int] = None
        self._reacquire_candidate_id: Optional[int] = None
        self._reacquire_count: int = 0

    def _new_track(self, det: FaceDetection, frame_idx: int) -> Track:
        center = ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)
        track = Track(
            track_id=self._next_id,
            bbox=(det.x1, det.y1, det.x2, det.y2),
            center=center,
            landmarks=det.landmarks,
            full_landmarks=det.full_landmarks,
            score=det.score,
            hits=1,
            last_seen_frame=frame_idx,
        )
        self.tracks[self._next_id] = track
        self._next_id += 1
        return track

    def update(self, detections: List[FaceDetection], frame_idx: int, frame_w: int) -> List[Track]:
        """Associate detections -> tracks. Returns the list of currently visible tracks."""
        track_ids = list(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        matched_tracks = set()

        # Build cost as IoU (preferred), then fall back to centroid distance.
        pairs: List[Tuple[float, int, int]] = []  # (score, track_id, det_idx)
        max_center_dist = self.max_center_dist_ratio * max(frame_w, 1)
        for tid in track_ids:
            tr = self.tracks[tid]
            pred = tr.predicted_bbox()
            for di in range(len(detections)):
                det = detections[di]
                dbox = (det.x1, det.y1, det.x2, det.y2)
                iou = _iou(pred, dbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, di))
                else:
                    dcx = (det.x1 + det.x2) / 2.0
                    dcy = (det.y1 + det.y2) / 2.0
                    dist = ((dcx - tr.center[0]) ** 2 + (dcy - tr.center[1]) ** 2) ** 0.5
                    if dist <= max_center_dist:
                        # Map distance to a pseudo-score below any real IoU match.
                        pairs.append((-(dist / max_center_dist), tid, di))

        # Greedy: highest score first, each track/det used once.
        pairs.sort(key=lambda p: p[0], reverse=True)
        for score, tid, di in pairs:
            if tid in matched_tracks or di not in unmatched_dets:
                continue
            self.tracks[tid].update_geometry(detections[di], frame_idx)
            matched_tracks.add(tid)
            unmatched_dets.discard(di)

        # Age unmatched tracks.
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].misses += 1

        # Spawn tracks for unmatched detections.
        for di in unmatched_dets:
            self._new_track(detections[di], frame_idx)

        # Drop stale tracks (and release lock if the locked track died).
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].misses > config.LOST_TARGET_FRAMES:
                if tid == self.locked_track_id:
                    self.locked_track_id = None
                del self.tracks[tid]

        return self.visible_tracks()

    def visible_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.misses == 0]

    def all_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    @property
    def locked_track(self) -> Optional[Track]:
        if self.locked_track_id is None:
            return None
        return self.tracks.get(self.locked_track_id)

    def acquire_lock(
        self, lock_name: str, require_stable_frames: bool = False
    ) -> Optional[Track]:
        """Bind the lock to the best visible track matching lock_name."""
        best: Optional[Track] = None
        for tr in self.visible_tracks():
            if tr.accepted and tr.name == lock_name:
                if best is None or tr.best_dist < best.best_dist:
                    best = tr

        if best is None:
            self._reacquire_candidate_id = None
            self._reacquire_count = 0
            return None

        if not require_stable_frames:
            self.locked_track_id = best.track_id
            self._reacquire_candidate_id = None
            self._reacquire_count = 0
            return best

        if best.track_id == self._reacquire_candidate_id:
            self._reacquire_count += 1
        else:
            self._reacquire_candidate_id = best.track_id
            self._reacquire_count = 1

        if self._reacquire_count >= config.SEARCH_REACQUIRE_FRAMES:
            self.locked_track_id = best.track_id
            self._reacquire_candidate_id = None
            self._reacquire_count = 0
            return best
        return None

    def release_lock(self) -> None:
        self.locked_track_id = None
