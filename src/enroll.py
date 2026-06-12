"""
Face enrollment module.
Captures multiple face samples, extracts embeddings, and stores a template.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from . import config
from .haar_5pt import HaarMediaPipeFaceDetector
from .align import FaceAligner
from .embed import ArcFaceEmbedder
from .camera_utils import _open_capture


def load_existing_db():
    """Load existing face database."""
    if not config.DB_NPZ_PATH.exists():
        return {}
    data = np.load(str(config.DB_NPZ_PATH), allow_pickle=True)
    return {k: data[k].astype(np.float32) for k in data.files}


def save_db(db: Dict[str, np.ndarray], metadata: dict):
    """Save face database and metadata."""
    config.ensure_dirs()
    np.savez(str(config.DB_NPZ_PATH), **db)
    config.DB_JSON_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute mean embedding and L2 normalize."""
    E = np.stack([e.reshape(-1) for e in embeddings], axis=0).astype(np.float32)
    mean = E.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)
    return mean.astype(np.float32)


def main():
    """
    Enrollment pipeline.
    Capture multiple face samples, compute template, store in database.
    """
    config.ensure_dirs()
    
    name = input("Enter person name to enroll (e.g., Alice): ").strip()
    if not name:
        print("No name provided. Exiting.")
        return False
    
    detector = HaarMediaPipeFaceDetector(min_size=config.HAAR_MIN_SIZE)
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)
    
    person_dir = config.ENROLL_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    db = load_existing_db()
    
    # Load existing samples from disk if re-enrolling
    existing_samples = []
    if list(person_dir.glob("*.jpg")):
        print(f"Found existing enrollment for {name}. Loading samples...")
        for img_path in sorted(person_dir.glob("*.jpg"))[:config.MAX_EXISTING_CROPS_PER_PERSON]:
            img = cv2.imread(str(img_path))
            if img is not None and img.shape[:2] == config.EMBEDDING_INPUT_SIZE:
                try:
                    emb, _ = embedder.embed(img)
                    existing_samples.append(emb)
                except Exception:
                    pass
        print(f"Loaded {len(existing_samples)} existing samples.")
    
    cap = _open_capture(config.CAMERA_INDEX)
    if cap is None:
        print("ERROR: Cannot open camera. Run: python -m src.camera_utils")
        return False
    
    cv2.namedWindow("Enrollment - Main")
    cv2.namedWindow("Enrollment - Aligned")
    cv2.resizeWindow("Enrollment - Aligned", 200, 200)
    
    new_samples = []
    status_msg = "Waiting for faces..."
    auto_mode = False
    last_auto_capture = 0
    
    print(f"\nEnrolling: {name}")
    print("Controls:")
    print("  SPACE  - Capture one sample")
    print("  a      - Toggle auto-capture")
    print("  s      - Save enrollment")
    print("  r      - Reset new samples (keep existing)")
    print("  q      - Quit")
    
    t0 = time.time()
    frame_count = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()
            
            vis = frame.copy()
            aligned_vis = np.zeros(
                (config.ALIGNMENT_OUTPUT_SIZE[1], config.ALIGNMENT_OUTPUT_SIZE[0], 3),
                dtype=np.uint8
            )
            
            faces = detector.detect(frame)
            
            if faces:
                f = faces[0]
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                
                for (x, y) in f.landmarks.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                aligned_vis, _ = aligner.align(frame, f.landmarks)
                
                # Auto capture
                if auto_mode and (time.time() - last_auto_capture) >= config.AUTO_CAPTURE_INTERVAL_SECONDS:
                    emb, _ = embedder.embed(aligned_vis)
                    new_samples.append(emb)
                    last_auto_capture = time.time()
                    
                    ts = int(time.time() * 1000)
                    cv2.imwrite(str(person_dir / f"{ts}.jpg"), aligned_vis)
                    status_msg = f"Auto-captured {len(new_samples)}"
            else:
                cv2.imshow("Enrollment - Aligned", aligned_vis)
            
            # UI overlay
            total_samples = len(existing_samples) + len(new_samples)
            info_lines = [
                f"ENROLL: {name}",
                f"Existing: {len(existing_samples)} | New: {len(new_samples)} | Total: {total_samples}",
                f"Need: {config.MIN_SAMPLES_TO_SAVE}-{config.MAX_SAMPLES_TO_SAVE} (target {config.SAMPLES_NEEDED_FOR_ENROLLMENT})",
                f"Auto: {'ON' if auto_mode else 'OFF'} (a)",
                f"FPS: {fps:.1f}",
                status_msg,
            ]
            
            y = 30
            for line in info_lines:
                cv2.putText(
                    vis, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                y += 25
            
            cv2.imshow("Enrollment - Main", vis)
            cv2.imshow("Enrollment - Aligned", aligned_vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                auto_mode = not auto_mode
                status_msg = f"Auto mode {'ON' if auto_mode else 'OFF'}"
            elif key == ord("r"):
                new_samples.clear()
                status_msg = "New samples reset"
            elif key == ord(" "):  # SPACE
                if not faces:
                    status_msg = "No face detected"
                else:
                    emb, _ = embedder.embed(aligned_vis)
                    new_samples.append(emb)
                    ts = int(time.time() * 1000)
                    cv2.imwrite(str(person_dir / f"{ts}.jpg"), aligned_vis)
                    status_msg = f"Captured {len(new_samples)}"
            elif key == ord("s"):
                total = len(existing_samples) + len(new_samples)
                if total < config.MIN_SAMPLES_TO_SAVE:
                    status_msg = f"Need at least {config.MIN_SAMPLES_TO_SAVE} samples ({total})"
                    continue

                # Compute template (SRS: use up to MAX_SAMPLES_TO_SAVE)
                all_embeddings = (existing_samples + new_samples)[: config.MAX_SAMPLES_TO_SAVE]
                template = mean_embedding(all_embeddings)
                
                db[name] = template
                
                metadata = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_dim": int(template.size),
                    "names": sorted(db.keys()),
                    "samples_used": total,
                }
                
                save_db(db, metadata)
                
                print(f"\n✓ Enrolled '{name}' with {total} samples")
                return True
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("Enrollment cancelled.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
