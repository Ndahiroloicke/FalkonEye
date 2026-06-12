"""
Threshold evaluation module.
Analyzes genuine and impostor distance distributions.
Recommends optimal recognition threshold.
"""

import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

from . import config
from .embed import ArcFaceEmbedder


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    similarity = float(np.dot(a, b))
    return 1.0 - similarity


def load_people_data():
    """Load all enrolled people and their samples."""
    if not config.ENROLL_DIR.exists():
        print("ERROR: No enrollment data found.")
        return {}
    
    people = {}
    for person_dir in config.ENROLL_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        
        images = list(person_dir.glob("*.jpg"))
        if not images:
            continue
        
        people[person_dir.name] = sorted(images)
    
    return people


def evaluate():
    """Run threshold evaluation."""
    config.ensure_dirs()
    
    people_data = load_people_data()
    
    if not people_data:
        print("ERROR: No enrollment samples found. Run enrollment first.")
        return False
    
    print(f"\nLoading embeddings from {len(people_data)} enrolled people...")
    
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)
    
    # Load embeddings per person
    embeddings_per_person = {}
    for name, img_paths in people_data.items():
        embeddings = []
        for path in img_paths[:config.MAX_EXISTING_CROPS_PER_PERSON]:
            img = cv2.imread(str(path))
            if img is None or img.shape[:2] != config.EMBEDDING_INPUT_SIZE:
                continue
            try:
                emb, _ = embedder.embed(img)
                embeddings.append(emb)
            except Exception:
                pass
        
        if len(embeddings) >= 5:
            embeddings_per_person[name] = embeddings
    
    if len(embeddings_per_person) < 1:
        print("ERROR: Not enough valid embeddings.")
        return False
    
    # Compute genuine distances (same person)
    genuine_distances = []
    for name, embeddings in embeddings_per_person.items():
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = cosine_distance(embeddings[i], embeddings[j])
                genuine_distances.append(dist)
    
    # Compute impostor distances (different people)
    impostor_distances = []
    names = sorted(embeddings_per_person.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for emb_i in embeddings_per_person[names[i]]:
                for emb_j in embeddings_per_person[names[j]]:
                    dist = cosine_distance(emb_i, emb_j)
                    impostor_distances.append(dist)
    
    genuine = np.array(genuine_distances, dtype=np.float32)
    impostor = np.array(impostor_distances, dtype=np.float32)
    
    # Statistics
    print("\n" + "="*60)
    print("DISTANCE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    def stats_str(arr):
        if len(arr) == 0:
            return "n=0 (empty)"
        return (
            f"n={len(arr)} | "
            f"mean={arr.mean():.3f} | "
            f"std={arr.std():.3f} | "
            f"min={arr.min():.3f} | "
            f"max={arr.max():.3f} | "
            f"p25={np.percentile(arr, 25):.3f} | "
            f"p50={np.percentile(arr, 50):.3f} | "
            f"p75={np.percentile(arr, 75):.3f}"
        )
    
    print(f"\nGenuine distances (same person): {stats_str(genuine)}")
    print(f"Impostor distances (diff people): {stats_str(impostor)}")
    
    # Threshold sweep
    print("\n" + "="*60)
    print("THRESHOLD SWEEP")
    print("="*60)
    print("\nDistance | FAR (%) | FRR (%)")
    print("-" * 30)
    
    start, end, step = config.THRESHOLD_SWEEP_RANGE
    thresholds = np.arange(start, end + 1e-9, step)
    
    best_threshold = None
    best_far = float('inf')
    best_frr = float('inf')
    
    for thr in thresholds:
        if len(impostor) > 0:
            far = float(np.mean(impostor <= thr)) * 100
        else:
            far = 0
        
        if len(genuine) > 0:
            frr = float(np.mean(genuine > thr)) * 100
        else:
            frr = 0
        
        # Print sampled thresholds
        if len(thresholds) > 0 and int((thr - start) / step) % max(1, len(thresholds) // 10) == 0:
            print(f"  {thr:.2f}   |  {far:6.2f}  |  {frr:6.2f}")
        
        # Find best threshold meeting target FAR
        if far <= (config.TARGET_FAR * 100):
            if frr < best_frr:
                best_frr = frr
                best_threshold = thr
                best_far = far
    
    # Recommendation
    print("\n" + "="*60)
    if best_threshold is not None:
        print(f"RECOMMENDED THRESHOLD: {best_threshold:.2f}")
        print(f"  FAR: {best_far:.2f}% (target: {config.TARGET_FAR*100:.2f}%)")
        print(f"  FRR: {best_frr:.2f}%")
        print(f"\nEquivalent cosine similarity: {1.0 - best_threshold:.3f}")
    else:
        print(f"No threshold met target FAR ({config.TARGET_FAR*100:.2f}%).")
        print("Consider:")
        print("  - Enrolling more samples")
        print("  - Using higher quality images")
        print("  - Relaxing the FAR target")
    
    print("="*60)
    return True


if __name__ == "__main__":
    success = evaluate()
    sys.exit(0 if success else 1)
