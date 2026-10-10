#!/usr/bin/env python3
"""Controlled test: enroll fresh template and immediately test recognition.

- Preserves backup of existing enrollment before re-enrollment.
- Compares old vs fresh template recognition scores.
- Restores original enrollment unconditionally on completion.
"""

import sys
import time
import hashlib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uniface_unlock.enrollment import FaceEnrollment
from uniface_unlock.recognizer import FaceRecognizer
from uniface_unlock.detector import FaceDetector
from uniface_unlock.camera import Camera, check_camera
from uniface_unlock.config import get_config


def sha16(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def collect_recognition_frames(recognizer, detector, template, device, target_frames=8, max_attempts=40):
    sims = []
    with Camera(device=device, fps=10) as cam:
        attempts = 0
        while len(sims) < target_frames and attempts < max_attempts:
            attempts += 1
            frame = cam.read()
            if frame is None:
                continue
            try:
                face = detector.validate_exactly_one_face(frame)
            except Exception:
                continue
            try:
                is_match, sim = recognizer.verify(frame, face.landmarks, template)
            except Exception:
                continue
            w = float(face.bbox[2] - face.bbox[0])
            h = float(face.bbox[3] - face.bbox[1])
            sims.append((sim, is_match, w, h))
            print(f"    frame {len(sims):2d}: bbox {w:.0f}x{h:.0f}px  sim={sim:.3f}  match={is_match}")
            time.sleep(0.15)
    return sims


def main():
    enrollment_dir = Path.home() / ".config" / "uniface-unlock"
    emb_file = enrollment_dir / "enrollment.npy"
    backup_file = enrollment_dir / "backups" / "enrollment.pre-controlled-test.npy"

    if not emb_file.exists():
        print("ERROR: No existing enrollment found.")
        return 1

    # Step 1: Backup existing
    old_emb = np.load(emb_file)
    backup_file.write_bytes(emb_file.read_bytes())
    backup_file.chmod(0o600)
    print(f"Step 1: Backed up existing template: shape={old_emb.shape} sha16={sha16(old_emb)}")

    config = get_config()
    device = config.get("camera.device", "/dev/video0")
    if not check_camera(device):
        print(f"ERROR: Camera {device} unavailable.")
        return 1

    recognizer = FaceRecognizer()
    detector = FaceDetector()
    enrollment = FaceEnrollment()

    try:
        # Step 2: Test existing template
        print(f"\nStep 2: Testing EXISTING template (shape={old_emb.shape}, sha16={sha16(old_emb)})...")
        old_results = collect_recognition_frames(recognizer, detector, old_emb, device, target_frames=8)
        if old_results:
            old_sims = [r[0] for r in old_results]
            old_matches = sum(1 for r in old_results if r[1])
            print(f"  Old template stats: n={len(old_sims)} mean={np.mean(old_sims):.3f} min={np.min(old_sims):.3f} max={np.max(old_sims):.3f} std={np.std(old_sims):.3f} matches={old_matches}/{len(old_results)}")
        else:
            print("  No frames collected for old template.")
            old_sims = []

        # Step 3: Fresh enrollment
        print("\nStep 3: Enrolling FRESH template with current code (5 samples)...")
        enrollment.enroll(num_samples=5, camera_device=device)
        new_emb = enrollment.load_enrollment()
        print(f"  Fresh template enrolled: shape={new_emb.shape} dtype={new_emb.dtype} norm={np.linalg.norm(new_emb):.6f} sha16={sha16(new_emb)}")

        # Step 4: Test fresh template immediately
        print(f"\nStep 4: Testing FRESH template immediately (same session, positioning)...")
        new_results = collect_recognition_frames(recognizer, detector, new_emb, device, target_frames=8)
        if new_results:
            new_sims = [r[0] for r in new_results]
            new_matches = sum(1 for r in new_results if r[1])
            print(f"  Fresh template stats: n={len(new_sims)} mean={np.mean(new_sims):.3f} min={np.min(new_sims):.3f} max={np.max(new_sims):.3f} std={np.std(new_sims):.3f} matches={new_matches}/{len(new_results)}")
        else:
            print("  No frames collected for fresh template.")
            new_sims = []

        # Step 5: Comparison
        print("\n" + "=" * 60)
        print("COMPARISON: Old vs Fresh Enrollment")
        print("=" * 60)
        if old_emb.shape == new_emb.shape:
            cos_between = float(np.dot(old_emb, new_emb) / (np.linalg.norm(old_emb) * np.linalg.norm(new_emb)))
            print(f"Cosine similarity between old and fresh template: {cos_between:.3f}")
        else:
            print(f"Shape mismatch: old={old_emb.shape} vs fresh={new_emb.shape} — cannot compute similarity")

        if old_sims and new_sims:
            print(f"Old template recognition:   mean={np.mean(old_sims):.3f} [{np.min(old_sims):.3f} – {np.max(old_sims):.3f}]  match_rate={old_matches}/{len(old_results)}")
            print(f"Fresh template recognition: mean={np.mean(new_sims):.3f} [{np.min(new_sims):.3f} – {np.max(new_sims):.3f}]  match_rate={new_matches}/{len(new_results)}")
            delta = np.mean(new_sims) - np.mean(old_sims)
            print(f"Difference (fresh - old):   {delta:+.3f}")

    finally:
        # Step 6: Restore original enrollment unconditionally
        print(f"\nStep 6: RESTORING ORIGINAL ENROLLMENT from backup...")
        emb_file.write_bytes(backup_file.read_bytes())
        emb_file.chmod(0o600)
        restored_emb = np.load(emb_file)
        print(f"  Restored: shape={restored_emb.shape} sha16={sha16(restored_emb)} match={np.array_equal(old_emb, restored_emb)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())