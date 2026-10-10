#!/usr/bin/env python3
"""Controlled test: enroll fresh template and immediately test recognition.

Preserves backup of existing enrollment before any re-enrollment.
Compares old and new enrollment results where technically valid.

Usage:
    python tests/controlled_enrollment_test.py [--live-frames N]
"""

import sys
import time
import hashlib
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uniface_unlock.enrollment import FaceEnrollment
from uniface_unlock.recognizer import FaceRecognizer
from uniface_unlock.detector import FaceDetector
from uniface_unlock.camera import Camera, check_camera
from uniface_unlock.config import get_config


def sha256_16(data):
    """Return first 16 chars of SHA256."""
    if isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def backup_existing_enrollment():
    """Backup existing enrollment if present."""
    enrollment_dir = Path.home() / ".config" / "uniface-unlock"
    emb_file = enrollment_dir / "enrollment.npy"
    meta_file = enrollment_dir / "enrollment.meta.json"
    
    if not emb_file.exists():
        return None
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = enrollment_dir / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    for f in [emb_file, meta_file]:
        if f.exists():
            bf = backup_dir / f"{f.stem}.{ts}{f.suffix}"
            bf.write_bytes(f.read_bytes())
            bf.chmod(0o600)
            print(f"Backed up {f.name} -> {bf.name}")
    
    return backup_dir


def test_recognition(enrollment, recognizer, detector, camera_device, num_frames=10):
    """Run recognition test and return similarity stats."""
    tmpl = enrollment.load_enrollment()
    sims = []
    
    with Camera(device=camera_device, fps=10) as cam:
        collected = 0
        attempts = 0
        while collected < num_frames and attempts < num_frames * 20:
            attempts += 1
            frame = cam.read()
            if frame is None:
                continue
            
            try:
                face = detector.validate_exactly_one_face(frame)
            except Exception:
                continue
            
            try:
                is_match, sim = recognizer.verify(frame, face.landmarks, tmpl)
            except Exception:
                continue
            
            w = float(face.bbox[2] - face.bbox[0])
            h = float(face.bbox[3] - face.bbox[1])
            sims.append((sim, is_match, w, h))
            collected += 1
            print(f"  frame {collected}: bbox {w:.0f}x{h:.0f}px  sim={sim:.3f}  match={is_match}")
            time.sleep(0.15)
    
    if sims:
        sims_only = [s[0] for s in sims]
        return {
            'n': len(sims_only),
            'mean': float(np.mean(sims_only)),
            'std': float(np.std(sims_only)),
            'min': float(np.min(sims_only)),
            'max': float(np.max(sims_only)),
            'match_rate': sum(1 for s in sims if s[1]) / len(sims),
        }
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Controlled enrollment test")
    parser.add_argument("--live-frames", "-n", type=int, default=15,
                        help="Number of live frames to test")
    parser.add_argument("--enroll-samples", type=int, default=10,
                        help="Samples to collect during enrollment")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CONTROLLED ENROLLMENT TEST")
    print("=" * 60)
    
    # Backup existing
    backup_dir = backup_existing_enrollment()
    
    config = get_config()
    device = config.get("camera.device", "/dev/video0")
    
    if not check_camera(device):
        print(f"❌ Camera not available: {device}")
        return 1
    
    print(f"\nCamera: {device}")
    print(f"Resolution: {config.get('camera.resolution')}")
    
    # Initialize components
    enrollment = FaceEnrollment()
    recognizer = FaceRecognizer()
    detector = FaceDetector()
    
    # Check existing enrollment
    has_old = enrollment.is_enrolled()
    old_meta = enrollment.load_meta() if has_old else None
    
    if has_old:
        print("\n" + "=" * 60)
        print("EXISTING ENROLLMENT ANALYSIS")
        print("=" * 60)
        emb = enrollment.load_enrollment()
        print(f"Template file:     {enrollment.embedding_path.with_suffix('.npy')}")
        print(f"Template sha256:   {sha256_16(emb)}")
        print(f"Shape:             {emb.shape}")
        print(f"Dtype:             {emb.dtype}")
        print(f"L2 norm:           {float(np.linalg.norm(emb)):.6f}")
        print(f"Finite:            {np.isfinite(emb).all()}")
        
        if old_meta:
            print(f"Enrollment time:   {old_meta.get('created_utc', 'unknown')}")
            print(f"Model class:       {old_meta.get('recognition_class', 'unknown')}")
            print(f"Model sha256:      {old_meta.get('recognition_model_sha256', 'unknown')[:16]}")
            print(f"Input size:        {old_meta.get('recognition_input_size', 'unknown')}")
            print(f"Input mean/std:    {old_meta.get('recognition_input_mean', 'unknown')}/{old_meta.get('recognition_input_std', 'unknown')}")
        
        print("\nTesting existing template...")
        old_stats = test_recognition(enrollment, recognizer, detector, device, args.live_frames)
        if old_stats:
            print(f"  Stats: n={old_stats['n']} mean={old_stats['mean']:.3f} min={old_stats['min']:.3f} max={old_stats['max']:.3f} std={old_stats['std']:.3f} match_rate={old_stats['match_rate']:.1%}")
        else:
            print("  No valid frames collected")
            old_stats = None
    
    # Fresh enrollment
    print("\n" + "=" * 60)
    print("FRESH ENROLLMENT")
    print("=" * 60)
    print(f"Collecting {args.enroll_samples} samples...")
    
    # Re-enroll (overwrites existing)
    enrollment.enroll(num_samples=args.enroll_samples, camera_device=device)
    
    # Reload
    new_emb = enrollment.load_enrollment()
    new_meta = enrollment.load_meta()
    
    print(f"\nNew template:")
    print(f"  Template sha256: {sha256_16(new_emb)}")
    print(f"  Shape:           {new_emb.shape}")
    print(f"  Dtype:           {new_emb.dtype}")
    print(f"  L2 norm:         {float(np.linalg.norm(new_emb)):.6f}")
    print(f"  Finite:          {np.isfinite(new_emb).all()}")
    
    if new_meta:
        print(f"  Enrollment time: {new_meta.get('created_utc', 'unknown')}")
        print(f"  Model class:     {new_meta.get('recognition_class', 'unknown')}")
        print(f"  Model sha256:    {new_meta.get('recognition_model_sha256', 'unknown')[:16]}")
        print(f"  Input size:      {new_meta.get('recognition_input_size', 'unknown')}")
        print(f"  Input mean/std:  {new_meta.get('recognition_input_mean', 'unknown')}/{new_meta.get('recognition_input_std', 'unknown')}")
    
    print("\nTesting fresh template...")
    new_stats = test_recognition(enrollment, recognizer, detector, device, args.live_frames)
    if new_stats:
        print(f"  Stats: n={new_stats['n']} mean={new_stats['mean']:.3f} min={new_stats['min']:.3f} max={new_stats['max']:.3f} std={new_stats['std']:.3f} match_rate={new_stats['match_rate']:.1%}")
    else:
        print("  No valid frames collected")
        new_stats = None
    
    # Comparison
    if has_old and old_stats and new_stats:
        print("\n" + "=" * 60)
        print("COMPARISON (old vs new enrollment)")
        print("=" * 60)
        print(f"  Old template mean similarity: {old_stats['mean']:.3f}  match_rate: {old_stats['match_rate']:.1%}")
        print(f"  New template mean similarity: {new_stats['mean']:.3f}  match_rate: {new_stats['match_rate']:.1%}")
        print(f"  Difference (new - old):       {new_stats['mean'] - old_stats['mean']:+.3f}")
        
        # Template similarity (are old and new templates similar?)
        old_emb = np.load(Path.home() / ".config" / "uniface-unlock" / "backups" / f"enrollment.{backup_dir.name.split('.')[-1]}.npy") if backup_dir else None
        # Actually just use the loaded one since we didn't save old separately
        # We can load from the backup if it exists
        if backup_dir:
            backup_emb_files = list(backup_dir.glob("enrollment.*.npy"))
            if backup_emb_files:
                latest_backup = sorted(backup_emb_files)[-1]
                old_emb_backup = np.load(latest_backup)
                sim_old_new = float(np.dot(old_emb_backup, new_emb) / (np.linalg.norm(old_emb_backup) * np.linalg.norm(new_emb)))
                print(f"\n  Template-to-template cosine similarity: {sim_old_new:.3f}")
                if sim_old_new < 0.5:
                    print("  ⚠️  WARNING: Old and new templates are significantly different!")
                    print("     This suggests enrollment conditions or model changed.")
    
    # Clean up: restore old enrollment from backup
    if backup_dir:
        print("\n" + "=" * 60)
        print("RESTORING ORIGINAL ENROLLMENT")
        print("=" * 60)
        backup_files = list(backup_dir.glob("enrollment.*.npy"))
        if backup_files:
            latest_backup = sorted(backup_files)[-1]
            enrollment.save_enrollment(np.load(latest_backup))
            enrollment.save_enrollment_meta()
            print(f"Restored from {latest_backup.name}")
    
    print("\n✅ Test complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())