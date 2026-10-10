#!/usr/bin/env python3
"""Numerical verification of the alignment path.

Checks:
 1. Landmark geometry in the raw frame (left/right eye order, eye-line angle).
 2. Transformed landmark positions in aligned space vs the arcface reference
    template (should be near-exact for a similarity warp).
 3. Mirror/axis checks: eye distance and eye-line dy preserved after warp.
 4. Crop validity: fraction of pure-black border pixels (warp coverage).
 5. Temporal stability: embedding similarity between consecutive frames.
 6. Sensitivity: embedding vs slight roll (crop rotated +/-5 deg first).
"""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")

from uniface.face_utils import estimate_norm, face_alignment, transform_points_2d, compute_similarity
from uniface_unlock.detector import FaceDetector
from uniface_unlock.recognizer import FaceRecognizer
from uniface_unlock.camera import Camera


REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


def black_frac(img):
    m = img.mean(axis=2)
    edge = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
    return float((edge < 1.0).mean())


def main():
    det = FaceDetector()
    rec = FaceRecognizer()

    frame = None
    face = None
    prev_emb = None
    consec = []
    with Camera(device="/dev/video0", fps=10) as cam:
        for i in range(30):
            f = cam.read()
            if f is None:
                continue
            try:
                fa = det.validate_exactly_one_face(f)
            except Exception:
                continue
            lm = np.asarray(fa.landmarks, dtype=np.float32)

            emb = rec.extract_embedding(f, lm)
            if prev_emb is not None:
                consec.append(float(compute_similarity(prev_emb, emb)))
            prev_emb = emb

            if frame is None:
                frame, face, lm0 = f, fa, lm

            if len(consec) >= 6:
                break
            time.sleep(0.15)

    if frame is None:
        print("NO FACE CAPTURED")
        return 1

    print("== 1. raw landmark geometry ==")
    le, re_, nose, lmouth, rmouth = lm0
    print(f"  left_eye=({le[0]:.1f},{le[1]:.1f}) right_eye=({re_[0]:.1f},{re_[1]:.1f})")
    print(f"  nose=({nose[0]:.1f},{nose[1]:.1f}) l_mouth=({lmouth[0]:.1f},{lmouth[1]:.1f}) r_mouth=({rmouth[0]:.1f},{rmouth[1]:.1f})")
    print(f"  left_eye.x < right_eye.x : {le[0] < re_[0]}   (expected True, no mirror)")
    dx = re_[0] - le[0]; dy = re_[1] - le[1]
    print(f"  eye-line: dx={dx:.1f} dy={dy:.1f} angle={np.degrees(np.arctan2(dy, dx)):.1f}deg")

    print("== 2. aligned-space landmarks vs reference ==")
    M, M_inv = estimate_norm(lm0, 112)
    aligned_pts = transform_points_2d(lm0, M)
    err = np.linalg.norm(aligned_pts - REF, axis=1)
    for i, (a, r) in enumerate(zip(aligned_pts, REF)):
        print(f"  pt{i}: aligned=({a[0]:6.2f},{a[1]:6.2f}) ref=({r[0]:6.2f},{r[1]:6.2f}) err={err[i]:.3f}px")
    print(f"  max err = {err.max():.3f}px  (similarity warp => should be < ~1px)")

    print("== 3. warp preserves geometry ==")
    aligned_img, _ = face_alignment(frame, lm0, 112)
    a_le, a_re = aligned_pts[0], aligned_pts[1]
    raw_eye_dist = float(np.hypot(*(re_ - le)))
    raw_face_w = float(fa.bbox[2] - fa.bbox[0])
    scale = 35.2372 / float(np.hypot(*(re_ - le))) * np.hypot(*(REF[1] - REF[0])) / float(np.hypot(*(re_ - le)))
    # simpler: aligned eye dist / expected ref eye dist
    ref_eye_dist = float(np.hypot(*(REF[1] - REF[0])))  # 35.24
    a_eye_dist = float(np.hypot(*(a_re - a_le)))
    print(f"  aligned eye dist {a_eye_dist:.2f} vs ref {ref_eye_dist:.2f}  (expect equal)")
    print(f"  aligned eye-line dy = {abs(a_re[1]-a_le[1]):.2f}px (expect ~0.2)")
    print(f"  raw face bbox w={raw_face_w:.0f}px -> aligned scale ~{ref_eye_dist/raw_eye_dist:.2f}x")

    print("== 4. crop validity ==")
    bf = black_frac(aligned_img)
    print(f"  aligned 112x112 black border fraction: {bf:.3f}")
    print(f"  crop stats: min={aligned_img.min()} max={aligned_img.max()} mean={aligned_img.mean():.1f}")

    print("== 5. temporal embedding stability (consecutive frames, ~150ms apart) ==")
    if consec:
        print(f"  n={len(consec)} mean={np.mean(consec):.3f} min={np.min(consec):.3f} max={np.max(consec):.3f}")

    print("== 6. roll sensitivity: rotate frame +/-5deg before pipeline ==")
    h, w = frame.shape[:2]
    c = (w/2, h/2)
    for ang in (-5, 5):
        R = cv2.getRotationMatrix2D(c, ang, 1.0)
        rot = cv2.warpAffine(frame, R, (w, h), borderValue=(0, 0, 0))
        # landmarks must be rotated too - emulate the detector seeing the rotated face
        ones = np.ones((5, 1), dtype=np.float32)
        lm_rot = (R @ np.hstack([lm0, ones]).T).T
        emb_rot = rec.extract_embedding(rot, lm_rot.astype(np.float32))
        s = float(compute_similarity(rec.extract_embedding(frame, lm0), emb_rot))
        print(f"  sim(original, rolled {ang:+d}deg re-aligned) = {s:.3f}")

    print("== 7. crop-size sensitivity: re-align at 2x/0.5x distance ==")
    # emulate distance change by scaling the image (alignment normalizes scale,
    # so sim should stay high if pipeline is scale-correct)
    for sf in (1.5, 0.6):
        big = cv2.resize(frame, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA if sf < 1 else cv2.INTER_CUBIC)
        lm_s = (lm0 * sf).astype(np.float32)
        emb_s = rec.extract_embedding(big, lm_s)
        s = float(compute_similarity(rec.extract_embedding(frame, lm0), emb_s))
        print(f"  sim(original, scale x{sf}) = {s:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
