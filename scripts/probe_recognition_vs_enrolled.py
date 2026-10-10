"""Decisive probe: does SCRFD input_size change recognition vs the ENROLLED template?

The enrolled embedding was produced with the default 640x640 detector. This probe
measures, on the same live frames, whether switching to a smaller detector input
moves the cosine similarity to the enrolled template enough to flip the
configured 0.70 decision. It also reports the margin to the threshold.

No biometric values are printed -- only shape, similarity, and margin.
Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="640,480,320")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--device", default="/dev/video0")
    args = ap.parse_args()

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    import cv2
    from uniface.detection import SCRFD
    from uniface.recognition import MobileFace
    from uniface_unlock.enrollment import FaceEnrollment
    from uniface_unlock.config import get_config

    sizes = [int(s) for s in args.sizes.split(",")]
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frames = []
    attempts = 0
    while len(frames) < args.frames and attempts < args.frames * 6:
        attempts += 1
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame.copy())
    cap.release()
    if not frames:
        raise SystemExit("no frames captured")

    enrollment = FaceEnrollment()
    if not enrollment.is_enrolled():
        raise SystemExit("no enrollment present -- cannot measure decision impact")
    tmpl = enrollment.load_enrollment()
    print(f"enrolled template shape={tmpl.shape} dtype={tmpl.dtype}")

    detectors = {}
    recognizer = MobileFace(providers=["CPUExecutionProvider"])
    threshold = float(get_config().get("recognition.threshold", 0.7))

    per_size = {}
    for size in sizes:
        det = SCRFD(input_size=(size, size))
        sims = []
        for frame in frames:
            faces = det.detect(frame)
            if not faces:
                continue
            face = max(faces, key=lambda f: float(f.confidence))
            try:
                emb = recognizer.get_normalized_embedding(frame, face.landmarks)
                sims.append(float(np.dot(emb, tmpl)))
            except Exception:
                continue
        if not sims:
            print(f"size={size}: no usable faces")
            del det
            continue
        a = np.array(sims)
        per_size[size] = a
        print(f"size={size}: n={len(a)} mean={a.mean():.3f} min={a.min():.3f} "
              f"max={a.max():.3f} std={a.std():.3f} "
              f"match_rate={np.mean(a >= threshold)*100:.1f}% "
              f"margin_to_threshold(min)={a.min()-threshold:+.3f}")
        del det

    if 640 in per_size and 480 in per_size:
        diff = per_size[480] - per_size[640]
        print(f"\ndelta(sim_480 - sim_640): mean={diff.mean():+.3f} "
              f"max_abs={np.abs(diff).max():.3f} "
              f"decision_flips={int(np.sum((per_size[480] >= threshold) != (per_size[640] >= threshold)))}")
    if 640 in per_size and 320 in per_size:
        diff = per_size[320] - per_size[640]
        print(f"delta(sim_320 - sim_640): mean={diff.mean():+.3f} "
              f"max_abs={np.abs(diff).max():.3f} "
              f"decision_flips={int(np.sum((per_size[320] >= threshold) != (per_size[640] >= threshold)))}")

    print(f"\nconfigured recognition threshold: {threshold:.2f} (unchanged)")


if __name__ == "__main__":
    main()