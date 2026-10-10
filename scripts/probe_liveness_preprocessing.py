#!/usr/bin/env python3
"""Probe MiniFASNet preprocessing variants against a real face.

Compares:
  A: current uniface wrapper (BGR uint8 -> float32, no scale, no swap)
  B: upstream Silent-Face-Anti-Spoofing (blob: /127.5 - 1.0, swapRB)
  C: same as B but keeping BGR (scale+mean only, no channel swap)

A well-formed model should give a consistent, confident REAL for a genuine
face. Large inter-variant disagreement or unstable labels indicate the fed
preprocessing does not match what the ONNX graph expects.
"""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")

from uniface.spoofing import MiniFASNet
from uniface.common import softmax
from uniface_unlock.detector import FaceDetector
from uniface_unlock.camera import Camera


def variant_a(spoofer, image, bbox):
    """Current wrapper: crop, astype(float32), CHW. (already in wrapper)"""
    return spoofer.preprocess(image, bbox)


def variant_b(spoofer, image, bbox):
    """Upstream-style: blobFromImage(scale=1/127.5, mean=127.5, swapRB=True)."""
    crop = spoofer._crop_face(image, spoofer._xyxy_to_xywh(bbox))
    return cv2.dnn.blobFromImage(
        crop, 1.0 / 127.5, spoofer.input_size,
        (127.5, 127.5, 127.5), swapRB=True,
    )


def variant_c(spoofer, image, bbox):
    """Same scaling but no channel swap (BGR kept)."""
    crop = spoofer._crop_face(image, spoofer._xyxy_to_xywh(bbox))
    return cv2.dnn.blobFromImage(
        crop, 1.0 / 127.5, spoofer.input_size,
        (127.5, 127.5, 127.5), swapRB=False,
    )


def probs_for(spoofer, tensor):
    out = spoofer.session.run([spoofer.output_name], {spoofer.input_name: tensor})[0]
    p = softmax(out)[0]
    label = int(np.argmax(p))
    # graph class map: 0=fake, 1=real, 2=mask (per wrapper postprocess label==1 -> real)
    name = {0: "FAKE", 1: "REAL", 2: "MASK"}.get(label, str(label))
    return [round(float(x), 3) for x in p], name


def main():
    det = FaceDetector()
    spoof = MiniFASNet()  # V2, 80x80, scale 2.7

    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rows = []
    with Camera(device="/dev/video0", fps=10) as cam:
        attempts = 0
        while len(rows) < n_frames and attempts < n_frames * 6:
            attempts += 1
            frame = cam.read()
            if frame is None:
                continue
            try:
                face = det.validate_exactly_one_face(frame)
            except Exception:
                continue
            pa, na = probs_for(spoof, variant_a(spoof, frame, face.bbox))
            pb, nb = probs_for(spoof, variant_b(spoof, frame, face.bbox))
            pc, nc = probs_for(spoof, variant_c(spoof, frame, face.bbox))
            rows.append((pa, na, pb, nb, pc, nc))
            print(f"f{len(rows):2d}  A(rawfloat)={na} {pa}   B(blob+rgb)={nb} {pb}   C(blob+bgr)={nc} {pc}")
            time.sleep(0.25)

    print()
    for tag, idx, lbl in [("A rawfloat", 0, 1), ("B blob+rgb", 2, 3), ("C blob+bgr", 4, 5)]:
        labels = [r[lbl] for r in rows]
        real_p = [ (r[idx][1] if len(r[idx]) > 1 else 0.0) for r in rows ]
        print(f"{tag}: labels={labels.count('REAL')}/{len(rows)} REAL, "
              f"real_prob mean={np.mean(real_p):.3f} std={np.std(real_p):.3f} "
              f"min={min(real_p):.3f} max={max(real_p):.3f}")


if __name__ == "__main__":
    main()
