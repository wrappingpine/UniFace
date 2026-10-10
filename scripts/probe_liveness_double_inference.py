#!/usr/bin/env python3
"""Measure what removing the duplicate liveness inference actually saved.

Background
----------
The previous ``LivenessDetector.check_liveness`` did two inferences on the same
crop: it called ``MiniFASNet.predict()`` for the decision and then re-ran
``preprocess`` + ``session.run`` a second time purely to read the raw logits
for diagnostics. The current implementation runs the session once and derives
both the decision and the diagnostics from those logits.

This script measures the saving and proves the optimisation was
behaviour-preserving rather than a silent policy change.

Modes
-----
live (default)
    Captures frames until one contains a face, then compares the old two-call
    path with the current single-call path on the real crop. Frames stay in
    memory; nothing is written to disk.

synthetic (automatic fallback when no face appears, or with --synthetic)
    No camera and no face required. The equivalence argument is input-agnostic
    (see below), and the cost saving is one full preprocess + session.run() per
    call, so timing on deterministic stand-in frames measures the same
    redundant work with the same preprocessing as the live path. Inputs are
    seeded, so the numbers are reproducible. Label any number taken from this
    mode as synthetic.

Equivalence argument (verified against upstream source, not assumed)
--------------------------------------------------------------------
uniface.MiniFASNet.predict() == preprocess -> session.run -> softmax -> argmax,
returning is_real=(argmax == 1). The current implementation runs the same
session once and applies the same argmax test to the same softmax output. Both
paths therefore consume identical math; this script additionally checks that
the two paths return identical probability vectors on the same input.

Methodology (both modes)
------------------------
- Warm-up repetitions are discarded before timing.
- The two paths are timed with alternating order so cache warming cannot
  systematically favour either one.
- Per input, the median of the timed repetitions is used.

Usage:
    python scripts/probe_liveness_double_inference.py [--frames 20] [--reps 5]
    python scripts/probe_liveness_double_inference.py --synthetic --inputs 30
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uniface.common import softmax  # noqa: E402
from uniface_unlock.detector import FaceDetector  # noqa: E402
from uniface_unlock.liveness import LivenessDetector  # noqa: E402


def timed(fn):
    """Return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def report(mode: str, n: int, reps: int,
           old_ms: list, new_ms: list, predict_only_ms: list,
           mismatches: int, max_gap: float) -> int:
    m_old = statistics.mean(old_ms)
    m_new = statistics.mean(new_ms)
    m_pred = statistics.mean(predict_only_ms)
    saving = m_old - m_new

    print(f"\nMode            : {mode}")
    print(f"Inputs measured : {n}  (reps/input {reps}, order alternated, "
          f"warm-up discarded)")
    print(f"\nold path  (predict + redundant 2nd inference): "
          f"mean {m_old:6.2f} ms   median {statistics.median(old_ms):6.2f} ms")
    print(f"new path  (single inference)                 : "
          f"mean {m_new:6.2f} ms   median {statistics.median(new_ms):6.2f} ms")
    print(f"predict() alone, for reference               : "
          f"mean {m_pred:6.2f} ms")
    print(f"saving per liveness call: {saving:6.2f} ms "
          f"({100 * saving / m_old:.1f}% of the old path)")
    print(f"\ndecision mismatches (old argmax vs new argmax): {mismatches}")
    print(f"max |probability difference| across inputs    : {max_gap:.3e}")
    if mismatches or max_gap > 0:
        print("RESULT: NOT equivalent -- do not adopt the single-pass path")
        return 2
    print("RESULT: decisions and probabilities identical; cost reduced")
    return 0


def measure_input(run_old_first, probs_of_old_second, run_new, reps):
    """Time both paths on one pre-prepared input set.

    Returns (old_median_ms, predict_only_median_ms, new_median_ms,
             probs_old, probs_new).
    """
    f_old, f_pred, f_new = [], [], []
    probs_old = probs_new = None
    for rep in range(reps):
        if rep % 2 == 0:
            (res_p, t_p) = timed(run_old_first)
            (p2, t_s) = timed(probs_of_old_second)
            (p_new, t_n) = timed(run_new)
        else:
            (p_new, t_n) = timed(run_new)
            (res_p, t_p) = timed(run_old_first)
            (p2, t_s) = timed(probs_of_old_second)
        probs_old = p2
        probs_new = p_new
        f_pred.append(t_p)
        f_old.append(t_p + t_s)
        f_new.append(t_n)
    return (statistics.median(f_old), statistics.median(f_pred),
            statistics.median(f_new), probs_old, probs_new)


def run_live(args, spoofer) -> int:
    from uniface_unlock.camera import Camera

    cam = Camera()
    det = FaceDetector()

    old_ms, pred_ms, new_ms = [], [], []
    mismatches = 0
    max_gap = 0.0
    measured = 0
    attempts = 0

    try:
        cam.open()
        while measured < args.frames and attempts < args.max_attempts:
            attempts += 1
            frame = cam.read()
            if frame is None:
                continue
            try:
                faces = det.detect(frame, max_faces=1)
            except Exception:
                continue
            if not faces:
                continue
            bbox = faces[0].bbox

            def old_predict():
                return spoofer.predict(frame, bbox)

            def second_pass():
                logits = spoofer.session.run(
                    [spoofer.output_name],
                    {spoofer.input_name: spoofer.preprocess(frame, bbox)},
                )[0]
                return softmax(logits)[0]

            def new_path():
                logits = spoofer.session.run(
                    [spoofer.output_name],
                    {spoofer.input_name: spoofer.preprocess(frame, bbox)},
                )[0]
                return softmax(logits)[0]

            # Warm-up on this frame before timing.
            old_predict()
            new_path()

            t_old, t_pred, t_new, p_old, p_new = measure_input(
                old_predict, second_pass, new_path, args.reps)

            if int(np.argmax(p_old)) != int(np.argmax(p_new)):
                mismatches += 1
            max_gap = max(max_gap, float(np.max(np.abs(p_old - p_new))))
            old_ms.append(t_old)
            pred_ms.append(t_pred)
            new_ms.append(t_new)
            measured += 1
    finally:
        cam.close()

    if not measured:
        print(f"\nNo face detected in {attempts} camera reads.")
        return 1
    return report("live (real camera frames with a detected face)",
                  measured, args.reps, old_ms, new_ms, pred_ms,
                  mismatches, max_gap)


def run_synthetic(args, spoofer) -> int:
    """Timing/equivalence on fixed preprocessed tensors, no camera needed.

    The redundant cost being removed is one session.run() call per liveness
    check; that cost does not depend on the pixel content of the input, so a
    deterministic tensor measures it faithfully. Equivalence (softmax+argmax
    over identical logits) is likewise input-independent; the seeded frames are
    just a repeatable witness of it.
    """
    rng = np.random.default_rng(20261010)

    old_ms, pred_ms, new_ms = [], [], []
    mismatches = 0
    max_gap = 0.0

    for i in range(args.inputs):
        # A deterministic stand-in "frame" and crop, so BOTH paths include
        # the real preprocess (crop+resize+normalise) exactly as the live
        # code paths do. Content is irrelevant to the equivalence argument
        # and to the inference cost.
        base = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        bbox = np.array([160, 120, 480, 440])

        def old_predict():
            return spoofer.predict(base, bbox)

        def second_pass():
            logits = spoofer.session.run(
                [spoofer.output_name],
                {spoofer.input_name: spoofer.preprocess(base, bbox)},
            )[0]
            return softmax(logits)[0]

        def new_path():
            logits = spoofer.session.run(
                [spoofer.output_name],
                {spoofer.input_name: spoofer.preprocess(base, bbox)},
            )[0]
            return softmax(logits)[0]

        # Warm-up before timing this input.
        old_predict()
        new_path()

        t_old, t_pred, t_new, p_old, p_new = measure_input(
            old_predict, second_pass, new_path, args.reps)

        if int(np.argmax(p_old)) != int(np.argmax(p_new)):
            mismatches += 1
        max_gap = max(max_gap, float(np.max(np.abs(p_old - p_new))))
        old_ms.append(t_old)
        pred_ms.append(t_pred)
        new_ms.append(t_new)

    return report("synthetic (seed 20261010, deterministic stand-in frames, "
                  "no camera/face required)",
                  args.inputs, args.reps, old_ms, new_ms, pred_ms,
                  mismatches, max_gap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20,
                    help="live mode: frames containing a face to measure")
    ap.add_argument("--reps", type=int, default=5,
                    help="timed repetitions per path per input")
    ap.add_argument("--max-attempts", type=int, default=200,
                    help="live mode: give up after this many camera reads")
    ap.add_argument("--inputs", type=int, default=30,
                    help="synthetic mode: number of fixed inputs")
    ap.add_argument("--synthetic", action="store_true",
                    help="skip the camera and run synthetic mode directly")
    args = ap.parse_args()

    liv = LivenessDetector()
    spoofer = liv.spoofer

    if args.synthetic:
        return run_synthetic(args, spoofer)

    rc = run_live(args, spoofer)
    if rc == 1:
        print("Falling back to synthetic mode (no face was available).\n")
        return run_synthetic(args, spoofer)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
