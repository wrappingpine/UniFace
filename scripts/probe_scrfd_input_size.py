"""Evidence probe: SCRFD input_size trade-off on the real webcam.

Runs one detector configuration at a time (only one ONNX session alive at any
moment, so peak RSS stays low on this 5.7 GiB laptop). Frames are captured
once and kept in memory only -- nothing is written to disk.

For each input size it reports:
  * cold (first) inference latency
  * warm latency: n, mean, median, p90, max
  * per-frame detection agreement with the 640x640 reference:
      - faces found
      - confidence delta
      - bbox IoU
      - 5-point landmark error, normalised by inter-ocular distance
  * recognition impact: cosine similarity between the embedding computed from
    this size's landmarks and the one computed from the reference landmarks
  * process RSS delta around the model load

Usage: python scripts/probe_scrfd_input_size.py [--sizes 640,480,320] [--frames 8]
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import numpy as np


def rss_mib() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def interocular(lms) -> float:
    return float(np.linalg.norm(np.asarray(lms[0], np.float32) - np.asarray(lms[1], np.float32)))


def capture_frames(n: int, device: str):
    import cv2

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frames = []
    attempts = 0
    while len(frames) < n and attempts < n * 5:
        attempts += 1
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame.copy())
        time.sleep(0.05)
    cap.release()
    if not frames:
        raise SystemExit("no frames captured")
    return frames


def run_size(size: int, frames, reference: list | None):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from uniface.detection import SCRFD
    from uniface.recognition import MobileFace

    r0 = rss_mib()
    t_load = time.perf_counter()
    det = SCRFD(input_size=(size, size))
    load_ms = (time.perf_counter() - t_load) * 1000
    r_loaded = rss_mib()

    latencies: list[float] = []
    results = []
    recognizer = MobileFace(providers=["CPUExecutionProvider"])
    r_models = rss_mib()

    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        faces = det.detect(frame)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        entry = {"n": len(faces), "ms": ms}
        if faces:
            face = max(faces, key=lambda f: float(f.confidence))
            entry["conf"] = float(face.confidence)
            entry["bbox"] = np.asarray(face.bbox, np.float32)
            entry["landmarks"] = np.asarray(face.landmarks, np.float32)
            entry["area"] = float((entry["bbox"][2] - entry["bbox"][0])
                                  * (entry["bbox"][3] - entry["bbox"][1]))
            # recognition impact on this detector's own landmarks
            try:
                emb = recognizer.get_normalized_embedding(frame, entry["landmarks"])
                entry["emb"] = emb
            except Exception as exc:  # pragma: no cover - diagnostic only
                entry["emb"] = None
                entry["emb_error"] = str(exc)
        results.append(entry)

        if idx == 0:
            cold_ms = ms
        else:
            warm_ms = ms

    warm = latencies[1:] or latencies
    summary = {
        "size": size,
        "load_ms": load_ms,
        "rss_after_detector": r_loaded,
        "rss_after_models": r_models,
        "rss_delta": r_models - r0,
        "cold_ms": latencies[0],
        "n_warm": len(warm),
        "mean": statistics.fmean(warm),
        "median": statistics.median(warm),
        "p90": sorted(warm)[max(0, int(len(warm) * 0.9) - 1)],
        "max": max(warm),
        "min": min(warm),
        "frames_with_face": sum(1 for r in results if r["n"] >= 1),
        "results": results,
    }

    if reference is not None:
        ious, lm_errs, sims, conf_deltas, misses = [], [], [], [], 0
        for mine, ref in zip(results, reference):
            if ref.get("n", 0) == 0:
                continue
            if mine.get("n", 0) == 0:
                misses += 1
                continue
            ious.append(iou(mine["bbox"], ref["bbox"]))
            base = interocular(ref["landmarks"])
            if base > 1e-6:
                # landmark 0/1 order is fixed by the model; compare pointwise
                lm_errs.append(float(np.mean(np.linalg.norm(
                    mine["landmarks"] - ref["landmarks"], axis=1))) / base)
            conf_deltas.append(mine["conf"] - ref["conf"])
            if mine.get("emb") is not None and ref.get("emb") is not None:
                sims.append(float(np.dot(mine["emb"], ref["emb"])))
        summary["reference_misses"] = misses
        summary["mean_iou"] = statistics.fmean(ious) if ious else None
        summary["min_iou"] = min(ious) if ious else None
        summary["mean_lm_err_iod"] = statistics.fmean(lm_errs) if lm_errs else None
        summary["max_lm_err_iod"] = max(lm_errs) if lm_errs else None
        summary["mean_emb_sim_vs_ref"] = statistics.fmean(sims) if sims else None
        summary["min_emb_sim_vs_ref"] = min(sims) if sims else None
        summary["mean_conf_delta"] = statistics.fmean(conf_deltas) if conf_deltas else None

    del det, recognizer
    gc.collect()
    return summary, results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="640,480,320")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--device", default="/dev/video0")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    frames = capture_frames(args.frames, args.device)
    print(f"captured {len(frames)} frames (in-memory only) @ {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"baseline RSS before any model: {rss_mib():.1f} MiB\n")

    reference = None
    rows = []
    for size in sizes:
        summary, results = run_size(size, frames, reference)
        if reference is None:
            reference = results
        rows.append(summary)

    hdr = (f"{'size':>5} {'load_ms':>8} {'cold_ms':>8} {'mean':>7} {'med':>7} "
           f"{'p90':>7} {'max':>7} {'min':>7} {'n':>3} {'RSS+MiB':>8} {'faces':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['size']:>5} {r['load_ms']:>8.0f} {r['cold_ms']:>8.0f} "
              f"{r['mean']:>7.1f} {r['median']:>7.1f} {r['p90']:>7.1f} {r['max']:>7.1f} "
              f"{r['min']:>7.1f} {r['n_warm']:>3} {r['rss_delta']:>8.1f} "
              f"{r['frames_with_face']:>3}/{len(frames)}")

    print("\nagreement vs 640x640 reference (same frames, same weights):")
    print(f"{'size':>5} {'miss':>5} {'IoU mean':>9} {'IoU min':>8} "
          f"{'lm err/IOD':>11} {'max lm':>7} {'emb sim mean':>13} {'emb min':>8} {'dconf':>7}")
    for r in rows[1:]:
        def f(v, fmt="{:>" + "8.3f}"):
            return (fmt.format(v) if v is not None else f"{'n/a':>8}")
        print(f"{r['size']:>5} {r.get('reference_misses', 0):>5} "
              f"{f(r.get('mean_iou'))} {f(r.get('min_iou'))} "
              f"{f(r.get('mean_lm_err_iod'), '{:>11.4f}')} {f(r.get('max_lm_err_iod'))} "
              f"{f(r.get('mean_emb_sim_vs_ref'), '{:>13.4f}')} {f(r.get('min_emb_sim_vs_ref'))} "
              f"{f(r.get('mean_conf_delta'), '{:>7.3f}')}")

    print(f"\nRSS after teardown: {rss_mib():.1f} MiB")


if __name__ == "__main__":
    main()
