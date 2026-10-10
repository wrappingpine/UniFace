"""Frame/face quality assessment.

Rejects bad samples (blur, bad exposure, implausible landmarks, invalid
alignment crops) instead of silently feeding them into enrollment or
recognition. All functions are pure (image in, verdict out) so they are
unit-testable without a camera.

Evidence motivating this module (2026-10-10): recognition similarity of the
SAME real face against the stored template moved from ~0.77 (historical auth
successes) to 0.63 -> 0.58 -> 0.36 over the day as camera capture conditions
drifted (V4L2 brightness at +64/max, exposure at max). Simulated photometric
transforms reproduce the reported 0.25 similarity plateau (brightness +64:
0.09, gain 1.4: 0.10, sigma-20 noise: 0.12, contrast 1.5: 0.49). MiniFASNet
real/mask probability also flips under the same transforms (real 0.95 ->
mask 0.53 at gain 2.0). Quality gating plus re-enrollment under stable
conditions is the in-policy remedy; thresholds and liveness are untouched.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import get_config

logger = logging.getLogger(__name__)

# Defaults; overridable via config["quality"]
QUALITY_DEFAULTS = {
    "enabled": True,
    "min_laplacian_variance": 12.0,   # blur: variance of Laplacian on the face crop
                                      # (live webcam background wall measures 9-18;
                                      # textured faces measure far higher; tune via
                                      # config quality.min_laplacian_variance)
    "min_mean_brightness": 35.0,      # exposure: mean gray level of face crop
    "max_mean_brightness": 225.0,
    "max_clipped_fraction": 0.15,     # exposure: fraction of pixels at 0 or 255
    "min_border_black_fraction": 0.0, # crop validity: allowed fraction of pure-black border
    "max_border_black_fraction": 0.35,
    "min_eye_dist_ratio": 0.15,       # eye distance / bbox width (wide yaw lowers this)
    "max_eye_dist_ratio": 0.75,
    "landmarks_margin_px": 2.0,       # landmarks must sit inside the frame by this margin
}


def _q(key: str):
    return get_config().get(f"quality.{key}", QUALITY_DEFAULTS[key])


def laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness proxy: variance of the Laplacian (higher = sharper)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def clipped_fraction(gray: np.ndarray) -> float:
    """Fraction of pixels saturated at 0 or 255."""
    return float(((gray <= 0) | (gray >= 255)).mean())


def border_black_fraction(bgr: np.ndarray, threshold: float = 1.0) -> float:
    """Fraction of the outer border of a (aligned) crop that is pure black.

    A similarity warp fills the 112x112 canvas for in-frame faces; a large
    black border means the face/landmarks sit near or outside the frame edge,
    i.e. a clipped alignment crop.
    """
    gray = bgr.mean(axis=2) if bgr.ndim == 3 else bgr
    edge = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return float((edge < threshold).mean())


def landmarks_plausible(
    landmarks: np.ndarray,
    bbox: np.ndarray,
    frame_shape: tuple[int, int],
) -> list[str]:
    """Geometry checks on 5-point landmarks (SCRFD order: Leye, Reye, Nose, Lmouth, Rmouth).

    Returns a list of failure strings (empty = plausible).
    """
    problems: list[str] = []
    lm = np.asarray(landmarks, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.float32)

    if lm.shape != (5, 2) or not np.isfinite(lm).all():
        return ["landmarks shape/finite"]

    h, w = frame_shape[:2]
    margin = _q("landmarks_margin_px")

    # All landmarks inside frame
    if (
        lm[:, 0].min() < margin or lm[:, 1].min() < margin
        or lm[:, 0].max() > w - 1 - margin or lm[:, 1].max() > h - 1 - margin
    ):
        problems.append("landmarks outside frame")

    le, re_, nose, lmouth, rmouth = lm

    # No mirroring: detector left eye must have smaller x than right eye
    if not le[0] < re_[0]:
        problems.append("eyes horizontally swapped")

    # Vertical ordering: eyes above nose above mouth
    eye_y = (le[1] + re_[1]) / 2
    mouth_y = (lmouth[1] + rmouth[1]) / 2
    if not eye_y < nose[1] < mouth_y:
        problems.append("eye/nose/mouth vertical order")

    # Eye distance plausible relative to bbox width
    bbox_w = float(bbox[2] - bbox[0])
    eye_dist = float(np.linalg.norm(re_ - le))
    if bbox_w > 0:
        ratio = eye_dist / bbox_w
        if not (_q("min_eye_dist_ratio") <= ratio <= _q("max_eye_dist_ratio")):
            problems.append(f"eye_dist_ratio {ratio:.2f} out of range")

    # Non-degenerate triangles (avoid collapsed landmarks)
    def tri_area(a, b, c):
        # 2D cross product magnitude (np.cross removed 2-D support in NumPy 2)
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0

    eye_dist2 = float(np.linalg.norm(re_ - le))
    if eye_dist2 <= 0:
        problems.append("zero eye distance")
    else:
        norm_area = tri_area(le, re_, nose) / (eye_dist2 ** 2)
        if norm_area < 0.02:
            problems.append("degenerate landmark triangle")

    return problems


def assess_face(
    frame: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray,
    aligned_crop: np.ndarray | None = None,
) -> dict:
    """Assess one detected face. Returns {"ok": bool, "problems": [...], "metrics": {...}}.

    `aligned_crop` is the 112x112 aligned face (if available) used for blur,
    exposure and crop-validity checks; falls back to the bbox crop otherwise.
    """
    if not _q("enabled"):
        return {"ok": True, "problems": [], "metrics": {}}

    problems: list[str] = []
    metrics: dict = {}

    problems += landmarks_plausible(landmarks, bbox, frame.shape)

    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(frame.shape[1], int(bbox[2])), min(frame.shape[0], int(bbox[3]))
    crop = aligned_crop if aligned_crop is not None else frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        problems.append("empty crop")
        return {"ok": False, "problems": problems, "metrics": metrics}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    lv = laplacian_variance(gray)
    mean_b = float(gray.mean())
    cf = clipped_fraction(gray)
    metrics.update(laplacian_var=round(lv, 1), mean_brightness=round(mean_b, 1), clipped=round(cf, 3))

    if lv < _q("min_laplacian_variance"):
        problems.append(f"blur (laplacian_var={lv:.1f})")
    if mean_b < _q("min_mean_brightness"):
        problems.append(f"underexposed (mean={mean_b:.0f})")
    if mean_b > _q("max_mean_brightness"):
        problems.append(f"overexposed (mean={mean_b:.0f})")
    if cf > _q("max_clipped_fraction"):
        problems.append(f"clipping (frac={cf:.2f})")

    if aligned_crop is not None:
        bb = border_black_fraction(aligned_crop)
        metrics["border_black"] = round(bb, 3)
        if bb > _q("max_border_black_fraction"):
            problems.append(f"clipped alignment crop (black border={bb:.2f})")

    return {"ok": len(problems) == 0, "problems": problems, "metrics": metrics}
