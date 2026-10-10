"""Unit tests for the frame/face quality gate (quality.py).

Pure-function tests: synthetic images only, no camera.
"""

import numpy as np
import pytest

from uniface_unlock.quality import (
    assess_face,
    border_black_fraction,
    clipped_fraction,
    laplacian_variance,
    landmarks_plausible,
)

# 640x480 frame, plausible 5-point landmarks (SCRFD order). Eye distance
# (~0.5 of bbox width) matches the probed real-camera geometry (135px bbox,
# 65px eye distance -> 0.49).
GOOD_LM = np.array(
    [[210, 200], [270, 198], [240, 240], [218, 278], [262, 276]],
    dtype=np.float32,
)
GOOD_BBOX = np.array([180, 170, 300, 310], dtype=np.float32)


def _frame(value=128, noise=0.0, shape=(480, 640, 3)):
    rng = np.random.default_rng(42)
    img = np.full(shape, value, dtype=np.uint8)
    if noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def _textured_face_frame():
    """Frame with a textured face region so blur/exposure checks are meaningful."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    return img


class TestBlur:
    def test_sharp_texture_high_variance(self):
        assert laplacian_variance(_textured_face_frame()[:, :, 0]) > 100

    def test_flat_image_low_variance(self):
        assert laplacian_variance(_frame(128)[:, :, 0]) < 1e-6


class TestExposure:
    def test_clipped_fraction_dark(self):
        assert clipped_fraction(np.zeros((50, 50), np.uint8)) == 1.0

    def test_clipped_fraction_mid(self):
        assert clipped_fraction(np.full((50, 50), 128, np.uint8)) == 0.0

    def test_underexposed_frame_flagged(self):
        verdict = assess_face(_frame(5), GOOD_BBOX, GOOD_LM)
        assert not verdict["ok"]
        assert any("underexposed" in p for p in verdict["problems"])

    def test_overexposed_frame_flagged(self):
        verdict = assess_face(_frame(250), GOOD_BBOX, GOOD_LM)
        assert not verdict["ok"]
        assert any("overexposed" in p for p in verdict["problems"])


class TestLandmarks:
    def test_good_landmarks_pass(self):
        assert landmarks_plausible(GOOD_LM, GOOD_BBOX, (480, 640)) == []

    def test_swapped_eyes_detected(self):
        swapped = GOOD_LM.copy()
        swapped[[0, 1]] = swapped[[1, 0]]
        problems = landmarks_plausible(swapped, GOOD_BBOX, (480, 640))
        assert any("eyes" in p for p in problems)

    def test_outside_frame_detected(self):
        shifted = GOOD_LM + 600
        problems = landmarks_plausible(shifted, GOOD_BBOX, (480, 640))
        assert any("outside frame" in p for p in problems)

    def test_wrong_vertical_order_detected(self):
        bad = GOOD_LM.copy()
        bad[2] = [240, 100]  # nose above eyes
        problems = landmarks_plausible(bad, GOOD_BBOX, (480, 640))
        assert any("vertical order" in p for p in problems)

    def test_degenerate_landmarks_detected(self):
        collapsed = np.tile(GOOD_LM[0], (5, 1)).astype(np.float32)
        problems = landmarks_plausible(collapsed, GOOD_BBOX, (480, 640))
        assert len(problems) > 0

    def test_nan_landmarks_detected(self):
        nan_lm = GOOD_LM.copy()
        nan_lm[2, 1] = np.nan
        problems = landmarks_plausible(nan_lm, GOOD_BBOX, (480, 640))
        assert any("shape/finite" in p for p in problems)

    def test_nan_landmarks_rejected_by_assess(self):
        nan_lm = GOOD_LM.copy()
        nan_lm[2, 1] = np.nan
        verdict = assess_face(_textured_face_frame(), GOOD_BBOX, nan_lm)
        assert not verdict["ok"]


class TestCropValidity:
    def test_black_border_fraction_all_black(self):
        img = np.zeros((112, 112, 3), np.uint8)
        assert border_black_fraction(img) == 1.0

    def test_black_border_fraction_clean(self):
        img = np.full((112, 112, 3), 128, np.uint8)
        assert border_black_fraction(img) == 0.0

    def test_half_clipped_crop_flagged(self):
        crop = np.full((112, 112, 3), 128, np.uint8)
        crop[:, 56:] = 0  # right half pure black => clipped warp
        verdict = assess_face(_textured_face_frame(), GOOD_BBOX, GOOD_LM, aligned_crop=crop)
        assert not verdict["ok"]
        assert any("clipped alignment crop" in p for p in verdict["problems"])


class TestAssessFace:
    def test_good_frame_passes(self):
        verdict = assess_face(_textured_face_frame(), GOOD_BBOX, GOOD_LM)
        assert verdict["ok"], verdict["problems"]

    def test_blur_detected(self):
        import cv2
        blurred = cv2.GaussianBlur(_textured_face_frame(), (0, 0), 15)
        verdict = assess_face(blurred, GOOD_BBOX, GOOD_LM)
        assert not verdict["ok"]
        assert any("blur" in p for p in verdict["problems"])

    def test_disabled_gate_always_ok(self):
        from uniface_unlock import config as config_mod
        cfg = config_mod.get_config()
        old = cfg.get("quality.enabled")
        cfg.set("quality.enabled", False)
        try:
            verdict = assess_face(_frame(0), GOOD_BBOX, GOOD_LM + 600)
            assert verdict["ok"]
        finally:
            cfg.set("quality.enabled", old if old is not None else True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
