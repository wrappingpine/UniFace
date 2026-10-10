"""Regression tests for the preview panel composition (Task A).

Guards against the historical crash:
    ValueError: all the input array dimensions except for the concatenation
    axis must match exactly
(caused by stacking a 240-px-tall raw panel with 224-px-tall crop panels),
and verifies:
- proportional, distortion-free fit to a common height
- divider/panel channel and height consistency
- None/failed panels become placeholders instead of raising
- grayscale/alpha inputs are normalized to 3 channels
"""

import numpy as np
import pytest

from uniface_unlock.preview import (
    DIVIDER_WIDTH,
    LABEL_BAR_HEIGHT,
    PANEL_HEIGHT,
    compose_preview,
    fit_panel,
    make_divider,
    placeholder_panel,
    set_message_canvas,
    to_bgr,
)


class TestFitPanel:
    def test_exact_height_and_preserved_aspect(self):
        # 640x480 raw frame -> 4:3 aspect preserved at height 240 -> width 320
        raw = np.zeros((480, 640, 3), np.uint8)
        panel = fit_panel(raw, 240)
        assert panel.shape[0] == 240
        assert panel.shape[1] == 320  # 640 * 240 / 480, no distortion

    def test_square_aligned_112_fits_to_240_square(self):
        al = np.zeros((112, 112, 3), np.uint8)
        panel = fit_panel(al, 240)
        assert panel.shape == (240, 240, 3)

    def test_odd_shapes_all_land_on_same_height(self):
        shapes = [(480, 640), (480, 641), (112, 112), (80, 80), (240, 321), (1, 7)]
        heights = set()
        widths = []
        for h, w in shapes:
            p = fit_panel(np.zeros((h, w, 3), np.uint8), PANEL_HEIGHT)
            assert p.shape[0] == PANEL_HEIGHT
            heights.add(p.shape[0])
            widths.append(p.shape[1])
        assert heights == {PANEL_HEIGHT}
        assert all(w >= 1 for w in widths)

    def test_grayscale_and_alpha_converted_to_3_channels(self):
        assert to_bgr(np.zeros((50, 50), np.uint8)).shape[2] == 3
        assert to_bgr(np.zeros((50, 50, 1), np.uint8)).shape[2] == 3
        assert to_bgr(np.zeros((50, 50, 4), np.uint8)).shape[2] == 3

    def test_upscale_uses_nearest_for_pixel_structure(self):
        # 112 -> 240 upscale: nearest keeps hard pixel blocks (no smoothing).
        al = np.zeros((112, 112, 3), np.uint8)
        al[0:56, :, :] = 255  # hard horizontal edge at mid-height
        panel = fit_panel(al, 240)
        # With INTER_NEAREST there must be no blended row values (only 0/255).
        uniq = np.unique(panel[:, 120])
        assert set(uniq.tolist()) <= {0, 255}


class TestDividersAndPlaceholders:
    def test_divider_matches_panel_height_and_channels(self):
        d = make_divider(PANEL_HEIGHT)
        assert d.shape == (PANEL_HEIGHT, DIVIDER_WIDTH, 3)

    def test_placeholder_panel_shape(self):
        p = placeholder_panel(PANEL_HEIGHT, 240)
        assert p.shape == (PANEL_HEIGHT, 240, 3)
        assert p.dtype == np.uint8


class TestComposePreview:
    def test_all_three_real_panels_stack_without_error(self):
        """The historical crash repro: 640x480 raw + 112x112 aligned + 80x80 fas."""
        raw = np.full((480, 640, 3), 7, np.uint8)
        al = np.full((112, 112, 3), 200, np.uint8)
        fs = np.full((80, 80, 3), 90, np.uint8)
        canvas = compose_preview(raw=raw, aligned=al, fas=fs, lines=["x"])
        assert canvas.ndim == 3 and canvas.shape[2] == 3
        assert canvas.shape[0] == PANEL_HEIGHT + LABEL_BAR_HEIGHT
        # expected width: 320 (raw fit) + 240 (aligned fit) + 240 (fas fit)
        #               + 2 dividers
        assert canvas.shape[1] == 320 + 240 + 240 + 2 * DIVIDER_WIDTH

    @pytest.mark.parametrize("kwargs", [
        dict(raw=None, aligned=None, fas=None),
        dict(raw=np.zeros((480, 640, 3), np.uint8), aligned=None, fas=None),
        dict(raw=None, aligned=np.zeros((112, 112, 3), np.uint8),
             fas=np.zeros((80, 80, 3), np.uint8)),
        dict(raw=np.zeros((480, 640, 3), np.uint8),
             aligned=np.zeros((112, 112, 3), np.uint8), fas=None),
    ])
    def test_missing_panels_become_placeholders_not_crashes(self, kwargs):
        canvas = compose_preview(**kwargs, lines=["quality: OK"])
        assert canvas.shape[0] == PANEL_HEIGHT + LABEL_BAR_HEIGHT
        assert canvas.shape[2] == 3

    def test_failed_detection_path_canvas_still_composes(self):
        """Detection failure keeps a raw frame but has no crops -> placeholders."""
        frame = np.zeros((480, 640, 3), np.uint8)
        canvas = compose_preview(raw=frame, aligned=None, fas=None,
                                 lines=["no face detected"])
        assert canvas.ndim == 3

    def test_empty_image_falls_back_to_placeholder(self):
        canvas = compose_preview(raw=np.zeros((0, 0, 3), np.uint8),
                                 aligned=None, fas=None)
        assert canvas.shape[0] == PANEL_HEIGHT + LABEL_BAR_HEIGHT

    def test_odd_width_panel_does_not_break_hstack(self):
        # A width that produces a rounding edge: 641-wide raw at height 240
        canvas = compose_preview(raw=np.zeros((180, 641, 3), np.uint8),
                                 aligned=np.zeros((112, 112, 3), np.uint8),
                                 fas=np.zeros((80, 80, 3), np.uint8))
        assert canvas.shape[0] == PANEL_HEIGHT + LABEL_BAR_HEIGHT

    def test_model_input_crops_not_mutated(self):
        al = np.full((112, 112, 3), 200, np.uint8)
        fs = np.full((80, 80, 3), 90, np.uint8)
        al_copy, fs_copy = al.copy(), fs.copy()
        compose_preview(raw=np.zeros((480, 640, 3), np.uint8), aligned=al, fas=fs)
        assert np.array_equal(al, al_copy)
        assert np.array_equal(fs, fs_copy)

    def test_message_canvas_when_no_frame(self):
        canvas = set_message_canvas("camera read failed")
        assert canvas.ndim == 3 and canvas.shape[2] == 3
        assert canvas.shape[0] == PANEL_HEIGHT + LABEL_BAR_HEIGHT

    def test_label_lines_color_coding(self):
        """Quality OK line is green, REJECT amber; both must render."""
        canvas_ok = compose_preview(lines=["quality: OK"])
        canvas_rej = compose_preview(lines=["quality: REJECT: blur"])
        assert canvas_ok.shape == canvas_rej.shape
        # at least some non-black pixels in the label bar for both
        bar_ok = canvas_ok[PANEL_HEIGHT:, :, :]
        bar_rej = canvas_rej[PANEL_HEIGHT:, :, :]
        assert bar_ok.max() > 0 and bar_rej.max() > 0


class TestQtBackendPick:
    def test_returns_a_known_backend_or_xcb(self):
        from uniface_unlock.preview import pick_qt_backend
        b = pick_qt_backend()
        assert isinstance(b, str) and b
        # never crashes on a headless box; falls back to xcb or offscreen
        assert b in {"xcb", "wayland", "minimal", "offscreen"}

    def test_does_not_force_wayland_without_plugin(self):
        """If no probed plugin dir has libqwayland.so, the picker must not return wayland."""
        from uniface_unlock.preview import pick_qt_backend
        import glob, os
        import cv2
        dirs = [
            os.path.join(cv2.__path__[0], "qt", "plugins", "platforms"),
            "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms",
            "/usr/lib/qt5/plugins/platforms",
        ]
        has_wayland = any(
            os.path.basename(p).startswith("libqwayland")
            for d in dirs for p in glob.glob(os.path.join(d, "libq*.so"))
        )
        b = pick_qt_backend()
        if not has_wayland:
            assert b != "wayland"
