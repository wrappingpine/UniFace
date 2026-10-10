"""Preview canvas composition for `uniface-unlock preview` (local-only).

The preview shows three panels side by side:
  1. RAW      - the full camera frame (BGR, 640x480) with bbox + landmarks drawn
  2. ALIGNED  - the exact aligned 112x112 crop fed to MobileFace (BGR)
  3. FAS      - the exact 80x80 crop fed to MiniFASNet (BGR)

This module only performs PRESENTATION transforms (proportional resize to a
common display height, padding, dividers, labels). The model-input crops
themselves are never modified here - callers pass them through untouched.

Historic bug: the panels were resized to fixed, mismatched sizes (240 vs 224
px high), so np.hstack raised
    ValueError: all the input array dimensions except for the concatenation
    axis must match exactly
whenever a face was detected. All panels are now height-matched with an
aspect-preserving fit plus (rare) padding, and missing/None panels become
placeholder panels instead of crashing the loop.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

# Default display height every panel is fitted to before hstack.
PANEL_HEIGHT = 240
# Width/height of the vertical divider strips between panels.
DIVIDER_WIDTH = 4
# Extra pixel band below the panels reserved for the text labels.
LABEL_BAR_HEIGHT = 66
# Width used for placeholder panels when an image is unavailable.
PLACEHOLDER_WIDTH = 240
# Background color for placeholders/dividers (BGR).
NEUTRAL_GRAY = 128


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Return a 3-channel BGR view of *img*, converting grayscale if needed.

    Guarantees every panel and divider has identical channel counts, so
    hstack/vconcat can never fail on channel mismatch.
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"unsupported image shape: {img.shape}")
    return img


def fit_panel(img: np.ndarray, height: int = PANEL_HEIGHT) -> np.ndarray:
    """Proportionally resize *img* to exactly *height* pixels tall.

    Aspect ratio is preserved (no face distortion); width is rounded and
    clamped to >= 1. Upscales use INTER_NEAREST so model-input crops keep
    their pixel structure visually; downscales use INTER_AREA.
    """
    img = to_bgr(np.asarray(img))
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        return placeholder_panel(height)
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * height / h)))
    interp = cv2.INTER_NEAREST if height > h else cv2.INTER_AREA
    return cv2.resize(img, (new_w, height), interpolation=interp)


def placeholder_panel(height: int = PANEL_HEIGHT,
                      width: int = PLACEHOLDER_WIDTH,
                      text: str = "no image") -> np.ndarray:
    """Neutral gray panel with a short message; used when a frame/face is missing."""
    panel = np.full((height, width, 3), NEUTRAL_GRAY, np.uint8)
    cv2.putText(panel, text[:28], (6, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return panel


def make_divider(height: int, width: int = DIVIDER_WIDTH) -> np.ndarray:
    """Vertical divider strip matching the panel height and 3 channels."""
    return np.full((height, width, 3), NEUTRAL_GRAY, np.uint8)


def compose_preview(
    raw: Optional[np.ndarray] = None,
    aligned: Optional[np.ndarray] = None,
    fas: Optional[np.ndarray] = None,
    lines: Sequence[str] = (),
    panel_height: int = PANEL_HEIGHT,
) -> np.ndarray:
    """Build the side-by-side preview canvas.

    Any of raw/aligned/fas may be None (frame read failure, no face, model
    error) - that slot is rendered as a placeholder panel so the loop can
    keep running. Returns a BGR canvas of fixed height
    (panel_height + LABEL_BAR_HEIGHT) whose width is the sum of the three
    fitted panels plus two dividers.
    """
    panels = []
    for img, name in ((raw, "no frame"), (aligned, "no face"), (fas, "no face")):
        if img is None:
            panels.append(placeholder_panel(panel_height, text=name))
        else:
            try:
                panels.append(fit_panel(img, panel_height))
            except Exception:
                panels.append(placeholder_panel(panel_height, text=f"bad {name}"))

    div = make_divider(panel_height)
    top = np.hstack([panels[0], div, panels[1], div, panels[2]])

    canvas = np.zeros((panel_height + LABEL_BAR_HEIGHT, top.shape[1], 3), np.uint8)
    canvas[:panel_height, :, :] = top

    y = panel_height + 16
    for line in lines[:3]:
        color = (255, 255, 255)
        if line.startswith("quality:") and "REJECT" in line:
            color = (0, 128, 255)
        elif line.startswith("quality: OK"):
            color = (0, 255, 0)
        cv2.putText(canvas, line[:120], (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y += 18
    return canvas


def pick_qt_backend() -> str:
    """Choose the Qt platform plugin that is actually installed on this box.

    cv2's bundled Qt (used by imshow) only ships libqxcb.so in this venv, so on
    a Wayland desktop the default picks 'wayland', finds no plugin, and falls
    back to xcb with font warnings. We probe the plugin directory and only
    advertise a backend whose shared library exists, so the preview window
    opens cleanly on X11, Wayland, or headless Linux without hard-coding a
    platform that breaks other desktops.
    """
    import glob
    import os

    plugin_dirs = []
    try:
        import cv2
        plugin_dirs.append(os.path.join(cv2.__path__[0], "qt", "plugins", "platforms"))
    except Exception:
        pass
    plugin_dirs.append("/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
    plugin_dirs.append("/usr/lib/qt5/plugins/platforms")

    available = set()
    for d in plugin_dirs:
        for path in glob.glob(os.path.join(d, "libq*.so")):
            base = os.path.basename(path)
            if base.startswith("libqxcb"):
                available.add("xcb")
            elif base.startswith("libqwayland"):
                available.add("wayland")
            elif base.startswith("libqminimal"):
                available.add("minimal")
            elif base.startswith("libqoffscreen"):
                available.add("offscreen")

    # Prefer the session the user is actually in, but only if the plugin exists.
    for candidate in (
        os.environ.get("WAYLAND_DISPLAY") and "wayland",
        os.environ.get("DISPLAY") and "xcb",
    ):
        if candidate in available:
            return candidate
    if "offscreen" in available:
        return "offscreen"  # headless-safe, never crashes
    return "xcb"  # last resort; cv2 ships libqxcb.so in this venv


def set_message_canvas(message: str,
                       width: int = 640,
                       height: int = PANEL_HEIGHT + LABEL_BAR_HEIGHT) -> np.ndarray:
    """Full-width placeholder shown when no frame is available at all."""
    canvas = np.full((height, width, 3), 40, np.uint8)
    cv2.putText(canvas, message[:70], (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas
