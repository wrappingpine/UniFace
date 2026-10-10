"""Camera abstraction for V4L2/Linux support."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Camera-related errors."""
    pass


class Camera:
    """V4L2 camera abstraction with configurable resolution and FPS.

    Requirements:
    - V4L2/Linux support
    - Configurable /dev/videoX
    - Configurable resolution
    - Configurable FPS
    - Clean startup/shutdown
    - Camera failure handling
    - No resource leaks
    - Graceful recovery if camera disappears
    """

    DEFAULT_DEVICE = "/dev/video0"
    DEFAULT_RESOLUTION = (640, 480)
    DEFAULT_FPS = 10

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
        fps: int = DEFAULT_FPS,
    ) -> None:
        self.device = device
        self.resolution = resolution
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        """Open camera device."""
        if self._is_open:
            logger.warning("Camera already open")
            return

        if not self._device_exists():
            raise CameraError(f"Camera device {self.device} does not exist")

        # Use V4L2 backend explicitly
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        if not self._cap or not self._cap.isOpened():
            raise CameraError(f"Failed to open camera device {self.device}")

        # Set properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Verify settings
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Camera opened: {self.device} "
            f"@ {actual_width}x{actual_height} {actual_fps:.1f} FPS"
        )

        self._is_open = True

        # Warm up camera
        self.read()

    def close(self) -> None:
        """Close camera and release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info("Camera closed")

    def read(self) -> Optional[np.ndarray]:
        """Read a single frame from camera.

        Returns:
            Frame as numpy array (BGR) or None if read failed
        """
        if not self._is_open or self._cap is None:
            raise CameraError("Camera not open")

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                return None
            return frame
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None

    def is_opened(self) -> bool:
        """Check if camera is open."""
        return self._is_open and self._cap is not None and self._cap.isOpened()

    def _device_exists(self) -> bool:
        """Check if camera device exists."""
        return Path(self.device).exists()

    @contextmanager
    def grab_frames(self, max_frames: int = 1):
        """Context manager for grabbing multiple frames."""
        try:
            yield self
        finally:
            pass

    def get_camera_info(self) -> dict:
        """Get camera information."""
        info = {
            "device": self.device,
            "resolution": self.resolution,
            "fps": self.fps,
            "exists": self._device_exists(),
            "is_open": self.is_opened(),
        }

        if self._cap and self._cap.isOpened():
            info.update({
                "actual_width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "actual_height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "actual_fps": self._cap.get(cv2.CAP_PROP_FPS),
                "backend": int(self._cap.get(cv2.CAP_PROP_BACKEND)),
            })

        return info


def list_cameras(max_devices: int = 10) -> list[str]:
    """List available camera devices.

    Returns:
        List of /dev/videoX paths
    """
    cameras = []
    for i in range(max_devices):
        device = f"/dev/video{i}"
        if Path(device).exists():
            cameras.append(device)
    return cameras


def check_camera(device: str = "/dev/video0") -> bool:
    """Check if camera is available and functional.

    Args:
        device: Camera device path

    Returns:
        True if camera is accessible
    """
    try:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            return False

        ret, _ = cap.read()
        cap.release()
        return ret
    except Exception:
        return False
