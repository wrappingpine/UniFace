"""Face detection module using UniFace."""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from uniface.detection import SCRFD, RetinaFace, BlazeFace
from uniface.types import Face

from .config import get_config

logger = logging.getLogger(__name__)


class DetectionError(Exception):
    """Detection-related errors."""
    pass


class FaceDetector:
    """Face detection with exactly one face validation.

    Requirements:
    - Detect faces
    - Return bounding box
    - Return landmarks if available
    - Reject frames with zero faces
    - Reject frames with multiple faces
    - Configurable detection threshold
    - Measure inference time
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        config = get_config()
        model_name = config.get("detection.model", "SCRFD")
        self.threshold = config.get("detection.threshold", 0.7)
        self.enable_landmarks = config.get("detection.enable_landmarks", True)
        self.min_face_size = config.get("detection.min_face_size", 10000)  # min pixels

        # Detection input size (square). The default (640, 640) is what the
        # enrolled template was produced with. A smaller size is measurably
        # faster but changes the bbox/landmark geometry slightly; it is only
        # used when the user has explicitly opted in via config and the
        # recognition decision against the enrolled template has been verified
        # to be unaffected (see scripts/probe_recognition_vs_enrolled.py).
        self.input_size = self._parse_input_size(
            config.get("detection.input_size", 640)
        )

        # Initialize detector based on config
        if model_name.upper() == "SCRFD":
            self.detector = SCRFD(input_size=self.input_size)
        elif model_name.upper() == "RETINAFACE":
            self.detector = RetinaFace()
        elif model_name.upper() == "BLAZEFACE":
            self.detector = BlazeFace()
        else:
            logger.warning(f"Unknown detector {model_name}, using SCRFD")
            self.detector = SCRFD(input_size=self.input_size)

        logger.info(
            f"Face detector initialized: {model_name} input_size={self.input_size}"
        )
        self._initialized = True

    @staticmethod
    def _parse_input_size(value) -> Tuple[int, int]:
        """Normalise a detection input_size config value to (w, h).

        Accepts an int (square) or a 2-element list/tuple. Defaults to 640.
        Raises ValueError on malformed input so bad config is caught at
        startup rather than silently producing wrong geometry.
        """
        if value is None:
            return (640, 640)
        if isinstance(value, (int, float)):
            n = int(value)
            if n <= 0:
                raise ValueError(f"detection.input_size must be positive, got {value}")
            return (n, n)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            w, h = int(value[0]), int(value[1])
            if w <= 0 or h <= 0:
                raise ValueError(f"detection.input_size must be positive, got {value}")
            return (w, h)
        raise ValueError(f"detection.input_size must be int or [w, h], got {value!r}")

    def detect(
        self,
        image: np.ndarray,
        max_faces: int = 1,
    ) -> List[Face]:
        """Detect faces in image.

        Args:
            image: Input image (BGR numpy array)
            max_faces: Maximum number of faces to return

        Returns:
            List of detected Face objects

        Raises:
            DetectionError: If detection fails
        """
        start_time = time.time()

        try:
            # Detect faces
            faces = self.detector.detect(image)

            # Filter by confidence threshold
            faces = [f for f in faces if f.confidence >= self.threshold]

            # Filter by minimum face size (area in pixels)
            filtered_faces = []
            for f in faces:
                bbox = f.bbox
                width = float(bbox[2] - bbox[0])
                height = float(bbox[3] - bbox[1])
                area = width * height
                if area >= self.min_face_size:
                    filtered_faces.append(f)
            faces = filtered_faces

            # Limit number of faces
            if max_faces is not None and len(faces) > max_faces:
                # Sort by confidence and take top N
                faces = sorted(faces, key=lambda f: f.confidence, reverse=True)[:max_faces]

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"Detection completed: {len(faces)} faces, "
                f"latency={latency_ms:.1f}ms"
            )

            return faces

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise DetectionError(f"Face detection failed: {e}")

    def validate_exactly_one_face(
        self,
        image: np.ndarray,
    ) -> Face:
        """Validate exactly one face is present.

        Args:
            image: Input image

        Returns:
            Single Face object

        Raises:
            DetectionError: If zero or multiple faces detected
        """
        faces = self.detect(image, max_faces=2)  # Get up to 2 to check

        num_faces = len(faces)

        if num_faces == 0:
            raise DetectionError("No face detected")

        if num_faces > 1:
            raise DetectionError(f"Multiple faces detected: {num_faces}")

        logger.info("Face detected")
        logger.debug(f"Faces detected: {num_faces}")
        return faces[0]

    def measure_latency(
        self,
        image: np.ndarray,
        num_runs: int = 10,
    ) -> float:
        """Measure average detection latency.

        Args:
            image: Test image
            num_runs: Number of runs for averaging

        Returns:
            Average latency in milliseconds
        """
        latencies = []

        for _ in range(num_runs):
            start = time.time()
            _ = self.detect(image)
            latencies.append((time.time() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"Average detection latency: {avg_latency:.1f}ms over {num_runs} runs")
        return avg_latency

    def get_info(self) -> dict:
        """Get detector information."""
        return {
            "model": self.detector.__class__.__name__,
            "threshold": self.threshold,
            "input_size": self.input_size,
            "enable_landmarks": self.enable_landmarks,
            "supports_landmarks": getattr(self.detector, 'supports_landmarks', False),
            "supports_alignment": getattr(self.detector, 'supports_alignment', False),
        }