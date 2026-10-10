"""Face enrollment module."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from .camera import Camera, CameraError
from .config import get_config
from .detector import FaceDetector, DetectionError
from .recognizer import FaceRecognizer, RecognitionError

logger = logging.getLogger(__name__)


class EnrollmentError(Exception):
    """Enrollment-related errors."""
    pass


class FaceEnrollment:
    """Face enrollment with robust embedding averaging.

    Requirements:
    - Open webcam
    - Detect exactly one face
    - Guide user to look naturally
    - Collect multiple samples
    - Reject poor-quality samples
    - Generate normalized embeddings
    - Average embeddings robustly
    - Store securely
    """

    def __init__(self) -> None:
        config = get_config()
        self.config = config
        self.enrollment_dir = Path.home() / ".config" / "uniface-unlock"
        self.enrollment_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_path = self.enrollment_dir / config.get("data.embedding_file")
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()

        # Set secure permissions
        self.enrollment_dir.chmod(0o700)

    def enroll(
        self,
        num_samples: int = 10,
        camera_device: str = "/dev/video0",
        resolution: tuple = (640, 480),
    ) -> Path:
        """Enroll a new face.

        Args:
            num_samples: Number of samples to collect
            camera_device: Camera device path
            resolution: Camera resolution

        Returns:
            Path to saved enrollment file

        Raises:
            EnrollmentError: If enrollment fails
        """
        logger.info("Starting face enrollment")
        logger.info(f"Collecting {num_samples} samples")

        embeddings = []
        collected = 0

        try:
            with Camera(device=camera_device, resolution=resolution, fps=10) as cam:
                logger.info("Camera opened, looking for face...")

                # Collect samples (with quality gating; keep scanning
                # up to 3x the requested frames so rejections do not starve
                # the sample budget)
                for i in tqdm(range(num_samples * 3), desc="Collecting samples"):
                    if collected >= num_samples:
                        break
                    frame = cam.read()
                    if frame is None:
                        raise EnrollmentError("Failed to read frame")

                    # Detect face
                    try:
                        face = self.detector.validate_exactly_one_face(frame)
                    except DetectionError as e:
                        logger.warning(f"Sample {i+1}: {e}")
                        continue

                    # Quality gate: reject bad samples instead of silently
                    # averaging them into the template (blur, bad exposure,
                    # implausible landmarks, clipped alignment crop).
                    from .quality import assess_face
                    verdict = assess_face(frame, face.bbox, face.landmarks)
                    if not verdict["ok"]:
                        logger.warning(
                            f"Sample {i+1} rejected (quality): {', '.join(verdict['problems'])}"
                        )
                        continue

                    # Extract embedding
                    try:
                        embedding = self.recognizer.extract_embedding(
                            frame,
                            face.landmarks
                        )
                        embeddings.append(embedding)
                        logger.info(f"Sample {i+1}: embedding collected")
                    except RecognitionError as e:
                        logger.warning(f"Sample {i+1}: {e}")
                        continue

                    collected += 1

                    # Brief pause between samples
                    import time
                    time.sleep(0.5)

        except CameraError as e:
            raise EnrollmentError(f"Camera error: {e}")

        if len(embeddings) < 3:
            raise EnrollmentError(
                f"Insufficient good-quality samples collected: {len(embeddings)} < 3. "
                "Improve lighting/positioning and try again."
            )

        # Average embeddings robustly
        logger.info(f"Averaging {len(embeddings)} embeddings")
        embeddings_array = np.stack(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)

        # Normalize
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        # Save enrollment
        self.save_enrollment(mean_embedding)
        self.save_enrollment_meta()

        logger.info(f"Enrollment completed: {self.embedding_path}")
        return self.embedding_path

    def save_enrollment(self, embedding: np.ndarray) -> None:
        """Save enrollment embedding securely.

        Args:
            embedding: Face embedding vector

        Raises:
            EnrollmentError: If the embedding is malformed (never written)
        """
        # Validate before touching disk so a bad template is never persisted.
        self._validate_embedding(embedding)

        # Save as numpy file
        embedding = np.asarray(embedding, dtype=np.float32)
        np.save(self.embedding_path.with_suffix('.npy'), embedding)

        # Set secure permissions
        self.embedding_path.with_suffix('.npy').chmod(0o600)

        logger.info(f"Enrollment saved to {self.embedding_path.with_suffix('.npy')}")

    def save_enrollment_meta(self) -> None:
        """Write a non-biometric metadata sidecar next to the template.

        Records model identity and template properties (not values) so that
        enrollment vs recognition pipeline drift can be detected later.
        """
        import hashlib
        import time

        emb_file = self.embedding_path.with_suffix('.npy')
        embedding = np.load(emb_file)
        recognizer = self.recognizer.recognizer
        detector = self.detector.detector

        meta = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recognition_class": recognizer.__class__.__name__,
            "recognition_model_path": str(recognizer.model_path),
            "recognition_model_sha256": self._sha256(recognizer.model_path),
            "recognition_input_size": list(recognizer.input_size),
            "recognition_input_mean": recognizer.input_mean,
            "recognition_input_std": recognizer.input_std,
            "detection_class": detector.__class__.__name__,
            "detection_model_sha256": self._sha256(getattr(detector, 'model_path', '')),
            "embedding_shape": list(embedding.shape),
            "embedding_dtype": str(embedding.dtype),
            "embedding_l2_norm": float(np.linalg.norm(embedding)),
            "embedding_finite": bool(np.isfinite(embedding).all()),
            "embedding_sha256": hashlib.sha256(
                np.ascontiguousarray(embedding).tobytes()
            ).hexdigest(),
        }

        meta_file = self.embedding_path.with_suffix('.meta.json')
        meta_file.write_text(json.dumps(meta, indent=2))
        meta_file.chmod(0o600)
        logger.info(f"Enrollment metadata written: {meta_file}")

    @staticmethod
    def _sha256(path) -> str:
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except Exception:
            return "unknown"

    def load_meta(self) -> dict | None:
        """Load enrollment metadata sidecar if present."""
        meta_file = self.embedding_path.with_suffix('.meta.json')
        if not meta_file.exists():
            return None
        try:
            return json.loads(meta_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to read enrollment metadata: {e}")
            return None

    def load_enrollment(self) -> np.ndarray:
        """Load and validate the enrolled face embedding.

        Validation is fail-closed and read-only: the stored file is never
        rewritten or normalised here. A template that is not a finite,
        non-degenerate float32 vector of the expected dimension is rejected
        rather than passed into cosine similarity, where NaN/inf or a wrong
        shape could silently produce a match or an exception mid-authentication.

        Returns:
            Face embedding vector (float32, shape (embedding_dim,))

        Raises:
            EnrollmentError: If enrollment is missing or malformed
        """
        enrollment_file = self.embedding_path.with_suffix('.npy')

        if not enrollment_file.exists():
            raise EnrollmentError(
                f"No enrollment found at {enrollment_file}. Run enroll first."
            )

        try:
            # allow_pickle stays at its default (False): a crafted .npy that
            # encodes a Python object must not be deserialised.
            embedding = np.load(enrollment_file)
        except Exception as e:
            raise EnrollmentError(f"Failed to load enrollment: {e}")

        self._validate_embedding(embedding)
        logger.info(f"Enrollment loaded: {enrollment_file}")
        return embedding

    def _validate_embedding(self, embedding: np.ndarray) -> None:
        """Reject malformed enrollment templates (fail closed)."""
        cfg = getattr(self, "config", None) or get_config()
        expected_dim = int(cfg.get("recognition.embedding_dim", 512))

        if not isinstance(embedding, np.ndarray):
            raise EnrollmentError(
                f"Enrollment must be a numpy array, got {type(embedding).__name__}"
            )
        if embedding.ndim != 1:
            raise EnrollmentError(
                f"Enrollment must be 1-dimensional, got shape {embedding.shape}"
            )
        if embedding.shape[0] != expected_dim:
            raise EnrollmentError(
                f"Enrollment has dim {embedding.shape[0]}, expected {expected_dim}"
            )
        if not np.issubdtype(embedding.dtype, np.floating):
            raise EnrollmentError(
                f"Enrollment dtype must be floating point, got {embedding.dtype}"
            )
        if not np.isfinite(embedding).all():
            raise EnrollmentError("Enrollment contains NaN or infinite values")

        norm = float(np.linalg.norm(embedding.astype(np.float64)))
        if not np.isfinite(norm) or norm < 1e-6:
            raise EnrollmentError("Enrollment has a degenerate (near-zero) norm")

    def is_enrolled(self) -> bool:
        """Check if a face is enrolled."""
        return self.embedding_path.with_suffix('.npy').exists()

    def delete_enrollment(self) -> None:
        """Delete enrolled face data."""
        enrollment_file = self.embedding_path.with_suffix('.npy')
        if enrollment_file.exists():
            enrollment_file.unlink()
            logger.info("Enrollment deleted")

    def get_info(self) -> dict:
        """Get enrollment information."""
        enrollment_file = self.embedding_path.with_suffix('.npy')
        return {
            "enrolled": enrollment_file.exists(),
            "path": str(enrollment_file),
            "directory": str(self.enrollment_dir),
        }
