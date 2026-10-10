"""Combined authentication module."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .camera import Camera, CameraError
from .config import get_config
from .detector import FaceDetector, DetectionError
from .enrollment import FaceEnrollment, EnrollmentError
from .liveness import LivenessDetector, LivenessError
from .quality import assess_face
from .recognizer import FaceRecognizer, RecognitionError

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Authentication-related errors."""
    
    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


class FaceAuthenticator:
    """Combined face authentication system.

    Authentication requires:
    - Exactly one face
    - Liveness passed
    - Recognition passed
    - Multiple consecutive successful frames
    """

    def __init__(self) -> None:
        config = get_config()
        self.required_frames = config.get("authentication.required_consecutive_frames", 3)
        self.frame_skip = config.get("authentication.frame_skip", 2)
        self.timeout_seconds = config.get("authentication.timeout_seconds", 30)

        # Use singleton pattern to avoid duplicate model initialization
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.liveness = LivenessDetector()
        self.enrollment = FaceEnrollment()

        logger.info(
            f"Authenticator initialized: "
            f"required_frames={self.required_frames}, "
            f"frame_skip={self.frame_skip}"
        )

        # Initialize state before any step() call
        self.state = self._initial_state()

    def authenticate(
        self,
        camera_device: str = "/dev/video0",
        resolution: tuple = (640, 480),
    ) -> bool:
        """Perform full authentication.

        Args:
            camera_device: Camera device path
            resolution: Camera resolution

        Returns:
            True if authenticated

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check enrollment
        if not self.enrollment.is_enrolled():
            raise AuthenticationError(
                "No face enrolled. Run 'uniface-unlock enroll' first.",
                code=3
            )

        # Load enrolled embedding
        enrolled_embedding = self.enrollment.load_enrollment()

        self.state = self._initial_state()
        start_time = time.time()

        logger.info("Starting authentication")
        logger.info("Looking for face...")

        try:
            with Camera(device=camera_device, resolution=resolution, fps=10) as cam:
                while True:
                    # Check timeout
                    if time.time() - start_time > self.timeout_seconds:
                        s = self.state
                        logger.info(
                            "auth run summary at timeout: "
                            f"frames_processed={s['frames_processed']}, "
                            f"quality_skips={s['quality_skips']}, "
                            f"success_count={s['success_count']}/{self.required_frames}, "
                            f"liveness_consistency={s['liveness_consistency']}/"
                            f"{self.liveness.consistency_required}"
                        )
                        raise AuthenticationError("Authentication timeout", code=5)

                    frame = cam.read()
                    if frame is None:
                        # A read failure is not a valid evidence frame.
                        # Fail closed: reset the consecutive-success counter.
                        self._reset_progress()
                        continue

                    self.state = self.step(frame, enrolled_embedding)

                    if self.state["authenticated"]:
                        logger.info("AUTH SUCCESS")
                        return True

        except CameraError as e:
            raise AuthenticationError(f"Camera failure: {e}", code=2)

    def _initial_state(self) -> dict:
        """Return a fresh authentication state."""
        return {
            "authenticated": False,
            "success_count": 0,
            "liveness_consistency": 0,
            "frames_processed": 0,
            "frames_skipped": 0,
            "quality_skips": 0,
        }

    def _reset_progress(self) -> None:
        """Reset the consecutive-success counters (fail closed)."""
        self.state["success_count"] = 0
        self.state["liveness_consistency"] = 0
        self.liveness.reset_smoothing()

    def step(self, frame: np.ndarray, enrolled_embedding: np.ndarray) -> dict:
        """Advance the state machine by one frame.

        This is the single source of truth for the authentication policy.
        The per-frame sequence (every frame that reaches this point) is:

          1. exactly one face (DetectionError -> reset, fail closed)
          1b. quality gate: bad frames are NEUTRAL skips (no progress, no reset)
          2. liveness: must return "REAL" on THIS frame
              - FAKE/UNKNOWN -> reset, fail closed
              - consistency_required consecutive REAL frames are required
                before recognition is even attempted
          3. recognition: must pass the 0.70 threshold on THIS frame
              - mismatch -> reset, fail closed
          4. all checks passed -> success_count += 1
              - success_count >= required_frames (3) -> authenticated = True

        Liveness and recognition are evaluated on the SAME frame. There is no
        cross-frame pairing: a liveness REAL frame followed by a recognition
        match on the next frame does NOT count as one success. The two are
        independent gates on one frame, and both must pass before success_count
        increments.

        Diagnostic logging: per frame we log the liveness decision
        (status + real/fake/mask probabilities), the similarity score, the
        quality verdict, and the resulting state transition. No images,
        embeddings, landmarks, or other biometric values are logged.
        """
        state = self.state
        state["frames_processed"] += 1

        try:
            # Step 1: exactly one face
            face = self.detector.validate_exactly_one_face(frame)
        except DetectionError:
            # no face or multiple faces -> reset, fail closed
            self._reset_progress()
            logger.info("auth frame: no/many faces -> reset")
            return state

        # Step 1b: quality gate. A low-quality frame (blur, bad exposure,
        # implausible landmarks, clipped crop) is a NEUTRAL skip: it does not
        # count toward authentication (so it cannot help an attacker) and it
        # does not reset accumulated progress (so a transient blur does not
        # force a timeout). Only full-quality REAL+match frames increment the
        # consecutive-success counter.
        try:
            verdict = assess_face(frame, face.bbox, face.landmarks)
        except Exception as e:  # gate must never become a new failure mode
            logger.debug(f"Quality assessment error (ignored): {e}")
            verdict = {"ok": True, "problems": [], "metrics": {}}
        if not verdict["ok"]:
            state["quality_skips"] += 1
            logger.info(f"Frame skipped (quality): {', '.join(verdict['problems'])}")
            return state

        try:
            # Step 2: liveness must pass
            status, _confidence, details = self.liveness.check_liveness(frame, face.bbox)
        except LivenessError:
            self._reset_progress()
            logger.info("auth frame: liveness error -> reset")
            return state

        if status != "REAL":
            # spoof / unknown -> reset, fail closed
            self._reset_progress()
            logger.info(
                f"auth frame: liveness={status} "
                f"(real={details.get('real_probability', 0):.3f}, "
                f"fake={details.get('fake_probability', 0):.3f}, "
                f"mask={details.get('mask_probability', 0):.3f}) -> reset"
            )
            return state

        # Liveness consistency: require N consecutive REAL liveness frames
        state["liveness_consistency"] += 1
        if state["liveness_consistency"] < self.liveness.consistency_required:
            state["success_count"] = 0
            logger.info(
                f"auth frame: liveness REAL ({state['liveness_consistency']}/"
                f"{self.liveness.consistency_required} consistency) "
                f"sim=deferred -> waiting on consistency"
            )
            return state

        try:
            # Step 3: recognition must pass
            is_match, similarity = self.recognizer.verify(
                frame, face.landmarks, enrolled_embedding
            )
        except RecognitionError:
            self._reset_progress()
            logger.info("auth frame: recognition error -> reset")
            return state

        if not is_match:
            # wrong identity -> reset, fail closed
            self._reset_progress()
            try:
                thr = float(getattr(self.recognizer, "threshold", 0.7))
            except (TypeError, ValueError):
                thr = 0.7
            logger.info(
                f"auth frame: liveness REAL, recognition MISMATCH "
                f"sim={similarity:.3f} < {thr:.2f} -> reset"
            )
            return state

        # All checks passed on this frame
        state["success_count"] += 1
        logger.info(
            f"auth frame: liveness REAL + recognition MATCH "
            f"sim={similarity:.3f} -> success_count {state['success_count']}/{self.required_frames}"
        )

        if state["success_count"] >= self.required_frames:
            state["authenticated"] = True

        return state

    def verify_single_frame(
        self,
        image: np.ndarray,
    ) -> dict:
        """Verify a single frame.

        Returns:
            Dict with detection, liveness, and recognition results
        """
        result = {
            "face_detected": False,
            "liveness_pass": False,
            "recognition_pass": False,
            "similarity": None,
            "liveness_confidence": None,
        }

        try:
            face = self.detector.validate_exactly_one_face(image)
            result["face_detected"] = True

            # Liveness
            status, confidence = self.liveness.check_liveness(image, face.bbox)
            result["liveness_confidence"] = confidence
            result["liveness_pass"] = (status == "REAL")

            if not result["liveness_pass"]:
                return result

            # Recognition
            if self.enrollment.is_enrolled():
                enrolled_embedding = self.enrollment.load_enrollment()
                is_match, similarity = self.recognizer.verify(
                    image, face.landmarks, enrolled_embedding
                )
                result["recognition_pass"] = is_match
                result["similarity"] = similarity

        except Exception as e:
            logger.debug(f"Verification failed: {e}")

        return result

    def get_info(self) -> dict:
        """Get authenticator information."""
        return {
            "enrolled": self.enrollment.is_enrolled(),
            "required_frames": self.required_frames,
            "frame_skip": self.frame_skip,
            "timeout_seconds": self.timeout_seconds,
        }
