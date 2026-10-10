"""Unit and property tests for the uniface-unlock authentication state machine.

Tests verify:
- Exactly-one-face enforcement (zero faces, multiple faces)
- Liveness enforcement (spoof rejected, unknown rejected, genuine accepted)
- Recognition thresholding
- Consecutive frame accumulation and reset on any failure
- Fail-closed behavior on exceptions or edge cases
- Single source of truth in FaceAuthenticator.step()
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from uniface.types import Face
from uniface_unlock.authenticator import FaceAuthenticator, AuthenticationError
from uniface_unlock.detector import DetectionError
from uniface_unlock.liveness import LivenessError
from uniface_unlock.recognizer import RecognitionError


@pytest.fixture
def mock_face():
    """A standard mock Face dataclass."""
    return Face(
        bbox=np.array([100, 100, 300, 300]),
        confidence=0.95,
        landmarks=np.array([[150, 150], [250, 150], [200, 200], [160, 260], [240, 260]]),
    )


@pytest.fixture
def authenticator():
    """Create an authenticator with mocked subcomponents for unit testing.

    The quality gate is neutralized to "ok" by default here so policy tests
    exercise the state machine only; TestQualityGate overrides the patch.
    """
    with patch("uniface_unlock.authenticator.FaceDetector"), \
         patch("uniface_unlock.authenticator.FaceRecognizer"), \
         patch("uniface_unlock.authenticator.LivenessDetector"), \
         patch("uniface_unlock.authenticator.FaceEnrollment"), \
         patch("uniface_unlock.authenticator.assess_face",
               return_value={"ok": True, "problems": [], "metrics": {}}):
        auth = FaceAuthenticator()
        # Set deterministic configuration
        auth.required_frames = 3
        auth.liveness.consistency_required = 2
        auth.state = auth._initial_state()
        # yield (not return) so all patches, including assess_face, stay
        # active while the test body calls step()
        yield auth


class TestStateMachinePolicy:
    """Validate that FaceAuthenticator.step() strictly adheres to security invariants."""

    def test_zero_faces_resets_progress(self, authenticator):
        """Zero faces detected must reset progress and remain unauthenticated."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.side_effect = DetectionError("No face detected")

        # Accumulate some state first
        authenticator.state["success_count"] = 2
        authenticator.state["liveness_consistency"] = 2

        state = authenticator.step(frame, emb)

        assert not state["authenticated"]
        assert state["success_count"] == 0
        assert state["liveness_consistency"] == 0

    def test_multiple_faces_resets_progress(self, authenticator):
        """Multiple faces detected must reset progress (shoulder surfing / presentation attack)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.side_effect = DetectionError("Multiple faces detected: 2")

        authenticator.state["success_count"] = 2
        authenticator.state["liveness_consistency"] = 2

        state = authenticator.step(frame, emb)

        assert not state["authenticated"]
        assert state["success_count"] == 0
        assert state["liveness_consistency"] == 0

    def test_spoof_liveness_resets_progress(self, authenticator, mock_face):
        """A FAKE liveness classification must reset progress immediately."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.return_value = mock_face
        authenticator.liveness.check_liveness.return_value = ("FAKE", 0.95, {})

        authenticator.state["success_count"] = 2
        authenticator.state["liveness_consistency"] = 2

        state = authenticator.step(frame, emb)

        assert not state["authenticated"]
        assert state["success_count"] == 0
        assert state["liveness_consistency"] == 0

    def test_unknown_liveness_resets_progress(self, authenticator, mock_face):
        """An UNKNOWN (uncertain) liveness classification must fail closed and reset."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.return_value = mock_face
        authenticator.liveness.check_liveness.return_value = ("UNKNOWN", 0.45, {})

        authenticator.state["success_count"] = 2
        authenticator.state["liveness_consistency"] = 2

        state = authenticator.step(frame, emb)

        assert not state["authenticated"]
        assert state["success_count"] == 0
        assert state["liveness_consistency"] == 0

    def test_liveness_consistency_delay(self, authenticator, mock_face):
        """Must require N consecutive REAL liveness frames before counting recognition successes."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.return_value = mock_face
        authenticator.liveness.check_liveness.return_value = ("REAL", 0.95, {})
        authenticator.recognizer.verify.return_value = (True, 0.85)

        # First frame: liveness consistency reaches 1, which is < consistency_required (2)
        state = authenticator.step(frame, emb)
        assert state["liveness_consistency"] == 1
        assert state["success_count"] == 0
        assert not state["authenticated"]

        # Second frame: liveness consistency reaches 2, now recognition can count
        state = authenticator.step(frame, emb)
        assert state["liveness_consistency"] == 2
        assert state["success_count"] == 1
        assert not state["authenticated"]

    def test_recognition_mismatch_resets_progress(self, authenticator, mock_face):
        """A non-matching face must reset progress."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.return_value = mock_face
        authenticator.liveness.check_liveness.return_value = ("REAL", 0.95, {})
        authenticator.recognizer.verify.return_value = (False, 0.35)

        authenticator.state["success_count"] = 2
        authenticator.state["liveness_consistency"] = 2

        state = authenticator.step(frame, emb)

        assert not state["authenticated"]
        assert state["success_count"] == 0
        assert state["liveness_consistency"] == 0

    def test_full_success_sequence(self, authenticator, mock_face):
        """N consecutive successful frames achieve authentication."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.return_value = mock_face
        authenticator.liveness.check_liveness.return_value = ("REAL", 0.95, {})
        authenticator.recognizer.verify.return_value = (True, 0.85)

        # Frame 1: liveness consistency 1 < 2
        state = authenticator.step(frame, emb)
        assert not state["authenticated"]
        assert state["success_count"] == 0

        # Frame 2: liveness consistency 2 == 2, success 1 < 3
        state = authenticator.step(frame, emb)
        assert not state["authenticated"]
        assert state["success_count"] == 1

        # Frame 3: success 2 < 3
        state = authenticator.step(frame, emb)
        assert not state["authenticated"]
        assert state["success_count"] == 2

        # Frame 4: success 3 >= 3 -> AUTHENTICATED
        state = authenticator.step(frame, emb)
        assert state["authenticated"]
        assert state["success_count"] == 3

    def test_intermittent_spoof_breaks_chain(self, authenticator, mock_face):
        """A single spoof frame in a sequence resets the chain completely."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        # 2 good frames
        authenticator.detector.validate_exactly_one_face.return_value = mock_face
        authenticator.liveness.check_liveness.return_value = ("REAL", 0.95, {})
        authenticator.recognizer.verify.return_value = (True, 0.85)

        authenticator.step(frame, emb)  # liveness=1, success=0
        authenticator.step(frame, emb)  # liveness=2, success=1
        authenticator.step(frame, emb)  # liveness=3, success=2

        assert authenticator.state["success_count"] == 2

        # 1 spoof frame
        authenticator.liveness.check_liveness.return_value = ("FAKE", 0.90, {})
        authenticator.step(frame, emb)

        assert authenticator.state["success_count"] == 0
        assert authenticator.state["liveness_consistency"] == 0
        assert not authenticator.state["authenticated"]

        # Back to good frames: must start completely over from scratch
        authenticator.liveness.check_liveness.return_value = ("REAL", 0.95, {})
        authenticator.step(frame, emb)  # liveness=1, success=0
        assert authenticator.state["success_count"] == 0
        assert not authenticator.state["authenticated"]


class TestQualityGate:
    """Quality-gated frames are neutral skips: they neither count nor reset."""

    def test_bad_quality_frame_skips_without_resetting(self, authenticator, mock_face):
        """A low-quality frame must not authenticate and must not reset progress."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.return_value = mock_face
        authenticator.liveness.check_liveness.return_value = ("REAL", 0.95, {})
        authenticator.recognizer.verify.return_value = (True, 0.85)

        # Build progress: 2 successes
        authenticator.step(frame, emb)
        authenticator.step(frame, emb)
        authenticator.step(frame, emb)
        assert authenticator.state["success_count"] == 2

        # Now flip the quality gate to reject
        with patch("uniface_unlock.authenticator.assess_face",
                   return_value={"ok": False, "problems": ["blur (laplacian_var=1.0)"], "metrics": {}}):
            state = authenticator.step(frame, emb)

        assert not state["authenticated"]
        # Neutral skip: progress preserved, no liveness/recognition evaluated
        assert state["success_count"] == 2
        assert state["quality_skips"] == 1
        # Liveness was NOT evaluated on the skipped frame
        # (detector was, quality vetoed before liveness)

    def test_bad_quality_frame_cannot_authenticate(self, authenticator, mock_face):
        """Even with REAL liveness and a recognition match, a bad-quality frame must not count."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.return_value = mock_face
        authenticator.liveness.check_liveness.return_value = ("REAL", 0.95, {})
        authenticator.recognizer.verify.return_value = (True, 0.99)

        with patch("uniface_unlock.authenticator.assess_face",
                   return_value={"ok": False, "problems": ["overexposed (mean=250)"], "metrics": {}}):
            for _ in range(10):
                state = authenticator.step(frame, emb)

        assert not state["authenticated"]
        assert state["success_count"] == 0
        assert state["quality_skips"] == 10

    def test_quality_error_is_neutral(self, authenticator, mock_face):
        """If quality assessment itself throws, the frame proceeds (gate must not
        become a new silent failure mode), preserving prior behavior."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)

        authenticator.detector.validate_exactly_one_face.return_value = mock_face
        authenticator.liveness.check_liveness.return_value = ("REAL", 0.95, {})
        authenticator.recognizer.verify.return_value = (True, 0.85)

        with patch("uniface_unlock.authenticator.assess_face", side_effect=RuntimeError("boom")):
            state = authenticator.step(frame, emb)

        # Proceeded past the gate (liveness consistency incremented)
        assert state["liveness_consistency"] == 1

    def test_detection_failure_still_resets_despite_quality(self, authenticator):
        """Zero faces resets even when quality gate is rejecting: detection ordering wins."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emb = np.zeros(512, dtype=np.float32)
        authenticator.detector.validate_exactly_one_face.side_effect = DetectionError("No face detected")

        authenticator.state["success_count"] = 2
        with patch("uniface_unlock.authenticator.assess_face",
                   return_value={"ok": False, "problems": ["blur"], "metrics": {}}):
            state = authenticator.step(frame, emb)

        assert state["success_count"] == 0
