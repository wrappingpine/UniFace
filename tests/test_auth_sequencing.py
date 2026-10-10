"""Authentication sequencing tests (Task B).

Documents and guards the real per-frame policy in FaceAuthenticator.step():
  detection -> quality gate -> liveness -> recognition -> consecutive counter

Confirms:
- liveness and recognition run on the SAME frame (no cross-frame pairing)
- recognition is deferred until liveness consistency is satisfied
- success_count only increments when the full gate passes on one frame
- FAKE / UNKNOWN liveness, mismatch, and errors reset the counter (fail closed)
- quality-gate skips are neutral (no increment, no reset)
- timeout raises and the camera context is closed
"""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from uniface.types import Face
from uniface_unlock.authenticator import FaceAuthenticator, AuthenticationError
from uniface_unlock.detector import DetectionError


def _details(real=0.9, fake=0.05, mask=0.05):
    return {"real_probability": real, "fake_probability": fake,
            "mask_probability": mask}


@pytest.fixture
def auth():
    with patch("uniface_unlock.authenticator.FaceDetector"), \
         patch("uniface_unlock.authenticator.FaceRecognizer"), \
         patch("uniface_unlock.authenticator.LivenessDetector"), \
         patch("uniface_unlock.authenticator.FaceEnrollment"), \
         patch("uniface_unlock.authenticator.assess_face",
               return_value={"ok": True, "problems": [], "metrics": {}}):
        a = FaceAuthenticator()
        a.required_frames = 3
        a.liveness.consistency_required = 2
        a.recognizer.threshold = 0.7
        a.state = a._initial_state()
        yield a


@pytest.fixture
def face():
    return Face(
        bbox=np.array([100, 100, 300, 300]),
        confidence=0.95,
        landmarks=np.array([[150, 150], [250, 150], [200, 200],
                            [160, 260], [240, 260]]),
    )


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
EMB = np.zeros(512, dtype=np.float32)


class TestSameFrameSequencing:
    def test_recognition_deferred_until_liveness_consistency(self, auth, face):
        """Recognition must NOT run until liveness consistency is met."""
        auth.detector.validate_exactly_one_face.return_value = face
        auth.liveness.check_liveness.return_value = ("REAL", 0.95, _details())
        auth.recognizer.verify.return_value = (True, 0.85)

        # Frame 1: consistency 1/2 -> recognition must not be called yet
        state = auth.step(FRAME, EMB)
        assert state["liveness_consistency"] == 1
        auth.recognizer.verify.assert_not_called()
        assert state["success_count"] == 0

        # Frame 2: consistency 2/2 -> recognition now runs
        state = auth.step(FRAME, EMB)
        auth.recognizer.verify.assert_called()
        assert state["success_count"] == 1

    def test_liveness_and_recognition_called_with_same_frame(self, auth, face):
        """Both gates receive the identical frame object in one step()."""
        auth.detector.validate_exactly_one_face.return_value = face
        auth.liveness.check_liveness.return_value = ("REAL", 0.95, _details())
        auth.recognizer.verify.return_value = (True, 0.85)
        auth.state["liveness_consistency"] = 2  # pre-satisfy consistency

        auth.step(FRAME, EMB)

        live_frame = auth.liveness.check_liveness.call_args[0][0]
        rec_frame = auth.recognizer.verify.call_args[0][0]
        assert live_frame is rec_frame is FRAME

    def test_recognition_match_but_liveness_fake_never_counts(self, auth, face):
        """A frame that would match recognition but fails liveness must not count."""
        auth.detector.validate_exactly_one_face.return_value = face
        auth.state["liveness_consistency"] = 2
        auth.state["success_count"] = 2
        # Liveness FAKE short-circuits before recognition is ever consulted.
        auth.liveness.check_liveness.return_value = ("FAKE", 0.9, _details(real=0.02, fake=0.96))
        auth.recognizer.verify.return_value = (True, 0.99)

        state = auth.step(FRAME, EMB)
        assert state["success_count"] == 0
        auth.recognizer.verify.assert_not_called()


class TestConsecutiveCounter:
    def test_full_gate_three_frames_authenticates(self, auth, face):
        auth.detector.validate_exactly_one_face.return_value = face
        auth.liveness.check_liveness.return_value = ("REAL", 0.95, _details())
        auth.recognizer.verify.return_value = (True, 0.85)

        auth.step(FRAME, EMB)   # consistency 1
        auth.step(FRAME, EMB)   # success 1
        auth.step(FRAME, EMB)   # success 2
        assert not auth.state["authenticated"]
        auth.step(FRAME, EMB)   # success 3 -> authenticated
        assert auth.state["authenticated"]

    def test_single_success_count_means_full_gate_passed(self, auth, face):
        """success_count==1 only ever happens after liveness REAL + recognition MATCH."""
        auth.detector.validate_exactly_one_face.return_value = face
        auth.liveness.check_liveness.return_value = ("REAL", 0.95, _details())
        auth.recognizer.verify.return_value = (True, 0.85)
        auth.step(FRAME, EMB)  # consistency only
        assert auth.state["success_count"] == 0
        auth.step(FRAME, EMB)  # first real success
        assert auth.state["success_count"] == 1
        auth.recognizer.verify.assert_called()

    def test_mismatch_resets_accumulated_successes(self, auth, face):
        auth.detector.validate_exactly_one_face.return_value = face
        auth.liveness.check_liveness.return_value = ("REAL", 0.95, _details())
        auth.recognizer.verify.return_value = (True, 0.85)
        auth.state["liveness_consistency"] = 2
        auth.step(FRAME, EMB)
        auth.step(FRAME, EMB)
        assert auth.state["success_count"] == 2
        # a sub-threshold frame resets to zero
        auth.recognizer.verify.return_value = (False, 0.61)
        state = auth.step(FRAME, EMB)
        assert state["success_count"] == 0
        assert not state["authenticated"]


class TestQualityGateNeutralSkip:
    def test_quality_skip_does_not_reset_or_count(self, auth, face):
        auth.detector.validate_exactly_one_face.return_value = face
        auth.state["liveness_consistency"] = 2
        auth.state["success_count"] = 2
        with patch("uniface_unlock.authenticator.assess_face",
                   return_value={"ok": False, "problems": ["blurry"], "metrics": {}}):
            state = auth.step(FRAME, EMB)
        assert state["quality_skips"] == 1
        # neutral: neither the counter nor consistency is touched
        assert state["success_count"] == 2
        assert state["liveness_consistency"] == 2

    def test_quality_skip_short_circuits_before_liveness(self, auth, face):
        auth.detector.validate_exactly_one_face.return_value = face
        auth.liveness.check_liveness.return_value = ("FAKE", 0.9, _details())
        with patch("uniface_unlock.authenticator.assess_face",
                   return_value={"ok": False, "problems": ["dark"], "metrics": {}}):
            auth.step(FRAME, EMB)
        auth.liveness.check_liveness.assert_not_called()


class TestFailureResets:
    @pytest.mark.parametrize("status,details", [
        ("FAKE", _details(real=0.02, fake=0.96)),
        ("UNKNOWN", _details(real=0.3, fake=0.3, mask=0.4)),
    ])
    def test_non_real_liveness_resets(self, auth, face, status, details):
        auth.detector.validate_exactly_one_face.return_value = face
        auth.liveness.check_liveness.return_value = (status, 0.5, details)
        auth.state["liveness_consistency"] = 2
        auth.state["success_count"] = 2
        state = auth.step(FRAME, EMB)
        assert state["success_count"] == 0
        assert state["liveness_consistency"] == 0

    def test_reset_clears_liveness_smoothing(self, auth, face):
        from uniface_unlock.liveness import LivenessError
        auth.detector.validate_exactly_one_face.return_value = face
        auth.liveness.check_liveness.side_effect = LivenessError("boom")
        auth.state["success_count"] = 2
        auth.step(FRAME, EMB)
        auth.liveness.reset_smoothing.assert_called()


class TestTimeoutAndCleanup:
    def test_timeout_raises_and_camera_closed(self, auth, face):
        """On timeout, authenticate() raises code 5 and closes the camera context."""
        auth.timeout_seconds = 0  # force immediate timeout
        auth.enrollment.is_enrolled.return_value = True
        auth.enrollment.load_enrollment.return_value = EMB

        cam = MagicMock()
        cam.__enter__ = MagicMock(return_value=cam)
        cam.__exit__ = MagicMock(return_value=False)
        cam.read.return_value = FRAME

        with patch("uniface_unlock.authenticator.Camera", return_value=cam):
            with pytest.raises(AuthenticationError) as ei:
                auth.authenticate()
        assert ei.value.code == 5
        # context manager protocol used -> __exit__ invoked for cleanup
        cam.__exit__.assert_called_once()

    def test_success_closes_camera(self, auth, face):
        auth.timeout_seconds = 30
        auth.enrollment.is_enrolled.return_value = True
        auth.enrollment.load_enrollment.return_value = EMB
        auth.detector.validate_exactly_one_face.return_value = face
        auth.liveness.check_liveness.return_value = ("REAL", 0.95, _details())
        auth.recognizer.verify.return_value = (True, 0.85)
        auth.required_frames = 1
        auth.liveness.consistency_required = 1

        cam = MagicMock()
        cam.__enter__ = MagicMock(return_value=cam)
        cam.__exit__ = MagicMock(return_value=False)
        cam.read.return_value = FRAME

        with patch("uniface_unlock.authenticator.Camera", return_value=cam):
            ok = auth.authenticate()
        assert ok is True
        cam.__exit__.assert_called_once()


class TestDiagnosticLogs:
    def test_liveness_rejection_logged_without_biometrics(self, auth, face, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="uniface_unlock.authenticator"):
            auth.detector.validate_exactly_one_face.return_value = face
            auth.liveness.check_liveness.return_value = ("FAKE", 0.9,
                                                          _details(real=0.02, fake=0.96))
            auth.step(FRAME, EMB)
        text = caplog.text
        assert "liveness=FAKE" in text
        # probabilities are allowed, raw biometric values are not
        assert "real=" in text and "fake=" in text
        assert "embedding" not in text.lower()
        assert "landmark" not in text.lower()
