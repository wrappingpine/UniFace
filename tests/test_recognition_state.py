"""Regression tests for recognition state, liveness rejection, and
consecutive-success resets (issue #3 follow-up).

Complements tests/test_auth_sequencing.py by focusing on:
- similarity at/above/below the configured threshold decides the match
- recognition exception and mismatch reset both counters
- liveness FAKE/UNKNOWN rejection resets accumulated progress
- progress survives neutral quality skips but never partial gates
"""

import numpy as np
import pytest
from unittest.mock import patch

from uniface.types import Face
from uniface_unlock.authenticator import FaceAuthenticator
from uniface_unlock.detector import DetectionError
from uniface_unlock.recognizer import RecognitionError

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
EMB = np.zeros(512, dtype=np.float32)


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
def ready_face(auth):
    """Face + liveness pre-satisfied so each step() reaches recognition."""
    face = Face(
        bbox=np.array([100, 100, 300, 300]),
        confidence=0.95,
        landmarks=np.array([[150, 150], [250, 150], [200, 200],
                            [160, 260], [240, 260]]),
    )
    auth.detector.validate_exactly_one_face.return_value = face
    auth.liveness.check_liveness.return_value = ("REAL", 0.95, _details())
    auth.state["liveness_consistency"] = 2
    return face


class TestRecognitionState:
    """The 0.70 threshold is the decision boundary; nothing else decides."""

    def test_similarity_above_threshold_counts_success(self, auth, ready_face):
        auth.recognizer.verify.return_value = (True, 0.85)
        state = auth.step(FRAME, EMB)
        assert state["success_count"] == 1
        assert not state["authenticated"]

    def test_similarity_at_threshold_boundary_counts(self, auth, ready_face):
        # verify() applies >= threshold itself; the state machine trusts is_match
        auth.recognizer.verify.return_value = (True, 0.700)
        state = auth.step(FRAME, EMB)
        assert state["success_count"] == 1

    def test_sub_threshold_similarity_resets(self, auth, ready_face):
        auth.state["success_count"] = 2
        auth.recognizer.verify.return_value = (False, 0.699)
        state = auth.step(FRAME, EMB)
        assert state["success_count"] == 0
        assert state["liveness_consistency"] == 0

    def test_recognition_exception_resets(self, auth, ready_face):
        auth.state["success_count"] = 2
        auth.recognizer.verify.side_effect = RecognitionError("model failure")
        state = auth.step(FRAME, EMB)
        assert state["success_count"] == 0
        auth.liveness.reset_smoothing.assert_called()

    def test_threshold_not_bypassed_by_liveness_confidence(self, auth, ready_face):
        """A very confident REAL liveness result must not skip recognition."""
        auth.liveness.check_liveness.return_value = ("REAL", 0.999,
                                                      _details(real=0.999))
        auth.recognizer.verify.return_value = (False, 0.10)
        state = auth.step(FRAME, EMB)
        auth.recognizer.verify.assert_called_once()
        assert state["success_count"] == 0


class TestLivenessRejectionResets:
    """Any non-REAL liveness verdict wipes accumulated progress."""

    @pytest.mark.parametrize("status", ["FAKE", "UNKNOWN"])
    def test_non_real_wipes_two_stored_successes(self, auth, ready_face, status):
        auth.state["success_count"] = 2
        auth.liveness.check_liveness.return_value = (status, 0.6,
                                                     _details(real=0.2))
        state = auth.step(FRAME, EMB)
        assert state["success_count"] == 0
        assert state["liveness_consistency"] == 0
        assert not state["authenticated"]

    def test_real_after_reset_needs_consistency_again(self, auth, ready_face):
        """After a FAKE wipe, two REAL frames restore consistency before counting."""
        auth.state["success_count"] = 2
        auth.liveness.check_liveness.side_effect = [
            ("FAKE", 0.9, _details(real=0.02, fake=0.96)),
            ("REAL", 0.95, _details()),
            ("REAL", 0.95, _details()),
            ("REAL", 0.95, _details()),
        ]
        auth.recognizer.verify.return_value = (True, 0.85)
        state = auth.step(FRAME, EMB)   # FAKE -> wipe
        assert state["success_count"] == 0
        state = auth.step(FRAME, EMB)   # consistency 1/2
        assert state["success_count"] == 0
        state = auth.step(FRAME, EMB)   # consistency 2/2, success 1
        assert state["success_count"] == 1
        state = auth.step(FRAME, EMB)   # success 2
        assert state["success_count"] == 2
        assert not state["authenticated"]


class TestConsecutiveSuccessResets:
    """success_count must be strictly consecutive across every gate."""

    def test_interrupted_sequence_restarts_from_zero(self, auth, ready_face):
        auth.recognizer.verify.return_value = (True, 0.85)
        auth.step(FRAME, EMB)          # success 1
        auth.step(FRAME, EMB)          # success 2
        # one bad-quality frame must NOT reset (neutral skip)...
        with patch("uniface_unlock.authenticator.assess_face",
                   return_value={"ok": False, "problems": ["blur"], "metrics": {}}):
            state = auth.step(FRAME, EMB)
        assert state["success_count"] == 2
        # ...but one sub-threshold frame must reset
        auth.recognizer.verify.return_value = (False, 0.44)
        state = auth.step(FRAME, EMB)
        assert state["success_count"] == 0
        # and the full run must restart through liveness consistency again
        auth.recognizer.verify.return_value = (True, 0.85)
        state = auth.step(FRAME, EMB)
        assert state["liveness_consistency"] == 1
        assert state["success_count"] == 0

    def test_authenticated_requires_three_consecutive(self, auth, ready_face):
        auth.recognizer.verify.return_value = (True, 0.9)
        auth.step(FRAME, EMB)
        auth.recognizer.verify.return_value = (False, 0.2)
        auth.step(FRAME, EMB)  # breaks the run
        auth.recognizer.verify.return_value = (True, 0.9)
        auth.step(FRAME, EMB)  # consistency 1/2 again
        auth.step(FRAME, EMB)  # success 1
        auth.step(FRAME, EMB)  # success 2
        assert not auth.state["authenticated"]
        auth.step(FRAME, EMB)  # success 3
        assert auth.state["authenticated"]


class TestTimeoutRunSummary:
    """Timeout must log a non-sensitive counter summary for diagnosis."""

    def _run_to_timeout(self, auth, caplog, logging, timeout):
        from uniface_unlock.authenticator import AuthenticationError
        from unittest.mock import MagicMock

        auth.timeout_seconds = timeout
        auth.enrollment.is_enrolled.return_value = True
        auth.enrollment.load_enrollment.return_value = EMB

        cam = MagicMock()
        cam.__enter__ = MagicMock(return_value=cam)
        cam.__exit__ = MagicMock(return_value=False)
        cam.read.return_value = FRAME

        # monotonic clock; time.time is also consumed by log-record creation,
        # so the exact iteration count is not asserted via the clock
        t = [0.0]

        def fake_clock():
            t[0] += 0.001
            return t[0]

        with caplog.at_level(logging.INFO, logger="uniface_unlock.authenticator"):
            with patch("uniface_unlock.authenticator.Camera", return_value=cam), \
                 patch("uniface_unlock.authenticator.time.time", side_effect=fake_clock):
                with pytest.raises(AuthenticationError) as ei:
                    auth.authenticate()
        assert ei.value.code == 5
        summaries = [r.getMessage() for r in caplog.records
                     if "auth run summary at timeout" in r.getMessage()]
        assert len(summaries) == 1
        return summaries[0]

    def test_timeout_logs_counter_summary_without_biometrics(self, auth, caplog):
        """Instant timeout: summary line present, counters keyed, no biometrics."""
        import logging
        msg = self._run_to_timeout(auth, caplog, logging, timeout=0)
        for key in ("frames_processed=", "quality_skips=", "success_count=",
                    "liveness_consistency="):
            assert key in msg
        lowered = msg.lower()
        for banned in ("embedding", "landmark", "similarity=0"):
            assert banned not in lowered

    def test_step_increments_frames_processed(self, auth):
        """The counters summarized at timeout are maintained by step()."""
        from uniface_unlock.detector import DetectionError
        auth.detector.validate_exactly_one_face.side_effect = DetectionError("no face")
        auth.step(FRAME, EMB)
        auth.step(FRAME, EMB)
        assert auth.state["frames_processed"] == 2
        assert auth.state["success_count"] == 0
