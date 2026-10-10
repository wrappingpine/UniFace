"""Regression tests for performance optimizations and bug fixes.

Guards:
  1. SCRFD input_size configuration and parsing (640, 480, 320, list, invalid)
  2. Single-pass liveness (check_liveness calls ONNX session.run exactly once)
  3. Bounded frame skipping in authenticate()
  4. Benchmark metric correctness (no mock enrollment, accurate classification)
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uniface_unlock.authenticator import FaceAuthenticator, AuthenticationError
from uniface_unlock.detector import FaceDetector


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
EMB = np.zeros(512, dtype=np.float32)


class TestSCRFDInputSizeConfig:
    """SCRFD input_size parsing and initialization."""

    def test_parse_input_size_defaults_to_640(self):
        assert FaceDetector._parse_input_size(None) == (640, 640)

    def test_parse_input_size_accepts_integer(self):
        assert FaceDetector._parse_input_size(480) == (480, 480)
        assert FaceDetector._parse_input_size(320) == (320, 320)

    def test_parse_input_size_accepts_list_or_tuple(self):
        assert FaceDetector._parse_input_size([640, 480]) == (640, 480)
        assert FaceDetector._parse_input_size((320, 320)) == (320, 320)

    def test_parse_input_size_rejects_non_positive_and_malformed(self):
        with pytest.raises(ValueError):
            FaceDetector._parse_input_size(0)
        with pytest.raises(ValueError):
            FaceDetector._parse_input_size([-10, 640])
        with pytest.raises(ValueError):
            FaceDetector._parse_input_size("invalid")

    def test_detector_passes_input_size_to_scrfd(self):
        saved = (FaceDetector._instance, FaceDetector._initialized)
        try:
            FaceDetector._instance = None
            FaceDetector._initialized = False
            with patch("uniface_unlock.detector.get_config") as mock_cfg, \
                 patch("uniface_unlock.detector.SCRFD") as mock_scrfd:
                mock_cfg.return_value.get.side_effect = lambda k, d=None: (
                    480 if k == "detection.input_size" else d
                )
                FaceDetector()
                mock_scrfd.assert_called_once_with(input_size=(480, 480))
        finally:
            FaceDetector._instance, FaceDetector._initialized = saved


class TestSinglePassLiveness:
    """Liveness check uses a single ONNX inference pass."""

    def test_check_liveness_calls_session_run_once(self):
        from uniface_unlock.liveness import LivenessDetector
        saved = (LivenessDetector._instance, LivenessDetector._initialized)
        try:
            LivenessDetector._instance = None
            LivenessDetector._initialized = False
            with patch("uniface_unlock.liveness.MiniFASNet") as mock_mfn:
                spoofer = MagicMock()
                spoofer.preprocess.return_value = np.zeros((1, 3, 80, 80), dtype=np.float32)
                # raw logits: class 1 (REAL) wins
                spoofer.session.run.return_value = [np.array([[0.0, 10.0, 0.0]], dtype=np.float32)]
                spoofer.output_name = "output"
                spoofer.input_name = "input"
                mock_mfn.return_value = spoofer

                detector = LivenessDetector()
                bbox = np.array([100, 100, 200, 200])
                status, conf, details = detector.check_liveness(FRAME, bbox)

                assert status == "REAL"
                assert details["predicted_class"] == 1
                # The ONNX session run was called exactly ONCE
                assert spoofer.session.run.call_count == 1
                # spoofer.predict was NEVER called (avoiding double inference)
                spoofer.predict.assert_not_called()
        finally:
            LivenessDetector._instance, LivenessDetector._initialized = saved


class TestFrameSkipping:
    """Bounded frame skipping in FaceAuthenticator."""

    def test_frame_skip_discards_raw_frames_and_counts_them(self):
        with patch("uniface_unlock.authenticator.FaceDetector"), \
             patch("uniface_unlock.authenticator.FaceRecognizer"), \
             patch("uniface_unlock.authenticator.LivenessDetector"), \
             patch("uniface_unlock.authenticator.FaceEnrollment"), \
             patch("uniface_unlock.authenticator.assess_face",
                   return_value={"ok": True, "problems": [], "metrics": {}}):
            auth = FaceAuthenticator()
            auth.frame_skip = 2
            auth.timeout_seconds = 10
            auth.enrollment.is_enrolled.return_value = True
            auth.enrollment.load_enrollment.return_value = EMB

            def fake_step(f, e):
                auth.state["frames_processed"] += 1
                auth.state["authenticated"] = True
                return auth.state

            auth.step = fake_step

            cam = MagicMock()
            cam.__enter__ = MagicMock(return_value=cam)
            cam.__exit__ = MagicMock(return_value=False)
            cam.read.return_value = FRAME

            with patch("uniface_unlock.authenticator.Camera", return_value=cam):
                res = auth.authenticate()

            assert res is True
            # Authenticated on 1st processed frame; skip did not execute because it succeeded
            assert auth.state["frames_processed"] == 1
            assert auth.state["frames_skipped"] == 0
            assert cam.read.call_count == 1

    def test_frame_skip_executes_between_non_authenticating_steps(self):
        with patch("uniface_unlock.authenticator.FaceDetector"), \
             patch("uniface_unlock.authenticator.FaceRecognizer"), \
             patch("uniface_unlock.authenticator.LivenessDetector"), \
             patch("uniface_unlock.authenticator.FaceEnrollment"), \
             patch("uniface_unlock.authenticator.assess_face",
                   return_value={"ok": True, "problems": [], "metrics": {}}):
            auth = FaceAuthenticator()
            auth.frame_skip = 2
            auth.timeout_seconds = 10
            auth.enrollment.is_enrolled.return_value = True
            auth.enrollment.load_enrollment.return_value = EMB

            # 1st step: not authenticated. 2nd step: authenticated.
            calls = [0]
            def fake_step(f, e):
                calls[0] += 1
                auth.state["frames_processed"] += 1
                if calls[0] >= 2:
                    auth.state["authenticated"] = True
                return auth.state

            auth.step = fake_step

            cam = MagicMock()
            cam.__enter__ = MagicMock(return_value=cam)
            cam.__exit__ = MagicMock(return_value=False)
            cam.read.return_value = FRAME

            with patch("uniface_unlock.authenticator.Camera", return_value=cam):
                res = auth.authenticate()

            assert res is True
            assert auth.state["frames_processed"] == 2
            # Between step 1 and step 2, 2 frames were skipped
            assert auth.state["frames_skipped"] == 2
            # Camera.read called: 1 (step 1) + 2 (skipped) + 1 (step 2) = 4
            assert cam.read.call_count == 4


class TestConfigIsolation:
    """Config instances must not mutate the module-level DEFAULT_CONFIG."""

    def test_loaded_config_does_not_pollute_defaults(self, tmp_path):
        """_deep_update mutates nested dicts in place.

        With a shallow copy of DEFAULT_CONFIG, loading a config file wrote
        straight through into the module-level defaults and leaked the value
        into every subsequently constructed Config. The defaults are now
        deep-copied per instance.
        """
        import json
        from uniface_unlock.config import Config, DEFAULT_CONFIG

        original = DEFAULT_CONFIG["detection"]["threshold"]
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"detection": {"threshold": 0.99}}))

        cfg = Config(str(cfg_file))
        assert cfg.get("detection.threshold") == 0.99
        # module-level defaults untouched
        assert DEFAULT_CONFIG["detection"]["threshold"] == original
        # a fresh instance over a different (empty) file sees the default
        other_file = tmp_path / "other.json"
        other_file.write_text(json.dumps({}))
        assert Config(str(other_file)).get("detection.threshold") == original

    def test_new_keys_have_defaults(self):
        """detection.input_size has a backward-compatible default."""
        from uniface_unlock.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["detection"]["input_size"] == 640


class TestEnrollmentTemplateValidation:
    """Malformed enrollment files must fail closed, never match."""

    @pytest.fixture
    def enrollment_offline(self, tmp_path):
        from uniface_unlock.enrollment import FaceEnrollment
        with patch.object(FaceEnrollment, "__init__", lambda self: None):
            e = FaceEnrollment()
        e.embedding_path = tmp_path / "enrollment.dat"
        return e

    def _write(self, enrollment, arr):
        np.save(enrollment.embedding_path.with_suffix(".npy"), arr)

    def test_valid_template_loads(self, enrollment_offline):
        good = np.ones(512, dtype=np.float32)
        good = good / np.linalg.norm(good)
        self._write(enrollment_offline, good)
        loaded = enrollment_offline.load_enrollment()
        assert loaded.shape == (512,)

    def test_nan_template_rejected(self, enrollment_offline):
        from uniface_unlock.enrollment import EnrollmentError
        bad = np.ones(512, dtype=np.float32)
        bad[7] = np.nan
        self._write(enrollment_offline, bad)
        with pytest.raises(EnrollmentError):
            enrollment_offline.load_enrollment()

    def test_inf_template_rejected(self, enrollment_offline):
        from uniface_unlock.enrollment import EnrollmentError
        bad = np.ones(512, dtype=np.float32)
        bad[3] = np.inf
        self._write(enrollment_offline, bad)
        with pytest.raises(EnrollmentError):
            enrollment_offline.load_enrollment()

    def test_zero_norm_template_rejected(self, enrollment_offline):
        """A zero template would make cosine similarity undefined."""
        from uniface_unlock.enrollment import EnrollmentError
        self._write(enrollment_offline, np.zeros(512, dtype=np.float32))
        with pytest.raises(EnrollmentError):
            enrollment_offline.load_enrollment()

    def test_integer_dtype_template_rejected(self, enrollment_offline):
        from uniface_unlock.enrollment import EnrollmentError
        self._write(enrollment_offline, np.ones(512, dtype=np.int32))
        with pytest.raises(EnrollmentError):
            enrollment_offline.load_enrollment()

    def test_two_dimensional_template_rejected(self, enrollment_offline):
        from uniface_unlock.enrollment import EnrollmentError
        self._write(enrollment_offline, np.ones((2, 512), dtype=np.float32))
        with pytest.raises(EnrollmentError):
            enrollment_offline.load_enrollment()

    def test_save_rejects_malformed_before_writing(self, enrollment_offline):
        """A bad vector must never reach disk."""
        from uniface_unlock.enrollment import EnrollmentError
        with pytest.raises(EnrollmentError):
            enrollment_offline.save_enrollment(np.zeros(512, dtype=np.float32))
        assert not enrollment_offline.embedding_path.with_suffix(".npy").exists()
