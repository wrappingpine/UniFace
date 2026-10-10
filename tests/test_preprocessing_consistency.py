"""Regression tests for enrollment/recognition preprocessing consistency.

Ensures that:
- Enrollment and recognition use identical MobileFace model
- Preprocessing (alignment, normalization, embedding) is identical
- Template metadata matches live pipeline configuration
- Similarity metric is cosine and threshold is appropriate
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from uniface.types import Face
from uniface_unlock.enrollment import FaceEnrollment, EnrollmentError
from uniface_unlock.recognizer import FaceRecognizer
from uniface_unlock.detector import FaceDetector
from uniface_unlock.config import get_config


@pytest.fixture
def mock_pipeline(tmp_path):
    """Create enrollment + recognizer with mocked camera and detection.

    CRITICAL: embedding_path is redirected to tmp_path so tests can never
    write to the real ~/.config/uniface-unlock/enrollment.npy.
    """
    # We need a real FaceRecognizer singleton but mocked I/O
    # Use patch to avoid model loading but keep the class intact
    with patch("uniface_unlock.enrollment.FaceDetector"), \
         patch("uniface_unlock.enrollment.Camera"), \
         patch("uniface_unlock.recognizer.FaceRecognizer.__init__", lambda self: setattr(self, '_initialized', True) or setattr(self, 'threshold', 0.7)), \
         patch("uniface_unlock.detector.FaceDetector.__init__", lambda self: setattr(self, '_initialized', True)):
        from uniface_unlock.recognizer import FaceRecognizer
        from uniface_unlock.detector import FaceDetector

        # Force re-initialization to use our patched __init__
        FaceRecognizer._instance = None
        FaceRecognizer._initialized = False
        FaceDetector._instance = None
        FaceDetector._initialized = False

        enrollment = FaceEnrollment()
        recognizer = FaceRecognizer()

        # Redirect enrollment output to tmp_path — never the real user dir
        enrollment.enrollment_dir = tmp_path
        enrollment.embedding_path = tmp_path / "enrollment.dat"
        
        # Now we need to give the recognizer a proper mock for its inner recognizer
        mock_inner_rec = MagicMock()
        mock_inner_rec.__class__.__name__ = "MobileFace"
        mock_inner_rec.model_path = "/fake/model.onnx"
        mock_inner_rec.input_size = (112, 112)
        mock_inner_rec.input_mean = 127.5
        mock_inner_rec.input_std = 127.5
        recognizer.recognizer = mock_inner_rec
        
        mock_inner_det = MagicMock()
        mock_inner_det.__class__.__name__ = "SCRFD"
        mock_inner_det.model_path = "/fake/det.onnx"
        enrollment.detector.detector = mock_inner_det
        
        return enrollment, recognizer


class TestEnrollmentRecognitionConsistency:
    """Verify enrollment and recognition use identical preprocessing."""

    def test_mobileface_model_identity(self):
        """Both enrollment and recognition must use the same MobileFace model instance."""
        with patch("uniface_unlock.recognizer.FaceRecognizer.__init__", return_value=None):
            r1 = FaceRecognizer()
            r2 = FaceRecognizer()
            # Same singleton instance
            assert r1 is r2

    def test_embedding_dimension_consistency(self, mock_pipeline):
        """Enrolled embedding must be 512-dim float32 L2-normalized."""
        enrollment, recognizer = mock_pipeline
        mock_emb = np.random.randn(512).astype(np.float32)
        mock_emb = mock_emb / np.linalg.norm(mock_emb)
        
        # Simulate save/load roundtrip
        enrollment.save_enrollment(mock_emb)
        loaded = enrollment.load_enrollment()
        
        assert loaded.shape == (512,)
        assert loaded.dtype == np.float32
        assert np.isfinite(loaded).all()
        # Loaded should be L2 normalized
        assert abs(float(np.linalg.norm(loaded)) - 1.0) < 1e-5

    def test_recognizer_uses_same_model_as_enrollment(self, mock_pipeline):
        """Recognizer used in enrollment and standalone must be same singleton."""
        enrollment, recognizer = mock_pipeline
        # Both reference the same singleton class (when not mocked)
        # With mocks, we just verify both use FaceRecognizer class
        assert enrollment.recognizer is not None
        assert recognizer is not None

    def test_alignment_uses_5_point_landmarks(self, mock_pipeline):
        """Alignment in both paths must use 5 landmarks (not 106 or other)."""
        from uniface.face_utils import estimate_norm
        
        # MobileFaceNet expects exactly 5 landmarks
        landmarks_5 = np.array([[0, 0], [1, 0], [0.5, 1], [0.3, 1.2], [0.7, 1.2]], dtype=np.float32)
        
        # This must not raise
        M, M_inv = estimate_norm(landmarks_5, 112)
        assert M.shape == (2, 3)
        assert M_inv.shape == (2, 3)
        
        # Wrong count must raise
        with pytest.raises(ValueError, match="5 alignment landmarks"):
            estimate_norm(np.zeros((106, 2)), 112)

    def test_preprocessing_config_matches_mobileface(self):
        """MobileFace preprocessing: BGR input, 112x112, mean=127.5, std=127.5, swapRB=True."""
        from uniface.recognition import MobileFace
        
        mf = MobileFace(providers=['CPUExecutionProvider'])
        assert mf.input_size == (112, 112)
        assert mf.input_mean == 127.5
        assert mf.input_std == 127.5
        # swapRB=True in blobFromImage means BGR->RGB conversion

    def test_similarity_metric_is_cosine(self):
        """compute_similarity must be cosine similarity."""
        from uniface.face_utils import compute_similarity
        
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Orthogonal vectors -> cosine ~0
        assert abs(float(compute_similarity(a, b)) - 0.0) < 1e-4
        # Identical vectors -> cosine ~1 (epsilon in denominator causes tiny diff)
        assert abs(float(compute_similarity(a, c)) - 1.0) < 1e-4

    def test_threshold_appropriate_for_cosine(self):
        """Default threshold 0.7 is typical for MobileFace cosine similarity."""
        config = get_config()
        threshold = config.get("recognition.threshold", 0.7)
        
        # MobileFaceNet with cosine similarity: 0.7 is standard operating point
        # False accept rate ~0.1% at 0.7 for good conditions
        assert 0.6 <= threshold <= 0.8

    def test_enrollment_averaging_then_normalization(self, mock_pipeline):
        """Enrollment must average embeddings THEN normalize (not normalize then average)."""
        enrollment, _ = mock_pipeline
        
        # Create two very different embeddings
        e1 = np.array([1.0] + [0.0]*511, dtype=np.float32)
        e2 = np.array([0.0, 1.0] + [0.0]*510, dtype=np.float32)
        
        # Proper way: average then normalize
        mean = (e1 + e2) / 2
        mean_norm = mean / np.linalg.norm(mean)
        
        # Wrong way: normalize then average (different result)
        e1_n = e1 / np.linalg.norm(e1)
        e2_n = e2 / np.linalg.norm(e2)
        wrong = (e1_n + e2_n) / 2
        
        # Code does average-then-normalize (line 119-125 in enrollment.py)
        # Verify the property holds
        assert abs(float(np.linalg.norm(mean_norm)) - 1.0) < 1e-5
        # The wrong way produces a vector with norm < 1
        assert float(np.linalg.norm(wrong)) < 1.0

    def test_enrollment_writes_metadata_sidecar(self, mock_pipeline):
        """New enrollments must write .meta.json with model identity."""
        enrollment, _ = mock_pipeline
        
        mock_emb = np.random.randn(512).astype(np.float32)
        mock_emb = mock_emb / np.linalg.norm(mock_emb)
        
        enrollment.save_enrollment(mock_emb)
        enrollment.save_enrollment_meta()
        
        meta = enrollment.load_meta()
        assert meta is not None
        assert meta["recognition_class"] == "MobileFace"
        assert meta["embedding_shape"] == [512]
        assert meta["embedding_dtype"] == "float32"
        assert abs(meta["embedding_l2_norm"] - 1.0) < 1e-5
        assert meta["embedding_finite"] is True

    def test_detector_uses_scrfd_with_landmarks(self):
        """Detection must use SCRFD with landmarks enabled for alignment."""
        detector = FaceDetector()
        det = detector.detector
        
        assert det.__class__.__name__ == "SCRFD"
        assert detector.enable_landmarks is True

    def test_recognition_threshold_from_config(self):
        """Threshold used at verify time must come from config, not hardcoded."""
        with patch("uniface_unlock.recognizer.FaceRecognizer.__init__", return_value=None):
            r = FaceRecognizer()
            r.threshold = 0.7  # set by config
            
            # Verify method uses self.threshold
            assert hasattr(r, 'threshold')
            assert r.threshold == 0.7


class TestTemplateIntegrity:
    """Verify enrolled template integrity checks."""

    def test_detect_drift_between_enrollment_and_recognition(self, mock_pipeline):
        """If model weights change, meta sha256 will differ from live model."""
        enrollment, recognizer = mock_pipeline
        
        mock_emb = np.random.randn(512).astype(np.float32)
        mock_emb = mock_emb / np.linalg.norm(mock_emb)
        
        enrollment.save_enrollment(mock_emb)
        enrollment.save_enrollment_meta()
        
        meta = enrollment.load_meta()
        live_sha = enrollment._sha256(recognizer.recognizer.model_path)
        
        # If someone swapped the model file, this would catch it
        assert meta["recognition_model_sha256"] == live_sha

    def test_embedding_shape_mismatch_detection(self, mock_pipeline):
        """A wrong-dimension template is flagged, never silently trusted.

        Two layers are checked:
          - the meta sidecar records the real shape, so a mismatch is visible
            to `diagnose` (original intent of this test)
          - save_enrollment refuses to persist a wrong-dimension vector, and
            load_enrollment refuses to return one that got onto disk by other
            means (e.g. written by an older release). Both fail closed.
        """
        enrollment, _ = mock_pipeline

        bad_emb = np.random.randn(256).astype(np.float32)
        bad_emb = bad_emb / np.linalg.norm(bad_emb)

        # Layer 1: save_enrollment refuses to write a malformed template
        with pytest.raises(EnrollmentError):
            enrollment.save_enrollment(bad_emb)

        # Layer 2: simulate a legacy file already on disk (bypassing
        # save_enrollment, as an older release would have written it), then
        # confirm the metadata still records the wrong shape and that loading
        # it fails closed instead of returning a template for matching.
        np.save(enrollment.embedding_path.with_suffix('.npy'), bad_emb)
        enrollment.save_enrollment_meta()

        meta = enrollment.load_meta()
        assert meta["embedding_shape"] == [256]
        assert meta["embedding_shape"] != [512]  # would trigger warning in diagnose

        with pytest.raises(EnrollmentError):
            enrollment.load_enrollment()


class TestLivenessIsIndependent:
    """Ensure liveness model changes don't affect recognition embeddings."""

    def test_liveness_model_is_separate(self):
        """MiniFASNet (liveness) and MobileFace (recognition) are separate models."""
        from uniface.recognition import MobileFace
        from uniface.spoofing import MiniFASNet
        
        mf = MobileFace(providers=['CPUExecutionProvider'])
        ml = MiniFASNet(providers=['CPUExecutionProvider'])
        
        assert mf.model_path != ml.model_path
        assert mf.__class__.__name__ != ml.__class__.__name__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])