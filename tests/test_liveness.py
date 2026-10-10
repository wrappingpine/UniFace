"""Unit tests for LivenessDetector interpretation of MiniFASNet outputs.

Tests verify:
- MiniFASNet V2 3-class ordering: [Fake, Real, Mask/Unknown]
- Index 1 is Real
- Index 0 is Fake
- Index 2 is Mask/Unknown
- Decision logic is fail-closed on current frame (never smoothed)
- Unknown classification when max_prob < threshold
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from uniface.types import SpoofingResult
from uniface_unlock.liveness import LivenessDetector


@pytest.fixture
def liveness():
    """Fresh LivenessDetector with a mocked MiniFASNet spoofer.

    LivenessDetector is a singleton: if any earlier test (in any collected
    file) already constructed a real instance, the patch of MiniFASNet would
    be bypassed by __init__'s early return and spoofer.predict would be a
    real bound method (AttributeError on .return_value). Reset the singleton
    before and after the test so the mock is guaranteed to take effect and
    other tests never observe this mocked instance.
    """
    saved = (LivenessDetector._instance, LivenessDetector._initialized)
    LivenessDetector._instance = None
    LivenessDetector._initialized = False
    with patch("uniface_unlock.liveness.MiniFASNet"):
        l = LivenessDetector()
        l.threshold = 0.5
        l.consistency_required = 2
        l._real_prob_history.clear()
        yield l
    LivenessDetector._instance, LivenessDetector._initialized = saved


class TestLivenessInterpretation:
    """Verify correct interpretation of MiniFASNet 3-class outputs."""

    def test_mini_fasnet_class_ordering(self, liveness):
        """MiniFASNet V2 must map index 1 to Real, 0 to Fake, 2 to Mask/Unknown."""
        # Mock spoofing result: what predict() returns
        mock_result = SpoofingResult(is_real=True, confidence=0.85)
        liveness.spoofer.predict.return_value = mock_result
        # Mock raw logits -> softmax probabilities
        # Index 0: Fake, Index 1: Real, Index 2: Mask
        fake_prob, real_prob, mask_prob = 0.05, 0.85, 0.10
        fake_sum = np.array([[fake_prob, real_prob, mask_prob]], dtype=np.float32)
        
        liveness.spoofer.preprocess.return_value = np.zeros((1, 1, 240, 240), dtype=np.float32)
        liveness.spoofer.session.run.return_value = [np.array([[0.1, 2.0, -1.0]], dtype=np.float32)]
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 300, 300])
        
        # Verify decision is driven by real_prob >= threshold and is_real
        with patch("uniface.common.softmax") as mock_softmax:
            mock_softmax.return_value = fake_sum
            status, confidence, details = liveness.check_liveness(image, bbox)
        
        # Status must be REAL given real_prob=0.85 >= 0.5 and is_real=True
        assert status == "REAL"
        assert confidence == pytest.approx(0.85)
        assert details["real_probability"] == pytest.approx(0.85)
        assert details["fake_probability"] == pytest.approx(0.05)

    def test_spoof_detected_as_fake(self, liveness):
        """A high Fake probability and is_real=False must return FAKE."""
        mock_result = SpoofingResult(is_real=False, confidence=0.90)
        liveness.spoofer.predict.return_value = mock_result
        
        fake_sum = np.array([[0.90, 0.05, 0.05]], dtype=np.float32)
        liveness.spoofer.preprocess.return_value = np.zeros((1, 1, 240, 240), dtype=np.float32)
        liveness.spoofer.session.run.return_value = [np.array([[2.0, 0.1, 0.1]])]
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 300, 300])
        
        with patch("uniface.common.softmax") as mock_softmax:
            mock_softmax.return_value = fake_sum
            status, confidence, details = liveness.check_liveness(image, bbox)
        
        assert status == "FAKE"
        assert details["fake_probability"] == pytest.approx(0.90)

    def test_mask_detected_as_fake(self, liveness):
        """A high Mask probability should be interpreted as FAKE (fail closed)."""
        # MiniFASNet may classify masks as class 2; policy: any non-Real is FAKE
        mock_result = SpoofingResult(is_real=False, confidence=0.80)
        liveness.spoofer.predict.return_value = mock_result
        
        fake_sum = np.array([[0.10, 0.10, 0.80]], dtype=np.float32)
        liveness.spoofer.preprocess.return_value = np.zeros((1, 1, 240, 240), dtype=np.float32)
        liveness.spoofer.session.run.return_value = [np.array([[-1.0, -1.0, 2.0]])]
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 300, 300])
        
        with patch("uniface.common.softmax") as mock_softmax:
            mock_softmax.return_value = fake_sum
            status, confidence, details = liveness.check_liveness(image, bbox)
        
        # Mask is non-REAL → fail closed → FAKE
        assert status == "FAKE"
        assert details["mask_probability"] == pytest.approx(0.80)

    def test_unknown_on_low_confidence(self, liveness):
        """Max probability below threshold → UNKNOWN (fail closed, never accept)."""
        mock_result = SpoofingResult(is_real=False, confidence=0.45)
        liveness.spoofer.predict.return_value = mock_result
        
        fake_sum = np.array([[0.35, 0.30, 0.35]], dtype=np.float32)
        liveness.spoofer.preprocess.return_value = np.zeros((1, 1, 240, 240), dtype=np.float32)
        liveness.spoofer.session.run.return_value = [np.array([[0.0, 0.0, 0.0]])]
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 300, 300])
        
        with patch("uniface.common.softmax") as mock_softmax:
            mock_softmax.return_value = fake_sum
            status, confidence, details = liveness.check_liveness(image, bbox)
        
        assert status == "UNKNOWN"
        assert details["confidence"] == pytest.approx(0.35)
        assert details["confidence"] < liveness.threshold

    def test_decision_uses_current_frame_not_smoothed(self, liveness):
        """The authentication decision must use current real_prob, not smoothed history."""
        # History has high smoothed value, but current frame is FAKE
        liveness._real_prob_history.extend([0.9, 0.9, 0.9, 0.9, 0.9])
        
        mock_result = SpoofingResult(is_real=False, confidence=0.90)
        liveness.spoofer.predict.return_value = mock_result
        
        fake_sum = np.array([[0.90, 0.05, 0.05]], dtype=np.float32)
        liveness.spoofer.preprocess.return_value = np.zeros((1, 1, 240, 240), dtype=np.float32)
        liveness.spoofer.session.run.return_value = [np.array([[2.0, 0.1, 0.1]])]
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 300, 300])
        
        with patch("uniface.common.softmax") as mock_softmax:
            mock_softmax.return_value = fake_sum
            status, confidence, details = liveness.check_liveness(image, bbox)
        
        # Must be FAKE despite smoothed history being high (fail closed)
        assert status == "FAKE"
        # Smoothed value is diagnostics only; it should contain some history
        # but the decision is based on current real_prob, not this value
        assert details["smoothed_real_probability"] > 0.0
        # Verify the decision isn't using smoothed value
        assert details["real_probability"] == pytest.approx(0.05)

    def test_real_prob_below_threshold_rejects(self, liveness):
        """Real probability below threshold must reject even if is_real=True."""
        mock_result = SpoofingResult(is_real=True, confidence=0.49)
        liveness.spoofer.predict.return_value = mock_result
        
        fake_sum = np.array([[0.30, 0.49, 0.21]], dtype=np.float32)
        liveness.spoofer.preprocess.return_value = np.zeros((1, 1, 240, 240), dtype=np.float32)
        liveness.spoofer.session.run.return_value = [np.array([[0.0, 0.0, 0.0]])]
        
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 300, 300])
        
        with patch("uniface.common.softmax") as mock_softmax:
            mock_softmax.return_value = fake_sum
            status, confidence, details = liveness.check_liveness(image, bbox)
        
        # Max probability is 0.49 which is below threshold → UNKNOWN, fail closed
        assert status == "UNKNOWN"
