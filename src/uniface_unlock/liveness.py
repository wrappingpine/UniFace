"""Anti-spoofing / liveness detection module using UniFace."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Tuple

import numpy as np

from uniface.spoofing import MiniFASNet
from uniface.types import SpoofingResult

from .config import get_config

logger = logging.getLogger(__name__)


class LivenessError(Exception):
    """Liveness detection errors."""
    pass


class LivenessDetector:
    """Anti-spoofing / liveness detection.

    Requirements:
    - Distinguish REAL / FAKE / UNKNOWN
    - Handle uncertain predictions conservatively
    - Fail closed (UNKNOWN → reject)
    - Use smoothing for stability
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
        model_name = config.get("liveness.model", "MiniFASNet")
        self.threshold = config.get("liveness.threshold", 0.5)
        self.consistency_required = config.get("liveness.consistency_required", 2)

        # Initialize MiniFASNet
        from uniface.constants import MiniFASNetWeights
        self.spoofer = MiniFASNet(
            model_name=MiniFASNetWeights.V2,
            providers=['CPUExecutionProvider']
        )

        # Smoothing buffer for stability
        self._real_prob_history = deque(maxlen=5)

        logger.info(f"Liveness detector initialized: {model_name} with threshold {self.threshold}, consistency={self.consistency_required}")
        self._initialized = True

    def check_liveness(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        raw_output: bool = False,
    ) -> Tuple[str, float, dict]:
        """Check if face is real or spoofed.

        Args:
            image: Input image (BGR)
            bbox: Face bounding box [x1, y1, x2, y2]
            raw_output: If True, return raw details

        Returns:
            (status, confidence, details) where status is "REAL", "FAKE", or "UNKNOWN"
        """
        start_time = time.time()

        try:
            # Use the predict method which handles preprocessing correctly
            result: SpoofingResult = self.spoofer.predict(image, bbox)
            latency_ms = (time.time() - start_time) * 1000

            # Get raw probabilities for detailed analysis
            input_tensor = self.spoofer.preprocess(image, bbox)
            raw_logits = self.spoofer.session.run(
                [self.spoofer.output_name],
                {self.spoofer.input_name: input_tensor}
            )[0]

            from uniface.common import softmax
            probs = softmax(raw_logits)[0]

            # MiniFASNet V2 outputs 3 classes: [Fake, Real, Mask/Unknown]
            fake_prob = float(probs[0])
            real_prob = float(probs[1])
            mask_prob = float(probs[2]) if len(probs) > 2 else 0.0

            # Use the model's own prediction logic (class 1 = real)
            predicted_class = int(np.argmax(probs))
            max_prob = float(np.max(probs))
            is_real = (predicted_class == 1)

            # Update smoothing buffer
            self._real_prob_history.append(real_prob)
            smoothed_real_prob = float(np.mean(self._real_prob_history))

            details = {
                'raw_logits': raw_logits.tolist(),
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'mask_probability': mask_prob,
                'predicted_class': predicted_class,
                'is_real': is_real,
                'confidence': max_prob,
                'smoothed_real_probability': smoothed_real_prob,
                'latency_ms': latency_ms,
            }

            # Decision logic - FAIL CLOSED.
            # The decision is driven by the CURRENT frame's real probability,
            # never by a lagging/smoothed value. A smoothed value would let a
            # spoof frame that follows genuine frames pass (fail-open), which
            # is unacceptable for anti-spoofing. The smoothed value is kept in
            # `details` for diagnostics only.
            if real_prob >= self.threshold and is_real:
                status = "REAL"
                logger.debug(
                    f"Liveness passed: real_prob={real_prob:.3f}, "
                    f"smoothed={smoothed_real_prob:.3f}, "
                    f"fake={fake_prob:.3f}, mask={mask_prob:.3f}"
                )
            elif max_prob < self.threshold:
                status = "UNKNOWN"
                logger.warning(
                    f"Liveness uncertain: max_prob={max_prob:.3f} < threshold={self.threshold}, "
                    f"fake={fake_prob:.3f}, real={real_prob:.3f}, mask={mask_prob:.3f}"
                )
            else:
                status = "FAKE"
                logger.warning(
                    f"Liveness failed: class={predicted_class}, "
                    f"fake={fake_prob:.3f}, real={real_prob:.3f}, mask={mask_prob:.3f}"
                )

            if raw_output:
                return status, max_prob, details
            return status, max_prob, details

        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            raise LivenessError(f"Liveness detection failed: {e}")

    def measure_latency(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        num_runs: int = 10,
    ) -> float:
        """Measure average liveness detection latency."""
        latencies = []

        for _ in range(num_runs):
            start = time.time()
            _ = self.spoofer.predict(image, bbox)
            latencies.append((time.time() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"Average liveness latency: {avg_latency:.1f}ms")
        return avg_latency

    def reset_smoothing(self) -> None:
        """Reset the smoothing buffer."""
        self._real_prob_history.clear()

    def get_info(self) -> dict:
        """Get liveness detector information."""
        return {
            "model": "MiniFASNet",
            "threshold": self.threshold,
            "consistency_required": self.consistency_required,
        }