"""Face recognition module using UniFace."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from uniface.recognition import MobileFace, AdaFace, ArcFace
from uniface.face_utils import compute_similarity

from .config import get_config

logger = logging.getLogger(__name__)


class RecognitionError(Exception):
    """Recognition-related errors."""
    pass


class FaceRecognizer:
    """Face recognition with enrollment and matching.

    Requirements:
    - Load enrolled embedding
    - Generate embedding from face
    - Calculate similarity
    - Configurable threshold
    - Display/debug similarity
    - Return match/non-match
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
        model_name = config.get("recognition.model", "MobileFace")

        # Initialize recognizer
        if model_name.upper() == "MOBILEFACE":
            self.recognizer = MobileFace(providers=['CPUExecutionProvider'])
        elif model_name.upper() == "ADAFACE":
            self.recognizer = AdaFace(providers=['CPUExecutionProvider'])
        elif model_name.upper() == "ARCFACE":
            self.recognizer = ArcFace(providers=['CPUExecutionProvider'])
        else:
            logger.warning(f"Unknown recognizer {model_name}, using MobileFace")
            self.recognizer = MobileFace(providers=['CPUExecutionProvider'])

        self.threshold = config.get("recognition.threshold", 0.7)
        logger.info(f"Face recognizer initialized: {model_name} with threshold {self.threshold}")
        self._initialized = True

    def extract_embedding(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """Extract face embedding from image.

        Args:
            image: Input image (BGR)
            landmarks: Facial landmarks

        Returns:
            Normalized embedding vector
        """
        start_time = time.time()

        try:
            embedding = self.recognizer.get_normalized_embedding(image, landmarks)

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Embedding extraction latency: {latency_ms:.1f}ms")

            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise RecognitionError(f"Failed to extract embedding: {e}")

    def compare(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compare two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity (0 to 1)
        """
        similarity = compute_similarity(embedding1, embedding2)
        return float(similarity)

    def verify(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        enrolled_embedding: np.ndarray,
    ) -> tuple[bool, float]:
        """Verify face against enrolled embedding.

        Args:
            image: Input image
            landmarks: Facial landmarks
            enrolled_embedding: Stored enrollment embedding

        Returns:
            (is_match, similarity_score)
        """
        extracted = self.extract_embedding(image, landmarks)
        similarity = self.compare(extracted, enrolled_embedding)
        is_match = similarity >= self.threshold

        logger.info(f"Recognition: similarity={similarity:.3f}, threshold={self.threshold}, match={is_match}")

        return is_match, similarity

    def measure_latency(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        num_runs: int = 10,
    ) -> float:
        """Measure average recognition latency."""
        latencies = []

        for _ in range(num_runs):
            start = time.time()
            _ = self.extract_embedding(image, landmarks)
            latencies.append((time.time() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"Average recognition latency: {avg_latency:.1f}ms")
        return avg_latency

    def get_info(self) -> dict:
        """Get recognizer information."""
        return {
            "model": self.recognizer.__class__.__name__,
            "threshold": self.threshold,
        }
