"""
Speech Emotion Recognition (SER) module using pre-trained wav2vec2 models.

This module provides both dimensional (valence/arousal) and categorical emotion
classification for audio segments.
"""

import logging
import threading
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray


class EmotionRecognizerLike(Protocol):
    """
    Protocol defining the interface for emotion recognizers.

    This protocol allows callers to work with either the real EmotionRecognizer
    or the DummyEmotionRecognizer without type-checker complaints.
    """

    def extract_emotion_dimensional(
        self, audio: NDArray[np.floating[Any]], sr: int
    ) -> dict[str, dict[str, Any]]:
        """
        Extract dimensional emotion features (valence, arousal, dominance).

        Args:
            audio: Audio samples as numpy array (mono)
            sr: Sample rate of the audio

        Returns:
            Dictionary with valence, arousal, dominance scores and levels.
        """
        ...

    def extract_emotion_categorical(
        self, audio: NDArray[np.floating[Any]], sr: int
    ) -> dict[str, dict[str, Any]]:
        """
        Extract categorical emotion classification.

        Args:
            audio: Audio samples as numpy array (mono)
            sr: Sample rate of the audio

        Returns:
            Dictionary with primary/secondary emotions and confidence scores.
        """
        ...


# Optional emotion dependencies - gracefully handle missing/broken packages
torch: Any
AutoModelForAudioClassification: Any
Wav2Vec2FeatureExtractor: Any
EMOTION_AVAILABLE: bool = False
try:
    # Suppress torchcodec/FFmpeg warnings from transformers' torch import chain.
    # These warnings are informational only and clutter the output without providing
    # actionable information for users of slower-whisper.
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")
    warnings.filterwarnings("ignore", message=".*FFmpeg.*")

    import torch
    from transformers import (  # type: ignore[no-redef]
        AutoModelForAudioClassification,
        Wav2Vec2FeatureExtractor,
    )

    EMOTION_AVAILABLE = True
except Exception:
    torch = cast(Any, None)
    AutoModelForAudioClassification = cast(Any, None)
    Wav2Vec2FeatureExtractor = cast(Any, None)

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    """Clamp a score to [0.0, 1.0] for JSON schema compliance."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class EmotionRecognizer:
    """
    Lazy-loading emotion recognition with model caching.

    Supports:
    - Dimensional: valence, arousal, dominance (MSP-Podcast dataset)
    - Categorical: angry, happy, sad, neutral, etc. (XLSR multilingual)
    """

    # Model specifications
    DIMENSIONAL_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    CATEGORICAL_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    # Minimum recommended segment length (seconds)
    MIN_SEGMENT_LENGTH = 0.5

    # Expected sampling rate for models (wav2vec2 standard)
    TARGET_SAMPLE_RATE = 16000

    def __init__(self):
        """Initialize recognizer with lazy loading."""
        if not EMOTION_AVAILABLE:
            raise RuntimeError(
                "Emotion recognition requires torch and transformers. "
                "Install with: uv sync --extra emotion"
            )
        self._dimensional_model = None
        self._dimensional_feature_extractor = None
        self._categorical_model = None
        self._categorical_feature_extractor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Thread-safety locks for lazy model loading
        self._dimensional_lock = threading.Lock()
        self._categorical_lock = threading.Lock()
        logger.info(f"EmotionRecognizer initialized on device: {self._device}")

    def _load_dimensional_model(self) -> None:
        """Lazy load dimensional emotion model (valence/arousal/dominance)."""
        if self._dimensional_model is None:
            with self._dimensional_lock:
                # Double-check after acquiring lock
                if self._dimensional_model is None:
                    from .cache import CachePaths

                    paths = CachePaths.from_env().ensure_dirs()
                    logger.info(f"Loading dimensional model: {self.DIMENSIONAL_MODEL}")
                    self._dimensional_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                        self.DIMENSIONAL_MODEL, cache_dir=str(paths.emotion_root)
                    )
                    self._dimensional_model = AutoModelForAudioClassification.from_pretrained(
                        self.DIMENSIONAL_MODEL, cache_dir=str(paths.emotion_root)
                    ).to(self._device)
                    self._dimensional_model.eval()
                    logger.info("Dimensional model loaded successfully")

    def _load_categorical_model(self) -> None:
        """Lazy load categorical emotion model."""
        if self._categorical_model is None:
            with self._categorical_lock:
                # Double-check after acquiring lock
                if self._categorical_model is None:
                    from .cache import CachePaths

                    paths = CachePaths.from_env().ensure_dirs()
                    logger.info(f"Loading categorical model: {self.CATEGORICAL_MODEL}")
                    self._categorical_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                        self.CATEGORICAL_MODEL, cache_dir=str(paths.emotion_root)
                    )
                    self._categorical_model = AutoModelForAudioClassification.from_pretrained(
                        self.CATEGORICAL_MODEL, cache_dir=str(paths.emotion_root)
                    ).to(self._device)
                    self._categorical_model.eval()
                    logger.info("Categorical model loaded successfully")

    def _validate_audio(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, bool]:
        """
        Validate and preprocess audio input.

        Args:
            audio: Audio samples as numpy array
            sr: Sample rate of the audio

        Returns:
            Tuple of (processed_audio, is_valid)
        """
        # Check if audio is too short
        duration = len(audio) / sr
        if duration < self.MIN_SEGMENT_LENGTH:
            logger.warning(
                f"Audio segment ({duration:.2f}s) is shorter than recommended "
                f"minimum ({self.MIN_SEGMENT_LENGTH}s). Results may be unreliable."
            )

        # Resample if needed
        if sr != self.TARGET_SAMPLE_RATE:
            logger.debug(f"Resampling from {sr}Hz to {self.TARGET_SAMPLE_RATE}Hz")
            # Simple resampling using numpy (for production, consider librosa.resample)
            audio = self._simple_resample(audio, sr, self.TARGET_SAMPLE_RATE)

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize to [-1, 1] range if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        return audio, True

    def _simple_resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> NDArray[np.float32]:
        """
        Resample audio using librosa with proper anti-aliasing.

        Uses sinc interpolation for high-quality resampling that preserves
        audio characteristics important for emotion recognition.
        """
        if orig_sr == target_sr:
            return cast(NDArray[np.float32], audio.astype(np.float32))

        try:
            import librosa

            # librosa.resample uses Kaiser window anti-aliasing by default
            resampled = librosa.resample(
                audio.astype(np.float32),
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type="soxr_hq",  # High-quality resampling
            )
            return cast(NDArray[np.float32], resampled.astype(np.float32))
        except ImportError:
            # Fallback to linear interpolation if librosa not available
            logger.warning(
                "librosa not available for resampling; using linear interpolation. "
                "Install with: uv sync --extra enrich-basic"
            )
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            resampled = np.interp(indices, np.arange(len(audio)), audio)
            return cast(NDArray[np.float32], resampled.astype(np.float32))

    def _classify_valence(self, score: float) -> str:
        """Classify valence score into level."""
        if score < 0.3:
            return "very_negative"
        if score < 0.4:
            return "negative"
        if score < 0.6:
            return "neutral"
        if score < 0.7:
            return "positive"
        return "very_positive"

    def _classify_arousal(self, score: float) -> str:
        """Classify arousal score into level."""
        if score < 0.3:
            return "very_low"
        if score < 0.4:
            return "low"
        if score < 0.6:
            return "medium"
        if score < 0.7:
            return "high"
        return "very_high"

    def _classify_dominance(self, score: float) -> str:
        """Classify dominance score into level."""
        if score < 0.3:
            return "very_submissive"
        if score < 0.4:
            return "submissive"
        if score < 0.6:
            return "neutral"
        if score < 0.7:
            return "dominant"
        return "very_dominant"

    def extract_emotion_dimensional(self, audio: np.ndarray, sr: int) -> dict[str, dict[str, Any]]:
        """
        Extract dimensional emotion features (valence, arousal, dominance).

        Args:
            audio: Audio samples as numpy array (mono)
            sr: Sample rate of the audio

        Returns:
            Dictionary with structure:
            {
                "valence": {"level": "negative", "score": 0.35},
                "arousal": {"level": "high", "score": 0.68},
                "dominance": {"level": "neutral", "score": 0.52}
            }

        Raises:
            ValueError: If audio validation fails
        """
        self._load_dimensional_model()

        # Validate and preprocess
        audio, is_valid = self._validate_audio(audio, sr)
        if not is_valid:
            raise ValueError("Audio validation failed")

        # Process audio through model
        assert self._dimensional_feature_extractor is not None
        assert self._dimensional_model is not None
        with torch.no_grad():
            inputs = self._dimensional_feature_extractor(
                audio, sampling_rate=self.TARGET_SAMPLE_RATE, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            outputs = self._dimensional_model(**inputs)
            predictions = outputs.logits.cpu().numpy()[0]

        # MSP-Dim model outputs: [arousal, dominance, valence]
        # Raw scores may slightly exceed [0, 1] range; clamp for schema compliance
        arousal_score = _clamp01(float(predictions[0]))
        dominance_score = _clamp01(float(predictions[1]))
        valence_score = _clamp01(float(predictions[2]))

        result = {
            "valence": {
                "level": self._classify_valence(valence_score),
                "score": round(valence_score, 3),
            },
            "arousal": {
                "level": self._classify_arousal(arousal_score),
                "score": round(arousal_score, 3),
            },
            "dominance": {
                "level": self._classify_dominance(dominance_score),
                "score": round(dominance_score, 3),
            },
        }

        logger.debug(f"Dimensional emotion: {result}")
        return result

    def extract_emotion_categorical(self, audio: np.ndarray, sr: int) -> dict[str, dict[str, Any]]:
        """
        Extract categorical emotion classification.

        Args:
            audio: Audio samples as numpy array (mono)
            sr: Sample rate of the audio

        Returns:
            Dictionary with structure:
            {
                "categorical": {
                    "primary": "frustrated",
                    "confidence": 0.81,
                    "secondary": "angry",
                    "secondary_confidence": 0.12,
                    "all_scores": {
                        "angry": 0.12,
                        "disgust": 0.03,
                        "fear": 0.02,
                        "happy": 0.01,
                        "neutral": 0.01,
                        "sad": 0.00,
                        ...
                    }
                }
            }

        Raises:
            ValueError: If audio validation fails
        """
        self._load_categorical_model()

        # Validate and preprocess
        audio, is_valid = self._validate_audio(audio, sr)
        if not is_valid:
            raise ValueError("Audio validation failed")

        # Process audio through model
        assert self._categorical_feature_extractor is not None
        assert self._categorical_model is not None
        with torch.no_grad():
            inputs = self._categorical_feature_extractor(
                audio, sampling_rate=self.TARGET_SAMPLE_RATE, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            outputs = self._categorical_model(**inputs)
            logits = outputs.logits.cpu()

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        # Get label names from model config
        id2label = self._categorical_model.config.id2label

        # Create scores dictionary
        all_scores = {id2label[i]: round(float(probs[i]), 3) for i in range(len(probs))}

        # Sort by confidence
        sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        primary_emotion, primary_conf = sorted_emotions[0]
        secondary_emotion, secondary_conf = (
            sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0.0)
        )

        result = {
            "categorical": {
                "primary": primary_emotion,
                "confidence": round(primary_conf, 3),
                "secondary": secondary_emotion,
                "secondary_confidence": round(secondary_conf, 3),
                "all_scores": all_scores,
            }
        }

        logger.debug(f"Categorical emotion: {primary_emotion} ({primary_conf:.2f})")
        return result


class DummyEmotionRecognizer:
    """Fallback emotion extractor that returns neutral scores."""

    def extract_emotion_dimensional(self, audio: np.ndarray, sr: int) -> dict[str, dict[str, Any]]:
        return {
            "valence": {"level": "neutral", "score": 0.5},
            "arousal": {"level": "medium", "score": 0.5},
            "dominance": {"level": "neutral", "score": 0.5},
        }

    def extract_emotion_categorical(self, audio: np.ndarray, sr: int) -> dict[str, dict[str, Any]]:
        return {
            "categorical": {
                "primary": "neutral",
                "confidence": 1.0,
                "secondary": None,
                "secondary_confidence": 0.0,
                "all_scores": {"neutral": 1.0},
            }
        }


# Singleton instance for easy access with thread-safe initialization
_recognizer_instance: EmotionRecognizerLike | None = None
_recognizer_lock = threading.Lock()


def get_emotion_recognizer() -> EmotionRecognizerLike:
    """
    Get or create the singleton EmotionRecognizer instance.

    Returns an object implementing EmotionRecognizerLike - either the real
    EmotionRecognizer (if dependencies are available) or DummyEmotionRecognizer.

    Thread-safe: Uses double-checked locking to avoid race conditions.
    """
    global _recognizer_instance
    if _recognizer_instance is None:
        with _recognizer_lock:
            # Double-check after acquiring lock
            if _recognizer_instance is None:
                if EMOTION_AVAILABLE:
                    _recognizer_instance = EmotionRecognizer()
                else:
                    logger.warning("Emotion dependencies unavailable; using dummy recognizer")
                    _recognizer_instance = DummyEmotionRecognizer()
    return _recognizer_instance


def cleanup_emotion_models() -> None:
    """
    Explicitly unload emotion recognition models from GPU/memory.

    Call this after batch processing to free GPU memory. The next call to
    get_emotion_recognizer() will re-initialize the singleton.

    Example:
        >>> from transcription.emotion import cleanup_emotion_models
        >>> # After batch processing...
        >>> cleanup_emotion_models()  # Free GPU memory
    """
    global _recognizer_instance
    with _recognizer_lock:
        if _recognizer_instance is not None:
            # Clear GPU cache if available
            try:
                if EMOTION_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared GPU cache")
            except Exception as e:
                logger.debug(f"Failed to clear GPU cache: {e}")

            _recognizer_instance = None
            logger.info("Emotion recognizer singleton released")


def extract_emotion_dimensional(audio: np.ndarray, sr: int) -> dict[str, dict[str, Any]]:
    """
    Convenience function to extract dimensional emotion features.

    Args:
        audio: Audio samples as numpy array (mono)
        sr: Sample rate of the audio

    Returns:
        Dictionary with valence, arousal, dominance scores and levels

    Example:
        >>> audio, sr = librosa.load("speech.wav", sr=16000)
        >>> result = extract_emotion_dimensional(audio, sr)
        >>> print(result)
        {
            "valence": {"level": "negative", "score": 0.42},
            "arousal": {"level": "very_high", "score": 0.78},
            "dominance": {"level": "neutral", "score": 0.51}
        }
    """
    recognizer = get_emotion_recognizer()
    return recognizer.extract_emotion_dimensional(audio, sr)


def extract_emotion_categorical(audio: np.ndarray, sr: int) -> dict[str, dict[str, Any]]:
    """
    Convenience function to extract categorical emotion classification.

    Args:
        audio: Audio samples as numpy array (mono)
        sr: Sample rate of the audio

    Returns:
        Dictionary with primary/secondary emotions and confidence scores

    Example:
        >>> audio, sr = librosa.load("speech.wav", sr=16000)
        >>> result = extract_emotion_categorical(audio, sr)
        >>> print(result)
        {
            "categorical": {
                "primary": "frustrated",
                "confidence": 0.81,
                "secondary": "angry",
                "secondary_confidence": 0.12,
                "all_scores": {...}
            }
        }
    """
    recognizer = get_emotion_recognizer()
    return recognizer.extract_emotion_categorical(audio, sr)


# Export public API
__all__ = [
    "DummyEmotionRecognizer",
    "EmotionRecognizer",
    "EmotionRecognizerLike",
    "cleanup_emotion_models",
    "extract_emotion_categorical",
    "extract_emotion_dimensional",
    "get_emotion_recognizer",
]
