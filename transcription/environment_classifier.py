"""Environment classification from audio health metrics.

This module provides audio environment classification based on
existing audio_health metrics:
- clean: High quality, good SNR
- noisy: Low SNR, background noise
- muffled: Low spectral centroid
- hissy: High centroid with low quality
- clipping: Significant clipping detected

Classification uses thresholds on audio health metrics and can be
configured for different sensitivity levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EnvironmentTag = Literal["clean", "noisy", "muffled", "hissy", "clipping", "unknown"]


@dataclass(slots=True)
class EnvironmentState:
    """Environment classification result.

    Attributes:
        tag: Primary environment classification.
        confidence: Classification confidence (0.0-1.0).
        contributing_factors: Factors that contributed to classification.
        secondary_tags: Additional applicable tags.
        quality_score: Overall audio quality score (0.0-1.0).
    """

    tag: EnvironmentTag = "unknown"
    confidence: float = 0.0
    contributing_factors: list[str] = field(default_factory=list)
    secondary_tags: list[EnvironmentTag] = field(default_factory=list)
    quality_score: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag": self.tag,
            "confidence": self.confidence,
            "contributing_factors": list(self.contributing_factors),
            "secondary_tags": list(self.secondary_tags),
            "quality_score": self.quality_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentState:
        """Create from dictionary."""
        return cls(
            tag=data.get("tag", "unknown"),
            confidence=data.get("confidence", 0.0),
            contributing_factors=list(data.get("contributing_factors", [])),
            secondary_tags=list(data.get("secondary_tags", [])),
            quality_score=data.get("quality_score", 0.5),
        )


@dataclass(slots=True)
class EnvironmentClassifierConfig:
    """Configuration for environment classification.

    Attributes:
        clean_quality_threshold: Minimum quality score for "clean".
        clean_snr_threshold: Minimum SNR (dB) for "clean".
        noisy_snr_threshold: Maximum SNR (dB) for "noisy".
        muffled_centroid_threshold: Maximum centroid (Hz) for "muffled".
        hissy_centroid_threshold: Minimum centroid (Hz) for "hissy".
        hissy_quality_threshold: Maximum quality for "hissy".
        clipping_ratio_threshold: Minimum clipping ratio for "clipping".
        enabled: Whether classification is enabled.
    """

    clean_quality_threshold: float = 0.7
    clean_snr_threshold: float = 20.0
    noisy_snr_threshold: float = 10.0
    muffled_centroid_threshold: float = 500.0
    hissy_centroid_threshold: float = 4000.0
    hissy_quality_threshold: float = 0.6
    clipping_ratio_threshold: float = 0.05
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clean_quality_threshold": self.clean_quality_threshold,
            "clean_snr_threshold": self.clean_snr_threshold,
            "noisy_snr_threshold": self.noisy_snr_threshold,
            "muffled_centroid_threshold": self.muffled_centroid_threshold,
            "hissy_centroid_threshold": self.hissy_centroid_threshold,
            "hissy_quality_threshold": self.hissy_quality_threshold,
            "clipping_ratio_threshold": self.clipping_ratio_threshold,
            "enabled": self.enabled,
        }


@dataclass(slots=True)
class AudioHealthMetrics:
    """Input metrics for environment classification.

    These metrics are typically extracted by the audio_health module.

    Attributes:
        quality_score: Overall quality score (0.0-1.0).
        snr_db: Signal-to-noise ratio in decibels.
        spectral_centroid_hz: Spectral centroid frequency in Hz.
        clipping_ratio: Ratio of clipped samples (0.0-1.0).
        noise_floor_db: Estimated noise floor in dB.
        peak_db: Peak amplitude in dB.
        rms_db: RMS amplitude in dB.
    """

    quality_score: float = 0.5
    snr_db: float | None = None
    spectral_centroid_hz: float | None = None
    clipping_ratio: float = 0.0
    noise_floor_db: float | None = None
    peak_db: float | None = None
    rms_db: float | None = None

    @classmethod
    def from_audio_health_snapshot(cls, snapshot: dict[str, Any]) -> AudioHealthMetrics:
        """Create from AudioHealthSnapshot dict.

        Args:
            snapshot: AudioHealthSnapshot as dictionary.

        Returns:
            AudioHealthMetrics instance.
        """
        return cls(
            quality_score=snapshot.get("quality_score", 0.5),
            snr_db=snapshot.get("snr_db"),
            spectral_centroid_hz=snapshot.get("spectral_centroid_hz"),
            clipping_ratio=snapshot.get("clipping_ratio", 0.0),
            noise_floor_db=snapshot.get("noise_floor_db"),
            peak_db=snapshot.get("peak_db"),
            rms_db=snapshot.get("rms_db"),
        )


class EnvironmentClassifier:
    """Audio environment classifier.

    Classifies audio environment based on audio health metrics.

    Example:
        >>> classifier = EnvironmentClassifier()
        >>> metrics = AudioHealthMetrics(
        ...     quality_score=0.85,
        ...     snr_db=25.0,
        ...     spectral_centroid_hz=2000.0,
        ...     clipping_ratio=0.0,
        ... )
        >>> result = classifier.classify(metrics)
        >>> print(result.tag)
        "clean"
    """

    def __init__(self, config: EnvironmentClassifierConfig | None = None):
        """Initialize classifier with configuration.

        Args:
            config: Classification configuration.
        """
        self.config = config or EnvironmentClassifierConfig()

    def classify(self, metrics: AudioHealthMetrics) -> EnvironmentState:
        """Classify audio environment from metrics.

        Args:
            metrics: Audio health metrics.

        Returns:
            EnvironmentState with classification results.
        """
        if not self.config.enabled:
            return EnvironmentState(tag="unknown", confidence=0.0)

        factors: list[str] = []
        tags_with_scores: list[tuple[EnvironmentTag, float, str]] = []

        # Check for clipping (highest priority issue)
        if metrics.clipping_ratio >= self.config.clipping_ratio_threshold:
            score = min(metrics.clipping_ratio / 0.1, 1.0)  # Normalize to 0-1
            tags_with_scores.append(("clipping", score, f"clipping_ratio={metrics.clipping_ratio:.2%}"))

        # Check for noisy environment
        if metrics.snr_db is not None and metrics.snr_db < self.config.noisy_snr_threshold:
            score = 1.0 - (metrics.snr_db / self.config.noisy_snr_threshold)
            tags_with_scores.append(("noisy", score, f"snr_db={metrics.snr_db:.1f}"))

        # Check for muffled audio
        if (
            metrics.spectral_centroid_hz is not None
            and metrics.spectral_centroid_hz < self.config.muffled_centroid_threshold
        ):
            score = 1.0 - (metrics.spectral_centroid_hz / self.config.muffled_centroid_threshold)
            tags_with_scores.append(("muffled", score, f"spectral_centroid={metrics.spectral_centroid_hz:.0f}Hz"))

        # Check for hissy audio
        if (
            metrics.spectral_centroid_hz is not None
            and metrics.spectral_centroid_hz > self.config.hissy_centroid_threshold
            and metrics.quality_score < self.config.hissy_quality_threshold
        ):
            centroid_score = (
                metrics.spectral_centroid_hz - self.config.hissy_centroid_threshold
            ) / 2000.0
            quality_factor = 1.0 - metrics.quality_score
            score = min(centroid_score + quality_factor, 1.0) / 2
            tags_with_scores.append(("hissy", score, "high_centroid+low_quality"))

        # Check for clean audio
        is_clean = (
            metrics.quality_score >= self.config.clean_quality_threshold
            and (metrics.snr_db is None or metrics.snr_db >= self.config.clean_snr_threshold)
            and metrics.clipping_ratio < self.config.clipping_ratio_threshold
        )

        if is_clean:
            score = metrics.quality_score
            tags_with_scores.append(("clean", score, f"quality_score={metrics.quality_score:.2f}"))

        # Determine primary tag (highest score)
        if not tags_with_scores:
            return EnvironmentState(
                tag="unknown",
                confidence=0.5,
                contributing_factors=["insufficient_metrics"],
                quality_score=metrics.quality_score,
            )

        # Sort by score descending
        tags_with_scores.sort(key=lambda x: x[1], reverse=True)

        primary_tag, primary_score, primary_factor = tags_with_scores[0]
        factors.append(primary_factor)

        # Collect secondary tags
        secondary_tags: list[EnvironmentTag] = []
        for tag, score, factor in tags_with_scores[1:]:
            if score > 0.3:  # Include if reasonably confident
                secondary_tags.append(tag)
                factors.append(factor)

        return EnvironmentState(
            tag=primary_tag,
            confidence=primary_score,
            contributing_factors=factors,
            secondary_tags=secondary_tags,
            quality_score=metrics.quality_score,
        )

    def classify_from_snapshot(self, snapshot: dict[str, Any]) -> EnvironmentState:
        """Classify from AudioHealthSnapshot dictionary.

        Convenience method for integration with audio_health module.

        Args:
            snapshot: AudioHealthSnapshot as dictionary.

        Returns:
            EnvironmentState with classification results.
        """
        metrics = AudioHealthMetrics.from_audio_health_snapshot(snapshot)
        return self.classify(metrics)


def classify_environment(
    quality_score: float = 0.5,
    snr_db: float | None = None,
    spectral_centroid_hz: float | None = None,
    clipping_ratio: float = 0.0,
    **kwargs: Any,
) -> EnvironmentState:
    """Convenience function for environment classification.

    Args:
        quality_score: Overall quality score (0.0-1.0).
        snr_db: Signal-to-noise ratio in dB.
        spectral_centroid_hz: Spectral centroid in Hz.
        clipping_ratio: Ratio of clipped samples.
        **kwargs: Additional metrics (ignored).

    Returns:
        EnvironmentState with classification results.
    """
    metrics = AudioHealthMetrics(
        quality_score=quality_score,
        snr_db=snr_db,
        spectral_centroid_hz=spectral_centroid_hz,
        clipping_ratio=clipping_ratio,
    )
    classifier = EnvironmentClassifier()
    return classifier.classify(metrics)


__all__ = [
    "EnvironmentClassifier",
    "EnvironmentClassifierConfig",
    "EnvironmentState",
    "EnvironmentTag",
    "AudioHealthMetrics",
    "classify_environment",
]
