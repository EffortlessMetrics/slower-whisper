"""Audio health analysis for streaming transcription.

Provides real-time audio quality metrics for monitoring and adaptive processing.
All operations are designed to be fast (<1ms per chunk) and bounded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AudioHealthSnapshot",
    "analyze_chunk_health",
    "AudioHealthAggregator",
]

# Constants for audio analysis
_INT16_MAX = 32767
_INT16_MIN = -32768
_CLIPPING_THRESHOLD = 0.99  # Samples above this ratio of max are considered clipped
_SPEECH_MIN_RMS = 0.01  # Minimum RMS for speech detection
_SPEECH_MAX_RMS = 0.9  # Maximum RMS (above this is likely noise/clipping)
_SPEECH_MIN_CENTROID = 200.0  # Minimum spectral centroid for speech (Hz)
_SPEECH_MAX_CENTROID = 4000.0  # Maximum spectral centroid for speech (Hz)
_SNR_PERCENTILE_HIGH = 90  # Top percentile for SNR calculation
_SNR_PERCENTILE_LOW = 10  # Bottom percentile for SNR calculation
_MIN_SAMPLES_FOR_FFT = 64  # Minimum samples needed for meaningful FFT


@dataclass(frozen=True, slots=True)
class AudioHealthSnapshot:
    """Snapshot of audio health metrics for a single chunk.

    Attributes:
        clipping_ratio: Ratio of clipped samples (0.0-1.0). Higher means more distortion.
        rms_energy: Root mean square energy level (0.0-1.0 for normalized audio).
        snr_proxy: Signal-to-noise ratio proxy in dB. Higher is better.
        spectral_centroid: Spectral centroid in Hz, or None if not computed.
            Speech typically has centroid between 200-4000 Hz.
        quality_score: Overall quality score (0.0-1.0). Higher is better.
        is_speech_likely: Whether the chunk likely contains speech based on
            energy and spectral characteristics.
    """

    clipping_ratio: float
    rms_energy: float
    snr_proxy: float
    spectral_centroid: float | None
    quality_score: float
    is_speech_likely: bool


def analyze_chunk_health(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
) -> AudioHealthSnapshot:
    """Analyze audio chunk health metrics.

    Converts PCM bytes to float array and computes various audio quality metrics
    including clipping detection, energy levels, SNR proxy, and spectral centroid.

    Args:
        pcm_bytes: Raw PCM audio data as 16-bit signed little-endian bytes.
        sample_rate: Sample rate in Hz. Defaults to 16000.

    Returns:
        AudioHealthSnapshot with computed metrics.

    Note:
        Designed to run in <1ms for typical chunk sizes (10-100ms of audio).
        Empty or very short chunks return a snapshot with default/zero values.
    """
    # Handle empty input
    if not pcm_bytes or len(pcm_bytes) < 2:
        return AudioHealthSnapshot(
            clipping_ratio=0.0,
            rms_energy=0.0,
            snr_proxy=0.0,
            spectral_centroid=None,
            quality_score=0.0,
            is_speech_likely=False,
        )

    # Convert PCM bytes (16-bit signed LE) to int16 numpy array
    samples_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    n_samples = len(samples_int16)

    if n_samples == 0:
        return AudioHealthSnapshot(
            clipping_ratio=0.0,
            rms_energy=0.0,
            snr_proxy=0.0,
            spectral_centroid=None,
            quality_score=0.0,
            is_speech_likely=False,
        )

    # Convert to float32 normalized to [-1.0, 1.0]
    samples_float = samples_int16.astype(np.float32) / _INT16_MAX

    # 1. Detect clipping (samples at or near +/- max value)
    clipping_threshold_int = int(_CLIPPING_THRESHOLD * _INT16_MAX)
    clipped_count = np.sum(np.abs(samples_int16) >= clipping_threshold_int)
    clipping_ratio = float(clipped_count / n_samples)

    # 2. Compute RMS energy
    rms_energy = float(np.sqrt(np.mean(samples_float**2)))

    # 3. Compute SNR proxy (ratio of top 10% energy to bottom 10%)
    snr_proxy = _compute_snr_proxy(samples_float)

    # 4. Compute spectral centroid using FFT (if enough samples)
    spectral_centroid: float | None = None
    if n_samples >= _MIN_SAMPLES_FOR_FFT:
        spectral_centroid = _compute_spectral_centroid(samples_float, sample_rate)

    # 5. Compute quality score (weighted combination)
    quality_score = _compute_quality_score(
        clipping_ratio=clipping_ratio,
        rms_energy=rms_energy,
        snr_proxy=snr_proxy,
        spectral_centroid=spectral_centroid,
    )

    # 6. Determine speech likelihood
    is_speech_likely = _is_speech_likely(
        rms_energy=rms_energy,
        spectral_centroid=spectral_centroid,
    )

    return AudioHealthSnapshot(
        clipping_ratio=clipping_ratio,
        rms_energy=rms_energy,
        snr_proxy=snr_proxy,
        spectral_centroid=spectral_centroid,
        quality_score=quality_score,
        is_speech_likely=is_speech_likely,
    )


def _compute_snr_proxy(samples: np.ndarray) -> float:
    """Compute SNR proxy as ratio of high to low energy percentiles.

    Args:
        samples: Float32 audio samples normalized to [-1.0, 1.0].

    Returns:
        SNR proxy in dB. Returns 0.0 if computation fails.
    """
    # Compute energy of samples
    energy = samples**2

    # Get percentiles
    high_energy = np.percentile(energy, _SNR_PERCENTILE_HIGH)
    low_energy = np.percentile(energy, _SNR_PERCENTILE_LOW)

    # Avoid division by zero and log of zero
    if low_energy < 1e-10:
        low_energy = 1e-10
    if high_energy < 1e-10:
        return 0.0

    # Convert ratio to dB
    snr_db = 10.0 * np.log10(high_energy / low_energy)

    # Clamp to reasonable range
    return float(np.clip(snr_db, 0.0, 60.0))


def _compute_spectral_centroid(samples: np.ndarray, sample_rate: int) -> float | None:
    """Compute spectral centroid using FFT.

    The spectral centroid is the "center of mass" of the spectrum and indicates
    where the "brightness" of the sound is concentrated.

    Args:
        samples: Float32 audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        Spectral centroid in Hz, or None if computation fails.
    """
    n_samples = len(samples)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n_samples)
    windowed = samples * window

    # Compute FFT magnitude spectrum (only positive frequencies)
    fft_result = np.fft.rfft(windowed)
    magnitude = np.abs(fft_result)

    # Compute frequency bins
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

    # Compute spectral centroid: sum(f * mag) / sum(mag)
    total_magnitude = np.sum(magnitude)
    if total_magnitude < 1e-10:
        return None

    centroid = float(np.sum(freqs * magnitude) / total_magnitude)

    # Clamp to reasonable range (0 to Nyquist)
    nyquist = sample_rate / 2.0
    return float(np.clip(centroid, 0.0, nyquist))


def _compute_quality_score(
    clipping_ratio: float,
    rms_energy: float,
    snr_proxy: float,
    spectral_centroid: float | None,
) -> float:
    """Compute overall quality score from individual metrics.

    Weights:
        - Clipping penalty: 30% (clipping severely degrades quality)
        - Energy score: 25% (too low = silence, too high = distortion)
        - SNR score: 30% (higher SNR = cleaner signal)
        - Spectral score: 15% (reasonable centroid suggests good audio)

    Args:
        clipping_ratio: Ratio of clipped samples (0.0-1.0).
        rms_energy: RMS energy level.
        snr_proxy: SNR proxy in dB.
        spectral_centroid: Spectral centroid in Hz, or None.

    Returns:
        Quality score from 0.0 (worst) to 1.0 (best).
    """
    # Clipping score: 1.0 = no clipping, 0.0 = severe clipping
    clipping_score = 1.0 - min(clipping_ratio * 10.0, 1.0)  # 10% clipping = 0 score

    # Energy score: optimal around 0.1-0.3 RMS
    if rms_energy < 0.005:
        # Too quiet (near silence)
        energy_score = rms_energy / 0.005 * 0.3
    elif rms_energy < 0.1:
        # Quiet but acceptable
        energy_score = 0.3 + (rms_energy - 0.005) / 0.095 * 0.7
    elif rms_energy < 0.5:
        # Good range
        energy_score = 1.0
    else:
        # Too loud, potential distortion
        energy_score = max(0.0, 1.0 - (rms_energy - 0.5) * 2.0)

    # SNR score: map 0-40 dB to 0-1
    snr_score = min(snr_proxy / 40.0, 1.0)

    # Spectral score: reasonable centroid for speech/music
    if spectral_centroid is None:
        spectral_score = 0.5  # Neutral if not computed
    elif _SPEECH_MIN_CENTROID <= spectral_centroid <= _SPEECH_MAX_CENTROID:
        spectral_score = 1.0
    elif spectral_centroid < _SPEECH_MIN_CENTROID:
        # Too low (rumble, DC offset)
        spectral_score = max(0.0, spectral_centroid / _SPEECH_MIN_CENTROID)
    else:
        # Too high (hiss, noise)
        spectral_score = max(0.0, 1.0 - (spectral_centroid - _SPEECH_MAX_CENTROID) / 4000.0)

    # Weighted combination
    quality = 0.30 * clipping_score + 0.25 * energy_score + 0.30 * snr_score + 0.15 * spectral_score

    return float(np.clip(quality, 0.0, 1.0))


def _is_speech_likely(
    rms_energy: float,
    spectral_centroid: float | None,
) -> bool:
    """Determine if chunk likely contains speech.

    Uses simple heuristics based on energy and spectral centroid.
    Speech typically has:
    - Moderate energy (not silence, not clipped noise)
    - Spectral centroid between 200-4000 Hz

    Args:
        rms_energy: RMS energy level.
        spectral_centroid: Spectral centroid in Hz, or None.

    Returns:
        True if speech is likely, False otherwise.
    """
    # Check energy bounds
    if rms_energy < _SPEECH_MIN_RMS or rms_energy > _SPEECH_MAX_RMS:
        return False

    # Check spectral centroid if available
    if spectral_centroid is not None:
        if spectral_centroid < _SPEECH_MIN_CENTROID:
            return False
        if spectral_centroid > _SPEECH_MAX_CENTROID:
            return False

    return True


class AudioHealthAggregator:
    """Aggregates audio health snapshots over a rolling window.

    Maintains a fixed-size window of recent snapshots and provides
    aggregated statistics for trend analysis and adaptive processing.

    Example:
        >>> aggregator = AudioHealthAggregator(window_size=10)
        >>> for chunk in audio_chunks:
        ...     snapshot = analyze_chunk_health(chunk)
        ...     aggregator.add_snapshot(snapshot)
        ...     if aggregator.get_aggregate().quality_score < 0.5:
        ...         print("Warning: audio quality degraded")
    """

    def __init__(self, window_size: int = 10) -> None:
        """Initialize the aggregator.

        Args:
            window_size: Number of snapshots to keep in the rolling window.
                Must be at least 1. Defaults to 10.

        Raises:
            ValueError: If window_size is less than 1.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}")

        self._window_size = window_size
        self._snapshots: deque[AudioHealthSnapshot] = deque(maxlen=window_size)

    @property
    def window_size(self) -> int:
        """Return the configured window size."""
        return self._window_size

    @property
    def count(self) -> int:
        """Return the current number of snapshots in the window."""
        return len(self._snapshots)

    def add_snapshot(self, snapshot: AudioHealthSnapshot) -> None:
        """Add a snapshot to the rolling window.

        If the window is full, the oldest snapshot is automatically removed.

        Args:
            snapshot: AudioHealthSnapshot to add.
        """
        self._snapshots.append(snapshot)

    def get_aggregate(self) -> AudioHealthSnapshot:
        """Compute aggregate statistics over the current window.

        Returns the average of all metrics in the window. For is_speech_likely,
        returns True if more than half of the snapshots indicate speech.

        Returns:
            AudioHealthSnapshot with averaged metrics. If the window is empty,
            returns a snapshot with zero/default values.
        """
        if not self._snapshots:
            return AudioHealthSnapshot(
                clipping_ratio=0.0,
                rms_energy=0.0,
                snr_proxy=0.0,
                spectral_centroid=None,
                quality_score=0.0,
                is_speech_likely=False,
            )

        snapshots: Sequence[AudioHealthSnapshot] = list(self._snapshots)
        n = len(snapshots)

        # Average scalar metrics
        avg_clipping = sum(s.clipping_ratio for s in snapshots) / n
        avg_rms = sum(s.rms_energy for s in snapshots) / n
        avg_snr = sum(s.snr_proxy for s in snapshots) / n
        avg_quality = sum(s.quality_score for s in snapshots) / n

        # Average spectral centroid (only over non-None values)
        centroids = [s.spectral_centroid for s in snapshots if s.spectral_centroid is not None]
        avg_centroid: float | None = None
        if centroids:
            avg_centroid = sum(centroids) / len(centroids)

        # Majority vote for speech likelihood
        speech_count = sum(1 for s in snapshots if s.is_speech_likely)
        is_speech = speech_count > n / 2

        return AudioHealthSnapshot(
            clipping_ratio=avg_clipping,
            rms_energy=avg_rms,
            snr_proxy=avg_snr,
            spectral_centroid=avg_centroid,
            quality_score=avg_quality,
            is_speech_likely=is_speech,
        )

    def reset(self) -> None:
        """Clear all snapshots from the window."""
        self._snapshots.clear()
