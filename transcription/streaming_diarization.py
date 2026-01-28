"""
Incremental diarization hooks for streaming pipelines (#86).

This module provides pluggable diarization backends for the WebSocket streaming
protocol. Diarization hooks accept accumulated audio and return speaker assignments
for integration with the `WebSocketStreamingSession`.

Key features:
- Async-compatible hooks matching `DiarizationHookProtocol`
- PyAnnote-based incremental diarization (full re-diarization on accumulated audio)
- Sliding window diarization for memory-constrained scenarios
- Graceful degradation when pyannote is unavailable

Usage:
    >>> from transcription.streaming_diarization import create_pyannote_hook
    >>> from transcription.streaming_ws import WebSocketStreamingSession, WebSocketSessionConfig
    >>>
    >>> # Create diarization hook
    >>> diarization_hook = create_pyannote_hook(device="cuda")
    >>>
    >>> # Configure session with diarization enabled
    >>> config = WebSocketSessionConfig(
    ...     enable_diarization=True,
    ...     diarization_interval_sec=30.0,
    ... )
    >>>
    >>> # Create session with hook
    >>> session = WebSocketStreamingSession(config=config, diarization_hook=diarization_hook)

See docs/STREAMING_ARCHITECTURE.md for protocol specification.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .streaming_ws import SpeakerAssignment

if TYPE_CHECKING:
    from .diarization import Diarizer

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(slots=True)
class IncrementalDiarizationConfig:
    """
    Configuration for incremental diarization hooks.

    Attributes:
        device: Device for diarization inference ("cuda", "cpu", or "auto").
        min_speakers: Minimum expected speakers (None = auto-detect).
        max_speakers: Maximum expected speakers (None = auto-detect).
        min_audio_duration_sec: Minimum audio duration before running diarization.
                               Helps avoid noisy results on very short clips.
        use_sliding_window: If True, only process the most recent audio window
                           instead of full accumulated buffer.
        window_duration_sec: Duration of sliding window in seconds (when enabled).
        window_overlap_sec: Overlap between sliding windows in seconds.
    """

    device: str = "auto"
    min_speakers: int | None = None
    max_speakers: int | None = None
    min_audio_duration_sec: float = 5.0
    use_sliding_window: bool = False
    window_duration_sec: float = 60.0
    window_overlap_sec: float = 10.0


# =============================================================================
# Type Aliases
# =============================================================================

# Async hook signature matching DiarizationHookProtocol in streaming_ws.py
DiarizationHook = Callable[[bytes, int], "asyncio.Future[list[SpeakerAssignment]]"]


# =============================================================================
# PyAnnote Hook Implementation
# =============================================================================


class PyAnnoteIncrementalDiarizer:
    """
    Incremental diarization using pyannote.audio.

    This hook wraps the batch `Diarizer` class and runs diarization on
    accumulated audio. While pyannote doesn't have true online diarization,
    this approach provides periodic speaker updates during streaming.

    Implementation notes:
    - Each call re-runs diarization on the full audio buffer
    - Speaker IDs are normalized to spk_N format
    - Thread-safe lazy initialization of pyannote pipeline
    - Temporary WAV files are used for pyannote input

    Example:
        >>> diarizer = PyAnnoteIncrementalDiarizer(device="cuda")
        >>> assignments = await diarizer(audio_buffer, sample_rate=16000)
        >>> for a in assignments:
        ...     print(f"[{a.start:.2f}s - {a.end:.2f}s] {a.speaker_id}")
    """

    def __init__(
        self,
        config: IncrementalDiarizationConfig | None = None,
    ) -> None:
        """
        Initialize the PyAnnote incremental diarizer.

        Args:
            config: Configuration for diarization behavior.
                   Defaults to IncrementalDiarizationConfig().
        """
        self.config = config or IncrementalDiarizationConfig()
        self._diarizer: Diarizer | None = None
        self._speaker_map: dict[str, int] = {}  # Maps backend IDs to normalized indices
        self._initialized = False

    def _ensure_diarizer(self) -> Diarizer:
        """Lazy-initialize the pyannote Diarizer on first use."""
        if self._diarizer is not None:
            return self._diarizer

        from .diarization import Diarizer

        self._diarizer = Diarizer(
            device=self.config.device,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )
        self._initialized = True
        logger.info(
            "Initialized PyAnnote diarizer: device=%s, min_speakers=%s, max_speakers=%s",
            self.config.device,
            self.config.min_speakers,
            self.config.max_speakers,
        )
        return self._diarizer

    def _normalize_speaker_id(self, raw_id: str) -> str:
        """
        Normalize backend speaker IDs to spk_N format.

        Maintains consistent mapping across incremental updates.

        Args:
            raw_id: Raw speaker ID from pyannote (e.g., "SPEAKER_00").

        Returns:
            Normalized speaker ID (e.g., "spk_0").
        """
        if raw_id not in self._speaker_map:
            self._speaker_map[raw_id] = len(self._speaker_map)
        return f"spk_{self._speaker_map[raw_id]}"

    def _audio_buffer_to_wav(self, audio_buffer: bytes, sample_rate: int) -> Path:
        """
        Convert raw PCM audio buffer to temporary WAV file.

        Args:
            audio_buffer: Raw PCM bytes (16-bit mono).
            sample_rate: Sample rate in Hz.

        Returns:
            Path to temporary WAV file (caller must clean up).
        """
        # Create temporary WAV file
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with wave.open(wav_path, "wb") as wav:
            wav.setnchannels(1)  # mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_buffer)

        return Path(wav_path)

    def _get_audio_duration_sec(self, audio_buffer: bytes, sample_rate: int) -> float:
        """Calculate audio duration from buffer size."""
        bytes_per_sample = 2  # 16-bit
        num_samples = len(audio_buffer) // bytes_per_sample
        return num_samples / sample_rate

    async def __call__(
        self,
        audio_buffer: bytes,
        sample_rate: int,
    ) -> list[SpeakerAssignment]:
        """
        Run diarization on accumulated audio buffer.

        This method is async-compatible for integration with the WebSocket
        streaming protocol. The actual diarization runs in a thread pool
        to avoid blocking the event loop.

        Args:
            audio_buffer: Raw PCM audio bytes (16-bit mono).
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of SpeakerAssignment objects covering the audio duration.
            Returns empty list if audio is too short or diarization fails.
        """
        # Check minimum duration
        duration = self._get_audio_duration_sec(audio_buffer, sample_rate)
        if duration < self.config.min_audio_duration_sec:
            logger.debug(
                "Audio duration %.2fs < minimum %.2fs, skipping diarization",
                duration,
                self.config.min_audio_duration_sec,
            )
            return []

        # Determine which portion of audio to process
        if self.config.use_sliding_window:
            audio_buffer, offset = self._apply_sliding_window(audio_buffer, sample_rate)
        else:
            offset = 0.0

        # Run diarization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            assignments = await loop.run_in_executor(
                None,
                self._run_diarization_sync,
                audio_buffer,
                sample_rate,
                offset,
            )
            return assignments
        except Exception as e:
            logger.warning("Incremental diarization failed: %s", e)
            return []

    def _apply_sliding_window(
        self,
        audio_buffer: bytes,
        sample_rate: int,
    ) -> tuple[bytes, float]:
        """
        Extract sliding window from audio buffer.

        Args:
            audio_buffer: Full accumulated audio.
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (window_buffer, time_offset).
        """
        bytes_per_sample = 2
        bytes_per_second = sample_rate * bytes_per_sample
        window_bytes = int(self.config.window_duration_sec * bytes_per_second)

        if len(audio_buffer) <= window_bytes:
            return audio_buffer, 0.0

        # Take the most recent window
        offset_bytes = len(audio_buffer) - window_bytes
        offset_seconds = offset_bytes / bytes_per_second
        return audio_buffer[offset_bytes:], offset_seconds

    def _run_diarization_sync(
        self,
        audio_buffer: bytes,
        sample_rate: int,
        time_offset: float,
    ) -> list[SpeakerAssignment]:
        """
        Synchronous diarization execution (runs in thread pool).

        Args:
            audio_buffer: Raw PCM audio bytes.
            sample_rate: Sample rate in Hz.
            time_offset: Time offset to add to all timestamps.

        Returns:
            List of SpeakerAssignment objects.
        """
        wav_path = None
        try:
            # Initialize diarizer (lazy)
            diarizer = self._ensure_diarizer()

            # Write audio to temporary WAV
            wav_path = self._audio_buffer_to_wav(audio_buffer, sample_rate)

            # Run diarization
            speaker_turns = diarizer.run(wav_path)

            # Convert to SpeakerAssignment with normalized IDs
            assignments = []
            for turn in speaker_turns:
                assignments.append(
                    SpeakerAssignment(
                        start=turn.start + time_offset,
                        end=turn.end + time_offset,
                        speaker_id=self._normalize_speaker_id(turn.speaker_id),
                        confidence=turn.confidence,
                    )
                )

            logger.debug(
                "Diarization complete: %d speaker turns, %d unique speakers",
                len(assignments),
                len(self._speaker_map),
            )
            return assignments

        finally:
            # Clean up temporary file
            if wav_path and wav_path.exists():
                try:
                    wav_path.unlink()
                except OSError as e:
                    logger.debug("Failed to remove temp WAV: %s", e)

    def reset_speaker_map(self) -> None:
        """
        Reset the speaker ID mapping.

        Call this when starting a new stream to ensure consistent
        speaker numbering from spk_0.
        """
        self._speaker_map.clear()
        logger.debug("Reset speaker ID mapping")


# =============================================================================
# Energy-Based Voice Activity Detection Hook
# =============================================================================


@dataclass
class EnergyVADConfig:
    """
    Configuration for simple energy-based voice activity detection.

    This provides a lightweight alternative to pyannote when only basic
    speaker segmentation is needed.

    Attributes:
        frame_duration_ms: Duration of each analysis frame in milliseconds.
        energy_threshold: RMS energy threshold for speech detection (0.0-1.0).
        min_speech_duration_sec: Minimum duration to consider as speech.
        min_silence_duration_sec: Minimum silence duration between segments.
    """

    frame_duration_ms: float = 30.0
    energy_threshold: float = 0.01
    min_speech_duration_sec: float = 0.3
    min_silence_duration_sec: float = 0.5


class EnergyVADDiarizer:
    """
    Simple energy-based diarization for testing and lightweight scenarios.

    This hook uses RMS energy to detect voice activity regions but does not
    perform actual speaker identification. All speech regions are assigned
    to "spk_0" by default.

    Use cases:
    - Testing diarization integration without pyannote dependency
    - Lightweight single-speaker scenarios
    - Baseline comparison for diarization quality

    Example:
        >>> diarizer = EnergyVADDiarizer()
        >>> assignments = await diarizer(audio_buffer, sample_rate=16000)
    """

    def __init__(self, config: EnergyVADConfig | None = None) -> None:
        """
        Initialize energy-based VAD diarizer.

        Args:
            config: Configuration for VAD behavior.
        """
        self.config = config or EnergyVADConfig()

    async def __call__(
        self,
        audio_buffer: bytes,
        sample_rate: int,
    ) -> list[SpeakerAssignment]:
        """
        Run energy-based voice activity detection.

        Args:
            audio_buffer: Raw PCM audio bytes (16-bit mono).
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of SpeakerAssignment objects (all assigned to spk_0).
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._detect_speech_regions,
            audio_buffer,
            sample_rate,
        )

    def _detect_speech_regions(
        self,
        audio_buffer: bytes,
        sample_rate: int,
    ) -> list[SpeakerAssignment]:
        """
        Detect speech regions using RMS energy.

        Args:
            audio_buffer: Raw PCM audio bytes.
            sample_rate: Sample rate in Hz.

        Returns:
            List of SpeakerAssignment objects for speech regions.
        """
        import struct

        # Convert bytes to samples
        num_samples = len(audio_buffer) // 2
        samples = struct.unpack(f"<{num_samples}h", audio_buffer)

        # Calculate frame parameters
        frame_samples = int(sample_rate * self.config.frame_duration_ms / 1000)
        num_frames = num_samples // frame_samples

        if num_frames == 0:
            return []

        # Calculate RMS energy per frame
        frame_energies = []
        for i in range(num_frames):
            start = i * frame_samples
            end = start + frame_samples
            frame = samples[start:end]
            rms = (sum(s * s for s in frame) / len(frame)) ** 0.5
            # Normalize to 0-1 range (assuming 16-bit audio)
            normalized_rms = rms / 32768.0
            frame_energies.append(normalized_rms)

        # Find speech regions
        is_speech = [e > self.config.energy_threshold for e in frame_energies]

        # Merge adjacent speech frames into regions
        regions: list[tuple[float, float]] = []
        in_speech = False
        region_start = 0.0

        for i, speech in enumerate(is_speech):
            time = i * self.config.frame_duration_ms / 1000

            if speech and not in_speech:
                # Start of speech
                region_start = time
                in_speech = True
            elif not speech and in_speech:
                # End of speech
                region_end = time
                if region_end - region_start >= self.config.min_speech_duration_sec:
                    regions.append((region_start, region_end))
                in_speech = False

        # Handle trailing speech
        if in_speech:
            region_end = len(is_speech) * self.config.frame_duration_ms / 1000
            if region_end - region_start >= self.config.min_speech_duration_sec:
                regions.append((region_start, region_end))

        # Merge close regions
        merged_regions = self._merge_close_regions(regions)

        # Convert to SpeakerAssignment (all spk_0)
        return [
            SpeakerAssignment(
                start=start,
                end=end,
                speaker_id="spk_0",
                confidence=0.5,  # Low confidence for energy-based detection
            )
            for start, end in merged_regions
        ]

    def _merge_close_regions(
        self,
        regions: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Merge regions that are close together."""
        if not regions:
            return []

        merged = [regions[0]]
        for start, end in regions[1:]:
            last_start, last_end = merged[-1]
            if start - last_end < self.config.min_silence_duration_sec:
                # Merge with previous region
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))

        return merged


# =============================================================================
# Mock Hook for Testing
# =============================================================================


class MockDiarizationHook:
    """
    Mock diarization hook for testing.

    Generates deterministic speaker assignments based on audio duration.
    Useful for unit tests and integration testing without pyannote.

    Pattern:
    - 0-3s: spk_0
    - 3-6s: spk_1
    - 6-9s: spk_0
    - ... (alternating every 3 seconds)

    Example:
        >>> hook = MockDiarizationHook(segment_duration=3.0)
        >>> assignments = await hook(audio_buffer, sample_rate=16000)
    """

    def __init__(
        self,
        segment_duration: float = 3.0,
        num_speakers: int = 2,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Initialize mock diarization hook.

        Args:
            segment_duration: Duration of each speaker segment in seconds.
            num_speakers: Number of speakers to cycle through.
            latency_ms: Simulated processing latency in milliseconds.
        """
        self.segment_duration = segment_duration
        self.num_speakers = num_speakers
        self.latency_ms = latency_ms

    async def __call__(
        self,
        audio_buffer: bytes,
        sample_rate: int,
    ) -> list[SpeakerAssignment]:
        """
        Generate mock speaker assignments.

        Args:
            audio_buffer: Raw PCM audio bytes.
            sample_rate: Sample rate in Hz.

        Returns:
            List of alternating speaker assignments.
        """
        # Simulate processing latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        # Calculate audio duration
        bytes_per_sample = 2
        num_samples = len(audio_buffer) // bytes_per_sample
        duration = num_samples / sample_rate

        # Generate alternating segments
        assignments = []
        current_time = 0.0
        speaker_idx = 0

        while current_time < duration:
            segment_end = min(current_time + self.segment_duration, duration)
            assignments.append(
                SpeakerAssignment(
                    start=current_time,
                    end=segment_end,
                    speaker_id=f"spk_{speaker_idx}",
                    confidence=0.9,
                )
            )
            current_time = segment_end
            speaker_idx = (speaker_idx + 1) % self.num_speakers

        return assignments


# =============================================================================
# Factory Functions
# =============================================================================


def create_pyannote_hook(
    device: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    min_audio_duration_sec: float = 5.0,
    use_sliding_window: bool = False,
    window_duration_sec: float = 60.0,
) -> PyAnnoteIncrementalDiarizer:
    """
    Create a PyAnnote-based incremental diarization hook.

    This is the recommended hook for production streaming diarization.
    Requires pyannote.audio and HF_TOKEN environment variable.

    Args:
        device: Device for inference ("cuda", "cpu", or "auto").
        min_speakers: Minimum expected speakers (None = auto-detect).
        max_speakers: Maximum expected speakers (None = auto-detect).
        min_audio_duration_sec: Minimum audio before running diarization.
        use_sliding_window: Only process recent audio window (memory optimization).
        window_duration_sec: Sliding window duration when enabled.

    Returns:
        Configured PyAnnoteIncrementalDiarizer instance.

    Example:
        >>> hook = create_pyannote_hook(device="cuda", max_speakers=4)
        >>> session = WebSocketStreamingSession(config, diarization_hook=hook)

    Requirements:
        - Install with: uv sync --extra diarization
        - Set HF_TOKEN environment variable
        - Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
    """
    config = IncrementalDiarizationConfig(
        device=device,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        min_audio_duration_sec=min_audio_duration_sec,
        use_sliding_window=use_sliding_window,
        window_duration_sec=window_duration_sec,
    )
    return PyAnnoteIncrementalDiarizer(config)


def create_energy_vad_hook(
    energy_threshold: float = 0.01,
    min_speech_duration_sec: float = 0.3,
) -> EnergyVADDiarizer:
    """
    Create an energy-based VAD diarization hook.

    This lightweight hook detects voice activity using RMS energy but does
    not perform speaker identification. All speech is assigned to spk_0.

    Useful for:
    - Testing without pyannote dependency
    - Single-speaker scenarios
    - Very low-latency requirements

    Args:
        energy_threshold: RMS energy threshold for speech detection.
        min_speech_duration_sec: Minimum duration for speech regions.

    Returns:
        Configured EnergyVADDiarizer instance.
    """
    config = EnergyVADConfig(
        energy_threshold=energy_threshold,
        min_speech_duration_sec=min_speech_duration_sec,
    )
    return EnergyVADDiarizer(config)


def create_mock_hook(
    segment_duration: float = 3.0,
    num_speakers: int = 2,
    latency_ms: float = 0.0,
) -> MockDiarizationHook:
    """
    Create a mock diarization hook for testing.

    Generates deterministic alternating speaker assignments without any
    actual diarization. Useful for unit tests and development.

    Args:
        segment_duration: Duration of each speaker segment.
        num_speakers: Number of speakers to cycle through.
        latency_ms: Simulated processing latency.

    Returns:
        Configured MockDiarizationHook instance.
    """
    return MockDiarizationHook(
        segment_duration=segment_duration,
        num_speakers=num_speakers,
        latency_ms=latency_ms,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Configuration
    "IncrementalDiarizationConfig",
    "EnergyVADConfig",
    # Hook implementations
    "PyAnnoteIncrementalDiarizer",
    "EnergyVADDiarizer",
    "MockDiarizationHook",
    # Factory functions
    "create_pyannote_hook",
    "create_energy_vad_hook",
    "create_mock_hook",
]
