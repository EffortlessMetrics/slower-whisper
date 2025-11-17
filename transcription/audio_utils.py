"""
Audio loading and slicing utilities for efficient segment extraction.

This module provides memory-efficient audio segment extraction using soundfile's
seeking capabilities. Designed for 16kHz mono WAV files normalized by the pipeline.
"""

from pathlib import Path

import numpy as np
import soundfile as sf


class AudioSegmentExtractor:
    """
    Memory-efficient audio segment extractor using soundfile seeking.

    This class enables extraction of specific time ranges from WAV files without
    loading the entire file into memory. It's optimized for 16kHz mono WAV files
    produced by the normalization pipeline.

    Example:
        >>> extractor = AudioSegmentExtractor("audio.wav")
        >>> audio, sr = extractor.extract_segment(10.5, 15.0)
        >>> print(f"Extracted {len(audio)} samples at {sr} Hz")
    """

    def __init__(self, wav_path: Path | str):
        """
        Initialize the extractor with a WAV file path.

        Args:
            wav_path: Path to the WAV file to extract segments from.

        Raises:
            FileNotFoundError: If the WAV file does not exist.
            ValueError: If the file is not a valid audio file.
            RuntimeError: If the audio file cannot be opened.
        """
        self.wav_path = Path(wav_path)

        # Validate file exists
        if not self.wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.wav_path}")

        if not self.wav_path.is_file():
            raise ValueError(f"Path is not a file: {self.wav_path}")

        # Probe the file to get metadata and validate readability
        try:
            with sf.SoundFile(str(self.wav_path), "r") as audio_file:
                self.sample_rate: int = audio_file.samplerate
                self.total_frames: int = len(audio_file)
                self.channels: int = audio_file.channels
                self.duration_seconds: float = self.total_frames / self.sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to open audio file '{self.wav_path}': {e}") from e

    def extract_segment(
        self,
        start_seconds: float,
        end_seconds: float,
        clamp: bool = True,
        min_duration: float = 0.0,
    ) -> tuple[np.ndarray, int]:
        """
        Extract an audio segment between start and end timestamps.

        Uses soundfile's seek functionality to read only the required frames,
        making this memory-efficient for large files.

        Args:
            start_seconds: Start time in seconds (inclusive).
            end_seconds: End time in seconds (exclusive).
            clamp: If True, clamp timestamps to valid file boundaries.
                   If False, raise ValueError for out-of-bounds timestamps.
            min_duration: Minimum required duration in seconds. If the segment
                          (after clamping) is shorter, raises ValueError.

        Returns:
            Tuple of (audio_array, sample_rate) where:
                - audio_array is a 1D numpy array of float32 samples
                - sample_rate is the audio sample rate in Hz

        Raises:
            ValueError: If timestamps are invalid, out of bounds (when clamp=False),
                        or result in a segment shorter than min_duration.
        """
        # Validate timestamp ordering
        if start_seconds > end_seconds:
            raise ValueError(f"Start time ({start_seconds}s) must be <= end time ({end_seconds}s)")

        # Handle negative timestamps
        if start_seconds < 0:
            if clamp:
                start_seconds = 0.0
            else:
                raise ValueError(f"Start time cannot be negative: {start_seconds}s")

        if end_seconds < 0:
            if clamp:
                end_seconds = 0.0
            else:
                raise ValueError(f"End time cannot be negative: {end_seconds}s")

        # Handle timestamps beyond file duration
        if start_seconds > self.duration_seconds:
            if clamp:
                start_seconds = self.duration_seconds
            else:
                raise ValueError(
                    f"Start time ({start_seconds}s) exceeds file duration "
                    f"({self.duration_seconds:.2f}s)"
                )

        if end_seconds > self.duration_seconds:
            if clamp:
                end_seconds = self.duration_seconds
            else:
                raise ValueError(
                    f"End time ({end_seconds}s) exceeds file duration "
                    f"({self.duration_seconds:.2f}s)"
                )

        # Check minimum duration requirement
        actual_duration = end_seconds - start_seconds
        if actual_duration < min_duration:
            raise ValueError(
                f"Segment duration ({actual_duration:.3f}s) is shorter than "
                f"minimum required duration ({min_duration}s)"
            )

        # Convert time to frames
        start_frame = int(start_seconds * self.sample_rate)
        end_frame = int(end_seconds * self.sample_rate)

        # Ensure we have at least one frame
        if start_frame >= end_frame:
            # This can happen with very short segments due to rounding
            end_frame = start_frame + 1

        # Clamp frames to valid range
        start_frame = max(0, min(start_frame, self.total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, self.total_frames))

        num_frames = end_frame - start_frame

        # Extract the segment using seek
        try:
            with sf.SoundFile(str(self.wav_path), "r") as audio_file:
                # Seek to start position
                audio_file.seek(start_frame)

                # Read the required number of frames
                audio_data = audio_file.read(num_frames, dtype="float32")

                # Handle multi-channel audio by taking first channel
                # (should be mono for normalized files, but be defensive)
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]

                return audio_data, self.sample_rate

        except Exception as e:
            raise RuntimeError(
                f"Failed to extract segment [{start_seconds}s - {end_seconds}s] "
                f"from '{self.wav_path}': {e}"
            ) from e

    def extract_segment_by_frames(
        self, start_frame: int, end_frame: int, clamp: bool = True
    ) -> tuple[np.ndarray, int]:
        """
        Extract an audio segment by frame indices (for advanced use).

        Args:
            start_frame: Starting frame index (inclusive).
            end_frame: Ending frame index (exclusive).
            clamp: If True, clamp frame indices to valid range.

        Returns:
            Tuple of (audio_array, sample_rate).

        Raises:
            ValueError: If frame indices are invalid.
        """
        if start_frame > end_frame:
            raise ValueError(f"Start frame ({start_frame}) must be <= end frame ({end_frame})")

        if start_frame < 0:
            if clamp:
                start_frame = 0
            else:
                raise ValueError(f"Start frame cannot be negative: {start_frame}")

        if end_frame > self.total_frames:
            if clamp:
                end_frame = self.total_frames
            else:
                raise ValueError(
                    f"End frame ({end_frame}) exceeds total frames ({self.total_frames})"
                )

        if start_frame >= end_frame:
            raise ValueError(f"Invalid frame range: [{start_frame}, {end_frame})")

        num_frames = end_frame - start_frame

        try:
            with sf.SoundFile(str(self.wav_path), "r") as audio_file:
                audio_file.seek(start_frame)
                audio_data = audio_file.read(num_frames, dtype="float32")

                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]

                return audio_data, self.sample_rate

        except Exception as e:
            raise RuntimeError(
                f"Failed to extract frames [{start_frame} - {end_frame}] "
                f"from '{self.wav_path}': {e}"
            ) from e

    def get_duration(self) -> float:
        """
        Get the total duration of the audio file in seconds.

        Returns:
            Duration in seconds.
        """
        return self.duration_seconds

    def get_info(self) -> dict:
        """
        Get metadata about the audio file.

        Returns:
            Dictionary with keys: 'sample_rate', 'total_frames', 'channels',
            'duration_seconds', 'file_path'.
        """
        return {
            "sample_rate": self.sample_rate,
            "total_frames": self.total_frames,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "file_path": str(self.wav_path),
        }

    def __repr__(self) -> str:
        """String representation of the extractor."""
        return (
            f"AudioSegmentExtractor('{self.wav_path.name}', "
            f"duration={self.duration_seconds:.2f}s, "
            f"sr={self.sample_rate}Hz, "
            f"channels={self.channels})"
        )


def load_full_audio(wav_path: Path | str) -> tuple[np.ndarray, int]:
    """
    Load an entire audio file into memory.

    This is a convenience function for cases where you need the full audio.
    For segment extraction, use AudioSegmentExtractor instead.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        Tuple of (audio_array, sample_rate).

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read.
    """
    wav_path = Path(wav_path)

    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    try:
        audio_data, sample_rate = sf.read(str(wav_path), dtype="float32")

        # Convert to mono if needed
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        return audio_data, sample_rate

    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{wav_path}': {e}") from e


def validate_wav_file(wav_path: Path | str) -> bool:
    """
    Validate that a file exists and is a readable WAV file.

    Args:
        wav_path: Path to the file to validate.

    Returns:
        True if valid, False otherwise.
    """
    try:
        wav_path = Path(wav_path)

        if not wav_path.exists() or not wav_path.is_file():
            return False

        # Try to open and read metadata
        with sf.SoundFile(str(wav_path), "r") as audio_file:
            # Just accessing these will validate the file
            _ = audio_file.samplerate
            _ = len(audio_file)

        return True

    except Exception:
        return False
