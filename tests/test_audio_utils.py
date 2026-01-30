"""Tests for audio_utils.py - AudioSegmentExtractor and related utilities.

This module provides comprehensive tests for:
- AudioSegmentExtractor class (constructor, extract_segment, extract_segment_by_frames, accessors)
- load_full_audio() function
- validate_wav_file() function
- Error handling, clamping vs raising behavior
- Multi-channel handling
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf

from transcription.audio_utils import (
    AudioSegmentExtractor,
    load_full_audio,
    validate_wav_file,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# ============================================================================
# Fixtures
# ============================================================================


def create_test_wav(
    path: Path,
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    frequency: float = 440.0,
) -> Path:
    """Create a test WAV file with a sine wave tone.

    Args:
        path: Output file path.
        duration_seconds: Duration of the audio in seconds.
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.
        frequency: Frequency of the sine wave tone in Hz.

    Returns:
        Path to the created WAV file.
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    # Convert to 16-bit PCM
    pcm_data = (audio_data * 32767).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        if channels == 1:
            wav_file.writeframes(pcm_data.tobytes())
        else:
            # Interleave channels
            multi_channel = np.column_stack([pcm_data] * channels)
            wav_file.writeframes(multi_channel.tobytes())

    return path


@pytest.fixture
def test_wav_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a standard test WAV file (1 second, 16kHz, mono)."""
    wav_path = tmp_path / "test_audio.wav"
    create_test_wav(wav_path, duration_seconds=1.0, sample_rate=16000, channels=1)
    yield wav_path


@pytest.fixture
def long_wav_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a longer test WAV file (10 seconds, 16kHz, mono)."""
    wav_path = tmp_path / "long_audio.wav"
    create_test_wav(wav_path, duration_seconds=10.0, sample_rate=16000, channels=1)
    yield wav_path


@pytest.fixture
def stereo_wav_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a stereo test WAV file (1 second, 16kHz, 2 channels)."""
    wav_path = tmp_path / "stereo_audio.wav"
    create_test_wav(wav_path, duration_seconds=1.0, sample_rate=16000, channels=2)
    yield wav_path


@pytest.fixture
def high_sample_rate_wav(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a high sample rate WAV file (1 second, 44100Hz, mono)."""
    wav_path = tmp_path / "high_sr_audio.wav"
    create_test_wav(wav_path, duration_seconds=1.0, sample_rate=44100, channels=1)
    yield wav_path


# ============================================================================
# AudioSegmentExtractor - Constructor Tests
# ============================================================================


class TestAudioSegmentExtractorConstructor:
    """Tests for AudioSegmentExtractor.__init__()."""

    def test_init_with_valid_wav_file(self, test_wav_file: Path) -> None:
        """Constructor should successfully initialize with a valid WAV file."""
        extractor = AudioSegmentExtractor(test_wav_file)

        assert extractor.wav_path == test_wav_file
        assert extractor.sample_rate == 16000
        assert extractor.channels == 1
        assert extractor.total_frames == 16000  # 1 second at 16kHz
        assert abs(extractor.duration_seconds - 1.0) < 0.001

    def test_init_with_string_path(self, test_wav_file: Path) -> None:
        """Constructor should accept string path as well as Path object."""
        extractor = AudioSegmentExtractor(str(test_wav_file))

        assert extractor.wav_path == test_wav_file
        assert extractor.sample_rate == 16000

    def test_init_with_nonexistent_file(self, tmp_path: Path) -> None:
        """Constructor should raise FileNotFoundError for missing files."""
        nonexistent = tmp_path / "does_not_exist.wav"

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            AudioSegmentExtractor(nonexistent)

    def test_init_with_directory_raises_value_error(self, tmp_path: Path) -> None:
        """Constructor should raise ValueError when given a directory."""
        with pytest.raises(ValueError, match="Path is not a file"):
            AudioSegmentExtractor(tmp_path)

    def test_init_with_invalid_audio_file(self, tmp_path: Path) -> None:
        """Constructor should raise RuntimeError for invalid audio files."""
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("This is not a WAV file")

        with pytest.raises(RuntimeError, match="Failed to open audio file"):
            AudioSegmentExtractor(invalid_file)

    def test_init_stores_correct_metadata(self, stereo_wav_file: Path) -> None:
        """Constructor should correctly store stereo file metadata."""
        extractor = AudioSegmentExtractor(stereo_wav_file)

        assert extractor.channels == 2
        assert extractor.sample_rate == 16000
        assert extractor.total_frames == 16000

    def test_init_with_high_sample_rate(self, high_sample_rate_wav: Path) -> None:
        """Constructor should handle different sample rates."""
        extractor = AudioSegmentExtractor(high_sample_rate_wav)

        assert extractor.sample_rate == 44100
        assert extractor.total_frames == 44100


# ============================================================================
# AudioSegmentExtractor - extract_segment() Tests
# ============================================================================


class TestAudioSegmentExtractorExtractSegment:
    """Tests for AudioSegmentExtractor.extract_segment()."""

    def test_extract_full_file(self, test_wav_file: Path) -> None:
        """Extracting 0.0 to duration should return entire file."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, sr = extractor.extract_segment(0.0, 1.0)

        assert sr == 16000
        assert len(audio) == 16000
        assert audio.dtype == np.float32

    def test_extract_segment_middle(self, long_wav_file: Path) -> None:
        """Extracting a segment from the middle of the file."""
        extractor = AudioSegmentExtractor(long_wav_file)
        audio, sr = extractor.extract_segment(2.0, 5.0)

        # 3 seconds at 16kHz = 48000 samples
        assert sr == 16000
        assert len(audio) == 48000

    def test_extract_segment_start(self, long_wav_file: Path) -> None:
        """Extracting a segment from the start of the file."""
        extractor = AudioSegmentExtractor(long_wav_file)
        audio, sr = extractor.extract_segment(0.0, 2.5)

        # 2.5 seconds at 16kHz = 40000 samples
        assert len(audio) == 40000

    def test_extract_segment_end(self, long_wav_file: Path) -> None:
        """Extracting a segment from the end of the file."""
        extractor = AudioSegmentExtractor(long_wav_file)
        audio, sr = extractor.extract_segment(8.0, 10.0)

        # 2 seconds at 16kHz = 32000 samples
        assert len(audio) == 32000

    def test_extract_very_short_segment(self, long_wav_file: Path) -> None:
        """Extracting a very short segment should work."""
        extractor = AudioSegmentExtractor(long_wav_file)
        audio, sr = extractor.extract_segment(1.0, 1.001)

        # Should have at least 1 sample due to frame clamping
        assert len(audio) >= 1

    def test_extract_segment_returns_mono(self, stereo_wav_file: Path) -> None:
        """Extracting from stereo file should return mono (first channel)."""
        extractor = AudioSegmentExtractor(stereo_wav_file)
        audio, sr = extractor.extract_segment(0.0, 0.5)

        # Should be 1D array (mono)
        assert audio.ndim == 1

    def test_extract_segment_invalid_order_raises(self, test_wav_file: Path) -> None:
        """Start time greater than end time should raise ValueError."""
        extractor = AudioSegmentExtractor(test_wav_file)

        with pytest.raises(ValueError, match="Start time.*must be <= end time"):
            extractor.extract_segment(0.5, 0.2)

    def test_extract_segment_negative_start_clamped(self, test_wav_file: Path) -> None:
        """Negative start time should be clamped to 0 when clamp=True."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, sr = extractor.extract_segment(-0.5, 0.5, clamp=True)

        # Should extract from 0.0 to 0.5 (8000 samples)
        assert len(audio) == 8000

    def test_extract_segment_negative_start_raises(self, test_wav_file: Path) -> None:
        """Negative start time should raise ValueError when clamp=False."""
        extractor = AudioSegmentExtractor(test_wav_file)

        with pytest.raises(ValueError, match="Start time cannot be negative"):
            extractor.extract_segment(-0.5, 0.5, clamp=False)

    def test_extract_segment_negative_end_clamped(self, test_wav_file: Path) -> None:
        """Negative end time should be clamped to 0 when clamp=True."""
        extractor = AudioSegmentExtractor(test_wav_file)

        # With both negative and clamped, both become 0, segment is 0 length
        # min_duration check will fail if min_duration > 0
        audio, sr = extractor.extract_segment(-0.5, -0.1, clamp=True, min_duration=0.0)

        # After clamping, start=0, end=0, then frame adjustment gives at least 1 sample
        assert len(audio) >= 1

    def test_extract_segment_negative_end_raises(self, test_wav_file: Path) -> None:
        """Negative end time should raise ValueError when clamp=False."""
        extractor = AudioSegmentExtractor(test_wav_file)

        # Note: When start > end (0.0 > -0.5), the ordering check triggers first.
        # To test the negative end time check, we need start < end with both negative.
        with pytest.raises(ValueError, match="Start time.*must be <= end time"):
            extractor.extract_segment(0.0, -0.5, clamp=False)

    def test_extract_segment_both_negative_raises(self, test_wav_file: Path) -> None:
        """Both negative with clamp=False should raise ValueError for start."""
        extractor = AudioSegmentExtractor(test_wav_file)

        # With start=-0.5, end=-0.1, ordering is valid, so negative check triggers
        with pytest.raises(ValueError, match="Start time cannot be negative"):
            extractor.extract_segment(-0.5, -0.1, clamp=False)

    def test_extract_segment_beyond_duration_clamped(self, test_wav_file: Path) -> None:
        """End time beyond duration should be clamped when clamp=True."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, sr = extractor.extract_segment(0.5, 5.0, clamp=True)

        # Should clamp end to 1.0, extracting 0.5 to 1.0 (8000 samples)
        assert len(audio) == 8000

    def test_extract_segment_beyond_duration_raises(self, test_wav_file: Path) -> None:
        """End time beyond duration should raise ValueError when clamp=False."""
        extractor = AudioSegmentExtractor(test_wav_file)

        with pytest.raises(ValueError, match="End time.*exceeds file duration"):
            extractor.extract_segment(0.5, 5.0, clamp=False)

    def test_extract_segment_start_beyond_duration_clamped(self, test_wav_file: Path) -> None:
        """Start time beyond duration should be clamped when clamp=True."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, sr = extractor.extract_segment(5.0, 10.0, clamp=True, min_duration=0.0)

        # Both clamped to duration, minimal segment returned
        assert len(audio) >= 1

    def test_extract_segment_start_beyond_duration_raises(self, test_wav_file: Path) -> None:
        """Start time beyond duration should raise ValueError when clamp=False."""
        extractor = AudioSegmentExtractor(test_wav_file)

        with pytest.raises(ValueError, match="Start time.*exceeds file duration"):
            extractor.extract_segment(5.0, 10.0, clamp=False)

    def test_extract_segment_min_duration_satisfied(self, long_wav_file: Path) -> None:
        """Segment meeting min_duration requirement should succeed."""
        extractor = AudioSegmentExtractor(long_wav_file)
        audio, sr = extractor.extract_segment(1.0, 3.0, min_duration=1.0)

        # 2 second segment meets 1 second minimum
        assert len(audio) == 32000

    def test_extract_segment_min_duration_not_met_raises(self, long_wav_file: Path) -> None:
        """Segment shorter than min_duration should raise ValueError."""
        extractor = AudioSegmentExtractor(long_wav_file)

        with pytest.raises(ValueError, match="Segment duration.*shorter than.*minimum"):
            extractor.extract_segment(1.0, 1.5, min_duration=1.0)

    def test_extract_segment_min_duration_after_clamping(self, test_wav_file: Path) -> None:
        """min_duration should be checked after clamping."""
        extractor = AudioSegmentExtractor(test_wav_file)

        # Request 0.5 to 5.0, clamped to 0.5 to 1.0 (0.5s duration)
        # min_duration=1.0 should fail
        with pytest.raises(ValueError, match="Segment duration.*shorter than.*minimum"):
            extractor.extract_segment(0.5, 5.0, clamp=True, min_duration=1.0)


# ============================================================================
# AudioSegmentExtractor - extract_segment_by_frames() Tests
# ============================================================================


class TestAudioSegmentExtractorExtractByFrames:
    """Tests for AudioSegmentExtractor.extract_segment_by_frames()."""

    def test_extract_by_frames_basic(self, test_wav_file: Path) -> None:
        """Basic frame extraction should work correctly."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, sr = extractor.extract_segment_by_frames(0, 8000)

        assert len(audio) == 8000
        assert sr == 16000

    def test_extract_by_frames_middle(self, long_wav_file: Path) -> None:
        """Frame extraction from middle of file."""
        extractor = AudioSegmentExtractor(long_wav_file)
        audio, sr = extractor.extract_segment_by_frames(32000, 64000)

        assert len(audio) == 32000

    def test_extract_by_frames_invalid_order_raises(self, test_wav_file: Path) -> None:
        """Start frame > end frame should raise ValueError."""
        extractor = AudioSegmentExtractor(test_wav_file)

        with pytest.raises(ValueError, match="Start frame.*must be <= end frame"):
            extractor.extract_segment_by_frames(8000, 4000)

    def test_extract_by_frames_negative_start_clamped(self, test_wav_file: Path) -> None:
        """Negative start frame should be clamped to 0 when clamp=True."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, sr = extractor.extract_segment_by_frames(-1000, 8000, clamp=True)

        assert len(audio) == 8000

    def test_extract_by_frames_negative_start_raises(self, test_wav_file: Path) -> None:
        """Negative start frame should raise ValueError when clamp=False."""
        extractor = AudioSegmentExtractor(test_wav_file)

        with pytest.raises(ValueError, match="Start frame cannot be negative"):
            extractor.extract_segment_by_frames(-1000, 8000, clamp=False)

    def test_extract_by_frames_beyond_total_clamped(self, test_wav_file: Path) -> None:
        """End frame beyond total frames should be clamped when clamp=True."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, sr = extractor.extract_segment_by_frames(8000, 20000, clamp=True)

        # Should clamp to 16000 total frames
        assert len(audio) == 8000

    def test_extract_by_frames_beyond_total_raises(self, test_wav_file: Path) -> None:
        """End frame beyond total frames should raise ValueError when clamp=False."""
        extractor = AudioSegmentExtractor(test_wav_file)

        with pytest.raises(ValueError, match="End frame.*exceeds total frames"):
            extractor.extract_segment_by_frames(8000, 20000, clamp=False)

    def test_extract_by_frames_empty_range_raises(self, test_wav_file: Path) -> None:
        """Empty frame range should raise ValueError."""
        extractor = AudioSegmentExtractor(test_wav_file)

        with pytest.raises(ValueError, match="Invalid frame range"):
            extractor.extract_segment_by_frames(8000, 8000, clamp=True)

    def test_extract_by_frames_returns_mono_from_stereo(self, stereo_wav_file: Path) -> None:
        """Frame extraction from stereo should return mono."""
        extractor = AudioSegmentExtractor(stereo_wav_file)
        audio, sr = extractor.extract_segment_by_frames(0, 8000)

        assert audio.ndim == 1


# ============================================================================
# AudioSegmentExtractor - Accessor Methods Tests
# ============================================================================


class TestAudioSegmentExtractorAccessors:
    """Tests for AudioSegmentExtractor accessor methods."""

    def test_get_duration(self, test_wav_file: Path) -> None:
        """get_duration() should return correct duration."""
        extractor = AudioSegmentExtractor(test_wav_file)

        duration = extractor.get_duration()

        assert abs(duration - 1.0) < 0.001

    def test_get_duration_long_file(self, long_wav_file: Path) -> None:
        """get_duration() should work for longer files."""
        extractor = AudioSegmentExtractor(long_wav_file)

        duration = extractor.get_duration()

        assert abs(duration - 10.0) < 0.001

    def test_get_info(self, test_wav_file: Path) -> None:
        """get_info() should return complete metadata dictionary."""
        extractor = AudioSegmentExtractor(test_wav_file)

        info = extractor.get_info()

        assert info["sample_rate"] == 16000
        assert info["total_frames"] == 16000
        assert info["channels"] == 1
        assert abs(info["duration_seconds"] - 1.0) < 0.001
        assert info["file_path"] == str(test_wav_file)

    def test_get_info_stereo(self, stereo_wav_file: Path) -> None:
        """get_info() should correctly report stereo channels."""
        extractor = AudioSegmentExtractor(stereo_wav_file)

        info = extractor.get_info()

        assert info["channels"] == 2

    def test_repr(self, test_wav_file: Path) -> None:
        """__repr__() should return informative string."""
        extractor = AudioSegmentExtractor(test_wav_file)

        repr_str = repr(extractor)

        assert "AudioSegmentExtractor" in repr_str
        assert "test_audio.wav" in repr_str
        assert "duration=" in repr_str
        assert "sr=16000Hz" in repr_str
        assert "channels=1" in repr_str

    def test_repr_stereo(self, stereo_wav_file: Path) -> None:
        """__repr__() should show correct channel count for stereo."""
        extractor = AudioSegmentExtractor(stereo_wav_file)

        repr_str = repr(extractor)

        assert "channels=2" in repr_str


# ============================================================================
# load_full_audio() Tests
# ============================================================================


class TestLoadFullAudio:
    """Tests for load_full_audio() function."""

    def test_load_full_audio_basic(self, test_wav_file: Path) -> None:
        """load_full_audio() should load entire file."""
        audio, sr = load_full_audio(test_wav_file)

        assert sr == 16000
        assert len(audio) == 16000
        assert audio.dtype == np.float32

    def test_load_full_audio_with_string_path(self, test_wav_file: Path) -> None:
        """load_full_audio() should accept string path."""
        audio, sr = load_full_audio(str(test_wav_file))

        assert sr == 16000
        assert len(audio) == 16000

    def test_load_full_audio_nonexistent_file(self, tmp_path: Path) -> None:
        """load_full_audio() should raise FileNotFoundError for missing files."""
        nonexistent = tmp_path / "missing.wav"

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            load_full_audio(nonexistent)

    def test_load_full_audio_invalid_file(self, tmp_path: Path) -> None:
        """load_full_audio() should raise RuntimeError for invalid files."""
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("Not a WAV file")

        with pytest.raises(RuntimeError, match="Failed to load audio file"):
            load_full_audio(invalid_file)

    def test_load_full_audio_stereo_returns_mono(self, stereo_wav_file: Path) -> None:
        """load_full_audio() should convert stereo to mono."""
        audio, sr = load_full_audio(stereo_wav_file)

        assert audio.ndim == 1

    def test_load_full_audio_long_file(self, long_wav_file: Path) -> None:
        """load_full_audio() should handle longer files."""
        audio, sr = load_full_audio(long_wav_file)

        assert len(audio) == 160000  # 10 seconds at 16kHz


# ============================================================================
# validate_wav_file() Tests
# ============================================================================


class TestValidateWavFile:
    """Tests for validate_wav_file() function."""

    def test_validate_wav_file_valid(self, test_wav_file: Path) -> None:
        """validate_wav_file() should return True for valid WAV files."""
        assert validate_wav_file(test_wav_file) is True

    def test_validate_wav_file_with_string_path(self, test_wav_file: Path) -> None:
        """validate_wav_file() should accept string path."""
        assert validate_wav_file(str(test_wav_file)) is True

    def test_validate_wav_file_nonexistent(self, tmp_path: Path) -> None:
        """validate_wav_file() should return False for missing files."""
        nonexistent = tmp_path / "missing.wav"

        assert validate_wav_file(nonexistent) is False

    def test_validate_wav_file_directory(self, tmp_path: Path) -> None:
        """validate_wav_file() should return False for directories."""
        assert validate_wav_file(tmp_path) is False

    def test_validate_wav_file_invalid_content(self, tmp_path: Path) -> None:
        """validate_wav_file() should return False for invalid audio files."""
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("This is not a WAV file")

        assert validate_wav_file(invalid_file) is False

    def test_validate_wav_file_empty_file(self, tmp_path: Path) -> None:
        """validate_wav_file() should return False for empty files."""
        empty_file = tmp_path / "empty.wav"
        empty_file.touch()

        assert validate_wav_file(empty_file) is False

    def test_validate_wav_file_stereo(self, stereo_wav_file: Path) -> None:
        """validate_wav_file() should return True for valid stereo files."""
        assert validate_wav_file(stereo_wav_file) is True


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_extract_exact_duration(self, test_wav_file: Path) -> None:
        """Extracting exactly to file duration should work."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, sr = extractor.extract_segment(0.0, extractor.duration_seconds)

        # Should get exactly the number of frames in the file
        assert len(audio) == extractor.total_frames

    def test_extract_same_start_and_end(self, test_wav_file: Path) -> None:
        """Same start and end time should return minimal segment."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, sr = extractor.extract_segment(0.5, 0.5, min_duration=0.0)

        # Due to frame adjustment, should get at least 1 sample
        assert len(audio) >= 1

    def test_multiple_extractions_from_same_extractor(self, long_wav_file: Path) -> None:
        """Multiple extractions should work correctly."""
        extractor = AudioSegmentExtractor(long_wav_file)

        audio1, sr1 = extractor.extract_segment(0.0, 1.0)
        audio2, sr2 = extractor.extract_segment(5.0, 6.0)
        audio3, sr3 = extractor.extract_segment(9.0, 10.0)

        assert len(audio1) == 16000
        assert len(audio2) == 16000
        assert len(audio3) == 16000

    def test_consistency_between_time_and_frame_extraction(self, long_wav_file: Path) -> None:
        """Time-based and frame-based extraction should produce consistent results."""
        extractor = AudioSegmentExtractor(long_wav_file)

        # Extract 1 second starting at 2 seconds
        audio_by_time, _ = extractor.extract_segment(2.0, 3.0)
        audio_by_frames, _ = extractor.extract_segment_by_frames(32000, 48000)

        # Should be the same
        assert len(audio_by_time) == len(audio_by_frames)
        np.testing.assert_array_almost_equal(audio_by_time, audio_by_frames, decimal=5)

    def test_extractor_attributes_immutable_after_init(self, test_wav_file: Path) -> None:
        """Extractor attributes should remain correct after extractions."""
        extractor = AudioSegmentExtractor(test_wav_file)

        original_sr = extractor.sample_rate
        original_frames = extractor.total_frames
        original_duration = extractor.duration_seconds

        # Perform some extractions
        extractor.extract_segment(0.0, 0.5)
        extractor.extract_segment_by_frames(0, 8000)

        # Attributes should be unchanged
        assert extractor.sample_rate == original_sr
        assert extractor.total_frames == original_frames
        assert extractor.duration_seconds == original_duration


class TestMultiChannelHandling:
    """Tests specifically for multi-channel audio handling."""

    def test_stereo_extraction_returns_first_channel(self, stereo_wav_file: Path) -> None:
        """Extracting from stereo should return first channel only."""
        extractor = AudioSegmentExtractor(stereo_wav_file)
        audio, sr = extractor.extract_segment(0.0, 0.5)

        # Should be 1D (mono)
        assert audio.ndim == 1
        assert len(audio) == 8000

    def test_stereo_load_full_returns_first_channel(self, stereo_wav_file: Path) -> None:
        """Loading full stereo file should return first channel only."""
        audio, sr = load_full_audio(stereo_wav_file)

        assert audio.ndim == 1

    def test_stereo_frame_extraction_returns_first_channel(self, stereo_wav_file: Path) -> None:
        """Frame extraction from stereo should return first channel."""
        extractor = AudioSegmentExtractor(stereo_wav_file)
        audio, sr = extractor.extract_segment_by_frames(0, 8000)

        assert audio.ndim == 1


class TestDataTypeConsistency:
    """Tests for ensuring consistent data types."""

    def test_extract_segment_returns_float32(self, test_wav_file: Path) -> None:
        """extract_segment() should always return float32."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, _ = extractor.extract_segment(0.0, 0.5)

        assert audio.dtype == np.float32

    def test_extract_by_frames_returns_float32(self, test_wav_file: Path) -> None:
        """extract_segment_by_frames() should always return float32."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, _ = extractor.extract_segment_by_frames(0, 8000)

        assert audio.dtype == np.float32

    def test_load_full_audio_returns_float32(self, test_wav_file: Path) -> None:
        """load_full_audio() should always return float32."""
        audio, _ = load_full_audio(test_wav_file)

        assert audio.dtype == np.float32

    def test_audio_values_normalized(self, test_wav_file: Path) -> None:
        """Audio values should be in normalized range [-1, 1]."""
        extractor = AudioSegmentExtractor(test_wav_file)
        audio, _ = extractor.extract_segment(0.0, 1.0)

        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)

# ============================================================================
# AudioSegmentExtractor - Context Manager Optimization Tests
# ============================================================================


class TestAudioSegmentExtractorContextManager:
    """Tests for AudioSegmentExtractor context manager functionality."""

    def test_context_manager_reuses_file_handle(self, test_wav_file: Path) -> None:
        """Verify that using the context manager reuses the internal file handle."""
        from unittest.mock import patch

        extractor = AudioSegmentExtractor(test_wav_file)

        assert extractor._file_handle is None

        with extractor:
            assert extractor._file_handle is not None
            assert not extractor._file_handle.closed

            # Verify extract_segment uses the handle
            with patch.object(extractor._file_handle, 'read', wraps=extractor._file_handle.read) as mock_read:
                data, sr = extractor.extract_segment(0.0, 0.1)
                assert len(data) > 0
                mock_read.assert_called()

            # Verify extract_segment_by_frames uses the handle
            with patch.object(extractor._file_handle, 'read', wraps=extractor._file_handle.read) as mock_read:
                data, sr = extractor.extract_segment_by_frames(0, 1600)
                assert len(data) > 0
                mock_read.assert_called()

        assert extractor._file_handle is None

    def test_fallback_behavior_without_context_manager(self, test_wav_file: Path) -> None:
        """Verify that without context manager, file is opened/closed each time."""
        from unittest.mock import patch

        extractor = AudioSegmentExtractor(test_wav_file)
        assert extractor._file_handle is None

        # spy on sf.SoundFile to see if it's instantiated
        with patch('soundfile.SoundFile', wraps=sf.SoundFile) as mock_sf:
            data, sr = extractor.extract_segment(0.0, 0.1)
            # It should be called once (open)
            mock_sf.assert_called()

        # And again
        with patch('soundfile.SoundFile', wraps=sf.SoundFile) as mock_sf:
            data, sr = extractor.extract_segment(0.1, 0.2)
            mock_sf.assert_called()

    def test_context_manager_avoids_repeated_opens(self, test_wav_file: Path) -> None:
        """Verify that context manager avoids repeated sf.SoundFile calls."""
        from unittest.mock import patch

        extractor = AudioSegmentExtractor(test_wav_file)

        with extractor:
            # Inside context, sf.SoundFile constructor should NOT be called for extractions
            # (It WAS called once for __enter__)

            with patch('soundfile.SoundFile', wraps=sf.SoundFile) as mock_sf:
                data, sr = extractor.extract_segment(0.0, 0.1)
                data, sr = extractor.extract_segment(0.1, 0.2)

                # Should NOT be called because we use the cached handle
                mock_sf.assert_not_called()

    def test_context_manager_handles_exception_in_enter(self, test_wav_file: Path) -> None:
        """Verify exception handling in __enter__."""
        from unittest.mock import patch

        extractor = AudioSegmentExtractor(test_wav_file)

        # Simulate file opening error
        with patch('soundfile.SoundFile', side_effect=RuntimeError("Open failed")):
            with pytest.raises(RuntimeError, match="Failed to open audio file"):
                with extractor:
                    pass

        assert extractor._file_handle is None
