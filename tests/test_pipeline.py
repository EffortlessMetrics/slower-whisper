"""
Unit tests for the transcription pipeline (transcription/pipeline.py).

This test module covers:
- PipelineFileResult and PipelineBatchResult dataclasses
- run_pipeline() orchestration logic
- Metadata building (_build_meta)
- Duration extraction (_get_duration_seconds)
- Error handling and recovery
- Skip-existing logic
- Diarization integration paths

These tests complement test_api_integration.py by testing the pipeline
internals directly rather than through the API layer.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcription.config import AppConfig, AsrConfig, Paths
from transcription.models import Segment, Transcript
from transcription.pipeline import (
    PipelineBatchResult,
    PipelineFileResult,
    _build_meta,
    _get_duration_seconds,
    run_pipeline,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_project_structure(tmp_path: Path) -> Paths:
    """
    Create a temporary project directory structure.

    Creates:
        raw_audio/      - for input recordings
        input_audio/    - for normalized WAVs
        whisper_json/   - for JSON transcripts
        transcripts/    - for TXT/SRT outputs

    Returns:
        Paths object configured for the temp directory
    """
    root = tmp_path / "project"
    root.mkdir()

    for subdir in ["raw_audio", "input_audio", "whisper_json", "transcripts"]:
        (root / subdir).mkdir()

    return Paths(root=root)


@pytest.fixture
def sample_app_config(temp_project_structure: Paths) -> AppConfig:
    """Create a sample AppConfig for testing."""
    asr_cfg = AsrConfig(
        model_name="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
    )
    return AppConfig(
        paths=temp_project_structure,
        asr=asr_cfg,
        skip_existing_json=False,
    )


@pytest.fixture
def sample_transcript() -> Transcript:
    """Create a sample transcript for testing."""
    return Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.5, text="Hello world"),
            Segment(id=1, start=1.5, end=3.0, text="Testing pipeline"),
        ],
        meta={
            "asr_backend": "faster-whisper",
            "asr_device": "cpu",
            "asr_compute_type": "int8",
        },
    )


# ============================================================================
# PipelineFileResult Tests
# ============================================================================


class TestPipelineFileResult:
    """Tests for the PipelineFileResult dataclass."""

    def test_file_result_success_status(self) -> None:
        """Test creating a successful file result."""
        # TODO: Implement test
        # Should verify:
        # - file_name is set correctly
        # - status is "success"
        # - error_message is None
        result = PipelineFileResult(
            file_name="audio.wav",
            status="success",
            error_message=None,
        )
        assert result.file_name == "audio.wav"
        assert result.status == "success"
        assert result.error_message is None

    def test_file_result_error_status(self) -> None:
        """Test creating an error file result with message."""
        # TODO: Implement test
        # Should verify:
        # - file_name is set correctly
        # - status is "error"
        # - error_message contains the error details
        result = PipelineFileResult(
            file_name="audio.wav",
            status="error",
            error_message="Transcription failed: model error",
        )
        assert result.file_name == "audio.wav"
        assert result.status == "error"
        assert "Transcription failed" in result.error_message

    def test_file_result_skipped_status(self) -> None:
        """Test creating a skipped file result."""
        # TODO: Implement test
        # Should verify:
        # - status is "skipped"
        # - error_message is None (not an error)
        result = PipelineFileResult(
            file_name="audio.wav",
            status="skipped",
        )
        assert result.status == "skipped"
        assert result.error_message is None

    def test_file_result_diarized_only_status(self) -> None:
        """Test creating a diarized-only file result."""
        # TODO: Implement test
        # Should verify:
        # - status is "diarized_only"
        # - This indicates existing transcript was upgraded with diarization
        result = PipelineFileResult(
            file_name="audio.wav",
            status="diarized_only",
        )
        assert result.status == "diarized_only"


# ============================================================================
# PipelineBatchResult Tests
# ============================================================================


class TestPipelineBatchResult:
    """Tests for the PipelineBatchResult dataclass."""

    def test_batch_result_creation(self) -> None:
        """Test creating a batch result with basic statistics."""
        # TODO: Implement test
        # Should verify all fields are set correctly
        result = PipelineBatchResult(
            total_files=10,
            processed=7,
            skipped=2,
            diarized_only=0,
            failed=1,
            total_audio_seconds=120.0,
            total_time_seconds=30.0,
        )
        assert result.total_files == 10
        assert result.processed == 7
        assert result.skipped == 2
        assert result.failed == 1

    def test_batch_result_rtf_calculation(self) -> None:
        """Test real-time factor (RTF) calculation."""
        # TODO: Implement test
        # RTF = processing time / audio duration
        # Lower RTF means faster than real-time
        result = PipelineBatchResult(
            total_files=5,
            processed=5,
            skipped=0,
            diarized_only=0,
            failed=0,
            total_audio_seconds=100.0,
            total_time_seconds=20.0,  # 5x faster than real-time
        )
        assert result.overall_rtf == pytest.approx(0.2)

    def test_batch_result_rtf_zero_audio(self) -> None:
        """Test RTF calculation when no audio was processed."""
        # TODO: Implement test
        # Should return 0.0 to avoid division by zero
        result = PipelineBatchResult(
            total_files=0,
            processed=0,
            skipped=0,
            diarized_only=0,
            failed=0,
            total_audio_seconds=0.0,
            total_time_seconds=0.0,
        )
        assert result.overall_rtf == 0.0

    def test_batch_result_with_file_results(self) -> None:
        """Test batch result with per-file results attached."""
        # TODO: Implement test
        # Should verify file_results list is properly populated
        file_results = [
            PipelineFileResult(file_name="a.wav", status="success"),
            PipelineFileResult(file_name="b.wav", status="error", error_message="Failed"),
        ]
        result = PipelineBatchResult(
            total_files=2,
            processed=1,
            skipped=0,
            diarized_only=0,
            failed=1,
            total_audio_seconds=10.0,
            total_time_seconds=5.0,
            file_results=file_results,
        )
        assert len(result.file_results) == 2
        assert result.file_results[0].status == "success"
        assert result.file_results[1].status == "error"


# ============================================================================
# _get_duration_seconds Tests
# ============================================================================


class TestGetDurationSeconds:
    """Tests for the _get_duration_seconds helper function."""

    def test_valid_wav_file(self, tmp_path: Path) -> None:
        """Test duration extraction from a valid WAV file."""
        # TODO: Implement test
        # Should create a WAV file and verify duration is correct
        import wave

        wav_path = tmp_path / "test.wav"
        sample_rate = 16000
        duration_samples = 16000  # 1 second

        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00" * duration_samples * 2)

        duration = _get_duration_seconds(wav_path)
        assert duration == pytest.approx(1.0, rel=0.01)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test duration extraction returns 0 for missing file."""
        # TODO: Implement test
        # Should gracefully handle missing files
        missing_path = tmp_path / "nonexistent.wav"
        duration = _get_duration_seconds(missing_path)
        assert duration == 0.0

    def test_invalid_wav_file(self, tmp_path: Path) -> None:
        """Test duration extraction returns 0 for invalid WAV file."""
        # TODO: Implement test
        # Should gracefully handle corrupted/invalid files
        invalid_path = tmp_path / "invalid.wav"
        invalid_path.write_text("not a wav file")
        duration = _get_duration_seconds(invalid_path)
        assert duration == 0.0


# ============================================================================
# _build_meta Tests
# ============================================================================


class TestBuildMeta:
    """Tests for the _build_meta metadata construction function."""

    def test_basic_metadata_fields(
        self, sample_app_config: AppConfig, sample_transcript: Transcript
    ) -> None:
        """Test that basic metadata fields are populated."""
        # TODO: Implement test
        # Should verify generated_at, model_name, device, etc.
        meta = _build_meta(
            sample_app_config,
            sample_transcript,
            Path("/path/to/audio.wav"),
            duration_sec=10.5,
        )
        assert "generated_at" in meta
        assert meta["model_name"] == "tiny"
        assert meta["audio_duration_sec"] == 10.5

    def test_preserves_asr_backend_metadata(
        self, sample_app_config: AppConfig, sample_transcript: Transcript
    ) -> None:
        """Test that ASR backend metadata from transcript is preserved."""
        # TODO: Implement test
        # Should keep asr_backend, asr_device, asr_compute_type from transcript
        meta = _build_meta(
            sample_app_config,
            sample_transcript,
            Path("/path/to/audio.wav"),
            duration_sec=10.0,
        )
        assert meta.get("asr_backend") == "faster-whisper"
        assert meta.get("asr_device") == "cpu"

    def test_prefers_actual_runtime_device(self, sample_app_config: AppConfig) -> None:
        """Test that actual runtime device takes precedence over config."""
        # TODO: Implement test
        # If transcript.meta has asr_device, that should be used over config.asr.device
        # This ensures metadata reflects what actually ran, not what was requested
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[],
            meta={
                "asr_backend": "dummy",
                "asr_device": "cpu",
                "asr_compute_type": "n/a",
            },
        )
        # Config says cuda, but transcript says cpu was actually used
        sample_app_config.asr.device = "cuda"

        meta = _build_meta(
            sample_app_config,
            transcript,
            Path("/path/to/audio.wav"),
            duration_sec=5.0,
        )
        # Should reflect what actually ran (cpu), not what was requested (cuda)
        assert meta.get("device") == "cpu"


# ============================================================================
# run_pipeline Tests
# ============================================================================


class TestRunPipeline:
    """Tests for the run_pipeline orchestration function."""

    @pytest.mark.slow
    def test_empty_directory_returns_zero_files(self, sample_app_config: AppConfig) -> None:
        """Test pipeline with no audio files returns empty result."""
        # TODO: Implement test
        # Should handle empty input gracefully
        result = run_pipeline(sample_app_config)
        assert result.total_files == 0
        assert result.processed == 0
        assert result.skipped == 0
        assert result.failed == 0

    @pytest.mark.slow
    def test_skip_existing_json_enabled(
        self, sample_app_config: AppConfig, sample_transcript: Transcript
    ) -> None:
        """Test that skip_existing_json skips files with existing JSON output."""
        # TODO: Implement test
        # Should:
        # 1. Create a WAV in input_audio/
        # 2. Create corresponding JSON in whisper_json/
        # 3. Run pipeline with skip_existing_json=True
        # 4. Verify file was skipped
        pass  # Placeholder for implementation

    @pytest.mark.slow
    def test_diarization_upgrade_existing_transcript(self, sample_app_config: AppConfig) -> None:
        """Test that existing transcripts can be upgraded with diarization."""
        # TODO: Implement test
        # Should:
        # 1. Create existing transcript without diarization
        # 2. Run pipeline with enable_diarization=True
        # 3. Verify transcript was upgraded (diarized_only count)
        pass  # Placeholder for implementation

    @pytest.mark.slow
    @patch("transcription.asr_engine.TranscriptionEngine")
    def test_transcription_error_continues_batch(
        self, mock_engine_class: MagicMock, sample_app_config: AppConfig, tmp_path: Path
    ) -> None:
        """Test that transcription error for one file doesn't abort batch."""
        # TODO: Implement test
        # Should:
        # 1. Create multiple audio files
        # 2. Mock engine to fail on first file, succeed on second
        # 3. Verify both files are attempted
        # 4. Verify failed count is 1, processed count is 1
        pass  # Placeholder for implementation

    @pytest.mark.slow
    @patch("transcription.audio_io.normalize_all")
    def test_normalization_failure_handling(
        self, mock_normalize: MagicMock, sample_app_config: AppConfig
    ) -> None:
        """Test pipeline handles normalization failures gracefully."""
        # TODO: Implement test
        # Should:
        # 1. Mock normalize_all to raise an error
        # 2. Verify pipeline handles error appropriately
        pass  # Placeholder for implementation


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the pipeline with real dependencies."""

    @pytest.mark.slow
    @pytest.mark.requires_enrich
    def test_full_pipeline_with_mock_asr(
        self, sample_app_config: AppConfig, tmp_path: Path
    ) -> None:
        """Test full pipeline execution with mocked ASR engine."""
        # TODO: Implement test
        # Should:
        # 1. Create a real audio file
        # 2. Mock ASR engine but let other components run
        # 3. Verify JSON, TXT, SRT outputs are created
        pass  # Placeholder for implementation

    @pytest.mark.slow
    def test_pipeline_metadata_consistency(
        self, sample_app_config: AppConfig, tmp_path: Path
    ) -> None:
        """Test that pipeline produces consistent metadata across files."""
        # TODO: Implement test
        # Should:
        # 1. Process multiple files
        # 2. Verify all outputs have consistent metadata structure
        pass  # Placeholder for implementation


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestPipelineEdgeCases:
    """Edge case tests for the pipeline."""

    def test_wav_file_with_zero_duration(self, tmp_path: Path) -> None:
        """Test handling of zero-length audio files."""
        # TODO: Implement test
        # Should handle gracefully without crashing
        pass  # Placeholder for implementation

    def test_very_long_filename(self, tmp_path: Path) -> None:
        """Test handling of very long filenames."""
        # TODO: Implement test
        # Should handle filesystem limits appropriately
        pass  # Placeholder for implementation

    def test_special_characters_in_filename(self, tmp_path: Path) -> None:
        """Test handling of special characters in filenames."""
        # TODO: Implement test
        # Should sanitize or handle special characters
        pass  # Placeholder for implementation
