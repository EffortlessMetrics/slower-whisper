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
- Output generation (JSON, TXT, SRT)

These tests complement test_api_integration.py by testing the pipeline
internals directly rather than through the API layer.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcription.config import AppConfig, AsrConfig, Paths, TranscriptionConfig
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


@pytest.fixture
def create_wav_file(tmp_path: Path):
    """Factory fixture to create WAV files with configurable duration."""

    def _create_wav(
        path: Path | None = None,
        duration_sec: float = 1.0,
        sample_rate: int = 16000,
    ) -> Path:
        if path is None:
            path = tmp_path / "test.wav"

        # Create audio data
        num_samples = int(sample_rate * duration_sec)
        audio = np.zeros(num_samples, dtype=np.int16)

        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio.tobytes())

        return path

    return _create_wav


# ============================================================================
# PipelineFileResult Tests
# ============================================================================


class TestPipelineFileResult:
    """Tests for the PipelineFileResult dataclass."""

    def test_file_result_success_status(self) -> None:
        """Test creating a successful file result."""
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
        result = PipelineFileResult(
            file_name="audio.wav",
            status="error",
            error_message="Transcription failed: model error",
        )
        assert result.file_name == "audio.wav"
        assert result.status == "error"
        assert "Transcription failed" in result.error_message  # type: ignore[operator]

    def test_file_result_skipped_status(self) -> None:
        """Test creating a skipped file result."""
        result = PipelineFileResult(
            file_name="audio.wav",
            status="skipped",
        )
        assert result.status == "skipped"
        assert result.error_message is None

    def test_file_result_diarized_only_status(self) -> None:
        """Test creating a diarized-only file result."""
        result = PipelineFileResult(
            file_name="audio.wav",
            status="diarized_only",
        )
        assert result.status == "diarized_only"

    def test_file_result_default_error_message(self) -> None:
        """Test that error_message defaults to None."""
        result = PipelineFileResult(file_name="test.wav", status="success")
        assert result.error_message is None

    def test_file_result_all_valid_statuses(self) -> None:
        """Test all valid status values are accepted."""
        valid_statuses = ["success", "skipped", "diarized_only", "error"]
        for status in valid_statuses:
            result = PipelineFileResult(file_name="test.wav", status=status)
            assert result.status == status


# ============================================================================
# PipelineBatchResult Tests
# ============================================================================


class TestPipelineBatchResult:
    """Tests for the PipelineBatchResult dataclass."""

    def test_batch_result_creation(self) -> None:
        """Test creating a batch result with basic statistics."""
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
        assert result.diarized_only == 0
        assert result.total_audio_seconds == 120.0
        assert result.total_time_seconds == 30.0

    def test_batch_result_rtf_calculation(self) -> None:
        """Test real-time factor (RTF) calculation."""
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

    def test_batch_result_rtf_only_time_no_audio(self) -> None:
        """Test RTF calculation with processing time but no audio duration."""
        result = PipelineBatchResult(
            total_files=1,
            processed=0,
            skipped=1,
            diarized_only=0,
            failed=0,
            total_audio_seconds=0.0,
            total_time_seconds=5.0,
        )
        assert result.overall_rtf == 0.0

    def test_batch_result_with_file_results(self) -> None:
        """Test batch result with per-file results attached."""
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

    def test_batch_result_default_file_results(self) -> None:
        """Test that file_results defaults to empty list."""
        result = PipelineBatchResult(
            total_files=0,
            processed=0,
            skipped=0,
            diarized_only=0,
            failed=0,
            total_audio_seconds=0.0,
            total_time_seconds=0.0,
        )
        assert result.file_results == []

    def test_batch_result_with_diarized_only_count(self) -> None:
        """Test batch result with diarized_only files."""
        result = PipelineBatchResult(
            total_files=5,
            processed=2,
            skipped=1,
            diarized_only=2,
            failed=0,
            total_audio_seconds=60.0,
            total_time_seconds=15.0,
        )
        assert result.diarized_only == 2
        # Total should still match
        assert result.processed + result.skipped + result.diarized_only + result.failed == 5


# ============================================================================
# _get_duration_seconds Tests
# ============================================================================


class TestGetDurationSeconds:
    """Tests for the _get_duration_seconds helper function."""

    def test_valid_wav_file(self, create_wav_file: Any) -> None:
        """Test duration extraction from a valid WAV file."""
        wav_path = create_wav_file(duration_sec=1.0)
        duration = _get_duration_seconds(wav_path)
        assert duration == pytest.approx(1.0, rel=0.01)

    def test_valid_wav_file_longer_duration(self, create_wav_file: Any) -> None:
        """Test duration extraction from a longer WAV file."""
        wav_path = create_wav_file(duration_sec=5.5)
        duration = _get_duration_seconds(wav_path)
        assert duration == pytest.approx(5.5, rel=0.01)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test duration extraction returns 0 for missing file."""
        missing_path = tmp_path / "nonexistent.wav"
        duration = _get_duration_seconds(missing_path)
        assert duration == 0.0

    def test_invalid_wav_file(self, tmp_path: Path) -> None:
        """Test duration extraction returns 0 for invalid WAV file."""
        invalid_path = tmp_path / "invalid.wav"
        invalid_path.write_text("not a wav file")
        duration = _get_duration_seconds(invalid_path)
        assert duration == 0.0

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test duration extraction returns 0 for empty file."""
        empty_path = tmp_path / "empty.wav"
        empty_path.touch()
        duration = _get_duration_seconds(empty_path)
        assert duration == 0.0

    def test_wav_file_with_zero_samples(self, tmp_path: Path) -> None:
        """Test duration extraction for WAV file with zero samples."""
        wav_path = tmp_path / "zero_samples.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"")
        duration = _get_duration_seconds(wav_path)
        assert duration == 0.0

    def test_wav_file_different_sample_rates(self, tmp_path: Path, create_wav_file: Any) -> None:
        """Test duration extraction with different sample rates."""
        # Create 1 second at 44100 Hz
        wav_path = tmp_path / "high_rate.wav"
        wav_path = create_wav_file(path=wav_path, duration_sec=2.0, sample_rate=44100)
        duration = _get_duration_seconds(wav_path)
        assert duration == pytest.approx(2.0, rel=0.01)


# ============================================================================
# _build_meta Tests
# ============================================================================


class TestBuildMeta:
    """Tests for the _build_meta metadata construction function."""

    def test_basic_metadata_fields(
        self, sample_app_config: AppConfig, sample_transcript: Transcript
    ) -> None:
        """Test that basic metadata fields are populated."""
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

    def test_includes_beam_size_and_vad_settings(self, temp_project_structure: Paths) -> None:
        """Test that beam_size and VAD settings are included in metadata."""
        asr_cfg = AsrConfig(
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
            beam_size=7,
            vad_min_silence_ms=300,
            language="es",
        )
        app_config = AppConfig(
            paths=temp_project_structure,
            asr=asr_cfg,
            skip_existing_json=False,
        )
        transcript = Transcript(
            file_name="test.wav",
            language="es",
            segments=[],
            meta={
                "asr_backend": "faster-whisper",
                "asr_device": "cuda",
                "asr_compute_type": "float16",
            },
        )

        meta = _build_meta(app_config, transcript, Path("/path/to/audio.wav"), 30.0)

        assert meta["model_name"] == "large-v3"
        assert meta["beam_size"] == 7
        assert meta["vad_min_silence_ms"] == 300
        assert meta["language_hint"] == "es"

    def test_includes_audio_file_name(
        self, sample_app_config: AppConfig, sample_transcript: Transcript
    ) -> None:
        """Test that audio file name is recorded in metadata."""
        meta = _build_meta(
            sample_app_config,
            sample_transcript,
            Path("/some/path/to/interview.wav"),
            duration_sec=60.0,
        )
        assert meta["audio_file"] == "test.wav"

    def test_handles_missing_transcript_meta(self, sample_app_config: AppConfig) -> None:
        """Test handling transcript with no meta dict."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[],
            meta=None,
        )

        meta = _build_meta(
            sample_app_config,
            transcript,
            Path("/path/to/audio.wav"),
            duration_sec=10.0,
        )

        # Should still have basic fields
        assert "generated_at" in meta
        assert meta["model_name"] == "tiny"


# ============================================================================
# run_pipeline Tests
# ============================================================================


class TestRunPipeline:
    """Tests for the run_pipeline orchestration function."""

    def test_empty_directory_returns_zero_files(self, sample_app_config: AppConfig) -> None:
        """Test pipeline with no audio files returns empty result."""
        result = run_pipeline(sample_app_config)
        assert result.total_files == 0
        assert result.processed == 0
        assert result.skipped == 0
        assert result.failed == 0

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_skip_existing_json_enabled(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that skip_existing_json skips files with existing JSON output."""
        # Enable skip_existing
        sample_app_config.skip_existing_json = True

        # Create a WAV file in input_audio
        wav_path = sample_app_config.paths.norm_dir / "test_audio.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Create existing JSON in whisper_json
        json_path = sample_app_config.paths.json_dir / "test_audio.json"
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "file": "test_audio.wav",
                    "language": "en",
                    "meta": {},
                    "segments": [],
                }
            ),
            encoding="utf-8",
        )

        result = run_pipeline(sample_app_config)

        # File should be skipped since JSON already exists
        assert result.total_files == 1
        assert result.skipped == 1
        assert result.processed == 0
        # Engine should not have been used for transcription
        mock_engine_class.return_value.transcribe_file.assert_not_called()

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_skip_existing_json_disabled_processes_all(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        sample_transcript: Transcript,
        create_wav_file: Any,
    ) -> None:
        """Test that skip_existing_json=False processes files even with existing JSON."""
        sample_app_config.skip_existing_json = False

        # Create a WAV file in input_audio
        wav_path = sample_app_config.paths.norm_dir / "test_audio.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Create existing JSON
        json_path = sample_app_config.paths.json_dir / "test_audio.json"
        json_path.write_text('{"segments": []}', encoding="utf-8")

        # Mock the transcription engine
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        # File should be processed (not skipped)
        assert result.total_files == 1
        assert result.processed == 1
        assert result.skipped == 0

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_transcription_error_continues_batch(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        sample_transcript: Transcript,
        create_wav_file: Any,
    ) -> None:
        """Test that transcription error for one file doesn't abort batch."""
        # Create two WAV files
        wav1 = sample_app_config.paths.norm_dir / "audio1.wav"
        wav2 = sample_app_config.paths.norm_dir / "audio2.wav"
        create_wav_file(path=wav1, duration_sec=1.0)
        create_wav_file(path=wav2, duration_sec=1.0)

        # Mock engine to fail on first file, succeed on second
        mock_engine = MagicMock()
        call_count = 0

        def side_effect(path: Path) -> Transcript:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Model error on first file")
            return Transcript(
                file_name=path.name,
                language="en",
                segments=[Segment(id=0, start=0.0, end=1.0, text="Success")],
            )

        mock_engine.transcribe_file.side_effect = side_effect
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        # Both files should be attempted
        assert result.total_files == 2
        assert result.failed == 1
        assert result.processed == 1
        assert mock_engine.transcribe_file.call_count == 2

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_successful_transcription_writes_all_outputs(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        sample_transcript: Transcript,
        create_wav_file: Any,
    ) -> None:
        """Test that successful transcription writes JSON, TXT, and SRT files."""
        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "test.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Mock the transcription engine
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        # Check outputs were created
        json_path = sample_app_config.paths.json_dir / "test.json"
        txt_path = sample_app_config.paths.transcripts_dir / "test.txt"
        srt_path = sample_app_config.paths.transcripts_dir / "test.srt"

        assert json_path.exists()
        assert txt_path.exists()
        assert srt_path.exists()

        # Verify JSON content
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["file"] == "test.wav"
        assert data["language"] == "en"
        assert len(data["segments"]) == 2

        assert result.processed == 1

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_pipeline_records_timing_statistics(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        sample_transcript: Transcript,
        create_wav_file: Any,
    ) -> None:
        """Test that pipeline records timing and audio duration statistics."""
        # Create a 3-second WAV file
        wav_path = sample_app_config.paths.norm_dir / "long_audio.wav"
        create_wav_file(path=wav_path, duration_sec=3.0)

        # Mock the transcription engine
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert result.total_audio_seconds == pytest.approx(3.0, rel=0.1)
        assert result.total_time_seconds > 0  # Should have some processing time
        assert result.processed == 1

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_pipeline_handles_multiple_files(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test pipeline processes multiple files in sorted order."""
        # Create multiple WAV files
        files = ["alpha.wav", "beta.wav", "gamma.wav"]
        for name in files:
            wav_path = sample_app_config.paths.norm_dir / name
            create_wav_file(path=wav_path, duration_sec=1.0)

        # Track order of processing
        processed_order: list[str] = []

        def side_effect(path: Path) -> Transcript:
            processed_order.append(path.name)
            return Transcript(
                file_name=path.name,
                language="en",
                segments=[Segment(id=0, start=0.0, end=1.0, text="Test")],
            )

        mock_engine = MagicMock()
        mock_engine.transcribe_file.side_effect = side_effect
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert result.total_files == 3
        assert result.processed == 3
        # Files should be processed in sorted order
        assert processed_order == sorted(files)

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_pipeline_file_results_track_individual_outcomes(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that file_results list tracks each file's outcome."""
        # Create two files
        wav1 = sample_app_config.paths.norm_dir / "success.wav"
        wav2 = sample_app_config.paths.norm_dir / "failure.wav"
        create_wav_file(path=wav1, duration_sec=1.0)
        create_wav_file(path=wav2, duration_sec=1.0)

        def side_effect(path: Path) -> Transcript:
            if "failure" in str(path):
                raise RuntimeError("Intentional failure")
            return Transcript(
                file_name=path.name,
                language="en",
                segments=[],
            )

        mock_engine = MagicMock()
        mock_engine.transcribe_file.side_effect = side_effect
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert len(result.file_results) == 2

        # Find results by file name
        results_by_name = {r.file_name: r for r in result.file_results}

        # Check success file
        assert "success.wav" in results_by_name
        assert results_by_name["success.wav"].status == "success"

        # Check failure file
        assert "failure.wav" in results_by_name
        assert results_by_name["failure.wav"].status == "error"
        assert "Intentional failure" in results_by_name["failure.wav"].error_message  # type: ignore[operator]


# ============================================================================
# Diarization Integration Tests
# ============================================================================


class TestPipelineDiarization:
    """Tests for diarization integration in the pipeline."""

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_diarization_upgrade_existing_transcript(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        sample_transcript: Transcript,
        create_wav_file: Any,
    ) -> None:
        """Test that existing transcripts can be upgraded with diarization."""
        sample_app_config.skip_existing_json = True

        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "test_diar.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Create existing JSON without diarization
        json_path = sample_app_config.paths.json_dir / "test_diar.json"
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "file": "test_diar.wav",
                    "language": "en",
                    "meta": {},  # No diarization metadata
                    "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
                }
            ),
            encoding="utf-8",
        )

        # Create diarization config
        diar_config = TranscriptionConfig(enable_diarization=True)

        # Mock diarization to add speaker info (imported dynamically in pipeline)
        def add_diarization(
            transcript: Transcript, wav_path: Path, config: TranscriptionConfig
        ) -> Transcript:
            transcript.meta = transcript.meta or {}
            transcript.meta["diarization"] = {"status": "success"}
            return transcript

        with patch("transcription.api._maybe_run_diarization", side_effect=add_diarization):
            result = run_pipeline(sample_app_config, diarization_config=diar_config)

        assert result.total_files == 1
        assert result.diarized_only == 1
        assert result.processed == 0  # Not a new transcription

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_skip_already_diarized_transcript(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that already diarized transcripts are skipped."""
        sample_app_config.skip_existing_json = True

        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "already_diar.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Create existing JSON WITH diarization already done
        json_path = sample_app_config.paths.json_dir / "already_diar.json"
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "file": "already_diar.wav",
                    "language": "en",
                    "meta": {"diarization": {"status": "success"}},
                    "segments": [],
                }
            ),
            encoding="utf-8",
        )

        diar_config = TranscriptionConfig(enable_diarization=True)
        result = run_pipeline(sample_app_config, diarization_config=diar_config)

        assert result.total_files == 1
        assert result.skipped == 1
        assert result.diarized_only == 0

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_diarization_error_marks_file_failed(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that diarization errors during upgrade are captured as failures."""
        sample_app_config.skip_existing_json = True

        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "diar_fail.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Create existing JSON without diarization
        json_path = sample_app_config.paths.json_dir / "diar_fail.json"
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "file": "diar_fail.wav",
                    "language": "en",
                    "meta": {},
                    "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
                }
            ),
            encoding="utf-8",
        )

        diar_config = TranscriptionConfig(enable_diarization=True)

        # Mock diarization to fail
        with patch(
            "transcription.api._maybe_run_diarization",
            side_effect=RuntimeError("Diarization model error"),
        ):
            result = run_pipeline(sample_app_config, diarization_config=diar_config)

        assert result.total_files == 1
        assert result.failed == 1
        assert result.diarized_only == 0

        # Check error message
        assert len(result.file_results) == 1
        assert result.file_results[0].status == "error"
        assert "Diarization failed" in result.file_results[0].error_message  # type: ignore[operator]

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_json_load_error_during_diarization_upgrade(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that JSON load errors during diarization upgrade are handled."""
        sample_app_config.skip_existing_json = True

        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "bad_json.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Create invalid JSON file
        json_path = sample_app_config.paths.json_dir / "bad_json.json"
        json_path.write_text("{ invalid json content", encoding="utf-8")

        diar_config = TranscriptionConfig(enable_diarization=True)
        result = run_pipeline(sample_app_config, diarization_config=diar_config)

        assert result.total_files == 1
        assert result.failed == 1
        assert result.diarized_only == 0

        # Check error message
        assert len(result.file_results) == 1
        assert result.file_results[0].status == "error"
        assert "Load failed" in result.file_results[0].error_message  # type: ignore[operator]

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_chunking_enabled_updates_existing_transcript(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that enable_chunking updates existing transcripts."""
        sample_app_config.skip_existing_json = True

        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "chunk_test.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Create existing JSON
        json_path = sample_app_config.paths.json_dir / "chunk_test.json"
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "file": "chunk_test.wav",
                    "language": "en",
                    "meta": {},
                    "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
                }
            ),
            encoding="utf-8",
        )

        # Create config with chunking enabled (not diarization)
        chunk_config = TranscriptionConfig(enable_diarization=False)
        # Simulate enable_chunking attribute (not a standard field, dynamically set)
        chunk_config.enable_chunking = True  # type: ignore[attr-defined]

        # Mock chunk building
        def mock_build_chunks(transcript: Transcript, config: TranscriptionConfig) -> Transcript:
            transcript.chunks = [{"id": 0, "segments": [0]}]
            return transcript

        with patch("transcription.api._maybe_build_chunks", side_effect=mock_build_chunks):
            result = run_pipeline(sample_app_config, diarization_config=chunk_config)

        # File should be skipped since no diarization enabled
        assert result.total_files == 1
        assert result.skipped == 1

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_chunking_error_during_skip_existing(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
        caplog: Any,
    ) -> None:
        """Test that chunking errors during skip-existing are logged but don't fail."""
        import logging

        sample_app_config.skip_existing_json = True

        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "chunk_err.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Create existing JSON
        json_path = sample_app_config.paths.json_dir / "chunk_err.json"
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "file": "chunk_err.wav",
                    "language": "en",
                    "meta": {},
                    "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
                }
            ),
            encoding="utf-8",
        )

        # Create config with chunking enabled
        chunk_config = TranscriptionConfig(enable_diarization=False)
        chunk_config.enable_chunking = True  # type: ignore[attr-defined]

        # Mock chunk building to raise error
        with patch(
            "transcription.api._maybe_build_chunks",
            side_effect=RuntimeError("Chunk error"),
        ):
            with caplog.at_level(logging.ERROR):
                result = run_pipeline(sample_app_config, diarization_config=chunk_config)

        # File should still be skipped (chunking error doesn't fail the file)
        assert result.total_files == 1
        assert result.skipped == 1
        assert "Failed to update chunks" in caplog.text

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_diarization_with_new_transcription(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        sample_transcript: Transcript,
        create_wav_file: Any,
    ) -> None:
        """Test that diarization runs on new transcriptions."""
        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "new_diar.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Mock transcription engine
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine

        # Create diarization config
        diar_config = TranscriptionConfig(enable_diarization=True)

        # Mock diarization
        def add_diarization(
            transcript: Transcript, wav_path: Path, config: TranscriptionConfig
        ) -> Transcript:
            transcript.meta = transcript.meta or {}
            transcript.meta["diarization"] = {"status": "success", "num_speakers": 2}
            return transcript

        with patch("transcription.api._maybe_run_diarization", side_effect=add_diarization):
            result = run_pipeline(sample_app_config, diarization_config=diar_config)

        assert result.processed == 1
        assert result.diarized_only == 0  # It's a new transcription

        # Check JSON was written with diarization
        json_path = sample_app_config.paths.json_dir / "new_diar.json"
        assert json_path.exists()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["meta"].get("diarization", {}).get("status") == "success"

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_chunking_with_new_transcription(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        sample_transcript: Transcript,
        create_wav_file: Any,
    ) -> None:
        """Test that chunking runs on new transcriptions when enabled."""
        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "new_chunk.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Mock transcription engine
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine

        # Create config with both diarization and chunking enabled
        chunk_config = TranscriptionConfig(enable_diarization=True)
        chunk_config.enable_chunking = True  # type: ignore[attr-defined]

        # Mock diarization and chunking
        def add_diarization(
            transcript: Transcript, wav_path: Path, config: TranscriptionConfig
        ) -> Transcript:
            return transcript

        def add_chunks(transcript: Transcript, config: TranscriptionConfig) -> Transcript:
            transcript.chunks = [{"id": 0, "segments": [0, 1]}]
            return transcript

        with patch("transcription.api._maybe_run_diarization", side_effect=add_diarization):
            with patch("transcription.api._maybe_build_chunks", side_effect=add_chunks):
                result = run_pipeline(sample_app_config, diarization_config=chunk_config)

        assert result.processed == 1

        # Check JSON was written with chunks
        json_path = sample_app_config.paths.json_dir / "new_chunk.json"
        assert json_path.exists()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "chunks" in data


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestPipelineErrorHandling:
    """Tests for error handling in the pipeline."""

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_all_files_fail_returns_correct_counts(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that all-failures scenario returns correct statistics."""
        # Create two WAV files
        wav1 = sample_app_config.paths.norm_dir / "fail1.wav"
        wav2 = sample_app_config.paths.norm_dir / "fail2.wav"
        create_wav_file(path=wav1, duration_sec=1.0)
        create_wav_file(path=wav2, duration_sec=1.0)

        mock_engine = MagicMock()
        mock_engine.transcribe_file.side_effect = RuntimeError("All fail")
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert result.total_files == 2
        assert result.failed == 2
        assert result.processed == 0

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_file_results_contain_error_messages(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that error messages are captured in file results."""
        wav_path = sample_app_config.paths.norm_dir / "error_test.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        mock_engine = MagicMock()
        mock_engine.transcribe_file.side_effect = RuntimeError("Specific error message")
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert len(result.file_results) == 1
        assert result.file_results[0].status == "error"
        assert "Specific error message" in result.file_results[0].error_message  # type: ignore[operator]

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_transcription_exception_does_not_crash_pipeline(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        sample_transcript: Transcript,
        create_wav_file: Any,
    ) -> None:
        """Test that unexpected exceptions don't crash the entire pipeline."""
        wav1 = sample_app_config.paths.norm_dir / "a.wav"
        wav2 = sample_app_config.paths.norm_dir / "b.wav"
        wav3 = sample_app_config.paths.norm_dir / "c.wav"
        create_wav_file(path=wav1, duration_sec=1.0)
        create_wav_file(path=wav2, duration_sec=1.0)
        create_wav_file(path=wav3, duration_sec=1.0)

        call_count = 0

        def side_effect(path: Path) -> Transcript:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Unexpected error")
            return Transcript(
                file_name=path.name,
                language="en",
                segments=[],
            )

        mock_engine = MagicMock()
        mock_engine.transcribe_file.side_effect = side_effect
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert result.total_files == 3
        assert result.processed == 2
        assert result.failed == 1


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestPipelineEdgeCases:
    """Edge case tests for the pipeline."""

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_wav_file_with_zero_duration(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        tmp_path: Path,
    ) -> None:
        """Test handling of zero-length audio files."""
        # Create a zero-sample WAV file
        wav_path = sample_app_config.paths.norm_dir / "zero.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"")

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = Transcript(
            file_name="zero.wav",
            language="en",
            segments=[],
        )
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert result.total_files == 1
        assert result.processed == 1
        # Duration should be 0 but not cause errors
        assert result.total_audio_seconds == 0.0

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_special_characters_in_filename(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test handling of special characters in filenames."""
        # Create file with spaces and special chars (where valid on filesystem)
        wav_path = sample_app_config.paths.norm_dir / "test file (1).wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = Transcript(
            file_name="test file (1).wav",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="Test")],
        )
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert result.processed == 1

        # Check output files were created with correct names
        json_path = sample_app_config.paths.json_dir / "test file (1).json"
        assert json_path.exists()

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_non_wav_files_in_input_dir_are_ignored(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that non-WAV files in input_audio are ignored."""
        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "valid.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        # Create some non-WAV files
        (sample_app_config.paths.norm_dir / "readme.txt").write_text("ignore me")
        (sample_app_config.paths.norm_dir / "audio.mp3").write_bytes(b"fake mp3")

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = Transcript(
            file_name="valid.wav",
            language="en",
            segments=[],
        )
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        # Only the WAV file should be processed
        assert result.total_files == 1
        assert result.processed == 1

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_empty_transcript_segments(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test handling of transcript with no segments."""
        wav_path = sample_app_config.paths.norm_dir / "silent.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = Transcript(
            file_name="silent.wav",
            language="en",
            segments=[],  # No segments
        )
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert result.processed == 1

        # JSON should still be written with empty segments
        json_path = sample_app_config.paths.json_dir / "silent.json"
        assert json_path.exists()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["segments"] == []

    def test_subdirectories_in_input_are_ignored(
        self,
        sample_app_config: AppConfig,
    ) -> None:
        """Test that subdirectories in input_audio are not processed."""
        # Create a subdirectory (shouldn't be processed as a file)
        subdir = sample_app_config.paths.norm_dir / "subdir"
        subdir.mkdir()

        result = run_pipeline(sample_app_config)

        # Should not attempt to process the directory
        assert result.total_files == 0


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the pipeline with real dependencies (mocked ASR)."""

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_full_pipeline_with_mock_asr(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test full pipeline execution with mocked ASR engine."""
        # Create a WAV file
        wav_path = sample_app_config.paths.norm_dir / "integration_test.wav"
        create_wav_file(path=wav_path, duration_sec=2.0)

        # Create realistic transcript
        transcript = Transcript(
            file_name="integration_test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=0.8, text="Hello, this is a test."),
                Segment(id=1, start=0.8, end=1.5, text="Testing the pipeline."),
                Segment(id=2, start=1.5, end=2.0, text="End of test."),
            ],
            meta={
                "asr_backend": "faster-whisper",
                "asr_device": "cpu",
                "asr_compute_type": "int8",
            },
        )

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = transcript
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        # Verify success
        assert result.processed == 1
        assert result.failed == 0

        # Verify all outputs exist and have correct content
        json_path = sample_app_config.paths.json_dir / "integration_test.json"
        txt_path = sample_app_config.paths.transcripts_dir / "integration_test.txt"
        srt_path = sample_app_config.paths.transcripts_dir / "integration_test.srt"

        assert json_path.exists()
        assert txt_path.exists()
        assert srt_path.exists()

        # Verify JSON content
        with open(json_path, encoding="utf-8") as f:
            json_data = json.load(f)
        assert json_data["file"] == "integration_test.wav"
        assert len(json_data["segments"]) == 3
        assert json_data["meta"]["model_name"] == "tiny"

        # Verify TXT content
        txt_content = txt_path.read_text(encoding="utf-8")
        assert "Hello, this is a test." in txt_content
        assert "Testing the pipeline." in txt_content

        # Verify SRT content
        srt_content = srt_path.read_text(encoding="utf-8")
        assert "1\n" in srt_content  # First subtitle number
        assert "-->" in srt_content  # SRT timestamp separator
        assert "Hello, this is a test." in srt_content

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_pipeline_metadata_consistency(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that pipeline produces consistent metadata across files."""
        # Create multiple files
        for i in range(3):
            wav_path = sample_app_config.paths.norm_dir / f"file_{i}.wav"
            create_wav_file(path=wav_path, duration_sec=1.0)

        def create_transcript(path: Path) -> Transcript:
            return Transcript(
                file_name=path.name,
                language="en",
                segments=[Segment(id=0, start=0.0, end=1.0, text=f"File {path.stem}")],
                meta={
                    "asr_backend": "faster-whisper",
                    "asr_device": "cpu",
                    "asr_compute_type": "int8",
                },
            )

        mock_engine = MagicMock()
        mock_engine.transcribe_file.side_effect = create_transcript
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)

        assert result.processed == 3

        # Verify all JSON files have consistent metadata structure
        for i in range(3):
            json_path = sample_app_config.paths.json_dir / f"file_{i}.json"
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            # All files should have same metadata fields
            assert "generated_at" in data["meta"]
            assert data["meta"]["model_name"] == "tiny"
            assert data["meta"]["device"] == "cpu"
            assert data["schema_version"] == 2


# ============================================================================
# Model Configuration Tests
# ============================================================================


class TestPipelineModelConfiguration:
    """Tests for different model configurations in the pipeline."""

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_different_model_configurations(
        self,
        mock_engine_class: MagicMock,
        temp_project_structure: Paths,
        create_wav_file: Any,
    ) -> None:
        """Test pipeline with different model configurations."""
        configurations = [
            {"model_name": "tiny", "device": "cpu", "compute_type": "int8"},
            {"model_name": "base", "device": "cpu", "compute_type": "float32"},
            {"model_name": "large-v3", "device": "cuda", "compute_type": "float16"},
        ]

        for cfg_dict in configurations:
            # Create fresh directory structure
            for subdir in ["raw_audio", "input_audio", "whisper_json", "transcripts"]:
                d = temp_project_structure.root / subdir
                d.mkdir(parents=True, exist_ok=True)
                # Clean existing files
                for f in d.glob("*"):
                    if f.is_file():
                        f.unlink()

            asr_cfg = AsrConfig(**cfg_dict)  # type: ignore[arg-type]
            app_config = AppConfig(
                paths=temp_project_structure,
                asr=asr_cfg,
                skip_existing_json=False,
            )

            wav_path = temp_project_structure.norm_dir / "test.wav"
            create_wav_file(path=wav_path, duration_sec=1.0)

            mock_engine = MagicMock()
            mock_engine.transcribe_file.return_value = Transcript(
                file_name="test.wav",
                language="en",
                segments=[],
                meta={
                    "asr_backend": "faster-whisper",
                    "asr_device": cfg_dict["device"],
                    "asr_compute_type": cfg_dict["compute_type"],
                },
            )
            mock_engine_class.return_value = mock_engine

            result = run_pipeline(app_config)
            assert result.processed == 1

            # Verify engine was created with correct config
            actual_cfg = mock_engine_class.call_args[0][0]
            assert actual_cfg.model_name == cfg_dict["model_name"]
            assert actual_cfg.device == cfg_dict["device"]
            assert actual_cfg.compute_type == cfg_dict["compute_type"]

    @patch("transcription.pipeline.TranscriptionEngine")
    def test_language_configuration_passed_to_engine(
        self,
        mock_engine_class: MagicMock,
        sample_app_config: AppConfig,
        create_wav_file: Any,
    ) -> None:
        """Test that language configuration is passed to the ASR engine."""
        sample_app_config.asr.language = "es"

        wav_path = sample_app_config.paths.norm_dir / "spanish.wav"
        create_wav_file(path=wav_path, duration_sec=1.0)

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = Transcript(
            file_name="spanish.wav",
            language="es",
            segments=[Segment(id=0, start=0.0, end=1.0, text="Hola mundo")],
        )
        mock_engine_class.return_value = mock_engine

        result = run_pipeline(sample_app_config)
        assert result.processed == 1

        # Verify engine was created with Spanish language config
        actual_cfg = mock_engine_class.call_args[0][0]
        assert actual_cfg.language == "es"
