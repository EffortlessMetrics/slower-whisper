"""
Comprehensive error handling tests for slower-whisper.

This test suite verifies that the library raises appropriate exceptions with
clear error messages in error scenarios:

1. Missing input files (audio files, JSON transcripts, config files)
2. Invalid configuration values (invalid devices, negative integers, malformed JSON)
3. Missing optional dependencies (enrichment features without required packages)
4. Corrupted audio files (invalid WAV format, truncated files)
5. Invalid JSON transcripts (corrupted JSON, missing required fields, wrong schema)
6. CLI error exit codes (proper exit codes for different error types)
7. API exception raising (custom exceptions with descriptive messages)

All tests use the custom exception hierarchy from transcription.exceptions.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from transcription import (
    AsrConfig,
    EnrichmentConfig,
    TranscriptionConfig,
    enrich_directory,
    enrich_transcript,
    load_transcript,
    save_transcript,
    transcribe_directory,
    transcribe_file,
)
from transcription.cli import _config_from_transcribe_args
from transcription.cli import main as cli_main
from transcription.config import validate_diarization_settings
from transcription.exceptions import (
    ConfigurationError,
    EnrichmentError,
    SlowerWhisperError,
    TranscriptionError,
)
from transcription.models import Segment, Transcript

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_project_root(tmp_path):
    """
    Create a temporary project root with expected directory structure.
    """
    root = tmp_path / "project"
    root.mkdir()

    (root / "raw_audio").mkdir()
    (root / "input_audio").mkdir()
    (root / "whisper_json").mkdir()
    (root / "transcripts").mkdir()

    return root


@pytest.fixture
def valid_transcript():
    """Create a valid Transcript object for testing."""
    return Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=2.5,
                text="Hello world",
                speaker=None,
                tone=None,
                audio_state=None,
            )
        ],
        meta={"model_name": "large-v3", "device": "cuda"},
    )


@pytest.fixture
def test_audio_file(tmp_path):
    """Create a test WAV file with synthetic audio."""
    import numpy as np
    import soundfile as sf

    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 200 * t)

    audio_path = tmp_path / "test.wav"
    sf.write(audio_path, audio.astype(np.float32), sr)

    return audio_path


# ============================================================================
# 1. Missing Input Files Tests
# ============================================================================


class TestMissingInputFiles:
    """Test error handling for missing input files."""

    def test_transcribe_file_missing_audio(self, temp_project_root):
        """Test transcribe_file raises TranscriptionError when audio file is missing."""
        nonexistent = temp_project_root / "nonexistent.wav"
        config = TranscriptionConfig(model="base", device="cpu")

        with pytest.raises(TranscriptionError) as exc_info:
            transcribe_file(nonexistent, temp_project_root, config)

        assert "Audio file not found" in str(exc_info.value)
        assert str(nonexistent) in str(exc_info.value)

    def test_transcribe_file_rejects_directory(self, temp_project_root):
        """Test transcribe_file raises when given a directory path."""
        directory = temp_project_root / "not_an_audio_file.wav"
        directory.mkdir()

        config = TranscriptionConfig(model="base", device="cpu")

        with pytest.raises(TranscriptionError) as exc_info:
            transcribe_file(directory, temp_project_root, config)

        message = str(exc_info.value)
        assert "not a file" in message
        assert str(directory) in message

    def test_load_transcript_missing_json(self, tmp_path):
        """Test load_transcript raises TranscriptionError when JSON file is missing."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(TranscriptionError) as exc_info:
            load_transcript(nonexistent)

        assert "Transcript file not found" in str(exc_info.value)
        assert str(nonexistent) in str(exc_info.value)

    def test_enrich_transcript_missing_audio(self, valid_transcript, tmp_path):
        """Test enrich_transcript raises EnrichmentError when audio file is missing."""
        nonexistent = tmp_path / "nonexistent.wav"
        config = EnrichmentConfig(enable_prosody=True)

        with pytest.raises(EnrichmentError) as exc_info:
            enrich_transcript(valid_transcript, nonexistent, config)

        assert "Audio file not found" in str(exc_info.value)
        assert str(nonexistent) in str(exc_info.value)

    def test_enrich_directory_missing_json_directory(self, tmp_path):
        """Test enrich_directory raises EnrichmentError when JSON directory is missing."""
        root = tmp_path / "project"
        root.mkdir()
        # Don't create whisper_json directory

        config = EnrichmentConfig(enable_prosody=True)

        with pytest.raises(EnrichmentError) as exc_info:
            enrich_directory(root, config)

        assert "JSON directory does not exist" in str(exc_info.value)
        assert "whisper_json" in str(exc_info.value)

    def test_enrich_directory_missing_audio_directory(self, tmp_path):
        """Test enrich_directory raises EnrichmentError when audio directory is missing."""
        root = tmp_path / "project"
        root.mkdir()
        (root / "whisper_json").mkdir()
        # Don't create input_audio directory

        config = EnrichmentConfig(enable_prosody=True)

        with pytest.raises(EnrichmentError) as exc_info:
            enrich_directory(root, config)

        assert "Audio directory does not exist" in str(exc_info.value)
        assert "input_audio" in str(exc_info.value)

    def test_enrich_directory_no_json_files(self, tmp_path):
        """Test enrich_directory raises EnrichmentError when no JSON files found."""
        root = tmp_path / "project"
        root.mkdir()
        (root / "whisper_json").mkdir()
        (root / "input_audio").mkdir()
        # No JSON files created

        config = EnrichmentConfig(enable_prosody=True)

        with pytest.raises(EnrichmentError) as exc_info:
            enrich_directory(root, config)

        assert "No JSON transcript files found" in str(exc_info.value)
        assert "whisper_json" in str(exc_info.value)

    def test_transcribe_directory_no_transcripts_generated(self, temp_project_root):
        """Test transcribe_directory raises TranscriptionError when no transcripts generated."""
        # Create empty raw_audio directory (no audio files to transcribe)
        config = TranscriptionConfig(model="base", device="cpu")

        # Mock the pipeline to not generate any files
        with patch("transcription.pipeline.run_pipeline"):
            with pytest.raises(TranscriptionError) as exc_info:
                transcribe_directory(temp_project_root, config)

            assert "No transcripts found" in str(exc_info.value)
            assert "whisper_json" in str(exc_info.value)


# ============================================================================
# 2. Invalid Configuration Values Tests
# ============================================================================


class TestInvalidConfiguration:
    """Test error handling for invalid configuration values."""

    def _cli_args(self, **overrides):
        """Build a minimal argparse-like namespace for CLI config tests."""
        base = {
            "config": None,
            "model": None,
            "device": None,
            "compute_type": None,
            "language": None,
            "task": None,
            "vad_min_silence_ms": None,
            "beam_size": None,
            "skip_existing_json": None,
            "enable_diarization": True,
            "diarization_device": None,
            "min_speakers": None,
            "max_speakers": None,
            "overlap_threshold": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_config_from_file_missing_file(self, tmp_path):
        """Test TranscriptionConfig.from_file raises FileNotFoundError for missing file."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError) as exc_info:
            TranscriptionConfig.from_file(nonexistent)

        assert "Configuration file not found" in str(exc_info.value)

    def test_config_from_file_invalid_json(self, tmp_path):
        """Test TranscriptionConfig.from_file raises JSONDecodeError for malformed JSON."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text("{ invalid json }", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "Invalid JSON" in str(exc_info.value)

    def test_config_from_file_not_dict(self, tmp_path):
        """Test TranscriptionConfig.from_file raises ValueError for non-dict JSON."""
        config_file = tmp_path / "array_config.json"
        config_file.write_text('["not", "a", "dict"]', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "must contain a JSON object" in str(exc_info.value)

    def test_config_from_file_invalid_compute_type(self, tmp_path):
        """Invalid compute_type in config files should raise ValueError."""
        config_file = tmp_path / "bad_compute_type.json"
        config_file.write_text('{"compute_type": "fp99"}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "Invalid compute_type" in str(exc_info.value)

    def test_config_from_env_invalid_compute_type(self, monkeypatch):
        """Environment compute_type should be validated and rejected when unknown."""
        monkeypatch.setenv("SLOWER_WHISPER_COMPUTE_TYPE", "fp99")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_env()

        assert "Invalid compute_type" in str(exc_info.value)

    def test_asr_config_invalid_compute_type(self):
        """Direct AsrConfig construction should validate compute_type."""
        with pytest.raises(ConfigurationError) as exc_info:
            AsrConfig(compute_type="fp99")

        assert "Invalid compute_type" in str(exc_info.value)

    def test_config_invalid_task_value(self, tmp_path):
        """Test TranscriptionConfig.from_file raises ValueError for invalid task."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"task": "invalid_task"}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "Invalid task value" in str(exc_info.value)
        assert "transcribe" in str(exc_info.value)
        assert "translate" in str(exc_info.value)

    def test_config_invalid_device_value(self, tmp_path):
        """Test TranscriptionConfig.from_file raises ValueError for invalid device."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"device": "invalid_device"}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "Invalid device value" in str(exc_info.value)
        assert "cuda" in str(exc_info.value)
        assert "cpu" in str(exc_info.value)

    def test_config_invalid_diarization_device_value(self, tmp_path):
        """TranscriptionConfig.from_file should validate diarization_device choices."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"diarization_device": "tpu"}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "Invalid diarization_device value" in str(exc_info.value)
        assert "cuda" in str(exc_info.value)
        assert "cpu" in str(exc_info.value)
        assert "auto" in str(exc_info.value)

    def test_config_negative_vad_min_silence_ms(self, tmp_path):
        """Test TranscriptionConfig.from_file raises ValueError for negative VAD silence."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"vad_min_silence_ms": -100}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "vad_min_silence_ms must be a non-negative integer" in str(exc_info.value)

    def test_config_zero_beam_size(self, tmp_path):
        """Test TranscriptionConfig.from_file raises ValueError for zero beam_size."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"beam_size": 0}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "beam_size must be a positive integer" in str(exc_info.value)

    def test_config_invalid_skip_existing_json_type(self, tmp_path):
        """Config files should reject non-boolean skip_existing_json values."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"skip_existing_json": "yes"}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "skip_existing_json must be a boolean" in str(exc_info.value)

    def test_config_invalid_enable_diarization_type(self, tmp_path):
        """Config files should reject non-boolean enable_diarization values."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"enable_diarization": "true"}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "enable_diarization must be a boolean" in str(exc_info.value)

    def test_config_invalid_overlap_threshold_from_file(self, tmp_path):
        """Config files should reject out-of-range diarization overlap thresholds."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"enable_diarization": true, "overlap_threshold": 1.2}', encoding="utf-8"
        )

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "overlap_threshold" in str(exc_info.value)

    def test_config_invalid_overlap_threshold_type_from_file(self, tmp_path):
        """Config files should reject non-numeric diarization overlap thresholds."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"enable_diarization": true, "overlap_threshold": "high"}', encoding="utf-8"
        )

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        assert "overlap_threshold" in str(exc_info.value)

    def test_config_from_env_invalid_device(self, monkeypatch):
        """Test TranscriptionConfig.from_env raises ValueError for invalid device."""
        monkeypatch.setenv("SLOWER_WHISPER_DEVICE", "invalid_device")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_env()

        assert "Invalid SLOWER_WHISPER_DEVICE" in str(exc_info.value)

    def test_config_from_env_invalid_task(self, monkeypatch):
        """Test TranscriptionConfig.from_env raises ValueError for invalid task."""
        monkeypatch.setenv("SLOWER_WHISPER_TASK", "invalid_task")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_env()

        assert "Invalid SLOWER_WHISPER_TASK" in str(exc_info.value)

    def test_config_from_env_invalid_boolean(self, monkeypatch):
        """Test TranscriptionConfig.from_env raises ValueError for invalid boolean."""
        monkeypatch.setenv("SLOWER_WHISPER_SKIP_EXISTING_JSON", "maybe")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_env()

        assert "Invalid SLOWER_WHISPER_SKIP_EXISTING_JSON" in str(exc_info.value)

    def test_config_from_env_non_integer_vad(self, monkeypatch):
        """Test TranscriptionConfig.from_env raises ValueError for non-integer VAD."""
        monkeypatch.setenv("SLOWER_WHISPER_VAD_MIN_SILENCE_MS", "not_a_number")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_env()

        assert "Invalid SLOWER_WHISPER_VAD_MIN_SILENCE_MS" in str(exc_info.value)

    def test_config_from_env_invalid_diarization_device(self, monkeypatch):
        """Test TranscriptionConfig.from_env validates diarization device choices."""
        monkeypatch.setenv("SLOWER_WHISPER_DIARIZATION_DEVICE", "tpu")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_env()

        assert "Invalid SLOWER_WHISPER_DIARIZATION_DEVICE" in str(exc_info.value)

    def test_config_from_env_invalid_overlap_threshold(self, monkeypatch):
        """Test TranscriptionConfig.from_env validates overlap_threshold range."""
        monkeypatch.setenv("SLOWER_WHISPER_OVERLAP_THRESHOLD", "1.5")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_env()

        assert "Invalid SLOWER_WHISPER_OVERLAP_THRESHOLD" in str(exc_info.value)

    def test_config_from_env_invalid_min_speakers(self, monkeypatch):
        """Test TranscriptionConfig.from_env rejects non-positive min_speakers."""
        monkeypatch.setenv("SLOWER_WHISPER_MIN_SPEAKERS", "0")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_env()

        assert "Invalid SLOWER_WHISPER_MIN_SPEAKERS" in str(exc_info.value)

    def test_cli_invalid_min_speakers_raises_configuration_error(self):
        """CLI helper should reject non-positive min_speakers values."""
        args = self._cli_args(min_speakers=0)

        with pytest.raises(ConfigurationError) as exc_info:
            _config_from_transcribe_args(args)

        assert "min_speakers must be a positive integer" in str(exc_info.value)

    def test_cli_invalid_max_speakers_raises_configuration_error(self):
        """CLI helper should reject non-positive max_speakers values."""
        args = self._cli_args(max_speakers=-1)

        with pytest.raises(ConfigurationError) as exc_info:
            _config_from_transcribe_args(args)

        assert "max_speakers must be a positive integer" in str(exc_info.value)

    def test_validate_diarization_settings_invalid_overlap(self):
        """Helper should reject overlap thresholds outside [0.0, 1.0]."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_diarization_settings(
                min_speakers=None, max_speakers=None, overlap_threshold=1.5
            )

        assert "overlap_threshold" in str(exc_info.value)

    def test_cli_min_speakers_cannot_exceed_max(self):
        """CLI helper should enforce min_speakers <= max_speakers."""
        args = self._cli_args(min_speakers=4, max_speakers=2)

        with pytest.raises(ConfigurationError) as exc_info:
            _config_from_transcribe_args(args)

        assert "min_speakers (4) cannot be greater than max_speakers (2)" in str(exc_info.value)

    def test_transcribe_directory_rejects_inconsistent_speaker_bounds(self, temp_project_root):
        """Programmatic API should validate inconsistent diarization bounds."""
        config = TranscriptionConfig(
            model="base",
            device="cpu",
            enable_diarization=True,
            min_speakers=3,
            max_speakers=1,
        )

        with pytest.raises(ConfigurationError) as exc_info:
            transcribe_directory(temp_project_root, config)

        assert "min_speakers (3) cannot be greater than max_speakers (1)" in str(exc_info.value)

    def test_transcribe_file_rejects_inconsistent_speaker_bounds(
        self, temp_project_root, test_audio_file
    ):
        """transcribe_file should fail fast on invalid diarization bounds."""
        config = TranscriptionConfig(
            model="base",
            device="cpu",
            enable_diarization=True,
            min_speakers=5,
            max_speakers=2,
        )

        with pytest.raises(ConfigurationError) as exc_info:
            transcribe_file(test_audio_file, temp_project_root, config)

        assert "min_speakers (5) cannot be greater than max_speakers (2)" in str(exc_info.value)

    def test_enrich_config_from_file_invalid_device(self, tmp_path):
        """Test EnrichmentConfig.from_file raises ValueError for invalid device."""
        config_file = tmp_path / "enrich_config.json"
        config_file.write_text('{"device": "gpu"}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            EnrichmentConfig.from_file(config_file)

        assert "Invalid device value" in str(exc_info.value)

    def test_enrich_config_from_file_non_boolean_field(self, tmp_path):
        """Test EnrichmentConfig.from_file raises ValueError for non-boolean field."""
        config_file = tmp_path / "enrich_config.json"
        config_file.write_text('{"enable_prosody": "yes"}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            EnrichmentConfig.from_file(config_file)

        assert "enable_prosody must be a boolean" in str(exc_info.value)

    def test_enrich_config_from_env_invalid_boolean(self, monkeypatch):
        """Test EnrichmentConfig.from_env raises ValueError for invalid boolean."""
        monkeypatch.setenv("SLOWER_WHISPER_ENRICH_ENABLE_PROSODY", "maybe")

        with pytest.raises(ValueError) as exc_info:
            EnrichmentConfig.from_env()

        assert "Invalid SLOWER_WHISPER_ENRICH_ENABLE_PROSODY" in str(exc_info.value)


# ============================================================================
# 3. Missing Optional Dependencies Tests
# ============================================================================


class TestMissingDependencies:
    """Test error handling for missing optional dependencies.

    Note: These tests verify error messages when enrichment dependencies are missing.
    Due to complex import dependencies, we document the expected behavior rather than
    mocking at runtime.
    """

    @pytest.mark.skip(reason="Complex to mock import-time dependencies; verified manually")
    def test_enrich_transcript_missing_dependencies_documented(self):
        """
        Document expected behavior when enrichment dependencies are missing.

        Expected: EnrichmentError with message containing:
        - "Missing required dependencies for audio enrichment"
        - Installation instructions (uv sync --extra full or pip install -e '.[full]')

        To verify manually:
        1. Uninstall enrichment dependencies
        2. Call enrich_transcript() with enable_prosody=True
        3. Verify proper EnrichmentError is raised
        """
        pytest.skip("Manual verification required")

    @pytest.mark.skip(reason="Complex to mock import-time dependencies; verified manually")
    def test_enrich_directory_missing_dependencies_documented(self):
        """
        Document expected behavior when enrichment dependencies are missing.

        Expected: EnrichmentError with message containing:
        - "Missing required dependencies for audio enrichment"
        - Installation instructions (pip install -e '.[full]' or uv sync --extra full)

        To verify manually:
        1. Uninstall enrichment dependencies (librosa, parselmouth, etc.)
        2. Call enrich_directory() with enable_prosody=True
        3. Verify proper EnrichmentError is raised
        """
        pytest.skip("Manual verification required")


# ============================================================================
# 4. Corrupted Audio Files Tests
# ============================================================================


class TestCorruptedAudioFiles:
    """Test error handling for corrupted or invalid audio files."""

    @pytest.mark.skip(reason="Requires enrichment dependencies; tested in integration")
    def test_enrich_transcript_invalid_wav_file(self, valid_transcript, tmp_path):
        """Test enrich_transcript handles corrupted WAV file gracefully."""
        # Create a fake WAV file (not actually valid WAV format)
        fake_wav = tmp_path / "corrupted.wav"
        fake_wav.write_bytes(b"This is not a valid WAV file")

        config = EnrichmentConfig(enable_prosody=True)

        # The function should raise EnrichmentError with informative message
        with pytest.raises(EnrichmentError) as exc_info:
            enrich_transcript(valid_transcript, fake_wav, config)

        assert "Failed to enrich transcript" in str(exc_info.value)

    def test_transcribe_file_normalization_failure(self, temp_project_root, tmp_path):
        """Test transcribe_file raises TranscriptionError when audio normalization fails."""
        # Create a fake audio file
        fake_audio = tmp_path / "bad_audio.mp3"
        fake_audio.write_bytes(b"Not real audio")

        config = TranscriptionConfig(model="base", device="cpu")

        # Mock normalization to simulate failure (it doesn't create the normalized file)
        with patch("transcription.audio_io.normalize_all"):
            with pytest.raises(TranscriptionError) as exc_info:
                transcribe_file(fake_audio, temp_project_root, config)

            assert "Audio normalization failed" in str(exc_info.value)
            assert "ffmpeg" in str(exc_info.value)


# ============================================================================
# 5. Invalid JSON Transcripts Tests
# ============================================================================


class TestInvalidJSONTranscripts:
    """Test error handling for corrupted or invalid JSON transcript files."""

    def test_load_transcript_malformed_json(self, tmp_path):
        """Test load_transcript raises TranscriptionError for malformed JSON."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ invalid json content }", encoding="utf-8")

        with pytest.raises(TranscriptionError) as exc_info:
            load_transcript(bad_json)

        assert "Failed to load transcript" in str(exc_info.value)
        assert "corrupted or have an invalid schema" in str(exc_info.value)

    def test_load_transcript_missing_required_field(self, tmp_path):
        """Test load_transcript raises TranscriptionError when required fields missing."""
        incomplete_json = tmp_path / "incomplete.json"
        data = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "en",
            "segments": [
                {
                    # Missing required fields: id, start, end, text
                    "text": "Hello"
                }
            ],
        }
        incomplete_json.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(TranscriptionError) as exc_info:
            load_transcript(incomplete_json)

        assert "Failed to load transcript" in str(exc_info.value)

    def test_save_transcript_write_failure(self, valid_transcript, tmp_path):
        """Test save_transcript raises TranscriptionError when file cannot be written."""
        # Try to write to a directory (not a file)
        invalid_path = tmp_path / "test.json"
        invalid_path.mkdir()  # Make it a directory

        with pytest.raises(TranscriptionError) as exc_info:
            save_transcript(valid_transcript, invalid_path)

        assert "Failed to save transcript" in str(exc_info.value)

    def test_transcribe_directory_corrupted_json(self, temp_project_root):
        """Test transcribe_directory raises TranscriptionError when loading corrupted JSON."""
        # Create a corrupted JSON file in whisper_json
        bad_json = temp_project_root / "whisper_json" / "corrupted.json"
        bad_json.write_text("{ invalid }", encoding="utf-8")

        config = TranscriptionConfig(model="base", device="cpu")

        # Mock pipeline to not do anything (just test loading phase)
        with patch("transcription.pipeline.run_pipeline"):
            with pytest.raises(TranscriptionError) as exc_info:
                transcribe_directory(temp_project_root, config)

            assert "Failed to load transcript" in str(exc_info.value)
            assert "corrupted.json" in str(exc_info.value)


# ============================================================================
# 6. CLI Error Exit Codes Tests
# ============================================================================


class TestCLIErrorExitCodes:
    """Test that CLI returns appropriate exit codes for different error types."""

    def test_cli_missing_audio_returns_exit_code_1(self, temp_project_root):
        """Test CLI returns exit code 1 for SlowerWhisperError (missing audio)."""
        # Create empty raw_audio directory (no audio files)
        argv = ["transcribe", "--root", str(temp_project_root), "--device", "cpu"]

        # Mock the pipeline to trigger TranscriptionError
        with patch("transcription.pipeline.run_pipeline"):
            exit_code = cli_main(argv)

        assert exit_code == 1  # SlowerWhisperError exit code

    def test_cli_invalid_config_returns_exit_code_1(self, temp_project_root, tmp_path, capsys):
        """Test CLI returns exit code 1 for invalid transcription configuration."""
        # Create invalid config file
        bad_config = tmp_path / "bad_config.json"
        bad_config.write_text('{"device": "invalid"}', encoding="utf-8")

        argv = ["transcribe", "--root", str(temp_project_root), "--config", str(bad_config)]

        exit_code = cli_main(argv)
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "Invalid transcription configuration" in err

    def test_cli_runtime_error_returns_exit_code_2(self, temp_project_root):
        """Test CLI returns exit code 2 for runtime errors."""
        argv = ["transcribe", "--root", str(temp_project_root)]

        # Mock to raise an unexpected exception
        with patch(
            "transcription.api.transcribe_directory", side_effect=RuntimeError("Unexpected")
        ):
            exit_code = cli_main(argv)

        assert exit_code == 2  # Unexpected error exit code

    def test_cli_success_returns_exit_code_0(self, temp_project_root):
        """Test CLI returns exit code 0 on success."""
        argv = ["transcribe", "--root", str(temp_project_root), "--device", "cpu"]

        # Mock successful transcription at the CLI level
        mock_transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[],
            meta={},
        )

        # Mock both the pipeline and directory function
        with (
            patch("transcription.pipeline.run_pipeline"),
            patch("transcription.writers.load_transcript_from_json", return_value=mock_transcript),
        ):
            # Create a fake JSON file for the mock to find
            json_path = temp_project_root / "whisper_json" / "test.json"
            json_path.parent.mkdir(exist_ok=True)
            json_path.write_text("{}", encoding="utf-8")

        exit_code = cli_main(argv)

        assert exit_code == 0  # Success

    def test_cli_invalid_enrich_config_returns_exit_code_1(
        self, temp_project_root, tmp_path, capsys
    ):
        """Test CLI enrich command returns exit code 1 for invalid configuration."""
        bad_config = tmp_path / "bad_enrich.json"
        bad_config.write_text('{"device": "tpu"}', encoding="utf-8")

        argv = ["enrich", "--root", str(temp_project_root), "--enrich-config", str(bad_config)]

        exit_code = cli_main(argv)
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "Invalid enrichment configuration" in err

    def test_cli_enrich_missing_json_returns_exit_code_1(self, temp_project_root):
        """Test CLI enrich command returns exit code 1 when JSON directory missing."""
        # Remove whisper_json directory
        import shutil

        shutil.rmtree(temp_project_root / "whisper_json")

        argv = ["enrich", "--root", str(temp_project_root)]

        exit_code = cli_main(argv)
        assert exit_code == 1  # EnrichmentError exit code


# ============================================================================
# 7. API Exception Raising Tests
# ============================================================================


class TestAPIExceptionRaising:
    """Test that API functions raise appropriate custom exceptions."""

    def test_exception_hierarchy(self):
        """Test that custom exceptions inherit from SlowerWhisperError."""
        assert issubclass(TranscriptionError, SlowerWhisperError)
        assert issubclass(EnrichmentError, SlowerWhisperError)
        assert issubclass(ConfigurationError, SlowerWhisperError)
        assert issubclass(SlowerWhisperError, Exception)

    def test_transcription_error_descriptive_message(self, tmp_path):
        """Test TranscriptionError includes descriptive message."""
        nonexistent = tmp_path / "missing.wav"
        config = TranscriptionConfig()

        with pytest.raises(TranscriptionError) as exc_info:
            transcribe_file(nonexistent, tmp_path, config)

        # Verify message is descriptive
        error_msg = str(exc_info.value)
        assert "Audio file not found" in error_msg
        assert str(nonexistent) in error_msg
        assert "verify the file path" in error_msg.lower()

    def test_enrichment_error_descriptive_message(self, tmp_path):
        """Test EnrichmentError includes descriptive message."""
        root = tmp_path / "project"
        root.mkdir()
        # Missing whisper_json directory

        config = EnrichmentConfig()

        with pytest.raises(EnrichmentError) as exc_info:
            enrich_directory(root, config)

        # Verify message is descriptive
        error_msg = str(exc_info.value)
        assert "JSON directory does not exist" in error_msg
        assert "whisper_json" in error_msg
        assert "transcription first" in error_msg.lower()

    def test_error_chaining_preserves_original_exception(self, tmp_path):
        """Test that exceptions are properly chained with __cause__."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ invalid }", encoding="utf-8")

        with pytest.raises(TranscriptionError) as exc_info:
            load_transcript(bad_json)

        # Verify the original exception is preserved
        assert exc_info.value.__cause__ is not None

    @pytest.mark.skip(reason="Requires enrichment dependencies; tested in integration")
    def test_enrich_directory_partial_failures_reported(self, temp_project_root, valid_transcript):
        """Test enrich_directory reports partial failures when some files fail."""
        # Create two JSON files
        json1 = temp_project_root / "whisper_json" / "test1.json"
        json2 = temp_project_root / "whisper_json" / "test2.json"
        save_transcript(valid_transcript, json1)
        save_transcript(valid_transcript, json2)

        # Create audio file for only one of them
        import numpy as np
        import soundfile as sf

        audio_path = temp_project_root / "input_audio" / "test1.wav"
        sr = 16000
        audio = np.zeros(int(sr * 2.0), dtype=np.float32)
        sf.write(audio_path, audio, sr)
        # test2.wav is missing

        config = EnrichmentConfig(enable_prosody=True)

        # Mock the enrichment to succeed for available files
        with patch(
            "transcription.audio_enrichment.enrich_transcript_audio", return_value=valid_transcript
        ):
            # Should succeed for test1 but skip test2 (missing audio)
            # This should NOT raise, just skip the missing file
            enrich_directory(temp_project_root, config)

            # We only enriched the one file with available audio
            # (test2 was skipped due to missing audio, not an error)
            # Note: actual behavior depends on implementation details


# ============================================================================
# 8. Edge Cases and Boundary Conditions
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions for error handling."""

    def test_empty_transcript_saves_successfully(self, tmp_path):
        """Test that empty transcript (no segments) can be saved."""
        empty_transcript = Transcript(
            file_name="empty.wav",
            language="en",
            segments=[],
            meta={},
        )

        json_path = tmp_path / "empty.json"
        save_transcript(empty_transcript, json_path)

        # Verify it can be loaded back
        loaded = load_transcript(json_path)
        assert len(loaded.segments) == 0

    def test_transcript_with_null_audio_state_loads_successfully(self, tmp_path):
        """Test that transcript with null audio_state loads correctly."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="Test",
                    speaker=None,
                    tone=None,
                    audio_state=None,  # Explicitly null
                )
            ],
            meta={},
        )

        json_path = tmp_path / "test.json"
        save_transcript(transcript, json_path)

        loaded = load_transcript(json_path)
        assert loaded.segments[0].audio_state is None

    def test_config_with_all_defaults_validates(self):
        """Test that default configuration values are all valid."""
        # Should not raise any exceptions
        trans_config = TranscriptionConfig()
        assert trans_config.model == "large-v3"
        assert trans_config.device == "cuda"

        enrich_config = EnrichmentConfig()
        assert enrich_config.enable_prosody is True
        assert enrich_config.device == "cpu"

    def test_config_from_empty_env(self, monkeypatch):
        """Test loading config from environment with no env vars set."""
        # Clear all relevant env vars
        for key in list(monkeypatch._setattr):
            if isinstance(key, tuple) and key[0] == os and "SLOWER_WHISPER" in str(key[1]):
                monkeypatch.delenv(key[1], raising=False)

        # Should return defaults without error
        config = TranscriptionConfig.from_env()
        assert config.model == "large-v3"

    def test_unknown_config_fields_ignored(self, tmp_path):
        """Test that unknown fields in config file are silently ignored."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "model": "base",
                    "unknown_field": "value",
                    "another_unknown": 123,
                }
            ),
            encoding="utf-8",
        )

        # Should not raise, just ignore unknown fields
        config = TranscriptionConfig.from_file(config_file)
        assert config.model == "base"
        assert not hasattr(config, "unknown_field")


# ============================================================================
# 9. Error Message Quality Tests
# ============================================================================


class TestErrorMessageQuality:
    """Test that error messages are helpful and actionable."""

    def test_transcription_error_suggests_solution(self, tmp_path):
        """Test TranscriptionError suggests how to fix the problem."""
        nonexistent = tmp_path / "missing.json"

        with pytest.raises(TranscriptionError) as exc_info:
            load_transcript(nonexistent)

        error_msg = str(exc_info.value)
        # Should mention the file path
        assert str(nonexistent) in error_msg
        # Should suggest verification
        assert "ensure" in error_msg.lower() or "verify" in error_msg.lower()

    @pytest.mark.skip(reason="Requires enrichment dependencies; tested in integration")
    def test_enrichment_error_suggests_installation(self, valid_transcript, test_audio_file):
        """Test EnrichmentError suggests installing dependencies."""
        # This test is documented but skipped as it requires complex mocking
        # Expected behavior: EnrichmentError with installation instructions
        pytest.skip("Complex dependency mocking; verified in integration tests")

    def test_config_error_mentions_valid_values(self, tmp_path):
        """Test configuration errors mention what values are valid."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"device": "gpu"}', encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_file(config_file)

        error_msg = str(exc_info.value)
        # Should list valid values
        assert "cuda" in error_msg
        assert "cpu" in error_msg


# ============================================================================
# Test Summary
# ============================================================================

"""
Test Coverage Summary:

1. Missing Input Files (6 tests)
   - Missing audio files for transcription
   - Missing JSON files for loading
   - Missing audio files for enrichment
   - Missing directories (JSON, audio)
   - No files found in directories

2. Invalid Configuration Values (13 tests)
   - File-based config errors (missing file, bad JSON, wrong type)
   - Invalid enum values (task, device)
   - Invalid numeric values (negative, zero)
   - Environment variable validation
   - Boolean parsing errors

3. Missing Optional Dependencies (2 tests)
   - Import errors for enrichment modules
   - Helpful error messages with installation instructions

4. Corrupted Audio Files (2 tests)
   - Invalid WAV format handling
   - Audio normalization failures

5. Invalid JSON Transcripts (4 tests)
   - Malformed JSON
   - Missing required fields
   - Write failures
   - Directory-level corruption handling

6. CLI Error Exit Codes (5 tests)
   - Exit code 0 for success
   - Exit code 1 for SlowerWhisperError (domain errors)
   - Exit code 2 for unexpected errors
   - Different error scenarios

7. API Exception Raising (5 tests)
   - Exception hierarchy verification
   - Descriptive error messages
   - Exception chaining
   - Partial failure reporting

8. Edge Cases (6 tests)
   - Empty transcripts
   - Null values
   - Default configurations
   - Unknown fields handling

9. Error Message Quality (3 tests)
   - Actionable suggestions
   - Installation instructions
   - Valid value listings

Total: 46 comprehensive error handling tests
"""
