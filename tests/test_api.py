"""
Unit tests for transcription.api module.

This test module provides focused unit tests for the public API functions and
internal helper functions in transcription/api.py. It complements the integration
tests in test_api_integration.py with more granular coverage.

Test coverage targets:
- Internal helper functions (_neutral_audio_state, _turns_have_metadata, etc.)
- Edge cases for public API functions
- Error handling paths
- Validation logic
"""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from transcription.api import (
    _get_wav_duration_seconds,
    _maybe_build_chunks,
    _maybe_run_diarization,
    _neutral_audio_state,
    _run_semantic_annotator,
    _run_speaker_analytics,
    _turns_have_metadata,
    enrich_directory,
    enrich_transcript,
    load_transcript,
    save_transcript,
    transcribe_directory,
    transcribe_file,
)
from transcription.config import EnrichmentConfig, TranscriptionConfig
from transcription.exceptions import EnrichmentError, TranscriptionError
from transcription.models import Segment, Transcript, Turn

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_transcript() -> Transcript:
    """Create a simple transcript for testing."""
    return Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="Hello"),
            Segment(id=1, start=1.0, end=2.0, text="World"),
        ],
    )


@pytest.fixture
def transcript_with_turns() -> Transcript:
    """Create a transcript with Turn dataclass objects."""
    return Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="Hello", speaker={"id": "spk_0"}),
            Segment(id=1, start=1.0, end=2.0, text="Hi", speaker={"id": "spk_1"}),
        ],
        turns=[
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=1.0,
                text="Hello",
                metadata={"question_count": 0},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=1.0,
                end=2.0,
                text="Hi",
                metadata={"question_count": 1},
            ),
        ],
    )


@pytest.fixture
def transcript_with_dict_turns() -> Transcript:
    """Create a transcript with dict-based turns (not Turn dataclass)."""
    return Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="Hello"),
        ],
        turns=[
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "segment_ids": [0],
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
                "metadata": {"question_count": 0},
            }
        ],
    )


@pytest.fixture
def test_wav_file(tmp_path: Path) -> Path:
    """Create a minimal valid WAV file for testing."""
    wav_path = tmp_path / "test.wav"

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(16000)  # 16kHz

        # Write 1 second of silence (16000 samples)
        samples = struct.pack("h" * 16000, *[0] * 16000)
        wav.writeframes(samples)

    return wav_path


@pytest.fixture
def temp_project_structure(tmp_path: Path) -> Path:
    """Create a temporary project directory structure."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "raw_audio").mkdir()
    (root / "input_audio").mkdir()
    (root / "whisper_json").mkdir()
    (root / "transcripts").mkdir()
    return root


# ============================================================================
# Tests for _get_wav_duration_seconds()
# ============================================================================


class TestGetWavDurationSeconds:
    """Tests for the _get_wav_duration_seconds helper."""

    def test_valid_wav_file(self, test_wav_file: Path) -> None:
        """Test reading duration from a valid WAV file."""
        duration = _get_wav_duration_seconds(test_wav_file)
        assert duration == pytest.approx(1.0, rel=0.01)

    def test_multi_second_wav(self, tmp_path: Path) -> None:
        """Test reading duration from a longer WAV file."""
        wav_path = tmp_path / "long.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            # 3 seconds of audio
            samples = struct.pack("h" * 48000, *[0] * 48000)
            wav.writeframes(samples)

        duration = _get_wav_duration_seconds(wav_path)
        assert duration == pytest.approx(3.0, rel=0.01)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test handling of non-existent file."""
        missing = tmp_path / "missing.wav"
        duration = _get_wav_duration_seconds(missing)
        assert duration == 0.0

    def test_invalid_wav_file(self, tmp_path: Path) -> None:
        """Test handling of invalid (corrupted) WAV file."""
        invalid = tmp_path / "invalid.wav"
        invalid.write_text("not a wav file")
        duration = _get_wav_duration_seconds(invalid)
        assert duration == 0.0

    def test_empty_wav_file(self, tmp_path: Path) -> None:
        """Test handling of empty WAV file."""
        empty = tmp_path / "empty.wav"
        with wave.open(str(empty), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            # No frames written

        duration = _get_wav_duration_seconds(empty)
        assert duration == 0.0

    def test_zero_framerate_handling(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling of WAV file with zero framerate (edge case)."""
        # Create a valid WAV first
        wav_path = tmp_path / "zero_rate.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            samples = struct.pack("h" * 16000, *[0] * 16000)
            wav.writeframes(samples)

        # Monkey-patch wave.open to return a mock with zero framerate
        original_wave_open = wave.open

        def mock_wave_open(path: str, mode: str = "rb"):
            if mode == "rb":
                mock_wav = MagicMock()
                mock_wav.getnframes.return_value = 16000
                mock_wav.getframerate.return_value = 0  # Zero framerate
                mock_wav.__enter__ = lambda s: mock_wav
                mock_wav.__exit__ = lambda s, *args: None
                return mock_wav
            return original_wave_open(path, mode)

        monkeypatch.setattr(wave, "open", mock_wave_open)
        duration = _get_wav_duration_seconds(wav_path)
        assert duration == 0.0


# ============================================================================
# Tests for _neutral_audio_state()
# ============================================================================


class TestNeutralAudioState:
    """Tests for the _neutral_audio_state helper."""

    def test_neutral_audio_state_without_error(self) -> None:
        """Test creating neutral audio state without error message."""
        state = _neutral_audio_state()

        assert state["prosody"]["pitch"]["level"] == "unknown"
        assert state["prosody"]["pitch"]["mean_hz"] is None
        assert state["prosody"]["energy"]["level"] == "unknown"
        assert state["prosody"]["rate"]["level"] == "unknown"
        assert state["prosody"]["pauses"]["count"] == 0

        assert state["emotion"]["valence"]["level"] == "neutral"
        assert state["emotion"]["valence"]["score"] == 0.5
        assert state["emotion"]["arousal"]["level"] == "medium"
        assert state["emotion"]["dominance"]["level"] == "neutral"

        assert state["rendering"] == "[audio: neutral]"
        assert state["extraction_status"]["prosody"] == "skipped"
        assert state["extraction_status"]["emotion_dimensional"] == "skipped"
        assert state["extraction_status"]["errors"] == []

    def test_neutral_audio_state_with_error(self) -> None:
        """Test creating neutral audio state with error message."""
        error_msg = "Audio file corrupted"
        state = _neutral_audio_state(error_msg)

        assert state["extraction_status"]["errors"] == [error_msg]
        # Other fields should still have defaults
        assert state["prosody"]["pitch"]["level"] == "unknown"
        assert state["emotion"]["valence"]["level"] == "neutral"

    def test_neutral_audio_state_error_none(self) -> None:
        """Test that None error creates empty error list."""
        state = _neutral_audio_state(None)
        assert state["extraction_status"]["errors"] == []


# ============================================================================
# Tests for _turns_have_metadata()
# ============================================================================


class TestTurnsHaveMetadata:
    """Tests for the _turns_have_metadata helper."""

    def test_empty_turns(self) -> None:
        """Test with empty turns list."""
        assert _turns_have_metadata([]) is False
        assert _turns_have_metadata(None) is False

    def test_dict_turns_with_metadata(self) -> None:
        """Test dict-based turns with metadata key."""
        turns = [
            {"id": "turn_0", "metadata": {"question_count": 1}},
            {"id": "turn_1", "metadata": {"question_count": 0}},
        ]
        assert _turns_have_metadata(turns) is True

    def test_dict_turns_with_meta(self) -> None:
        """Test dict-based turns with 'meta' key (alternative name)."""
        turns = [
            {"id": "turn_0", "meta": {"question_count": 1}},
        ]
        assert _turns_have_metadata(turns) is True

    def test_dict_turns_without_metadata(self) -> None:
        """Test dict-based turns without metadata."""
        turns = [
            {"id": "turn_0", "text": "Hello"},
            {"id": "turn_1", "text": "Hi"},
        ]
        assert _turns_have_metadata(turns) is False

    def test_dict_turns_partial_metadata(self) -> None:
        """Test dict-based turns where only some have metadata."""
        turns = [
            {"id": "turn_0", "metadata": {"question_count": 1}},
            {"id": "turn_1", "text": "No metadata"},
        ]
        assert _turns_have_metadata(turns) is False

    def test_turn_dataclass_with_metadata_attr(self) -> None:
        """Test Turn dataclass objects with metadata attribute."""
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=1.0,
                text="Hello",
                metadata={"question_count": 1},
            ),
        ]
        assert _turns_have_metadata(turns) is True

    def test_turn_dataclass_without_metadata(self) -> None:
        """Test Turn dataclass with None metadata."""
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=1.0,
                text="Hello",
                metadata=None,
            ),
        ]
        assert _turns_have_metadata(turns) is False

    def test_object_with_meta_attribute(self) -> None:
        """Test with objects that have 'meta' attribute instead of 'metadata'."""

        class TurnLike:
            def __init__(self, meta: dict[str, Any] | None):
                self.meta = meta

        turns = [TurnLike({"question_count": 1})]
        assert _turns_have_metadata(turns) is True

        turns_no_meta = [TurnLike(None)]
        assert _turns_have_metadata(turns_no_meta) is False

    def test_empty_metadata_dict(self) -> None:
        """Test with empty metadata dict (considered as no metadata)."""
        turns = [{"id": "turn_0", "metadata": {}}]
        # Empty dict is falsy, so should return False
        assert _turns_have_metadata(turns) is False


# ============================================================================
# Tests for _run_speaker_analytics()
# ============================================================================


class TestRunSpeakerAnalytics:
    """Tests for the _run_speaker_analytics helper."""

    def test_analytics_disabled(self, simple_transcript: Transcript) -> None:
        """Test with both turn_metadata and speaker_stats disabled."""
        config = EnrichmentConfig(
            enable_turn_metadata=False,
            enable_speaker_stats=False,
        )
        result = _run_speaker_analytics(simple_transcript, config)
        # Should return transcript unchanged
        assert result is simple_transcript

    @patch("transcription.turns.build_turns")
    @patch("transcription.turns_enrich.enrich_turns_metadata")
    def test_turn_metadata_builds_turns_when_missing(
        self,
        mock_enrich: MagicMock,
        mock_build: MagicMock,
        simple_transcript: Transcript,
    ) -> None:
        """Test that turns are built when enable_turn_metadata=True and no turns exist."""
        mock_build.return_value = simple_transcript
        config = EnrichmentConfig(
            enable_turn_metadata=True,
            enable_speaker_stats=False,
        )

        _run_speaker_analytics(simple_transcript, config)

        mock_build.assert_called_once()
        mock_enrich.assert_called_once_with(simple_transcript)

    @patch("transcription.speaker_stats.compute_speaker_stats")
    @patch("transcription.turns_enrich.enrich_turns_metadata")
    @patch("transcription.turns.build_turns")
    def test_speaker_stats_computation(
        self,
        mock_build: MagicMock,
        mock_enrich: MagicMock,
        mock_stats: MagicMock,
        simple_transcript: Transcript,
    ) -> None:
        """Test speaker stats computation when enabled."""
        mock_build.return_value = simple_transcript
        config = EnrichmentConfig(
            enable_turn_metadata=True,
            enable_speaker_stats=True,
        )

        _run_speaker_analytics(simple_transcript, config)

        mock_stats.assert_called_once_with(simple_transcript)


# ============================================================================
# Tests for _run_semantic_annotator()
# ============================================================================


class TestRunSemanticAnnotator:
    """Tests for the _run_semantic_annotator helper."""

    def test_semantic_annotator_disabled(self, simple_transcript: Transcript) -> None:
        """Test with semantic annotator disabled."""
        config = EnrichmentConfig(enable_semantic_annotator=False)
        result = _run_semantic_annotator(simple_transcript, config)
        assert result is simple_transcript

    @patch("transcription.semantic.KeywordSemanticAnnotator")
    def test_semantic_annotator_default(
        self,
        mock_annotator_cls: MagicMock,
        simple_transcript: Transcript,
    ) -> None:
        """Test default KeywordSemanticAnnotator when none provided."""
        mock_annotator = MagicMock()
        mock_annotator.annotate.return_value = simple_transcript
        mock_annotator_cls.return_value = mock_annotator

        config = EnrichmentConfig(enable_semantic_annotator=True)

        result = _run_semantic_annotator(simple_transcript, config)

        mock_annotator_cls.assert_called_once()
        mock_annotator.annotate.assert_called_once_with(simple_transcript)
        assert result is simple_transcript

    def test_semantic_annotator_custom(self, simple_transcript: Transcript) -> None:
        """Test with custom semantic annotator provided."""
        mock_annotator = MagicMock()
        mock_annotator.annotate.return_value = simple_transcript

        config = EnrichmentConfig(
            enable_semantic_annotator=True,
            semantic_annotator=mock_annotator,
        )

        result = _run_semantic_annotator(simple_transcript, config)

        mock_annotator.annotate.assert_called_once_with(simple_transcript)
        assert result is simple_transcript

    def test_semantic_annotator_failure_handled(
        self, simple_transcript: Transcript, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test graceful handling when semantic annotator fails."""
        mock_annotator = MagicMock()
        mock_annotator.annotate.side_effect = RuntimeError("Annotator crashed")

        config = EnrichmentConfig(
            enable_semantic_annotator=True,
            semantic_annotator=mock_annotator,
        )

        import logging

        with caplog.at_level(logging.WARNING):
            result = _run_semantic_annotator(simple_transcript, config)

        # Should return original transcript on failure
        assert result is simple_transcript
        assert "Semantic annotator failed" in caplog.text

    def test_semantic_annotator_returns_none(self, simple_transcript: Transcript) -> None:
        """Test handling when annotator returns None."""
        mock_annotator = MagicMock()
        mock_annotator.annotate.return_value = None

        config = EnrichmentConfig(
            enable_semantic_annotator=True,
            semantic_annotator=mock_annotator,
        )

        result = _run_semantic_annotator(simple_transcript, config)
        # Should fall back to original transcript
        assert result is simple_transcript


# ============================================================================
# Tests for _maybe_build_chunks()
# ============================================================================


class TestMaybeBuildChunks:
    """Tests for the _maybe_build_chunks helper."""

    def test_chunking_disabled(self, simple_transcript: Transcript) -> None:
        """Test with chunking disabled."""
        config = TranscriptionConfig(enable_chunking=False)
        result = _maybe_build_chunks(simple_transcript, config)
        assert result is simple_transcript

    @patch("transcription.chunking.build_chunks")
    def test_chunking_enabled(
        self, mock_build_chunks: MagicMock, simple_transcript: Transcript
    ) -> None:
        """Test chunking when enabled."""
        config = TranscriptionConfig(
            enable_chunking=True,
            chunk_target_duration_s=30.0,
            chunk_max_duration_s=60.0,
            chunk_target_tokens=200,
            chunk_pause_split_threshold_s=1.0,
        )

        _maybe_build_chunks(simple_transcript, config)

        mock_build_chunks.assert_called_once()
        call_args = mock_build_chunks.call_args
        assert call_args[0][0] is simple_transcript
        chunk_cfg = call_args[0][1]
        assert chunk_cfg.target_duration_s == 30.0
        assert chunk_cfg.max_duration_s == 60.0


# ============================================================================
# Tests for _maybe_run_diarization()
# ============================================================================


class TestMaybeRunDiarization:
    """Tests for the _maybe_run_diarization helper."""

    def test_diarization_disabled(self, simple_transcript: Transcript, test_wav_file: Path) -> None:
        """Test with diarization disabled."""
        config = TranscriptionConfig(enable_diarization=False)

        result = _maybe_run_diarization(simple_transcript, test_wav_file, config)

        assert result.meta is not None
        assert result.meta["diarization"]["requested"] is False
        assert result.meta["diarization"]["status"] == "disabled"

    def test_diarization_disabled_preserves_existing_meta(
        self, simple_transcript: Transcript, test_wav_file: Path
    ) -> None:
        """Test that existing meta is preserved when diarization is disabled."""
        simple_transcript.meta = {"existing_key": "existing_value"}
        config = TranscriptionConfig(enable_diarization=False)

        result = _maybe_run_diarization(simple_transcript, test_wav_file, config)

        assert result.meta is not None
        assert result.meta["existing_key"] == "existing_value"
        assert result.meta["diarization"]["status"] == "disabled"

    @patch("transcription.diarization.assign_speakers")
    @patch("transcription.turns.build_turns")
    @patch("transcription.diarization.Diarizer")
    def test_diarization_success(
        self,
        mock_diarizer_cls: MagicMock,
        mock_build_turns: MagicMock,
        mock_assign: MagicMock,
        simple_transcript: Transcript,
        test_wav_file: Path,
    ) -> None:
        """Test successful diarization."""
        # Setup mocks
        mock_diarizer = MagicMock()
        mock_diarizer.run.return_value = [
            SimpleNamespace(speaker_id="spk_0", start=0.0, end=1.0),
            SimpleNamespace(speaker_id="spk_1", start=1.0, end=2.0),
        ]
        mock_diarizer_cls.return_value = mock_diarizer

        mock_assign.return_value = simple_transcript
        mock_build_turns.return_value = simple_transcript
        simple_transcript.speakers = [{"id": "spk_0"}, {"id": "spk_1"}]

        config = TranscriptionConfig(
            enable_diarization=True,
            diarization_device="cpu",
            min_speakers=1,
            max_speakers=2,
            overlap_threshold=0.5,
        )

        result = _maybe_run_diarization(simple_transcript, test_wav_file, config)

        assert result.meta is not None
        assert result.meta["diarization"]["status"] == "ok"
        assert result.meta["diarization"]["requested"] is True
        assert result.meta["diarization"]["backend"] == "pyannote.audio"
        mock_diarizer_cls.assert_called_once_with(device="cpu", min_speakers=1, max_speakers=2)

    @patch("transcription.diarization.Diarizer")
    def test_diarization_import_error(
        self,
        mock_diarizer_cls: MagicMock,
        simple_transcript: Transcript,
        test_wav_file: Path,
    ) -> None:
        """Test diarization failure due to import error."""
        mock_diarizer_cls.side_effect = ImportError("pyannote.audio not installed")

        config = TranscriptionConfig(enable_diarization=True)

        result = _maybe_run_diarization(simple_transcript, test_wav_file, config)

        assert result.meta is not None
        assert result.meta["diarization"]["status"] == "skipped"
        assert result.meta["diarization"]["error_type"] == "missing_dependency"

    @patch("transcription.diarization.Diarizer")
    def test_diarization_auth_error(
        self,
        mock_diarizer_cls: MagicMock,
        simple_transcript: Transcript,
        test_wav_file: Path,
    ) -> None:
        """Test diarization failure due to auth error."""
        mock_diarizer_cls.side_effect = RuntimeError("Missing HF_TOKEN for pyannote.audio")

        config = TranscriptionConfig(enable_diarization=True)

        result = _maybe_run_diarization(simple_transcript, test_wav_file, config)

        assert result.meta is not None
        assert result.meta["diarization"]["status"] == "skipped"
        assert result.meta["diarization"]["error_type"] == "auth"

    @patch("transcription.diarization.Diarizer")
    def test_diarization_file_not_found(
        self,
        mock_diarizer_cls: MagicMock,
        simple_transcript: Transcript,
        test_wav_file: Path,
    ) -> None:
        """Test diarization failure due to file not found."""
        mock_diarizer_cls.side_effect = FileNotFoundError("Audio file not found")

        config = TranscriptionConfig(enable_diarization=True)

        result = _maybe_run_diarization(simple_transcript, test_wav_file, config)

        assert result.meta is not None
        assert result.meta["diarization"]["status"] == "error"
        assert result.meta["diarization"]["error_type"] == "file_not_found"

    @patch("transcription.diarization.Diarizer")
    def test_diarization_restores_original_state_on_failure(
        self,
        mock_diarizer_cls: MagicMock,
        simple_transcript: Transcript,
        test_wav_file: Path,
    ) -> None:
        """Test that original speaker annotations are restored on diarization failure."""
        # Set up original speaker annotations
        simple_transcript.segments[0].speaker = {"id": "original_spk"}
        simple_transcript.speakers = [{"id": "original_spk"}]
        simple_transcript.turns = [{"id": "original_turn"}]

        mock_diarizer_cls.side_effect = RuntimeError("Diarization crashed")

        config = TranscriptionConfig(enable_diarization=True)

        result = _maybe_run_diarization(simple_transcript, test_wav_file, config)

        # Original state should be restored
        assert result.segments[0].speaker == {"id": "original_spk"}
        assert result.speakers == [{"id": "original_spk"}]
        assert result.turns == [{"id": "original_turn"}]


# ============================================================================
# Tests for load_transcript()
# ============================================================================


class TestLoadTranscript:
    """Tests for the load_transcript public API function."""

    def test_load_valid_transcript(self, simple_transcript: Transcript, tmp_path: Path) -> None:
        """Test loading a valid transcript file."""
        json_path = tmp_path / "transcript.json"
        save_transcript(simple_transcript, json_path)

        loaded = load_transcript(json_path)

        assert loaded.file_name == simple_transcript.file_name
        assert loaded.language == simple_transcript.language
        assert len(loaded.segments) == 2

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test loading a non-existent file raises TranscriptionError."""
        missing = tmp_path / "missing.json"

        with pytest.raises(TranscriptionError, match="Transcript file not found"):
            load_transcript(missing)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON raises TranscriptionError."""
        invalid = tmp_path / "invalid.json"
        invalid.write_text("{invalid json")

        with pytest.raises(TranscriptionError, match="Failed to load transcript"):
            load_transcript(invalid)

    def test_load_with_strict_validation(
        self, simple_transcript: Transcript, tmp_path: Path
    ) -> None:
        """Test loading with strict schema validation enabled."""
        json_path = tmp_path / "transcript.json"
        save_transcript(simple_transcript, json_path)

        # Should pass validation for valid transcript
        loaded = load_transcript(json_path, strict=True)
        assert loaded.file_name == simple_transcript.file_name

    @patch("transcription.validation.validate_transcript_json")
    def test_load_strict_validation_failure(self, mock_validate: MagicMock, tmp_path: Path) -> None:
        """Test loading with strict validation when validation fails."""
        mock_validate.return_value = (
            False,
            ["Missing required field: segments", "Invalid schema version"],
        )

        # Create a file that exists
        json_path = tmp_path / "transcript.json"
        json_path.write_text('{"file": "test.wav"}')

        with pytest.raises(TranscriptionError, match="Schema validation failed"):
            load_transcript(json_path, strict=True)


# ============================================================================
# Tests for save_transcript()
# ============================================================================


class TestSaveTranscript:
    """Tests for the save_transcript public API function."""

    def test_save_valid_transcript(self, simple_transcript: Transcript, tmp_path: Path) -> None:
        """Test saving a valid transcript."""
        json_path = tmp_path / "output.json"
        save_transcript(simple_transcript, json_path)

        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["file"] == "test.wav"
        assert data["language"] == "en"

    def test_save_creates_parent_directories(
        self, simple_transcript: Transcript, tmp_path: Path
    ) -> None:
        """Test that save_transcript creates parent directories."""
        nested_path = tmp_path / "nested" / "dirs" / "output.json"

        save_transcript(simple_transcript, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_overwrites_existing(self, simple_transcript: Transcript, tmp_path: Path) -> None:
        """Test that save_transcript overwrites existing files."""
        json_path = tmp_path / "output.json"
        json_path.write_text('{"old": "data"}')

        save_transcript(simple_transcript, json_path)

        data = json.loads(json_path.read_text())
        assert "old" not in data
        assert data["file"] == "test.wav"

    def test_save_permission_error(self, simple_transcript: Transcript, tmp_path: Path) -> None:
        """Test save_transcript handles permission errors."""
        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        try:
            with pytest.raises(TranscriptionError, match="Failed to save transcript"):
                save_transcript(simple_transcript, readonly_dir / "output.json")
        finally:
            readonly_dir.chmod(0o755)


# ============================================================================
# Tests for transcribe_file()
# ============================================================================


class TestTranscribeFile:
    """Tests for the transcribe_file public API function."""

    def test_missing_audio_file(self, temp_project_structure: Path) -> None:
        """Test transcribe_file raises error for missing audio."""
        config = TranscriptionConfig()
        missing = temp_project_structure / "missing.wav"

        with pytest.raises(TranscriptionError, match="Audio file not found"):
            transcribe_file(missing, temp_project_structure, config)

    def test_audio_path_is_directory(self, temp_project_structure: Path) -> None:
        """Test transcribe_file raises error when path is a directory."""
        config = TranscriptionConfig()
        directory = temp_project_structure / "raw_audio"

        with pytest.raises(TranscriptionError, match="not a file"):
            transcribe_file(directory, temp_project_structure, config)

    @patch("transcription.asr_engine.TranscriptionEngine")
    @patch("transcription.audio_io.normalize_all")
    def test_normalization_failure(
        self,
        mock_normalize: MagicMock,
        mock_engine_cls: MagicMock,
        test_wav_file: Path,
        temp_project_structure: Path,
    ) -> None:
        """Test transcribe_file raises error when normalization fails."""
        # normalize_all doesn't create the expected output file
        mock_normalize.return_value = None

        config = TranscriptionConfig()

        with pytest.raises(TranscriptionError, match="Audio normalization failed"):
            transcribe_file(test_wav_file, temp_project_structure, config)


# ============================================================================
# Tests for transcribe_directory()
# ============================================================================


class TestTranscribeDirectory:
    """Tests for the transcribe_directory public API function."""

    @patch("transcription.pipeline.run_pipeline")
    def test_no_transcripts_found(
        self, mock_pipeline: MagicMock, temp_project_structure: Path
    ) -> None:
        """Test transcribe_directory raises error when no transcripts generated."""
        mock_pipeline.return_value = SimpleNamespace(processed=0, skipped=0, failed=0)

        config = TranscriptionConfig()

        with pytest.raises(TranscriptionError, match="No transcripts found"):
            transcribe_directory(temp_project_structure, config)

    def test_invalid_diarization_settings(self, temp_project_structure: Path) -> None:
        """Test transcribe_directory validates diarization settings."""
        from transcription.exceptions import ConfigurationError

        config = TranscriptionConfig(
            min_speakers=5,
            max_speakers=2,  # Invalid: min > max
        )

        with pytest.raises(ConfigurationError, match="min_speakers"):
            transcribe_directory(temp_project_structure, config)


# ============================================================================
# Tests for enrich_transcript()
# ============================================================================


class TestEnrichTranscript:
    """Tests for the enrich_transcript public API function."""

    def test_missing_audio_file(self, simple_transcript: Transcript, tmp_path: Path) -> None:
        """Test enrich_transcript raises error for missing audio."""
        config = EnrichmentConfig()
        missing = tmp_path / "missing.wav"

        with pytest.raises(EnrichmentError, match="Audio file not found"):
            enrich_transcript(simple_transcript, missing, config)

    @patch("transcription.api._run_semantic_annotator")
    @patch("transcription.api._run_speaker_analytics")
    def test_skip_existing_fully_enriched(
        self,
        mock_analytics: MagicMock,
        mock_semantic: MagicMock,
        test_wav_file: Path,
    ) -> None:
        """Test skip_existing=True skips already enriched transcripts."""
        # Create fully enriched transcript
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="Hello",
                    audio_state={"prosody": {}, "emotion": {}},
                ),
            ],
        )

        config = EnrichmentConfig(
            skip_existing=True,
            enable_turn_metadata=False,
            enable_speaker_stats=False,
        )

        result = enrich_transcript(transcript, test_wav_file, config)

        # Should return same transcript without processing
        assert result is transcript

    @patch("transcription.audio_enrichment.enrich_transcript_audio")
    def test_enrichment_failure_graceful_degradation(
        self,
        mock_enrich: MagicMock,
        simple_transcript: Transcript,
        test_wav_file: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test graceful degradation when enrichment fails."""
        mock_enrich.side_effect = RuntimeError("Enrichment crashed")

        config = EnrichmentConfig(
            enable_turn_metadata=False,
            enable_speaker_stats=False,
        )

        import logging

        with caplog.at_level(logging.WARNING):
            result = enrich_transcript(simple_transcript, test_wav_file, config)

        # Should return transcript with neutral audio_state
        assert result.segments[0].audio_state is not None
        assert result.segments[0].audio_state["rendering"] == "[audio: neutral]"
        assert "Enrichment crashed" in result.segments[0].audio_state["extraction_status"]["errors"]
        assert "Enrichment failed" in caplog.text


# ============================================================================
# Tests for enrich_directory()
# ============================================================================


class TestEnrichDirectory:
    """Tests for the enrich_directory public API function."""

    def test_missing_json_directory(self, tmp_path: Path) -> None:
        """Test enrich_directory raises error when JSON dir doesn't exist."""
        root = tmp_path / "project"
        root.mkdir()
        (root / "input_audio").mkdir()
        # whisper_json not created

        config = EnrichmentConfig()

        with pytest.raises(EnrichmentError, match="JSON directory does not exist"):
            enrich_directory(root, config)

    def test_missing_audio_directory(self, tmp_path: Path) -> None:
        """Test enrich_directory raises error when audio dir doesn't exist."""
        root = tmp_path / "project"
        root.mkdir()
        (root / "whisper_json").mkdir()
        # input_audio not created

        config = EnrichmentConfig()

        with pytest.raises(EnrichmentError, match="Audio directory does not exist"):
            enrich_directory(root, config)

    def test_no_json_files(self, temp_project_structure: Path) -> None:
        """Test enrich_directory raises error when no JSON files exist."""
        config = EnrichmentConfig()

        with pytest.raises(EnrichmentError, match="No JSON transcript files found"):
            enrich_directory(temp_project_structure, config)

    def test_all_transcripts_fail_raises_error(self, temp_project_structure: Path) -> None:
        """Test enrich_directory raises error when all transcripts fail."""
        # Create transcript but no corresponding audio
        transcript = Transcript(file_name="test.wav", language="en", segments=[])
        json_path = temp_project_structure / "whisper_json" / "test.json"
        save_transcript(transcript, json_path)

        config = EnrichmentConfig()

        with pytest.raises(EnrichmentError, match="Failed to enrich any transcripts"):
            enrich_directory(temp_project_structure, config)


# ============================================================================
# Round-trip and Integration Tests
# ============================================================================


class TestRoundTrip:
    """Round-trip tests for save and load operations."""

    def test_transcript_with_all_fields_roundtrip(self, tmp_path: Path) -> None:
        """Test round-trip with transcript containing all optional fields."""
        transcript = Transcript(
            file_name="complete.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="Hello",
                    speaker={"id": "spk_0", "confidence": 0.95},
                    tone="neutral",
                    audio_state={
                        "prosody": {"pitch": {"level": "normal"}},
                        "emotion": {"valence": {"level": "positive"}},
                    },
                ),
            ],
            speakers=[{"id": "spk_0", "label": "Alice"}],
            turns=[
                {
                    "id": "turn_0",
                    "speaker_id": "spk_0",
                    "segment_ids": [0],
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello",
                }
            ],
            meta={"model_name": "large-v3", "version": "1.0"},
        )

        json_path = tmp_path / "complete.json"
        save_transcript(transcript, json_path)
        loaded = load_transcript(json_path)

        assert loaded.file_name == "complete.wav"
        assert loaded.language == "en"
        assert loaded.segments[0].speaker == {"id": "spk_0", "confidence": 0.95}
        assert loaded.segments[0].audio_state is not None
        assert loaded.speakers is not None
        assert len(loaded.speakers) == 1
        assert loaded.turns is not None
        assert len(loaded.turns) == 1

    def test_path_types_accepted(self, simple_transcript: Transcript, tmp_path: Path) -> None:
        """Test that both str and Path types are accepted."""
        json_path_obj = tmp_path / "output.json"
        json_path_str = str(json_path_obj)

        # Save with Path
        save_transcript(simple_transcript, json_path_obj)
        loaded = load_transcript(json_path_str)  # Load with str
        assert loaded.file_name == "test.wav"

        # Save with str
        json_path_obj2 = tmp_path / "output2.json"
        save_transcript(simple_transcript, str(json_path_obj2))
        loaded2 = load_transcript(json_path_obj2)  # Load with Path
        assert loaded2.file_name == "test.wav"
