"""
Unit tests for core data models (transcription/models.py).

This test module covers:
- Segment dataclass and its optional fields
- Transcript dataclass and its diarization fields
- Word dataclass for word-level timestamps (v1.8+)
- DiarizationMeta for speaker diarization metadata
- TurnMeta for turn-level conversational signals
- Turn dataclass and serialization
- Chunk dataclass for RAG-ready slices
- SpeakerStats and nested summary dataclasses
- BatchFileResult and BatchProcessingResult
- EnrichmentFileResult and EnrichmentBatchResult
- Schema version constants

Tests focus on:
- Dataclass instantiation with defaults
- to_dict() serialization
- from_dict() deserialization (where applicable)
- Edge cases and optional field handling
"""

from __future__ import annotations

from typing import Any

import pytest

from transcription.models import (
    AUDIO_STATE_VERSION,
    SCHEMA_VERSION,
    WORD_ALIGNMENT_VERSION,
    BatchFileResult,
    BatchProcessingResult,
    Chunk,
    DiarizationMeta,
    EnrichmentBatchResult,
    EnrichmentFileResult,
    ProsodySummary,
    Segment,
    SentimentSummary,
    SpeakerStats,
    Transcript,
    Turn,
    TurnMeta,
    Word,
)

# ============================================================================
# Schema Version Constants
# ============================================================================


class TestSchemaVersionConstants:
    """Tests for schema version constants."""

    def test_schema_version_is_integer(self) -> None:
        """SCHEMA_VERSION should be an integer."""
        assert isinstance(SCHEMA_VERSION, int)
        assert SCHEMA_VERSION == 2

    def test_audio_state_version_format(self) -> None:
        """AUDIO_STATE_VERSION should be a semantic version string."""
        assert isinstance(AUDIO_STATE_VERSION, str)
        parts = AUDIO_STATE_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_word_alignment_version_format(self) -> None:
        """WORD_ALIGNMENT_VERSION should be a semantic version string."""
        assert isinstance(WORD_ALIGNMENT_VERSION, str)
        parts = WORD_ALIGNMENT_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ============================================================================
# Word Dataclass Tests
# ============================================================================


class TestWord:
    """Tests for the Word dataclass (v1.8+ word-level timestamps)."""

    def test_word_creation_minimal(self) -> None:
        """Test Word creation with required fields only."""
        word = Word(word="hello", start=0.0, end=0.5)
        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.probability == 1.0  # default
        assert word.speaker is None  # default

    def test_word_creation_with_all_fields(self) -> None:
        """Test Word creation with all fields specified."""
        word = Word(
            word="hello",
            start=0.0,
            end=0.5,
            probability=0.95,
            speaker="spk_0",
        )
        assert word.word == "hello"
        assert word.probability == 0.95
        assert word.speaker == "spk_0"

    def test_word_to_dict_without_speaker(self) -> None:
        """Test Word.to_dict() excludes speaker when None."""
        word = Word(word="test", start=1.0, end=1.5, probability=0.8)
        d = word.to_dict()
        assert d == {
            "word": "test",
            "start": 1.0,
            "end": 1.5,
            "probability": 0.8,
        }
        assert "speaker" not in d

    def test_word_to_dict_with_speaker(self) -> None:
        """Test Word.to_dict() includes speaker when set."""
        word = Word(word="test", start=1.0, end=1.5, probability=0.8, speaker="spk_1")
        d = word.to_dict()
        assert d["speaker"] == "spk_1"

    def test_word_from_dict_minimal(self) -> None:
        """Test Word.from_dict() with minimal data."""
        d = {"word": "hello", "start": 0.0, "end": 0.5}
        word = Word.from_dict(d)
        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.probability == 1.0
        assert word.speaker is None

    def test_word_from_dict_full(self) -> None:
        """Test Word.from_dict() with all fields."""
        d = {
            "word": "hello",
            "start": 0.0,
            "end": 0.5,
            "probability": 0.9,
            "speaker": "spk_0",
        }
        word = Word.from_dict(d)
        assert word.probability == 0.9
        assert word.speaker == "spk_0"

    def test_word_from_dict_missing_fields(self) -> None:
        """Test Word.from_dict() handles missing fields with defaults."""
        d: dict[str, Any] = {}
        word = Word.from_dict(d)
        assert word.word == ""
        assert word.start == 0.0
        assert word.end == 0.0
        assert word.probability == 1.0


# ============================================================================
# Segment Dataclass Tests
# ============================================================================


class TestSegment:
    """Tests for the Segment dataclass."""

    def test_segment_creation_minimal(self) -> None:
        """Test Segment creation with required fields only."""
        seg = Segment(id=0, start=0.0, end=1.5, text="Hello world")
        assert seg.id == 0
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "Hello world"
        assert seg.speaker is None
        assert seg.tone is None
        assert seg.audio_state is None
        assert seg.words is None

    def test_segment_with_speaker(self) -> None:
        """Test Segment with speaker diarization data."""
        speaker_data = {"id": "spk_0", "confidence": 0.95}
        seg = Segment(
            id=0,
            start=0.0,
            end=1.5,
            text="Hello",
            speaker=speaker_data,
        )
        assert seg.speaker == speaker_data
        assert seg.speaker["id"] == "spk_0"
        assert seg.speaker["confidence"] == 0.95

    def test_segment_with_audio_state(self) -> None:
        """Test Segment with enriched audio_state data."""
        audio_state = {
            "prosody": {
                "pitch": {"level": "high", "mean_hz": 250.0},
                "energy": {"level": "loud", "db_rms": -8.5},
            },
            "emotion": {
                "valence": {"level": "positive", "score": 0.7},
            },
            "rendering": "[audio: high pitch, loud]",
            "extraction_status": {"prosody": "success"},
        }
        seg = Segment(
            id=0,
            start=0.0,
            end=1.5,
            text="Hello",
            audio_state=audio_state,
        )
        assert seg.audio_state is not None
        assert seg.audio_state["prosody"]["pitch"]["level"] == "high"
        assert "rendering" in seg.audio_state

    def test_segment_with_words(self) -> None:
        """Test Segment with word-level timestamps (v1.8+)."""
        words = [
            Word(word="Hello", start=0.0, end=0.3, probability=0.95),
            Word(word="world", start=0.35, end=0.7, probability=0.92),
        ]
        seg = Segment(
            id=0,
            start=0.0,
            end=0.7,
            text="Hello world",
            words=words,
        )
        assert seg.words is not None
        assert len(seg.words) == 2
        assert seg.words[0].word == "Hello"
        assert seg.words[1].word == "world"

    def test_segment_all_fields(self) -> None:
        """Test Segment with all optional fields populated."""
        seg = Segment(
            id=5,
            start=10.0,
            end=15.5,
            text="Testing all fields",
            speaker={"id": "spk_1", "confidence": 0.88},
            tone="excited",
            audio_state={"prosody": {}, "rendering": "neutral"},
            words=[Word(word="Testing", start=10.0, end=10.5, probability=0.99)],
        )
        assert seg.id == 5
        assert seg.speaker is not None
        assert seg.tone == "excited"
        assert seg.audio_state is not None
        assert seg.words is not None


# ============================================================================
# Transcript Dataclass Tests
# ============================================================================


class TestTranscript:
    """Tests for the Transcript dataclass."""

    def test_transcript_creation_minimal(self) -> None:
        """Test Transcript creation with required fields only."""
        transcript = Transcript(file_name="test.wav", language="en")
        assert transcript.file_name == "test.wav"
        assert transcript.language == "en"
        assert transcript.segments == []
        assert transcript.meta is None
        assert transcript.speakers is None
        assert transcript.turns is None
        assert transcript.speaker_stats is None
        assert transcript.chunks is None

    def test_transcript_with_segments(self) -> None:
        """Test Transcript with populated segments."""
        segments = [
            Segment(id=0, start=0.0, end=1.0, text="Hello"),
            Segment(id=1, start=1.0, end=2.0, text="World"),
        ]
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=segments,
        )
        assert len(transcript.segments) == 2
        assert transcript.segments[0].text == "Hello"

    def test_transcript_with_metadata(self) -> None:
        """Test Transcript with metadata dictionary."""
        meta = {
            "generated_at": "2024-12-31T00:00:00Z",
            "model_name": "large-v3",
            "device": "cuda",
            "pipeline_version": "1.8.0",
        }
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            meta=meta,
        )
        assert transcript.meta is not None
        assert transcript.meta["model_name"] == "large-v3"

    def test_transcript_with_diarization_fields(self) -> None:
        """Test Transcript with v1.1+ diarization fields."""
        speakers = [
            {"id": "spk_0", "label": None, "total_speech_time": 10.5, "num_segments": 5},
            {"id": "spk_1", "label": None, "total_speech_time": 8.2, "num_segments": 4},
        ]
        turns = [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "start": 0.0,
                "end": 5.0,
                "segment_ids": [0, 1],
                "text": "Hello there",
            }
        ]
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            speakers=speakers,
            turns=turns,
        )
        assert transcript.speakers is not None
        assert len(transcript.speakers) == 2
        assert transcript.turns is not None
        assert len(transcript.turns) == 1

    def test_transcript_with_annotations(self) -> None:
        """Test Transcript with semantic annotations."""
        annotations = {
            "semantic": {
                "topics": ["meeting", "project"],
                "action_items": ["Review document"],
            }
        }
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            annotations=annotations,
        )
        assert transcript.annotations is not None
        assert "semantic" in transcript.annotations


# ============================================================================
# DiarizationMeta Tests
# ============================================================================


class TestDiarizationMeta:
    """Tests for the DiarizationMeta dataclass."""

    def test_diarization_meta_disabled(self) -> None:
        """Test DiarizationMeta for disabled diarization."""
        meta = DiarizationMeta(
            requested=False,
            status="disabled",
        )
        assert meta.requested is False
        assert meta.status == "disabled"
        assert meta.backend is None
        assert meta.num_speakers is None

    def test_diarization_meta_success(self) -> None:
        """Test DiarizationMeta for successful diarization."""
        meta = DiarizationMeta(
            requested=True,
            status="ok",
            backend="pyannote.audio",
            num_speakers=3,
        )
        assert meta.requested is True
        assert meta.status == "ok"
        assert meta.backend == "pyannote.audio"
        assert meta.num_speakers == 3

    def test_diarization_meta_error(self) -> None:
        """Test DiarizationMeta for failed diarization."""
        meta = DiarizationMeta(
            requested=True,
            status="error",
            backend="pyannote.audio",
            error_type="auth",
            error="Missing HF_TOKEN",
        )
        assert meta.status == "error"
        assert meta.error_type == "auth"
        assert meta.error == "Missing HF_TOKEN"

    def test_diarization_meta_message_error_sync(self) -> None:
        """Test that message and error fields stay in sync."""
        # When error is set, message should also be set
        meta = DiarizationMeta(
            requested=True,
            status="error",
            error="Test error",
        )
        assert meta.message == "Test error"
        assert meta.error == "Test error"

    def test_diarization_meta_to_dict(self) -> None:
        """Test DiarizationMeta.to_dict() serialization."""
        meta = DiarizationMeta(
            requested=True,
            status="ok",
            backend="pyannote.audio",
            num_speakers=2,
        )
        d = meta.to_dict()
        assert d["requested"] is True
        assert d["status"] == "ok"
        assert d["backend"] == "pyannote.audio"
        assert d["num_speakers"] == 2


# ============================================================================
# TurnMeta Tests
# ============================================================================


class TestTurnMeta:
    """Tests for the TurnMeta dataclass."""

    def test_turn_meta_defaults(self) -> None:
        """Test TurnMeta with default values."""
        meta = TurnMeta()
        assert meta.question_count == 0
        assert meta.interruption_started_here is False
        assert meta.avg_pause_ms is None
        assert meta.disfluency_ratio is None

    def test_turn_meta_with_values(self) -> None:
        """Test TurnMeta with conversational signals."""
        meta = TurnMeta(
            question_count=3,
            interruption_started_here=True,
            avg_pause_ms=250.5,
            disfluency_ratio=0.05,
        )
        assert meta.question_count == 3
        assert meta.interruption_started_here is True
        assert meta.avg_pause_ms == 250.5
        assert meta.disfluency_ratio == 0.05

    def test_turn_meta_to_dict(self) -> None:
        """Test TurnMeta.to_dict() serialization."""
        meta = TurnMeta(question_count=2, avg_pause_ms=100.0)
        d = meta.to_dict()
        assert d["question_count"] == 2
        assert d["avg_pause_ms"] == 100.0


# ============================================================================
# Turn Dataclass Tests
# ============================================================================


class TestTurn:
    """Tests for the Turn dataclass."""

    def test_turn_creation(self) -> None:
        """Test Turn creation with all fields."""
        turn = Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0, 1, 2],
            start=0.0,
            end=5.5,
            text="Hello there, how are you?",
        )
        assert turn.id == "turn_0"
        assert turn.speaker_id == "spk_0"
        assert turn.segment_ids == [0, 1, 2]
        assert turn.text == "Hello there, how are you?"

    def test_turn_with_metadata(self) -> None:
        """Test Turn with enriched metadata."""
        turn = Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0],
            start=0.0,
            end=1.0,
            text="Hello",
            metadata={"question_count": 1},
        )
        assert turn.metadata is not None
        assert turn.metadata["question_count"] == 1

    def test_turn_to_dict(self) -> None:
        """Test Turn.to_dict() serialization."""
        turn = Turn(
            id="turn_0",
            speaker_id="spk_1",
            segment_ids=[0, 1],
            start=0.0,
            end=2.0,
            text="Test",
            metadata={"key": "value"},
        )
        d = turn.to_dict()
        assert d["id"] == "turn_0"
        assert d["speaker_id"] == "spk_1"
        assert d["segment_ids"] == [0, 1]
        assert d["metadata"] == {"key": "value"}

    def test_turn_from_dict(self) -> None:
        """Test Turn.from_dict() deserialization."""
        d = {
            "id": "turn_5",
            "speaker_id": "spk_2",
            "segment_ids": [10, 11],
            "start": 30.0,
            "end": 35.0,
            "text": "Testing from_dict",
            "metadata": {"test": True},
        }
        turn = Turn.from_dict(d)
        assert turn.id == "turn_5"
        assert turn.speaker_id == "spk_2"
        assert turn.text == "Testing from_dict"

    def test_turn_from_dict_legacy_speaker_field(self) -> None:
        """Test Turn.from_dict() handles legacy 'speaker' field."""
        d = {
            "id": "turn_0",
            "speaker": "spk_legacy",  # Legacy field name
            "segment_ids": [0],
            "start": 0.0,
            "end": 1.0,
            "text": "Test",
        }
        turn = Turn.from_dict(d)
        assert turn.speaker_id == "spk_legacy"


# ============================================================================
# Chunk Dataclass Tests
# ============================================================================


class TestChunk:
    """Tests for the Chunk dataclass (RAG-ready slices)."""

    def test_chunk_creation_minimal(self) -> None:
        """Test Chunk creation with minimal fields."""
        chunk = Chunk(id="chunk_0", start=0.0, end=30.0)
        assert chunk.id == "chunk_0"
        assert chunk.segment_ids == []
        assert chunk.turn_ids == []
        assert chunk.token_count_estimate == 0

    def test_chunk_creation_full(self) -> None:
        """Test Chunk creation with all fields."""
        chunk = Chunk(
            id="chunk_1",
            start=30.0,
            end=60.0,
            segment_ids=[5, 6, 7],
            turn_ids=["turn_3", "turn_4"],
            speaker_ids=["spk_0", "spk_1"],
            token_count_estimate=150,
            text="The conversation continues...",
        )
        assert len(chunk.segment_ids) == 3
        assert len(chunk.turn_ids) == 2
        assert chunk.token_count_estimate == 150

    def test_chunk_to_dict(self) -> None:
        """Test Chunk.to_dict() serialization."""
        chunk = Chunk(
            id="chunk_0",
            start=0.0,
            end=30.0,
            segment_ids=[0, 1],
            token_count_estimate=100,
            text="Hello",
        )
        d = chunk.to_dict()
        assert d["id"] == "chunk_0"
        assert d["token_count_estimate"] == 100

    def test_chunk_from_dict(self) -> None:
        """Test Chunk.from_dict() deserialization."""
        d = {
            "id": "chunk_5",
            "start": 120.0,
            "end": 150.0,
            "segment_ids": [20, 21, 22],
            "turn_ids": ["turn_10"],
            "speaker_ids": ["spk_0"],
            "token_count_estimate": 200,
            "text": "Later in the conversation",
        }
        chunk = Chunk.from_dict(d)
        assert chunk.id == "chunk_5"
        assert chunk.start == 120.0
        assert chunk.token_count_estimate == 200


# ============================================================================
# SpeakerStats and Summary Dataclass Tests
# ============================================================================


class TestProsodySummary:
    """Tests for the ProsodySummary dataclass."""

    def test_prosody_summary_defaults(self) -> None:
        """Test ProsodySummary with default None values."""
        summary = ProsodySummary()
        assert summary.pitch_median_hz is None
        assert summary.energy_median_db is None

    def test_prosody_summary_with_values(self) -> None:
        """Test ProsodySummary with actual values."""
        summary = ProsodySummary(pitch_median_hz=220.5, energy_median_db=-12.3)
        assert summary.pitch_median_hz == 220.5
        assert summary.energy_median_db == -12.3

    def test_prosody_summary_to_dict(self) -> None:
        """Test ProsodySummary.to_dict() serialization."""
        summary = ProsodySummary(pitch_median_hz=200.0)
        d = summary.to_dict()
        assert d["pitch_median_hz"] == 200.0
        assert d["energy_median_db"] is None


class TestSentimentSummary:
    """Tests for the SentimentSummary dataclass."""

    def test_sentiment_summary_defaults(self) -> None:
        """Test SentimentSummary with default zero values."""
        summary = SentimentSummary()
        assert summary.positive == 0.0
        assert summary.neutral == 0.0
        assert summary.negative == 0.0

    def test_sentiment_summary_distribution(self) -> None:
        """Test SentimentSummary with a sentiment distribution."""
        summary = SentimentSummary(positive=0.4, neutral=0.5, negative=0.1)
        total = summary.positive + summary.neutral + summary.negative
        assert total == pytest.approx(1.0)


class TestSpeakerStats:
    """Tests for the SpeakerStats dataclass."""

    def test_speaker_stats_creation(self) -> None:
        """Test SpeakerStats creation with all fields."""
        stats = SpeakerStats(
            speaker_id="spk_0",
            total_talk_time=120.5,
            num_turns=15,
            avg_turn_duration=8.0,
            interruptions_initiated=2,
            interruptions_received=3,
            question_turns=5,
            prosody_summary=ProsodySummary(pitch_median_hz=180.0),
            sentiment_summary=SentimentSummary(positive=0.6, neutral=0.3, negative=0.1),
        )
        assert stats.speaker_id == "spk_0"
        assert stats.total_talk_time == 120.5
        assert stats.num_turns == 15
        assert stats.interruptions_initiated == 2

    def test_speaker_stats_to_dict(self) -> None:
        """Test SpeakerStats.to_dict() serialization."""
        stats = SpeakerStats(
            speaker_id="spk_1",
            total_talk_time=60.0,
            num_turns=10,
            avg_turn_duration=6.0,
            interruptions_initiated=1,
            interruptions_received=2,
            question_turns=3,
            prosody_summary=ProsodySummary(pitch_median_hz=250.0, energy_median_db=-10.0),
            sentiment_summary=SentimentSummary(positive=0.5, neutral=0.4, negative=0.1),
        )
        d = stats.to_dict()
        assert d["speaker_id"] == "spk_1"
        assert d["total_talk_time"] == 60.0
        assert d["prosody_summary"]["pitch_median_hz"] == 250.0
        assert d["sentiment_summary"]["positive"] == 0.5

    def test_speaker_stats_from_dict(self) -> None:
        """Test SpeakerStats.from_dict() deserialization."""
        d = {
            "speaker_id": "spk_2",
            "total_talk_time": 90.0,
            "num_turns": 12,
            "avg_turn_duration": 7.5,
            "interruptions_initiated": 0,
            "interruptions_received": 1,
            "question_turns": 4,
            "prosody_summary": {"pitch_median_hz": 200.0, "energy_median_db": None},
            "sentiment_summary": {"positive": 0.3, "neutral": 0.6, "negative": 0.1},
        }
        stats = SpeakerStats.from_dict(d)
        assert stats.speaker_id == "spk_2"
        assert stats.num_turns == 12
        assert stats.prosody_summary.pitch_median_hz == 200.0


# ============================================================================
# BatchFileResult and BatchProcessingResult Tests
# ============================================================================


class TestBatchFileResult:
    """Tests for the BatchFileResult dataclass."""

    def test_batch_file_result_success(self) -> None:
        """Test BatchFileResult for successful transcription."""
        result = BatchFileResult(
            file_path="/path/to/audio.wav",
            status="success",
            transcript=Transcript(file_name="audio.wav", language="en"),
        )
        assert result.status == "success"
        assert result.transcript is not None
        assert result.error_type is None

    def test_batch_file_result_error(self) -> None:
        """Test BatchFileResult for failed transcription."""
        result = BatchFileResult(
            file_path="/path/to/audio.wav",
            status="error",
            error_type="FileNotFoundError",
            error_message="Audio file not found",
        )
        assert result.status == "error"
        assert result.transcript is None
        assert result.error_type == "FileNotFoundError"

    def test_batch_file_result_to_dict(self) -> None:
        """Test BatchFileResult.to_dict() excludes transcript."""
        result = BatchFileResult(
            file_path="/path/to/audio.wav",
            status="success",
            transcript=Transcript(file_name="audio.wav", language="en"),
        )
        d = result.to_dict()
        assert "file_path" in d
        assert "status" in d
        assert "transcript" not in d  # Excluded from serialization


class TestBatchProcessingResult:
    """Tests for the BatchProcessingResult dataclass."""

    def test_batch_processing_result_creation(self) -> None:
        """Test BatchProcessingResult creation."""
        result = BatchProcessingResult(
            total_files=5,
            successful=4,
            failed=1,
        )
        assert result.total_files == 5
        assert result.successful == 4
        assert result.failed == 1

    def test_batch_processing_result_get_failures(self) -> None:
        """Test BatchProcessingResult.get_failures() method."""
        results = [
            BatchFileResult(file_path="a.wav", status="success"),
            BatchFileResult(file_path="b.wav", status="error", error_type="ASRError"),
            BatchFileResult(file_path="c.wav", status="success"),
        ]
        batch = BatchProcessingResult(
            total_files=3,
            successful=2,
            failed=1,
            results=results,
        )
        failures = batch.get_failures()
        assert len(failures) == 1
        assert failures[0].file_path == "b.wav"

    def test_batch_processing_result_get_transcripts(self) -> None:
        """Test BatchProcessingResult.get_transcripts() method."""
        t1 = Transcript(file_name="a.wav", language="en")
        t2 = Transcript(file_name="c.wav", language="en")
        results = [
            BatchFileResult(file_path="a.wav", status="success", transcript=t1),
            BatchFileResult(file_path="b.wav", status="error"),
            BatchFileResult(file_path="c.wav", status="success", transcript=t2),
        ]
        batch = BatchProcessingResult(
            total_files=3,
            successful=2,
            failed=1,
            results=results,
        )
        transcripts = batch.get_transcripts()
        assert len(transcripts) == 2


# ============================================================================
# EnrichmentFileResult and EnrichmentBatchResult Tests
# ============================================================================


class TestEnrichmentFileResult:
    """Tests for the EnrichmentFileResult dataclass."""

    def test_enrichment_file_result_success(self) -> None:
        """Test EnrichmentFileResult for successful enrichment."""
        result = EnrichmentFileResult(
            transcript_path="/path/to/transcript.json",
            status="success",
            enriched_transcript=Transcript(file_name="audio.wav", language="en"),
        )
        assert result.status == "success"
        assert result.enriched_transcript is not None
        assert result.warnings == []

    def test_enrichment_file_result_partial(self) -> None:
        """Test EnrichmentFileResult for partial enrichment."""
        result = EnrichmentFileResult(
            transcript_path="/path/to/transcript.json",
            status="partial",
            enriched_transcript=Transcript(file_name="audio.wav", language="en"),
            warnings=["Emotion extraction failed"],
        )
        assert result.status == "partial"
        assert len(result.warnings) == 1

    def test_enrichment_file_result_error(self) -> None:
        """Test EnrichmentFileResult for failed enrichment."""
        result = EnrichmentFileResult(
            transcript_path="/path/to/transcript.json",
            status="error",
            error_type="AudioNotFoundError",
            error_message="Audio file not found for transcript",
        )
        assert result.status == "error"
        assert result.error_type == "AudioNotFoundError"


class TestEnrichmentBatchResult:
    """Tests for the EnrichmentBatchResult dataclass."""

    def test_enrichment_batch_result_creation(self) -> None:
        """Test EnrichmentBatchResult creation."""
        result = EnrichmentBatchResult(
            total_files=10,
            successful=8,
            partial=1,
            failed=1,
        )
        assert result.total_files == 10
        assert result.successful == 8
        assert result.partial == 1
        assert result.failed == 1

    def test_enrichment_batch_result_get_failures(self) -> None:
        """Test EnrichmentBatchResult.get_failures() method."""
        results = [
            EnrichmentFileResult(transcript_path="a.json", status="success"),
            EnrichmentFileResult(transcript_path="b.json", status="error"),
            EnrichmentFileResult(transcript_path="c.json", status="partial"),
        ]
        batch = EnrichmentBatchResult(
            total_files=3,
            successful=1,
            partial=1,
            failed=1,
            results=results,
        )
        failures = batch.get_failures()
        assert len(failures) == 1
        assert failures[0].transcript_path == "b.json"

    def test_enrichment_batch_result_get_transcripts(self) -> None:
        """Test EnrichmentBatchResult.get_transcripts() includes partial results."""
        t1 = Transcript(file_name="a.wav", language="en")
        t2 = Transcript(file_name="c.wav", language="en")
        results = [
            EnrichmentFileResult(
                transcript_path="a.json", status="success", enriched_transcript=t1
            ),
            EnrichmentFileResult(transcript_path="b.json", status="error"),
            EnrichmentFileResult(
                transcript_path="c.json", status="partial", enriched_transcript=t2
            ),
        ]
        batch = EnrichmentBatchResult(
            total_files=3,
            successful=1,
            partial=1,
            failed=1,
            results=results,
        )
        transcripts = batch.get_transcripts()
        assert len(transcripts) == 2  # Includes both success and partial
