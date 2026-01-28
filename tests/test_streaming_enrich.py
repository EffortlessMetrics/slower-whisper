"""Tests for streaming enrichment session (v2.0).

This test suite validates the StreamingEnrichmentSession and StreamingEnrichmentConfig
classes from transcription/streaming_enrich.py:

1. StreamingEnrichmentConfig dataclass - field defaults and customization
2. StreamingEnrichmentSession initialization - file validation, extractor setup
3. ingest_chunk() - partial/final segment handling, enrichment, callbacks
4. end_of_stream() - flush, enrichment, turn finalization
5. reset() - state clearing, session reuse
6. _enrich_stream_segment() - feature extraction, error handling
7. Speaker turn tracking - detection, accumulation, callbacks
8. get_stats() - statistics dictionary structure

Testing strategy:
- Mock AudioSegmentExtractor to avoid real audio file dependencies
- Mock enrichment functions to control test outcomes
- Use recording callbacks to verify invocation patterns
- Test error paths with forced failures
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from transcription.streaming import (
    StreamChunk,
    StreamConfig,
    StreamEventType,
    StreamSegment,
)
from transcription.streaming_callbacks import (
    StreamingError,
)
from transcription.streaming_enrich import (
    StreamingEnrichmentConfig,
    StreamingEnrichmentSession,
)

# =============================================================================
# Helper Functions
# =============================================================================


def _chunk(start: float, end: float, text: str, speaker: str | None = None) -> StreamChunk:
    """Create a StreamChunk for testing."""
    return {"start": start, "end": end, "text": text, "speaker_id": speaker}


def _create_mock_extractor(duration: float = 60.0, sample_rate: int = 16000) -> MagicMock:
    """Create a mock AudioSegmentExtractor."""
    mock = MagicMock()
    mock.duration_seconds = duration
    mock.sample_rate = sample_rate
    mock.total_frames = int(duration * sample_rate)
    mock.wav_path = Path("/tmp/fake_audio.wav")
    return mock


def _create_mock_audio_state(
    prosody_status: str = "success",
    emotion_status: str = "success",
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Create a mock audio_state dict for enrichment mocking."""
    return {
        "prosody": {"pitch_mean": 150.0, "energy_mean": 0.5}
        if prosody_status == "success"
        else None,
        "emotion": {"valence": 0.5, "arousal": 0.5} if emotion_status == "success" else None,
        "rendering": "[audio: neutral]",
        "extraction_status": {
            "prosody": prosody_status,
            "emotion_dimensional": emotion_status,
            "emotion_categorical": "skipped",
            "errors": errors or [],
        },
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_extractor() -> MagicMock:
    """Provide a mock AudioSegmentExtractor."""
    return _create_mock_extractor()


@pytest.fixture
def base_config() -> StreamingEnrichmentConfig:
    """Provide a base config with all enrichment disabled."""
    return StreamingEnrichmentConfig(
        enable_prosody=False,
        enable_emotion=False,
        enable_categorical_emotion=False,
    )


@pytest.fixture
def enriched_config() -> StreamingEnrichmentConfig:
    """Provide a config with prosody and emotion enabled."""
    return StreamingEnrichmentConfig(
        enable_prosody=True,
        enable_emotion=True,
        enable_categorical_emotion=False,
    )


@pytest.fixture
def recording_callbacks() -> dict[str, Any]:
    """Fixture that provides a callbacks object that records all calls."""

    class RecordingCallbacks:
        def __init__(self) -> None:
            self.finalized_segments: list[StreamSegment] = []
            self.speaker_turns: list[dict] = []
            self.errors: list[StreamingError] = []

        def on_segment_finalized(self, segment: StreamSegment) -> None:
            self.finalized_segments.append(segment)

        def on_speaker_turn(self, turn: dict) -> None:
            self.speaker_turns.append(turn)

        def on_error(self, error: StreamingError) -> None:
            self.errors.append(error)

    return {"callbacks": RecordingCallbacks(), "instance": RecordingCallbacks}


@pytest.fixture
def failing_callbacks() -> dict[str, Any]:
    """Fixture that provides a callbacks object where on_segment_finalized raises."""

    class FailingCallbacks:
        def __init__(self) -> None:
            self.errors: list[StreamingError] = []
            self.call_count = 0

        def on_segment_finalized(self, segment: StreamSegment) -> None:
            self.call_count += 1
            raise RuntimeError("Simulated callback failure")

        def on_error(self, error: StreamingError) -> None:
            self.errors.append(error)

    return {"callbacks": FailingCallbacks(), "instance": FailingCallbacks}


# =============================================================================
# 1. Tests for StreamingEnrichmentConfig
# =============================================================================


class TestStreamingEnrichmentConfig:
    """Tests for StreamingEnrichmentConfig dataclass."""

    def test_default_values(self) -> None:
        """Config has expected default values."""
        config = StreamingEnrichmentConfig()

        assert config.enable_prosody is False
        assert config.enable_emotion is False
        assert config.enable_categorical_emotion is False
        assert config.speaker_baseline is None
        assert isinstance(config.base_config, StreamConfig)

    def test_custom_base_config(self) -> None:
        """Config accepts custom base_config."""
        base = StreamConfig(max_gap_sec=2.5)
        config = StreamingEnrichmentConfig(base_config=base)

        assert config.base_config.max_gap_sec == 2.5

    def test_enable_prosody_flag(self) -> None:
        """Config accepts enable_prosody flag."""
        config = StreamingEnrichmentConfig(enable_prosody=True)

        assert config.enable_prosody is True
        assert config.enable_emotion is False  # others unchanged

    def test_enable_emotion_flag(self) -> None:
        """Config accepts enable_emotion flag."""
        config = StreamingEnrichmentConfig(enable_emotion=True)

        assert config.enable_emotion is True
        assert config.enable_prosody is False  # others unchanged

    def test_enable_categorical_emotion_flag(self) -> None:
        """Config accepts enable_categorical_emotion flag."""
        config = StreamingEnrichmentConfig(enable_categorical_emotion=True)

        assert config.enable_categorical_emotion is True
        assert config.enable_emotion is False  # others unchanged

    def test_speaker_baseline_passthrough(self) -> None:
        """Config accepts and preserves speaker_baseline."""
        baseline = {"pitch_mean": 180.0, "energy_mean": 0.6}
        config = StreamingEnrichmentConfig(speaker_baseline=baseline)

        assert config.speaker_baseline == baseline

    def test_all_options_combined(self) -> None:
        """Config accepts all options at once."""
        base = StreamConfig(max_gap_sec=1.5)
        baseline = {"pitch_mean": 200.0}
        config = StreamingEnrichmentConfig(
            base_config=base,
            enable_prosody=True,
            enable_emotion=True,
            enable_categorical_emotion=True,
            speaker_baseline=baseline,
        )

        assert config.base_config.max_gap_sec == 1.5
        assert config.enable_prosody is True
        assert config.enable_emotion is True
        assert config.enable_categorical_emotion is True
        assert config.speaker_baseline == baseline


# =============================================================================
# 2. Tests for StreamingEnrichmentSession Initialization
# =============================================================================


class TestStreamingEnrichmentSessionInit:
    """Tests for session initialization."""

    def test_init_with_valid_wav(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Session initializes correctly with valid WAV path."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            assert session.wav_path == Path("/tmp/fake.wav")
            assert session.config is base_config
            assert session._chunk_count == 0
            assert session._segment_count == 0
            assert session._enrichment_errors == 0

    def test_init_file_not_found(self) -> None:
        """Session raises FileNotFoundError for missing WAV."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            side_effect=FileNotFoundError("Audio file not found"),
        ):
            with pytest.raises(FileNotFoundError, match="Audio file not found"):
                StreamingEnrichmentSession(
                    wav_path=Path("/nonexistent/audio.wav"),
                )

    def test_init_invalid_audio(self) -> None:
        """Session raises RuntimeError for invalid audio file."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            side_effect=RuntimeError("Failed to open audio file"),
        ):
            with pytest.raises(RuntimeError, match="Failed to open audio"):
                StreamingEnrichmentSession(
                    wav_path=Path("/tmp/invalid.wav"),
                )

    def test_init_with_callbacks(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Session accepts callbacks object."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            assert session._callbacks is callbacks

    def test_init_with_custom_config(self, mock_extractor: MagicMock) -> None:
        """Session accepts custom enrichment config."""
        config = StreamingEnrichmentConfig(
            base_config=StreamConfig(max_gap_sec=2.0),
            enable_prosody=True,
        )

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=config,
            )

            assert session.config.enable_prosody is True
            assert session.config.base_config.max_gap_sec == 2.0

    def test_init_default_config(self, mock_extractor: MagicMock) -> None:
        """Session creates default config when none provided."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(wav_path=Path("/tmp/fake.wav"))

            assert session.config is not None
            assert isinstance(session.config, StreamingEnrichmentConfig)
            assert session.config.enable_prosody is False

    def test_init_string_path_converted(self, mock_extractor: MagicMock) -> None:
        """Session converts string path to Path object."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(wav_path="/tmp/string_path.wav")

            assert isinstance(session.wav_path, Path)
            assert str(session.wav_path) == "/tmp/string_path.wav"


# =============================================================================
# 3. Tests for ingest_chunk()
# =============================================================================


class TestIngestChunk:
    """Tests for ingest_chunk() method."""

    def test_ingest_partial_segment(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """ingest_chunk returns PARTIAL_SEGMENT for new chunk."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            events = session.ingest_chunk(_chunk(0.0, 1.0, "hello", "spk_0"))

            assert len(events) == 1
            assert events[0].type == StreamEventType.PARTIAL_SEGMENT
            assert events[0].segment.text == "hello"

    def test_ingest_final_segment_on_gap(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """ingest_chunk returns FINAL_SEGMENT when gap exceeds threshold."""
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            session.ingest_chunk(_chunk(0.0, 0.4, "hello", "spk_0"))
            events = session.ingest_chunk(_chunk(2.0, 2.5, "world", "spk_0"))

            assert len(events) == 2
            assert events[0].type == StreamEventType.FINAL_SEGMENT
            assert events[0].segment.text == "hello"
            assert events[1].type == StreamEventType.PARTIAL_SEGMENT
            assert events[1].segment.text == "world"

    def test_ingest_final_segment_unenriched(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """FINAL segments have no audio_state when enrichment disabled."""
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            session.ingest_chunk(_chunk(0.0, 0.4, "hello", "spk_0"))
            events = session.ingest_chunk(_chunk(2.0, 2.5, "world", "spk_0"))

            final_segment = events[0].segment
            assert final_segment.audio_state is None

    def test_ingest_final_segment_enriched(
        self,
        mock_extractor: MagicMock,
        enriched_config: StreamingEnrichmentConfig,
    ) -> None:
        """FINAL segments have audio_state when enrichment enabled."""
        enriched_config.base_config.max_gap_sec = 0.5
        mock_audio_state = _create_mock_audio_state()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                return_value=mock_audio_state,
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=enriched_config,
                )

                session.ingest_chunk(_chunk(0.0, 0.4, "hello", "spk_0"))
                events = session.ingest_chunk(_chunk(2.0, 2.5, "world", "spk_0"))

                final_segment = events[0].segment
                assert final_segment.audio_state is not None
                assert final_segment.audio_state["prosody"] is not None

    def test_callback_on_segment_finalized(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """on_segment_finalized callback is invoked for FINAL segments."""
        callbacks = recording_callbacks["callbacks"]
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 0.4, "hello", "spk_0"))
            session.ingest_chunk(_chunk(2.0, 2.5, "world", "spk_0"))

            assert len(callbacks.finalized_segments) == 1
            assert callbacks.finalized_segments[0].text == "hello"

    def test_callback_exception_does_not_crash(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        failing_callbacks: dict[str, Any],
    ) -> None:
        """Callback exceptions are caught and don't crash the pipeline."""
        callbacks = failing_callbacks["callbacks"]
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Should not raise even though callback will fail
            session.ingest_chunk(_chunk(0.0, 0.4, "hello", "spk_0"))
            events = session.ingest_chunk(_chunk(2.0, 2.5, "world", "spk_0"))

            # Pipeline should continue
            assert len(events) == 2
            assert callbacks.call_count == 1

    def test_chunk_count_increments(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Chunk count is incremented on each ingest_chunk call."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            assert session._chunk_count == 0

            session.ingest_chunk(_chunk(0.0, 0.5, "one", "spk_0"))
            assert session._chunk_count == 1

            session.ingest_chunk(_chunk(0.6, 1.0, "two", "spk_0"))
            assert session._chunk_count == 2

    def test_segment_count_increments_on_final(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Segment count is incremented for FINAL segments."""
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            assert session._segment_count == 0

            session.ingest_chunk(_chunk(0.0, 0.4, "hello", "spk_0"))
            assert session._segment_count == 0  # partial

            session.ingest_chunk(_chunk(2.0, 2.5, "world", "spk_0"))
            assert session._segment_count == 1  # finalized first

    def test_speaker_change_finalizes_segment(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Speaker change triggers segment finalization."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            session.ingest_chunk(_chunk(0.0, 0.7, "hello", "spk_a"))
            events = session.ingest_chunk(_chunk(0.8, 1.3, "hi there", "spk_b"))

            assert len(events) == 2
            assert events[0].type == StreamEventType.FINAL_SEGMENT
            assert events[0].segment.speaker_id == "spk_a"
            assert events[1].type == StreamEventType.PARTIAL_SEGMENT
            assert events[1].segment.speaker_id == "spk_b"


# =============================================================================
# 4. Tests for end_of_stream()
# =============================================================================


class TestEndOfStream:
    """Tests for end_of_stream() method."""

    def test_end_of_stream_flushes_partial(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """end_of_stream flushes any partial segment as final."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello world", "spk_0"))
            events = session.end_of_stream()

            assert len(events) == 1
            assert events[0].type == StreamEventType.FINAL_SEGMENT
            assert events[0].segment.text == "hello world"

    def test_end_of_stream_enriches_final(
        self,
        mock_extractor: MagicMock,
        enriched_config: StreamingEnrichmentConfig,
    ) -> None:
        """end_of_stream enriches the final segment."""
        mock_audio_state = _create_mock_audio_state()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                return_value=mock_audio_state,
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=enriched_config,
                )

                session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))
                events = session.end_of_stream()

                assert events[0].segment.audio_state is not None

    def test_end_of_stream_finalizes_turn(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """end_of_stream finalizes any open speaker turn."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello", "spk_0"))
            session.end_of_stream()

            # Should have one turn finalized
            assert len(callbacks.speaker_turns) == 1
            assert callbacks.speaker_turns[0]["speaker_id"] == "spk_0"

    def test_end_of_stream_logs_stats(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """end_of_stream logs session statistics."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))

            with caplog.at_level(logging.INFO):
                session.end_of_stream()

            assert "Stream ended" in caplog.text or "chunks ingested" in caplog.text

    def test_end_of_stream_empty_session(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """end_of_stream returns empty list for empty session."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            events = session.end_of_stream()
            assert events == []

    def test_end_of_stream_invokes_callback(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """end_of_stream invokes on_segment_finalized callback."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "test message", "spk_0"))
            session.end_of_stream()

            assert len(callbacks.finalized_segments) == 1
            assert callbacks.finalized_segments[0].text == "test message"


# =============================================================================
# 5. Tests for reset()
# =============================================================================


class TestReset:
    """Tests for reset() method."""

    def test_reset_clears_counters(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """reset clears all session counters."""
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            # Generate some state
            session.ingest_chunk(_chunk(0.0, 0.4, "hello", "spk_0"))
            session.ingest_chunk(_chunk(2.0, 2.5, "world", "spk_0"))
            session.end_of_stream()

            assert session._chunk_count > 0
            assert session._segment_count > 0

            session.reset()

            assert session._chunk_count == 0
            assert session._segment_count == 0
            assert session._enrichment_errors == 0
            assert session._turn_counter == 0

    def test_reset_preserves_extractor(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """reset preserves the AudioSegmentExtractor."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            extractor_before = session._extractor

            session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))
            session.reset()

            assert session._extractor is extractor_before

    def test_reset_allows_reuse(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """reset allows session to be reused."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            # First stream
            session.ingest_chunk(_chunk(0.0, 1.0, "first stream", "spk_0"))
            events1 = session.end_of_stream()
            assert events1[0].segment.text == "first stream"

            session.reset()

            # Second stream
            session.ingest_chunk(_chunk(0.0, 1.5, "second stream", "spk_1"))
            events2 = session.end_of_stream()
            assert events2[0].segment.text == "second stream"

    def test_reset_clears_turn_state(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """reset clears speaker turn tracking state."""
        base_config.base_config.max_gap_sec = 0.5  # Force gap-based finalization

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            # Two chunks with gap - first gets finalized and tracked
            session.ingest_chunk(_chunk(0.0, 0.4, "hello", "spk_0"))
            session.ingest_chunk(_chunk(1.5, 2.0, "world", "spk_0"))

            # Turn state should be populated with finalized segment
            assert len(session._current_turn_segments) > 0
            assert session._current_turn_speaker == "spk_0"

            session.reset()

            assert session._current_turn_segments == []
            assert session._current_turn_speaker is None


# =============================================================================
# 6. Tests for _enrich_stream_segment()
# =============================================================================


class TestEnrichStreamSegment:
    """Tests for _enrich_stream_segment() method."""

    def test_skips_when_all_disabled(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Enrichment is skipped when all features disabled."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            segment = StreamSegment(start=0.0, end=1.0, text="test", speaker_id="spk_0")
            result = session._enrich_stream_segment(segment)

            # Should return unchanged segment
            assert result.audio_state is None
            assert result.text == "test"

    def test_enriches_with_prosody(
        self,
        mock_extractor: MagicMock,
    ) -> None:
        """Enrichment extracts prosody when enabled."""
        config = StreamingEnrichmentConfig(enable_prosody=True)
        mock_audio_state = _create_mock_audio_state()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                return_value=mock_audio_state,
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=config,
                )

                segment = StreamSegment(start=0.0, end=1.0, text="test", speaker_id="spk_0")
                result = session._enrich_stream_segment(segment)

                assert result.audio_state is not None
                assert result.audio_state["prosody"] is not None

    def test_enriches_with_emotion(
        self,
        mock_extractor: MagicMock,
    ) -> None:
        """Enrichment extracts emotion when enabled."""
        config = StreamingEnrichmentConfig(enable_emotion=True)
        mock_audio_state = _create_mock_audio_state()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                return_value=mock_audio_state,
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=config,
                )

                segment = StreamSegment(start=0.0, end=1.0, text="test", speaker_id="spk_0")
                result = session._enrich_stream_segment(segment)

                assert result.audio_state is not None
                assert result.audio_state["emotion"] is not None

    def test_handles_enrichment_failure(
        self,
        mock_extractor: MagicMock,
    ) -> None:
        """Enrichment failure results in error audio_state."""
        config = StreamingEnrichmentConfig(enable_prosody=True)

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                side_effect=RuntimeError("Enrichment failed"),
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=config,
                )

                segment = StreamSegment(start=0.0, end=1.0, text="test", speaker_id="spk_0")
                result = session._enrich_stream_segment(segment)

                # Should have error audio_state
                assert result.audio_state is not None
                assert result.audio_state["prosody"] is None
                assert "failed" in result.audio_state["extraction_status"]["prosody"]
                assert session._enrichment_errors == 1

    def test_handles_partial_errors(
        self,
        mock_extractor: MagicMock,
    ) -> None:
        """Enrichment with partial errors increments error count."""
        config = StreamingEnrichmentConfig(enable_prosody=True, enable_emotion=True)
        mock_audio_state = _create_mock_audio_state(
            prosody_status="success",
            emotion_status="failed",
            errors=["Emotion extraction failed"],
        )

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                return_value=mock_audio_state,
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=config,
                )

                segment = StreamSegment(start=0.0, end=1.0, text="test", speaker_id="spk_0")
                result = session._enrich_stream_segment(segment)

                assert result.audio_state is not None
                assert len(result.audio_state["extraction_status"]["errors"]) > 0
                assert session._enrichment_errors == 1

    def test_calls_on_error_callback(
        self,
        mock_extractor: MagicMock,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Enrichment failure invokes on_error callback."""
        callbacks = recording_callbacks["callbacks"]
        config = StreamingEnrichmentConfig(enable_prosody=True)

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                side_effect=RuntimeError("Enrichment failed"),
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=config,
                    callbacks=callbacks,
                )

                segment = StreamSegment(start=0.0, end=1.0, text="test", speaker_id="spk_0")
                session._enrich_stream_segment(segment)

                assert len(callbacks.errors) == 1
                assert "Failed to enrich segment" in callbacks.errors[0].context

    def test_preserves_segment_fields(
        self,
        mock_extractor: MagicMock,
    ) -> None:
        """Enrichment preserves original segment fields."""
        config = StreamingEnrichmentConfig(enable_prosody=True)
        mock_audio_state = _create_mock_audio_state()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                return_value=mock_audio_state,
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=config,
                )

                segment = StreamSegment(
                    start=5.5,
                    end=10.2,
                    text="test message",
                    speaker_id="spk_1",
                )
                result = session._enrich_stream_segment(segment)

                assert result.start == 5.5
                assert result.end == 10.2
                assert result.text == "test message"
                assert result.speaker_id == "spk_1"


# =============================================================================
# 7. Tests for Speaker Turn Tracking
# =============================================================================


class TestSpeakerTurnTracking:
    """Tests for speaker turn detection and callbacks."""

    def test_detects_speaker_change(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Speaker change is detected and triggers turn callback."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Finalize first segment by speaker change
            session.ingest_chunk(_chunk(0.0, 1.0, "hello", "spk_a"))
            session.ingest_chunk(_chunk(1.1, 2.0, "hi there", "spk_b"))

            # spk_a segment is finalized, but turn isn't finalized yet
            # Turn is finalized when the NEXT speaker's segment becomes final
            # and it has a different speaker_id
            assert len(callbacks.finalized_segments) == 1
            assert callbacks.finalized_segments[0].speaker_id == "spk_a"
            assert len(callbacks.speaker_turns) == 0  # Not finalized yet

            # Finalize spk_b by another speaker change
            session.ingest_chunk(_chunk(2.1, 3.0, "bye", "spk_a"))

            # Now spk_a turn (turn_0) should be finalized when spk_b segment was tracked
            # (speaker changed from spk_a to spk_b)
            assert len(callbacks.speaker_turns) == 1
            assert callbacks.speaker_turns[0]["speaker_id"] == "spk_a"

            # spk_b turn is still in progress, will be finalized at end_of_stream
            session.end_of_stream()

            # Now both turns should be finalized
            assert len(callbacks.speaker_turns) == 3
            assert callbacks.speaker_turns[0]["speaker_id"] == "spk_a"
            assert callbacks.speaker_turns[1]["speaker_id"] == "spk_b"
            assert callbacks.speaker_turns[2]["speaker_id"] == "spk_a"

    def test_accumulates_same_speaker(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Same speaker segments are accumulated in one turn."""
        callbacks = recording_callbacks["callbacks"]
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Two segments from same speaker with gap
            session.ingest_chunk(_chunk(0.0, 0.4, "first", "spk_0"))
            session.ingest_chunk(_chunk(1.5, 2.0, "second", "spk_0"))
            session.ingest_chunk(_chunk(3.5, 4.0, "third", "spk_0"))
            session.end_of_stream()

            # All should be in one turn
            assert len(callbacks.speaker_turns) == 1
            assert callbacks.speaker_turns[0]["speaker_id"] == "spk_0"
            assert len(callbacks.speaker_turns[0]["segment_ids"]) == 3

    def test_finalizes_turn_on_speaker_change(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Turn is finalized when speaker changes."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello from A", "spk_a"))
            session.ingest_chunk(_chunk(1.1, 2.0, "hello from B", "spk_b"))
            session.end_of_stream()

            assert len(callbacks.speaker_turns) == 2
            assert callbacks.speaker_turns[0]["speaker_id"] == "spk_a"
            assert callbacks.speaker_turns[1]["speaker_id"] == "spk_b"

    def test_finalizes_turn_on_end_of_stream(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Turn is finalized at end_of_stream."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello", "spk_0"))

            # No turn finalized yet
            assert len(callbacks.speaker_turns) == 0

            session.end_of_stream()

            # Turn should now be finalized
            assert len(callbacks.speaker_turns) == 1

    def test_build_turn_dict_structure(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """_build_turn_dict creates correct structure."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            # Set up some state
            session._segment_count = 3
            segments = [
                StreamSegment(start=0.0, end=1.0, text="hello", speaker_id="spk_0"),
                StreamSegment(start=1.1, end=2.0, text="world", speaker_id="spk_0"),
                StreamSegment(start=2.1, end=3.0, text="test", speaker_id="spk_0"),
            ]

            turn = session._build_turn_dict(
                turn_id=0,
                segments=segments,
                speaker_id="spk_0",
            )

            assert turn["id"] == "turn_0"
            assert turn["speaker_id"] == "spk_0"
            assert turn["start"] == 0.0
            assert turn["end"] == 3.0
            assert turn["text"] == "hello world test"
            assert len(turn["segment_ids"]) == 3

    def test_calls_on_speaker_turn_callback(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """on_speaker_turn callback is invoked with turn dict."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello", "spk_0"))
            session.end_of_stream()

            assert len(callbacks.speaker_turns) == 1
            turn = callbacks.speaker_turns[0]
            assert "id" in turn
            assert "speaker_id" in turn
            assert "start" in turn
            assert "end" in turn
            assert "text" in turn
            assert "segment_ids" in turn

    def test_unknown_speaker_id(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """None speaker_id is converted to 'unknown' in turn."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello", None))
            session.end_of_stream()

            assert len(callbacks.speaker_turns) == 1
            assert callbacks.speaker_turns[0]["speaker_id"] == "unknown"

    def test_build_turn_dict_empty_segments_raises(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """_build_turn_dict raises ValueError for empty segments."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            with pytest.raises(ValueError, match="Cannot build turn from empty"):
                session._build_turn_dict(turn_id=0, segments=[], speaker_id="spk_0")


# =============================================================================
# 8. Tests for get_stats()
# =============================================================================


class TestGetStats:
    """Tests for get_stats() method."""

    def test_stats_structure(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """get_stats returns expected keys."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            stats = session.get_stats()

            assert "chunk_count" in stats
            assert "segment_count" in stats
            assert "enrichment_errors" in stats
            assert "audio_duration_sec" in stats
            assert "audio_sample_rate" in stats

    def test_stats_after_processing(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """get_stats reflects processing state."""
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            session.ingest_chunk(_chunk(0.0, 0.4, "one", "spk_0"))
            session.ingest_chunk(_chunk(1.5, 2.0, "two", "spk_0"))
            session.ingest_chunk(_chunk(3.5, 4.0, "three", "spk_0"))
            session.end_of_stream()

            stats = session.get_stats()

            assert stats["chunk_count"] == 3
            assert stats["segment_count"] == 3
            assert stats["enrichment_errors"] == 0

    def test_stats_include_audio_info(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """get_stats includes audio file information."""
        mock_extractor.duration_seconds = 120.5
        mock_extractor.sample_rate = 16000

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            stats = session.get_stats()

            assert stats["audio_duration_sec"] == 120.5
            assert stats["audio_sample_rate"] == 16000

    def test_stats_initial_values(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """get_stats returns zeros for fresh session."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            stats = session.get_stats()

            assert stats["chunk_count"] == 0
            assert stats["segment_count"] == 0
            assert stats["enrichment_errors"] == 0


# =============================================================================
# 9. Additional Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_callbacks_can_be_none(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Session works correctly when callbacks is None."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=None,
            )

            # These should not raise
            events = session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))
            assert len(events) == 1

            final_events = session.end_of_stream()
            assert len(final_events) == 1

    def test_partial_callbacks_missing_methods(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Session works when callbacks lacks some methods."""

        class PartialCallbacks:
            def on_error(self, error: StreamingError) -> None:
                pass

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=PartialCallbacks(),
            )

            # Should not raise
            session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))
            events = session.end_of_stream()
            assert len(events) == 1

    def test_multiple_segments_multiple_callbacks(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Multiple finalized segments invoke multiple callbacks."""
        callbacks = recording_callbacks["callbacks"]
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 0.4, "one", "spk_0"))
            session.ingest_chunk(_chunk(1.5, 2.0, "two", "spk_0"))
            session.ingest_chunk(_chunk(3.5, 4.0, "three", "spk_0"))
            session.end_of_stream()

            assert len(callbacks.finalized_segments) == 3
            texts = [seg.text for seg in callbacks.finalized_segments]
            assert texts == ["one", "two", "three"]

    def test_single_segment_stream(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Session handles single segment stream correctly."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            events = session.ingest_chunk(_chunk(0.0, 1.0, "only one", "spk_0"))
            assert events[0].type == StreamEventType.PARTIAL_SEGMENT

            final_events = session.end_of_stream()
            assert len(final_events) == 1
            assert final_events[0].segment.text == "only one"

    def test_enrichment_errors_tracked(
        self,
        mock_extractor: MagicMock,
    ) -> None:
        """Enrichment errors are tracked in error count."""
        config = StreamingEnrichmentConfig(enable_prosody=True)
        config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                side_effect=RuntimeError("Enrichment failed"),
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=config,
                )

                session.ingest_chunk(_chunk(0.0, 0.4, "one", "spk_0"))
                session.ingest_chunk(_chunk(1.5, 2.0, "two", "spk_0"))
                session.end_of_stream()

                assert session._enrichment_errors == 2

    def test_on_error_invoked_for_partial_enrichment_errors(
        self,
        mock_extractor: MagicMock,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """on_error callback is invoked for partial enrichment errors."""
        callbacks = recording_callbacks["callbacks"]
        config = StreamingEnrichmentConfig(enable_prosody=True)
        mock_audio_state = _create_mock_audio_state(
            prosody_status="failed",
            errors=["Prosody extraction failed"],
        )

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            with patch(
                "transcription.streaming_enrich._enrich_segment_with_extractor",
                return_value=mock_audio_state,
            ):
                session = StreamingEnrichmentSession(
                    wav_path=Path("/tmp/fake.wav"),
                    config=config,
                    callbacks=callbacks,
                )

                session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))
                session.end_of_stream()

                # on_error should have been called for partial errors
                assert len(callbacks.errors) == 1
                assert callbacks.errors[0].recoverable is True

    def test_segment_with_no_speaker(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Session handles segments without speaker_id."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
            )

            events = session.ingest_chunk(_chunk(0.0, 1.0, "no speaker", None))
            assert events[0].segment.speaker_id is None

            final_events = session.end_of_stream()
            assert final_events[0].segment.speaker_id is None

    def test_multiple_speaker_transitions(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Multiple speaker transitions create multiple turns."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "A says", "spk_a"))
            session.ingest_chunk(_chunk(1.1, 2.0, "B says", "spk_b"))
            session.ingest_chunk(_chunk(2.1, 3.0, "A again", "spk_a"))
            session.end_of_stream()

            assert len(callbacks.speaker_turns) == 3
            speakers = [t["speaker_id"] for t in callbacks.speaker_turns]
            assert speakers == ["spk_a", "spk_b", "spk_a"]
