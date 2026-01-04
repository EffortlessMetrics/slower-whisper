"""
Tests for speaker_stats.py module.

Test coverage:
- compute_speaker_stats() with single speaker
- compute_speaker_stats() with multi-speaker scenarios
- _collect_segment_durations_by_speaker() edge cases
- _collect_prosody_by_speaker() with/without audio_state
- _collect_sentiment_by_speaker() with various emotion data
- Missing/incomplete audio_state handling
- Empty transcripts
- Interruption counting from turn metadata
- Question turn counting
- Prosody aggregation (median pitch/energy)
- Sentiment distribution normalization
- Edge cases: null speakers, zero-duration segments
"""

from __future__ import annotations

import pytest

from transcription.models import (
    Segment,
    Transcript,
    Turn,
)
from transcription.speaker_stats import (
    _collect_prosody_by_speaker,
    _collect_segment_durations_by_speaker,
    _collect_sentiment_by_speaker,
    compute_speaker_stats,
)

# ============================================================================
# Test fixtures and helpers
# ============================================================================


def make_segment(
    seg_id: int,
    start: float,
    end: float,
    text: str,
    speaker_id: str | None = None,
    audio_state: dict | None = None,
) -> Segment:
    """Helper to create Segment for testing."""
    seg = Segment(id=seg_id, start=start, end=end, text=text)
    if speaker_id is not None:
        seg.speaker = {"id": speaker_id, "confidence": 0.9}
    if audio_state is not None:
        seg.audio_state = audio_state
    return seg


def make_transcript(
    segments: list[Segment],
    turns: list[Turn | dict] | None = None,
    speakers: list[dict] | None = None,
) -> Transcript:
    """Helper to create Transcript for testing."""
    if speakers is None:
        # Auto-build speakers list from segments
        speaker_ids = {seg.speaker["id"] for seg in segments if seg.speaker is not None}
        speakers = [
            {"id": sid, "label": None, "total_speech_time": 0.0, "num_segments": 0}
            for sid in sorted(speaker_ids)
        ]
    return Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        speakers=speakers,
        turns=turns or [],
    )


def make_prosody_audio_state(pitch_hz: float | None = None, energy_db: float | None = None) -> dict:
    """Helper to create audio_state dict with prosody data."""
    state: dict = {"prosody": {}}
    if pitch_hz is not None:
        state["prosody"]["pitch"] = {"level": "medium", "mean_hz": pitch_hz}
    if energy_db is not None:
        state["prosody"]["energy"] = {"level": "medium", "db_rms": energy_db}
    return state


def make_emotion_audio_state(valence_level: str, valence_score: float) -> dict:
    """Helper to create audio_state dict with emotion data."""
    return {
        "emotion": {
            "valence": {"level": valence_level.lower(), "score": valence_score},
            "arousal": {"level": "medium", "score": 0.5},
        }
    }


# ============================================================================
# Tests for _collect_segment_durations_by_speaker
# ============================================================================


class TestCollectSegmentDurations:
    """Tests for _collect_segment_durations_by_speaker() helper."""

    def test_single_speaker_single_segment(self):
        """Single segment from one speaker."""
        segments = [make_segment(0, 0.0, 2.5, "Hello", "spk_0")]
        transcript = make_transcript(segments)

        durations = _collect_segment_durations_by_speaker(transcript)

        assert len(durations) == 1
        assert "spk_0" in durations
        assert durations["spk_0"] == [2.5]

    def test_single_speaker_multiple_segments(self):
        """Multiple segments from one speaker."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", "spk_0"),
            make_segment(1, 2.1, 5.0, "world", "spk_0"),
            make_segment(2, 5.5, 8.0, "today", "spk_0"),
        ]
        transcript = make_transcript(segments)

        durations = _collect_segment_durations_by_speaker(transcript)

        assert len(durations) == 1
        assert "spk_0" in durations
        assert durations["spk_0"] == [2.0, 2.9, 2.5]

    def test_multi_speaker_segments(self):
        """Segments from multiple speakers."""
        segments = [
            make_segment(0, 0.0, 2.0, "A1", "spk_0"),
            make_segment(1, 2.5, 4.0, "B1", "spk_1"),
            make_segment(2, 4.5, 6.0, "A2", "spk_0"),
        ]
        transcript = make_transcript(segments)

        durations = _collect_segment_durations_by_speaker(transcript)

        assert len(durations) == 2
        assert "spk_0" in durations
        assert "spk_1" in durations
        assert durations["spk_0"] == [2.0, 1.5]
        assert durations["spk_1"] == [1.5]

    def test_null_speaker_assigned_to_spk_0(self):
        """Segments without speaker are assigned to spk_0."""
        segments = [
            make_segment(0, 0.0, 2.0, "No speaker"),
            make_segment(1, 2.5, 4.0, "Also no speaker"),
        ]
        transcript = make_transcript(segments, speakers=[])

        durations = _collect_segment_durations_by_speaker(transcript)

        assert len(durations) == 1
        assert "spk_0" in durations
        assert durations["spk_0"] == [2.0, 1.5]

    def test_zero_duration_segment(self):
        """Segment with end <= start gets 0.0 duration."""
        segments = [
            make_segment(0, 5.0, 5.0, "Instant", "spk_0"),  # zero duration
            make_segment(1, 10.0, 9.0, "Negative", "spk_0"),  # negative duration
        ]
        transcript = make_transcript(segments)

        durations = _collect_segment_durations_by_speaker(transcript)

        assert "spk_0" in durations
        # max(end - start, 0.0) ensures non-negative
        assert durations["spk_0"] == [0.0, 0.0]

    def test_empty_transcript(self):
        """Empty transcript returns empty dict."""
        transcript = make_transcript([], speakers=[])

        durations = _collect_segment_durations_by_speaker(transcript)

        assert durations == {}


# ============================================================================
# Tests for _collect_prosody_by_speaker
# ============================================================================


class TestCollectProsody:
    """Tests for _collect_prosody_by_speaker() helper."""

    def test_single_speaker_with_prosody(self):
        """Single speaker with prosody data."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", "spk_0", make_prosody_audio_state(220.0, -10.5)),
            make_segment(1, 2.5, 4.0, "world", "spk_0", make_prosody_audio_state(245.3, -8.2)),
        ]
        transcript = make_transcript(segments)

        prosody = _collect_prosody_by_speaker(transcript)

        assert len(prosody) == 1
        assert "spk_0" in prosody
        assert prosody["spk_0"]["pitch"] == [220.0, 245.3]
        assert prosody["spk_0"]["energy"] == [-10.5, -8.2]

    def test_multi_speaker_with_prosody(self):
        """Multiple speakers with prosody data."""
        segments = [
            make_segment(0, 0.0, 2.0, "A1", "spk_0", make_prosody_audio_state(200.0, -12.0)),
            make_segment(1, 2.5, 4.0, "B1", "spk_1", make_prosody_audio_state(180.0, -15.0)),
            make_segment(2, 4.5, 6.0, "A2", "spk_0", make_prosody_audio_state(210.0, -11.0)),
        ]
        transcript = make_transcript(segments)

        prosody = _collect_prosody_by_speaker(transcript)

        assert len(prosody) == 2
        assert prosody["spk_0"]["pitch"] == [200.0, 210.0]
        assert prosody["spk_0"]["energy"] == [-12.0, -11.0]
        assert prosody["spk_1"]["pitch"] == [180.0]
        assert prosody["spk_1"]["energy"] == [-15.0]

    def test_missing_audio_state(self):
        """Segments without audio_state are skipped."""
        segments = [
            make_segment(0, 0.0, 2.0, "No audio state", "spk_0"),
            make_segment(1, 2.5, 4.0, "Has audio state", "spk_0", make_prosody_audio_state(220.0)),
        ]
        transcript = make_transcript(segments)

        prosody = _collect_prosody_by_speaker(transcript)

        assert "spk_0" in prosody
        assert prosody["spk_0"]["pitch"] == [220.0]

    def test_null_audio_state(self):
        """Segments with audio_state=None are handled."""
        segments = [
            make_segment(0, 0.0, 2.0, "Null audio state", "spk_0", None),
        ]
        transcript = make_transcript(segments)

        prosody = _collect_prosody_by_speaker(transcript)

        # Should return empty defaultdict structure
        assert len(prosody) == 0

    def test_malformed_prosody_data(self):
        """Malformed prosody data is skipped gracefully."""
        segments = [
            # Missing pitch.mean_hz
            make_segment(
                0,
                0.0,
                2.0,
                "Missing pitch",
                "spk_0",
                {"prosody": {"energy": {"db_rms": -10.0}}},
            ),
            # prosody is not a dict
            make_segment(1, 2.5, 4.0, "Bad prosody", "spk_0", {"prosody": "invalid"}),
            # pitch is not a dict
            make_segment(2, 5.0, 7.0, "Bad pitch", "spk_0", {"prosody": {"pitch": "invalid"}}),
        ]
        transcript = make_transcript(segments)

        prosody = _collect_prosody_by_speaker(transcript)

        # Should only collect the valid energy value
        assert "spk_0" in prosody
        assert prosody["spk_0"]["energy"] == [-10.0]
        assert prosody["spk_0"].get("pitch", []) == []

    def test_partial_prosody_data(self):
        """Segments with only pitch or only energy are handled."""
        segments = [
            make_segment(0, 0.0, 2.0, "Pitch only", "spk_0", make_prosody_audio_state(220.0)),
            make_segment(
                1, 2.5, 4.0, "Energy only", "spk_0", make_prosody_audio_state(None, -10.0)
            ),
        ]
        transcript = make_transcript(segments)

        prosody = _collect_prosody_by_speaker(transcript)

        assert prosody["spk_0"]["pitch"] == [220.0]
        assert prosody["spk_0"]["energy"] == [-10.0]

    def test_empty_transcript(self):
        """Empty transcript returns empty dict."""
        transcript = make_transcript([], speakers=[])

        prosody = _collect_prosody_by_speaker(transcript)

        assert prosody == {}


# ============================================================================
# Tests for _collect_sentiment_by_speaker
# ============================================================================


class TestCollectSentiment:
    """Tests for _collect_sentiment_by_speaker() helper."""

    def test_single_speaker_with_sentiment(self):
        """Single speaker with sentiment data."""
        segments = [
            make_segment(0, 0.0, 2.0, "Good!", "spk_0", make_emotion_audio_state("positive", 0.8)),
            make_segment(1, 2.5, 4.0, "Great!", "spk_0", make_emotion_audio_state("positive", 0.9)),
            make_segment(2, 4.5, 6.0, "Okay.", "spk_0", make_emotion_audio_state("neutral", 0.5)),
        ]
        transcript = make_transcript(segments)

        sentiment = _collect_sentiment_by_speaker(transcript)

        assert len(sentiment) == 1
        assert "spk_0" in sentiment
        assert sentiment["spk_0"]["positive"] == 2
        assert sentiment["spk_0"]["neutral"] == 1
        assert sentiment["spk_0"]["negative"] == 0

    def test_multi_speaker_with_sentiment(self):
        """Multiple speakers with different sentiment patterns."""
        segments = [
            make_segment(0, 0.0, 2.0, "Happy", "spk_0", make_emotion_audio_state("positive", 0.8)),
            make_segment(1, 2.5, 4.0, "Angry", "spk_1", make_emotion_audio_state("negative", 0.7)),
            make_segment(2, 4.5, 6.0, "Calm", "spk_0", make_emotion_audio_state("neutral", 0.5)),
        ]
        transcript = make_transcript(segments)

        sentiment = _collect_sentiment_by_speaker(transcript)

        assert len(sentiment) == 2
        assert sentiment["spk_0"]["positive"] == 1
        assert sentiment["spk_0"]["neutral"] == 1
        assert sentiment["spk_1"]["negative"] == 1

    def test_missing_emotion_data(self):
        """Segments without emotion data are skipped."""
        segments = [
            make_segment(0, 0.0, 2.0, "No emotion", "spk_0"),
            make_segment(
                1, 2.5, 4.0, "Has emotion", "spk_0", make_emotion_audio_state("positive", 0.8)
            ),
        ]
        transcript = make_transcript(segments)

        sentiment = _collect_sentiment_by_speaker(transcript)

        assert "spk_0" in sentiment
        assert sentiment["spk_0"]["positive"] == 1

    def test_malformed_emotion_data(self):
        """Malformed emotion data is skipped gracefully."""
        segments = [
            # emotion is not a dict
            make_segment(0, 0.0, 2.0, "Bad emotion", "spk_0", {"emotion": "invalid"}),
            # valence is not a dict
            make_segment(1, 2.5, 4.0, "Bad valence", "spk_0", {"emotion": {"valence": "invalid"}}),
            # level is invalid
            make_segment(
                2,
                5.0,
                7.0,
                "Invalid level",
                "spk_0",
                {"emotion": {"valence": {"level": "unknown", "score": 0.5}}},
            ),
        ]
        transcript = make_transcript(segments)

        sentiment = _collect_sentiment_by_speaker(transcript)

        # All segments should be skipped, resulting in empty counter for spk_0
        # defaultdict will have spk_0 key with empty Counter if accessed during processing
        assert len(sentiment) == 0 or sentiment.get("spk_0", {}) == {}

    def test_case_insensitive_levels(self):
        """Sentiment levels are case-insensitive."""
        segments = [
            make_segment(
                0,
                0.0,
                2.0,
                "Upper",
                "spk_0",
                {"emotion": {"valence": {"level": "POSITIVE", "score": 0.8}}},
            ),
            make_segment(
                1,
                2.5,
                4.0,
                "Mixed",
                "spk_0",
                {"emotion": {"valence": {"level": "Neutral", "score": 0.5}}},
            ),
        ]
        transcript = make_transcript(segments)

        sentiment = _collect_sentiment_by_speaker(transcript)

        assert sentiment["spk_0"]["positive"] == 1
        assert sentiment["spk_0"]["neutral"] == 1

    def test_empty_transcript(self):
        """Empty transcript returns empty dict."""
        transcript = make_transcript([], speakers=[])

        sentiment = _collect_sentiment_by_speaker(transcript)

        assert sentiment == {}


# ============================================================================
# Tests for compute_speaker_stats (main function)
# ============================================================================


class TestComputeSpeakerStats:
    """Tests for compute_speaker_stats() main function."""

    def test_single_speaker_basic(self):
        """Single speaker, basic stats without turns."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", "spk_0"),
            make_segment(1, 2.5, 5.0, "world", "spk_0"),
        ]
        transcript = make_transcript(segments)

        stats = compute_speaker_stats(transcript)

        assert len(stats) == 1
        assert stats[0]["speaker_id"] == "spk_0"
        assert stats[0]["total_talk_time"] == 4.5  # 2.0 + 2.5
        assert stats[0]["num_turns"] == 0  # No turns provided
        assert stats[0]["avg_turn_duration"] == 0.0
        assert stats[0]["interruptions_initiated"] == 0
        assert stats[0]["interruptions_received"] == 0
        assert stats[0]["question_turns"] == 0

    def test_single_speaker_with_prosody(self):
        """Single speaker with prosody aggregation."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", "spk_0", make_prosody_audio_state(200.0, -12.0)),
            make_segment(1, 2.5, 5.0, "world", "spk_0", make_prosody_audio_state(220.0, -10.0)),
            make_segment(2, 5.5, 8.0, "today", "spk_0", make_prosody_audio_state(210.0, -11.0)),
        ]
        transcript = make_transcript(segments)

        stats = compute_speaker_stats(transcript)

        assert len(stats) == 1
        prosody = stats[0]["prosody_summary"]
        # Median of [200.0, 210.0, 220.0] = 210.0
        assert prosody["pitch_median_hz"] == 210.0
        # Median of [-12.0, -11.0, -10.0] = -11.0
        assert prosody["energy_median_db"] == -11.0

    def test_single_speaker_with_sentiment(self):
        """Single speaker with sentiment distribution."""
        segments = [
            make_segment(0, 0.0, 2.0, "Good!", "spk_0", make_emotion_audio_state("positive", 0.8)),
            make_segment(1, 2.5, 4.0, "Great!", "spk_0", make_emotion_audio_state("positive", 0.9)),
            make_segment(2, 4.5, 6.0, "Okay.", "spk_0", make_emotion_audio_state("neutral", 0.5)),
            make_segment(3, 6.5, 8.0, "Bad.", "spk_0", make_emotion_audio_state("negative", 0.3)),
        ]
        transcript = make_transcript(segments)

        stats = compute_speaker_stats(transcript)

        assert len(stats) == 1
        sentiment = stats[0]["sentiment_summary"]
        # 2 positive, 1 neutral, 1 negative out of 4 total
        assert sentiment["positive"] == 0.5  # 2/4
        assert sentiment["neutral"] == 0.25  # 1/4
        assert sentiment["negative"] == 0.25  # 1/4

    def test_multi_speaker_stats(self):
        """Multiple speakers with different statistics."""
        segments = [
            make_segment(0, 0.0, 3.0, "Speaker 0", "spk_0"),
            make_segment(1, 3.5, 5.0, "Speaker 1", "spk_1"),
            make_segment(2, 5.5, 9.0, "Speaker 0 again", "spk_0"),
        ]
        transcript = make_transcript(segments)

        stats = compute_speaker_stats(transcript)

        assert len(stats) == 2
        # Stats should be sorted by speaker_id (from durations dict iteration)
        spk_0_stats = next(s for s in stats if s["speaker_id"] == "spk_0")
        spk_1_stats = next(s for s in stats if s["speaker_id"] == "spk_1")

        assert spk_0_stats["total_talk_time"] == 6.5  # 3.0 + 3.5
        assert spk_1_stats["total_talk_time"] == 1.5

    def test_turns_with_metadata(self):
        """Stats with turns including question counts."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello?", "spk_0"),
            make_segment(1, 2.5, 4.0, "Hi there", "spk_1"),
        ]
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=2.0,
                text="Hello?",
                metadata={"question_count": 1, "interruption_started_here": False},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=2.5,
                end=4.0,
                text="Hi there",
                metadata={"question_count": 0, "interruption_started_here": False},
            ),
        ]
        transcript = make_transcript(segments, turns=turns)

        stats = compute_speaker_stats(transcript)

        spk_0_stats = next(s for s in stats if s["speaker_id"] == "spk_0")
        spk_1_stats = next(s for s in stats if s["speaker_id"] == "spk_1")

        assert spk_0_stats["num_turns"] == 1
        assert spk_0_stats["question_turns"] == 1
        assert spk_1_stats["num_turns"] == 1
        assert spk_1_stats["question_turns"] == 0

    def test_interruption_counting(self):
        """Interruptions are counted from turn metadata."""
        segments = [
            make_segment(0, 0.0, 2.0, "A speaks", "spk_0"),
            make_segment(1, 1.5, 4.0, "B interrupts", "spk_1"),
            make_segment(2, 4.5, 6.0, "A speaks again", "spk_0"),
        ]
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=2.0,
                text="A speaks",
                metadata={"interruption_started_here": False},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=1.5,
                end=4.0,
                text="B interrupts",
                metadata={"interruption_started_here": True},  # B interrupted A
            ),
            Turn(
                id="turn_2",
                speaker_id="spk_0",
                segment_ids=[2],
                start=4.5,
                end=6.0,
                text="A speaks again",
                metadata={"interruption_started_here": False},
            ),
        ]
        transcript = make_transcript(segments, turns=turns)

        stats = compute_speaker_stats(transcript)

        spk_0_stats = next(s for s in stats if s["speaker_id"] == "spk_0")
        spk_1_stats = next(s for s in stats if s["speaker_id"] == "spk_1")

        # spk_1 interrupted spk_0 once
        assert spk_1_stats["interruptions_initiated"] == 1
        assert spk_0_stats["interruptions_received"] == 1
        assert spk_0_stats["interruptions_initiated"] == 0
        assert spk_1_stats["interruptions_received"] == 0

    def test_same_speaker_interruption_not_counted(self):
        """Interruptions within same speaker are not counted."""
        segments = [
            make_segment(0, 0.0, 2.0, "A speaks", "spk_0"),
            make_segment(1, 2.5, 4.0, "A continues", "spk_0"),
        ]
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=2.0,
                text="A speaks",
                metadata={"interruption_started_here": False},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_0",
                segment_ids=[1],
                start=2.5,
                end=4.0,
                text="A continues",
                metadata={"interruption_started_here": True},  # Self-interruption
            ),
        ]
        transcript = make_transcript(segments, turns=turns)

        stats = compute_speaker_stats(transcript)

        spk_0_stats = next(s for s in stats if s["speaker_id"] == "spk_0")
        # Self-interruptions are not counted
        assert spk_0_stats["interruptions_initiated"] == 0
        assert spk_0_stats["interruptions_received"] == 0

    def test_avg_turn_duration(self):
        """Average turn duration is computed correctly."""
        segments = [
            make_segment(0, 0.0, 2.0, "Turn 1", "spk_0"),
            make_segment(1, 2.5, 6.0, "Turn 2", "spk_0"),
            make_segment(2, 6.5, 9.0, "Turn 3", "spk_0"),
        ]
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=2.0,
                text="Turn 1",
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_0",
                segment_ids=[1],
                start=2.5,
                end=6.0,
                text="Turn 2",
            ),
            Turn(
                id="turn_2",
                speaker_id="spk_0",
                segment_ids=[2],
                start=6.5,
                end=9.0,
                text="Turn 3",
            ),
        ]
        transcript = make_transcript(segments, turns=turns)

        stats = compute_speaker_stats(transcript)

        spk_0_stats = stats[0]
        assert spk_0_stats["total_talk_time"] == 8.0  # 2.0 + 3.5 + 2.5
        assert spk_0_stats["num_turns"] == 3
        assert spk_0_stats["avg_turn_duration"] == pytest.approx(8.0 / 3.0)

    def test_empty_transcript(self):
        """Empty transcript returns empty stats list."""
        transcript = make_transcript([], speakers=[])

        stats = compute_speaker_stats(transcript)

        assert stats == []

    def test_null_speakers_assigned_to_spk_0(self):
        """Segments without speaker info are assigned to spk_0."""
        segments = [
            make_segment(0, 0.0, 2.0, "No speaker 1"),
            make_segment(1, 2.5, 4.0, "No speaker 2"),
        ]
        transcript = make_transcript(segments, speakers=[])

        stats = compute_speaker_stats(transcript)

        assert len(stats) == 1
        assert stats[0]["speaker_id"] == "spk_0"
        assert stats[0]["total_talk_time"] == 3.5  # 2.0 + 1.5

    def test_no_prosody_data_returns_none(self):
        """Stats without prosody data have None for prosody medians."""
        segments = [
            make_segment(0, 0.0, 2.0, "No prosody", "spk_0"),
        ]
        transcript = make_transcript(segments)

        stats = compute_speaker_stats(transcript)

        prosody = stats[0]["prosody_summary"]
        assert prosody["pitch_median_hz"] is None
        assert prosody["energy_median_db"] is None

    def test_no_sentiment_data_returns_zeros(self):
        """Stats without sentiment data have zero distribution."""
        segments = [
            make_segment(0, 0.0, 2.0, "No sentiment", "spk_0"),
        ]
        transcript = make_transcript(segments)

        stats = compute_speaker_stats(transcript)

        sentiment = stats[0]["sentiment_summary"]
        # With no sentiment counts, total_sentiments=1 (to avoid div by zero)
        # so all percentages are 0/1 = 0.0
        assert sentiment["positive"] == 0.0
        assert sentiment["neutral"] == 0.0
        assert sentiment["negative"] == 0.0

    def test_stats_attached_to_transcript(self):
        """compute_speaker_stats() attaches stats to transcript.speaker_stats."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", "spk_0"),
        ]
        transcript = make_transcript(segments)

        assert not hasattr(transcript, "speaker_stats") or transcript.speaker_stats is None

        stats = compute_speaker_stats(transcript)

        # Stats should be attached to transcript
        assert hasattr(transcript, "speaker_stats")
        assert transcript.speaker_stats == stats

    def test_turn_dict_compatibility(self):
        """compute_speaker_stats() works with Turn dicts instead of dataclasses."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello?", "spk_0"),
        ]
        turns_as_dicts = [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "segment_ids": [0],
                "start": 0.0,
                "end": 2.0,
                "text": "Hello?",
                "metadata": {"question_count": 1, "interruption_started_here": False},
            }
        ]
        transcript = make_transcript(segments, turns=turns_as_dicts)

        stats = compute_speaker_stats(transcript)

        assert stats[0]["question_turns"] == 1

    def test_median_with_even_number_of_values(self):
        """Median calculation works correctly with even number of values."""
        segments = [
            make_segment(0, 0.0, 2.0, "S1", "spk_0", make_prosody_audio_state(200.0, -12.0)),
            make_segment(1, 2.5, 4.0, "S2", "spk_0", make_prosody_audio_state(220.0, -10.0)),
            make_segment(2, 4.5, 6.0, "S3", "spk_0", make_prosody_audio_state(210.0, -14.0)),
            make_segment(3, 6.5, 8.0, "S4", "spk_0", make_prosody_audio_state(230.0, -8.0)),
        ]
        transcript = make_transcript(segments)

        stats = compute_speaker_stats(transcript)

        prosody = stats[0]["prosody_summary"]
        # Median of [200.0, 210.0, 220.0, 230.0] = (210.0 + 220.0) / 2 = 215.0
        assert prosody["pitch_median_hz"] == 215.0
        # Median of [-14.0, -12.0, -10.0, -8.0] = (-12.0 + -10.0) / 2 = -11.0
        assert prosody["energy_median_db"] == -11.0

    def test_complex_multi_speaker_scenario(self):
        """Complex scenario with multiple speakers, turns, prosody, and sentiment."""
        segments = [
            make_segment(
                0,
                0.0,
                2.0,
                "Hello?",
                "spk_0",
                {
                    **make_prosody_audio_state(220.0, -10.0),
                    **make_emotion_audio_state("positive", 0.8),
                },
            ),
            make_segment(
                1,
                2.5,
                4.0,
                "Hi there",
                "spk_1",
                {
                    **make_prosody_audio_state(180.0, -12.0),
                    **make_emotion_audio_state("neutral", 0.5),
                },
            ),
            make_segment(
                2,
                4.5,
                7.0,
                "How are you?",
                "spk_0",
                {
                    **make_prosody_audio_state(210.0, -11.0),
                    **make_emotion_audio_state("positive", 0.7),
                },
            ),
            make_segment(
                3,
                7.5,
                9.0,
                "Not great.",
                "spk_1",
                {
                    **make_prosody_audio_state(190.0, -13.0),
                    **make_emotion_audio_state("negative", 0.6),
                },
            ),
        ]
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=2.0,
                text="Hello?",
                metadata={"question_count": 1, "interruption_started_here": False},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=2.5,
                end=4.0,
                text="Hi there",
                metadata={"question_count": 0, "interruption_started_here": False},
            ),
            Turn(
                id="turn_2",
                speaker_id="spk_0",
                segment_ids=[2],
                start=4.5,
                end=7.0,
                text="How are you?",
                metadata={"question_count": 1, "interruption_started_here": False},
            ),
            Turn(
                id="turn_3",
                speaker_id="spk_1",
                segment_ids=[3],
                start=7.5,
                end=9.0,
                text="Not great.",
                metadata={"question_count": 0, "interruption_started_here": True},
            ),
        ]
        transcript = make_transcript(segments, turns=turns)

        stats = compute_speaker_stats(transcript)

        assert len(stats) == 2

        spk_0_stats = next(s for s in stats if s["speaker_id"] == "spk_0")
        spk_1_stats = next(s for s in stats if s["speaker_id"] == "spk_1")

        # spk_0: 2 turns, 2 questions, total time 4.5s (2.0 + 2.5)
        assert spk_0_stats["num_turns"] == 2
        assert spk_0_stats["question_turns"] == 2
        assert spk_0_stats["total_talk_time"] == 4.5
        assert spk_0_stats["avg_turn_duration"] == pytest.approx(4.5 / 2.0)
        assert spk_0_stats["interruptions_received"] == 1
        assert spk_0_stats["interruptions_initiated"] == 0

        # spk_1: 2 turns, 0 questions, total time 3.0s (1.5 + 1.5)
        assert spk_1_stats["num_turns"] == 2
        assert spk_1_stats["question_turns"] == 0
        assert spk_1_stats["total_talk_time"] == 3.0
        assert spk_1_stats["avg_turn_duration"] == pytest.approx(3.0 / 2.0)
        assert spk_1_stats["interruptions_initiated"] == 1
        assert spk_1_stats["interruptions_received"] == 0

        # Prosody for spk_0: median of [210.0, 220.0], [-11.0, -10.0]
        assert spk_0_stats["prosody_summary"]["pitch_median_hz"] == 215.0
        assert spk_0_stats["prosody_summary"]["energy_median_db"] == -10.5

        # Prosody for spk_1: median of [180.0, 190.0], [-13.0, -12.0]
        assert spk_1_stats["prosody_summary"]["pitch_median_hz"] == 185.0
        assert spk_1_stats["prosody_summary"]["energy_median_db"] == -12.5

        # Sentiment for spk_0: 2 positive, 0 neutral, 0 negative
        assert spk_0_stats["sentiment_summary"]["positive"] == 1.0
        assert spk_0_stats["sentiment_summary"]["neutral"] == 0.0
        assert spk_0_stats["sentiment_summary"]["negative"] == 0.0

        # Sentiment for spk_1: 0 positive, 1 neutral, 1 negative
        assert spk_1_stats["sentiment_summary"]["positive"] == 0.0
        assert spk_1_stats["sentiment_summary"]["neutral"] == 0.5
        assert spk_1_stats["sentiment_summary"]["negative"] == 0.5
