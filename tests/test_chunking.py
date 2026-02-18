from typing import Any

import pytest

from slower_whisper.pipeline.chunking import (
    ChunkingConfig,
    _compute_split_score,
    _count_turn_boundaries,
    _detect_overlapping_speech,
    _detect_rapid_turn_taking,
    _is_turn_boundary,
    build_chunks,
)
from slower_whisper.pipeline.models import Chunk, Segment, Transcript, Turn


def test_chunks_respect_turn_boundaries_when_hitting_max_duration():
    segments = [
        Segment(id=0, start=0.0, end=4.0, text="hello there", speaker={"id": "spk_0"}),
        Segment(id=1, start=4.0, end=8.0, text="more words", speaker={"id": "spk_0"}),
        Segment(id=2, start=8.0, end=12.0, text="reply", speaker={"id": "spk_1"}),
    ]
    turns: list[Turn | dict[str, Any]] = [
        Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0, 1],
            start=0.0,
            end=8.0,
            text="hello there more words",
            metadata={},
        ),
        Turn(
            id="turn_1",
            speaker_id="spk_1",
            segment_ids=[2],
            start=8.0,
            end=12.0,
            text="reply",
            metadata={},
        ),
    ]
    transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

    cfg = ChunkingConfig(
        target_duration_s=6.0, max_duration_s=8.0, target_tokens=400, pause_split_threshold_s=0.5
    )
    chunks = build_chunks(transcript, cfg)

    assert len(chunks) == 2
    assert chunks[0].turn_ids == ["turn_0"]
    assert chunks[1].turn_ids == ["turn_1"]
    assert set(chunks[0].segment_ids) == {0, 1}
    assert set(chunks[1].segment_ids) == {2}


def test_pause_triggers_split_at_target():
    segments = [
        Segment(id=0, start=0.0, end=3.0, text="first", speaker={"id": "spk_0"}),
        Segment(id=1, start=5.5, end=8.5, text="second", speaker={"id": "spk_0"}),
        Segment(id=2, start=9.0, end=11.0, text="third", speaker={"id": "spk_0"}),
    ]
    turns: list[Turn | dict[str, Any]] = [
        Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0],
            start=0.0,
            end=3.0,
            text="first",
            metadata={},
        ),
        Turn(
            id="turn_1",
            speaker_id="spk_0",
            segment_ids=[1],
            start=5.5,
            end=8.5,
            text="second",
            metadata={},
        ),
        Turn(
            id="turn_2",
            speaker_id="spk_0",
            segment_ids=[2],
            start=9.0,
            end=11.0,
            text="third",
            metadata={},
        ),
    ]
    transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

    cfg = ChunkingConfig(
        target_duration_s=5.0, max_duration_s=9.0, target_tokens=400, pause_split_threshold_s=1.5
    )
    chunks = build_chunks(transcript, cfg)

    assert len(chunks) == 2
    assert chunks[0].turn_ids == ["turn_0", "turn_1"]
    assert chunks[1].turn_ids == ["turn_2"]


def test_segment_fallback_when_no_turns():
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="hello"),
        Segment(id=1, start=2.1, end=4.0, text="world"),
    ]
    transcript = Transcript(file_name="demo.wav", language="en", segments=segments)

    cfg = ChunkingConfig(
        target_duration_s=10.0, max_duration_s=12.0, target_tokens=5, pause_split_threshold_s=0.5
    )
    chunks = build_chunks(transcript, cfg)

    assert len(chunks) == 1
    assert chunks[0].segment_ids == [0, 1]
    assert chunks[0].turn_ids == ["seg_0", "seg_1"]


# ==============================================================================
# ChunkingConfig validation tests
# ==============================================================================


class TestChunkingConfigValidation:
    """Tests for ChunkingConfig parameter validation."""

    def test_turn_affinity_valid_range(self):
        """turn_affinity accepts values in [0.0, 1.0]."""
        cfg_low = ChunkingConfig(turn_affinity=0.0)
        assert cfg_low.turn_affinity == 0.0

        cfg_mid = ChunkingConfig(turn_affinity=0.5)
        assert cfg_mid.turn_affinity == 0.5

        cfg_high = ChunkingConfig(turn_affinity=1.0)
        assert cfg_high.turn_affinity == 1.0

    def test_turn_affinity_invalid_below_zero(self):
        """turn_affinity < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="turn_affinity must be in"):
            ChunkingConfig(turn_affinity=-0.1)

    def test_turn_affinity_invalid_above_one(self):
        """turn_affinity > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="turn_affinity must be in"):
            ChunkingConfig(turn_affinity=1.5)

    def test_cross_turn_penalty_valid_range(self):
        """cross_turn_penalty accepts values in [0.0, 2.0]."""
        cfg_low = ChunkingConfig(cross_turn_penalty=0.0)
        assert cfg_low.cross_turn_penalty == 0.0

        cfg_mid = ChunkingConfig(cross_turn_penalty=1.0)
        assert cfg_mid.cross_turn_penalty == 1.0

        cfg_high = ChunkingConfig(cross_turn_penalty=2.0)
        assert cfg_high.cross_turn_penalty == 2.0

    def test_cross_turn_penalty_invalid_below_zero(self):
        """cross_turn_penalty < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="cross_turn_penalty must be in"):
            ChunkingConfig(cross_turn_penalty=-0.5)

    def test_cross_turn_penalty_invalid_above_two(self):
        """cross_turn_penalty > 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="cross_turn_penalty must be in"):
            ChunkingConfig(cross_turn_penalty=2.5)

    def test_min_turn_gap_valid(self):
        """min_turn_gap_s accepts non-negative values."""
        cfg = ChunkingConfig(min_turn_gap_s=0.0)
        assert cfg.min_turn_gap_s == 0.0

        cfg_positive = ChunkingConfig(min_turn_gap_s=0.5)
        assert cfg_positive.min_turn_gap_s == 0.5

    def test_min_turn_gap_invalid_negative(self):
        """min_turn_gap_s < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_turn_gap_s must be >= 0.0"):
            ChunkingConfig(min_turn_gap_s=-0.1)


# ==============================================================================
# Turn boundary detection tests
# ==============================================================================


class TestTurnBoundaryDetection:
    """Tests for turn boundary detection helpers."""

    def test_is_turn_boundary_different_speakers(self):
        """Different speaker IDs indicate a turn boundary."""
        unit1 = {"speaker_id": "spk_0"}
        unit2 = {"speaker_id": "spk_1"}
        assert _is_turn_boundary(unit1, unit2) is True

    def test_is_turn_boundary_same_speaker(self):
        """Same speaker ID means no turn boundary."""
        unit1 = {"speaker_id": "spk_0"}
        unit2 = {"speaker_id": "spk_0"}
        assert _is_turn_boundary(unit1, unit2) is False

    def test_is_turn_boundary_missing_speaker(self):
        """Missing speaker ID means no turn boundary can be detected."""
        unit1 = {"speaker_id": "spk_0"}
        unit2 = {"speaker_id": None}
        assert _is_turn_boundary(unit1, unit2) is False

        unit3 = {"speaker_id": None}
        unit4 = {"speaker_id": "spk_0"}
        assert _is_turn_boundary(unit3, unit4) is False

        unit5: dict[str, Any] = {}
        unit6 = {"speaker_id": "spk_0"}
        assert _is_turn_boundary(unit5, unit6) is False

    def test_count_turn_boundaries_multiple(self):
        """Count multiple speaker transitions."""
        units = [
            {"speaker_id": "spk_0"},
            {"speaker_id": "spk_0"},  # Same, no boundary
            {"speaker_id": "spk_1"},  # Boundary +1
            {"speaker_id": "spk_0"},  # Boundary +1
            {"speaker_id": "spk_0"},  # Same, no boundary
        ]
        assert _count_turn_boundaries(units) == 2

    def test_count_turn_boundaries_single_unit(self):
        """Single unit has no boundaries."""
        units = [{"speaker_id": "spk_0"}]
        assert _count_turn_boundaries(units) == 0

    def test_count_turn_boundaries_empty(self):
        """Empty list has no boundaries."""
        assert _count_turn_boundaries([]) == 0


# ==============================================================================
# Overlapping speech detection tests
# ==============================================================================


class TestOverlappingSpeechDetection:
    """Tests for overlapping speech detection."""

    def test_detect_overlapping_speech_with_overlap(self):
        """Detect when segments overlap in time."""
        units = [
            {"start": 0.0, "end": 5.0},
            {"start": 4.0, "end": 8.0},  # Starts before previous ends
        ]
        assert _detect_overlapping_speech(units) is True

    def test_detect_overlapping_speech_no_overlap(self):
        """No overlap when segments are sequential."""
        units = [
            {"start": 0.0, "end": 5.0},
            {"start": 5.0, "end": 8.0},  # Starts exactly when previous ends
        ]
        assert _detect_overlapping_speech(units) is False

    def test_detect_overlapping_speech_with_gap(self):
        """No overlap when there's a gap between segments."""
        units = [
            {"start": 0.0, "end": 5.0},
            {"start": 6.0, "end": 8.0},  # Gap of 1 second
        ]
        assert _detect_overlapping_speech(units) is False

    def test_detect_overlapping_speech_single_unit(self):
        """Single unit cannot have overlap."""
        units = [{"start": 0.0, "end": 5.0}]
        assert _detect_overlapping_speech(units) is False


# ==============================================================================
# Rapid turn-taking detection tests
# ==============================================================================


class TestRapidTurnTakingDetection:
    """Tests for rapid turn-taking detection."""

    def test_detect_rapid_turn_taking_with_quick_switch(self):
        """Detect rapid turn-taking with minimal gap between speakers."""
        units = [
            {"start": 0.0, "end": 5.0, "speaker_id": "spk_0"},
            {"start": 5.1, "end": 8.0, "speaker_id": "spk_1"},  # Gap of 0.1s
        ]
        assert _detect_rapid_turn_taking(units, min_turn_gap_s=0.3) is True

    def test_detect_rapid_turn_taking_with_normal_gap(self):
        """No rapid turn-taking when gap is above threshold."""
        units = [
            {"start": 0.0, "end": 5.0, "speaker_id": "spk_0"},
            {"start": 5.5, "end": 8.0, "speaker_id": "spk_1"},  # Gap of 0.5s
        ]
        assert _detect_rapid_turn_taking(units, min_turn_gap_s=0.3) is False

    def test_detect_rapid_turn_taking_same_speaker(self):
        """Same speaker transitions don't count as rapid turn-taking."""
        units = [
            {"start": 0.0, "end": 5.0, "speaker_id": "spk_0"},
            {"start": 5.0, "end": 8.0, "speaker_id": "spk_0"},  # Same speaker, no gap
        ]
        assert _detect_rapid_turn_taking(units, min_turn_gap_s=0.3) is False

    def test_detect_rapid_turn_taking_missing_speaker(self):
        """Missing speaker IDs don't trigger rapid turn-taking detection."""
        units = [
            {"start": 0.0, "end": 5.0, "speaker_id": "spk_0"},
            {"start": 5.0, "end": 8.0, "speaker_id": None},  # Missing speaker
        ]
        assert _detect_rapid_turn_taking(units, min_turn_gap_s=0.3) is False


# ==============================================================================
# Split score computation tests
# ==============================================================================


class TestSplitScoreComputation:
    """Tests for the split score computation."""

    def test_split_score_at_turn_boundary_with_high_affinity(self):
        """Turn boundary with high affinity should increase score."""
        config = ChunkingConfig(turn_affinity=1.0, cross_turn_penalty=1.0)
        score_at_boundary = _compute_split_score(
            current_duration=30.0,
            token_estimate=400,
            gap=1.5,
            is_turn_boundary=True,
            config=config,
        )
        score_not_at_boundary = _compute_split_score(
            current_duration=30.0,
            token_estimate=400,
            gap=1.5,
            is_turn_boundary=False,
            config=config,
        )
        assert score_at_boundary > score_not_at_boundary

    def test_split_score_with_zero_affinity(self):
        """Zero turn_affinity means turn boundaries don't affect score."""
        config = ChunkingConfig(turn_affinity=0.0, cross_turn_penalty=1.0)
        score_at_boundary = _compute_split_score(
            current_duration=30.0,
            token_estimate=400,
            gap=1.5,
            is_turn_boundary=True,
            config=config,
        )
        score_not_at_boundary = _compute_split_score(
            current_duration=30.0,
            token_estimate=400,
            gap=1.5,
            is_turn_boundary=False,
            config=config,
        )
        assert score_at_boundary == score_not_at_boundary

    def test_split_score_increases_with_duration(self):
        """Score should increase as duration approaches target."""
        config = ChunkingConfig(turn_affinity=0.5)
        score_short = _compute_split_score(
            current_duration=10.0,  # 1/3 of target
            token_estimate=100,
            gap=0.5,
            is_turn_boundary=False,
            config=config,
        )
        score_long = _compute_split_score(
            current_duration=30.0,  # At target
            token_estimate=100,
            gap=0.5,
            is_turn_boundary=False,
            config=config,
        )
        assert score_long > score_short


# ==============================================================================
# Chunk metadata tests
# ==============================================================================


class TestChunkMetadata:
    """Tests for chunk turn-aware metadata."""

    def test_chunk_crosses_turn_boundary_metadata(self):
        """Chunks spanning multiple turns have crosses_turn_boundary=True."""
        segments = [
            Segment(id=0, start=0.0, end=10.0, text="hello", speaker={"id": "spk_0"}),
            Segment(id=1, start=10.0, end=20.0, text="world", speaker={"id": "spk_1"}),
        ]
        turns: list[Turn | dict[str, Any]] = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=10.0,
                text="hello",
                metadata={},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=10.0,
                end=20.0,
                text="world",
                metadata={},
            ),
        ]
        transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

        # Force a single chunk by using high limits
        cfg = ChunkingConfig(
            target_duration_s=100.0,
            max_duration_s=100.0,
            target_tokens=1000,
            turn_affinity=0.0,  # Disable turn-based splitting
        )
        chunks = build_chunks(transcript, cfg)

        assert len(chunks) == 1
        assert chunks[0].crosses_turn_boundary is True
        assert chunks[0].turn_boundary_count == 1

    def test_chunk_single_turn_no_boundary_crossing(self):
        """Single-turn chunks have crosses_turn_boundary=False."""
        segments = [
            Segment(id=0, start=0.0, end=5.0, text="hello there", speaker={"id": "spk_0"}),
        ]
        turns: list[Turn | dict[str, Any]] = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=5.0,
                text="hello there",
                metadata={},
            ),
        ]
        transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

        cfg = ChunkingConfig()
        chunks = build_chunks(transcript, cfg)

        assert len(chunks) == 1
        assert chunks[0].crosses_turn_boundary is False
        assert chunks[0].turn_boundary_count == 0

    def test_chunk_rapid_turn_taking_metadata(self):
        """Chunks with rapid turn-taking are flagged."""
        segments = [
            Segment(id=0, start=0.0, end=5.0, text="hello", speaker={"id": "spk_0"}),
            Segment(id=1, start=5.1, end=10.0, text="world", speaker={"id": "spk_1"}),
        ]
        turns: list[Turn | dict[str, Any]] = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=5.0,
                text="hello",
                metadata={},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=5.1,  # Only 0.1s gap
                end=10.0,
                text="world",
                metadata={},
            ),
        ]
        transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

        cfg = ChunkingConfig(
            target_duration_s=100.0,
            max_duration_s=100.0,
            target_tokens=1000,
            turn_affinity=0.0,
            min_turn_gap_s=0.3,  # Gap of 0.1s is less than threshold
        )
        chunks = build_chunks(transcript, cfg)

        assert len(chunks) == 1
        assert chunks[0].has_rapid_turn_taking is True

    def test_chunk_overlapping_speech_metadata(self):
        """Chunks with overlapping speech are flagged."""
        segments = [
            Segment(id=0, start=0.0, end=6.0, text="hello", speaker={"id": "spk_0"}),
            Segment(id=1, start=5.0, end=10.0, text="world", speaker={"id": "spk_1"}),
        ]
        turns: list[Turn | dict[str, Any]] = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=6.0,
                text="hello",
                metadata={},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=5.0,  # Starts before previous ends (overlap)
                end=10.0,
                text="world",
                metadata={},
            ),
        ]
        transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

        cfg = ChunkingConfig(
            target_duration_s=100.0,
            max_duration_s=100.0,
            target_tokens=1000,
            turn_affinity=0.0,
        )
        chunks = build_chunks(transcript, cfg)

        assert len(chunks) == 1
        assert chunks[0].has_overlapping_speech is True


# ==============================================================================
# Turn affinity behavior tests
# ==============================================================================


class TestTurnAffinityBehavior:
    """Tests for turn_affinity parameter effects on chunking."""

    def test_high_affinity_splits_at_turn_boundaries(self):
        """High turn_affinity causes splits at speaker changes."""
        segments = [
            Segment(id=0, start=0.0, end=20.0, text="hello " * 50, speaker={"id": "spk_0"}),
            Segment(id=1, start=20.0, end=40.0, text="world " * 50, speaker={"id": "spk_1"}),
        ]
        turns: list[Turn | dict[str, Any]] = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=20.0,
                text="hello " * 50,
                metadata={},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=20.0,
                end=40.0,
                text="world " * 50,
                metadata={},
            ),
        ]
        transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

        # With high affinity, should split at turn boundary even before hard limit
        cfg_high = ChunkingConfig(
            target_duration_s=30.0,
            max_duration_s=50.0,
            target_tokens=500,
            turn_affinity=1.0,
        )
        chunks_high = build_chunks(transcript, cfg_high)

        # High affinity should create 2 chunks (one per turn)
        assert len(chunks_high) == 2
        assert chunks_high[0].turn_ids == ["turn_0"]
        assert chunks_high[1].turn_ids == ["turn_1"]

    def test_zero_affinity_ignores_turn_boundaries(self):
        """Zero turn_affinity chunks based purely on duration/tokens."""
        segments = [
            Segment(id=0, start=0.0, end=10.0, text="hello", speaker={"id": "spk_0"}),
            Segment(id=1, start=10.0, end=20.0, text="world", speaker={"id": "spk_1"}),
        ]
        turns: list[Turn | dict[str, Any]] = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=10.0,
                text="hello",
                metadata={},
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=10.0,
                end=20.0,
                text="world",
                metadata={},
            ),
        ]
        transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

        cfg = ChunkingConfig(
            target_duration_s=30.0,
            max_duration_s=50.0,
            target_tokens=500,
            turn_affinity=0.0,
        )
        chunks = build_chunks(transcript, cfg)

        # Should be one chunk since we're under all limits
        assert len(chunks) == 1
        assert set(chunks[0].turn_ids) == {"turn_0", "turn_1"}


# ==============================================================================
# Chunk.to_dict and from_dict tests
# ==============================================================================


class TestChunkSerialization:
    """Tests for Chunk serialization with new metadata fields."""

    def test_chunk_to_dict_includes_new_fields(self):
        """to_dict includes all turn-aware metadata fields."""
        chunk = Chunk(
            id="chunk_0",
            start=0.0,
            end=10.0,
            segment_ids=[0, 1],
            turn_ids=["turn_0", "turn_1"],
            speaker_ids=["spk_0", "spk_1"],
            token_count_estimate=50,
            text="hello world",
            crosses_turn_boundary=True,
            turn_boundary_count=1,
            has_rapid_turn_taking=True,
            has_overlapping_speech=False,
        )

        d = chunk.to_dict()

        assert d["crosses_turn_boundary"] is True
        assert d["turn_boundary_count"] == 1
        assert d["has_rapid_turn_taking"] is True
        assert d["has_overlapping_speech"] is False

    def test_chunk_from_dict_loads_new_fields(self):
        """from_dict properly loads turn-aware metadata fields."""
        d = {
            "id": "chunk_0",
            "start": 0.0,
            "end": 10.0,
            "segment_ids": [0, 1],
            "turn_ids": ["turn_0"],
            "speaker_ids": ["spk_0"],
            "token_count_estimate": 50,
            "text": "hello",
            "crosses_turn_boundary": True,
            "turn_boundary_count": 2,
            "has_rapid_turn_taking": True,
            "has_overlapping_speech": True,
        }

        chunk = Chunk.from_dict(d)

        assert chunk.crosses_turn_boundary is True
        assert chunk.turn_boundary_count == 2
        assert chunk.has_rapid_turn_taking is True
        assert chunk.has_overlapping_speech is True

    def test_chunk_from_dict_defaults_new_fields(self):
        """from_dict defaults new fields to False/0 for backward compatibility."""
        d = {
            "id": "chunk_0",
            "start": 0.0,
            "end": 10.0,
            # No new fields
        }

        chunk = Chunk.from_dict(d)

        assert chunk.crosses_turn_boundary is False
        assert chunk.turn_boundary_count == 0
        assert chunk.has_rapid_turn_taking is False
        assert chunk.has_overlapping_speech is False


# ==============================================================================
# Edge cases tests
# ==============================================================================


class TestChunkingEdgeCases:
    """Tests for edge cases in chunking."""

    def test_empty_transcript(self):
        """Empty transcript produces no chunks."""
        transcript = Transcript(file_name="demo.wav", language="en", segments=[])

        cfg = ChunkingConfig()
        chunks = build_chunks(transcript, cfg)

        assert chunks == []

    def test_single_segment_no_speaker(self):
        """Single segment without speaker info still produces a chunk."""
        segments = [
            Segment(id=0, start=0.0, end=5.0, text="hello"),
        ]
        transcript = Transcript(file_name="demo.wav", language="en", segments=segments)

        cfg = ChunkingConfig()
        chunks = build_chunks(transcript, cfg)

        assert len(chunks) == 1
        assert chunks[0].segment_ids == [0]
        assert chunks[0].crosses_turn_boundary is False

    def test_many_rapid_speaker_changes(self):
        """Handle many rapid speaker changes (interview-style)."""
        segments = []
        turns: list[Turn | dict[str, Any]] = []
        for i in range(10):
            speaker = f"spk_{i % 2}"
            segments.append(
                Segment(
                    id=i,
                    start=float(i * 2),
                    end=float(i * 2 + 1.8),
                    text=f"utterance {i}",
                    speaker={"id": speaker},
                )
            )
            turns.append(
                Turn(
                    id=f"turn_{i}",
                    speaker_id=speaker,
                    segment_ids=[i],
                    start=float(i * 2),
                    end=float(i * 2 + 1.8),
                    text=f"utterance {i}",
                    metadata={},
                )
            )

        transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

        cfg = ChunkingConfig(
            target_duration_s=10.0,
            max_duration_s=15.0,
            turn_affinity=0.5,
            min_turn_gap_s=0.3,
        )
        chunks = build_chunks(transcript, cfg)

        # All chunks should detect rapid turn-taking (0.2s gaps)
        for chunk in chunks:
            if chunk.turn_boundary_count > 0:
                assert chunk.has_rapid_turn_taking is True
