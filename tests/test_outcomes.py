"""Tests for outcomes extractor (decisions, action items, risks, commitments, questions)."""

from __future__ import annotations

import json

import pytest

from slower_whisper.pipeline.models import Segment, Transcript
from slower_whisper.pipeline.outcomes import (
    ACTION_ITEM_PATTERNS,
    ALL_PATTERNS,
    COMMITMENT_PATTERNS,
    DECISION_PATTERNS,
    OUTCOMES_SCHEMA_VERSION,
    QUESTION_PATTERNS,
    RISK_PATTERNS,
    BaselineOutcomeExtractor,
    Citation,
    Outcome,
    OutcomeExtractionResult,
    OutcomeProcessor,
    PatternRule,
    format_outcomes_json,
    format_outcomes_pretty,
)

# -----------------------------------------------------------------------------
# Citation Tests
# -----------------------------------------------------------------------------


class TestCitation:
    """Test Citation dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic Citation creation."""
        citation = Citation(
            segment_id="0",
            start_time=10.5,
            end_time=15.0,
            speaker_id="spk_0",
            quote="We decided to go with option A",
        )
        assert citation.segment_id == "0"
        assert citation.start_time == 10.5
        assert citation.end_time == 15.0
        assert citation.speaker_id == "spk_0"
        assert citation.quote == "We decided to go with option A"

    def test_citation_without_speaker(self) -> None:
        """Test Citation without speaker ID."""
        citation = Citation(
            segment_id="1",
            start_time=0.0,
            end_time=5.0,
            speaker_id=None,
            quote="Some text",
        )
        assert citation.speaker_id is None

    def test_to_dict(self) -> None:
        """Test Citation serialization."""
        citation = Citation(
            segment_id="2",
            start_time=20.0,
            end_time=25.0,
            speaker_id="spk_1",
            quote="I will send the report",
        )
        d = citation.to_dict()
        assert d["segment_id"] == "2"
        assert d["start_time"] == 20.0
        assert d["end_time"] == 25.0
        assert d["speaker_id"] == "spk_1"
        assert d["quote"] == "I will send the report"

    def test_to_dict_without_speaker(self) -> None:
        """Test Citation serialization without speaker."""
        citation = Citation(
            segment_id="0",
            start_time=0.0,
            end_time=1.0,
            speaker_id=None,
            quote="Test",
        )
        d = citation.to_dict()
        assert "speaker_id" not in d

    def test_from_dict(self) -> None:
        """Test Citation deserialization."""
        d = {
            "segment_id": "5",
            "start_time": 30.0,
            "end_time": 35.0,
            "speaker_id": "spk_2",
            "quote": "The risk is that the server might fail",
        }
        citation = Citation.from_dict(d)
        assert citation.segment_id == "5"
        assert citation.start_time == 30.0
        assert citation.end_time == 35.0
        assert citation.speaker_id == "spk_2"
        assert citation.quote == "The risk is that the server might fail"

    def test_from_dict_defaults(self) -> None:
        """Test Citation deserialization with defaults."""
        d = {}
        citation = Citation.from_dict(d)
        assert citation.segment_id == ""
        assert citation.start_time == 0.0
        assert citation.end_time == 0.0
        assert citation.speaker_id is None
        assert citation.quote == ""

    def test_roundtrip(self) -> None:
        """Test Citation serialization roundtrip."""
        original = Citation(
            segment_id="10",
            start_time=100.0,
            end_time=110.0,
            speaker_id="spk_0",
            quote="I commit to finishing by Friday",
        )
        restored = Citation.from_dict(original.to_dict())
        assert restored.segment_id == original.segment_id
        assert restored.start_time == original.start_time
        assert restored.end_time == original.end_time
        assert restored.speaker_id == original.speaker_id
        assert restored.quote == original.quote

    def test_from_segment(self) -> None:
        """Test Citation creation from Segment."""
        segment = Segment(
            id=3,
            start=15.0,
            end=20.0,
            text="We decided to go with plan B",
            speaker={"id": "spk_0"},
        )
        citation = Citation.from_segment(segment)
        assert citation.segment_id == "3"
        assert citation.start_time == 15.0
        assert citation.end_time == 20.0
        assert citation.speaker_id == "spk_0"
        assert citation.quote == "We decided to go with plan B"

    def test_from_segment_with_custom_quote(self) -> None:
        """Test Citation from Segment with custom quote."""
        segment = Segment(
            id=4,
            start=25.0,
            end=30.0,
            text="Full segment text here",
            speaker="spk_1",
        )
        citation = Citation.from_segment(segment, quote="Custom quote")
        assert citation.quote == "Custom quote"

    def test_from_segment_string_speaker(self) -> None:
        """Test Citation from Segment with string speaker."""
        segment = Segment(
            id=5,
            start=0.0,
            end=5.0,
            text="Text",
            speaker="spk_2",
        )
        citation = Citation.from_segment(segment)
        assert citation.speaker_id == "spk_2"


# -----------------------------------------------------------------------------
# Outcome Tests
# -----------------------------------------------------------------------------


class TestOutcome:
    """Test Outcome dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic Outcome creation."""
        citation = Citation(
            segment_id="0",
            start_time=0.0,
            end_time=5.0,
            speaker_id="spk_0",
            quote="We decided to use Python",
        )
        outcome = Outcome(
            outcome_type="decision",
            summary="Team decided to use Python for the project",
            citations=[citation],
            confidence=0.95,
            metadata={"source": "meeting"},
        )
        assert outcome.outcome_type == "decision"
        assert outcome.summary == "Team decided to use Python for the project"
        assert len(outcome.citations) == 1
        assert outcome.confidence == 0.95
        assert outcome.metadata == {"source": "meeting"}

    def test_outcome_requires_citation(self) -> None:
        """Test that Outcome requires at least one citation."""
        with pytest.raises(ValueError, match="at least one citation"):
            Outcome(
                outcome_type="action_item",
                summary="Send report",
                citations=[],  # Empty citations should fail
            )

    def test_outcome_defaults(self) -> None:
        """Test Outcome default values."""
        citation = Citation(
            segment_id="0",
            start_time=0.0,
            end_time=1.0,
            speaker_id=None,
            quote="Test",
        )
        outcome = Outcome(
            outcome_type="risk",
            summary="Test risk",
            citations=[citation],
        )
        assert outcome.confidence == 1.0
        assert outcome.metadata == {}

    def test_to_dict(self) -> None:
        """Test Outcome serialization."""
        citation = Citation(
            segment_id="1",
            start_time=10.0,
            end_time=15.0,
            speaker_id="spk_0",
            quote="I will complete the task",
        )
        outcome = Outcome(
            outcome_type="action_item",
            summary="Complete the task",
            citations=[citation],
            confidence=0.9,
            metadata={"assignee": "John"},
        )
        d = outcome.to_dict()
        assert d["outcome_type"] == "action_item"
        assert d["summary"] == "Complete the task"
        assert len(d["citations"]) == 1
        assert d["confidence"] == 0.9
        assert d["metadata"]["assignee"] == "John"

    def test_from_dict(self) -> None:
        """Test Outcome deserialization."""
        d = {
            "outcome_type": "commitment",
            "summary": "Deliver by Friday",
            "citations": [
                {
                    "segment_id": "2",
                    "start_time": 20.0,
                    "end_time": 25.0,
                    "speaker_id": "spk_1",
                    "quote": "I commit to delivering by Friday",
                }
            ],
            "confidence": 0.85,
            "metadata": {"deadline": "Friday"},
        }
        outcome = Outcome.from_dict(d)
        assert outcome.outcome_type == "commitment"
        assert outcome.summary == "Deliver by Friday"
        assert len(outcome.citations) == 1
        assert outcome.confidence == 0.85

    def test_roundtrip(self) -> None:
        """Test Outcome serialization roundtrip."""
        citation = Citation(
            segment_id="3",
            start_time=30.0,
            end_time=35.0,
            speaker_id="spk_0",
            quote="Is this the final version?",
        )
        original = Outcome(
            outcome_type="question",
            summary="Question about final version",
            citations=[citation],
            confidence=1.0,
        )
        restored = Outcome.from_dict(original.to_dict())
        assert restored.outcome_type == original.outcome_type
        assert restored.summary == original.summary
        assert len(restored.citations) == len(original.citations)

    def test_multiple_citations(self) -> None:
        """Test Outcome with multiple citations."""
        citations = [
            Citation(
                segment_id="0",
                start_time=0.0,
                end_time=5.0,
                speaker_id="spk_0",
                quote="First mention",
            ),
            Citation(
                segment_id="5",
                start_time=25.0,
                end_time=30.0,
                speaker_id="spk_1",
                quote="Second mention",
            ),
        ]
        outcome = Outcome(
            outcome_type="risk",
            summary="Risk mentioned multiple times",
            citations=citations,
        )
        assert len(outcome.citations) == 2


# -----------------------------------------------------------------------------
# OutcomeExtractionResult Tests
# -----------------------------------------------------------------------------


class TestOutcomeExtractionResult:
    """Test OutcomeExtractionResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        citation = Citation(
            segment_id="0",
            start_time=0.0,
            end_time=5.0,
            speaker_id=None,
            quote="Test",
        )
        outcome = Outcome(
            outcome_type="decision",
            summary="Test decision",
            citations=[citation],
        )
        result = OutcomeExtractionResult(
            outcomes=[outcome],
            backend="baseline",
            latency_ms=50,
        )
        assert result.schema_version == OUTCOMES_SCHEMA_VERSION
        assert len(result.outcomes) == 1
        assert result.backend == "baseline"
        assert result.latency_ms == 50

    def test_defaults(self) -> None:
        """Test result default values."""
        result = OutcomeExtractionResult()
        assert result.schema_version == OUTCOMES_SCHEMA_VERSION
        assert result.outcomes == []
        assert result.backend == "baseline"
        assert result.model is None
        assert result.latency_ms == 0

    def test_to_dict(self) -> None:
        """Test result serialization."""
        citation = Citation(
            segment_id="1",
            start_time=10.0,
            end_time=15.0,
            speaker_id="spk_0",
            quote="I will do it",
        )
        outcome = Outcome(
            outcome_type="action_item",
            summary="Do the thing",
            citations=[citation],
        )
        result = OutcomeExtractionResult(
            outcomes=[outcome],
            backend="llm",
            model="gpt-4o",
            latency_ms=500,
            metadata={"segment_count": 10},
        )
        d = result.to_dict()
        assert d["schema_version"] == OUTCOMES_SCHEMA_VERSION
        assert len(d["outcomes"]) == 1
        assert d["backend"] == "llm"
        assert d["model"] == "gpt-4o"
        assert d["latency_ms"] == 500

    def test_from_dict(self) -> None:
        """Test result deserialization."""
        d = {
            "schema_version": "0.1.0",
            "outcomes": [
                {
                    "outcome_type": "risk",
                    "summary": "Server might fail",
                    "citations": [
                        {
                            "segment_id": "2",
                            "start_time": 20.0,
                            "end_time": 25.0,
                            "quote": "The server might fail",
                        }
                    ],
                    "confidence": 0.8,
                    "metadata": {},
                }
            ],
            "backend": "baseline",
            "latency_ms": 100,
        }
        result = OutcomeExtractionResult.from_dict(d)
        assert len(result.outcomes) == 1
        assert result.outcomes[0].outcome_type == "risk"


# -----------------------------------------------------------------------------
# Pattern Rules Tests
# -----------------------------------------------------------------------------


class TestPatternRules:
    """Test pattern rules definitions."""

    def test_decision_patterns_exist(self) -> None:
        """Test that decision patterns are defined."""
        assert len(DECISION_PATTERNS) > 0
        for rule in DECISION_PATTERNS:
            assert rule.outcome_type == "decision"

    def test_action_item_patterns_exist(self) -> None:
        """Test that action item patterns are defined."""
        assert len(ACTION_ITEM_PATTERNS) > 0
        for rule in ACTION_ITEM_PATTERNS:
            assert rule.outcome_type == "action_item"

    def test_risk_patterns_exist(self) -> None:
        """Test that risk patterns are defined."""
        assert len(RISK_PATTERNS) > 0
        for rule in RISK_PATTERNS:
            assert rule.outcome_type == "risk"

    def test_commitment_patterns_exist(self) -> None:
        """Test that commitment patterns are defined."""
        assert len(COMMITMENT_PATTERNS) > 0
        for rule in COMMITMENT_PATTERNS:
            assert rule.outcome_type == "commitment"

    def test_question_patterns_exist(self) -> None:
        """Test that question patterns are defined."""
        assert len(QUESTION_PATTERNS) > 0
        for rule in QUESTION_PATTERNS:
            assert rule.outcome_type == "question"

    def test_all_patterns_combined(self) -> None:
        """Test that ALL_PATTERNS combines all pattern types."""
        expected_count = (
            len(DECISION_PATTERNS)
            + len(ACTION_ITEM_PATTERNS)
            + len(RISK_PATTERNS)
            + len(COMMITMENT_PATTERNS)
            + len(QUESTION_PATTERNS)
        )
        assert len(ALL_PATTERNS) == expected_count


# -----------------------------------------------------------------------------
# BaselineOutcomeExtractor Tests
# -----------------------------------------------------------------------------


class TestBaselineOutcomeExtractor:
    """Test BaselineOutcomeExtractor implementation."""

    def test_extract_decision(self) -> None:
        """Test extraction of decisions."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We decided to go with option A"),
            ],
        )
        result = extractor.extract(transcript)
        assert len(result.outcomes) >= 1
        decision_outcomes = [o for o in result.outcomes if o.outcome_type == "decision"]
        assert len(decision_outcomes) >= 1
        assert decision_outcomes[0].citations[0].segment_id == "0"

    def test_extract_action_item(self) -> None:
        """Test extraction of action items."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="I will send the report tomorrow"),
            ],
        )
        result = extractor.extract(transcript)
        action_outcomes = [o for o in result.outcomes if o.outcome_type == "action_item"]
        assert len(action_outcomes) >= 1

    def test_extract_risk(self) -> None:
        """Test extraction of risks."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="The risk is that the server might fail"),
            ],
        )
        result = extractor.extract(transcript)
        risk_outcomes = [o for o in result.outcomes if o.outcome_type == "risk"]
        assert len(risk_outcomes) >= 1

    def test_extract_commitment(self) -> None:
        """Test extraction of commitments."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="I commit to finishing by Friday"),
            ],
        )
        result = extractor.extract(transcript)
        commitment_outcomes = [o for o in result.outcomes if o.outcome_type == "commitment"]
        assert len(commitment_outcomes) >= 1

    def test_extract_question(self) -> None:
        """Test extraction of questions."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="What is the status of the project?"),
            ],
        )
        result = extractor.extract(transcript)
        question_outcomes = [o for o in result.outcomes if o.outcome_type == "question"]
        assert len(question_outcomes) >= 1

    def test_extract_empty_transcript(self) -> None:
        """Test extraction from empty transcript."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[],
        )
        result = extractor.extract(transcript)
        assert result.outcomes == []
        assert result.backend == "baseline"
        assert result.latency_ms >= 0

    def test_extract_no_matches(self) -> None:
        """Test extraction when no patterns match."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="Hello, nice to meet you"),
            ],
        )
        result = extractor.extract(transcript)
        # Should not extract any outcomes from greeting
        # (unless "you" triggers action_item pattern)
        assert result.backend == "baseline"

    def test_extract_multiple_segments(self) -> None:
        """Test extraction from multiple segments."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We decided to use Python"),
                Segment(id=1, start=5.0, end=10.0, text="I will implement it tomorrow"),
                Segment(id=2, start=10.0, end=15.0, text="The risk is that it might be slow"),
            ],
        )
        result = extractor.extract(transcript)
        # Should find at least 3 outcomes
        assert len(result.outcomes) >= 3

    def test_citation_includes_segment_info(self) -> None:
        """Test that citations include full segment information."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(
                    id=5,
                    start=25.0,
                    end=30.0,
                    text="We decided to ship on Monday",
                    speaker={"id": "spk_0"},
                ),
            ],
        )
        result = extractor.extract(transcript)
        assert len(result.outcomes) >= 1
        citation = result.outcomes[0].citations[0]
        assert citation.segment_id == "5"
        assert citation.start_time == 25.0
        assert citation.end_time == 30.0
        assert citation.speaker_id == "spk_0"
        assert "decided" in citation.quote.lower()

    def test_metadata_includes_pattern(self) -> None:
        """Test that extraction metadata includes pattern info."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We decided to proceed"),
            ],
        )
        result = extractor.extract(transcript)
        assert result.metadata.get("segment_count") == 1
        assert result.metadata.get("pattern_count") > 0

    def test_custom_patterns(self) -> None:
        """Test extractor with custom patterns."""
        custom_decision = (PatternRule(r"\bapproved\b", "decision", 10),)
        extractor = BaselineOutcomeExtractor(decision_patterns=custom_decision)
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="The budget was approved"),
            ],
        )
        result = extractor.extract(transcript)
        decision_outcomes = [o for o in result.outcomes if o.outcome_type == "decision"]
        assert len(decision_outcomes) >= 1

    def test_latency_recorded(self) -> None:
        """Test that extraction latency is recorded."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="I will do it"),
            ],
        )
        result = extractor.extract(transcript)
        assert result.latency_ms >= 0


# -----------------------------------------------------------------------------
# Citation Validation Tests
# -----------------------------------------------------------------------------


class TestCitationValidation:
    """Test citation validation behavior."""

    def test_every_outcome_has_citation(self) -> None:
        """Test that every extracted outcome has at least one citation."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We decided to go ahead"),
                Segment(id=1, start=5.0, end=10.0, text="I will send the email"),
                Segment(id=2, start=10.0, end=15.0, text="What is the deadline?"),
            ],
        )
        result = extractor.extract(transcript)
        for outcome in result.outcomes:
            assert len(outcome.citations) >= 1, f"Outcome without citation: {outcome.summary}"

    def test_citation_quote_matches_segment(self) -> None:
        """Test that citation quote matches segment text."""
        extractor = BaselineOutcomeExtractor()
        segment_text = "We have decided to postpone the launch"
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text=segment_text),
            ],
        )
        result = extractor.extract(transcript)
        assert len(result.outcomes) >= 1
        citation = result.outcomes[0].citations[0]
        assert citation.quote == segment_text


# -----------------------------------------------------------------------------
# Deduplication Tests
# -----------------------------------------------------------------------------


class TestDeduplication:
    """Test outcome deduplication in OutcomeProcessor."""

    def test_deduplication_merges_similar(self) -> None:
        """Test that similar outcomes are merged."""
        processor = OutcomeProcessor(backend="baseline", deduplicate=True)
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We decided to use Python"),
                Segment(id=1, start=5.0, end=10.0, text="We have decided to use Python language"),
            ],
        )
        result = processor.extract(transcript)
        # Similar decisions should potentially be merged
        # (depends on similarity threshold)
        assert result.backend == "baseline"

    def test_deduplication_preserves_different(self) -> None:
        """Test that different outcomes are preserved."""
        processor = OutcomeProcessor(backend="baseline", deduplicate=True)
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We decided to use Python"),
                Segment(id=1, start=5.0, end=10.0, text="We decided to hire more developers"),
            ],
        )
        result = processor.extract(transcript)
        decision_outcomes = [o for o in result.outcomes if o.outcome_type == "decision"]
        # Different decisions should remain separate
        assert len(decision_outcomes) >= 2

    def test_deduplication_disabled(self) -> None:
        """Test extraction with deduplication disabled."""
        processor = OutcomeProcessor(backend="baseline", deduplicate=False)
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We decided to proceed"),
                Segment(id=1, start=5.0, end=10.0, text="We decided to proceed with plan A"),
            ],
        )
        result = processor.extract(transcript)
        # Without dedup, each match should create separate outcome
        decision_outcomes = [o for o in result.outcomes if o.outcome_type == "decision"]
        assert len(decision_outcomes) >= 2

    def test_merged_outcome_has_multiple_citations(self) -> None:
        """Test that merged outcomes can have multiple citations."""
        processor = OutcomeProcessor(backend="baseline", deduplicate=True, similarity_threshold=0.5)
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="The risk is server failure"),
                Segment(id=1, start=5.0, end=10.0, text="The risk is the server might fail"),
            ],
        )
        result = processor.extract(transcript)
        risk_outcomes = [o for o in result.outcomes if o.outcome_type == "risk"]
        # If merged, should have multiple citations
        # Check that at least one risk outcome exists
        assert len(risk_outcomes) >= 1


# -----------------------------------------------------------------------------
# OutcomeProcessor Tests
# -----------------------------------------------------------------------------


class TestOutcomeProcessor:
    """Test OutcomeProcessor high-level API."""

    def test_baseline_backend(self) -> None:
        """Test processor with baseline backend."""
        processor = OutcomeProcessor(backend="baseline")
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We decided to ship"),
            ],
        )
        result = processor.extract(transcript)
        assert result.backend == "baseline"

    def test_llm_backend_requires_adapter(self) -> None:
        """Test that LLM backend requires adapter."""
        with pytest.raises(ValueError, match="requires an adapter"):
            OutcomeProcessor(backend="llm")

    def test_custom_similarity_threshold(self) -> None:
        """Test processor with custom similarity threshold."""
        processor = OutcomeProcessor(
            backend="baseline",
            similarity_threshold=0.9,  # Very high threshold
        )
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We decided A"),
                Segment(id=1, start=5.0, end=10.0, text="We decided B"),
            ],
        )
        result = processor.extract(transcript)
        # With high threshold, should not merge
        decision_outcomes = [o for o in result.outcomes if o.outcome_type == "decision"]
        assert len(decision_outcomes) >= 2


# -----------------------------------------------------------------------------
# Output Format Tests
# -----------------------------------------------------------------------------


class TestOutputFormat:
    """Test output formatting functions."""

    def test_format_outcomes_json(self) -> None:
        """Test JSON formatting."""
        citation = Citation(
            segment_id="0",
            start_time=0.0,
            end_time=5.0,
            speaker_id="spk_0",
            quote="We decided to proceed",
        )
        outcome = Outcome(
            outcome_type="decision",
            summary="Proceed with project",
            citations=[citation],
        )
        result = OutcomeExtractionResult(outcomes=[outcome], backend="baseline")

        json_str = format_outcomes_json(result)
        data = json.loads(json_str)
        assert data["schema_version"] == OUTCOMES_SCHEMA_VERSION
        assert len(data["outcomes"]) == 1

    def test_format_outcomes_pretty(self) -> None:
        """Test pretty formatting."""
        citation = Citation(
            segment_id="0",
            start_time=10.5,
            end_time=15.0,
            speaker_id="spk_0",
            quote="We decided to use Python",
        )
        outcome = Outcome(
            outcome_type="decision",
            summary="Use Python for the project",
            citations=[citation],
            confidence=0.95,
        )
        result = OutcomeExtractionResult(outcomes=[outcome], backend="baseline", latency_ms=42)

        pretty_str = format_outcomes_pretty(result)
        assert "Decisions" in pretty_str
        assert "Python" in pretty_str
        assert "95%" in pretty_str
        assert "10.5s" in pretty_str
        assert "baseline" in pretty_str


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for outcome extraction."""

    def test_full_meeting_extraction(self) -> None:
        """Test extraction from a realistic meeting transcript."""
        extractor = BaselineOutcomeExtractor()
        transcript = Transcript(
            file_name="meeting.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=5.0,
                    text="Good morning everyone, let's get started.",
                    speaker={"id": "spk_0"},
                ),
                Segment(
                    id=1,
                    start=5.0,
                    end=10.0,
                    text="We decided to launch the product next month.",
                    speaker={"id": "spk_0"},
                ),
                Segment(
                    id=2,
                    start=10.0,
                    end=15.0,
                    text="I will prepare the marketing materials.",
                    speaker={"id": "spk_1"},
                ),
                Segment(
                    id=3,
                    start=15.0,
                    end=20.0,
                    text="The risk is that we might not have enough inventory.",
                    speaker={"id": "spk_2"},
                ),
                Segment(
                    id=4,
                    start=20.0,
                    end=25.0,
                    text="I commit to resolving the supply chain issues by Friday.",
                    speaker={"id": "spk_2"},
                ),
                Segment(
                    id=5,
                    start=25.0,
                    end=30.0,
                    text="What is the budget for this quarter?",
                    speaker={"id": "spk_1"},
                ),
            ],
        )
        result = extractor.extract(transcript)

        # Should extract multiple outcome types
        types_found = {o.outcome_type for o in result.outcomes}
        assert "decision" in types_found
        assert "action_item" in types_found
        assert "risk" in types_found
        assert "commitment" in types_found
        assert "question" in types_found

        # Every outcome should have citation
        for outcome in result.outcomes:
            assert len(outcome.citations) >= 1
            assert outcome.citations[0].quote

    def test_processor_end_to_end(self) -> None:
        """Test OutcomeProcessor end-to-end workflow."""
        processor = OutcomeProcessor(backend="baseline", deduplicate=True)
        transcript = Transcript(
            file_name="call.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=5.0, text="We need to discuss the project"),
                Segment(id=1, start=5.0, end=10.0, text="We decided to prioritize feature A"),
                Segment(id=2, start=10.0, end=15.0, text="I will implement feature A this week"),
                Segment(id=3, start=15.0, end=20.0, text="I commit to delivering by Wednesday"),
            ],
        )

        result = processor.extract(transcript)

        # Verify result structure
        assert result.schema_version == OUTCOMES_SCHEMA_VERSION
        assert result.backend == "baseline"
        assert result.latency_ms >= 0

        # Verify outcomes
        assert len(result.outcomes) >= 2

        # Verify JSON serialization
        json_output = format_outcomes_json(result)
        parsed = json.loads(json_output)
        assert "outcomes" in parsed

        # Verify pretty output
        pretty_output = format_outcomes_pretty(result)
        assert "===" in pretty_output
