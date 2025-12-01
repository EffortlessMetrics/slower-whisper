"""Tests for semantic annotation module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from transcription.api import load_transcript
from transcription.models import Segment, Transcript
from transcription.semantic import (
    KeywordSemanticAnnotator,
    NoOpSemanticAnnotator,
    SemanticAnnotator,
)
from transcription.writers import write_json


class TestNoOpSemanticAnnotator:
    """Test NoOpSemanticAnnotator."""

    def test_returns_transcript_unchanged(self) -> None:
        """Verify noop annotator returns transcript as-is."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="Hello")],
        )
        annotator = NoOpSemanticAnnotator()
        result = annotator.annotate(transcript)
        assert result is transcript
        assert result.annotations is None


class TestKeywordSemanticAnnotator:
    """Test KeywordSemanticAnnotator."""

    def test_tags_pricing_keywords(self) -> None:
        """Verify pricing keywords are detected."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="What is the price for this?"),
                Segment(id=1, start=2.0, end=4.0, text="Let me check the quote."),
            ],
        )
        annotator = KeywordSemanticAnnotator()
        result = annotator.annotate(transcript)

        assert result.annotations is not None
        semantic = result.annotations.get("semantic", {})
        assert "pricing" in semantic.get("tags", [])

    def test_tags_escalation_keywords(self) -> None:
        """Verify escalation keywords are detected."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="I need to speak to a supervisor."),
            ],
        )
        annotator = KeywordSemanticAnnotator()
        result = annotator.annotate(transcript)

        semantic = result.annotations.get("semantic", {})
        assert "escalation" in semantic.get("tags", [])

    def test_tags_churn_risk_keywords(self) -> None:
        """Verify churn risk keywords are detected."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="I want to cancel my subscription."),
            ],
        )
        annotator = KeywordSemanticAnnotator()
        result = annotator.annotate(transcript)

        semantic = result.annotations.get("semantic", {})
        assert "churn_risk" in semantic.get("tags", [])

    def test_multiple_tags(self) -> None:
        """Verify multiple tags can be detected."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="The pricing is too high."),
                Segment(id=1, start=2.0, end=4.0, text="I want to cancel unless I get a discount."),
            ],
        )
        annotator = KeywordSemanticAnnotator()
        result = annotator.annotate(transcript)

        semantic = result.annotations.get("semantic", {})
        tags = semantic.get("tags", [])
        assert "pricing" in tags
        assert "churn_risk" in tags

    def test_no_tags_when_no_matches(self) -> None:
        """Verify no tags when no keywords match."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="The weather is nice today."),
            ],
        )
        annotator = KeywordSemanticAnnotator()
        result = annotator.annotate(transcript)

        semantic = result.annotations.get("semantic", {})
        tags = semantic.get("tags", [])
        assert tags == []

    def test_matches_recorded(self) -> None:
        """Verify keyword matches are recorded."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="What is the budget?"),
            ],
        )
        annotator = KeywordSemanticAnnotator()
        result = annotator.annotate(transcript)

        semantic = result.annotations.get("semantic", {})
        matches = semantic.get("matches", [])
        assert any(m["keyword"] == "budget" for m in matches)

    def test_custom_keyword_map(self) -> None:
        """Verify custom keyword map works."""
        custom_map = {
            "greeting": ("hello", "hi", "hey"),
            "farewell": ("goodbye", "bye"),
        }
        annotator = KeywordSemanticAnnotator(keyword_map=custom_map)

        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="Hello there!")],
        )
        result = annotator.annotate(transcript)

        semantic = result.annotations.get("semantic", {})
        assert "greeting" in semantic.get("tags", [])

    def test_case_insensitive(self) -> None:
        """Verify matching is case-insensitive."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="WHAT IS THE PRICE?"),
            ],
        )
        annotator = KeywordSemanticAnnotator()
        result = annotator.annotate(transcript)

        semantic = result.annotations.get("semantic", {})
        assert "pricing" in semantic.get("tags", [])


class TestSemanticAnnotationRoundtrip:
    """Test semantic annotations survive JSON round-trip."""

    def test_json_roundtrip(self) -> None:
        """Verify annotations are preserved through JSON."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="Check the pricing please."),
            ],
        )

        # Annotate
        annotator = KeywordSemanticAnnotator()
        annotated = annotator.annotate(transcript)

        # Write to JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            write_json(annotated, json_path)

            # Read back
            loaded = load_transcript(json_path)

            # Verify annotations preserved
            assert loaded.annotations is not None
            semantic = loaded.annotations.get("semantic", {})
            assert "pricing" in semantic.get("tags", [])
        finally:
            json_path.unlink(missing_ok=True)

    def test_empty_annotations_roundtrip(self) -> None:
        """Verify empty annotations work correctly."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=1.0, text="Just a test."),
            ],
        )

        # Annotate (no matches)
        annotator = KeywordSemanticAnnotator()
        annotated = annotator.annotate(transcript)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            write_json(annotated, json_path)
            loaded = load_transcript(json_path)

            # Should have annotations dict with semantic key
            assert loaded.annotations is not None
            semantic = loaded.annotations.get("semantic", {})
            assert semantic.get("tags", []) == []
        finally:
            json_path.unlink(missing_ok=True)


class TestProtocolCompliance:
    """Test SemanticAnnotator protocol compliance."""

    def test_noop_satisfies_protocol(self) -> None:
        """Verify NoOpSemanticAnnotator satisfies protocol."""
        annotator: SemanticAnnotator = NoOpSemanticAnnotator()
        transcript = Transcript(file_name="test.wav", language="en", segments=[])
        result = annotator.annotate(transcript)
        assert isinstance(result, Transcript)

    def test_keyword_satisfies_protocol(self) -> None:
        """Verify KeywordSemanticAnnotator satisfies protocol."""
        annotator: SemanticAnnotator = KeywordSemanticAnnotator()
        transcript = Transcript(file_name="test.wav", language="en", segments=[])
        result = annotator.annotate(transcript)
        assert isinstance(result, Transcript)
