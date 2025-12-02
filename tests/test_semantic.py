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

    def test_empty_transcript_has_empty_semantic_block(self) -> None:
        """No-signal transcripts still emit semantic block with empty lists."""
        transcript = Transcript(file_name="test.wav", language="en", segments=[])
        annotator = KeywordSemanticAnnotator()

        result = annotator.annotate(transcript)
        semantic = (result.annotations or {}).get("semantic", {})
        assert semantic.get("keywords") == []
        assert semantic.get("risk_tags") == []
        assert semantic.get("actions") == []

    def test_detects_escalation_and_churn_keywords(self) -> None:
        """Escalation/churn lexicons map to risk tags and keywords."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=2.0, text="This is unacceptable, I want a manager."),
                Segment(id=1, start=2.0, end=4.0, text="Otherwise I will switch to a competitor."),
            ],
        )
        annotator = KeywordSemanticAnnotator()
        result = annotator.annotate(transcript)

        semantic = (result.annotations or {}).get("semantic", {})
        keywords = semantic.get("keywords", [])
        risk_tags = semantic.get("risk_tags", [])

        assert "escalate" in keywords or "unacceptable" in keywords
        assert "switch" in keywords or "competitor" in keywords
        assert "escalation" in risk_tags
        assert "churn_risk" in risk_tags

    def test_actions_capture_speaker_ids(self) -> None:
        """Action detection attaches speaker_id and segment_ids."""
        transcript = Transcript(
            file_name="call.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=2.0,
                    text="I'll send the invoice after this call.",
                    speaker={"id": "spk_0"},
                ),
                Segment(
                    id=1,
                    start=2.0,
                    end=4.0,
                    text="We will call you back tomorrow.",
                    speaker={"id": "spk_1"},
                ),
            ],
        )
        annotator = KeywordSemanticAnnotator()
        result = annotator.annotate(transcript)

        semantic = (result.annotations or {}).get("semantic", {})
        actions = semantic.get("actions", [])
        assert len(actions) == 2

        action_by_text = {a["text"]: a for a in actions}
        assert action_by_text["I'll send the invoice after this call."]["speaker_id"] == "spk_0"
        assert action_by_text["We will call you back tomorrow."]["speaker_id"] == "spk_1"
        assert action_by_text["I'll send the invoice after this call."]["segment_ids"] == [0]

    def test_idempotent_annotations(self) -> None:
        """Re-running annotator does not duplicate tags or actions."""
        transcript = Transcript(
            file_name="call.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=2.0,
                    text="I'll follow up after I speak to my manager.",
                    speaker={"id": "spk_0"},
                )
            ],
        )
        annotator = KeywordSemanticAnnotator()
        once = annotator.annotate(transcript)
        twice = annotator.annotate(once)

        semantic = (twice.annotations or {}).get("semantic", {})
        assert semantic.get("risk_tags", []) == ["escalation"]
        assert len(semantic.get("actions", [])) == 1


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
            assert semantic.get("keywords", []) == []
            assert semantic.get("risk_tags", []) == []
            assert semantic.get("actions", []) == []
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
