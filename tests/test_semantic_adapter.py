"""Tests for semantic adapter protocol and implementations (Track 3, #88)."""

from __future__ import annotations

import pytest

from slower_whisper.pipeline.models import Segment, Transcript
from slower_whisper.pipeline.semantic_adapter import (
    SEMANTIC_SCHEMA_VERSION,
    ActionItem,
    ChunkContext,
    LocalKeywordAdapter,
    NoOpSemanticAdapter,
    NormalizedAnnotation,
    ProviderHealth,
    SemanticAdapter,
    SemanticAnnotation,
    create_adapter,
)


class TestActionItem:
    """Test ActionItem dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic ActionItem creation."""
        action = ActionItem(
            text="I'll send the report",
            speaker_id="spk_0",
            segment_ids=[0, 1],
            pattern=r"\bi'll\s+",
            confidence=0.95,
        )
        assert action.text == "I'll send the report"
        assert action.speaker_id == "spk_0"
        assert action.segment_ids == [0, 1]
        assert action.pattern == r"\bi'll\s+"
        assert action.confidence == 0.95

    def test_defaults(self) -> None:
        """Test ActionItem default values."""
        action = ActionItem(text="Do something")
        assert action.speaker_id is None
        assert action.segment_ids == []
        assert action.pattern is None
        assert action.confidence == 1.0

    def test_to_dict(self) -> None:
        """Test ActionItem serialization."""
        action = ActionItem(
            text="Follow up",
            speaker_id="spk_1",
            segment_ids=[2],
            pattern=r"\bfollow up\b",
        )
        d = action.to_dict()
        assert d["text"] == "Follow up"
        assert d["speaker_id"] == "spk_1"
        assert d["segment_ids"] == [2]
        assert d["pattern"] == r"\bfollow up\b"
        assert d["confidence"] == 1.0

    def test_to_dict_minimal(self) -> None:
        """Test ActionItem serialization with minimal fields."""
        action = ActionItem(text="Task")
        d = action.to_dict()
        assert d["text"] == "Task"
        assert d["confidence"] == 1.0
        # Optional fields should not be present
        assert "speaker_id" not in d
        assert "segment_ids" not in d
        assert "pattern" not in d

    def test_from_dict(self) -> None:
        """Test ActionItem deserialization."""
        d = {
            "text": "Send email",
            "speaker_id": "spk_0",
            "segment_ids": [1, 2, 3],
            "pattern": r"\bsend\b",
            "confidence": 0.8,
        }
        action = ActionItem.from_dict(d)
        assert action.text == "Send email"
        assert action.speaker_id == "spk_0"
        assert action.segment_ids == [1, 2, 3]
        assert action.pattern == r"\bsend\b"
        assert action.confidence == 0.8

    def test_roundtrip(self) -> None:
        """Test ActionItem serialization roundtrip."""
        original = ActionItem(
            text="Review PR",
            speaker_id="spk_2",
            segment_ids=[5, 6],
            confidence=0.9,
        )
        restored = ActionItem.from_dict(original.to_dict())
        assert restored.text == original.text
        assert restored.speaker_id == original.speaker_id
        assert restored.segment_ids == original.segment_ids
        assert restored.confidence == original.confidence


class TestNormalizedAnnotation:
    """Test NormalizedAnnotation dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic NormalizedAnnotation creation."""
        annotation = NormalizedAnnotation(
            topics=["pricing", "support"],
            intent="question",
            sentiment="negative",
            action_items=[ActionItem(text="Call back")],
            risk_tags=["churn_risk"],
        )
        assert annotation.topics == ["pricing", "support"]
        assert annotation.intent == "question"
        assert annotation.sentiment == "negative"
        assert len(annotation.action_items) == 1
        assert annotation.risk_tags == ["churn_risk"]

    def test_defaults(self) -> None:
        """Test NormalizedAnnotation default values."""
        annotation = NormalizedAnnotation()
        assert annotation.topics == []
        assert annotation.intent is None
        assert annotation.sentiment is None
        assert annotation.action_items == []
        assert annotation.risk_tags == []

    def test_to_dict(self) -> None:
        """Test NormalizedAnnotation serialization."""
        annotation = NormalizedAnnotation(
            topics=["billing"],
            intent="objection",
            sentiment="positive",
            action_items=[ActionItem(text="Fix issue")],
            risk_tags=["escalation"],
        )
        d = annotation.to_dict()
        assert d["topics"] == ["billing"]
        assert d["intent"] == "objection"
        assert d["sentiment"] == "positive"
        assert len(d["action_items"]) == 1
        assert d["action_items"][0]["text"] == "Fix issue"
        assert d["risk_tags"] == ["escalation"]

    def test_from_dict(self) -> None:
        """Test NormalizedAnnotation deserialization."""
        d = {
            "topics": ["contract"],
            "intent": "statement",
            "sentiment": "neutral",
            "action_items": [{"text": "Review terms", "confidence": 0.7}],
            "risk_tags": ["pricing"],
        }
        annotation = NormalizedAnnotation.from_dict(d)
        assert annotation.topics == ["contract"]
        assert annotation.intent == "statement"
        assert annotation.sentiment == "neutral"
        assert len(annotation.action_items) == 1
        assert annotation.action_items[0].text == "Review terms"
        assert annotation.risk_tags == ["pricing"]


class TestSemanticAnnotation:
    """Test SemanticAnnotation dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic SemanticAnnotation creation."""
        annotation = SemanticAnnotation(
            provider="anthropic",
            model="claude-3-5-sonnet",
            normalized=NormalizedAnnotation(topics=["pricing"]),
            confidence=0.9,
            latency_ms=150,
        )
        assert annotation.schema_version == SEMANTIC_SCHEMA_VERSION
        assert annotation.provider == "anthropic"
        assert annotation.model == "claude-3-5-sonnet"
        assert annotation.normalized.topics == ["pricing"]
        assert annotation.confidence == 0.9
        assert annotation.latency_ms == 150

    def test_defaults(self) -> None:
        """Test SemanticAnnotation default values."""
        annotation = SemanticAnnotation()
        assert annotation.schema_version == SEMANTIC_SCHEMA_VERSION
        assert annotation.provider == "local"
        assert annotation.model == "unknown"
        assert isinstance(annotation.normalized, NormalizedAnnotation)
        assert annotation.confidence == 1.0
        assert annotation.latency_ms == 0
        assert annotation.raw_model_output is None

    def test_to_dict(self) -> None:
        """Test SemanticAnnotation serialization."""
        annotation = SemanticAnnotation(
            provider="openai",
            model="gpt-4",
            normalized=NormalizedAnnotation(intent="question"),
            confidence=0.85,
            latency_ms=200,
            raw_model_output={"raw": "data"},
        )
        d = annotation.to_dict()
        assert d["schema_version"] == SEMANTIC_SCHEMA_VERSION
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4"
        assert d["normalized"]["intent"] == "question"
        assert d["confidence"] == 0.85
        assert d["latency_ms"] == 200
        assert d["raw_model_output"] == {"raw": "data"}

    def test_to_dict_no_raw_output(self) -> None:
        """Test that raw_model_output is omitted when None."""
        annotation = SemanticAnnotation(provider="local")
        d = annotation.to_dict()
        assert "raw_model_output" not in d

    def test_from_dict(self) -> None:
        """Test SemanticAnnotation deserialization."""
        d = {
            "schema_version": "0.1.0",
            "provider": "local",
            "model": "keyword-v1",
            "normalized": {"topics": ["pricing"], "risk_tags": ["churn_risk"]},
            "confidence": 1.0,
            "latency_ms": 5,
        }
        annotation = SemanticAnnotation.from_dict(d)
        assert annotation.schema_version == "0.1.0"
        assert annotation.provider == "local"
        assert annotation.model == "keyword-v1"
        assert annotation.normalized.topics == ["pricing"]
        assert annotation.normalized.risk_tags == ["churn_risk"]


class TestChunkContext:
    """Test ChunkContext dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic ChunkContext creation."""
        context = ChunkContext(
            speaker_id="spk_0",
            segment_ids=[0, 1, 2],
            start=10.5,
            end=25.0,
            previous_chunks=["Hello", "How are you?"],
            turn_id="turn_5",
            language="en",
        )
        assert context.speaker_id == "spk_0"
        assert context.segment_ids == [0, 1, 2]
        assert context.start == 10.5
        assert context.end == 25.0
        assert context.previous_chunks == ["Hello", "How are you?"]
        assert context.turn_id == "turn_5"
        assert context.language == "en"

    def test_defaults(self) -> None:
        """Test ChunkContext default values."""
        context = ChunkContext()
        assert context.speaker_id is None
        assert context.segment_ids == []
        assert context.start == 0.0
        assert context.end == 0.0
        assert context.previous_chunks == []
        assert context.turn_id is None
        assert context.language == "en"

    def test_to_dict(self) -> None:
        """Test ChunkContext serialization."""
        context = ChunkContext(
            speaker_id="spk_1",
            segment_ids=[3, 4],
            start=5.0,
            end=10.0,
            language="de",
        )
        d = context.to_dict()
        assert d["speaker_id"] == "spk_1"
        assert d["segment_ids"] == [3, 4]
        assert d["start"] == 5.0
        assert d["end"] == 10.0
        assert d["language"] == "de"

    def test_from_dict(self) -> None:
        """Test ChunkContext deserialization."""
        d = {
            "speaker_id": "spk_2",
            "segment_ids": [5],
            "start": 15.0,
            "end": 20.0,
            "previous_chunks": ["Previous text"],
            "turn_id": "turn_3",
            "language": "fr",
        }
        context = ChunkContext.from_dict(d)
        assert context.speaker_id == "spk_2"
        assert context.segment_ids == [5]
        assert context.start == 15.0
        assert context.end == 20.0
        assert context.previous_chunks == ["Previous text"]
        assert context.turn_id == "turn_3"
        assert context.language == "fr"

    def test_from_segments(self) -> None:
        """Test ChunkContext creation from segments."""
        segments = [
            Segment(id=0, start=0.0, end=2.0, text="Hello", speaker={"id": "spk_0"}),
            Segment(id=1, start=2.0, end=4.0, text="World", speaker={"id": "spk_0"}),
        ]
        context = ChunkContext.from_segments(
            segments,
            previous_chunks=["Previous"],
            turn_id="turn_1",
        )
        assert context.speaker_id == "spk_0"
        assert context.segment_ids == [0, 1]
        assert context.start == 0.0
        assert context.end == 4.0
        assert context.previous_chunks == ["Previous"]
        assert context.turn_id == "turn_1"

    def test_from_segments_empty(self) -> None:
        """Test ChunkContext creation from empty segments."""
        context = ChunkContext.from_segments([])
        assert context.speaker_id is None
        assert context.segment_ids == []

    def test_from_segments_string_speaker(self) -> None:
        """Test ChunkContext creation with string speaker."""
        segments = [
            Segment(id=0, start=0.0, end=1.0, text="Test", speaker="spk_1"),
        ]
        context = ChunkContext.from_segments(segments)
        assert context.speaker_id == "spk_1"


class TestProviderHealth:
    """Test ProviderHealth dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic ProviderHealth creation."""
        health = ProviderHealth(
            available=True,
            quota_remaining=100,
            latency_ms=5,
        )
        assert health.available is True
        assert health.quota_remaining == 100
        assert health.error is None
        assert health.latency_ms == 5

    def test_unavailable(self) -> None:
        """Test unavailable ProviderHealth."""
        health = ProviderHealth(
            available=False,
            error="API rate limit exceeded",
            latency_ms=0,
        )
        assert health.available is False
        assert health.error == "API rate limit exceeded"

    def test_to_dict(self) -> None:
        """Test ProviderHealth serialization."""
        health = ProviderHealth(available=True, quota_remaining=50)
        d = health.to_dict()
        assert d["available"] is True
        assert d["quota_remaining"] == 50
        assert "error" not in d

    def test_to_dict_with_error(self) -> None:
        """Test ProviderHealth serialization with error."""
        health = ProviderHealth(available=False, error="Connection failed")
        d = health.to_dict()
        assert d["available"] is False
        assert d["error"] == "Connection failed"


class TestLocalKeywordAdapter:
    """Test LocalKeywordAdapter implementation."""

    def test_annotate_chunk_basic(self) -> None:
        """Test basic chunk annotation."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext(speaker_id="spk_0", start=0.0, end=5.0)
        result = adapter.annotate_chunk("I'll send the report tomorrow.", context)

        assert isinstance(result, SemanticAnnotation)
        assert result.provider == "local"
        assert result.model == "keyword-v1"
        assert result.schema_version == SEMANTIC_SCHEMA_VERSION
        assert len(result.normalized.action_items) > 0
        assert result.latency_ms >= 0

    def test_annotate_chunk_with_risk_tags(self) -> None:
        """Test chunk annotation detects risk tags."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext(speaker_id="spk_0", start=0.0, end=5.0)
        result = adapter.annotate_chunk(
            "This is unacceptable! I want to cancel my subscription.", context
        )

        assert "escalation" in result.normalized.risk_tags
        assert "churn_risk" in result.normalized.risk_tags

    def test_annotate_chunk_with_pricing(self) -> None:
        """Test chunk annotation detects pricing topics."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext(speaker_id="spk_0", start=0.0, end=5.0)
        result = adapter.annotate_chunk("What's the price for the premium plan?", context)

        assert "pricing" in result.normalized.risk_tags

    def test_annotate_chunk_empty_text(self) -> None:
        """Test annotation with empty text."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext()
        result = adapter.annotate_chunk("", context)

        assert result.normalized.topics == []
        assert result.normalized.risk_tags == []
        assert result.normalized.action_items == []

    def test_annotate_chunk_records_latency(self) -> None:
        """Test that annotation records latency."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext()
        result = adapter.annotate_chunk("Test text", context)

        # Latency should be recorded (might be 0 for fast operations)
        assert result.latency_ms >= 0

    def test_health_check(self) -> None:
        """Test health check returns available."""
        adapter = LocalKeywordAdapter()
        health = adapter.health_check()

        assert health.available is True
        assert health.quota_remaining is None
        assert health.error is None

    def test_raw_model_output_included(self) -> None:
        """Test that raw model output is included for debugging."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext(speaker_id="spk_0", start=0.0, end=5.0)
        result = adapter.annotate_chunk("I want to escalate this issue.", context)

        assert result.raw_model_output is not None
        assert "keywords" in result.raw_model_output


class TestNoOpSemanticAdapter:
    """Test NoOpSemanticAdapter implementation."""

    def test_annotate_chunk_returns_empty(self) -> None:
        """Test that annotation returns empty result."""
        adapter = NoOpSemanticAdapter()
        context = ChunkContext()
        result = adapter.annotate_chunk("Any text here", context)

        assert result.provider == "noop"
        assert result.model == "none"
        assert result.confidence == 0.0
        assert result.normalized.topics == []
        assert result.normalized.risk_tags == []
        assert result.normalized.action_items == []

    def test_health_check(self) -> None:
        """Test health check returns available."""
        adapter = NoOpSemanticAdapter()
        health = adapter.health_check()

        assert health.available is True


class TestCreateAdapter:
    """Test create_adapter factory function."""

    def test_create_local_adapter(self) -> None:
        """Test creating local adapter."""
        adapter = create_adapter("local")
        assert isinstance(adapter, LocalKeywordAdapter)

    def test_create_noop_adapter(self) -> None:
        """Test creating noop adapter."""
        adapter = create_adapter("noop")
        assert isinstance(adapter, NoOpSemanticAdapter)

    def test_unknown_provider_raises(self) -> None:
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown semantic provider"):
            create_adapter("unknown-provider")


class TestProtocolCompliance:
    """Test SemanticAdapter protocol compliance."""

    def test_local_adapter_satisfies_protocol(self) -> None:
        """Verify LocalKeywordAdapter satisfies protocol."""
        adapter: SemanticAdapter = LocalKeywordAdapter()
        context = ChunkContext()
        result = adapter.annotate_chunk("Test", context)
        assert isinstance(result, SemanticAnnotation)

        health = adapter.health_check()
        assert isinstance(health, ProviderHealth)

    def test_noop_adapter_satisfies_protocol(self) -> None:
        """Verify NoOpSemanticAdapter satisfies protocol."""
        adapter: SemanticAdapter = NoOpSemanticAdapter()
        context = ChunkContext()
        result = adapter.annotate_chunk("Test", context)
        assert isinstance(result, SemanticAnnotation)

        health = adapter.health_check()
        assert isinstance(health, ProviderHealth)


class TestSemanticAnnotationIntegration:
    """Integration tests for semantic annotations with transcript."""

    def test_annotate_transcript_chunks(self) -> None:
        """Test annotating transcript segments as chunks."""
        adapter = LocalKeywordAdapter()

        # Create a transcript
        transcript = Transcript(
            file_name="call.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=3.0,
                    text="What's the price for your enterprise plan?",
                    speaker={"id": "spk_0"},
                ),
                Segment(
                    id=1,
                    start=3.0,
                    end=6.0,
                    text="I'll send you a quote after this call.",
                    speaker={"id": "spk_1"},
                ),
            ],
        )

        # Annotate each segment
        annotations = []
        for segment in transcript.segments:
            context = ChunkContext.from_segments([segment])
            annotation = adapter.annotate_chunk(segment.text, context)
            annotations.append(annotation)

        # Verify first segment (pricing question)
        assert "pricing" in annotations[0].normalized.risk_tags

        # Verify second segment (action item)
        assert len(annotations[1].normalized.action_items) > 0
