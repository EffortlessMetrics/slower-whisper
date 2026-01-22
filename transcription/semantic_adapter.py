"""Semantic adapter protocol and annotation schema (Track 3, #88).

This module defines the contract for semantic annotation providers (LLM-based
and rule-based), including standardized annotation schemas with versioning.

Key components:
- SemanticAnnotation: Versioned annotation dataclass with provenance
- NormalizedAnnotation: Provider-agnostic annotation structure
- ActionItem: Individual action item representation
- ChunkContext: Context for chunk-level annotation
- ProviderHealth: Health check result for providers
- SemanticAdapter: Protocol for all semantic providers
- LocalKeywordAdapter: Wrapper for KeywordSemanticAnnotator

Schema version: 0.1.0

See docs/SEMANTIC_BENCHMARK.md for evaluation methodology.
See ROADMAP.md Track 3 for design rationale.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from .models import Segment
    from .semantic import KeywordSemanticAnnotator

logger = logging.getLogger(__name__)

# Schema version for semantic annotations
SEMANTIC_SCHEMA_VERSION = "0.1.0"


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class ActionItem:
    """Individual action item detected in text.

    Attributes:
        text: The full text of the action item (the sentence/phrase).
        speaker_id: Speaker who committed to this action (if identifiable).
        segment_ids: List of segment IDs where this action was detected.
        pattern: The regex pattern that matched (for rule-based detection).
        confidence: Confidence score for this action item (0.0-1.0).
    """

    text: str
    speaker_id: str | None = None
    segment_ids: list[int] = field(default_factory=list)
    pattern: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        result: dict[str, Any] = {
            "text": self.text,
            "confidence": self.confidence,
        }
        if self.speaker_id is not None:
            result["speaker_id"] = self.speaker_id
        if self.segment_ids:
            result["segment_ids"] = list(self.segment_ids)
        if self.pattern is not None:
            result["pattern"] = self.pattern
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActionItem:
        """Deserialize from dict."""
        return cls(
            text=str(d.get("text", "")),
            speaker_id=d.get("speaker_id"),
            segment_ids=list(d.get("segment_ids", [])),
            pattern=d.get("pattern"),
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass(slots=True)
class NormalizedAnnotation:
    """Provider-agnostic normalized annotation structure.

    This dataclass standardizes semantic annotations across different
    providers (local keyword matching, OpenAI, Anthropic, etc.), ensuring
    consistent downstream processing regardless of source.

    Attributes:
        topics: List of topic labels (e.g., ["pricing", "contract_terms"]).
        intent: Detected intent of the utterance/chunk.
                Common values: "question", "objection", "statement",
                "request", "confirmation", "greeting", "closing".
        sentiment: Overall sentiment classification.
                   Values: "positive", "negative", "neutral", or None.
        action_items: List of detected action items/commitments.
        risk_tags: List of risk indicators (e.g., ["escalation", "churn_risk"]).
    """

    topics: list[str] = field(default_factory=list)
    intent: str | None = None
    sentiment: Literal["positive", "negative", "neutral"] | None = None
    action_items: list[ActionItem] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "topics": list(self.topics),
            "intent": self.intent,
            "sentiment": self.sentiment,
            "action_items": [a.to_dict() for a in self.action_items],
            "risk_tags": list(self.risk_tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormalizedAnnotation:
        """Deserialize from dict."""
        action_items_raw = d.get("action_items", [])
        action_items = [
            ActionItem.from_dict(a) if isinstance(a, dict) else a for a in action_items_raw
        ]
        return cls(
            topics=list(d.get("topics", [])),
            intent=d.get("intent"),
            sentiment=d.get("sentiment"),
            action_items=action_items,
            risk_tags=list(d.get("risk_tags", [])),
        )


@dataclass(slots=True)
class SemanticAnnotation:
    """Complete semantic annotation with versioning and provenance.

    This is the top-level annotation structure that includes both the
    normalized annotation and metadata about how/when it was produced.

    Attributes:
        schema_version: Version of the annotation schema (e.g., "0.1.0").
        provider: Provider identifier ("local", "anthropic", "openai").
        model: Model identifier (e.g., "keyword-v1", "claude-3-5-sonnet").
        normalized: The standardized annotation data.
        confidence: Overall confidence score for the annotation (0.0-1.0).
        latency_ms: Time taken to produce this annotation in milliseconds.
        raw_model_output: Optional raw output from the model (for debugging).
    """

    schema_version: str = SEMANTIC_SCHEMA_VERSION
    provider: str = "local"
    model: str = "unknown"
    normalized: NormalizedAnnotation = field(default_factory=NormalizedAnnotation)
    confidence: float = 1.0
    latency_ms: int = 0
    raw_model_output: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "model": self.model,
            "normalized": self.normalized.to_dict(),
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }
        if self.raw_model_output is not None:
            result["raw_model_output"] = self.raw_model_output
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SemanticAnnotation:
        """Deserialize from dict."""
        normalized_raw = d.get("normalized", {})
        normalized = (
            NormalizedAnnotation.from_dict(normalized_raw)
            if isinstance(normalized_raw, dict)
            else normalized_raw
        )
        return cls(
            schema_version=str(d.get("schema_version", SEMANTIC_SCHEMA_VERSION)),
            provider=str(d.get("provider", "local")),
            model=str(d.get("model", "unknown")),
            normalized=normalized,
            confidence=float(d.get("confidence", 1.0)),
            latency_ms=int(d.get("latency_ms", 0)),
            raw_model_output=d.get("raw_model_output"),
        )


@dataclass(slots=True)
class ChunkContext:
    """Context for chunk-level annotation.

    Provides surrounding context to help semantic providers make more
    accurate annotations. This includes speaker information, timing,
    and previous conversation context.

    Attributes:
        speaker_id: Speaker identifier for this chunk.
        segment_ids: List of segment IDs included in this chunk.
        start: Start time of the chunk in seconds.
        end: End time of the chunk in seconds.
        previous_chunks: List of previous chunk texts for context.
                        Ordered oldest to newest, limited by provider config.
        turn_id: Optional turn identifier if chunk is part of a turn.
        language: Language code for the chunk (e.g., "en").
    """

    speaker_id: str | None = None
    segment_ids: list[int] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    previous_chunks: list[str] = field(default_factory=list)
    turn_id: str | None = None
    language: str = "en"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "speaker_id": self.speaker_id,
            "segment_ids": list(self.segment_ids),
            "start": self.start,
            "end": self.end,
            "previous_chunks": list(self.previous_chunks),
            "turn_id": self.turn_id,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkContext:
        """Deserialize from dict."""
        return cls(
            speaker_id=d.get("speaker_id"),
            segment_ids=list(d.get("segment_ids", [])),
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            previous_chunks=list(d.get("previous_chunks", [])),
            turn_id=d.get("turn_id"),
            language=str(d.get("language", "en")),
        )

    @classmethod
    def from_segments(
        cls,
        segments: list[Segment],
        previous_chunks: list[str] | None = None,
        turn_id: str | None = None,
        language: str = "en",
    ) -> ChunkContext:
        """Create ChunkContext from a list of segments.

        Args:
            segments: List of Segment objects to create context from.
            previous_chunks: Optional list of previous chunk texts.
            turn_id: Optional turn identifier.
            language: Language code.

        Returns:
            ChunkContext populated from the segments.
        """
        if not segments:
            return cls(
                previous_chunks=previous_chunks or [],
                turn_id=turn_id,
                language=language,
            )

        # Extract speaker ID from first segment
        first_speaker = getattr(segments[0], "speaker", None)
        speaker_id = None
        if isinstance(first_speaker, dict):
            speaker_id = first_speaker.get("id")
        elif isinstance(first_speaker, str):
            speaker_id = first_speaker

        return cls(
            speaker_id=speaker_id,
            segment_ids=[seg.id for seg in segments],
            start=segments[0].start,
            end=segments[-1].end,
            previous_chunks=previous_chunks or [],
            turn_id=turn_id,
            language=language,
        )


@dataclass(slots=True)
class ProviderHealth:
    """Health check result for semantic providers.

    Used to verify provider availability and check quota/rate limits
    before making annotation requests.

    Attributes:
        available: Whether the provider is currently available.
        quota_remaining: Remaining API quota (requests/tokens/etc).
                        None if not applicable or unknown.
        error: Error message if provider is unavailable.
        latency_ms: Response time for health check in milliseconds.
    """

    available: bool = True
    quota_remaining: int | None = None
    error: str | None = None
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        result: dict[str, Any] = {
            "available": self.available,
            "latency_ms": self.latency_ms,
        }
        if self.quota_remaining is not None:
            result["quota_remaining"] = self.quota_remaining
        if self.error is not None:
            result["error"] = self.error
        return result


# -----------------------------------------------------------------------------
# Protocol
# -----------------------------------------------------------------------------


class SemanticAdapter(Protocol):
    """Protocol for semantic annotation providers.

    All semantic backends (local keyword matching, cloud LLMs, local LLMs)
    must implement this interface to be used in the annotation pipeline.

    The protocol provides two methods:
    - annotate_chunk: Annotate a single text chunk with context
    - health_check: Verify provider availability and quota

    Implementations should:
    - Handle errors gracefully (return low-confidence annotations on failure)
    - Respect rate limits and quotas
    - Include timing information in results
    - Support the standardized NormalizedAnnotation format

    Example implementation:
        >>> class MyAdapter:
        ...     def annotate_chunk(
        ...         self, text: str, context: ChunkContext
        ...     ) -> SemanticAnnotation:
        ...         # Perform annotation...
        ...         return SemanticAnnotation(
        ...             provider="my-provider",
        ...             model="my-model-v1",
        ...             normalized=NormalizedAnnotation(topics=["topic1"]),
        ...             confidence=0.9,
        ...             latency_ms=42,
        ...         )
        ...
        ...     def health_check(self) -> ProviderHealth:
        ...         return ProviderHealth(available=True)
    """

    def annotate_chunk(self, text: str, context: ChunkContext) -> SemanticAnnotation:
        """Annotate a single chunk of text with semantic information.

        Args:
            text: The text to annotate (typically 60-120 seconds of conversation).
            context: Surrounding context including speaker, timing, and history.

        Returns:
            SemanticAnnotation with normalized annotations and provenance.

        Note:
            Implementations should not raise exceptions. Instead, return
            a low-confidence annotation with error information in raw_model_output.
        """
        ...

    def health_check(self) -> ProviderHealth:
        """Check provider availability and quota status.

        Returns:
            ProviderHealth indicating availability and any limitations.

        Note:
            This method should be lightweight and fast. It should not
            make expensive API calls if possible.
        """
        ...


# -----------------------------------------------------------------------------
# Local Keyword Adapter
# -----------------------------------------------------------------------------


class LocalKeywordAdapter:
    """Adapter wrapping KeywordSemanticAnnotator for the SemanticAdapter protocol.

    This adapter provides a bridge from the existing rule-based
    KeywordSemanticAnnotator to the new SemanticAdapter protocol,
    enabling consistent usage across local and cloud providers.

    The adapter:
    - Converts text to a temporary Transcript for the annotator
    - Extracts keywords, risk tags, and actions from the result
    - Maps legacy format to NormalizedAnnotation
    - Tracks latency for benchmarking

    Attributes:
        _annotator: The underlying KeywordSemanticAnnotator instance.
    """

    def __init__(self, annotator: KeywordSemanticAnnotator | None = None) -> None:
        """Initialize the adapter.

        Args:
            annotator: Optional KeywordSemanticAnnotator instance.
                      If None, creates a default instance.
        """
        from .semantic import KeywordSemanticAnnotator

        self._annotator = annotator or KeywordSemanticAnnotator()

    def annotate_chunk(self, text: str, context: ChunkContext) -> SemanticAnnotation:
        """Annotate text using rule-based keyword matching.

        Args:
            text: The text to annotate.
            context: Chunk context (speaker, timing, etc.).

        Returns:
            SemanticAnnotation with extracted keywords, risks, and actions.
        """
        from .models import Segment, Transcript

        start_time = time.perf_counter_ns()

        # Build temporary transcript for the annotator
        segment = Segment(
            id=0,
            start=context.start,
            end=context.end,
            text=text,
            speaker={"id": context.speaker_id} if context.speaker_id else None,
        )
        transcript = Transcript(
            file_name="chunk",
            language=context.language,
            segments=[segment],
            annotations={},
        )

        # Run the annotator
        annotated = self._annotator.annotate(transcript)

        # Extract semantic results
        semantic = annotated.annotations.get("semantic", {}) if annotated.annotations else {}
        keywords = semantic.get("keywords", [])
        risk_tags = semantic.get("risk_tags", [])
        actions_raw = semantic.get("actions", [])

        # Convert legacy actions to ActionItem objects
        action_items = [
            ActionItem(
                text=str(a.get("text", "")),
                speaker_id=a.get("speaker_id"),
                segment_ids=a.get("segment_ids", []),
                pattern=a.get("pattern"),
                confidence=1.0,  # Rule-based = full confidence
            )
            for a in actions_raw
        ]

        # Compute latency
        end_time = time.perf_counter_ns()
        latency_ms = (end_time - start_time) // 1_000_000

        # Map keywords to topics (keywords are the "topics" in local mode)
        # Risk tags are semantic categories, not topics
        topics = [kw for kw in keywords if kw not in risk_tags]

        # Build normalized annotation
        normalized = NormalizedAnnotation(
            topics=topics,
            intent=None,  # Keyword annotator doesn't detect intent
            sentiment=None,  # Keyword annotator doesn't detect sentiment
            action_items=action_items,
            risk_tags=risk_tags,
        )

        return SemanticAnnotation(
            schema_version=SEMANTIC_SCHEMA_VERSION,
            provider="local",
            model="keyword-v1",
            normalized=normalized,
            confidence=1.0,  # Rule-based matching is deterministic
            latency_ms=latency_ms,
            raw_model_output=semantic,  # Include raw output for debugging
        )

    def health_check(self) -> ProviderHealth:
        """Check adapter health (always available for local adapter).

        Returns:
            ProviderHealth indicating the local adapter is available.
        """
        return ProviderHealth(
            available=True,
            quota_remaining=None,  # No quota for local
            error=None,
            latency_ms=0,
        )


# -----------------------------------------------------------------------------
# No-Op Adapter
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class NoOpSemanticAdapter:
    """No-op adapter that returns empty annotations.

    Useful as a placeholder or for testing when semantic annotation
    is disabled.
    """

    def annotate_chunk(self, text: str, context: ChunkContext) -> SemanticAnnotation:
        """Return an empty annotation.

        Args:
            text: Ignored.
            context: Ignored.

        Returns:
            Empty SemanticAnnotation with no-op provider.
        """
        return SemanticAnnotation(
            provider="noop",
            model="none",
            normalized=NormalizedAnnotation(),
            confidence=0.0,
            latency_ms=0,
        )

    def health_check(self) -> ProviderHealth:
        """Return healthy status.

        Returns:
            ProviderHealth indicating available.
        """
        return ProviderHealth(available=True)


# -----------------------------------------------------------------------------
# Factory function
# -----------------------------------------------------------------------------


def create_adapter(
    provider: str = "local",
    **kwargs: Any,
) -> SemanticAdapter:
    """Create a semantic adapter by provider name.

    Args:
        provider: Provider identifier. Supported values:
                 - "local": LocalKeywordAdapter (rule-based)
                 - "noop": NoOpSemanticAdapter (placeholder)
                 Future: "anthropic", "openai"
        **kwargs: Provider-specific configuration.

    Returns:
        A SemanticAdapter implementation.

    Raises:
        ValueError: If provider is unknown.

    Example:
        >>> adapter = create_adapter("local")
        >>> annotation = adapter.annotate_chunk("I'll send the report", context)
    """
    if provider == "local":
        return LocalKeywordAdapter(**kwargs)
    elif provider == "noop":
        return NoOpSemanticAdapter()
    else:
        raise ValueError(
            f"Unknown semantic provider: {provider}. "
            f"Supported: 'local', 'noop'. "
            f"Future: 'anthropic', 'openai'."
        )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # Schema version
    "SEMANTIC_SCHEMA_VERSION",
    # Data classes
    "ActionItem",
    "NormalizedAnnotation",
    "SemanticAnnotation",
    "ChunkContext",
    "ProviderHealth",
    # Protocol
    "SemanticAdapter",
    # Implementations
    "LocalKeywordAdapter",
    "NoOpSemanticAdapter",
    # Factory
    "create_adapter",
]
