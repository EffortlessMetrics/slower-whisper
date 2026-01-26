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
- LocalLLMSemanticAdapter: Local LLM-based semantic annotator (#89)
- OpenAISemanticAdapter: Cloud LLM adapter using OpenAI API (#90)
- AnthropicSemanticAdapter: Cloud LLM adapter using Anthropic API (#90)

Cloud adapter features (#90):
- Retry logic with exponential backoff for transient errors
- Configurable timeout handling (default: 30 seconds)
- Rate limit error detection and retry
- Error classification (rate_limit, timeout, transient, malformed, fatal)
- Graceful degradation on API failures

Schema version: 0.1.0

See docs/SEMANTIC_BENCHMARK.md for evaluation methodology.
See ROADMAP.md Track 3 for design rationale.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

if TYPE_CHECKING:
    from .models import Segment
    from .semantic import KeywordSemanticAnnotator

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _run_async_safely(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine safely from both sync and async contexts.

    This handles the case where asyncio.run() would fail because there's
    already a running event loop (e.g., when called from FastAPI endpoints
    or other async code).

    Args:
        coro: Coroutine to execute.

    Returns:
        Result of the coroutine.
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
        # Already in async context - run in a thread to avoid blocking
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop - use asyncio.run directly
        return asyncio.run(coro)


# -----------------------------------------------------------------------------
# Prompt Templates for Local LLM Semantic Extraction
# -----------------------------------------------------------------------------

TOPIC_EXTRACTION_PROMPT = """Extract topics from this conversation chunk.

CONTEXT (previous chunks):
{context}

CURRENT CHUNK:
Speaker: {speaker_id}
Timestamp: {start}s - {end}s
Text: {text}

Extract 1-3 main topics discussed. Focus on:
- Business topics (pricing, billing, features, support)
- Technical topics (bugs, integrations, setup)
- Relationship topics (satisfaction, complaints, requests)

Respond with ONLY valid JSON (no other text):
{{"topics": [
  {{"label": "topic_name", "confidence": 0.0-1.0, "evidence": "brief quote"}}
]}}
"""

RISK_DETECTION_PROMPT = """Detect risk signals in this conversation chunk.

CONTEXT (previous chunks):
{context}

CURRENT CHUNK:
Speaker: {speaker_id}
Timestamp: {start}s - {end}s
Text: {text}

Identify any of these risk signals:
- escalation: Customer requests manager/supervisor, uses threatening language
- churn_risk: Customer mentions leaving, canceling, competitor products
- compliance: Legal threats, regulatory mentions, recording concerns
- sentiment_negative: Strong negative emotion, frustration, anger
- pricing_objection: Budget concerns, price complaints, discount demands
- competitor_mention: References to competing products/services
- customer_frustration: Repeated issues, long wait times, unresolved problems
- agent_error: Mistakes, miscommunication, incorrect information

For each risk found, assess severity:
- critical: Immediate action required (legal threat, explicit churn intent)
- high: Urgent attention needed (escalation request, strong frustration)
- medium: Monitor closely (pricing concerns, mild frustration)
- low: Note for context (competitor mention, minor concerns)

Respond with ONLY valid JSON (no other text):
{{"risks": [
  {{"type": "risk_type", "severity": "level", "confidence": 0.0-1.0, "evidence": "brief quote"}}
]}}
"""

ACTION_EXTRACTION_PROMPT = """Extract action items from this conversation chunk.

CONTEXT (previous chunks):
{context}

CURRENT CHUNK:
Speaker: {speaker_id}
Timestamp: {start}s - {end}s
Text: {text}

Identify explicit commitments and tasks:
- Look for phrases like "I will", "I'll", "Let me", "We'll send", "I can"
- Note who made the commitment (speaker ID or role: agent/customer)
- Extract deadlines if mentioned (today, tomorrow, end of week, specific dates)
- Assess priority based on urgency signals

Only extract clear, actionable commitments. Do NOT include:
- Vague intentions ("maybe", "might", "could")
- Questions ("should I?", "do you want me to?")
- Past actions already completed

Respond with ONLY valid JSON (no other text):
{{"actions": [
  {{"description": "clear action description", "assignee": "speaker_id or null", "due": "deadline or null", "priority": "low|medium|high|null", "verbatim": "exact quote"}}
]}}
"""

COMBINED_EXTRACTION_PROMPT = """Analyze this conversation chunk and extract structured annotations.

CONTEXT (previous chunks):
{context}

CURRENT CHUNK:
Speaker: {speaker_id}
Timestamp: {start}s - {end}s
Text: {text}

Extract the following:

1. TOPICS: Main subjects discussed (1-3 topics max)
   - Use lowercase labels with underscores (e.g., "pricing", "technical_support")
   - Confidence: 0.9+ for explicit mentions, 0.7-0.9 for strong inference

2. RISKS: Any concerning signals
   - Types: escalation, churn_risk, compliance, sentiment_negative, pricing_objection, competitor_mention, customer_frustration, agent_error
   - Severity: low, medium, high, critical

3. ACTIONS: Explicit commitments and tasks
   - Only clear commitments (not suggestions or wishes)
   - Include assignee if identifiable

Respond with ONLY valid JSON (no other text):
{{
  "topics": [{{"label": "topic", "confidence": 0.0-1.0, "evidence": "quote"}}],
  "risks": [{{"type": "type", "severity": "level", "confidence": 0.0-1.0, "evidence": "quote"}}],
  "actions": [{{"description": "task", "assignee": "speaker|null", "due": "deadline|null", "priority": "level|null"}}]
}}

If no items found for a category, return empty array [].
"""

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
# Cloud LLM Semantic Extraction Prompt
# -----------------------------------------------------------------------------

CLOUD_SEMANTIC_EXTRACTION_SYSTEM_PROMPT = """\
You are a semantic analysis expert. Analyze the provided conversation text \
and extract semantic information.

Output a JSON object with exactly this structure:
{
  "topics": ["list of 1-5 topic labels like 'pricing', 'contract_terms'"],
  "intent": "question, objection, statement, request, confirmation, greeting, closing, or null",
  "sentiment": "positive, negative, neutral, or null",
  "action_items": [
    {
      "text": "the full text of the action/commitment",
      "speaker_id": "speaker ID if identifiable, or null",
      "confidence": 0.0-1.0
    }
  ],
  "risk_tags": ["list of risk indicators like 'escalation', 'churn_risk', 'compliance_concern'"]
}

Rules:
- Return ONLY valid JSON, no markdown or explanation
- topics: Identify key themes discussed (max 5)
- intent: The primary intent of the speaker(s)
- sentiment: Overall emotional tone
- action_items: Any commitments, promises, or tasks mentioned
- risk_tags: Potential concerns or risks identified
- If uncertain about a field, use null or empty list

Focus on accuracy over completeness."""

CLOUD_SEMANTIC_EXTRACTION_USER_TEMPLATE = """Analyze this conversation segment:

Speaker: {speaker_id}
Context: {context}

Text:
{text}

Return the JSON semantic analysis."""


# -----------------------------------------------------------------------------
# Cloud LLM Adapters (#90)
# -----------------------------------------------------------------------------


class CloudLLMSemanticAdapter:
    """Base class for cloud LLM semantic adapters.

    Provides common functionality for OpenAI and Anthropic adapters including:
    - Prompt building
    - JSON response parsing
    - Error handling with graceful degradation
    - Guardrails integration
    - Retry logic with exponential backoff
    - Timeout handling
    - Rate limit retry

    Subclasses must implement:
    - _get_provider_name() -> str
    - _get_model_name() -> str
    - _make_request_async(system: str, user: str) -> tuple[str, int]
    - _check_health_impl() -> ProviderHealth
    """

    # Default retry configuration
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_INITIAL_BACKOFF_MS = 1000
    DEFAULT_MAX_BACKOFF_MS = 32000
    DEFAULT_TIMEOUT_MS = 30000

    # Error patterns for classification
    RATE_LIMIT_PATTERNS = (
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "quota exceeded",
    )

    TIMEOUT_PATTERNS = (
        "timeout",
        "timed out",
        "time out",
        "deadline exceeded",
        "request timeout",
    )

    TRANSIENT_PATTERNS = (
        "connection",
        "network",
        "temporary",
        "unavailable",
        "server error",
        "500",
        "502",
        "503",
        "504",
    )

    def __init__(
        self,
        guardrails: Any | None = None,
        context_window: int = 3,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff_ms: int = DEFAULT_INITIAL_BACKOFF_MS,
        max_backoff_ms: int = DEFAULT_MAX_BACKOFF_MS,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        """Initialize the cloud adapter.

        Args:
            guardrails: Optional LLMGuardrails for rate limiting, cost tracking, PII.
                       If None, a default guardrails instance is created.
            context_window: Number of previous chunks to include for context.
            max_retries: Maximum number of retry attempts for transient errors.
            initial_backoff_ms: Initial backoff delay in milliseconds.
            max_backoff_ms: Maximum backoff delay in milliseconds.
            timeout_ms: Request timeout in milliseconds.
        """
        from .llm_guardrails import LLMGuardrails

        self._guardrails = guardrails or LLMGuardrails()
        self._context_window = context_window
        self._max_retries = max_retries
        self._initial_backoff_ms = initial_backoff_ms
        self._max_backoff_ms = max_backoff_ms
        self._timeout_ms = timeout_ms

    def _build_prompt(self, text: str, context: ChunkContext) -> tuple[str, str]:
        """Build system and user prompts for semantic extraction.

        Args:
            text: The text to annotate.
            context: Chunk context with speaker, timing, etc.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        # Build context string from previous chunks
        context_str = ""
        if context.previous_chunks:
            recent = context.previous_chunks[-self._context_window :]
            context_str = " | ".join(recent)
        else:
            context_str = "(start of conversation)"

        speaker = context.speaker_id or "unknown"

        user_prompt = CLOUD_SEMANTIC_EXTRACTION_USER_TEMPLATE.format(
            speaker_id=speaker,
            context=context_str,
            text=text,
        )

        return CLOUD_SEMANTIC_EXTRACTION_SYSTEM_PROMPT, user_prompt

    def _parse_response(self, response_text: str) -> NormalizedAnnotation:
        """Parse JSON response into NormalizedAnnotation.

        Args:
            response_text: Raw text response from LLM.

        Returns:
            NormalizedAnnotation parsed from response.

        Raises:
            ValueError: If response is not valid JSON or missing required fields.
        """
        # Strip markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)

        # Parse action items
        action_items = []
        for item in data.get("action_items", []):
            if isinstance(item, dict):
                action_items.append(
                    ActionItem(
                        text=str(item.get("text", "")),
                        speaker_id=item.get("speaker_id"),
                        confidence=float(item.get("confidence", 0.8)),
                    )
                )

        # Parse sentiment
        sentiment_raw = data.get("sentiment")
        sentiment: Literal["positive", "negative", "neutral"] | None = None
        if sentiment_raw in ("positive", "negative", "neutral"):
            sentiment = sentiment_raw

        return NormalizedAnnotation(
            topics=list(data.get("topics", [])),
            intent=data.get("intent"),
            sentiment=sentiment,
            action_items=action_items,
            risk_tags=list(data.get("risk_tags", [])),
        )

    def _classify_error(self, error: Exception) -> str:
        """Classify an error for retry decision.

        Args:
            error: The exception to classify.

        Returns:
            Error type: "rate_limit", "timeout", "transient", "malformed", or "fatal".
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check for rate limit errors
        if any(pattern in error_str for pattern in self.RATE_LIMIT_PATTERNS):
            return "rate_limit"

        # Check for timeout errors
        if any(pattern in error_str for pattern in self.TIMEOUT_PATTERNS):
            return "timeout"
        if "asyncio.timeout" in error_type or "TimeoutError" in type(error).__name__:
            return "timeout"

        # Check for transient/connection errors
        if any(pattern in error_str for pattern in self.TRANSIENT_PATTERNS):
            return "transient"

        # Check for JSON parsing errors (malformed response)
        if isinstance(error, json.JSONDecodeError):
            return "malformed"
        if "json" in error_str and ("decode" in error_str or "parse" in error_str):
            return "malformed"

        # Default to fatal (non-recoverable)
        return "fatal"

    def _should_retry(self, error_type: str, attempt: int) -> bool:
        """Determine if a request should be retried.

        Args:
            error_type: The classified error type.
            attempt: Current attempt number (0-indexed).

        Returns:
            True if the request should be retried.
        """
        if attempt >= self._max_retries:
            return False

        # Retry on rate limit, timeout, and transient errors
        return error_type in ("rate_limit", "timeout", "transient")

    def _calculate_backoff(self, attempt: int, error_type: str) -> float:
        """Calculate backoff delay with jitter.

        Args:
            attempt: Current attempt number (0-indexed).
            error_type: The classified error type.

        Returns:
            Backoff delay in seconds.
        """
        # Exponential backoff: initial * 2^attempt
        base_delay_ms = self._initial_backoff_ms * (2**attempt)

        # Rate limit errors get longer initial delay
        if error_type == "rate_limit":
            base_delay_ms = max(base_delay_ms, 5000)  # At least 5 seconds

        # Cap at max backoff
        delay_ms = min(base_delay_ms, self._max_backoff_ms)

        # Add jitter (0-25% of delay)
        jitter = random.uniform(0, 0.25) * delay_ms
        delay_ms += jitter

        return float(delay_ms / 1000.0)  # Return seconds

    def _get_provider_name(self) -> str:
        """Return provider identifier (e.g., 'openai', 'anthropic')."""
        raise NotImplementedError

    def _get_model_name(self) -> str:
        """Return model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet')."""
        raise NotImplementedError

    async def _make_request_async(self, system: str, user: str) -> tuple[str, int]:
        """Make async LLM request.

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            Tuple of (response_text, latency_ms).

        Raises:
            Exception: On API failure.
        """
        raise NotImplementedError

    def _make_request_sync(self, system: str, user: str) -> tuple[str, int]:
        """Make synchronous LLM request (deprecated, use _make_request_with_retry).

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            Tuple of (response_text, latency_ms).

        Raises:
            Exception: On API failure.
        """
        raise NotImplementedError

    def _check_health_impl(self) -> ProviderHealth:
        """Check provider health (implementation).

        Returns:
            ProviderHealth indicating availability.
        """
        raise NotImplementedError

    async def _make_request_with_retry(
        self, system: str, user: str
    ) -> tuple[str, int, list[dict[str, Any]]]:
        """Make LLM request with retry logic and timeout handling.

        Implements exponential backoff with jitter for rate limit,
        timeout, and transient errors.

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            Tuple of (response_text, latency_ms, retry_history).
            retry_history contains details of any retry attempts.

        Raises:
            Exception: On non-recoverable failure after all retries exhausted.
        """
        retry_history: list[dict[str, Any]] = []
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                # Apply timeout to the request
                timeout_seconds = self._timeout_ms / 1000.0
                response_text, latency_ms = await asyncio.wait_for(
                    self._make_request_async(system, user),
                    timeout=timeout_seconds,
                )
                return response_text, latency_ms, retry_history

            except TimeoutError as e:
                last_error = e
                error_type = "timeout"
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)

            # Record retry attempt
            retry_info = {
                "attempt": attempt,
                "error_type": error_type,
                "error_message": str(last_error),
            }

            # Decide whether to retry
            if not self._should_retry(error_type, attempt):
                retry_info["action"] = "give_up"
                retry_history.append(retry_info)
                logger.warning(
                    f"{self._get_provider_name()} request failed (attempt {attempt + 1}): "
                    f"{error_type} - {last_error}"
                )
                raise last_error

            # Calculate and apply backoff
            backoff_seconds = self._calculate_backoff(attempt, error_type)
            retry_info["backoff_seconds"] = backoff_seconds
            retry_info["action"] = "retry"
            retry_history.append(retry_info)

            logger.info(
                f"{self._get_provider_name()} request failed (attempt {attempt + 1}), "
                f"retrying in {backoff_seconds:.2f}s: {error_type} - {last_error}"
            )

            await asyncio.sleep(backoff_seconds)

        # Should not reach here, but just in case
        raise last_error if last_error else RuntimeError("Unexpected retry loop exit")

    def annotate_chunk(self, text: str, context: ChunkContext) -> SemanticAnnotation:
        """Annotate text using cloud LLM with retry logic.

        Args:
            text: The text to annotate.
            context: Chunk context (speaker, timing, etc.).

        Returns:
            SemanticAnnotation with extracted semantics.

        Note:
            Does not raise exceptions. On failure, returns low-confidence
            annotation with error in raw_model_output.
        """
        start_time = time.perf_counter_ns()

        try:
            # Build prompts
            system_prompt, user_prompt = self._build_prompt(text, context)

            # Make request with retry
            async def _annotate() -> tuple[str, list[dict[str, Any]]]:
                response_text, _, retry_history = await self._make_request_with_retry(
                    system_prompt, user_prompt
                )
                return response_text, retry_history

            response_text, retry_history = _run_async_safely(_annotate())

            # Parse response
            normalized = self._parse_response(response_text)

            # Compute latency
            end_time = time.perf_counter_ns()
            latency_ms = (end_time - start_time) // 1_000_000

            raw_output: dict[str, Any] = {"response": response_text}
            if retry_history:
                raw_output["retry_history"] = retry_history

            return SemanticAnnotation(
                schema_version=SEMANTIC_SCHEMA_VERSION,
                provider=self._get_provider_name(),
                model=self._get_model_name(),
                normalized=normalized,
                confidence=0.85,  # Cloud LLMs have good but not perfect accuracy
                latency_ms=latency_ms,
                raw_model_output=raw_output,
            )

        except Exception as e:
            # Classify the error for better reporting
            error_type = self._classify_error(e)

            # Log error and return low-confidence annotation
            logger.warning(
                f"{self._get_provider_name()} annotation failed ({error_type}): {e}",
                exc_info=True,
            )

            end_time = time.perf_counter_ns()
            latency_ms = (end_time - start_time) // 1_000_000

            return SemanticAnnotation(
                schema_version=SEMANTIC_SCHEMA_VERSION,
                provider=self._get_provider_name(),
                model=self._get_model_name(),
                normalized=NormalizedAnnotation(),  # Empty annotation
                confidence=0.0,  # No confidence on error
                latency_ms=latency_ms,
                raw_model_output={
                    "error": str(e),
                    "error_type": error_type,
                },
            )

    def health_check(self) -> ProviderHealth:
        """Check provider availability and quota.

        Returns:
            ProviderHealth with availability and remaining quota.
        """
        start_time = time.perf_counter_ns()

        try:
            result = self._check_health_impl()
            end_time = time.perf_counter_ns()
            result.latency_ms = (end_time - start_time) // 1_000_000
            return result
        except Exception as e:
            end_time = time.perf_counter_ns()
            latency_ms = (end_time - start_time) // 1_000_000
            return ProviderHealth(
                available=False,
                error=str(e),
                latency_ms=latency_ms,
            )


class OpenAISemanticAdapter(CloudLLMSemanticAdapter):
    """Semantic adapter using OpenAI API (GPT-4o, GPT-4, etc.).

    This adapter:
    - Wraps OpenAIProvider from llm_client
    - Applies guardrails (rate limit, cost budget, PII detection)
    - Builds prompts for semantic extraction
    - Parses JSON responses into SemanticAnnotation
    - Implements retry logic with exponential backoff
    - Handles timeout, rate limit, and transient errors

    Example:
        >>> adapter = OpenAISemanticAdapter(
        ...     api_key="sk-...",  # Or use OPENAI_API_KEY env
        ...     model="gpt-4o",
        ... )
        >>> annotation = adapter.annotate_chunk(text, context)
        >>> health = adapter.health_check()
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        guardrails: Any | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        context_window: int = 3,
        max_retries: int = CloudLLMSemanticAdapter.DEFAULT_MAX_RETRIES,
        initial_backoff_ms: int = CloudLLMSemanticAdapter.DEFAULT_INITIAL_BACKOFF_MS,
        max_backoff_ms: int = CloudLLMSemanticAdapter.DEFAULT_MAX_BACKOFF_MS,
        timeout_ms: int = CloudLLMSemanticAdapter.DEFAULT_TIMEOUT_MS,
    ) -> None:
        """Initialize OpenAI semantic adapter.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use (default: gpt-4o).
            base_url: Optional custom API base URL (for Azure, proxies, etc.).
            guardrails: Optional LLMGuardrails for rate limiting, cost tracking, PII.
            temperature: Sampling temperature (default: 0.3 for consistency).
            max_tokens: Maximum tokens in response (default: 1024).
            context_window: Number of previous chunks to include for context.
            max_retries: Maximum number of retry attempts for transient errors.
            initial_backoff_ms: Initial backoff delay in milliseconds.
            max_backoff_ms: Maximum backoff delay in milliseconds.
            timeout_ms: Request timeout in milliseconds.
        """
        import os

        from .historian.llm_client import LLMConfig, OpenAIProvider
        from .llm_guardrails import GuardedLLMProvider

        super().__init__(
            guardrails=guardrails,
            context_window=context_window,
            max_retries=max_retries,
            initial_backoff_ms=initial_backoff_ms,
            max_backoff_ms=max_backoff_ms,
            timeout_ms=timeout_ms,
        )

        # Resolve API key from parameter or environment
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

        config = LLMConfig(
            provider="openai",
            model=model,
            api_key=self._api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._provider = OpenAIProvider(config)
        self._guarded_provider = GuardedLLMProvider(self._provider, self._guardrails)
        self._model = model
        self._base_url = base_url

    def _get_provider_name(self) -> str:
        """Return 'openai' provider identifier."""
        return "openai"

    def _get_model_name(self) -> str:
        """Return configured model name."""
        return self._model

    async def _make_request_async(self, system: str, user: str) -> tuple[str, int]:
        """Make async request via guarded provider.

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            Tuple of (response_text, latency_ms).
        """
        response = await self._guarded_provider.complete(system, user)
        return response.text, response.duration_ms

    def _make_request_sync(self, system: str, user: str) -> tuple[str, int]:
        """Make synchronous request via guarded provider (legacy).

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            Tuple of (response_text, latency_ms).
        """
        return _run_async_safely(self._make_request_async(system, user))

    def _check_health_impl(self) -> ProviderHealth:
        """Check OpenAI API availability.

        Makes a minimal request to verify connectivity and API key validity.

        Returns:
            ProviderHealth with availability status.
        """
        # Check API key is available (already resolved in __init__)
        if not self._api_key:
            return ProviderHealth(
                available=False,
                error="OpenAI API key not configured",
            )

        # Check remaining budget
        stats = self._guardrails.stats
        budget = self._guardrails.cost_budget_usd
        remaining = max(0, budget - stats.total_cost_usd)

        # Estimate remaining requests based on cost per request
        # Approximate cost per semantic extraction: ~$0.002 for GPT-4o
        estimated_cost_per_request = 0.002
        quota_remaining = int(remaining / estimated_cost_per_request) if remaining > 0 else 0

        # Try a lightweight API check (list models endpoint is fast)
        try:
            import openai

            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            client = openai.OpenAI(**client_kwargs)
            # Just check if we can list models - lightweight health check
            list(client.models.list())

            return ProviderHealth(
                available=True,
                quota_remaining=quota_remaining,
            )
        except ImportError:
            return ProviderHealth(
                available=False,
                error="openai package not installed",
            )
        except Exception as e:
            return ProviderHealth(
                available=False,
                error=f"OpenAI API error: {e}",
            )


class AnthropicSemanticAdapter(CloudLLMSemanticAdapter):
    """Semantic adapter using Anthropic API (Claude models).

    This adapter:
    - Wraps AnthropicProvider from llm_client
    - Applies guardrails (rate limit, cost budget, PII detection)
    - Builds prompts for semantic extraction
    - Parses JSON responses into SemanticAnnotation

    Example:
        >>> adapter = AnthropicSemanticAdapter(
        ...     api_key="sk-ant-...",  # Or use ANTHROPIC_API_KEY env
        ...     model="claude-sonnet-4-20250514",
        ... )
        >>> annotation = adapter.annotate_chunk(text, context)
        >>> health = adapter.health_check()
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        guardrails: Any | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        context_window: int = 3,
        max_retries: int = CloudLLMSemanticAdapter.DEFAULT_MAX_RETRIES,
        initial_backoff_ms: int = CloudLLMSemanticAdapter.DEFAULT_INITIAL_BACKOFF_MS,
        max_backoff_ms: int = CloudLLMSemanticAdapter.DEFAULT_MAX_BACKOFF_MS,
        timeout_ms: int = CloudLLMSemanticAdapter.DEFAULT_TIMEOUT_MS,
    ) -> None:
        """Initialize Anthropic semantic adapter.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Model to use (default: claude-sonnet-4-20250514).
            guardrails: Optional LLMGuardrails for rate limiting, cost tracking, PII.
            temperature: Sampling temperature (default: 0.3 for consistency).
            max_tokens: Maximum tokens in response (default: 1024).
            context_window: Number of previous chunks to include for context.
            max_retries: Maximum number of retry attempts for transient errors.
            initial_backoff_ms: Initial backoff delay in milliseconds.
            max_backoff_ms: Maximum backoff delay in milliseconds.
            timeout_ms: Request timeout in milliseconds.
        """
        import os

        from .historian.llm_client import AnthropicProvider, LLMConfig
        from .llm_guardrails import GuardedLLMProvider

        super().__init__(
            guardrails=guardrails,
            context_window=context_window,
            max_retries=max_retries,
            initial_backoff_ms=initial_backoff_ms,
            max_backoff_ms=max_backoff_ms,
            timeout_ms=timeout_ms,
        )

        # Resolve API key from parameter or environment
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        config = LLMConfig(
            provider="anthropic",
            model=model,
            api_key=self._api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._provider = AnthropicProvider(config)
        self._guarded_provider = GuardedLLMProvider(self._provider, self._guardrails)
        self._model = model

    def _get_provider_name(self) -> str:
        """Return 'anthropic' provider identifier."""
        return "anthropic"

    def _get_model_name(self) -> str:
        """Return configured model name."""
        return self._model

    async def _make_request_async(self, system: str, user: str) -> tuple[str, int]:
        """Make async request via guarded provider.

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            Tuple of (response_text, latency_ms).
        """
        response = await self._guarded_provider.complete(system, user)
        return response.text, response.duration_ms

    def _make_request_sync(self, system: str, user: str) -> tuple[str, int]:
        """Make synchronous request via guarded provider (legacy).

        Args:
            system: System prompt.
            user: User prompt.

        Returns:
            Tuple of (response_text, latency_ms).
        """
        return _run_async_safely(self._make_request_async(system, user))

    def _check_health_impl(self) -> ProviderHealth:
        """Check Anthropic API availability.

        Makes a minimal request to verify connectivity and API key validity.

        Returns:
            ProviderHealth with availability status.
        """
        # Check API key is available (already resolved in __init__)
        if not self._api_key:
            return ProviderHealth(
                available=False,
                error="Anthropic API key not configured",
            )

        # Check remaining budget
        stats = self._guardrails.stats
        budget = self._guardrails.cost_budget_usd
        remaining = max(0, budget - stats.total_cost_usd)

        # Estimate remaining requests based on cost per request
        # Approximate cost per semantic extraction: ~$0.003 for Claude Sonnet
        estimated_cost_per_request = 0.003
        quota_remaining = int(remaining / estimated_cost_per_request) if remaining > 0 else 0

        # Try a lightweight API check using the count_tokens endpoint
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._api_key)
            # Count tokens is a lightweight way to verify API access
            client.messages.count_tokens(
                model=self._model,
                messages=[{"role": "user", "content": "test"}],
            )

            return ProviderHealth(
                available=True,
                quota_remaining=quota_remaining,
            )
        except ImportError:
            return ProviderHealth(
                available=False,
                error="anthropic package not installed",
            )
        except Exception as e:
            return ProviderHealth(
                available=False,
                error=f"Anthropic API error: {e}",
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
# Local LLM Semantic Adapter (#89)
# -----------------------------------------------------------------------------


class LocalLLMSemanticAdapter:
    """Local LLM-based semantic annotator implementing SemanticAdapter protocol.

    This adapter uses local models (via transformers) for semantic annotation,
    providing topic extraction, risk detection, and action item extraction
    without requiring cloud API access.

    Features:
    - Optional dependency handling (torch/transformers not required at import)
    - Lazy model loading (only loads model on first use)
    - Structured JSON output parsing with validation
    - Graceful error handling for malformed LLM responses
    - Configurable extraction modes (topics, risks, actions, or combined)
    - Deterministic output normalization with validation

    The adapter uses the LocalLLMProvider from local_llm_provider.py for
    actual model inference. If torch/transformers are not installed,
    health_check() returns unavailable and annotate_chunk() returns
    an empty annotation with confidence 0.

    Attributes:
        model: Model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct").
        temperature: LLM temperature for generation (default: 0.1).
        max_tokens: Maximum tokens for LLM response (default: 1024).
        extraction_mode: What to extract ("combined", "topics", "risks", "actions").

    Example:
        >>> adapter = LocalLLMSemanticAdapter(model="Qwen/Qwen2.5-7B-Instruct")
        >>> context = ChunkContext(speaker_id="agent", start=0.0, end=30.0)
        >>> annotation = adapter.annotate_chunk("I'll send the report tomorrow", context)
        >>> print(annotation.normalized.action_items)
        [ActionItem(text='Send the report', due='tomorrow', ...)]
    """

    # Default model if none specified
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # Valid risk types for validation
    VALID_RISK_TYPES = frozenset(
        {
            "escalation",
            "churn_risk",
            "compliance",
            "sentiment_negative",
            "legal",
            "pricing_objection",
            "competitor_mention",
            "customer_frustration",
            "agent_error",
            "other",
        }
    )

    # Valid severity levels for validation
    VALID_SEVERITIES = frozenset({"low", "medium", "high", "critical"})

    # Valid priority levels for validation
    VALID_PRIORITIES = frozenset({"low", "medium", "high"})

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        extraction_mode: Literal["combined", "topics", "risks", "actions"] = "combined",
    ) -> None:
        """Initialize the local LLM semantic adapter.

        Args:
            model: Model identifier for local LLM. If None, uses DEFAULT_MODEL.
            temperature: LLM temperature for generation (lower = more deterministic).
            max_tokens: Maximum tokens for LLM response.
            extraction_mode: What to extract. Options:
                - "combined": Extract all (topics, risks, actions) in one call
                - "topics": Only extract topics
                - "risks": Only extract risks
                - "actions": Only extract action items
        """
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extraction_mode = extraction_mode
        self._provider: Any = None
        self._provider_available: bool | None = None

    def _lazy_load_provider(self) -> bool:
        """Lazy load the LocalLLMProvider on first use.

        Returns:
            True if provider is available and loaded, False otherwise.
        """
        if self._provider_available is not None:
            return self._provider_available

        try:
            # Use dedicated local_llm_provider module with proper optional dep handling
            from .local_llm_provider import LocalLLMProvider, is_available

            if not is_available():
                logger.warning(
                    "LocalLLMSemanticAdapter: torch/transformers not installed. "
                    "Install with: pip install 'slower-whisper[emotion]'"
                )
                self._provider_available = False
                return False

            self._provider = LocalLLMProvider(
                model_name=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self._provider_available = True
            logger.info(f"LocalLLMSemanticAdapter: Provider ready with model {self.model}")
            return True
        except ImportError as e:
            logger.warning(f"LocalLLMSemanticAdapter: Failed to import LLM provider: {e}")
            self._provider_available = False
            return False
        except Exception as e:
            logger.warning(f"LocalLLMSemanticAdapter: Failed to initialize provider: {e}")
            self._provider_available = False
            return False

    def _build_prompt(self, text: str, context: ChunkContext) -> str:
        """Build the appropriate prompt based on extraction mode.

        Args:
            text: The text to annotate.
            context: Chunk context information.

        Returns:
            Formatted prompt string.
        """
        # Build context string from previous chunks
        context_str = (
            "\n".join(context.previous_chunks[-3:])  # Last 3 chunks
            if context.previous_chunks
            else "No prior context"
        )

        # Common format kwargs
        format_kwargs = {
            "context": context_str,
            "speaker_id": context.speaker_id or "unknown",
            "start": f"{context.start:.1f}",
            "end": f"{context.end:.1f}",
            "text": text,
        }

        # Select prompt based on extraction mode
        if self.extraction_mode == "topics":
            return TOPIC_EXTRACTION_PROMPT.format(**format_kwargs)
        elif self.extraction_mode == "risks":
            return RISK_DETECTION_PROMPT.format(**format_kwargs)
        elif self.extraction_mode == "actions":
            return ACTION_EXTRACTION_PROMPT.format(**format_kwargs)
        else:  # combined (default)
            return COMBINED_EXTRACTION_PROMPT.format(**format_kwargs)

    def _extract_json_from_response(self, response: str) -> str | None:
        """Extract JSON from LLM response, handling markdown code blocks.

        Args:
            response: Raw LLM response string.

        Returns:
            Extracted JSON string, or None if no JSON found.
        """
        response = response.strip()

        # Try to extract from markdown code block
        if "```" in response:
            # Pattern to match JSON in code blocks
            code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
            matches = re.findall(code_block_pattern, response)
            if matches:
                return str(matches[0]).strip()

        # Try to find JSON object directly
        # Look for outermost { } pair
        brace_start = response.find("{")
        if brace_start != -1:
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(response[brace_start:], start=brace_start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return response[brace_start : i + 1]

        return None

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse and validate LLM JSON response.

        Args:
            response: Raw LLM response string.

        Returns:
            Parsed and validated dictionary with topics, risks, and actions.
            Returns empty structure if parsing fails.
        """
        # Default empty structure
        empty_result: dict[str, Any] = {"topics": [], "risks": [], "actions": []}

        # Extract JSON from response
        json_str = self._extract_json_from_response(response)
        if not json_str:
            logger.warning("LocalLLMSemanticAdapter: No JSON found in response")
            return empty_result

        # Parse JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"LocalLLMSemanticAdapter: JSON parse error: {e}")
            return empty_result

        if not isinstance(parsed, dict):
            logger.warning("LocalLLMSemanticAdapter: Response is not a JSON object")
            return empty_result

        # Validate and normalize topics
        topics = []
        for topic in parsed.get("topics", []):
            if isinstance(topic, dict) and "label" in topic:
                validated_topic = {
                    "label": str(topic["label"]).lower().replace(" ", "_"),
                    "confidence": self._clamp_confidence(topic.get("confidence", 0.5)),
                    "evidence": str(topic.get("evidence", ""))[:100],  # Limit length
                }
                topics.append(validated_topic)

        # Validate and normalize risks
        risks = []
        for risk in parsed.get("risks", []):
            if isinstance(risk, dict) and "type" in risk:
                risk_type = str(risk["type"]).lower()
                if risk_type not in self.VALID_RISK_TYPES:
                    risk_type = "other"

                severity = str(risk.get("severity", "medium")).lower()
                if severity not in self.VALID_SEVERITIES:
                    severity = "medium"

                validated_risk = {
                    "type": risk_type,
                    "severity": severity,
                    "confidence": self._clamp_confidence(risk.get("confidence", 0.5)),
                    "evidence": str(risk.get("evidence", ""))[:100],
                }
                risks.append(validated_risk)

        # Validate and normalize actions
        actions = []
        for action in parsed.get("actions", []):
            if isinstance(action, dict) and "description" in action:
                priority = action.get("priority")
                if priority and str(priority).lower() not in self.VALID_PRIORITIES:
                    priority = None
                elif priority:
                    priority = str(priority).lower()

                validated_action = {
                    "description": str(action["description"]),
                    "assignee": action.get("assignee"),
                    "due": action.get("due"),
                    "priority": priority,
                    "verbatim": str(action.get("verbatim", ""))[:200],
                    "confidence": self._clamp_confidence(action.get("confidence", 0.8)),
                }
                actions.append(validated_action)

        return {"topics": topics, "risks": risks, "actions": actions}

    def _clamp_confidence(self, value: Any) -> float:
        """Clamp confidence value to valid range [0.0, 1.0].

        Args:
            value: Value to clamp.

        Returns:
            Float clamped to [0.0, 1.0].
        """
        try:
            conf = float(value)
            return max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            return 0.5

    def _build_normalized_annotation(
        self, parsed: dict[str, Any], context: ChunkContext
    ) -> NormalizedAnnotation:
        """Build NormalizedAnnotation from parsed LLM response.

        Args:
            parsed: Parsed and validated LLM response.
            context: Chunk context information.

        Returns:
            NormalizedAnnotation with topics, risks, and action items.
        """
        # Extract topics as simple labels
        topics = [t["label"] for t in parsed.get("topics", [])]

        # Extract risk tags (just the type strings)
        risk_tags = [r["type"] for r in parsed.get("risks", [])]

        # Build ActionItem objects
        action_items = []
        for action in parsed.get("actions", []):
            action_item = ActionItem(
                text=action["description"],
                speaker_id=action.get("assignee"),
                segment_ids=list(context.segment_ids) if context.segment_ids else [],
                pattern=None,  # LLM-based, no regex pattern
                confidence=action.get("confidence", 0.8),
            )
            action_items.append(action_item)

        # Detect intent from context (simple heuristic)
        intent = self._detect_intent(parsed)

        # Detect sentiment from risks
        sentiment = self._detect_sentiment(parsed)

        return NormalizedAnnotation(
            topics=topics,
            intent=intent,
            sentiment=sentiment,
            action_items=action_items,
            risk_tags=risk_tags,
        )

    def _detect_intent(self, parsed: dict[str, Any]) -> str | None:
        """Detect intent from parsed response.

        Args:
            parsed: Parsed LLM response.

        Returns:
            Intent string or None.
        """
        # Simple heuristic based on topics and risks
        risks = parsed.get("risks", [])
        for risk in risks:
            if risk.get("type") == "escalation":
                return "request"
            if risk.get("type") == "pricing_objection":
                return "objection"

        actions = parsed.get("actions", [])
        if actions:
            return "statement"  # Commitment/statement

        return None

    def _detect_sentiment(
        self, parsed: dict[str, Any]
    ) -> Literal["positive", "negative", "neutral"] | None:
        """Detect sentiment from parsed response.

        Args:
            parsed: Parsed LLM response.

        Returns:
            Sentiment classification or None.
        """
        risks = parsed.get("risks", [])
        negative_indicators = {
            "escalation",
            "churn_risk",
            "sentiment_negative",
            "customer_frustration",
        }

        for risk in risks:
            if risk.get("type") in negative_indicators:
                severity = risk.get("severity", "medium")
                if severity in ("high", "critical"):
                    return "negative"

        # Default to None (not enough signal)
        return None

    def _compute_overall_confidence(self, parsed: dict[str, Any]) -> float:
        """Compute overall confidence score for the annotation.

        Args:
            parsed: Parsed LLM response.

        Returns:
            Overall confidence score (0.0-1.0).
        """
        confidences = []

        for topic in parsed.get("topics", []):
            confidences.append(topic.get("confidence", 0.5))

        for risk in parsed.get("risks", []):
            confidences.append(risk.get("confidence", 0.5))

        for action in parsed.get("actions", []):
            confidences.append(action.get("confidence", 0.8))

        if not confidences:
            return 0.5  # Default confidence when no extractions

        return float(sum(confidences) / len(confidences))

    def annotate_chunk(self, text: str, context: ChunkContext) -> SemanticAnnotation:
        """Annotate a single chunk of text with semantic information.

        Uses local LLM to extract topics, risks, and action items from the text.
        Handles errors gracefully, returning low-confidence annotations on failure.

        Args:
            text: The text to annotate (typically 60-120 seconds of conversation).
            context: Surrounding context including speaker, timing, and history.

        Returns:
            SemanticAnnotation with normalized annotations and provenance.
        """
        start_time = time.perf_counter_ns()

        # Check if provider is available
        if not self._lazy_load_provider():
            logger.warning(
                "LocalLLMSemanticAdapter: Provider not available, returning empty annotation"
            )
            end_time = time.perf_counter_ns()
            return SemanticAnnotation(
                provider="local-llm",
                model=self.model,
                normalized=NormalizedAnnotation(),
                confidence=0.0,
                latency_ms=(end_time - start_time) // 1_000_000,
                raw_model_output={"error": "Provider not available"},
            )

        # Build prompt
        prompt = self._build_prompt(text, context)

        # System prompt for structured extraction
        system_prompt = (
            "You are a semantic annotation assistant for conversation transcripts. "
            "Your task is to extract structured information. "
            "Always respond with ONLY valid JSON - no explanations or other text."
        )

        # Call LLM - use synchronous generate for the local provider
        try:
            response = self._provider.generate(prompt, system_prompt=system_prompt)
            raw_output = response.text
        except Exception as e:
            logger.warning(f"LocalLLMSemanticAdapter: LLM call failed: {e}")
            end_time = time.perf_counter_ns()
            return SemanticAnnotation(
                provider="local-llm",
                model=self.model,
                normalized=NormalizedAnnotation(),
                confidence=0.0,
                latency_ms=(end_time - start_time) // 1_000_000,
                raw_model_output={"error": str(e)},
            )

        # Parse response
        parsed = self._parse_llm_response(raw_output)

        # Build normalized annotation
        normalized = self._build_normalized_annotation(parsed, context)

        # Compute confidence
        confidence = self._compute_overall_confidence(parsed)

        # Compute latency
        end_time = time.perf_counter_ns()
        latency_ms = (end_time - start_time) // 1_000_000

        return SemanticAnnotation(
            schema_version=SEMANTIC_SCHEMA_VERSION,
            provider="local-llm",
            model=self.model,
            normalized=normalized,
            confidence=confidence,
            latency_ms=latency_ms,
            raw_model_output={"parsed": parsed, "raw": raw_output[:500]},  # Truncate raw
        )

    def health_check(self) -> ProviderHealth:
        """Check provider availability.

        Returns:
            ProviderHealth indicating availability and any errors.
        """
        start_time = time.perf_counter_ns()

        if not self._lazy_load_provider():
            end_time = time.perf_counter_ns()
            return ProviderHealth(
                available=False,
                quota_remaining=None,
                error="LocalLLMProvider not available - check transformers/torch installation",
                latency_ms=(end_time - start_time) // 1_000_000,
            )

        end_time = time.perf_counter_ns()
        return ProviderHealth(
            available=True,
            quota_remaining=None,  # No quota for local models
            error=None,
            latency_ms=(end_time - start_time) // 1_000_000,
        )


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
                 - "local-llm": LocalLLMSemanticAdapter (local LLM-based, #89)
                 - "openai": OpenAISemanticAdapter (GPT-4o, etc., #90)
                 - "anthropic": AnthropicSemanticAdapter (Claude, #90)
                 - "noop": NoOpSemanticAdapter (placeholder)
        **kwargs: Provider-specific configuration.
                 For "local-llm":
                 - model: Model identifier (default: "Qwen/Qwen2.5-7B-Instruct")
                 - temperature: LLM temperature (default: 0.1)
                 - max_tokens: Max tokens for response (default: 1024)
                 - extraction_mode: "combined", "topics", "risks", "actions"
                 For "openai":
                 - api_key: OpenAI API key (or use OPENAI_API_KEY env)
                 - model: Model to use (default: "gpt-4o")
                 - base_url: Optional custom API base URL
                 - guardrails: Optional LLMGuardrails instance
                 - temperature: Sampling temperature (default: 0.3)
                 - max_tokens: Max tokens for response (default: 1024)
                 - context_window: Number of previous chunks for context (default: 3)
                 For "anthropic":
                 - api_key: Anthropic API key (or use ANTHROPIC_API_KEY env)
                 - model: Model to use (default: "claude-sonnet-4-20250514")
                 - guardrails: Optional LLMGuardrails instance
                 - temperature: Sampling temperature (default: 0.3)
                 - max_tokens: Max tokens for response (default: 1024)
                 - context_window: Number of previous chunks for context (default: 3)

    Returns:
        A SemanticAdapter implementation.

    Raises:
        ValueError: If provider is unknown.

    Example:
        >>> adapter = create_adapter("local")
        >>> annotation = adapter.annotate_chunk("I'll send the report", context)

        >>> # Use local LLM adapter
        >>> adapter = create_adapter("local-llm", model="Qwen/Qwen2.5-3B-Instruct")
        >>> annotation = adapter.annotate_chunk("I'll send the report", context)

        >>> # Use OpenAI adapter
        >>> adapter = create_adapter("openai", model="gpt-4o")
        >>> annotation = adapter.annotate_chunk("I'll send the report", context)

        >>> # Use Anthropic adapter
        >>> adapter = create_adapter("anthropic", model="claude-sonnet-4-20250514")
        >>> annotation = adapter.annotate_chunk("I'll send the report", context)
    """
    if provider == "local":
        return LocalKeywordAdapter(**kwargs)
    elif provider == "local-llm":
        return LocalLLMSemanticAdapter(**kwargs)
    elif provider == "openai":
        return OpenAISemanticAdapter(**kwargs)
    elif provider == "anthropic":
        return AnthropicSemanticAdapter(**kwargs)
    elif provider == "noop":
        return NoOpSemanticAdapter()
    else:
        raise ValueError(
            f"Unknown semantic provider: {provider}. "
            f"Supported: 'local', 'local-llm', 'openai', 'anthropic', 'noop'."
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
    "LocalLLMSemanticAdapter",
    "CloudLLMSemanticAdapter",
    "OpenAISemanticAdapter",
    "AnthropicSemanticAdapter",
    "NoOpSemanticAdapter",
    # Prompt templates (local LLM)
    "TOPIC_EXTRACTION_PROMPT",
    "RISK_DETECTION_PROMPT",
    "ACTION_EXTRACTION_PROMPT",
    "COMBINED_EXTRACTION_PROMPT",
    # Prompt templates (cloud LLM)
    "CLOUD_SEMANTIC_EXTRACTION_SYSTEM_PROMPT",
    "CLOUD_SEMANTIC_EXTRACTION_USER_TEMPLATE",
    # Factory
    "create_adapter",
]
