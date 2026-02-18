"""Evidence-grade outcomes extractor for meetings and conversations.

This module provides extraction of decisions, action items, risks, commitments,
and questions from transcripts with full citation support (provenance).

Key components:
- Citation: Source reference pointing back to specific segments
- Outcome: Extracted outcome with type, summary, and required citations
- BaselineOutcomeExtractor: Deterministic rule-based extraction
- LLMOutcomeExtractor: LLM-based extraction using semantic adapter
- OutcomeProcessor: High-level processor supporting multiple backends

Schema version: 0.1.0

See docs/OUTCOMES.md for usage examples (when available).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from .models import Segment, Transcript
    from .semantic_adapter import SemanticAdapter

logger = logging.getLogger(__name__)

# Schema version for outcomes
OUTCOMES_SCHEMA_VERSION = "0.1.0"

# Outcome types
OutcomeType = Literal["decision", "action_item", "risk", "commitment", "question"]


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class Citation:
    """Source reference for an extracted outcome.

    Citations provide provenance, linking outcomes back to the exact
    source segments in the transcript. Every outcome MUST have at least
    one citation.

    Attributes:
        segment_id: ID of the source segment (string for compatibility).
        start_time: Start time of the segment in seconds.
        end_time: End time of the segment in seconds.
        speaker_id: Optional speaker identifier.
        quote: Exact text from the segment supporting this outcome.
    """

    segment_id: str
    start_time: float
    end_time: float
    speaker_id: str | None
    quote: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize citation to JSON-compatible dict."""
        result: dict[str, Any] = {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "quote": self.quote,
        }
        if self.speaker_id is not None:
            result["speaker_id"] = self.speaker_id
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Citation:
        """Deserialize citation from dict."""
        return cls(
            segment_id=str(d.get("segment_id", "")),
            start_time=float(d.get("start_time", 0.0)),
            end_time=float(d.get("end_time", 0.0)),
            speaker_id=d.get("speaker_id"),
            quote=str(d.get("quote", "")),
        )

    @classmethod
    def from_segment(cls, segment: Segment, quote: str | None = None) -> Citation:
        """Create a citation from a Segment object.

        Args:
            segment: The source segment.
            quote: Optional specific quote. If None, uses segment.text.

        Returns:
            Citation referencing the segment.
        """
        speaker_id = None
        speaker = getattr(segment, "speaker", None)
        if isinstance(speaker, dict):
            speaker_id = speaker.get("id")
        elif isinstance(speaker, str):
            speaker_id = speaker

        return cls(
            segment_id=str(segment.id),
            start_time=segment.start,
            end_time=segment.end,
            speaker_id=speaker_id,
            quote=quote or segment.text,
        )


@dataclass(slots=True)
class Outcome:
    """Extracted outcome with evidence-grade citations.

    An Outcome represents a decision, action item, risk, commitment, or
    question extracted from the transcript. Every outcome MUST include
    at least one citation back to source segments.

    Attributes:
        outcome_type: Classification of the outcome.
        summary: Human-readable summary of the outcome.
        citations: List of citations (MUST have at least one).
        confidence: Confidence score (0.0-1.0).
        metadata: Additional type-specific metadata.
    """

    outcome_type: OutcomeType
    summary: str
    citations: list[Citation]
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that citations is not empty."""
        if not self.citations:
            raise ValueError("Outcome must have at least one citation")

    def to_dict(self) -> dict[str, Any]:
        """Serialize outcome to JSON-compatible dict."""
        return {
            "outcome_type": self.outcome_type,
            "summary": self.summary,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Outcome:
        """Deserialize outcome from dict."""
        citations_raw = d.get("citations", [])
        citations = [Citation.from_dict(c) for c in citations_raw]
        return cls(
            outcome_type=d.get("outcome_type", "action_item"),
            summary=str(d.get("summary", "")),
            citations=citations,
            confidence=float(d.get("confidence", 1.0)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass(slots=True)
class OutcomeExtractionResult:
    """Result of outcome extraction from a transcript.

    Attributes:
        schema_version: Version of the outcomes schema.
        outcomes: List of extracted outcomes.
        backend: Backend used for extraction ("baseline" or "llm").
        model: Model identifier (for LLM backends).
        latency_ms: Total extraction time in milliseconds.
        metadata: Additional extraction metadata.
    """

    schema_version: str = OUTCOMES_SCHEMA_VERSION
    outcomes: list[Outcome] = field(default_factory=list)
    backend: str = "baseline"
    model: str | None = None
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to JSON-compatible dict."""
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "backend": self.backend,
            "latency_ms": self.latency_ms,
        }
        if self.model is not None:
            result["model"] = self.model
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OutcomeExtractionResult:
        """Deserialize result from dict."""
        outcomes_raw = d.get("outcomes", [])
        outcomes = [Outcome.from_dict(o) for o in outcomes_raw]
        return cls(
            schema_version=str(d.get("schema_version", OUTCOMES_SCHEMA_VERSION)),
            outcomes=outcomes,
            backend=str(d.get("backend", "baseline")),
            model=d.get("model"),
            latency_ms=int(d.get("latency_ms", 0)),
            metadata=dict(d.get("metadata", {})),
        )


# -----------------------------------------------------------------------------
# Pattern Definitions for Baseline Extractor
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PatternRule:
    """A pattern rule for outcome extraction.

    Attributes:
        pattern: Regex pattern to match.
        outcome_type: Type of outcome this pattern indicates.
        priority: Priority for deduplication (higher = keep).
    """

    pattern: str
    outcome_type: OutcomeType
    priority: int = 0


# Decision patterns
DECISION_PATTERNS: tuple[PatternRule, ...] = (
    PatternRule(r"\bwe decided\b", "decision", 10),
    PatternRule(r"\bthe decision is\b", "decision", 10),
    PatternRule(r"\bwe(?:'re| are) going with\b", "decision", 9),
    PatternRule(r"\bwe(?:'ve| have) decided\b", "decision", 10),
    PatternRule(r"\blet(?:'s|'s| us) go with\b", "decision", 8),
    PatternRule(r"\bfinal decision\b", "decision", 10),
    PatternRule(r"\bdecided to\b", "decision", 8),
    PatternRule(r"\bwe will proceed with\b", "decision", 9),
    PatternRule(r"\bagreed to\b", "decision", 7),
    PatternRule(r"\bwe(?:'re| are) choosing\b", "decision", 7),
)

# Action item patterns
ACTION_ITEM_PATTERNS: tuple[PatternRule, ...] = (
    PatternRule(r"\bi will\b", "action_item", 8),
    PatternRule(r"\bi['']ll\b", "action_item", 8),
    PatternRule(r"\byou need to\b", "action_item", 9),
    PatternRule(r"\baction item[:\s]", "action_item", 10),
    PatternRule(r"\btodo[:\s]", "action_item", 10),
    PatternRule(r"\bto[ -]?do[:\s]", "action_item", 10),
    PatternRule(r"\bwe need to\b", "action_item", 7),
    PatternRule(r"\bplease\s+\w+\b", "action_item", 5),
    PatternRule(r"\bcan you\b", "action_item", 4),
    PatternRule(r"\blet me\b", "action_item", 6),
    PatternRule(r"\bi(?:'ll| will) send\b", "action_item", 9),
    PatternRule(r"\bi(?:'ll| will) follow up\b", "action_item", 9),
    PatternRule(r"\bfollow up on\b", "action_item", 7),
    PatternRule(r"\bfollow-up\b", "action_item", 6),
    PatternRule(r"\bnext step[s]?\b", "action_item", 6),
    PatternRule(r"\btake this offline\b", "action_item", 5),
)

# Risk patterns
RISK_PATTERNS: tuple[PatternRule, ...] = (
    PatternRule(r"\brisk is\b", "risk", 10),
    PatternRule(r"\bconcern about\b", "risk", 8),
    PatternRule(r"\bmight fail\b", "risk", 9),
    PatternRule(r"\bworried about\b", "risk", 8),
    PatternRule(r"\bpotential issue\b", "risk", 8),
    PatternRule(r"\bpotential problem\b", "risk", 8),
    PatternRule(r"\brisk of\b", "risk", 9),
    PatternRule(r"\brisks?\b", "risk", 5),
    PatternRule(r"\bdanger(?:ous)?\b", "risk", 7),
    PatternRule(r"\bcould go wrong\b", "risk", 8),
    PatternRule(r"\bworst case\b", "risk", 7),
    PatternRule(r"\bif .+ fails?\b", "risk", 6),
    PatternRule(r"\bblocker\b", "risk", 8),
    PatternRule(r"\bblocking issue\b", "risk", 9),
)

# Commitment patterns
COMMITMENT_PATTERNS: tuple[PatternRule, ...] = (
    PatternRule(r"\bi commit to\b", "commitment", 10),
    PatternRule(r"\bi promise\b", "commitment", 10),
    PatternRule(
        r"\bby (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "commitment", 7
    ),
    PatternRule(r"\bby (?:tomorrow|next week|end of (?:day|week|month))\b", "commitment", 8),
    PatternRule(r"\bdeadline\b", "commitment", 6),
    PatternRule(r"\bdue date\b", "commitment", 6),
    PatternRule(r"\bguarantee\b", "commitment", 9),
    PatternRule(r"\bi(?:'ll| will) have it (?:done|ready|finished)\b", "commitment", 9),
    PatternRule(r"\bdeliverable\b", "commitment", 5),
    PatternRule(r"\bmilestone\b", "commitment", 5),
)

# Question patterns (for completeness)
QUESTION_PATTERNS: tuple[PatternRule, ...] = (
    PatternRule(r"\?\s*$", "question", 10),  # Ends with question mark
    PatternRule(
        r"^(?:what|who|when|where|why|how|which|can|could|would|should|is|are|do|does|did)\b",
        "question",
        8,
    ),
    PatternRule(r"\bany questions\b", "question", 5),
    PatternRule(r"\bopen question\b", "question", 9),
)

# All patterns combined
ALL_PATTERNS: tuple[PatternRule, ...] = (
    *DECISION_PATTERNS,
    *ACTION_ITEM_PATTERNS,
    *RISK_PATTERNS,
    *COMMITMENT_PATTERNS,
    *QUESTION_PATTERNS,
)


# -----------------------------------------------------------------------------
# Extractor Protocol
# -----------------------------------------------------------------------------


class OutcomeExtractor(Protocol):
    """Protocol for outcome extraction backends."""

    def extract(self, transcript: Transcript) -> OutcomeExtractionResult:
        """Extract outcomes from a transcript.

        Args:
            transcript: The source transcript.

        Returns:
            OutcomeExtractionResult with extracted outcomes and metadata.
        """
        ...


# -----------------------------------------------------------------------------
# Baseline Extractor (Rule-Based)
# -----------------------------------------------------------------------------


class BaselineOutcomeExtractor:
    """Deterministic rule-based outcome extractor.

    Uses regex patterns to identify decisions, action items, risks,
    commitments, and questions. Every extraction includes citation
    back to the source segment.

    Example:
        >>> extractor = BaselineOutcomeExtractor()
        >>> result = extractor.extract(transcript)
        >>> for outcome in result.outcomes:
        ...     print(f"{outcome.outcome_type}: {outcome.summary}")
    """

    def __init__(
        self,
        decision_patterns: tuple[PatternRule, ...] | None = None,
        action_item_patterns: tuple[PatternRule, ...] | None = None,
        risk_patterns: tuple[PatternRule, ...] | None = None,
        commitment_patterns: tuple[PatternRule, ...] | None = None,
        question_patterns: tuple[PatternRule, ...] | None = None,
    ) -> None:
        """Initialize the baseline extractor.

        Args:
            decision_patterns: Custom decision patterns (or use defaults).
            action_item_patterns: Custom action item patterns (or use defaults).
            risk_patterns: Custom risk patterns (or use defaults).
            commitment_patterns: Custom commitment patterns (or use defaults).
            question_patterns: Custom question patterns (or use defaults).
        """
        self._decision_patterns = decision_patterns or DECISION_PATTERNS
        self._action_item_patterns = action_item_patterns or ACTION_ITEM_PATTERNS
        self._risk_patterns = risk_patterns or RISK_PATTERNS
        self._commitment_patterns = commitment_patterns or COMMITMENT_PATTERNS
        self._question_patterns = question_patterns or QUESTION_PATTERNS

        # Pre-compile all patterns
        self._compiled_patterns: list[tuple[re.Pattern[str], PatternRule]] = []
        all_patterns = (
            *self._decision_patterns,
            *self._action_item_patterns,
            *self._risk_patterns,
            *self._commitment_patterns,
            *self._question_patterns,
        )
        for rule in all_patterns:
            try:
                compiled = re.compile(rule.pattern, re.IGNORECASE)
                self._compiled_patterns.append((compiled, rule))
            except re.error as e:
                logger.warning(f"Invalid pattern '{rule.pattern}': {e}")

    def extract(self, transcript: Transcript) -> OutcomeExtractionResult:
        """Extract outcomes from transcript using pattern matching.

        Args:
            transcript: The source transcript.

        Returns:
            OutcomeExtractionResult with extracted outcomes.
        """
        import time

        start_time = time.perf_counter_ns()
        outcomes: list[Outcome] = []

        for segment in transcript.segments:
            text = getattr(segment, "text", "") or ""
            text = text.strip()
            if not text:
                continue

            # Test each pattern against the segment text
            for compiled, rule in self._compiled_patterns:
                if compiled.search(text):
                    # Create citation from the segment
                    citation = Citation.from_segment(segment, quote=text)

                    # Create outcome
                    outcome = Outcome(
                        outcome_type=rule.outcome_type,
                        summary=text,
                        citations=[citation],
                        confidence=1.0,  # Rule-based = deterministic
                        metadata={
                            "pattern": rule.pattern,
                            "priority": rule.priority,
                        },
                    )
                    outcomes.append(outcome)
                    # Only match first pattern per segment per type
                    # to avoid duplicates from similar patterns
                    break

        end_time = time.perf_counter_ns()
        latency_ms = (end_time - start_time) // 1_000_000

        return OutcomeExtractionResult(
            schema_version=OUTCOMES_SCHEMA_VERSION,
            outcomes=outcomes,
            backend="baseline",
            model=None,
            latency_ms=latency_ms,
            metadata={
                "segment_count": len(transcript.segments),
                "pattern_count": len(self._compiled_patterns),
            },
        )


# -----------------------------------------------------------------------------
# LLM Extractor
# -----------------------------------------------------------------------------


# Prompt for LLM-based outcome extraction
LLM_OUTCOME_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert at extracting structured outcomes from meeting transcripts.
Your task is to identify decisions, action items, risks, commitments, and questions.

CRITICAL: Every outcome MUST include a citation with the exact quote from the transcript.

Output a JSON object with this structure:
{
  "outcomes": [
    {
      "outcome_type": "decision|action_item|risk|commitment|question",
      "summary": "Clear summary of the outcome",
      "segment_id": "ID of the source segment",
      "quote": "EXACT quote from the transcript supporting this outcome",
      "confidence": 0.0-1.0,
      "metadata": {}
    }
  ]
}

Extraction rules:
- decision: Explicit choices made ("we decided", "going with", "agreed to")
- action_item: Tasks to be done ("I will", "you need to", "action item:")
- risk: Potential problems ("risk is", "might fail", "concerned about")
- commitment: Promises with timelines ("I commit to", "by tomorrow", "deadline")
- question: Unresolved questions (ends with ?, open questions)

Rules:
- Return ONLY valid JSON, no markdown or explanation
- Every outcome MUST have a quote from the source segment
- The quote MUST be the exact text from the transcript
- If no outcomes found, return {"outcomes": []}
"""

LLM_OUTCOME_EXTRACTION_USER_TEMPLATE = """Analyze this transcript and extract all outcomes.

SEGMENTS:
{segments}

Extract outcomes with citations. Return JSON."""


class LLMOutcomeExtractor:
    """LLM-based outcome extractor using semantic adapter.

    Uses cloud or local LLM to extract outcomes with higher accuracy
    than rule-based approaches. Validates that all citations actually
    exist in the source transcript.

    Example:
        >>> from transcription.semantic_adapter import create_adapter
        >>> adapter = create_adapter("openai", model="gpt-4o")
        >>> extractor = LLMOutcomeExtractor(adapter)
        >>> result = extractor.extract(transcript)
    """

    def __init__(
        self,
        adapter: SemanticAdapter,
        max_segments_per_chunk: int = 50,
    ) -> None:
        """Initialize the LLM extractor.

        Args:
            adapter: Semantic adapter for LLM calls.
            max_segments_per_chunk: Max segments per LLM call (for long transcripts).
        """
        self._adapter = adapter
        self._max_segments_per_chunk = max_segments_per_chunk

    def _format_segments(self, segments: list[Segment]) -> str:
        """Format segments for LLM prompt.

        Args:
            segments: List of segments to format.

        Returns:
            Formatted string for LLM prompt.
        """
        lines = []
        for segment in segments:
            speaker = ""
            speaker_meta = getattr(segment, "speaker", None)
            if isinstance(speaker_meta, dict):
                speaker = speaker_meta.get("id", "")
            elif isinstance(speaker_meta, str):
                speaker = speaker_meta

            line = f"[{segment.id}] ({segment.start:.1f}s-{segment.end:.1f}s)"
            if speaker:
                line += f" {speaker}:"
            line += f" {segment.text}"
            lines.append(line)
        return "\n".join(lines)

    def _build_segment_index(self, transcript: Transcript) -> dict[str, Segment]:
        """Build index of segment ID to segment for validation.

        Args:
            transcript: Source transcript.

        Returns:
            Dict mapping segment ID (as string) to Segment.
        """
        return {str(seg.id): seg for seg in transcript.segments}

    def _validate_citation(
        self, citation_data: dict[str, Any], segment_index: dict[str, Segment]
    ) -> Citation | None:
        """Validate and create citation from LLM output.

        Args:
            citation_data: Raw citation data from LLM.
            segment_index: Index of segments for validation.

        Returns:
            Citation if valid, None otherwise.
        """
        segment_id = str(citation_data.get("segment_id", ""))
        quote = str(citation_data.get("quote", ""))

        if not segment_id or segment_id not in segment_index:
            logger.warning(f"LLM returned invalid segment_id: {segment_id}")
            return None

        segment = segment_index[segment_id]

        # Validate that quote exists in segment text (fuzzy match)
        segment_text_lower = segment.text.lower()
        quote_lower = quote.lower()

        # Allow partial matches (LLM might truncate or paraphrase)
        if quote_lower not in segment_text_lower:
            # Try checking if significant words match
            quote_words = set(quote_lower.split())
            segment_words = set(segment_text_lower.split())
            overlap = len(quote_words & segment_words)
            if overlap < len(quote_words) * 0.5:
                logger.warning(f"Quote does not match segment text: '{quote[:50]}...'")
                # Still create citation but use segment text
                quote = segment.text

        return Citation.from_segment(segment, quote=quote)

    def _parse_llm_response(
        self, response_text: str, segment_index: dict[str, Segment]
    ) -> list[Outcome]:
        """Parse LLM response and validate citations.

        Args:
            response_text: Raw LLM response.
            segment_index: Index of segments for validation.

        Returns:
            List of validated outcomes.
        """
        outcomes: list[Outcome] = []

        # Strip markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"LLMOutcomeExtractor: JSON parse error: {e}")
            return outcomes

        raw_outcomes = data.get("outcomes", [])
        for raw_outcome in raw_outcomes:
            if not isinstance(raw_outcome, dict):
                continue

            outcome_type = raw_outcome.get("outcome_type", "action_item")
            if outcome_type not in ("decision", "action_item", "risk", "commitment", "question"):
                outcome_type = "action_item"

            summary = str(raw_outcome.get("summary", ""))
            confidence = float(raw_outcome.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))

            # Validate citation
            citation = self._validate_citation(raw_outcome, segment_index)
            if citation is None:
                # Skip outcomes without valid citations
                continue

            outcome = Outcome(
                outcome_type=outcome_type,
                summary=summary,
                citations=[citation],
                confidence=confidence,
                metadata=raw_outcome.get("metadata", {}),
            )
            outcomes.append(outcome)

        return outcomes

    def extract(self, transcript: Transcript) -> OutcomeExtractionResult:
        """Extract outcomes from transcript using LLM.

        Args:
            transcript: The source transcript.

        Returns:
            OutcomeExtractionResult with extracted outcomes.
        """
        import time

        from .semantic_adapter import ChunkContext

        start_time = time.perf_counter_ns()
        all_outcomes: list[Outcome] = []

        # Build segment index for citation validation
        segment_index = self._build_segment_index(transcript)

        # Process in chunks if transcript is long
        segments = list(transcript.segments)
        for i in range(0, len(segments), self._max_segments_per_chunk):
            chunk_segments = segments[i : i + self._max_segments_per_chunk]

            # Format segments for prompt
            segments_text = self._format_segments(chunk_segments)
            user_prompt = LLM_OUTCOME_EXTRACTION_USER_TEMPLATE.format(segments=segments_text)

            # Create context for adapter
            context = ChunkContext(
                start=chunk_segments[0].start if chunk_segments else 0.0,
                end=chunk_segments[-1].end if chunk_segments else 0.0,
                segment_ids=[seg.id for seg in chunk_segments],
            )

            # Call LLM via adapter
            # We'll use annotate_chunk but with our custom prompt
            # For better results, we'd want a dedicated method, but this works
            try:
                # Build combined prompt
                combined_prompt = f"{LLM_OUTCOME_EXTRACTION_SYSTEM_PROMPT}\n\n{user_prompt}"
                annotation = self._adapter.annotate_chunk(combined_prompt, context)

                # Extract raw response
                raw_output = annotation.raw_model_output or {}
                response_text = raw_output.get("response", "")

                if response_text:
                    outcomes = self._parse_llm_response(response_text, segment_index)
                    all_outcomes.extend(outcomes)
            except Exception as e:
                logger.warning(f"LLMOutcomeExtractor: LLM call failed: {e}")

        end_time = time.perf_counter_ns()
        latency_ms = (end_time - start_time) // 1_000_000

        # Get model info from adapter
        model = None
        try:
            model = getattr(self._adapter, "_model", None)
            if model is None:
                model = getattr(self._adapter, "model", None)
        except AttributeError:
            pass

        return OutcomeExtractionResult(
            schema_version=OUTCOMES_SCHEMA_VERSION,
            outcomes=all_outcomes,
            backend="llm",
            model=model,
            latency_ms=latency_ms,
            metadata={
                "segment_count": len(transcript.segments),
            },
        )


# -----------------------------------------------------------------------------
# Outcome Processor (High-Level API)
# -----------------------------------------------------------------------------


class OutcomeProcessor:
    """High-level processor for outcome extraction.

    Supports multiple backends (baseline, LLM) and provides deduplication
    and citation merging.

    Example:
        >>> processor = OutcomeProcessor(backend="baseline")
        >>> result = processor.extract(transcript)
        >>> print(f"Found {len(result.outcomes)} outcomes")

        >>> # With LLM backend
        >>> from transcription.semantic_adapter import create_adapter
        >>> adapter = create_adapter("openai")
        >>> processor = OutcomeProcessor(backend="llm", adapter=adapter)
        >>> result = processor.extract(transcript)
    """

    def __init__(
        self,
        backend: Literal["baseline", "llm"] = "baseline",
        adapter: SemanticAdapter | None = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.8,
    ) -> None:
        """Initialize the outcome processor.

        Args:
            backend: Extraction backend ("baseline" or "llm").
            adapter: Semantic adapter for LLM backend (required if backend="llm").
            deduplicate: Whether to deduplicate similar outcomes.
            similarity_threshold: Threshold for considering outcomes similar (0.0-1.0).

        Raises:
            ValueError: If backend="llm" but no adapter provided.
        """
        self._backend = backend
        self._deduplicate = deduplicate
        self._similarity_threshold = similarity_threshold

        if backend == "llm":
            if adapter is None:
                raise ValueError("LLM backend requires an adapter")
            self._extractor: OutcomeExtractor = LLMOutcomeExtractor(adapter)
        else:
            self._extractor = BaselineOutcomeExtractor()

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple word-based similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score (0.0-1.0).
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _deduplicate_outcomes(self, outcomes: list[Outcome]) -> list[Outcome]:
        """Deduplicate similar outcomes, merging citations.

        Args:
            outcomes: List of outcomes to deduplicate.

        Returns:
            Deduplicated list with merged citations.
        """
        if not outcomes:
            return []

        # Group by outcome type first
        by_type: dict[OutcomeType, list[Outcome]] = {}
        for outcome in outcomes:
            by_type.setdefault(outcome.outcome_type, []).append(outcome)

        deduplicated: list[Outcome] = []

        for outcome_type, type_outcomes in by_type.items():
            # Track which outcomes have been merged
            merged_indices: set[int] = set()

            for i, outcome1 in enumerate(type_outcomes):
                if i in merged_indices:
                    continue

                # Find similar outcomes
                merged_citations = list(outcome1.citations)
                best_confidence = outcome1.confidence
                best_summary = outcome1.summary
                best_metadata = dict(outcome1.metadata)

                for j, outcome2 in enumerate(type_outcomes[i + 1 :], start=i + 1):
                    if j in merged_indices:
                        continue

                    similarity = self._compute_similarity(outcome1.summary, outcome2.summary)
                    if similarity >= self._similarity_threshold:
                        # Merge citations
                        for citation in outcome2.citations:
                            # Avoid duplicate citations
                            if not any(
                                c.segment_id == citation.segment_id for c in merged_citations
                            ):
                                merged_citations.append(citation)

                        # Keep higher confidence
                        if outcome2.confidence > best_confidence:
                            best_confidence = outcome2.confidence
                            best_summary = outcome2.summary
                            best_metadata = dict(outcome2.metadata)

                        merged_indices.add(j)

                # Create merged outcome
                merged = Outcome(
                    outcome_type=outcome_type,
                    summary=best_summary,
                    citations=merged_citations,
                    confidence=best_confidence,
                    metadata=best_metadata,
                )
                deduplicated.append(merged)

        return deduplicated

    def extract(self, transcript: Transcript) -> OutcomeExtractionResult:
        """Extract outcomes from transcript.

        Args:
            transcript: The source transcript.

        Returns:
            OutcomeExtractionResult with extracted outcomes.
        """
        result = self._extractor.extract(transcript)

        if self._deduplicate:
            result.outcomes = self._deduplicate_outcomes(result.outcomes)

        return result


# -----------------------------------------------------------------------------
# Output Formatting
# -----------------------------------------------------------------------------


def format_outcomes_json(result: OutcomeExtractionResult) -> str:
    """Format outcome extraction result as JSON string.

    Args:
        result: The extraction result.

    Returns:
        JSON string with pretty formatting.
    """
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def format_outcomes_pretty(result: OutcomeExtractionResult) -> str:
    """Format outcome extraction result for human reading.

    Args:
        result: The extraction result.

    Returns:
        Human-readable string.
    """
    lines: list[str] = []
    lines.append(f"=== Outcomes ({len(result.outcomes)} found) ===\n")

    # Group by type
    by_type: dict[str, list[Outcome]] = {}
    for outcome in result.outcomes:
        by_type.setdefault(outcome.outcome_type, []).append(outcome)

    type_order = ["decision", "action_item", "commitment", "risk", "question"]
    type_labels = {
        "decision": "Decisions",
        "action_item": "Action Items",
        "commitment": "Commitments",
        "risk": "Risks",
        "question": "Open Questions",
    }

    for outcome_type in type_order:
        outcomes = by_type.get(outcome_type, [])
        if not outcomes:
            continue

        lines.append(f"\n{type_labels.get(outcome_type, outcome_type)} ({len(outcomes)}):")
        lines.append("-" * 40)

        for i, outcome in enumerate(outcomes, 1):
            lines.append(f"\n{i}. {outcome.summary}")
            lines.append(f"   Confidence: {outcome.confidence:.0%}")

            for citation in outcome.citations:
                time_str = f"{citation.start_time:.1f}s"
                speaker_str = f" [{citation.speaker_id}]" if citation.speaker_id else ""
                lines.append(
                    f'   @ {time_str}{speaker_str}: "{citation.quote[:80]}..."'
                    if len(citation.quote) > 80
                    else f'   @ {time_str}{speaker_str}: "{citation.quote}"'
                )

    lines.append(f"\n\nBackend: {result.backend}")
    if result.model:
        lines.append(f"Model: {result.model}")
    lines.append(f"Latency: {result.latency_ms}ms")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # Schema version
    "OUTCOMES_SCHEMA_VERSION",
    # Types
    "OutcomeType",
    # Data classes
    "Citation",
    "Outcome",
    "OutcomeExtractionResult",
    # Pattern rules
    "PatternRule",
    "DECISION_PATTERNS",
    "ACTION_ITEM_PATTERNS",
    "RISK_PATTERNS",
    "COMMITMENT_PATTERNS",
    "QUESTION_PATTERNS",
    "ALL_PATTERNS",
    # Protocol
    "OutcomeExtractor",
    # Implementations
    "BaselineOutcomeExtractor",
    "LLMOutcomeExtractor",
    "OutcomeProcessor",
    # Output formatting
    "format_outcomes_json",
    "format_outcomes_pretty",
]
