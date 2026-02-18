"""Type definitions for the Conversation Store.

This module defines:
- Query filter dataclasses (TextQuery, TimeRangeQuery, SpeakerQuery, etc.)
- Combined query dataclass (StoreQuery)
- Result TypedDicts (SegmentHit, TranscriptSummary, ActionItemRow)
- Operation dataclasses (IngestResult, IngestOptions, ExportOptions)
- Custom exceptions (StoreError, DuplicateError, SchemaVersionError)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, TypedDict

# =============================================================================
# Exceptions
# =============================================================================


class StoreError(Exception):
    """Base exception for store errors."""

    pass


class DuplicateError(StoreError):
    """Raised when attempting to insert a duplicate record."""

    def __init__(self, transcript_id: str, message: str | None = None):
        self.transcript_id = transcript_id
        super().__init__(message or f"Transcript already exists: {transcript_id}")


class SchemaVersionError(StoreError):
    """Raised when database schema version is incompatible."""

    def __init__(
        self,
        found_version: int,
        expected_version: int,
        message: str | None = None,
    ):
        self.found_version = found_version
        self.expected_version = expected_version
        super().__init__(
            message
            or f"Schema version mismatch: found {found_version}, expected {expected_version}"
        )


class QueryError(StoreError):
    """Raised when a query is invalid or fails."""

    pass


class IngestError(StoreError):
    """Raised when transcript ingestion fails."""

    pass


class ExportError(StoreError):
    """Raised when export fails."""

    pass


# =============================================================================
# Enums
# =============================================================================


class ActionStatus(str, Enum):
    """Status of an action item."""

    OPEN = "open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DuplicateHandling(str, Enum):
    """How to handle duplicate entries during ingest."""

    SKIP = "skip"
    REPLACE = "replace"
    ERROR = "error"


class ExportFormat(str, Enum):
    """Export format options."""

    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


# =============================================================================
# Query Filter Dataclasses
# =============================================================================


@dataclass(slots=True)
class TextQuery:
    """Full-text search query filter.

    Attributes:
        text: Search text (uses FTS5 MATCH syntax).
        match_type: How to match text.
            - 'phrase': Exact phrase match (default)
            - 'any': Match any word
            - 'all': Match all words
            - 'prefix': Prefix match (e.g., 'trans*')
        case_sensitive: Whether search is case-sensitive (default: False).
    """

    text: str
    match_type: Literal["phrase", "any", "all", "prefix"] = "phrase"
    case_sensitive: bool = False

    def to_fts_query(self) -> str:
        """Convert to FTS5 query syntax.

        Returns:
            FTS5 query string.
        """
        text = self.text.strip()
        if not text:
            return ""

        if self.match_type == "phrase":
            # Wrap in quotes for phrase search
            return f'"{text}"'
        elif self.match_type == "any":
            # OR between words
            words = text.split()
            return " OR ".join(words)
        elif self.match_type == "all":
            # AND between words (implicit in FTS5)
            words = text.split()
            return " ".join(words)
        else:  # match_type == "prefix"
            # Add asterisk for prefix match
            if not text.endswith("*"):
                text = f"{text}*"
            return text


@dataclass(slots=True)
class TimeRangeQuery:
    """Time range filter for segments (in seconds).

    Attributes:
        start: Minimum start time in seconds (inclusive).
        end: Maximum end time in seconds (inclusive).
    """

    start: float | None = None
    end: float | None = None


@dataclass
class TimeRange:
    """A time range for filtering queries by datetime.

    Both start and end are optional:
    - If only start is set, filter from start onwards
    - If only end is set, filter up to end
    - If both are set, filter within the range
    """

    start: datetime | None = None
    end: datetime | None = None

    @classmethod
    def parse(cls, range_str: str) -> TimeRange:
        """Parse a time range string like '2024-01-01-2024-12-31' or '2024-01-01-'.

        Formats:
        - 'START-END': Both start and end dates (YYYY-MM-DD)
        - 'START-': From start date onwards
        - '-END': Up to end date

        Raises:
            ValueError: If the format is invalid
        """
        if not range_str or range_str == "-":
            return cls()

        parts = range_str.split("-", maxsplit=1)
        if len(parts) == 1:
            # Single date - treat as start
            return cls(start=datetime.fromisoformat(parts[0]))

        # Handle YYYY-MM-DD format which has internal hyphens
        # Look for the pattern where we have date-date or date- or -date
        # Dates are 10 chars: YYYY-MM-DD
        if len(range_str) >= 21 and range_str[10] == "-":
            # Format: YYYY-MM-DD-YYYY-MM-DD
            start_str = range_str[:10]
            end_str = range_str[11:]
            start = datetime.fromisoformat(start_str) if start_str else None
            end = datetime.fromisoformat(end_str) if end_str else None
            return cls(start=start, end=end)
        elif len(range_str) == 11 and range_str.endswith("-"):
            # Format: YYYY-MM-DD-
            return cls(start=datetime.fromisoformat(range_str[:10]))
        elif len(range_str) == 11 and range_str.startswith("-"):
            # Format: -YYYY-MM-DD
            return cls(end=datetime.fromisoformat(range_str[1:]))
        else:
            # Try simple parse
            start = datetime.fromisoformat(parts[0]) if parts[0] else None
            end = datetime.fromisoformat(parts[1]) if len(parts) > 1 and parts[1] else None
            return cls(start=start, end=end)


@dataclass(slots=True)
class SpeakerQuery:
    """Speaker filter.

    Attributes:
        speaker_ids: List of speaker IDs to include.
        exclude: If True, exclude these speakers instead of including.
    """

    speaker_ids: list[str] = field(default_factory=list)
    exclude: bool = False


@dataclass(slots=True)
class TranscriptQuery:
    """Transcript filter.

    Attributes:
        transcript_ids: List of transcript IDs to include.
        file_names: List of file names to include.
        languages: List of language codes to include.
    """

    transcript_ids: list[str] = field(default_factory=list)
    file_names: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SemanticQuery:
    """Semantic annotation filter.

    Attributes:
        topics: List of topic labels to match.
        risk_tags: List of risk tags to match.
        intents: List of intents to match.
        sentiments: List of sentiments to match.
        has_action_items: If True, only include segments with action items.
    """

    topics: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    sentiments: list[str] = field(default_factory=list)
    has_action_items: bool | None = None


@dataclass(slots=True)
class DateRangeQuery:
    """Date range filter for ingestion time.

    Attributes:
        after: Only include transcripts ingested after this datetime (ISO format).
        before: Only include transcripts ingested before this datetime (ISO format).
    """

    after: str | None = None
    before: str | None = None


# =============================================================================
# Combined Query
# =============================================================================


@dataclass
class StoreQuery:
    """Combined query with multiple filter types.

    All filters are combined with AND logic. Empty/None filters are ignored.

    Attributes:
        text: Full-text search query.
        time_range: Time range filter for segments.
        speakers: Speaker filter.
        transcripts: Transcript filter.
        semantic: Semantic annotation filter.
        date_range: Date range filter for ingestion time.
        limit: Maximum number of results (default: 100).
        offset: Number of results to skip (default: 0).
        order_by: Column to order by (default: 'start_time').
        order_desc: If True, order descending (default: False).

    Example:
        >>> query = StoreQuery(
        ...     text=TextQuery("pricing discussion"),
        ...     speakers=SpeakerQuery(speaker_ids=["agent"]),
        ...     limit=50,
        ... )
    """

    text: TextQuery | None = None
    time_range: TimeRangeQuery | None = None
    speakers: SpeakerQuery | None = None
    transcripts: TranscriptQuery | None = None
    semantic: SemanticQuery | None = None
    date_range: DateRangeQuery | None = None
    limit: int = 100
    offset: int = 0
    order_by: str = "start_time"
    order_desc: bool = False

    def is_empty(self) -> bool:
        """Check if query has no filters (would return all results).

        Returns:
            True if no filters are set.
        """
        return (
            self.text is None
            and self.time_range is None
            and self.speakers is None
            and self.transcripts is None
            and self.semantic is None
            and self.date_range is None
        )


# Legacy alias for backward compatibility
@dataclass
class QueryFilter:
    """Filter criteria for querying the conversation store.

    All filters are optional and combined with AND logic.

    Note: This is a legacy type. Use StoreQuery for new code.
    """

    # Semantic search
    text: str | None = None

    # Metadata filters
    speaker_id: str | None = None
    time_range: TimeRange | None = None
    topic: str | None = None
    tags: list[str] = field(default_factory=list)

    # Source file filter
    source_file: str | None = None

    # Result limits
    limit: int = 10


# =============================================================================
# Result TypedDicts
# =============================================================================


class SegmentHit(TypedDict, total=False):
    """Search result for a segment.

    Attributes:
        segment_id: Database row ID.
        transcript_id: Parent transcript ID.
        segment_index: Index within transcript.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        text: Segment text.
        speaker_id: Speaker ID.
        speaker_confidence: Speaker confidence score.
        file_name: Source file name.
        language: Language code.
        rank: FTS5 relevance rank (lower is better).
        snippet: FTS5 snippet with highlighting.
        audio_state: Audio state dict (if requested).
    """

    segment_id: int
    transcript_id: str
    segment_index: int
    start_time: float
    end_time: float
    text: str
    speaker_id: str | None
    speaker_confidence: float | None
    file_name: str
    language: str | None
    rank: float
    snippet: str
    audio_state: dict[str, Any] | None


class TranscriptSummary(TypedDict, total=False):
    """Summary of a transcript.

    Attributes:
        transcript_id: Unique transcript ID.
        file_name: Source file name.
        language: Language code.
        duration_seconds: Total duration.
        word_count: Total word count.
        segment_count: Number of segments.
        speaker_count: Number of speakers.
        speakers: List of speaker IDs.
        ingested_at: When transcript was ingested.
        source_path: Original source file path.
        has_annotations: Whether semantic annotations exist.
        action_item_count: Number of action items.
    """

    transcript_id: str
    file_name: str
    language: str | None
    duration_seconds: float | None
    word_count: int | None
    segment_count: int | None
    speaker_count: int | None
    speakers: list[str]
    ingested_at: str | None
    source_path: str | None
    has_annotations: bool
    action_item_count: int


class ActionItemRow(TypedDict, total=False):
    """Action item record.

    Attributes:
        action_id: Database row ID.
        transcript_id: Parent transcript ID.
        text: Action item text.
        assignee: Who is responsible.
        due: Due date/time.
        priority: Priority level.
        confidence: Detection confidence.
        segment_ids: List of segment indices.
        pattern: Regex pattern that matched.
        status: Current status.
        created_at: When action was created.
        updated_at: When status was last updated.
        file_name: Source file name.
    """

    action_id: int
    transcript_id: str
    text: str
    assignee: str | None
    due: str | None
    priority: str | None
    confidence: float | None
    segment_ids: list[int]
    pattern: str | None
    status: str
    created_at: str | None
    updated_at: str | None
    file_name: str


class StoreStats(TypedDict, total=False):
    """Store statistics.

    Attributes:
        transcript_count: Total number of transcripts.
        segment_count: Total number of segments.
        word_count: Total number of words.
        speaker_count: Total unique speakers.
        action_item_count: Total action items.
        open_action_items: Open action items count.
        completed_action_items: Completed action items count.
        total_duration_seconds: Sum of all transcript durations.
        database_size_bytes: Size of database file.
        oldest_ingestion: Datetime of oldest ingestion.
        newest_ingestion: Datetime of newest ingestion.
        topics: List of (topic, count) pairs.
    """

    transcript_count: int
    segment_count: int
    word_count: int
    speaker_count: int
    action_item_count: int
    open_action_items: int
    completed_action_items: int
    total_duration_seconds: float
    database_size_bytes: int
    oldest_ingestion: str | None
    newest_ingestion: str | None
    topics: list[tuple[str, int]]


# =============================================================================
# Legacy Types (backward compatibility)
# =============================================================================


@dataclass
class ConversationEntry:
    """A single conversation entry stored in the database.

    Represents a segment or turn from a transcript with associated metadata.

    Note: This is a legacy type. Use SegmentHit for new code.
    """

    id: str
    text: str
    speaker_id: str | None
    start_time: float
    end_time: float
    source_file: str
    ingested_at: datetime
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "source_file": self.source_file,
            "ingested_at": self.ingested_at.isoformat(),
            "tags": self.tags,
            "topics": self.topics,
            "metadata": self.metadata,
        }


@dataclass
class QueryResult:
    """Result from a store query.

    Note: This is a legacy type for backward compatibility.
    """

    entries: list[ConversationEntry]
    total_count: int
    query_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "total_count": self.total_count,
            "query_time_ms": self.query_time_ms,
        }


@dataclass
class ActionItem:
    """An action item extracted from conversations.

    Action items are extracted from transcript semantic annotations
    and can be tracked for completion.
    """

    id: str
    text: str
    status: ActionStatus
    source_entry_id: str
    source_file: str
    created_at: datetime
    completed_at: datetime | None = None
    assignee: str | None = None
    due_date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status.value,
            "source_entry_id": self.source_entry_id,
            "source_file": self.source_file,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "assignee": self.assignee,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "metadata": self.metadata,
        }


# =============================================================================
# Operation Dataclasses
# =============================================================================


@dataclass(slots=True)
class IngestOptions:
    """Options for transcript ingestion.

    Attributes:
        skip_duplicates: If True, skip transcripts with existing IDs (default: True).
            If False, raise DuplicateError on conflict.
        generate_transcript_id: If True, generate UUID for transcript_id.
            If False, use file_name hash.
        extract_annotations: If True, extract and store semantic annotations.
        store_words: If True, store word-level alignments if available.
        store_audio_state: If True, store audio_state for each segment.
        source_path: Optional source path to store for reference.
    """

    skip_duplicates: bool = True
    generate_transcript_id: bool = True
    extract_annotations: bool = True
    store_words: bool = True
    store_audio_state: bool = True
    source_path: str | None = None


@dataclass(slots=True)
class IngestResult:
    """Result of transcript ingestion.

    Attributes:
        transcript_id: The transcript ID (generated or existing).
        status: Operation status ('success', 'skipped', 'error').
        receipt_id: Provenance receipt ID.
        segment_count: Number of segments ingested.
        word_count: Number of words ingested.
        speaker_count: Number of speakers found.
        action_item_count: Number of action items extracted.
        error_message: Error message if status is 'error'.
        duration_ms: Operation duration in milliseconds.
    """

    transcript_id: str
    status: Literal["success", "skipped", "error"]
    receipt_id: str
    segment_count: int = 0
    word_count: int = 0
    speaker_count: int = 0
    action_item_count: int = 0
    error_message: str | None = None
    duration_ms: int = 0


@dataclass(slots=True)
class ExportOptions:
    """Options for exporting search results.

    Attributes:
        format: Export format (jsonl, json, csv).
        include_audio_state: Include audio_state in export.
        include_annotations: Include semantic annotations in export.
        include_words: Include word-level alignments in export.
        output_path: Path to write export file (if None, returns data).
        pretty_print: For JSON format, use pretty printing.
    """

    format: ExportFormat = ExportFormat.JSONL
    include_audio_state: bool = False
    include_annotations: bool = True
    include_words: bool = False
    output_path: str | None = None
    pretty_print: bool = False


@dataclass(slots=True)
class ExportResult:
    """Result of export operation.

    Attributes:
        status: Operation status ('success', 'error').
        record_count: Number of records exported.
        output_path: Path to output file (if written).
        data: Export data (if not written to file).
        error_message: Error message if status is 'error'.
        duration_ms: Operation duration in milliseconds.
    """

    status: Literal["success", "error"]
    record_count: int = 0
    output_path: str | None = None
    data: str | None = None
    error_message: str | None = None
    duration_ms: int = 0


# Type alias for output formats
OutputFormat = Literal["table", "json", "jsonl", "csv"]


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Exceptions
    "StoreError",
    "DuplicateError",
    "SchemaVersionError",
    "QueryError",
    "IngestError",
    "ExportError",
    # Enums
    "ActionStatus",
    "DuplicateHandling",
    "ExportFormat",
    # Query filters
    "TextQuery",
    "TimeRangeQuery",
    "TimeRange",
    "SpeakerQuery",
    "TranscriptQuery",
    "SemanticQuery",
    "DateRangeQuery",
    # Combined query
    "StoreQuery",
    "QueryFilter",  # Legacy
    # Result types
    "SegmentHit",
    "TranscriptSummary",
    "ActionItemRow",
    "StoreStats",
    # Legacy types
    "ConversationEntry",
    "QueryResult",
    "ActionItem",
    # Operation types
    "IngestOptions",
    "IngestResult",
    "ExportOptions",
    "ExportResult",
    # Type aliases
    "OutputFormat",
]
