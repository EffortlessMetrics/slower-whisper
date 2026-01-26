"""Conversation Store module for slower-whisper.

Provides persistent storage and semantic querying of transcripts using SQLite
with FTS5 full-text search support.

Main components:
- ConversationStore / SQLiteConversationStore: Main interface for storing and querying
- StoreQuery: Combined query with multiple filter types
- TextQuery, SpeakerQuery, etc.: Individual filter components
- SegmentHit, TranscriptSummary, ActionItemRow: Result types
- IngestOptions, IngestResult: Ingestion configuration and results
- ExportOptions, ExportResult: Export configuration and results

Example usage:
    >>> from transcription.store import ConversationStore, StoreQuery, TextQuery
    >>> store = ConversationStore.open("transcripts.db")
    >>> result = store.ingest("meeting.json")
    >>> hits = store.search(StoreQuery(text=TextQuery("pricing")))
    >>> store.close()
"""

from __future__ import annotations

# Re-export store class
from .store import ConversationStore, SQLiteConversationStore, get_default_store_path

# Re-export all types
from .types import (
    ActionItem,
    ActionItemRow,
    # Enums
    ActionStatus,
    # Legacy types
    ConversationEntry,
    DateRangeQuery,
    DuplicateError,
    DuplicateHandling,
    ExportError,
    ExportFormat,
    ExportOptions,
    ExportResult,
    IngestError,
    # Operation types
    IngestOptions,
    IngestResult,
    # Type aliases
    OutputFormat,
    QueryError,
    QueryFilter,  # Legacy
    QueryResult,
    SchemaVersionError,
    # Result types
    SegmentHit,
    SemanticQuery,
    SpeakerQuery,
    # Exceptions
    StoreError,
    # Combined query
    StoreQuery,
    StoreStats,
    # Query filters
    TextQuery,
    TimeRange,
    TimeRangeQuery,
    TranscriptQuery,
    TranscriptSummary,
)

__all__ = [
    # Store class
    "ConversationStore",
    "SQLiteConversationStore",
    "get_default_store_path",
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
    "QueryFilter",
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
