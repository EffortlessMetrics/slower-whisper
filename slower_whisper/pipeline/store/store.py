"""SQLite-backed Conversation Store implementation.

This module provides the main ConversationStore class for:
- Storing and searching transcripts with FTS5
- Ingesting transcripts from JSON files
- Querying segments with combined filters
- Managing action items
- Exporting search results

The store uses SQLite with FTS5 for full-text search, enabling
efficient text search across large numbers of transcripts.

Example:
    >>> from transcription.store import ConversationStore, StoreQuery, TextQuery
    >>> store = ConversationStore.open("transcripts.db")
    >>> store.ingest(Path("meeting.json"))
    >>> results = store.search(StoreQuery(text=TextQuery("pricing")))
    >>> store.close()
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schema import CHECK_SCHEMA_VERSION_SQL, SCHEMA_V1, SCHEMA_VERSION
from .types import (
    ActionItem,
    ActionItemRow,
    ActionStatus,
    ConversationEntry,
    DuplicateError,
    DuplicateHandling,
    ExportError,
    ExportFormat,
    ExportOptions,
    ExportResult,
    IngestError,
    IngestOptions,
    IngestResult,
    QueryError,
    QueryFilter,
    QueryResult,
    SchemaVersionError,
    SegmentHit,
    SpeakerQuery,
    StoreError,
    StoreQuery,
    StoreStats,
    TextQuery,
    TranscriptQuery,
    TranscriptSummary,
)

if TYPE_CHECKING:
    from ..models import Transcript

logger = logging.getLogger(__name__)


def get_default_store_path() -> Path:
    """Get the default store database path.

    Returns ~/.slower-whisper/store.db, creating the directory if needed.
    """
    home = Path.home()
    store_dir = home / ".slower-whisper"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / "store.db"


class SQLiteConversationStore:
    """SQLite-backed conversation store with FTS5 search.

    This class provides a local database for storing and searching
    transcripts. Features include:

    - Full-text search using SQLite FTS5
    - Combined query filters (text, speaker, time range, etc.)
    - Action item tracking
    - Provenance receipts for audit trails
    - JSONL/JSON/CSV export

    The store is designed to be used with context managers for
    proper resource cleanup:

        >>> with ConversationStore.open("store.db") as store:
        ...     store.ingest(transcript_path)
        ...     results = store.search(query)

    Or explicitly managed:

        >>> store = ConversationStore.open("store.db")
        >>> try:
        ...     store.ingest(transcript_path)
        ... finally:
        ...     store.close()

    Attributes:
        path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize store with database path.

        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        self._path = Path(db_path) if db_path else get_default_store_path()
        self._conn: sqlite3.Connection | None = None
        # Auto-connect for backward compatibility
        self._connect()
        self._init_schema()

    @classmethod
    def open(cls, path: str | Path, create: bool = True) -> SQLiteConversationStore:
        """Open or create a conversation store database.

        Args:
            path: Path to SQLite database file.
            create: If True, create database if it doesn't exist.
                   If False, raise FileNotFoundError for missing db.

        Returns:
            Initialized SQLiteConversationStore instance.

        Raises:
            FileNotFoundError: If database doesn't exist and create=False.
            SchemaVersionError: If database has incompatible schema version.
        """
        path = Path(path)

        if not create and not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")

        return cls(path)

    def _connect(self) -> None:
        """Create database connection."""
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode, we manage transactions
        )
        # Enable foreign keys
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Enable WAL mode for better concurrency
        self._conn.execute("PRAGMA journal_mode = WAL")
        # Return rows as dictionaries
        self._conn.row_factory = sqlite3.Row

    def _init_schema(self) -> None:
        """Initialize or verify database schema."""
        if self._conn is None:
            raise StoreError("Store not connected")

        # Check if database has tables
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='store_meta'"
        )
        has_schema = cursor.fetchone() is not None

        if not has_schema:
            # Create fresh schema
            logger.info(f"Creating schema v{SCHEMA_VERSION} in {self._path}")
            self._conn.executescript(SCHEMA_V1)
        else:
            # Verify schema version
            cursor = self._conn.execute(CHECK_SCHEMA_VERSION_SQL)
            row = cursor.fetchone()
            if row:
                found_version = int(row["value"])
                if found_version != SCHEMA_VERSION:
                    raise SchemaVersionError(found_version, SCHEMA_VERSION)

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection (legacy compatibility)."""
        if self._conn is None:
            self._connect()
        assert self._conn is not None  # Guaranteed by _connect() above
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions.

        Yields:
            Database cursor within a transaction.

        Raises:
            StoreError: If store is not connected.
        """
        if self._conn is None:
            raise StoreError("Store not connected")

        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN")
            yield cursor
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    @property
    def path(self) -> Path:
        """Return database path."""
        return self._path

    @property
    def db_path(self) -> Path:
        """Return database path (legacy alias)."""
        return self._path

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SQLiteConversationStore:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    # =========================================================================
    # Ingestion
    # =========================================================================

    def ingest(
        self,
        transcript_json: Path | str,
        options: IngestOptions | None = None,
    ) -> IngestResult:
        """Ingest a transcript JSON file into the store.

        Parses the transcript JSON and inserts all segments, speakers,
        turns, and annotations into the database. Creates a provenance
        receipt for tracking.

        Args:
            transcript_json: Path to transcript JSON file.
            options: Ingestion options (defaults to IngestOptions()).

        Returns:
            IngestResult with operation summary.

        Raises:
            FileNotFoundError: If transcript file doesn't exist.
            DuplicateError: If transcript already exists and skip_duplicates=False.
            IngestError: If transcript format is invalid.
        """
        start_time = time.perf_counter_ns()
        options = options or IngestOptions()
        transcript_path = Path(transcript_json)

        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        # Generate receipt ID
        receipt_id = str(uuid.uuid4())

        # Load transcript JSON
        try:
            with open(transcript_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise IngestError(f"Invalid JSON in {transcript_path}: {e}") from e

        # Generate or derive transcript ID
        if options.generate_transcript_id:
            transcript_id = str(uuid.uuid4())
        else:
            # Hash the file name for deterministic ID
            file_name = data.get("file", data.get("file_name", str(transcript_path)))
            transcript_id = hashlib.sha256(file_name.encode()).hexdigest()[:16]

        # Check for duplicates
        if self._transcript_exists(transcript_id):
            if options.skip_duplicates:
                duration_ms = (time.perf_counter_ns() - start_time) // 1_000_000
                return IngestResult(
                    transcript_id=transcript_id,
                    status="skipped",
                    receipt_id=receipt_id,
                    duration_ms=duration_ms,
                )
            else:
                raise DuplicateError(transcript_id)

        # Extract data from JSON
        file_name = data.get("file", data.get("file_name", ""))
        language = data.get("language", "")
        segments = data.get("segments", [])
        speakers = data.get("speakers", [])
        turns = data.get("turns", [])
        annotations = data.get("annotations", {})
        meta = data.get("meta", {})
        schema_version = data.get("schema_version", 1)

        # Calculate summary stats
        word_count = sum(len(s.get("text", "").split()) for s in segments)
        duration_seconds = max((s.get("end", 0) for s in segments), default=0)
        speaker_ids = set()
        for seg in segments:
            speaker = seg.get("speaker")
            if isinstance(speaker, dict):
                sid = speaker.get("id")
                if sid:
                    speaker_ids.add(sid)
            elif speaker:
                speaker_ids.add(speaker)

        try:
            with self._transaction() as cursor:
                # Insert transcript record
                cursor.execute(
                    """
                    INSERT INTO transcripts (
                        transcript_id, file_name, language, duration_seconds,
                        word_count, segment_count, speaker_count,
                        meta_json, annotations_json, source_path, source_schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transcript_id,
                        file_name,
                        language,
                        duration_seconds,
                        word_count,
                        len(segments),
                        len(speaker_ids),
                        json.dumps(meta) if meta else None,
                        json.dumps(annotations) if annotations else None,
                        options.source_path or str(transcript_path),
                        schema_version,
                    ),
                )

                # Insert segments
                for seg in segments:
                    speaker = seg.get("speaker")
                    speaker_id = None
                    speaker_confidence = None
                    if isinstance(speaker, dict):
                        speaker_id = speaker.get("id")
                        speaker_confidence = speaker.get("confidence")
                    elif speaker:
                        speaker_id = speaker

                    audio_state = seg.get("audio_state")
                    audio_state_json = (
                        json.dumps(audio_state)
                        if audio_state and options.store_audio_state
                        else None
                    )

                    cursor.execute(
                        """
                        INSERT INTO segments (
                            transcript_id, segment_index, start_time, end_time,
                            text, speaker_id, speaker_confidence, tone, audio_state_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            transcript_id,
                            seg.get("id", 0),
                            seg.get("start", 0.0),
                            seg.get("end", 0.0),
                            seg.get("text", ""),
                            speaker_id,
                            speaker_confidence,
                            seg.get("tone"),
                            audio_state_json,
                        ),
                    )

                    segment_db_id = cursor.lastrowid

                    # Insert words if present and requested
                    if options.store_words:
                        words = seg.get("words", [])
                        for word_idx, word in enumerate(words):
                            cursor.execute(
                                """
                                INSERT INTO words (
                                    segment_id, word_index, word, start_time, end_time,
                                    probability, speaker_id
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    segment_db_id,
                                    word_idx,
                                    word.get("word", ""),
                                    word.get("start", 0.0),
                                    word.get("end", 0.0),
                                    word.get("probability"),
                                    word.get("speaker"),
                                ),
                            )

                # Insert speakers
                for speaker in speakers:
                    cursor.execute(
                        """
                        INSERT INTO speakers (
                            transcript_id, speaker_id, label, total_speech_time, num_segments
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            transcript_id,
                            speaker.get("id", ""),
                            speaker.get("label"),
                            speaker.get("total_speech_time"),
                            speaker.get("num_segments"),
                        ),
                    )

                # Insert turns
                for turn in turns:
                    cursor.execute(
                        """
                        INSERT INTO turns (
                            transcript_id, turn_id, speaker_id, start_time, end_time,
                            text, segment_ids_json, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            transcript_id,
                            turn.get("id", ""),
                            turn.get("speaker_id", turn.get("speaker", "")),
                            turn.get("start", 0.0),
                            turn.get("end", 0.0),
                            turn.get("text", ""),
                            json.dumps(turn.get("segment_ids", [])),
                            json.dumps(turn.get("metadata", {})),
                        ),
                    )

                # Extract and insert action items from annotations
                action_item_count = 0
                if options.extract_annotations and annotations:
                    semantic = annotations.get("semantic", {})
                    actions = semantic.get("actions", [])
                    for action in actions:
                        cursor.execute(
                            """
                            INSERT INTO action_items (
                                transcript_id, text, assignee, priority, confidence,
                                segment_ids_json, pattern
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                transcript_id,
                                action.get("text", ""),
                                action.get("speaker_id"),
                                action.get("priority"),
                                action.get("confidence", 1.0),
                                json.dumps(action.get("segment_ids", [])),
                                action.get("pattern"),
                            ),
                        )
                        action_item_count += 1

                # Create receipt
                duration_ms = (time.perf_counter_ns() - start_time) // 1_000_000
                cursor.execute(
                    """
                    INSERT INTO receipts (
                        receipt_id, transcript_id, operation, status, source,
                        options_json, result_json, duration_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt_id,
                        transcript_id,
                        "ingest",
                        "success",
                        str(transcript_path),
                        json.dumps(
                            {
                                "skip_duplicates": options.skip_duplicates,
                                "store_words": options.store_words,
                                "store_audio_state": options.store_audio_state,
                            }
                        ),
                        json.dumps(
                            {
                                "segment_count": len(segments),
                                "word_count": word_count,
                                "speaker_count": len(speaker_ids),
                                "action_item_count": action_item_count,
                            }
                        ),
                        duration_ms,
                    ),
                )

        except sqlite3.Error as e:
            raise IngestError(f"Database error during ingest: {e}") from e

        return IngestResult(
            transcript_id=transcript_id,
            status="success",
            receipt_id=receipt_id,
            segment_count=len(segments),
            word_count=word_count,
            speaker_count=len(speaker_ids),
            action_item_count=action_item_count,
            duration_ms=duration_ms,
        )

    def ingest_transcript(
        self,
        transcript: Transcript,
        options: IngestOptions | None = None,
    ) -> IngestResult:
        """Ingest a Transcript object directly into the store.

        Alternative to ingest() that accepts an in-memory Transcript
        object instead of reading from a JSON file.

        Args:
            transcript: Transcript object to ingest.
            options: Ingestion options.

        Returns:
            IngestResult with operation summary.
        """
        start_time = time.perf_counter_ns()
        options = options or IngestOptions()
        receipt_id = str(uuid.uuid4())

        # Generate or derive transcript ID
        if options.generate_transcript_id:
            transcript_id = str(uuid.uuid4())
        else:
            transcript_id = hashlib.sha256(transcript.file_name.encode()).hexdigest()[:16]

        # Check for duplicates
        if self._transcript_exists(transcript_id):
            if options.skip_duplicates:
                duration_ms = (time.perf_counter_ns() - start_time) // 1_000_000
                return IngestResult(
                    transcript_id=transcript_id,
                    status="skipped",
                    receipt_id=receipt_id,
                    duration_ms=duration_ms,
                )
            else:
                raise DuplicateError(transcript_id)

        # Calculate summary stats
        word_count = transcript.word_count()
        speaker_ids = set(transcript.speaker_ids())

        try:
            with self._transaction() as cursor:
                # Insert transcript record
                cursor.execute(
                    """
                    INSERT INTO transcripts (
                        transcript_id, file_name, language, duration_seconds,
                        word_count, segment_count, speaker_count,
                        meta_json, annotations_json, source_path, source_schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transcript_id,
                        transcript.file_name,
                        transcript.language,
                        transcript.duration,
                        word_count,
                        len(transcript.segments),
                        len(speaker_ids),
                        json.dumps(transcript.meta) if transcript.meta else None,
                        json.dumps(transcript.annotations) if transcript.annotations else None,
                        options.source_path,
                        2,  # Transcript objects are schema v2
                    ),
                )

                # Insert segments
                for seg in transcript.segments:
                    speaker_id = None
                    speaker_confidence = None
                    if seg.speaker is not None:
                        speaker_id = seg.speaker.get("id")
                        speaker_confidence = seg.speaker.get("confidence")

                    audio_state_json = (
                        json.dumps(seg.audio_state)
                        if seg.audio_state and options.store_audio_state
                        else None
                    )

                    cursor.execute(
                        """
                        INSERT INTO segments (
                            transcript_id, segment_index, start_time, end_time,
                            text, speaker_id, speaker_confidence, tone, audio_state_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            transcript_id,
                            seg.id,
                            seg.start,
                            seg.end,
                            seg.text,
                            speaker_id,
                            speaker_confidence,
                            seg.tone,
                            audio_state_json,
                        ),
                    )

                    segment_db_id = cursor.lastrowid

                    # Insert words if present
                    if options.store_words and seg.words:
                        for word_idx, word in enumerate(seg.words):
                            cursor.execute(
                                """
                                INSERT INTO words (
                                    segment_id, word_index, word, start_time, end_time,
                                    probability, speaker_id
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    segment_db_id,
                                    word_idx,
                                    word.word,
                                    word.start,
                                    word.end,
                                    word.probability,
                                    word.speaker,
                                ),
                            )

                # Insert speakers
                if transcript.speakers:
                    for speaker in transcript.speakers:
                        cursor.execute(
                            """
                            INSERT INTO speakers (
                                transcript_id, speaker_id, label, total_speech_time, num_segments
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                transcript_id,
                                speaker.get("id", ""),
                                speaker.get("label"),
                                speaker.get("total_speech_time"),
                                speaker.get("num_segments"),
                            ),
                        )

                # Insert turns
                if transcript.turns:
                    for turn in transcript.turns:
                        if hasattr(turn, "to_dict"):
                            turn_dict = turn.to_dict()
                        else:
                            turn_dict = turn

                        cursor.execute(
                            """
                            INSERT INTO turns (
                                transcript_id, turn_id, speaker_id, start_time, end_time,
                                text, segment_ids_json, metadata_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                transcript_id,
                                turn_dict.get("id", ""),
                                turn_dict.get("speaker_id", ""),
                                turn_dict.get("start", 0.0),
                                turn_dict.get("end", 0.0),
                                turn_dict.get("text", ""),
                                json.dumps(turn_dict.get("segment_ids", [])),
                                json.dumps(turn_dict.get("metadata", {})),
                            ),
                        )

                # Extract action items from annotations
                action_item_count = 0
                if options.extract_annotations and transcript.annotations:
                    semantic = transcript.annotations.get("semantic", {})
                    actions = semantic.get("actions", [])
                    for action in actions:
                        cursor.execute(
                            """
                            INSERT INTO action_items (
                                transcript_id, text, assignee, priority, confidence,
                                segment_ids_json, pattern
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                transcript_id,
                                action.get("text", ""),
                                action.get("speaker_id"),
                                action.get("priority"),
                                action.get("confidence", 1.0),
                                json.dumps(action.get("segment_ids", [])),
                                action.get("pattern"),
                            ),
                        )
                        action_item_count += 1

                # Create receipt
                duration_ms = (time.perf_counter_ns() - start_time) // 1_000_000
                cursor.execute(
                    """
                    INSERT INTO receipts (
                        receipt_id, transcript_id, operation, status, source,
                        duration_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt_id,
                        transcript_id,
                        "ingest",
                        "success",
                        transcript.file_name,
                        duration_ms,
                    ),
                )

        except sqlite3.Error as e:
            raise IngestError(f"Database error during ingest: {e}") from e

        return IngestResult(
            transcript_id=transcript_id,
            status="success",
            receipt_id=receipt_id,
            segment_count=len(transcript.segments),
            word_count=word_count,
            speaker_count=len(speaker_ids),
            action_item_count=action_item_count,
            duration_ms=duration_ms,
        )

    def ingest_file(
        self,
        json_path: Path,
        tags: list[str] | None = None,
        on_duplicate: DuplicateHandling = DuplicateHandling.SKIP,
    ) -> int:
        """Ingest a transcript JSON file into the store (legacy API).

        This is the legacy ingestion method for backward compatibility.
        New code should use ingest() instead.

        Args:
            json_path: Path to the transcript JSON file
            tags: Optional tags to apply (not used in new schema)
            on_duplicate: How to handle duplicate entries

        Returns:
            Number of segments ingested

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If on_duplicate is ERROR and duplicates exist
        """
        options = IngestOptions(
            skip_duplicates=(on_duplicate == DuplicateHandling.SKIP),
            generate_transcript_id=False,  # Use deterministic IDs for legacy
        )

        try:
            result = self.ingest(json_path, options)
            return result.segment_count
        except DuplicateError as e:
            if on_duplicate == DuplicateHandling.ERROR:
                raise ValueError(f"Entries already exist for source file: {json_path.name}") from e
            return 0

    def _transcript_exists(self, transcript_id: str) -> bool:
        """Check if a transcript exists in the store."""
        if self._conn is None:
            raise StoreError("Store not connected")

        cursor = self._conn.execute(
            "SELECT 1 FROM transcripts WHERE transcript_id = ?",
            (transcript_id,),
        )
        return cursor.fetchone() is not None

    # =========================================================================
    # Search
    # =========================================================================

    def search(self, query: StoreQuery) -> list[SegmentHit]:
        """Search for segments matching the query.

        Combines full-text search with filter predicates to find
        matching segments. Results include FTS5 ranking and snippets.

        Args:
            query: Combined query with filters.

        Returns:
            List of SegmentHit results.

        Raises:
            QueryError: If query is invalid.
        """
        if self._conn is None:
            raise StoreError("Store not connected")

        # Build SQL query
        sql_parts = []
        params: list[Any] = []

        # Base query depends on whether we have text search
        if query.text and query.text.text.strip():
            fts_query = query.text.to_fts_query()
            sql_parts.append(
                """
                SELECT
                    s.id as segment_id,
                    s.transcript_id,
                    s.segment_index,
                    s.start_time,
                    s.end_time,
                    s.text,
                    s.speaker_id,
                    s.speaker_confidence,
                    t.file_name,
                    t.language,
                    bm25(segments_fts) as rank,
                    snippet(segments_fts, 0, '<mark>', '</mark>', '...', 32) as snippet,
                    s.audio_state_json
                FROM segments_fts
                JOIN segments s ON segments_fts.rowid = s.id
                JOIN transcripts t ON s.transcript_id = t.transcript_id
                WHERE segments_fts MATCH ?
                """
            )
            params.append(fts_query)
        else:
            sql_parts.append(
                """
                SELECT
                    s.id as segment_id,
                    s.transcript_id,
                    s.segment_index,
                    s.start_time,
                    s.end_time,
                    s.text,
                    s.speaker_id,
                    s.speaker_confidence,
                    t.file_name,
                    t.language,
                    0 as rank,
                    s.text as snippet,
                    s.audio_state_json
                FROM segments s
                JOIN transcripts t ON s.transcript_id = t.transcript_id
                WHERE 1=1
                """
            )

        # Add filter predicates
        if query.time_range:
            if query.time_range.start is not None:
                sql_parts.append("AND s.start_time >= ?")
                params.append(query.time_range.start)
            if query.time_range.end is not None:
                sql_parts.append("AND s.end_time <= ?")
                params.append(query.time_range.end)

        if query.speakers and query.speakers.speaker_ids:
            placeholders = ",".join("?" * len(query.speakers.speaker_ids))
            if query.speakers.exclude:
                sql_parts.append(f"AND s.speaker_id NOT IN ({placeholders})")
            else:
                sql_parts.append(f"AND s.speaker_id IN ({placeholders})")
            params.extend(query.speakers.speaker_ids)

        if query.transcripts:
            if query.transcripts.transcript_ids:
                placeholders = ",".join("?" * len(query.transcripts.transcript_ids))
                sql_parts.append(f"AND s.transcript_id IN ({placeholders})")
                params.extend(query.transcripts.transcript_ids)
            if query.transcripts.file_names:
                placeholders = ",".join("?" * len(query.transcripts.file_names))
                sql_parts.append(f"AND t.file_name IN ({placeholders})")
                params.extend(query.transcripts.file_names)
            if query.transcripts.languages:
                placeholders = ",".join("?" * len(query.transcripts.languages))
                sql_parts.append(f"AND t.language IN ({placeholders})")
                params.extend(query.transcripts.languages)

        if query.date_range:
            if query.date_range.after:
                sql_parts.append("AND t.ingested_at > ?")
                params.append(query.date_range.after)
            if query.date_range.before:
                sql_parts.append("AND t.ingested_at < ?")
                params.append(query.date_range.before)

        # Order by
        order_col = "rank" if query.text else query.order_by
        order_dir = "DESC" if query.order_desc else "ASC"
        sql_parts.append(f"ORDER BY {order_col} {order_dir}")

        # Limit and offset
        sql_parts.append("LIMIT ? OFFSET ?")
        params.extend([query.limit, query.offset])

        # Execute query
        sql = "\n".join(sql_parts)
        try:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            raise QueryError(f"Search query failed: {e}") from e

        # Convert to SegmentHit
        results: list[SegmentHit] = []
        for row in rows:
            audio_state = None
            if row["audio_state_json"]:
                try:
                    audio_state = json.loads(row["audio_state_json"])
                except json.JSONDecodeError:
                    pass

            hit: SegmentHit = {
                "segment_id": row["segment_id"],
                "transcript_id": row["transcript_id"],
                "segment_index": row["segment_index"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "text": row["text"],
                "speaker_id": row["speaker_id"],
                "speaker_confidence": row["speaker_confidence"],
                "file_name": row["file_name"],
                "language": row["language"],
                "rank": row["rank"],
                "snippet": row["snippet"],
                "audio_state": audio_state,
            }
            results.append(hit)

        return results

    def query(self, filter: QueryFilter) -> QueryResult:
        """Query the store for matching entries (legacy API).

        This is the legacy query method for backward compatibility.
        New code should use search() instead.

        Args:
            filter: Query filter criteria

        Returns:
            QueryResult with matching entries
        """
        start_time_ns = time.perf_counter_ns()

        # Convert legacy filter to StoreQuery
        store_query = StoreQuery(limit=filter.limit)

        if filter.text:
            store_query.text = TextQuery(text=filter.text, match_type="phrase")

        if filter.speaker_id:
            store_query.speakers = SpeakerQuery(speaker_ids=[filter.speaker_id])

        if filter.source_file:
            store_query.transcripts = TranscriptQuery(file_names=[filter.source_file])

        # Execute search
        hits = self.search(store_query)

        # Convert to legacy ConversationEntry format
        entries = []
        for hit in hits:
            entries.append(
                ConversationEntry(
                    id=f"{hit['transcript_id']}:{hit['segment_index']}",
                    text=hit["text"],
                    speaker_id=hit.get("speaker_id"),
                    start_time=hit["start_time"],
                    end_time=hit["end_time"],
                    source_file=hit["file_name"],
                    ingested_at=datetime.now(),  # Not available in new schema
                    tags=[],
                    topics=[],
                    metadata={},
                )
            )

        query_time_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000

        return QueryResult(
            entries=entries,
            total_count=len(entries),
            query_time_ms=query_time_ms,
        )

    def list_entries(self, limit: int = 10) -> QueryResult:
        """List recent entries in the store (legacy API).

        Args:
            limit: Maximum number of entries to return

        Returns:
            QueryResult with entries
        """
        return self.query(QueryFilter(limit=limit))

    # =========================================================================
    # Transcript Management
    # =========================================================================

    def list_transcripts(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TranscriptSummary]:
        """List all transcripts in the store.

        Args:
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of TranscriptSummary.
        """
        if self._conn is None:
            raise StoreError("Store not connected")

        cursor = self._conn.execute(
            """
            SELECT
                t.transcript_id,
                t.file_name,
                t.language,
                t.duration_seconds,
                t.word_count,
                t.segment_count,
                t.speaker_count,
                t.ingested_at,
                t.source_path,
                (SELECT COUNT(*) FROM annotations a WHERE a.transcript_id = t.transcript_id) > 0 as has_annotations,
                (SELECT COUNT(*) FROM action_items ai WHERE ai.transcript_id = t.transcript_id) as action_item_count
            FROM transcripts t
            ORDER BY t.ingested_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        results: list[TranscriptSummary] = []
        for row in cursor.fetchall():
            # Get speaker IDs
            speaker_cursor = self._conn.execute(
                "SELECT speaker_id FROM speakers WHERE transcript_id = ?",
                (row["transcript_id"],),
            )
            speakers = [r["speaker_id"] for r in speaker_cursor.fetchall()]

            summary: TranscriptSummary = {
                "transcript_id": row["transcript_id"],
                "file_name": row["file_name"],
                "language": row["language"],
                "duration_seconds": row["duration_seconds"],
                "word_count": row["word_count"],
                "segment_count": row["segment_count"],
                "speaker_count": row["speaker_count"],
                "speakers": speakers,
                "ingested_at": row["ingested_at"],
                "source_path": row["source_path"],
                "has_annotations": bool(row["has_annotations"]),
                "action_item_count": row["action_item_count"],
            }
            results.append(summary)

        return results

    def get_transcript(self, transcript_id: str) -> TranscriptSummary | None:
        """Get a transcript summary by ID.

        Args:
            transcript_id: Transcript ID.

        Returns:
            TranscriptSummary or None if not found.
        """
        if self._conn is None:
            raise StoreError("Store not connected")

        cursor = self._conn.execute(
            """
            SELECT
                t.transcript_id,
                t.file_name,
                t.language,
                t.duration_seconds,
                t.word_count,
                t.segment_count,
                t.speaker_count,
                t.ingested_at,
                t.source_path,
                (SELECT COUNT(*) FROM annotations a WHERE a.transcript_id = t.transcript_id) > 0 as has_annotations,
                (SELECT COUNT(*) FROM action_items ai WHERE ai.transcript_id = t.transcript_id) as action_item_count
            FROM transcripts t
            WHERE t.transcript_id = ?
            """,
            (transcript_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Get speaker IDs
        speaker_cursor = self._conn.execute(
            "SELECT speaker_id FROM speakers WHERE transcript_id = ?",
            (transcript_id,),
        )
        speakers = [r["speaker_id"] for r in speaker_cursor.fetchall()]

        return {
            "transcript_id": row["transcript_id"],
            "file_name": row["file_name"],
            "language": row["language"],
            "duration_seconds": row["duration_seconds"],
            "word_count": row["word_count"],
            "segment_count": row["segment_count"],
            "speaker_count": row["speaker_count"],
            "speakers": speakers,
            "ingested_at": row["ingested_at"],
            "source_path": row["source_path"],
            "has_annotations": bool(row["has_annotations"]),
            "action_item_count": row["action_item_count"],
        }

    def delete_transcript(self, transcript_id: str) -> bool:
        """Delete a transcript and all associated data.

        Args:
            transcript_id: Transcript ID to delete.

        Returns:
            True if transcript was deleted, False if not found.
        """
        if self._conn is None:
            raise StoreError("Store not connected")

        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM transcripts WHERE transcript_id = ?",
                (transcript_id,),
            )
            return cursor.rowcount > 0

    # =========================================================================
    # Action Items
    # =========================================================================

    def get_action_items(
        self,
        transcript_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ActionItemRow]:
        """Get action items with optional filters.

        Args:
            transcript_id: Filter by transcript ID.
            status: Filter by status ('open', 'completed', 'cancelled').
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of ActionItemRow.
        """
        if self._conn is None:
            raise StoreError("Store not connected")

        sql_parts = [
            """
            SELECT
                ai.id as action_id,
                ai.transcript_id,
                ai.text,
                ai.assignee,
                ai.due,
                ai.priority,
                ai.confidence,
                ai.segment_ids_json,
                ai.pattern,
                ai.status,
                ai.created_at,
                ai.updated_at,
                t.file_name
            FROM action_items ai
            JOIN transcripts t ON ai.transcript_id = t.transcript_id
            WHERE 1=1
            """
        ]
        params: list[Any] = []

        if transcript_id:
            sql_parts.append("AND ai.transcript_id = ?")
            params.append(transcript_id)

        if status:
            sql_parts.append("AND ai.status = ?")
            params.append(status)

        sql_parts.append("ORDER BY ai.created_at DESC LIMIT ? OFFSET ?")
        params.extend([limit, offset])

        cursor = self._conn.execute("\n".join(sql_parts), params)

        results: list[ActionItemRow] = []
        for row in cursor.fetchall():
            segment_ids = []
            if row["segment_ids_json"]:
                try:
                    segment_ids = json.loads(row["segment_ids_json"])
                except json.JSONDecodeError:
                    pass

            item: ActionItemRow = {
                "action_id": row["action_id"],
                "transcript_id": row["transcript_id"],
                "text": row["text"],
                "assignee": row["assignee"],
                "due": row["due"],
                "priority": row["priority"],
                "confidence": row["confidence"],
                "segment_ids": segment_ids,
                "pattern": row["pattern"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "file_name": row["file_name"],
            }
            results.append(item)

        return results

    def update_action_item_status(
        self,
        action_id: int,
        status: str,
    ) -> bool:
        """Update the status of an action item.

        Args:
            action_id: Action item database ID.
            status: New status ('open', 'completed', 'cancelled').

        Returns:
            True if action was updated, False if not found.
        """
        if self._conn is None:
            raise StoreError("Store not connected")

        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE action_items
                SET status = ?, updated_at = datetime('now')
                WHERE id = ?
                """,
                (status, action_id),
            )
            return cursor.rowcount > 0

    def list_actions(self, status: ActionStatus | None = None, limit: int = 50) -> list[ActionItem]:
        """List action items (legacy API).

        Args:
            status: Filter by status (None for all)
            limit: Maximum number of items to return

        Returns:
            List of ActionItem objects
        """
        status_str = status.value if status else None
        rows = self.get_action_items(status=status_str, limit=limit)

        actions = []
        for row in rows:
            actions.append(
                ActionItem(
                    id=str(row["action_id"]),
                    text=row["text"],
                    status=ActionStatus(row["status"]),
                    source_entry_id=f"{row['transcript_id']}:{row['segment_ids'][0] if row['segment_ids'] else 0}",
                    source_file=row["file_name"],
                    created_at=datetime.fromisoformat(row["created_at"])
                    if row["created_at"]
                    else datetime.now(),
                    completed_at=None,
                    assignee=row["assignee"],
                    due_date=None,
                    metadata={},
                )
            )

        return actions

    def complete_action(self, action_id: str) -> bool:
        """Mark an action item as completed (legacy API).

        Args:
            action_id: ID of the action item to complete

        Returns:
            True if the action was found and updated, False otherwise
        """
        try:
            return self.update_action_item_status(int(action_id), "completed")
        except ValueError:
            return False

    # =========================================================================
    # Export
    # =========================================================================

    def export(
        self,
        query: StoreQuery | QueryFilter | None = None,
        options: ExportOptions | None = None,
        format: str = "jsonl",
    ) -> ExportResult | list[dict[str, Any]]:
        """Export search results to file or string.

        Args:
            query: Query to filter results (StoreQuery or legacy QueryFilter).
            options: Export options.
            format: Output format hint for legacy API (jsonl, json, csv).

        Returns:
            ExportResult with data or file path, or list of dicts for legacy API.
        """
        start_time = time.perf_counter_ns()
        options = options or ExportOptions()

        # Handle legacy QueryFilter
        if isinstance(query, QueryFilter) or query is None:
            if query is None:
                query = QueryFilter(limit=10000)
            result = self.query(query)
            return [entry.to_dict() for entry in result.entries]

        # New API with StoreQuery
        results = self.search(query)

        # Format output
        if options.format == ExportFormat.JSONL:
            lines = []
            for hit in results:
                # Remove audio_state if not requested
                hit_dict = dict(hit)
                if not options.include_audio_state:
                    hit_dict.pop("audio_state", None)
                lines.append(json.dumps(hit_dict))
            data = "\n".join(lines)

        elif options.format == ExportFormat.JSON:
            output = []
            for hit in results:
                hit_dict = dict(hit)
                if not options.include_audio_state:
                    hit_dict.pop("audio_state", None)
                output.append(hit_dict)
            if options.pretty_print:
                data = json.dumps(output, indent=2)
            else:
                data = json.dumps(output)

        elif options.format == ExportFormat.CSV:
            output_io = io.StringIO()
            if results:
                fieldnames = [
                    "transcript_id",
                    "segment_index",
                    "start_time",
                    "end_time",
                    "text",
                    "speaker_id",
                    "file_name",
                ]
                writer = csv.DictWriter(output_io, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for hit in results:
                    writer.writerow(dict(hit))
            data = output_io.getvalue()

        else:
            raise StoreError(f"Unsupported format: {options.format}")

        duration_ms = (time.perf_counter_ns() - start_time) // 1_000_000

        # Write to file if path specified
        if options.output_path:
            Path(options.output_path).write_text(data, encoding="utf-8")
            return ExportResult(
                status="success",
                record_count=len(results),
                output_path=options.output_path,
                duration_ms=duration_ms,
            )
        else:
            return ExportResult(
                status="success",
                record_count=len(results),
                data=data,
                duration_ms=duration_ms,
            )

    def export_parquet(
        self,
        output_path: str,
        query: StoreQuery | None = None,
    ) -> ExportResult:
        """Export search results to Parquet format.

        This method requires pyarrow to be installed. Parquet is an efficient
        columnar format suitable for data analysis workflows.

        Args:
            output_path: Path to write the Parquet file.
            query: Query to filter results (defaults to all segments).

        Returns:
            ExportResult with operation status.

        Raises:
            ImportError: If pyarrow is not installed.
            ExportError: If export fails.
        """
        start_time = time.perf_counter_ns()

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for Parquet export. Install with: pip install pyarrow"
            ) from e

        # Execute query
        query = query or StoreQuery(limit=100000)  # Large default limit
        results = self.search(query)

        if not results:
            # Create empty parquet with schema
            schema = pa.schema(
                [
                    ("segment_id", pa.int64()),
                    ("transcript_id", pa.string()),
                    ("segment_index", pa.int64()),
                    ("start_time", pa.float64()),
                    ("end_time", pa.float64()),
                    ("text", pa.string()),
                    ("speaker_id", pa.string()),
                    ("speaker_confidence", pa.float64()),
                    ("file_name", pa.string()),
                    ("language", pa.string()),
                    ("rank", pa.float64()),
                    ("snippet", pa.string()),
                ]
            )
            table = pa.Table.from_pydict({}, schema=schema)
        else:
            # Build columns
            columns: dict[str, list[Any]] = {
                "segment_id": [],
                "transcript_id": [],
                "segment_index": [],
                "start_time": [],
                "end_time": [],
                "text": [],
                "speaker_id": [],
                "speaker_confidence": [],
                "file_name": [],
                "language": [],
                "rank": [],
                "snippet": [],
            }

            for hit in results:
                columns["segment_id"].append(hit.get("segment_id"))
                columns["transcript_id"].append(hit.get("transcript_id"))
                columns["segment_index"].append(hit.get("segment_index"))
                columns["start_time"].append(hit.get("start_time"))
                columns["end_time"].append(hit.get("end_time"))
                columns["text"].append(hit.get("text"))
                columns["speaker_id"].append(hit.get("speaker_id"))
                columns["speaker_confidence"].append(hit.get("speaker_confidence"))
                columns["file_name"].append(hit.get("file_name"))
                columns["language"].append(hit.get("language"))
                columns["rank"].append(hit.get("rank"))
                columns["snippet"].append(hit.get("snippet"))

            table = pa.Table.from_pydict(columns)

        try:
            pq.write_table(table, output_path, compression="snappy")
        except Exception as e:
            raise ExportError(f"Failed to write Parquet file: {e}") from e

        duration_ms = (time.perf_counter_ns() - start_time) // 1_000_000

        return ExportResult(
            status="success",
            record_count=len(results),
            output_path=output_path,
            duration_ms=duration_ms,
        )

    def export_jsonl(self, output_path: str, query: StoreQuery | None = None) -> ExportResult:
        """Export search results to JSONL format.

        Convenience method for JSONL export.

        Args:
            output_path: Path to write the JSONL file.
            query: Query to filter results.

        Returns:
            ExportResult with operation status.
        """
        options = ExportOptions(format=ExportFormat.JSONL, output_path=output_path)
        query = query or StoreQuery(limit=100000)
        result = self.export(query, options=options)
        if isinstance(result, list):
            # Legacy API fallback
            return ExportResult(status="success", record_count=len(result))
        return result

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> StoreStats:
        """Get store statistics.

        Returns:
            StoreStats with aggregate counts and metrics.
        """
        if self._conn is None:
            raise StoreError("Store not connected")

        # Get counts
        cursor = self._conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM transcripts) as transcript_count,
                (SELECT COUNT(*) FROM segments) as segment_count,
                (SELECT COALESCE(SUM(word_count), 0) FROM transcripts) as word_count,
                (SELECT COUNT(DISTINCT speaker_id) FROM speakers) as speaker_count,
                (SELECT COUNT(*) FROM action_items) as action_item_count,
                (SELECT COUNT(*) FROM action_items WHERE status = 'open') as open_action_items,
                (SELECT COUNT(*) FROM action_items WHERE status = 'completed') as completed_action_items,
                (SELECT COALESCE(SUM(duration_seconds), 0) FROM transcripts) as total_duration,
                (SELECT MIN(ingested_at) FROM transcripts) as oldest_ingestion,
                (SELECT MAX(ingested_at) FROM transcripts) as newest_ingestion
            """
        )
        row = cursor.fetchone()

        # Get database size
        db_size = 0
        if self._path.exists():
            db_size = os.path.getsize(self._path)

        return {
            "transcript_count": row["transcript_count"],
            "segment_count": row["segment_count"],
            "word_count": row["word_count"],
            "speaker_count": row["speaker_count"],
            "action_item_count": row["action_item_count"],
            "open_action_items": row["open_action_items"],
            "completed_action_items": row["completed_action_items"],
            "total_duration_seconds": row["total_duration"],
            "database_size_bytes": db_size,
            "oldest_ingestion": row["oldest_ingestion"],
            "newest_ingestion": row["newest_ingestion"],
        }

    def get_stats(self) -> StoreStats:
        """Get statistics about the store (legacy alias).

        Returns:
            StoreStats with aggregate information
        """
        return self.stats()

    # =========================================================================
    # Maintenance
    # =========================================================================

    def vacuum(self) -> None:
        """Compact the database and reclaim unused space.

        This should be called periodically after many deletions.
        """
        if self._conn is None:
            raise StoreError("Store not connected")

        self._conn.execute("VACUUM")


# Alias for backward compatibility
ConversationStore = SQLiteConversationStore

__all__ = [
    "SQLiteConversationStore",
    "ConversationStore",
    "get_default_store_path",
]
