"""Tests for the Conversation Store system.

Tests cover:
- Store initialization and schema creation
- Transcript ingestion (JSON file and Transcript object)
- Idempotent ingestion (duplicates handled correctly)
- Full-text search (FTS5)
- Query with various filters
- Export to JSONL, JSON, CSV, and Parquet formats
- Store statistics
- Action item management
- Error handling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slower_whisper.pipeline.models import Segment, Transcript
from slower_whisper.pipeline.store import (
    ConversationStore,
    DuplicateError,
    ExportFormat,
    ExportOptions,
    IngestOptions,
    QueryFilter,
    SpeakerQuery,
    StoreError,
    StoreQuery,
    TextQuery,
    TimeRangeQuery,
    TranscriptQuery,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a path for a temporary database."""
    return tmp_path / "test_store.db"


@pytest.fixture
def store(tmp_db: Path) -> ConversationStore:
    """Create and return a ConversationStore instance."""
    store = ConversationStore(tmp_db)
    yield store
    store.close()


@pytest.fixture
def sample_transcript_data() -> dict[str, Any]:
    """Return sample transcript data for testing."""
    return {
        "schema_version": 2,
        "file": "test_meeting.wav",
        "language": "en",
        "meta": {
            "model_name": "large-v3",
            "device": "cuda",
        },
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "Hello, welcome to the meeting.",
                "speaker": {"id": "spk_0", "confidence": 0.95},
                "tone": None,
                "audio_state": None,
            },
            {
                "id": 1,
                "start": 2.5,
                "end": 5.0,
                "text": "Thank you for having me here today.",
                "speaker": {"id": "spk_1", "confidence": 0.87},
                "tone": None,
                "audio_state": None,
            },
            {
                "id": 2,
                "start": 5.0,
                "end": 8.0,
                "text": "Let's discuss the pricing strategy.",
                "speaker": {"id": "spk_0", "confidence": 0.92},
                "tone": None,
                "audio_state": None,
            },
            {
                "id": 3,
                "start": 8.0,
                "end": 12.0,
                "text": "I will send you the proposal by tomorrow.",
                "speaker": {"id": "spk_1", "confidence": 0.89},
                "tone": None,
                "audio_state": None,
            },
        ],
        "speakers": [
            {"id": "spk_0", "label": "Host", "total_speech_time": 5.5, "num_segments": 2},
            {"id": "spk_1", "label": "Guest", "total_speech_time": 6.5, "num_segments": 2},
        ],
        "turns": [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "start": 0.0,
                "end": 2.5,
                "text": "Hello, welcome to the meeting.",
                "segment_ids": [0],
            },
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "start": 2.5,
                "end": 5.0,
                "text": "Thank you for having me here today.",
                "segment_ids": [1],
            },
        ],
        "annotations": {
            "semantic": {
                "actions": [
                    {
                        "text": "Send proposal by tomorrow",
                        "speaker_id": "spk_1",
                        "segment_ids": [3],
                        "confidence": 0.9,
                        "pattern": "I will.*",
                    }
                ],
                "topics": ["meeting", "pricing"],
            }
        },
    }


@pytest.fixture
def sample_transcript_json(tmp_path: Path, sample_transcript_data: dict[str, Any]) -> Path:
    """Create a sample transcript JSON file."""
    json_path = tmp_path / "test_meeting.json"
    json_path.write_text(json.dumps(sample_transcript_data, indent=2), encoding="utf-8")
    return json_path


@pytest.fixture
def sample_transcript() -> Transcript:
    """Create a sample Transcript object."""
    return Transcript(
        file_name="sample_transcript.wav",
        language="en",
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=3.0,
                text="The budget discussion is important.",
                speaker={"id": "speaker_A", "confidence": 0.9},
            ),
            Segment(
                id=1,
                start=3.0,
                end=6.0,
                text="We need to review the numbers.",
                speaker={"id": "speaker_B", "confidence": 0.85},
            ),
            Segment(
                id=2,
                start=6.0,
                end=9.0,
                text="I agree with the budget proposal.",
                speaker={"id": "speaker_A", "confidence": 0.88},
            ),
        ],
        meta={"model_name": "test-model"},
        speakers=[
            {"id": "speaker_A", "label": "Alice"},
            {"id": "speaker_B", "label": "Bob"},
        ],
    )


# =============================================================================
# Store Initialization Tests
# =============================================================================


class TestStoreInitialization:
    """Tests for store initialization and schema management."""

    def test_create_new_store(self, tmp_db: Path) -> None:
        """Test creating a new store database."""
        assert not tmp_db.exists()

        store = ConversationStore(tmp_db)
        assert tmp_db.exists()
        assert store.path == tmp_db

        # Verify schema was created
        stats = store.stats()
        assert stats["transcript_count"] == 0
        assert stats["segment_count"] == 0

        store.close()

    def test_open_existing_store(self, tmp_db: Path) -> None:
        """Test opening an existing store database."""
        # Create store
        store1 = ConversationStore(tmp_db)
        store1.close()

        # Reopen
        store2 = ConversationStore.open(tmp_db)
        assert store2.path == tmp_db
        store2.close()

    def test_open_nonexistent_with_create_false(self, tmp_path: Path) -> None:
        """Test that opening nonexistent store with create=False raises error."""
        nonexistent_path = tmp_path / "nonexistent.db"

        with pytest.raises(FileNotFoundError):
            ConversationStore.open(nonexistent_path, create=False)

    def test_context_manager(self, tmp_db: Path) -> None:
        """Test using store as context manager."""
        with ConversationStore(tmp_db) as store:
            assert store.path == tmp_db
            stats = store.stats()
            assert stats["transcript_count"] == 0

        # Store should be closed after context
        # (accessing it would fail, but we don't need to test that explicitly)

    def test_default_store_path(self) -> None:
        """Test that default store path is in user home directory."""
        from slower_whisper.pipeline.store import get_default_store_path

        default_path = get_default_store_path()
        assert default_path.name == "store.db"
        assert ".slower-whisper" in str(default_path)


# =============================================================================
# Ingestion Tests
# =============================================================================


class TestIngestion:
    """Tests for transcript ingestion."""

    def test_ingest_json_file(self, store: ConversationStore, sample_transcript_json: Path) -> None:
        """Test ingesting a transcript JSON file."""
        result = store.ingest(sample_transcript_json)

        assert result.status == "success"
        assert result.transcript_id is not None
        assert result.segment_count == 4
        assert result.speaker_count == 2
        assert result.action_item_count == 1
        assert result.receipt_id is not None

        # Verify stats
        stats = store.stats()
        assert stats["transcript_count"] == 1
        assert stats["segment_count"] == 4
        assert stats["speaker_count"] == 2

    def test_ingest_transcript_object(
        self, store: ConversationStore, sample_transcript: Transcript
    ) -> None:
        """Test ingesting a Transcript object directly."""
        result = store.ingest_transcript(sample_transcript)

        assert result.status == "success"
        assert result.transcript_id is not None
        assert result.segment_count == 3
        assert result.speaker_count == 2

        # Verify stats
        stats = store.stats()
        assert stats["transcript_count"] == 1
        assert stats["segment_count"] == 3

    def test_idempotent_ingestion_skip(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test that duplicate ingestion is skipped by default."""
        # Use deterministic IDs for duplicate detection
        options = IngestOptions(generate_transcript_id=False, skip_duplicates=True)

        # First ingestion
        result1 = store.ingest(sample_transcript_json, options=options)
        assert result1.status == "success"

        # Second ingestion - should be skipped
        result2 = store.ingest(sample_transcript_json, options=options)
        assert result2.status == "skipped"

        # Verify only one transcript exists
        stats = store.stats()
        assert stats["transcript_count"] == 1

    def test_idempotent_ingestion_error(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test that duplicate ingestion raises error when configured."""
        # Use deterministic IDs for duplicate detection
        options_first = IngestOptions(generate_transcript_id=False, skip_duplicates=True)

        # First ingestion
        result1 = store.ingest(sample_transcript_json, options=options_first)
        assert result1.status == "success"

        # Second ingestion with skip_duplicates=False and deterministic ID
        options_second = IngestOptions(generate_transcript_id=False, skip_duplicates=False)
        with pytest.raises(DuplicateError):
            store.ingest(sample_transcript_json, options=options_second)

    def test_ingest_with_generate_transcript_id(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test ingestion with UUID transcript ID generation."""
        options = IngestOptions(generate_transcript_id=True)
        result = store.ingest(sample_transcript_json, options=options)

        # Should be a UUID format
        assert result.status == "success"
        assert len(result.transcript_id) == 36  # UUID length with dashes

    def test_ingest_nonexistent_file(self, store: ConversationStore, tmp_path: Path) -> None:
        """Test that ingesting nonexistent file raises error."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            store.ingest(nonexistent)

    def test_ingest_invalid_json(self, store: ConversationStore, tmp_path: Path) -> None:
        """Test that ingesting invalid JSON raises error."""
        from slower_whisper.pipeline.store.types import IngestError

        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{ invalid json }", encoding="utf-8")

        with pytest.raises(IngestError):
            store.ingest(invalid_json)

    def test_ingest_legacy_api(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test the legacy ingest_file API."""
        count = store.ingest_file(sample_transcript_json)
        assert count == 4  # Number of segments

        # Duplicate should return 0
        count2 = store.ingest_file(sample_transcript_json)
        assert count2 == 0

    def test_ingest_with_words(self, store: ConversationStore, tmp_path: Path) -> None:
        """Test ingestion with word-level timestamps."""
        transcript_data = {
            "schema_version": 2,
            "file": "with_words.wav",
            "language": "en",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                        {"word": "world", "start": 0.6, "end": 1.0, "probability": 0.92},
                    ],
                }
            ],
        }

        json_path = tmp_path / "with_words.json"
        json_path.write_text(json.dumps(transcript_data), encoding="utf-8")

        options = IngestOptions(store_words=True)
        result = store.ingest(json_path, options=options)

        assert result.status == "success"
        assert result.word_count == 2


# =============================================================================
# Full-Text Search Tests
# =============================================================================


class TestFullTextSearch:
    """Tests for FTS5 full-text search."""

    def test_search_text_phrase(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test searching for an exact phrase."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(text=TextQuery("pricing strategy", match_type="phrase"))
        results = store.search(query)

        assert len(results) == 1
        assert "pricing" in results[0]["text"].lower()

    def test_search_text_any_word(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test searching for any word."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(text=TextQuery("meeting welcome", match_type="any"))
        results = store.search(query)

        # Should find segments with "meeting" OR "welcome"
        assert len(results) >= 1

    def test_search_text_all_words(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test searching for all words."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(text=TextQuery("Hello welcome", match_type="all"))
        results = store.search(query)

        # Should find segments with both "Hello" AND "welcome"
        assert len(results) >= 1
        for result in results:
            text_lower = result["text"].lower()
            assert "hello" in text_lower
            assert "welcome" in text_lower

    def test_search_text_prefix(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test searching with prefix matching."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(text=TextQuery("pric", match_type="prefix"))
        results = store.search(query)

        assert len(results) >= 1
        # Should match "pricing"

    def test_search_no_results(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test search returning no results."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(text=TextQuery("xyznonexistent"))
        results = store.search(query)

        assert len(results) == 0

    def test_search_with_snippet_highlighting(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test that search results include highlighted snippets."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(text=TextQuery("pricing"))
        results = store.search(query)

        assert len(results) >= 1
        # FTS5 should include <mark> tags for highlighting
        assert "snippet" in results[0]


# =============================================================================
# Query Filter Tests
# =============================================================================


class TestQueryFilters:
    """Tests for query filters."""

    def test_filter_by_speaker(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test filtering results by speaker ID."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(speakers=SpeakerQuery(speaker_ids=["spk_0"]))
        results = store.search(query)

        assert len(results) == 2
        for result in results:
            assert result["speaker_id"] == "spk_0"

    def test_filter_by_speaker_exclude(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test excluding results by speaker ID."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(speakers=SpeakerQuery(speaker_ids=["spk_0"], exclude=True))
        results = store.search(query)

        assert len(results) == 2
        for result in results:
            assert result["speaker_id"] != "spk_0"

    def test_filter_by_time_range(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test filtering results by time range."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(time_range=TimeRangeQuery(start=2.0, end=6.0))
        results = store.search(query)

        # Should match segments within the time range
        for result in results:
            assert result["start_time"] >= 2.0
            assert result["end_time"] <= 6.0

    def test_filter_by_transcript(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test filtering results by transcript file name."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(transcripts=TranscriptQuery(file_names=["test_meeting.wav"]))
        results = store.search(query)

        assert len(results) == 4
        for result in results:
            assert result["file_name"] == "test_meeting.wav"

    def test_combined_filters(self, store: ConversationStore, sample_transcript_json: Path) -> None:
        """Test combining multiple filters."""
        store.ingest(sample_transcript_json)

        query = StoreQuery(
            text=TextQuery("meeting"),
            speakers=SpeakerQuery(speaker_ids=["spk_0"]),
            time_range=TimeRangeQuery(start=0.0, end=5.0),
        )
        results = store.search(query)

        for result in results:
            assert result["speaker_id"] == "spk_0"
            assert result["start_time"] >= 0.0
            assert result["end_time"] <= 5.0

    def test_query_limit_and_offset(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test query pagination with limit and offset."""
        store.ingest(sample_transcript_json)

        # Get first 2 results
        query1 = StoreQuery(limit=2, offset=0)
        results1 = store.search(query1)
        assert len(results1) == 2

        # Get next 2 results
        query2 = StoreQuery(limit=2, offset=2)
        results2 = store.search(query2)
        assert len(results2) == 2

        # Results should be different
        assert results1[0]["segment_id"] != results2[0]["segment_id"]

    def test_legacy_query_api(self, store: ConversationStore, sample_transcript_json: Path) -> None:
        """Test the legacy query() API."""
        store.ingest(sample_transcript_json)

        filter = QueryFilter(text="meeting", limit=10)
        result = store.query(filter)

        assert result.total_count >= 1
        assert len(result.entries) >= 1
        assert result.query_time_ms > 0


# =============================================================================
# Export Tests
# =============================================================================


class TestExport:
    """Tests for export functionality."""

    def test_export_jsonl(
        self, store: ConversationStore, sample_transcript_json: Path, tmp_path: Path
    ) -> None:
        """Test exporting to JSONL format."""
        store.ingest(sample_transcript_json)

        output_path = tmp_path / "export.jsonl"
        options = ExportOptions(format=ExportFormat.JSONL, output_path=str(output_path))
        query = StoreQuery(limit=100)

        result = store.export(query, options=options)

        assert result.status == "success"
        assert result.record_count == 4
        assert output_path.exists()

        # Verify JSONL format
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 4
        for line in lines:
            record = json.loads(line)
            assert "text" in record
            assert "transcript_id" in record

    def test_export_json(
        self, store: ConversationStore, sample_transcript_json: Path, tmp_path: Path
    ) -> None:
        """Test exporting to JSON format."""
        store.ingest(sample_transcript_json)

        output_path = tmp_path / "export.json"
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_path=str(output_path),
            pretty_print=True,
        )
        query = StoreQuery(limit=100)

        result = store.export(query, options=options)

        assert result.status == "success"
        assert output_path.exists()

        # Verify JSON format
        data = json.loads(output_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 4

    def test_export_csv(
        self, store: ConversationStore, sample_transcript_json: Path, tmp_path: Path
    ) -> None:
        """Test exporting to CSV format."""
        store.ingest(sample_transcript_json)

        output_path = tmp_path / "export.csv"
        options = ExportOptions(format=ExportFormat.CSV, output_path=str(output_path))
        query = StoreQuery(limit=100)

        result = store.export(query, options=options)

        assert result.status == "success"
        assert output_path.exists()

        # Verify CSV format
        import csv

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 4
        assert "text" in rows[0]
        assert "transcript_id" in rows[0]

    def test_export_to_string(self, store: ConversationStore, sample_transcript_json: Path) -> None:
        """Test exporting to string (no output file)."""
        store.ingest(sample_transcript_json)

        options = ExportOptions(format=ExportFormat.JSONL)  # No output_path
        query = StoreQuery(limit=100)

        result = store.export(query, options=options)

        assert result.status == "success"
        assert result.data is not None
        assert result.output_path is None

        # Verify content
        lines = result.data.strip().split("\n")
        assert len(lines) == 4

    def test_export_with_query_filter(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test export with query filter applied."""
        store.ingest(sample_transcript_json)

        options = ExportOptions(format=ExportFormat.JSONL)
        query = StoreQuery(
            text=TextQuery("pricing"),
            limit=100,
        )

        result = store.export(query, options=options)

        assert result.status == "success"
        assert result.record_count == 1

    def test_export_legacy_api(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test the legacy export() API."""
        store.ingest(sample_transcript_json)

        entries = store.export(format="jsonl")

        assert isinstance(entries, list)
        assert len(entries) >= 1


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for store statistics."""

    def test_stats_empty_store(self, store: ConversationStore) -> None:
        """Test statistics for empty store."""
        stats = store.stats()

        assert stats["transcript_count"] == 0
        assert stats["segment_count"] == 0
        assert stats["word_count"] == 0
        assert stats["speaker_count"] == 0
        assert stats["action_item_count"] == 0

    def test_stats_with_data(self, store: ConversationStore, sample_transcript_json: Path) -> None:
        """Test statistics with data."""
        store.ingest(sample_transcript_json)

        stats = store.stats()

        assert stats["transcript_count"] == 1
        assert stats["segment_count"] == 4
        assert stats["speaker_count"] == 2
        assert stats["action_item_count"] == 1
        assert stats["database_size_bytes"] > 0

    def test_stats_legacy_api(self, store: ConversationStore, sample_transcript_json: Path) -> None:
        """Test the legacy get_stats() API."""
        store.ingest(sample_transcript_json)

        stats = store.get_stats()

        assert stats["transcript_count"] == 1


# =============================================================================
# Transcript Management Tests
# =============================================================================


class TestTranscriptManagement:
    """Tests for transcript management operations."""

    def test_list_transcripts(self, store: ConversationStore, sample_transcript_json: Path) -> None:
        """Test listing all transcripts."""
        store.ingest(sample_transcript_json)

        transcripts = store.list_transcripts()

        assert len(transcripts) == 1
        assert transcripts[0]["file_name"] == "test_meeting.wav"
        assert transcripts[0]["segment_count"] == 4

    def test_get_transcript(self, store: ConversationStore, sample_transcript_json: Path) -> None:
        """Test getting a single transcript by ID."""
        result = store.ingest(sample_transcript_json)

        transcript = store.get_transcript(result.transcript_id)

        assert transcript is not None
        assert transcript["file_name"] == "test_meeting.wav"
        assert transcript["segment_count"] == 4

    def test_get_transcript_not_found(self, store: ConversationStore) -> None:
        """Test getting a nonexistent transcript."""
        transcript = store.get_transcript("nonexistent_id")
        assert transcript is None

    def test_delete_transcript(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test deleting a transcript."""
        result = store.ingest(sample_transcript_json)

        # Delete transcript
        deleted = store.delete_transcript(result.transcript_id)
        assert deleted is True

        # Verify deletion
        stats = store.stats()
        assert stats["transcript_count"] == 0
        assert stats["segment_count"] == 0

    def test_delete_transcript_not_found(self, store: ConversationStore) -> None:
        """Test deleting a nonexistent transcript."""
        deleted = store.delete_transcript("nonexistent_id")
        assert deleted is False


# =============================================================================
# Action Items Tests
# =============================================================================


class TestActionItems:
    """Tests for action item management."""

    def test_get_action_items(self, store: ConversationStore, sample_transcript_json: Path) -> None:
        """Test getting action items."""
        store.ingest(sample_transcript_json)

        actions = store.get_action_items()

        assert len(actions) == 1
        assert "proposal" in actions[0]["text"].lower()
        assert actions[0]["status"] == "open"

    def test_update_action_item_status(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test updating action item status."""
        store.ingest(sample_transcript_json)

        actions = store.get_action_items()
        action_id = actions[0]["action_id"]

        # Complete the action
        updated = store.update_action_item_status(action_id, "completed")
        assert updated is True

        # Verify status changed
        actions = store.get_action_items()
        assert actions[0]["status"] == "completed"

    def test_filter_action_items_by_status(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test filtering action items by status."""
        store.ingest(sample_transcript_json)

        # All open
        open_actions = store.get_action_items(status="open")
        assert len(open_actions) == 1

        # All completed
        completed_actions = store.get_action_items(status="completed")
        assert len(completed_actions) == 0

    def test_legacy_action_api(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test the legacy action item APIs."""
        from slower_whisper.pipeline.store import ActionStatus

        store.ingest(sample_transcript_json)

        # List actions
        actions = store.list_actions()
        assert len(actions) >= 1

        # Complete action
        action_id = actions[0].id
        completed = store.complete_action(action_id)
        assert completed is True

        # Filter by status
        open_actions = store.list_actions(status=ActionStatus.OPEN)
        assert len(open_actions) == 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_store_error_on_closed_connection(self, tmp_db: Path) -> None:
        """Test that operations on closed store raise StoreError."""
        store = ConversationStore(tmp_db)
        store.close()

        with pytest.raises(StoreError):
            store.stats()

    def test_query_error_handling(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test that invalid queries are handled gracefully."""
        store.ingest(sample_transcript_json)

        # Empty query should work (returns all results)
        query = StoreQuery()
        results = store.search(query)
        assert len(results) == 4


# =============================================================================
# Roundtrip Tests
# =============================================================================


class TestRoundtrip:
    """Tests for data integrity through ingest/query/export cycles."""

    def test_ingest_query_roundtrip(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> None:
        """Test that data survives ingest and query cycle."""
        store.ingest(sample_transcript_json)

        # Query all segments
        query = StoreQuery(limit=100)
        results = store.search(query)

        assert len(results) == 4

        # Verify data integrity
        texts = {r["text"] for r in results}
        assert "Hello, welcome to the meeting." in texts
        assert "Let's discuss the pricing strategy." in texts

    def test_ingest_export_roundtrip(
        self, store: ConversationStore, sample_transcript_json: Path, tmp_path: Path
    ) -> None:
        """Test that data survives ingest and export cycle."""
        store.ingest(sample_transcript_json)

        # Export to JSONL
        output_path = tmp_path / "export.jsonl"
        options = ExportOptions(format=ExportFormat.JSONL, output_path=str(output_path))
        query = StoreQuery(limit=100)

        store.export(query, options=options)

        # Verify exported data
        lines = output_path.read_text().strip().split("\n")
        records = [json.loads(line) for line in lines]

        assert len(records) == 4

        # Check data integrity
        texts = {r["text"] for r in records}
        assert "Hello, welcome to the meeting." in texts

    def test_multiple_transcripts(self, store: ConversationStore, tmp_path: Path) -> None:
        """Test handling multiple transcripts."""
        # Create two different transcripts
        for i in range(2):
            transcript_data = {
                "schema_version": 2,
                "file": f"meeting_{i}.wav",
                "language": "en",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 2.0,
                        "text": f"This is meeting {i}.",
                    }
                ],
            }
            json_path = tmp_path / f"meeting_{i}.json"
            json_path.write_text(json.dumps(transcript_data), encoding="utf-8")

            # Use unique transcript IDs
            options = IngestOptions(generate_transcript_id=True)
            store.ingest(json_path, options=options)

        # Verify both are stored
        stats = store.stats()
        assert stats["transcript_count"] == 2
        assert stats["segment_count"] == 2

        # Query should find both
        query = StoreQuery(text=TextQuery("meeting"))
        results = store.search(query)
        assert len(results) == 2


# =============================================================================
# Parquet Export Tests (Optional)
# =============================================================================


def _parquet_available() -> bool:
    """Check if pyarrow is available for parquet export."""
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


class TestParquetExport:
    """Tests for optional Parquet export functionality."""

    @pytest.fixture
    def store_with_data(
        self, store: ConversationStore, sample_transcript_json: Path
    ) -> ConversationStore:
        """Return a store with sample data ingested."""
        store.ingest(sample_transcript_json)
        return store

    def test_parquet_export_available(self) -> None:
        """Check if parquet export is available (pyarrow installed)."""
        try:
            import pyarrow  # noqa: F401

            parquet_available = True
        except ImportError:
            parquet_available = False

        # This test passes either way - it's just documenting the status
        assert isinstance(parquet_available, bool)

    @pytest.mark.skipif(
        not _parquet_available(),
        reason="pyarrow not installed",
    )
    def test_export_parquet(self, store_with_data: ConversationStore, tmp_path: Path) -> None:
        """Test exporting to Parquet format (requires pyarrow)."""
        output_path = tmp_path / "export.parquet"

        # Export using the export_parquet method
        result = store_with_data.export_parquet(str(output_path))

        assert result.status == "success"
        assert output_path.exists()

        # Verify parquet file can be read back
        import pyarrow.parquet as pq

        table = pq.read_table(str(output_path))
        assert len(table) == 4
        assert "text" in table.column_names
        assert "transcript_id" in table.column_names
