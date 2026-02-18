"""SQLite schema for the Conversation Store.

This module defines the database schema for storing and searching transcripts.
The schema supports:
- Transcript metadata and segments with speaker/word alignment
- Full-text search (FTS5) for segment text and action items
- Semantic annotations (topics, risks, action items)
- Streaming event storage for replay/debugging
- Schema versioning for migrations

Schema version: 1
"""

from __future__ import annotations

# Current schema version - increment when making breaking changes
SCHEMA_VERSION = 1

SCHEMA_V1 = """
-- ============================================================================
-- Schema Metadata
-- ============================================================================

-- Tracks schema version for migrations
CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Insert initial schema version
INSERT OR IGNORE INTO store_meta (key, value) VALUES ('schema_version', '1');


-- ============================================================================
-- Core Transcript Tables
-- ============================================================================

-- Main transcript table - one row per ingested transcript file
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Unique identifier for the transcript (UUID)
    transcript_id TEXT UNIQUE NOT NULL,
    -- Original filename
    file_name TEXT NOT NULL,
    -- Detected language code (e.g., 'en', 'es')
    language TEXT,
    -- Total duration in seconds
    duration_seconds REAL,
    -- Word count across all segments
    word_count INTEGER,
    -- Number of segments
    segment_count INTEGER,
    -- Number of speakers detected
    speaker_count INTEGER,
    -- JSON-encoded transcript metadata (model, device, diarization status, etc.)
    meta_json TEXT,
    -- JSON-encoded annotations (semantic, topics, etc.)
    annotations_json TEXT,
    -- Original source path (for reference, may be deleted)
    source_path TEXT,
    -- When this transcript was ingested
    ingested_at TEXT DEFAULT (datetime('now')),
    -- Schema version of the source JSON
    source_schema_version INTEGER,
    -- Indexing
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_transcripts_file_name ON transcripts(file_name);
CREATE INDEX IF NOT EXISTS idx_transcripts_language ON transcripts(language);
CREATE INDEX IF NOT EXISTS idx_transcripts_ingested_at ON transcripts(ingested_at);


-- Segments table - individual speech segments
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Foreign key to transcripts
    transcript_id TEXT NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    -- Segment index within the transcript (0-based)
    segment_index INTEGER NOT NULL,
    -- Timing in seconds
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    -- Transcribed text
    text TEXT NOT NULL,
    -- Speaker ID (e.g., 'spk_0', 'SPEAKER_01')
    speaker_id TEXT,
    -- Speaker confidence score (0.0-1.0)
    speaker_confidence REAL,
    -- Tone label (if available)
    tone TEXT,
    -- JSON-encoded audio_state (prosody, emotion, etc.)
    audio_state_json TEXT,
    -- Unique constraint
    UNIQUE(transcript_id, segment_index)
);

CREATE INDEX IF NOT EXISTS idx_segments_transcript_id ON segments(transcript_id);
CREATE INDEX IF NOT EXISTS idx_segments_speaker_id ON segments(speaker_id);
CREATE INDEX IF NOT EXISTS idx_segments_start_time ON segments(start_time);
CREATE INDEX IF NOT EXISTS idx_segments_text ON segments(text);


-- Words table - word-level alignments (optional, only if word_timestamps enabled)
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Foreign key to segments
    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    -- Word index within the segment (0-based)
    word_index INTEGER NOT NULL,
    -- The word text
    word TEXT NOT NULL,
    -- Timing in seconds
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    -- ASR confidence (0.0-1.0)
    probability REAL,
    -- Speaker ID at word level (may differ from segment speaker)
    speaker_id TEXT,
    -- Unique constraint
    UNIQUE(segment_id, word_index)
);

CREATE INDEX IF NOT EXISTS idx_words_segment_id ON words(segment_id);


-- ============================================================================
-- Speaker & Turn Tables
-- ============================================================================

-- Speakers table - speaker metadata per transcript
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Foreign key to transcripts
    transcript_id TEXT NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    -- Speaker ID (e.g., 'spk_0')
    speaker_id TEXT NOT NULL,
    -- Display label (e.g., 'Customer', 'Agent')
    label TEXT,
    -- Total speech time in seconds
    total_speech_time REAL,
    -- Number of segments by this speaker
    num_segments INTEGER,
    -- JSON-encoded speaker stats (from speaker_stats)
    stats_json TEXT,
    -- Unique constraint
    UNIQUE(transcript_id, speaker_id)
);

CREATE INDEX IF NOT EXISTS idx_speakers_transcript_id ON speakers(transcript_id);
CREATE INDEX IF NOT EXISTS idx_speakers_speaker_id ON speakers(speaker_id);


-- Turns table - conversational turns (groups of consecutive segments by same speaker)
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Foreign key to transcripts
    transcript_id TEXT NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    -- Turn ID from the transcript
    turn_id TEXT NOT NULL,
    -- Speaker for this turn
    speaker_id TEXT NOT NULL,
    -- Timing in seconds
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    -- Concatenated text from all segments in the turn
    text TEXT NOT NULL,
    -- JSON array of segment indices in this turn
    segment_ids_json TEXT,
    -- JSON-encoded turn metadata (question_count, interruption, etc.)
    metadata_json TEXT,
    -- Unique constraint
    UNIQUE(transcript_id, turn_id)
);

CREATE INDEX IF NOT EXISTS idx_turns_transcript_id ON turns(transcript_id);
CREATE INDEX IF NOT EXISTS idx_turns_speaker_id ON turns(speaker_id);
CREATE INDEX IF NOT EXISTS idx_turns_start_time ON turns(start_time);


-- ============================================================================
-- Semantic Annotation Tables
-- ============================================================================

-- Annotations table - stores semantic annotations per transcript or segment
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Foreign key to transcripts
    transcript_id TEXT NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    -- Optional segment index (NULL = transcript-level annotation)
    segment_index INTEGER,
    -- Annotation type (e.g., 'semantic', 'topic', 'risk')
    annotation_type TEXT NOT NULL,
    -- Provider that generated this annotation
    provider TEXT,
    -- Model used (e.g., 'keyword-v1', 'gpt-4o')
    model TEXT,
    -- Schema version of the annotation
    schema_version TEXT,
    -- Confidence score (0.0-1.0)
    confidence REAL,
    -- Latency in milliseconds
    latency_ms INTEGER,
    -- JSON-encoded normalized annotation data
    normalized_json TEXT,
    -- JSON-encoded raw model output (for debugging)
    raw_output_json TEXT,
    -- When this annotation was created
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_annotations_transcript_id ON annotations(transcript_id);
CREATE INDEX IF NOT EXISTS idx_annotations_type ON annotations(annotation_type);
CREATE INDEX IF NOT EXISTS idx_annotations_provider ON annotations(provider);


-- Action items table - extracted commitments and tasks
CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Foreign key to transcripts
    transcript_id TEXT NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
    -- The action item text
    text TEXT NOT NULL,
    -- Who is responsible (speaker ID or 'agent'/'customer')
    assignee TEXT,
    -- Due date/time if mentioned (free text)
    due TEXT,
    -- Priority level (low, medium, high)
    priority TEXT,
    -- Confidence score (0.0-1.0)
    confidence REAL,
    -- JSON array of segment indices where this action was detected
    segment_ids_json TEXT,
    -- Regex pattern that matched (for rule-based detection)
    pattern TEXT,
    -- Status tracking (open, completed, cancelled)
    status TEXT DEFAULT 'open',
    -- When this action was created
    created_at TEXT DEFAULT (datetime('now')),
    -- When status was last updated
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_action_items_transcript_id ON action_items(transcript_id);
CREATE INDEX IF NOT EXISTS idx_action_items_assignee ON action_items(assignee);
CREATE INDEX IF NOT EXISTS idx_action_items_status ON action_items(status);
CREATE INDEX IF NOT EXISTS idx_action_items_priority ON action_items(priority);


-- ============================================================================
-- Provenance & Receipts
-- ============================================================================

-- Receipts table - provenance tracking for ingested data
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Unique receipt ID (UUID)
    receipt_id TEXT UNIQUE NOT NULL,
    -- Foreign key to transcripts (optional, may be for other operations)
    transcript_id TEXT REFERENCES transcripts(transcript_id) ON DELETE SET NULL,
    -- Operation type (ingest, export, annotate, etc.)
    operation TEXT NOT NULL,
    -- Status (success, partial, error)
    status TEXT NOT NULL,
    -- Source file path or URI
    source TEXT,
    -- JSON-encoded operation options/parameters
    options_json TEXT,
    -- JSON-encoded result summary
    result_json TEXT,
    -- Error message if status is error
    error_message TEXT,
    -- Duration of operation in milliseconds
    duration_ms INTEGER,
    -- When operation was performed
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_receipts_transcript_id ON receipts(transcript_id);
CREATE INDEX IF NOT EXISTS idx_receipts_operation ON receipts(operation);
CREATE INDEX IF NOT EXISTS idx_receipts_status ON receipts(status);
CREATE INDEX IF NOT EXISTS idx_receipts_created_at ON receipts(created_at);


-- ============================================================================
-- Streaming Event Storage
-- ============================================================================

-- Streaming events table - for session replay and debugging
CREATE TABLE IF NOT EXISTS streaming_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Session ID for grouping events
    session_id TEXT NOT NULL,
    -- Foreign key to transcripts (set when session completes)
    transcript_id TEXT REFERENCES transcripts(transcript_id) ON DELETE SET NULL,
    -- Event sequence number within session
    sequence_num INTEGER NOT NULL,
    -- Event type (chunk, segment, turn, semantic, error, etc.)
    event_type TEXT NOT NULL,
    -- Timestamp when event occurred
    event_time TEXT NOT NULL,
    -- JSON-encoded event payload
    payload_json TEXT,
    -- Unique constraint
    UNIQUE(session_id, sequence_num)
);

CREATE INDEX IF NOT EXISTS idx_streaming_events_session_id ON streaming_events(session_id);
CREATE INDEX IF NOT EXISTS idx_streaming_events_event_type ON streaming_events(event_type);
CREATE INDEX IF NOT EXISTS idx_streaming_events_event_time ON streaming_events(event_time);


-- ============================================================================
-- Full-Text Search (FTS5)
-- ============================================================================

-- FTS5 virtual table for segment text search
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    content='segments',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- FTS5 virtual table for action item search
CREATE VIRTUAL TABLE IF NOT EXISTS action_items_fts USING fts5(
    text,
    content='action_items',
    content_rowid='id',
    tokenize='porter unicode61'
);


-- ============================================================================
-- FTS Sync Triggers
-- ============================================================================

-- Triggers to keep segments_fts in sync with segments table
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Triggers to keep action_items_fts in sync with action_items table
CREATE TRIGGER IF NOT EXISTS action_items_ai AFTER INSERT ON action_items BEGIN
    INSERT INTO action_items_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS action_items_ad AFTER DELETE ON action_items BEGIN
    INSERT INTO action_items_fts(action_items_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS action_items_au AFTER UPDATE ON action_items BEGIN
    INSERT INTO action_items_fts(action_items_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO action_items_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

# SQL to check schema version
CHECK_SCHEMA_VERSION_SQL = """
SELECT value FROM store_meta WHERE key = 'schema_version';
"""

# SQL to update schema version
UPDATE_SCHEMA_VERSION_SQL = """
INSERT OR REPLACE INTO store_meta (key, value, updated_at)
VALUES ('schema_version', ?, datetime('now'));
"""

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_V1",
    "CHECK_SCHEMA_VERSION_SQL",
    "UPDATE_SCHEMA_VERSION_SQL",
]
