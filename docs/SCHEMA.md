# Transcript Schema Documentation

**Current Version:** v2
**Normative Source:** [`transcription/schemas/transcript-v2.schema.json`](../transcription/schemas/transcript-v2.schema.json)

This document provides a human-readable guide to the slower-whisper transcript schema. For automated validation, always use the JSON Schema files directly.

---

## Table of Contents

1. [Stability Contract](#stability-contract)
2. [Schema Overview](#schema-overview)
3. [Root Fields](#root-fields)
4. [Segments](#segments)
5. [Word-Level Timestamps](#word-level-timestamps-v18)
6. [Speaker Diarization](#speaker-diarization-v11)
7. [Turns](#turns-v11)
8. [Speaker Statistics](#speaker-statistics-v12)
9. [Chunks (RAG Export)](#chunks-rag-export-v13)
10. [Audio Enrichment](#audio-enrichment)
11. [Semantic Annotations](#semantic-annotations)
12. [Metadata](#metadata)
13. [Receipt Contract](#receipt-contract-v21)
14. [Streaming Events Schema](#streaming-events-schema-v20)
15. [Schema Migration](#schema-migration)
16. [Validation](#validation)
17. [Examples](#examples)
18. [See Also](#see-also)

---

## Stability Contract

The schema follows a tiered stability model to balance evolution with backward compatibility:

| Tier | Fields | Commitment |
|------|--------|------------|
| **Stable** | `schema_version`, `file`, `language`, `segments[].{id,start,end,text}` | Will not change in v2.x |
| **Stable (v1.1+)** | `speakers`, `turns`, `annotations` | Shape stable; may add optional fields |
| **Stable (v1.8+)** | `segments[].words[].{word,start,end,probability}` | Shape stable |
| **Extension** | `segments[].audio_state`, `speaker_stats`, `chunks`, `meta` | Optional; may evolve |

**Breaking change policy:** Changes to stable fields require a major version bump.

---

## Schema Overview

A transcript document represents the complete output of the transcription pipeline for a single audio file. The schema supports progressive enrichment, from basic ASR output to fully-annotated conversation analytics.

```
Transcript
├── schema_version (int, required)
├── file (string, required)
├── language (string, required)
├── segments[] (required)
│   ├── id, start, end, text
│   ├── speaker (optional, v1.1+)
│   ├── words[] (optional, v1.8+)
│   └── audio_state (optional)
├── speakers[] (optional, v1.1+)
├── turns[] (optional, v1.1+)
├── speaker_stats[] (optional, v1.2+)
├── chunks[] (optional, v1.3+)
├── annotations (optional)
└── meta (optional)
    ├── diarization
    └── receipt (v2.1+)
```

---

## Root Fields

These are the **serialized JSON keys** (not Python attribute names):

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `schema_version` | `integer` | Yes | Always `2` for this schema |
| `file` | `string` | Yes | Audio filename (e.g., `"meeting.wav"`) |
| `language` | `string` | Yes | ISO 639-1 language code (e.g., `"en"`, `"es"`, `"fr"`) |
| `segments` | `array` | Yes | Ordered list of transcript segments |
| `speakers` | `array\|null` | No | Global speaker table (v1.1+) |
| `turns` | `array\|null` | No | Speaker turns (v1.1+) |
| `speaker_stats` | `array\|null` | No | Per-speaker analytics (v1.2+) |
| `chunks` | `array\|null` | No | RAG-ready chunks (v1.3+) |
| `annotations` | `object\|null` | No | Semantic annotations |
| `meta` | `object\|null` | No | Processing metadata and provenance |

---

## Segments

Segments are the atomic units of transcription. Each represents a contiguous span of transcribed speech.

### Segment Shape

```json
{
  "id": 0,
  "start": 0.0,
  "end": 2.5,
  "text": "Hello world.",
  "speaker": {"id": "spk_0", "confidence": 0.95},
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.98},
    {"word": "world.", "start": 0.6, "end": 1.0, "probability": 0.92}
  ],
  "audio_state": null
}
```

### Segment Fields

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `id` | `integer` | Yes | Unique segment index (0-based, sequential) |
| `start` | `number` | Yes | Start time in seconds |
| `end` | `number` | Yes | End time in seconds |
| `text` | `string` | Yes | Transcribed text |
| `speaker` | `object\|null` | No | Speaker attribution (v1.1+) |
| `tone` | `string\|null` | No | Reserved for future tone tagging |
| `audio_state` | `object\|null` | No | Enriched audio features |
| `words` | `array\|null` | No | Word-level timestamps (v1.8+) |

### Speaker Attribution Object

When diarization is enabled, segments include speaker information:

```json
{
  "id": "spk_0",
  "confidence": 0.95
}
```

| Key | Type | Description |
|-----|------|-------------|
| `id` | `string` | Normalized speaker ID (pattern: `^spk_[0-9]+$`) |
| `confidence` | `number` | Assignment confidence (0.0-1.0, based on overlap ratio) |

---

## Word-Level Timestamps (v1.8+)

When `word_timestamps=True` is enabled during transcription, segments include per-word timing.

### Word Shape

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `word` | `string` | Yes | The transcribed word token |
| `start` | `number` | Yes | Word start time in seconds |
| `end` | `number` | Yes | Word end time in seconds |
| `probability` | `number` | No | ASR confidence (0.0-1.0, default: 1.0) |
| `speaker` | `string\|null` | No | Speaker ID if word-level diarization is applied |

### Example

```json
{
  "words": [
    {"word": "Hello", "start": 0.00, "end": 0.35, "probability": 0.98},
    {"word": "everyone,", "start": 0.40, "end": 0.75, "probability": 0.95},
    {"word": "welcome", "start": 0.80, "end": 1.10, "probability": 0.97},
    {"word": "to", "start": 1.12, "end": 1.20, "probability": 0.99},
    {"word": "the", "start": 1.22, "end": 1.30, "probability": 0.99},
    {"word": "meeting.", "start": 1.32, "end": 1.70, "probability": 0.94}
  ]
}
```

---

## Speaker Diarization (v1.1+)

The `speakers` array provides a global table of detected speakers with aggregate statistics.

### Speaker Entry Shape

```json
{
  "id": "spk_0",
  "label": "Speaker A",
  "total_speech_time": 45.2,
  "num_segments": 12
}
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `id` | `string` | Yes | Normalized speaker ID (`spk_0`, `spk_1`, etc.) |
| `label` | `string\|null` | No | Human-readable label (optional) |
| `total_speech_time` | `number` | Yes | Total speech duration in seconds |
| `num_segments` | `integer` | Yes | Number of segments attributed to this speaker |

---

## Turns (v1.1+)

Turns group contiguous segments by the same speaker into conversational units.

### Turn Shape

```json
{
  "id": "turn_0",
  "speaker_id": "spk_0",
  "segment_ids": [0, 1, 2],
  "start": 0.0,
  "end": 8.5,
  "text": "Hello everyone, welcome to the meeting. Let's get started.",
  "metadata": {
    "question_count": 0,
    "interruption_started_here": false,
    "avg_pause_ms": 250.0,
    "disfluency_ratio": 0.02
  }
}
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `id` | `string` | Yes | Turn identifier (`turn_0`, `turn_1`, etc.) |
| `speaker_id` | `string` | Yes | Speaker ID for this turn |
| `segment_ids` | `array[integer]` | Yes | List of segment IDs in this turn |
| `start` | `number` | Yes | Turn start time in seconds |
| `end` | `number` | Yes | Turn end time in seconds |
| `text` | `string` | No | Concatenated text from all segments |
| `metadata` | `object` | No | Turn-level enrichment metadata |

### Turn Metadata Fields

| Key | Type | Description |
|-----|------|-------------|
| `question_count` | `integer` | Number of questions in the turn |
| `interruption_started_here` | `boolean` | Whether this turn began as an interruption |
| `avg_pause_ms` | `number\|null` | Average pause duration within the turn |
| `disfluency_ratio` | `number\|null` | Ratio of disfluent words (fillers, repairs) |

---

## Speaker Statistics (v1.2+)

The `speaker_stats` array provides aggregated analytics per speaker.

### SpeakerStats Shape

```json
{
  "speaker_id": "spk_0",
  "total_talk_time": 120.5,
  "num_turns": 15,
  "avg_turn_duration": 8.03,
  "interruptions_initiated": 2,
  "interruptions_received": 1,
  "question_turns": 5,
  "prosody_summary": {
    "pitch_median_hz": 185.3,
    "energy_median_db": -12.5
  },
  "sentiment_summary": {
    "positive": 0.45,
    "neutral": 0.50,
    "negative": 0.05
  }
}
```

| Key | Type | Description |
|-----|------|-------------|
| `speaker_id` | `string` | Speaker identifier |
| `total_talk_time` | `number` | Total speech time in seconds |
| `num_turns` | `integer` | Number of conversational turns |
| `avg_turn_duration` | `number` | Average turn length in seconds |
| `interruptions_initiated` | `integer` | Times this speaker interrupted others |
| `interruptions_received` | `integer` | Times this speaker was interrupted |
| `question_turns` | `integer` | Turns containing questions |
| `prosody_summary` | `object` | Pitch and energy aggregates |
| `sentiment_summary` | `object` | Sentiment distribution (positive/neutral/negative) |

---

## Chunks (RAG Export) (v1.3+)

Chunks provide RAG-friendly slices of the transcript for retrieval pipelines.

### Chunk Shape

```json
{
  "id": "chunk_0",
  "start": 0.0,
  "end": 45.0,
  "segment_ids": [0, 1, 2, 3, 4],
  "turn_ids": ["turn_0", "turn_1"],
  "speaker_ids": ["spk_0", "spk_1"],
  "token_count_estimate": 150,
  "text": "Hello everyone... [chunk text]",
  "crosses_turn_boundary": true,
  "turn_boundary_count": 1,
  "has_rapid_turn_taking": false,
  "has_overlapping_speech": false
}
```

### Chunk Fields

| Key | Type | Description |
|-----|------|-------------|
| `id` | `string` | Chunk identifier |
| `start` | `number` | Chunk start time in seconds |
| `end` | `number` | Chunk end time in seconds |
| `segment_ids` | `array[integer]` | Segment IDs included in chunk |
| `turn_ids` | `array[string]` | Turn IDs included in chunk |
| `speaker_ids` | `array[string]` | Unique speakers in chunk |
| `token_count_estimate` | `integer` | Estimated token count for LLM context |
| `text` | `string` | Concatenated text |
| `crosses_turn_boundary` | `boolean` | Contains segments from multiple turns |
| `turn_boundary_count` | `integer` | Number of speaker transitions |
| `has_rapid_turn_taking` | `boolean` | Contains rapid speaker switches |
| `has_overlapping_speech` | `boolean` | Contains overlapping speech |

---

## Audio Enrichment

The `audio_state` field on segments contains extracted audio features when enrichment is enabled.

### Audio State Structure

```json
{
  "prosody": {
    "pitch": {
      "level": "high",
      "mean_hz": 245.3,
      "std_hz": 32.1,
      "variation": "moderate",
      "contour": "rising"
    },
    "energy": {
      "level": "high",
      "db_rms": -8.2,
      "variation": "low"
    },
    "rate": {
      "level": "neutral",
      "syllables_per_sec": 5.3,
      "words_per_sec": 2.8
    },
    "pauses": {
      "count": 1,
      "longest_ms": 250,
      "density": "sparse",
      "density_per_sec": 0.4
    }
  },
  "emotion": {
    "valence": {"level": "positive", "score": 0.72},
    "arousal": {"level": "high", "score": 0.68},
    "dominance": {"level": "neutral", "score": 0.51},
    "categorical": {
      "primary": "happy",
      "confidence": 0.89,
      "secondary": "neutral",
      "secondary_confidence": 0.08,
      "all_scores": {
        "angry": 0.01, "disgusted": 0.00, "fearful": 0.00,
        "happy": 0.89, "neutral": 0.08, "sad": 0.01, "surprised": 0.01
      }
    }
  },
  "rendering": "[audio: high pitch, high energy, normal rate, positive tone]",
  "extraction_status": {
    "prosody": "success",
    "emotion_dimensional": "success",
    "emotion_categorical": "success",
    "errors": []
  }
}
```

### Prosody Features

#### Pitch Features

| Key | Type | Description |
|-----|------|-------------|
| `level` | `enum` | `very_low`, `low`, `neutral`, `high`, `very_high`, `unknown` |
| `mean_hz` | `number\|null` | Mean pitch in Hertz |
| `std_hz` | `number\|null` | Standard deviation in Hertz |
| `variation` | `enum\|null` | `low`, `moderate`, `high`, `unknown` |
| `contour` | `enum\|null` | `rising`, `falling`, `flat`, `unknown` |

#### Energy Features

| Key | Type | Description |
|-----|------|-------------|
| `level` | `enum` | `very_low`, `low`, `neutral`, `high`, `very_high`, `unknown` |
| `db_rms` | `number\|null` | RMS energy in decibels |
| `variation` | `enum\|null` | `low`, `moderate`, `high`, `unknown` |

#### Rate Features

| Key | Type | Description |
|-----|------|-------------|
| `level` | `enum` | `very_low`, `low`, `neutral`, `high`, `very_high`, `unknown` |
| `syllables_per_sec` | `number\|null` | Syllable articulation rate |
| `words_per_sec` | `number\|null` | Word articulation rate |

#### Pause Features

| Key | Type | Description |
|-----|------|-------------|
| `count` | `integer` | Number of detected pauses |
| `longest_ms` | `number\|null` | Longest pause in milliseconds |
| `density` | `enum\|null` | `very_sparse`, `sparse`, `moderate`, `frequent`, `very_frequent`, `unknown` |
| `density_per_sec` | `number\|null` | Pauses per second |

### Emotion Features

#### Dimensional Emotion (VAD Model)

| Dimension | Level Values | Score Range |
|-----------|--------------|-------------|
| `valence` | `very_negative`, `negative`, `neutral`, `positive`, `very_positive` | 0.0-1.0 |
| `arousal` | `very_low`, `low`, `medium`, `high`, `very_high` | 0.0-1.0 |
| `dominance` | `very_submissive`, `submissive`, `neutral`, `dominant`, `very_dominant` | 0.0-1.0 |

#### Categorical Emotion

| Key | Type | Description |
|-----|------|-------------|
| `primary` | `string` | Primary emotion label |
| `confidence` | `number` | Confidence score (0.0-1.0) |
| `secondary` | `string\|null` | Second most likely emotion |
| `secondary_confidence` | `number\|null` | Secondary confidence |
| `all_scores` | `object` | Confidence scores for all emotion categories |

Common emotion labels: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`

---

## Semantic Annotations

The `annotations` field stores semantic analysis results.

### Annotations Structure

```json
{
  "semantic": {
    "keywords": ["project", "deadline", "budget"],
    "risk_tags": ["escalation", "churn_risk"],
    "actions": [
      {
        "text": "Send the report by Friday",
        "speaker_id": "spk_0",
        "segment_ids": [5],
        "pattern": "commitment"
      }
    ],
    "tags": ["escalation"],
    "matches": [
      {"tag": "escalation", "keyword": "urgent", "risk": "high", "segment_id": 3}
    ]
  }
}
```

| Key | Type | Description |
|-----|------|-------------|
| `keywords` | `array[string]` | Normalized keywords detected |
| `risk_tags` | `array[string]` | Risk-oriented tags (e.g., `escalation`, `churn_risk`) |
| `actions` | `array[object]` | Detected action items and commitments |
| `tags` | `array[string]` | Semantic tags (deprecated; use `risk_tags`) |
| `matches` | `array[object]` | Keyword matches with segment references |

---

## Metadata

The `meta` field captures processing provenance and configuration.

### Meta Structure

```json
{
  "model_name": "faster-whisper-large-v3",
  "model_size": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "beam_size": 5,
  "temperature": 0,
  "duration_sec": 120.5,
  "pipeline_version": "2.0.0",
  "transcribed_at": "2026-01-15T10:30:00Z",
  "audio_enrichment": {
    "enriched_at": "2026-01-15T10:30:05Z",
    "total_segments": 15,
    "success_count": 14,
    "partial_count": 1,
    "failed_count": 0,
    "features_enabled": {
      "prosody": true,
      "emotion_dimensional": true,
      "emotion_categorical": true
    }
  },
  "diarization": {
    "requested": true,
    "status": "ok",
    "backend": "pyannote.audio",
    "num_speakers": 2
  },
  "receipt": {
    "tool_version": "2.1.0",
    "schema_version": 2,
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "config_hash": "a1b2c3d4e5f6",
    "run_id": "run-20260128-143052-x7k9p2",
    "created_at": "2026-01-28T14:30:52Z",
    "git_commit": "80a507f"
  }
}
```

### Diarization Status

| Key | Type | Description |
|-----|------|-------------|
| `requested` | `boolean` | Whether diarization was requested |
| `status` | `enum` | `disabled`, `skipped`, `ok`, `error` |
| `backend` | `string\|null` | Backend used (e.g., `pyannote.audio`) |
| `num_speakers` | `integer\|null` | Number of speakers detected |
| `error_type` | `enum\|null` | `missing_dependency`, `auth`, `file_not_found`, `unknown` |
| `message` | `string\|null` | Human-readable status message |
| `error` | `string\|null` | Detailed error information |

---

## Receipt Contract (v2.1+)

The `meta.receipt` field provides provenance tracking for reproducibility and traceability.

### Receipt Fields

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `tool_version` | `string` | Yes | Package version (e.g., `"2.1.0"`) |
| `schema_version` | `integer` | Yes | JSON schema version |
| `model` | `string` | Yes | ASR model name |
| `device` | `string` | Yes | Resolved device (`cuda`, `cpu`) |
| `compute_type` | `string` | Yes | Compute type (`float16`, `int8`) |
| `config_hash` | `string` | Yes | SHA-256 hash of config (12 chars) |
| `run_id` | `string` | Yes | Unique run identifier |
| `created_at` | `string` | Yes | ISO 8601 timestamp (UTC) |
| `git_commit` | `string` | No | Short git commit hash |

### Run ID Format

Run IDs follow the format: `run-YYYYMMDD-HHMMSS-XXXXXX`

- `YYYYMMDD`: Date in year-month-day format
- `HHMMSS`: Time in hour-minute-second format
- `XXXXXX`: 6 random alphanumeric characters

Example: `run-20260128-143052-x7k9p2`

### Usage

```python
from transcription.receipt import build_receipt
from transcription.writers import add_receipt_to_meta

# Build a receipt
receipt = build_receipt(
    model="large-v3",
    device="cuda",
    compute_type="float16",
)

# Add to transcript metadata
transcript.meta = add_receipt_to_meta(transcript.meta, receipt)
```

---

## Streaming Events Schema (v2.0+)

For WebSocket streaming, events are wrapped in an `EventEnvelope`. The schema is defined in [`transcription/schemas/stream_event.schema.json`](../transcription/schemas/stream_event.schema.json).

### Event Envelope Structure

```json
{
  "event_id": 42,
  "stream_id": "str-550e8400-e29b-41d4-a716-446655440000",
  "segment_id": "seg-5",
  "type": "FINALIZED",
  "ts_server": 1706450000000,
  "ts_audio_start": 10.5,
  "ts_audio_end": 12.8,
  "payload": { ... }
}
```

### Envelope Fields

| Key | Type | Description |
|-----|------|-------------|
| `event_id` | `integer` | Monotonically increasing event ID (starts at 1) |
| `stream_id` | `string` | Unique stream identifier (`str-{uuid4}`) |
| `segment_id` | `string\|null` | Segment identifier (`seg-{seq}`) or null for session events |
| `type` | `enum` | Event type (see below) |
| `ts_server` | `integer` | Server timestamp (Unix epoch milliseconds) |
| `ts_audio_start` | `number\|null` | Audio timestamp start (seconds) |
| `ts_audio_end` | `number\|null` | Audio timestamp end (seconds) |
| `payload` | `object` | Event-type-specific payload |

### Server Message Types

| Type | Description |
|------|-------------|
| `SESSION_STARTED` | Session initialization confirmed |
| `PARTIAL` | In-progress transcription (text may change) |
| `FINALIZED` | Final transcription (text immutable) |
| `SPEAKER_TURN` | Speaker turn completed |
| `SEMANTIC_UPDATE` | Semantic annotations for a turn |
| `DIARIZATION_UPDATE` | Incremental diarization results |
| `ERROR` | Error with recovery guidance |
| `SESSION_ENDED` | Session completed with statistics |
| `PONG` | Heartbeat response |
| `PHYSICS_UPDATE` | Conversation dynamics metrics (v2.1) |
| `AUDIO_HEALTH` | Audio quality metrics (v2.1) |
| `VAD_ACTIVITY` | Voice activity detection state (v2.1) |
| `BARGE_IN` | User interrupted TTS playback (v2.1) |
| `END_OF_TURN_HINT` | Speaker may have finished (v2.1) |

See [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) for complete protocol documentation.

---

## Schema Migration

The schema module provides utilities for migrating between schema versions.

### Version Detection

```python
from transcription.schema import detect_schema_version

version = detect_schema_version(data)  # Returns "v1" or "v2"
```

### Automatic Migration

```python
from transcription.schema import auto_migrate

# Migrates v1 data to v2
migrated = auto_migrate(data, target_version="v2")
```

### Manual Migration

```python
from transcription.schema import migrate_v1_to_v2

v2_data = migrate_v1_to_v2(v1_data)
```

### Migration: v1 to v2

The v1 to v2 migration performs these transformations:

1. Adds `schema_version: 2`
2. Ensures all required fields exist
3. Initializes optional fields to `null`
4. Normalizes segment structure
5. Migrates `meta.diarization` structure

---

## Validation

### Command Line

```bash
# Validate a transcript against the JSON Schema
python -m jsonschema -i output.json transcription/schemas/transcript-v2.schema.json
```

### Python API

```python
from transcription.schema import validate_data

# Raises SchemaValidationError if invalid
validate_data(data, "transcript-v2")
```

### Schema Registry

```python
from transcription.schema import get_schema, list_schemas, get_schema_hash

# List available schemas
schemas = list_schemas()  # ['transcript-v2', 'pr-dossier-v2', 'stream_event']

# Get schema info
schema = get_schema("transcript-v2")
print(f"Version: {schema.version}, Hash: {schema.hash}")

# Verify schema integrity
from transcription.schema import verify_schema_integrity
is_valid = verify_schema_integrity("transcript-v2", expected_hash)
```

---

## Examples

### Minimal Transcript

```json
{
  "schema_version": 2,
  "file": "example.wav",
  "language": "en",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world."}
  ],
  "speakers": null,
  "turns": null,
  "meta": null
}
```

### Transcript with Diarization

```json
{
  "schema_version": 2,
  "file": "meeting.wav",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "Good morning everyone.",
      "speaker": {"id": "spk_0", "confidence": 0.92},
      "words": null,
      "audio_state": null
    },
    {
      "id": 1,
      "start": 3.5,
      "end": 5.8,
      "text": "Morning! Ready to start?",
      "speaker": {"id": "spk_1", "confidence": 0.88},
      "words": null,
      "audio_state": null
    }
  ],
  "speakers": [
    {"id": "spk_0", "label": null, "total_speech_time": 3.2, "num_segments": 1},
    {"id": "spk_1", "label": null, "total_speech_time": 2.3, "num_segments": 1}
  ],
  "turns": [
    {"id": "turn_0", "speaker_id": "spk_0", "segment_ids": [0], "start": 0.0, "end": 3.2, "text": "Good morning everyone."},
    {"id": "turn_1", "speaker_id": "spk_1", "segment_ids": [1], "start": 3.5, "end": 5.8, "text": "Morning! Ready to start?"}
  ],
  "meta": {
    "diarization": {
      "requested": true,
      "status": "ok",
      "backend": "pyannote.audio",
      "num_speakers": 2
    }
  }
}
```

### Fully Enriched Transcript

See the JSON Schema file for a complete example with all optional fields populated:
[`transcription/schemas/transcript-v2.schema.json#examples`](../transcription/schemas/transcript-v2.schema.json)

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md) - Python API usage
- [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) - WebSocket streaming protocol
- [CONFIGURATION.md](CONFIGURATION.md) - Pipeline configuration options
- [BENCHMARKS.md](BENCHMARKS.md) - Benchmark evaluation framework

---

## Python Type Reference

The schema maps to Python dataclasses defined in `transcription/models.py`:

| JSON Type | Python Class |
|-----------|--------------|
| Transcript (root) | `Transcript` |
| Segment | `Segment` |
| Word | `Word` |
| Turn | `Turn` |
| Chunk | `Chunk` |
| SpeakerStats | `SpeakerStats` |
| ProsodySummary | `ProsodySummary` |
| SentimentSummary | `SentimentSummary` |
| DiarizationMeta | `DiarizationMeta` |
| TurnMeta | `TurnMeta` |

### Constants

```python
from transcription.models import (
    SCHEMA_VERSION,          # 2
    AUDIO_STATE_VERSION,     # "1.0.0"
    WORD_ALIGNMENT_VERSION,  # "1.0.0"
)
```
