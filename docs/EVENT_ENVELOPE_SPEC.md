# Event Envelope Specification

**Version:** 2.1.0
**Status:** Stable
**Schema:** [`transcription/schemas/stream_event.schema.json`](../transcription/schemas/stream_event.schema.json)

This document describes the event envelope format used in the WebSocket streaming protocol. Every server-to-client message uses this envelope structure.

## Overview

The event envelope provides a consistent wrapper for all streaming events, enabling:
- **Event ordering** via monotonically increasing `event_id`
- **Session correlation** via `stream_id`
- **Timing information** via `ts_server` and audio timestamps
- **Type-safe payloads** specific to each event type

## Envelope Structure

```json
{
  "schema_version": "2.1.0",
  "event_id": 1,
  "stream_id": "str-550e8400-e29b-41d4-a716-446655440000",
  "segment_id": "seg-0",
  "type": "FINALIZED",
  "ts_server": 1706400000000,
  "ts_audio_start": 0.0,
  "ts_audio_end": 2.5,
  "payload": { ... }
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | integer | Monotonically increasing ID per stream. Starts at 1. Gaps indicate dropped PARTIAL events. |
| `stream_id` | string | Unique stream identifier: `str-{uuid4}`. Immutable for session duration. |
| `type` | string | Server message type (see [Event Types](#event-types)). |
| `ts_server` | integer | Server timestamp in Unix epoch milliseconds. |
| `payload` | object | Event-specific payload (schema varies by type). |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version (semver). Default: `"2.1.0"`. |
| `segment_id` | string \| null | Segment identifier: `seg-{seq}`. Null for non-segment events. |
| `ts_audio_start` | number \| null | Audio timestamp start (seconds). Null for non-audio events. |
| `ts_audio_end` | number \| null | Audio timestamp end (seconds). Null for non-audio events. |

## Event Types

### Core Events (v2.0)

| Type | Description | Has Segment ID |
|------|-------------|----------------|
| `SESSION_STARTED` | Session initialization confirmed | No |
| `PARTIAL` | In-progress transcription (may change) | Yes |
| `FINALIZED` | Final transcription (immutable) | Yes |
| `SPEAKER_TURN` | Speaker turn boundary | No |
| `SEMANTIC_UPDATE` | Semantic annotations for a turn | No |
| `DIARIZATION_UPDATE` | Incremental speaker assignments | No |
| `ERROR` | Error information | No |
| `SESSION_ENDED` | Session completed with stats | No |
| `PONG` | Keep-alive response | No |

### Reflex Events (v2.1)

| Type | Description | Has Segment ID |
|------|-------------|----------------|
| `PHYSICS_UPDATE` | Conversation dynamics metrics | No |
| `AUDIO_HEALTH` | Audio quality metrics | No |
| `VAD_ACTIVITY` | Voice activity detection state | No |
| `BARGE_IN` | User speech during TTS playback | No |
| `END_OF_TURN_HINT` | Speaker may have finished | No |

## Payload Schemas

### SESSION_STARTED

```json
{
  "session_id": "str-550e8400-e29b-41d4-a716-446655440000"
}
```

Confirms session initialization. The `session_id` matches the envelope `stream_id`.

### SESSION_ENDED

```json
{
  "stats": {
    "chunks_received": 100,
    "bytes_received": 320000,
    "segments_partial": 15,
    "segments_finalized": 10,
    "events_sent": 45,
    "events_dropped": 0,
    "errors": 0,
    "backpressure_events": 0,
    "resume_attempts": 0,
    "duration_sec": 30.5
  }
}
```

### PARTIAL

```json
{
  "segment": {
    "start": 0.0,
    "end": 1.5,
    "text": "Hello, how are",
    "speaker_id": "spk_0"
  },
  "confidence": 0.85
}
```

Text may change with subsequent PARTIAL events. Same `segment_id` links updates.

### FINALIZED

```json
{
  "segment": {
    "start": 0.0,
    "end": 2.5,
    "text": "Hello, how are you today?",
    "speaker_id": "spk_0",
    "audio_state": {
      "prosody": {
        "pitch": { "level": "neutral", "mean_hz": 150.0 },
        "energy": { "level": "moderate", "mean_db": -20.0 },
        "rate": { "level": "moderate", "syllables_per_sec": 4.5 }
      },
      "emotion": {
        "valence": { "level": "positive", "score": 0.7 },
        "arousal": { "level": "moderate", "score": 0.5 }
      },
      "categorical_emotion": {
        "label": "happy",
        "confidence": 0.75
      },
      "rendering": "[audio: neutral pitch, positive]"
    }
  }
}
```

Text is immutable after FINALIZED. `audio_state` is null if enrichment is disabled.

### SPEAKER_TURN

```json
{
  "turn": {
    "id": "turn-0",
    "speaker_id": "spk_0",
    "start": 0.0,
    "end": 5.0,
    "segment_ids": ["seg-0", "seg-1"],
    "text": "Hello, how are you today? I hope you're doing well."
  },
  "previous_speaker": null
}
```

### SEMANTIC_UPDATE

```json
{
  "turn_id": "turn-0",
  "keywords": ["greeting", "wellbeing"],
  "risk_tags": [],
  "actions": [],
  "question_count": 1,
  "intent": "greeting",
  "sentiment": "positive",
  "context_size": 1
}
```

### DIARIZATION_UPDATE

```json
{
  "update_number": 1,
  "audio_duration": 10.5,
  "num_speakers": 2,
  "speaker_ids": ["spk_0", "spk_1"],
  "assignments": [
    { "start": 0.0, "end": 5.0, "speaker_id": "spk_0", "confidence": 0.95 },
    { "start": 5.5, "end": 10.5, "speaker_id": "spk_1", "confidence": 0.92 }
  ]
}
```

### ERROR

```json
{
  "code": "ASR_TIMEOUT",
  "message": "ASR processing timed out after 30 seconds",
  "recoverable": true,
  "details": { "timeout_ms": 30000 }
}
```

**Error Codes:**
- `ASR_TIMEOUT` - ASR processing timeout (recoverable)
- `ASR_FAILURE` - ASR processing failed (recoverable)
- `ENRICHMENT_FAILURE` - Enrichment failed (recoverable)
- `SEQUENCE_ERROR` - Invalid message sequence (recoverable)
- `BUFFER_OVERFLOW` - Event buffer full (recoverable, PARTIAL dropped)
- `RESUME_GAP` - Resume gap too large (not recoverable)
- `SESSION_ERROR` - Session-level error (not recoverable)
- `INVALID_MESSAGE` - Malformed client message (recoverable)
- `RATE_LIMITED` - Rate limit exceeded (recoverable)
- `DIARIZATION_FAILURE` - Diarization failed (recoverable)
- `SESSION_MISMATCH` - Session ID mismatch (not recoverable)

### PONG

```json
{
  "timestamp": 1706400000000,
  "server_timestamp": 1706400000050
}
```

Echo of client's PING timestamp plus server timestamp for latency measurement.

### PHYSICS_UPDATE (v2.1)

```json
{
  "speaker_talk_times": {
    "spk_0": 15.5,
    "spk_1": 12.3
  },
  "total_duration_sec": 30.0,
  "interruption_count": 2,
  "interruption_rate": 4.0,
  "mean_response_latency_sec": 0.8,
  "speaker_transitions": 5,
  "overlap_duration_sec": 1.2
}
```

### AUDIO_HEALTH (v2.1)

```json
{
  "clipping_ratio": 0.01,
  "rms_energy": -25.0,
  "snr_proxy": 20.5,
  "spectral_centroid": 2500.0,
  "quality_score": 0.85,
  "is_speech_likely": true
}
```

### VAD_ACTIVITY (v2.1)

```json
{
  "energy_level": -30.0,
  "is_speech": false,
  "silence_duration_sec": 2.5
}
```

### BARGE_IN (v2.1)

```json
{
  "energy": -20.0,
  "tts_elapsed_sec": 1.5
}
```

Emitted when user speech is detected during TTS playback.

### END_OF_TURN_HINT (v2.1)

```json
{
  "confidence": 0.85,
  "silence_duration_sec": 1.0,
  "prosodic_cues": {
    "falling_intonation": true,
    "final_lengthening": false
  }
}
```

## Ordering Guarantees

1. **event_id is monotonically increasing** within a stream
2. **Gaps in event_id indicate dropped PARTIAL events** (backpressure)
3. **FINALIZED events are never dropped** (delivery guarantee)
4. **Same segment_id links PARTIAL → FINALIZED** for a segment
5. **SESSION_STARTED is always event_id=1**
6. **SESSION_ENDED is always the final event**

## Backpressure Handling

When the server's event queue fills:
1. PARTIAL events may be dropped (client sees event_id gaps)
2. FINALIZED events are always delivered
3. BUFFER_OVERFLOW error is emitted to notify client
4. Client should adjust processing rate or buffer size

## Resume Protocol

Clients can resume from disconnection:
1. Send `RESUME_SESSION` with `last_event_id`
2. Server replays events from replay buffer (if available)
3. If gap too large, `RESUME_GAP` error is sent
4. Client must restart session on unrecoverable gap

## Version Compatibility

- Check `schema_version` field for protocol version
- v2.0 clients should ignore unknown event types
- v2.1 events (`PHYSICS_UPDATE`, `AUDIO_HEALTH`, etc.) are optional extensions
- Server may omit v2.1 events if features are disabled

## See Also

- [Streaming Architecture](STREAMING_ARCHITECTURE.md) - Full protocol documentation
- [Streaming API](STREAMING_API.md) - Client integration guide
- [JSON Schema](../transcription/schemas/stream_event.schema.json) - Machine-readable schema
