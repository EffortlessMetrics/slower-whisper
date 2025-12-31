# Streaming Architecture

**Version:** v1.8.0 (with v1.9.0 and v2.0.0 planned features)
**Last Updated:** 2025-12-31

This document provides comprehensive documentation for slower-whisper's streaming architecture, covering the real-time transcription pipeline, enrichment layers, and semantic annotation.

---

## Table of Contents

1. [Overview](#overview)
2. [Message Flow](#message-flow)
3. [Event Types](#event-types)
4. [Session Classes](#session-classes)
5. [Callback Interface (v1.9.0)](#callback-interface-v190)
6. [WebSocket Protocol (v2.0.0)](#websocket-protocol-v200)
7. [Performance Characteristics](#performance-characteristics)
8. [Examples](#examples)
9. [Integration Patterns](#integration-patterns)
10. [Monitoring and Metrics](#monitoring-and-metrics)

---

## Overview

The streaming architecture enables real-time transcription and enrichment of audio conversations. It is designed around three core principles:

1. **Event-Driven**: All state changes emit typed events that downstream consumers can process
2. **Layered Processing**: Each layer (ASR, enrichment, semantic) operates independently
3. **Graceful Degradation**: Failures in enrichment do not block transcription

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STREAMING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │  Audio   │───▶│  ASR Engine      │───▶│   StreamingSession         │ │
│  │  Input   │    │  (faster-whisper)│    │   (base state machine)     │ │
│  └──────────┘    └──────────────────┘    └─────────────┬──────────────┘ │
│                                                        │                 │
│                        StreamChunk                     │                 │
│                    ┌───────────────────────────────────┘                 │
│                    │                                                     │
│                    ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    ENRICHMENT LAYER (Optional)                       ││
│  │  ┌────────────────────────┐    ┌────────────────────────┐           ││
│  │  │ StreamingEnrichment    │    │ LiveSemanticSession    │           ││
│  │  │ Session                │    │                        │           ││
│  │  │ - Prosody extraction   │    │ - Turn detection       │           ││
│  │  │ - Emotion recognition  │    │ - Keyword extraction   │           ││
│  │  │ - Audio state          │    │ - Risk tag detection   │           ││
│  │  └────────────┬───────────┘    └──────────┬─────────────┘           ││
│  │               │                           │                          ││
│  └───────────────┼───────────────────────────┼──────────────────────────┘│
│                  │                           │                           │
│                  ▼                           ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                        EVENT STREAM                                  ││
│  │                                                                      ││
│  │  PARTIAL_SEGMENT ──▶ FINAL_SEGMENT ──▶ SEMANTIC_UPDATE              ││
│  │       │                    │                   │                     ││
│  │       └────────────────────┴───────────────────┘                     ││
│  │                            │                                         ││
│  │                            ▼                                         ││
│  │                    Downstream Consumers                              ││
│  │              (UI, LLM, Storage, Analytics)                           ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Module | Purpose |
|-----------|--------|---------|
| `StreamingSession` | `streaming.py` | Base state machine for post-ASR chunks |
| `StreamingEnrichmentSession` | `streaming_enrich.py` | Audio feature extraction layer |
| `LiveSemanticSession` | `streaming_semantic.py` | Turn-aware semantic annotation |

---

## Message Flow

### Data Flow Sequence

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MESSAGE FLOW DIAGRAM                             │
└─────────────────────────────────────────────────────────────────────────┘

 Audio              ASR                 Streaming           Enrichment
 Input            Engine                Session              Layer
   │                 │                     │                    │
   │  audio chunk    │                     │                    │
   │────────────────▶│                     │                    │
   │                 │                     │                    │
   │                 │   StreamChunk       │                    │
   │                 │    {start, end,     │                    │
   │                 │     text,           │                    │
   │                 │     speaker_id}     │                    │
   │                 │────────────────────▶│                    │
   │                 │                     │                    │
   │                 │                     │──┐                 │
   │                 │                     │  │ State Machine   │
   │                 │                     │  │ - validate      │
   │                 │                     │  │ - extend/close  │
   │                 │                     │◀─┘                 │
   │                 │                     │                    │
   │                 │                     │  PARTIAL_SEGMENT   │
   │                 │                     │───────────────────▶│ (pass-through)
   │                 │                     │                    │
   │                 │                     │  FINAL_SEGMENT     │
   │                 │                     │───────────────────▶│
   │                 │                     │                    │──┐
   │                 │                     │                    │  │ Extract
   │                 │                     │                    │  │ - prosody
   │                 │                     │                    │  │ - emotion
   │                 │                     │                    │◀─┘
   │                 │                     │                    │
   │                 │                     │                    │
   │                 │                     │    StreamEvent     │
   │                 │                     │◀───────────────────│
   │                 │                     │    (enriched)      │
   │                 │                     │                    │
```

### Chunk → Segment State Transitions

```
┌────────────────────────────────────────────────────────────────────────┐
│                    STATE TRANSITION DIAGRAM                             │
└────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │     EMPTY       │
                         │   (no partial)  │
                         └────────┬────────┘
                                  │
                         chunk arrives
                                  │
                                  ▼
                         ┌─────────────────┐
              ┌──────────│    PARTIAL      │◀────────────┐
              │          │   (in-flight)   │             │
              │          └────────┬────────┘             │
              │                   │                      │
         same speaker        different speaker       same speaker
         gap <= max          OR gap > max            gap <= max
              │                   │                      │
              │                   ▼                      │
              │          ┌─────────────────┐             │
              │          │   FINALIZED     │             │
              │          │                 │             │
              │          └────────┬────────┘             │
              │                   │                      │
              │          emit FINAL_SEGMENT              │
              │                   │                      │
              │                   ▼                      │
              │          ┌─────────────────┐             │
              │          │  NEW PARTIAL    │─────────────┘
              │          │  (from chunk)   │
              │          └─────────────────┘
              │
              │
              │  extend segment
              │  emit PARTIAL_SEGMENT
              │
              └──────────────────────────────────────────┘
```

### Finalization Rules

A segment is finalized when:

1. **Speaker change**: New chunk has different `speaker_id`
2. **Time gap**: Gap between chunks exceeds `max_gap_sec` (default: 1.0s)
3. **End of stream**: `end_of_stream()` called explicitly

---

## Event Types

### StreamEventType Enum

```python
class StreamEventType(Enum):
    """Event types emitted by streaming sessions."""

    PARTIAL_SEGMENT = "partial_segment"   # Segment may change
    FINAL_SEGMENT = "final_segment"       # Segment complete, no more changes
    SEMANTIC_UPDATE = "semantic_update"   # Semantic annotation attached (v1.7+)
```

### Event Type Details

#### PARTIAL_SEGMENT

Emitted after each chunk when the segment is still building.

```python
StreamEvent(
    type=StreamEventType.PARTIAL_SEGMENT,
    segment=StreamSegment(
        start=0.0,
        end=2.5,
        text="Hello world",
        speaker_id="spk_0",
        audio_state=None  # Not enriched for partials
    ),
    semantic=None
)
```

**When to use**: Show live transcription in UI, update live caption display.

**Guarantees**:
- Text may grow or change with subsequent chunks
- Same segment may emit multiple PARTIAL events
- audio_state is always None (enrichment deferred to FINAL)

#### FINAL_SEGMENT

Emitted when a segment is complete and will not change.

```python
StreamEvent(
    type=StreamEventType.FINAL_SEGMENT,
    segment=StreamSegment(
        start=0.0,
        end=4.2,
        text="Hello world, how are you today?",
        speaker_id="spk_0",
        audio_state={
            "prosody": {"pitch": {"level": "high", "mean_hz": 245.3}},
            "emotion": {"valence": {"level": "positive", "score": 0.72}},
            "rendering": "[audio: high pitch, positive]"
        }
    ),
    semantic=None
)
```

**When to use**: Persist to database, send to LLM, update final transcript.

**Guarantees**:
- Text is final and will not change
- audio_state populated (if enrichment enabled)
- Exactly one FINAL event per segment

#### SEMANTIC_UPDATE (v1.7+)

Emitted when semantic annotation completes for a finalized turn.

```python
StreamEvent(
    type=StreamEventType.SEMANTIC_UPDATE,
    segment=StreamSegment(
        start=0.0,
        end=12.5,
        text="Can you send me the proposal by Friday?",
        speaker_id="spk_1"
    ),
    semantic=SemanticUpdatePayload(
        turn=Turn(id="turn_0", speaker_id="spk_1", ...),
        keywords=["proposal", "deadline"],
        risk_tags=[],
        actions=[{"text": "send proposal", "pattern": "imperative"}],
        question_count=1,
        context_size=5
    )
)
```

**When to use**: Route to action item workflow, detect escalation, trigger LLM analysis.

**Guarantees**:
- Only emitted for finalized turns (not segments)
- semantic field contains SemanticUpdatePayload
- Context window reflects recent conversation history

#### Future Event Types (v2.0.0)

| Event Type | Description |
|------------|-------------|
| `SPEAKER_TURN` | Explicit speaker change notification |
| `ERROR` | Processing error with recovery information |
| `HEARTBEAT` | Connection keep-alive (WebSocket only) |

---

## Session Classes

### StreamingSession (Base)

The core state machine for processing post-ASR chunks.

```python
from transcription.streaming import (
    StreamChunk,
    StreamConfig,
    StreamingSession,
    StreamEvent,
    StreamEventType,
)

# Configuration
config = StreamConfig(
    max_gap_sec=1.0  # Gap threshold to finalize segment
)

# Initialize session
session = StreamingSession(config)

# Process chunks
chunk: StreamChunk = {
    "start": 0.0,
    "end": 2.5,
    "text": "Hello world",
    "speaker_id": "spk_0"
}

events = session.ingest_chunk(chunk)
for event in events:
    if event.type == StreamEventType.PARTIAL_SEGMENT:
        print(f"[PARTIAL] {event.segment.text}")
    elif event.type == StreamEventType.FINAL_SEGMENT:
        print(f"[FINAL] {event.segment.text}")

# Finalize stream
final_events = session.end_of_stream()
```

#### StreamConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_gap_sec` | float | 1.0 | Gap threshold to start new segment |

#### StreamChunk (Input)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `start` | float | Yes | Start time in seconds |
| `end` | float | Yes | End time in seconds |
| `text` | str | Yes | Transcribed text |
| `speaker_id` | str \| None | No | Speaker identifier |

#### StreamSegment (State/Output)

| Field | Type | Description |
|-------|------|-------------|
| `start` | float | Segment start time |
| `end` | float | Segment end time |
| `text` | str | Aggregated text |
| `speaker_id` | str \| None | Speaker identifier |
| `audio_state` | dict \| None | Audio enrichment data (v1.7+) |

### StreamingEnrichmentSession

Extends base session with real-time audio feature extraction.

```python
from pathlib import Path
from transcription.streaming_enrich import (
    StreamingEnrichmentSession,
    StreamingEnrichmentConfig,
)
from transcription.streaming import StreamConfig, StreamEventType

# Configure enrichment
config = StreamingEnrichmentConfig(
    base_config=StreamConfig(max_gap_sec=1.0),
    enable_prosody=True,       # Extract pitch, energy, rate
    enable_emotion=True,       # Extract valence, arousal
    enable_categorical_emotion=False,  # Skip categorical labels
    speaker_baseline=None      # Use absolute thresholds
)

# Initialize with audio file
wav_path = Path("normalized_audio.wav")
session = StreamingEnrichmentSession(wav_path, config)

# Process chunks (same as base session)
events = session.ingest_chunk(chunk)

# FINAL segments now have audio_state populated
for event in events:
    if event.type == StreamEventType.FINAL_SEGMENT:
        audio_state = event.segment.audio_state
        if audio_state:
            print(f"Rendering: {audio_state['rendering']}")
            # "[audio: high pitch, loud volume, positive]"

# Get session statistics
stats = session.get_stats()
print(f"Processed {stats['chunk_count']} chunks")
print(f"Finalized {stats['segment_count']} segments")
print(f"Errors: {stats['enrichment_errors']}")

# Reset for reuse with same audio file
session.reset()
```

#### StreamingEnrichmentConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_config` | StreamConfig | StreamConfig() | Base session configuration |
| `enable_prosody` | bool | False | Extract prosodic features |
| `enable_emotion` | bool | False | Extract dimensional emotion |
| `enable_categorical_emotion` | bool | False | Extract categorical labels |
| `speaker_baseline` | dict \| None | None | Baseline for speaker-relative normalization |

### LiveSemanticSession

Turn-aware semantic annotation for streaming conversations.

```python
from transcription.streaming_semantic import (
    LiveSemanticSession,
    LiveSemanticsConfig,
    SemanticUpdatePayload,
)
from transcription.streaming import StreamChunk, StreamEventType

# Configure semantics
config = LiveSemanticsConfig(
    turn_gap_sec=2.0,              # Gap to finalize turn
    context_window_turns=10,       # Keep 10 recent turns
    context_window_sec=120.0,      # Keep last 2 minutes
    enable_question_detection=True,
    enable_action_detection=True,
)

# Initialize session
session = LiveSemanticSession(config)

# Process chunks
for chunk in stream_of_chunks:
    events = session.ingest_chunk(chunk)

    for event in events:
        if event.type == StreamEventType.SEMANTIC_UPDATE:
            payload: SemanticUpdatePayload = event.semantic

            print(f"Turn {payload.turn.id} by {payload.turn.speaker_id}")
            print(f"  Keywords: {payload.keywords}")
            print(f"  Risk tags: {payload.risk_tags}")
            print(f"  Actions: {payload.actions}")
            print(f"  Questions: {payload.question_count}")

# Get context window for LLM
context = session.get_context_window()  # List of Turn objects
llm_context = session.render_context_for_llm()  # Formatted string

# Finalize stream
final_events = session.end_of_stream()
```

#### LiveSemanticsConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `turn_gap_sec` | float | 2.0 | Gap to finalize turn |
| `context_window_turns` | int | 10 | Max turns in context |
| `context_window_sec` | float | 120.0 | Max context age (seconds) |
| `enable_question_detection` | bool | True | Count questions |
| `enable_action_detection` | bool | True | Detect action items |

#### SemanticUpdatePayload

| Field | Type | Description |
|-------|------|-------------|
| `turn` | Turn | Finalized Turn object |
| `keywords` | list[str] | Extracted keywords |
| `risk_tags` | list[str] | Risk indicators |
| `actions` | list[dict] | Detected action items |
| `question_count` | int | Number of questions |
| `context_size` | int | Current context window size |

---

## Callback Interface (v1.9.0)

The v1.9.0 release introduces a standardized callback interface for event-driven downstream processing.

### StreamCallbacks Protocol

```python
from typing import Protocol
from transcription.models import Segment, Turn
from transcription.streaming_semantic import SemanticUpdatePayload

class StreamCallbacks(Protocol):
    """Protocol for streaming event callbacks (v1.9.0+).

    Implement this protocol to receive streaming events as they occur.
    All methods are optional; unimplemented methods are no-ops.
    """

    def on_partial_segment(self, segment: Segment) -> None:
        """Called when a partial segment is emitted.

        Args:
            segment: The partial segment (may change with more chunks).
        """
        ...

    def on_segment_finalized(self, segment: Segment) -> None:
        """Called when a segment is finalized (complete, won't change).

        Args:
            segment: The finalized segment with optional audio_state.
        """
        ...

    def on_speaker_turn(self, turn: Turn) -> None:
        """Called when a speaker turn is detected.

        Args:
            turn: The detected turn boundary.
        """
        ...

    def on_semantic_update(self, payload: SemanticUpdatePayload) -> None:
        """Called when semantic annotation is complete for a turn.

        Args:
            payload: Semantic data including keywords, risk tags, actions.
        """
        ...

    def on_error(self, error: Exception, recoverable: bool) -> None:
        """Called when a processing error occurs.

        Args:
            error: The exception that occurred.
            recoverable: Whether processing can continue.
        """
        ...
```

### Usage with Callbacks

```python
from transcription.streaming_enrich import StreamingEnrichmentSession

class MyCallbacks:
    """Custom callback implementation."""

    def on_segment_finalized(self, segment):
        print(f"Finalized: {segment.text}")
        # Persist to database, update UI, etc.

    def on_semantic_update(self, payload):
        if "escalation" in payload.risk_tags:
            self.alert_manager(payload.turn)

    def on_error(self, error, recoverable):
        logger.error(f"Stream error: {error}", exc_info=True)
        if not recoverable:
            self.abort_session()

# Initialize with callbacks (v1.9.0+)
session = StreamingEnrichmentSession(
    config=config,
    callbacks=MyCallbacks()  # New parameter
)

# Events automatically dispatched to callbacks
for chunk in chunks:
    session.ingest_chunk(chunk)  # Callbacks invoked internally
```

### Async Callback Support (v1.9.0+)

```python
import asyncio
from typing import Protocol

class AsyncStreamCallbacks(Protocol):
    """Async variant for non-blocking callbacks."""

    async def on_segment_finalized(self, segment: Segment) -> None: ...
    async def on_semantic_update(self, payload: SemanticUpdatePayload) -> None: ...
    async def on_error(self, error: Exception, recoverable: bool) -> None: ...

class AsyncProcessor:
    async def on_segment_finalized(self, segment):
        await self.db.insert(segment)
        await self.notify_clients(segment)

# Async session wrapper (v1.9.0+)
async with AsyncStreamingSession(config, AsyncProcessor()) as session:
    async for chunk in audio_stream:
        await session.ingest_chunk(chunk)
```

---

## WebSocket Protocol (v2.0.0)

The v2.0.0 release introduces a WebSocket-based real-time streaming API.

### Connection Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WEBSOCKET CONNECTION LIFECYCLE                        │
└─────────────────────────────────────────────────────────────────────────┘

 Client                                                      Server
   │                                                           │
   │  WS CONNECT /stream                                       │
   │──────────────────────────────────────────────────────────▶│
   │                                                           │
   │                           CONNECTED                       │
   │◀──────────────────────────────────────────────────────────│
   │                                                           │
   │  START_SESSION {config: {...}}                            │
   │──────────────────────────────────────────────────────────▶│
   │                                                           │
   │                   SESSION_STARTED {session_id}            │
   │◀──────────────────────────────────────────────────────────│
   │                                                           │
   │  AUDIO_CHUNK {data: base64, sequence: 1}                  │
   │──────────────────────────────────────────────────────────▶│
   │                                                           │
   │                   PARTIAL {segment: {...}}                │
   │◀──────────────────────────────────────────────────────────│
   │                                                           │
   │  AUDIO_CHUNK {data: base64, sequence: 2}                  │
   │──────────────────────────────────────────────────────────▶│
   │                                                           │
   │                   FINALIZED {segment: {...}}              │
   │◀──────────────────────────────────────────────────────────│
   │                                                           │
   │                   SEMANTIC_UPDATE {payload: {...}}        │
   │◀──────────────────────────────────────────────────────────│
   │                                                           │
   │  END_SESSION                                              │
   │──────────────────────────────────────────────────────────▶│
   │                                                           │
   │                   SESSION_ENDED {stats: {...}}            │
   │◀──────────────────────────────────────────────────────────│
   │                                                           │
   │  WS CLOSE                                                 │
   │◀─────────────────────────────────────────────────────────▶│
   │                                                           │
```

### Message Types

#### Client → Server

| Type | Payload | Description |
|------|---------|-------------|
| `START_SESSION` | `{config: StreamConfig}` | Initialize streaming session |
| `AUDIO_CHUNK` | `{data: string, sequence: int}` | Base64-encoded audio chunk |
| `END_SESSION` | `{}` | Finalize and close session |
| `PING` | `{timestamp: int}` | Heartbeat request |

#### Server → Client

| Type | Payload | Description |
|------|---------|-------------|
| `SESSION_STARTED` | `{session_id: string}` | Session ready |
| `PARTIAL` | `{segment: StreamSegment}` | Partial segment update |
| `FINALIZED` | `{segment: StreamSegment}` | Finalized segment |
| `SPEAKER_TURN` | `{turn: Turn}` | Speaker change detected |
| `SEMANTIC_UPDATE` | `{payload: SemanticUpdatePayload}` | Semantic annotation |
| `ERROR` | `{code: string, message: string, recoverable: bool}` | Error notification |
| `SESSION_ENDED` | `{stats: dict}` | Session statistics |
| `PONG` | `{timestamp: int}` | Heartbeat response |

### Example: WebSocket Client (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/stream');

ws.onopen = () => {
    // Start session with configuration
    ws.send(JSON.stringify({
        type: 'START_SESSION',
        config: {
            max_gap_sec: 1.0,
            enable_prosody: true,
            enable_emotion: true
        }
    }));
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);

    switch (message.type) {
        case 'SESSION_STARTED':
            console.log(`Session started: ${message.session_id}`);
            startAudioCapture();
            break;

        case 'PARTIAL':
            updateLiveCaption(message.segment.text);
            break;

        case 'FINALIZED':
            appendToTranscript(message.segment);
            break;

        case 'SEMANTIC_UPDATE':
            handleSemantics(message.payload);
            break;

        case 'ERROR':
            if (!message.recoverable) {
                handleFatalError(message);
            }
            break;
    }
};

function sendAudioChunk(audioData, sequence) {
    ws.send(JSON.stringify({
        type: 'AUDIO_CHUNK',
        data: btoa(audioData),  // Base64 encode
        sequence: sequence
    }));
}
```

### REST Endpoints (v2.0.0)

Companion REST endpoints for session management:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream/sessions` | POST | Create new session (returns session_id) |
| `/stream/sessions/{id}/status` | GET | Get session status and stats |
| `/stream/sessions/{id}` | DELETE | Force-close session |
| `/stream/config` | GET | Get default streaming configuration |

---

## Performance Characteristics

### Latency Targets

| Operation | Target (P50) | Target (P95) | Current (v1.8) |
|-----------|--------------|--------------|----------------|
| Chunk ingestion | < 1ms | < 5ms | ~0.5ms |
| Prosody extraction | < 10ms | < 30ms | ~15ms |
| Emotion extraction | < 100ms | < 200ms | ~150ms |
| Full enrichment | < 120ms | < 250ms | ~170ms |
| Semantic annotation | < 20ms | < 50ms | ~25ms |
| End-to-end (with enrichment) | < 150ms | < 300ms | ~200ms |

### Memory Usage

| Component | Per-Session | Notes |
|-----------|-------------|-------|
| StreamingSession | ~10 KB | Base state machine |
| StreamingEnrichmentSession | ~50 MB | Audio extractor + models |
| LiveSemanticSession | ~5 KB + context | Context window scales with turns |
| Emotion model (loaded) | ~2 GB | Shared across sessions |

### Throughput

| Metric | Target | Current (v1.8) |
|--------|--------|----------------|
| Concurrent sessions (GPU) | > 10 | ~8 |
| Concurrent sessions (CPU) | > 3 | ~3 |
| Chunks per second per session | > 10 | ~20 |
| Audio real-time factor | < 0.3x | ~0.25x |

### Scaling Recommendations

**Horizontal Scaling:**
- One session per audio stream
- Use message queue (Redis Streams, Kafka) for distribution
- Stateless design enables load balancing

**Vertical Scaling:**
- GPU-accelerated ASR and emotion (CUDA)
- CPU-only prosody extraction
- Separate GPU instances for high-throughput LLM

**Memory Management:**
- Use `context_window_turns` to bound context memory
- Call `session.reset()` between streams for same audio file
- Lazy model loading reduces cold start memory

---

## Examples

### Example 1: Basic Streaming

```python
"""Minimal streaming example without enrichment."""

from transcription.streaming import (
    StreamChunk,
    StreamConfig,
    StreamingSession,
    StreamEventType,
)

# Simulate ASR output chunks
chunks: list[StreamChunk] = [
    {"start": 0.0, "end": 1.5, "text": "Hello", "speaker_id": "spk_0"},
    {"start": 1.5, "end": 3.0, "text": "world", "speaker_id": "spk_0"},
    {"start": 4.5, "end": 6.0, "text": "How are you?", "speaker_id": "spk_1"},
]

# Initialize session
session = StreamingSession(StreamConfig(max_gap_sec=1.0))

# Process chunks
for chunk in chunks:
    events = session.ingest_chunk(chunk)
    for event in events:
        if event.type == StreamEventType.PARTIAL_SEGMENT:
            print(f"[PARTIAL] {event.segment.speaker_id}: {event.segment.text}")
        elif event.type == StreamEventType.FINAL_SEGMENT:
            print(f"[FINAL] {event.segment.speaker_id}: {event.segment.text}")

# Finalize
for event in session.end_of_stream():
    print(f"[FINAL] {event.segment.speaker_id}: {event.segment.text}")
```

Output:
```
[PARTIAL] spk_0: Hello
[PARTIAL] spk_0: Hello world
[FINAL] spk_0: Hello world
[PARTIAL] spk_1: How are you?
[FINAL] spk_1: How are you?
```

### Example 2: Streaming with Audio Enrichment

```python
"""Streaming with real-time prosody and emotion extraction."""

from pathlib import Path
from transcription.streaming_enrich import (
    StreamingEnrichmentSession,
    StreamingEnrichmentConfig,
)
from transcription.streaming import StreamConfig, StreamEventType

# Configuration
config = StreamingEnrichmentConfig(
    base_config=StreamConfig(max_gap_sec=1.0),
    enable_prosody=True,
    enable_emotion=True,
)

# Initialize with audio file
wav_path = Path("conversation.wav")
session = StreamingEnrichmentSession(wav_path, config)

# Simulate chunks from ASR
chunks = [
    {"start": 0.0, "end": 2.5, "text": "This is exciting!", "speaker_id": "spk_0"},
    {"start": 3.0, "end": 5.0, "text": "I agree completely.", "speaker_id": "spk_1"},
]

for chunk in chunks:
    events = session.ingest_chunk(chunk)
    for event in events:
        if event.type == StreamEventType.FINAL_SEGMENT:
            seg = event.segment
            audio = seg.audio_state or {}

            print(f"\n[FINAL] {seg.speaker_id}: {seg.text}")
            print(f"  Rendering: {audio.get('rendering', 'N/A')}")

            if prosody := audio.get("prosody"):
                print(f"  Pitch: {prosody.get('pitch', {}).get('level', 'N/A')}")
                print(f"  Energy: {prosody.get('energy', {}).get('level', 'N/A')}")

            if emotion := audio.get("emotion"):
                print(f"  Valence: {emotion.get('valence', {}).get('level', 'N/A')}")

# Finalize and get stats
for event in session.end_of_stream():
    seg = event.segment
    print(f"\n[FINAL] {seg.speaker_id}: {seg.text}")

stats = session.get_stats()
print(f"\n--- Stats ---")
print(f"Chunks: {stats['chunk_count']}")
print(f"Segments: {stats['segment_count']}")
print(f"Errors: {stats['enrichment_errors']}")
```

### Example 3: Live Semantic Annotation

```python
"""Turn-aware semantic annotation for action item detection."""

from transcription.streaming_semantic import (
    LiveSemanticSession,
    LiveSemanticsConfig,
)
from transcription.streaming import StreamChunk, StreamEventType

# Configuration
config = LiveSemanticsConfig(
    turn_gap_sec=2.0,
    context_window_turns=10,
    enable_action_detection=True,
)

session = LiveSemanticSession(config)

# Conversation chunks
chunks: list[StreamChunk] = [
    {"start": 0.0, "end": 3.0, "text": "Can you review the proposal?", "speaker_id": "spk_0"},
    {"start": 3.5, "end": 5.0, "text": "Sure, I'll do it by Friday.", "speaker_id": "spk_1"},
    {"start": 8.0, "end": 11.0, "text": "Great, also schedule a follow-up meeting.", "speaker_id": "spk_0"},
]

action_items = []

for chunk in chunks:
    events = session.ingest_chunk(chunk)

    for event in events:
        if event.type == StreamEventType.SEMANTIC_UPDATE:
            payload = event.semantic
            print(f"\nTurn: {payload.turn.speaker_id}: {payload.turn.text}")
            print(f"  Keywords: {payload.keywords}")
            print(f"  Questions: {payload.question_count}")

            if payload.actions:
                for action in payload.actions:
                    action_items.append({
                        "speaker": payload.turn.speaker_id,
                        "action": action,
                        "timestamp": payload.turn.start,
                    })
                    print(f"  ACTION: {action}")

# Finalize
for event in session.end_of_stream():
    if event.type == StreamEventType.SEMANTIC_UPDATE:
        for action in event.semantic.actions:
            action_items.append({
                "speaker": event.semantic.turn.speaker_id,
                "action": action,
            })

# Get LLM-ready context
llm_context = session.render_context_for_llm()
print(f"\n--- LLM Context ---\n{llm_context}")
print(f"\n--- Action Items ---")
for item in action_items:
    print(f"  - {item}")
```

### Example 4: Full Pipeline Integration

```python
"""Complete streaming pipeline with all layers."""

from pathlib import Path
from transcription.streaming import StreamConfig, StreamChunk, StreamEventType
from transcription.streaming_enrich import (
    StreamingEnrichmentSession,
    StreamingEnrichmentConfig,
)
from transcription.streaming_semantic import (
    LiveSemanticSession,
    LiveSemanticsConfig,
)

class StreamingPipeline:
    """Unified streaming pipeline combining enrichment and semantics."""

    def __init__(self, wav_path: Path):
        # Enrichment layer
        self.enrich_session = StreamingEnrichmentSession(
            wav_path,
            StreamingEnrichmentConfig(
                base_config=StreamConfig(max_gap_sec=1.0),
                enable_prosody=True,
                enable_emotion=True,
            )
        )

        # Semantic layer
        self.semantic_session = LiveSemanticSession(
            LiveSemanticsConfig(
                turn_gap_sec=2.0,
                context_window_turns=15,
            )
        )

        self.finalized_segments = []
        self.semantic_updates = []

    def process_chunk(self, chunk: StreamChunk) -> dict:
        """Process a single chunk through all layers."""
        result = {"partial": None, "finalized": [], "semantic": []}

        # Layer 1: Enrichment
        enrich_events = self.enrich_session.ingest_chunk(chunk)

        for event in enrich_events:
            if event.type == StreamEventType.PARTIAL_SEGMENT:
                result["partial"] = event.segment

            elif event.type == StreamEventType.FINAL_SEGMENT:
                self.finalized_segments.append(event.segment)
                result["finalized"].append(event.segment)

                # Layer 2: Semantic (process finalized segments)
                semantic_events = self.semantic_session.ingest_chunk(chunk)
                for sem_event in semantic_events:
                    if sem_event.type == StreamEventType.SEMANTIC_UPDATE:
                        self.semantic_updates.append(sem_event.semantic)
                        result["semantic"].append(sem_event.semantic)

        return result

    def finalize(self) -> dict:
        """End stream and get final results."""
        result = {"finalized": [], "semantic": []}

        # Finalize enrichment
        for event in self.enrich_session.end_of_stream():
            self.finalized_segments.append(event.segment)
            result["finalized"].append(event.segment)

        # Finalize semantics
        for event in self.semantic_session.end_of_stream():
            if event.type == StreamEventType.SEMANTIC_UPDATE:
                self.semantic_updates.append(event.semantic)
                result["semantic"].append(event.semantic)

        return result

    def get_summary(self) -> dict:
        """Get pipeline summary."""
        return {
            "segments": len(self.finalized_segments),
            "semantic_updates": len(self.semantic_updates),
            "enrich_stats": self.enrich_session.get_stats(),
            "context_window": self.semantic_session.get_context_window(),
        }

# Usage
pipeline = StreamingPipeline(Path("meeting.wav"))

for chunk in stream_of_chunks:
    result = pipeline.process_chunk(chunk)

    if result["partial"]:
        update_live_display(result["partial"].text)

    for segment in result["finalized"]:
        persist_segment(segment)

    for semantic in result["semantic"]:
        if semantic.risk_tags:
            trigger_alert(semantic)

final = pipeline.finalize()
summary = pipeline.get_summary()
```

---

## Integration Patterns

### Pattern 1: Streaming + LLM Routing

Route finalized segments to different LLM workflows based on intent.

```python
from enum import Enum
from transcription.streaming import StreamingSession, StreamEventType
from transcription.semantic import KeywordSemanticAnnotator

class WorkflowType(Enum):
    QUESTION = "question"
    ACTION_ITEM = "action"
    ESCALATION = "escalation"
    GENERAL = "general"

def route_segment(segment, annotator) -> WorkflowType:
    """Classify segment intent for routing."""
    # Quick keyword-based classification
    text_lower = segment.text.lower()

    if "?" in segment.text:
        return WorkflowType.QUESTION
    if any(kw in text_lower for kw in ["will do", "i'll", "by friday"]):
        return WorkflowType.ACTION_ITEM
    if any(kw in text_lower for kw in ["escalate", "manager", "unacceptable"]):
        return WorkflowType.ESCALATION
    return WorkflowType.GENERAL

# Process and route
session = StreamingSession()
annotator = KeywordSemanticAnnotator()
workflow_queues = {wf: [] for wf in WorkflowType}

for chunk in chunks:
    for event in session.ingest_chunk(chunk):
        if event.type == StreamEventType.FINAL_SEGMENT:
            workflow = route_segment(event.segment, annotator)
            workflow_queues[workflow].append(event.segment)

            if workflow == WorkflowType.ESCALATION:
                trigger_immediate_review(event.segment)
```

### Pattern 2: Progressive Context Building

Build rolling context for incremental LLM summarization.

```python
from collections import deque
from transcription.streaming import StreamingSession, StreamEventType

class ContextWindow:
    def __init__(self, max_turns: int = 10, max_age_sec: float = 120.0):
        self.max_turns = max_turns
        self.max_age_sec = max_age_sec
        self.turns = deque(maxlen=max_turns)

    def add(self, segment):
        self.turns.append(segment)
        self._prune_old()

    def _prune_old(self):
        if not self.turns:
            return
        cutoff = self.turns[-1].end - self.max_age_sec
        while self.turns and self.turns[0].end < cutoff:
            self.turns.popleft()

    def render_for_llm(self) -> str:
        lines = ["Recent conversation:"]
        for seg in self.turns:
            speaker = seg.speaker_id or "Unknown"
            lines.append(f"[{seg.start:.1f}s] {speaker}: {seg.text}")
        return "\n".join(lines)

# Usage
context = ContextWindow(max_turns=15, max_age_sec=180.0)
session = StreamingSession()

for chunk in chunks:
    for event in session.ingest_chunk(chunk):
        if event.type == StreamEventType.FINAL_SEGMENT:
            context.add(event.segment)

            # Trigger summary every 5 segments
            if len(context.turns) % 5 == 0:
                llm_prompt = f"{context.render_for_llm()}\n\nSummarize:"
                # summary = call_llm(llm_prompt)
```

### Pattern 3: Real-Time Metrics

Track streaming performance metrics.

```python
import time
from dataclasses import dataclass, field

@dataclass
class StreamMetrics:
    chunks_ingested: int = 0
    segments_finalized: int = 0
    segments_partial: int = 0
    semantic_updates: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    def record_event(self, event_type: str):
        if event_type == "partial":
            self.segments_partial += 1
        elif event_type == "finalized":
            self.segments_finalized += 1
        elif event_type == "semantic":
            self.semantic_updates += 1
        elif event_type == "error":
            self.errors += 1
        self.chunks_ingested += 1

    def report(self) -> dict:
        elapsed = time.time() - self.start_time
        return {
            "elapsed_sec": round(elapsed, 2),
            "chunks_per_sec": round(self.chunks_ingested / max(elapsed, 0.001), 2),
            "segments_finalized": self.segments_finalized,
            "error_rate": round(self.errors / max(self.chunks_ingested, 1), 4),
        }

# Usage with session
metrics = StreamMetrics()
session = StreamingSession()

for chunk in chunks:
    try:
        events = session.ingest_chunk(chunk)
        for event in events:
            if event.type == StreamEventType.FINAL_SEGMENT:
                metrics.record_event("finalized")
            else:
                metrics.record_event("partial")
    except Exception as e:
        metrics.record_event("error")
        logger.error(f"Stream error: {e}")

print(metrics.report())
```

---

## Monitoring and Metrics

### Key Metrics to Track

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `stream.chunk_rate` | Chunks ingested per second | < 5/s (stalled) |
| `stream.latency_p95` | End-to-end latency (P95) | > 500ms |
| `stream.error_rate` | Errors per chunk | > 1% |
| `stream.context_size` | Context window turns | > 50 (memory) |
| `enrichment.duration_ms` | Enrichment time per segment | > 300ms |
| `semantic.turn_rate` | Turns finalized per minute | N/A (informational) |

### Prometheus Metrics Example

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
chunks_total = Counter('stream_chunks_total', 'Total chunks ingested')
segments_total = Counter('stream_segments_total', 'Total segments finalized',
                         ['type'])  # partial, finalized
errors_total = Counter('stream_errors_total', 'Total stream errors',
                       ['error_type'])

# Histograms
latency_histogram = Histogram(
    'stream_latency_seconds',
    'End-to-end latency',
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
)
enrichment_histogram = Histogram(
    'enrichment_duration_seconds',
    'Enrichment processing time',
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5]
)

# Gauges
active_sessions = Gauge('stream_active_sessions', 'Number of active sessions')
context_window_size = Gauge('stream_context_window_size', 'Context window size')

# Usage in session
class InstrumentedSession:
    def ingest_chunk(self, chunk):
        chunks_total.inc()
        start = time.time()

        try:
            events = self._session.ingest_chunk(chunk)
            for event in events:
                segments_total.labels(type=event.type.value).inc()
            latency_histogram.observe(time.time() - start)
            return events
        except Exception as e:
            errors_total.labels(error_type=type(e).__name__).inc()
            raise
```

### Logging Configuration

```python
import logging

# Configure streaming logger
logging.getLogger("transcription.streaming").setLevel(logging.INFO)
logging.getLogger("transcription.streaming_enrich").setLevel(logging.DEBUG)
logging.getLogger("transcription.streaming_semantic").setLevel(logging.INFO)

# Structured logging format
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s [%(name)s] %(message)s'
))
```

---

## Related Documentation

- [ROADMAP.md](/ROADMAP.md) - v1.9.0 and v2.0.0 streaming features
- [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md) - Public API summary
- [LLM_PROMPT_PATTERNS.md](LLM_PROMPT_PATTERNS.md) - LLM integration patterns
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration management
- [PERFORMANCE.md](PERFORMANCE.md) - Performance benchmarks

---

## Changelog

| Version | Changes |
|---------|---------|
| v1.7.0 | Initial streaming enrichment and live semantics |
| v1.8.0 | Word-level alignment support in segments |
| v1.9.0 | Callback interface (planned) |
| v2.0.0 | WebSocket protocol, LLM-backed semantics (planned) |
