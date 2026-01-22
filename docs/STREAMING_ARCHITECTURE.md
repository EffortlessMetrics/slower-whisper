# Streaming Architecture

**Version:** v1.9.0 (with v2.0.0 planned features)
**Last Updated:** 2026-01-21

This document provides comprehensive documentation for slower-whisper's streaming architecture, covering the real-time transcription pipeline, enrichment layers, and semantic annotation.

---

## Table of Contents

1. [Overview](#overview)
2. [Message Flow](#message-flow)
3. [Event Types](#event-types)
4. [Session Classes](#session-classes)
5. [Event Callback API (v1.9.0)](#event-callback-api-v190)
6. [WebSocket Protocol (v2.0.0)](#websocket-protocol-v200)
7. [Performance Characteristics](#performance-characteristics)
8. [Examples](#examples)
9. [Integration Patterns](#integration-patterns)
10. [Monitoring and Metrics](#monitoring-and-metrics)
11. [Event Envelope Specification (v2.0.0)](#event-envelope-specification-v200)
    - [Event Envelope Structure](#event-envelope-structure)
    - [ID Contracts](#id-contracts)
    - [Event Types Specification](#event-types-specification)
    - [Ordering Guarantees](#ordering-guarantees)
    - [Backpressure Contract](#backpressure-contract)
    - [Resume Contract](#resume-contract-v20--best-effort)
    - [Security Posture](#security-posture-v20)
    - [JSON Schema Reference](#json-schema-reference)

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

## Event Callback API (v1.9.0)

The v1.9.0 release introduces a standardized callback interface for event-driven downstream processing. The callback API enables real-time reactive patterns like logging, persistence, alerting, and UI updates without polling or manual event processing.

### Overview

**Key Design Principles:**

1. **Protocol-Based**: Type-safe callback interface using Python `Protocol`
2. **Optional Methods**: Implement only the callbacks you need
3. **Exception Isolation**: Callback errors never crash the streaming pipeline
4. **Sync-First**: Synchronous callbacks for simplicity (async support planned for v2.0)

**Architecture:**

```
StreamChunk → StreamingEnrichmentSession → Events → Callbacks → Your Code
                      │                        │
                      │                   on_segment_finalized()
                      │                   on_speaker_turn()
                      │                   on_semantic_update()
                      │                   on_error()
                      │
                invoke_callback_safely() catches exceptions
```

---

### StreamCallbacks Protocol

The `StreamCallbacks` protocol defines four optional callback methods for different event types:

```python
from typing import Protocol
from transcription.streaming import StreamSegment
from transcription.streaming_semantic import SemanticUpdatePayload
from transcription.streaming_callbacks import StreamingError

class StreamCallbacks(Protocol):
    """Protocol for streaming event callbacks.

    Implement this protocol to receive real-time notifications as
    segments are finalized, speaker turns are detected, and semantic
    annotations are computed.

    All callbacks are optional - implement only what you need.
    Unimplemented methods default to no-op behavior.

    Important: Callback implementations must not raise exceptions.
    If a callback raises, it will be caught, logged, and on_error
    will be invoked. The streaming pipeline continues regardless.
    """

    def on_segment_finalized(self, segment: StreamSegment) -> None:
        """Called when a segment is finalized with enrichment complete.

        This is the primary callback for consuming enriched segments.
        The segment will have audio_state populated if enrichment was
        enabled and succeeded.

        Args:
            segment: The finalized segment with optional audio_state.
                     Guaranteed to have start, end, text, and speaker_id.
        """
        ...

    def on_speaker_turn(self, turn: dict) -> None:
        """Called when a speaker turn is detected.

        A turn is a contiguous sequence of segments from the same speaker.
        This callback fires when a turn boundary is detected (speaker change
        or long pause).

        Args:
            turn: Turn dictionary with keys:
                - id: Turn identifier (e.g., "turn_0")
                - speaker_id: Speaker identifier
                - start: Turn start time in seconds
                - end: Turn end time in seconds
                - segment_ids: List of segment IDs in this turn
                - text: Concatenated text from all segments
        """
        ...

    def on_semantic_update(self, payload: SemanticUpdatePayload) -> None:
        """Called when semantic annotations are computed for a turn.

        Semantic updates include keywords, risk tags, and action items
        extracted from the turn text.

        Args:
            payload: SemanticUpdatePayload with:
                - turn: The annotated Turn object
                - keywords: List of extracted keywords
                - risk_tags: List of detected risk flags
                - actions: List of action items
                - context_summary: Recent conversation context
        """
        ...

    def on_error(self, error: StreamingError) -> None:
        """Called when an error occurs during streaming.

        This includes enrichment failures, callback exceptions, and
        other recoverable errors. The streaming pipeline continues
        after recoverable errors.

        Args:
            error: StreamingError with exception details and context.
        """
        ...
```

---

### StreamingError Dataclass

The `StreamingError` dataclass provides structured error context:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class StreamingError:
    """Error context for callback error handling.

    Attributes:
        exception: The exception that occurred.
        context: Human-readable context about where the error occurred.
        segment_start: Start time of segment being processed (if applicable).
        segment_end: End time of segment being processed (if applicable).
        recoverable: Whether the pipeline can continue after this error.
    """

    exception: Exception
    context: str
    segment_start: float | None = None
    segment_end: float | None = None
    recoverable: bool = True
```

**Example Error Objects:**

```python
# Enrichment error (recoverable)
StreamingError(
    exception=RuntimeError("Prosody extraction failed"),
    context="Enrichment completed with partial errors",
    segment_start=12.5,
    segment_end=15.8,
    recoverable=True
)

# Callback exception (recoverable)
StreamingError(
    exception=DatabaseError("Connection timeout"),
    context="Exception in callback on_segment_finalized",
    segment_start=None,
    segment_end=None,
    recoverable=True
)
```

---

### invoke_callback_safely Helper

The `invoke_callback_safely` function ensures callback exceptions never crash the pipeline:

```python
def invoke_callback_safely(
    callbacks: StreamCallbacks | object | None,
    method_name: str,
    *args,
    **kwargs,
) -> bool:
    """Invoke a callback method safely, catching and logging any exceptions.

    This function ensures that callback exceptions never crash the streaming
    pipeline. If a callback raises, it's logged and on_error is invoked
    (if available and not the failing method).

    Args:
        callbacks: The callbacks object (may be None).
        method_name: Name of the method to invoke (e.g., "on_segment_finalized").
        *args: Positional arguments to pass to the callback.
        **kwargs: Keyword arguments to pass to the callback.

    Returns:
        True if the callback was invoked successfully, False if it failed
        or callbacks was None.
    """
```

**Internal Behavior:**

1. Returns `False` if `callbacks` is `None` or method doesn't exist
2. Invokes the callback method with provided arguments
3. If callback raises:
   - Logs the exception with full traceback
   - Constructs a `StreamingError` object
   - Invokes `on_error(error)` (if available and not the failing method)
   - Returns `False` (pipeline continues)

**Example Usage in Session:**

```python
# In StreamingEnrichmentSession.ingest_chunk()
if event.type == StreamEventType.FINAL_SEGMENT:
    # Invoke callback for finalized segment
    invoke_callback_safely(
        self._callbacks,
        "on_segment_finalized",
        enriched_segment,
    )
```

---

### Integration with StreamingEnrichmentSession

The `StreamingEnrichmentSession` class integrates callbacks at key pipeline stages:

```python
from pathlib import Path
from transcription.streaming_enrich import (
    StreamingEnrichmentSession,
    StreamingEnrichmentConfig,
)
from transcription.streaming import StreamConfig

# Create callbacks object
class MyCallbacks:
    def on_segment_finalized(self, segment):
        print(f"[{segment.start:.2f}s] {segment.text}")
        self.db.insert(segment)

    def on_error(self, error):
        if not error.recoverable:
            self.alert_ops_team(error)

# Initialize session with callbacks
config = StreamingEnrichmentConfig(
    base_config=StreamConfig(max_gap_sec=1.0),
    enable_prosody=True,
    enable_emotion=True,
)

session = StreamingEnrichmentSession(
    wav_path=Path("audio.wav"),
    config=config,
    callbacks=MyCallbacks()  # Pass callbacks to session
)

# Process chunks - callbacks invoked automatically
for chunk in chunks:
    events = session.ingest_chunk(chunk)
    # No need to manually process events; callbacks already invoked

# Finalize stream
final_events = session.end_of_stream()
```

**Callback Invocation Points:**

| Session Method | Callback Invoked | When |
|----------------|------------------|------|
| `ingest_chunk()` | `on_segment_finalized()` | After enriching FINAL segments |
| `ingest_chunk()` | `on_error()` | On enrichment errors |
| `end_of_stream()` | `on_segment_finalized()` | For any remaining segments |
| `_enrich_stream_segment()` | `on_error()` | On enrichment failures |

---

### Example: LoggingCallbacks Implementation

The `examples/streaming/callback_demo.py` demonstrates a complete callback implementation:

```python
from dataclasses import dataclass, field
from transcription.streaming import StreamSegment
from transcription.streaming_callbacks import StreamingError

@dataclass
class LoggingCallbacks:
    """
    Example callback implementation that logs all events to console.

    Attributes:
        verbose: If True, prints detailed segment info including audio_state.
        segment_count: Running count of finalized segments.
        error_count: Running count of errors encountered.
        segments: List of all finalized segments (for post-processing).
    """

    verbose: bool = False
    segment_count: int = field(default=0, init=False)
    error_count: int = field(default=0, init=False)
    segments: list[StreamSegment] = field(default_factory=list, init=False)

    def on_segment_finalized(self, segment: StreamSegment) -> None:
        """Log finalized segment with optional audio_state."""
        self.segment_count += 1
        self.segments.append(segment)

        # Format timestamp range
        time_range = f"{segment.start:>6.2f}s - {segment.end:>6.2f}s"
        speaker = segment.speaker_id or "unknown"

        print(f"\n[FINALIZED #{self.segment_count}] {time_range} | speaker={speaker}")
        print(f'  Text: "{segment.text}"')

        # Show audio_state if present and verbose
        if segment.audio_state and self.verbose:
            rendering = segment.audio_state.get("rendering", "[no rendering]")
            print(f"  Audio: {rendering}")

            # Show extraction status
            status = segment.audio_state.get("extraction_status", {})
            prosody_status = status.get("prosody", "n/a")
            emotion_status = status.get("emotion_dimensional", "n/a")
            print(f"  Status: prosody={prosody_status}, emotion={emotion_status}")

    def on_error(self, error: StreamingError) -> None:
        """Log errors with context."""
        self.error_count += 1

        severity = "RECOVERABLE" if error.recoverable else "FATAL"
        print(f"\n[ERROR - {severity}] {error.context}")
        print(f"  Exception: {error.exception}")

        if error.segment_start is not None and error.segment_end is not None:
            print(f"  Segment: {error.segment_start:.2f}s - {error.segment_end:.2f}s")

    def print_summary(self) -> None:
        """Print summary statistics at the end of the stream."""
        print("\n" + "=" * 60)
        print("CALLBACK SUMMARY")
        print("=" * 60)
        print(f"Segments finalized: {self.segment_count}")
        print(f"Errors encountered: {self.error_count}")

        if self.segments:
            total_duration = self.segments[-1].end - self.segments[0].start
            print(f"Total duration: {total_duration:.2f}s")

            # Count segments with audio enrichment
            enriched = sum(1 for s in self.segments if s.audio_state)
            print(f"Segments with audio_state: {enriched}/{self.segment_count}")


# Usage
callbacks = LoggingCallbacks(verbose=True)
session = StreamingEnrichmentSession(
    wav_path=Path("audio.wav"),
    config=config,
    callbacks=callbacks
)

# Process stream
for chunk in chunks:
    session.ingest_chunk(chunk)

session.end_of_stream()

# Print final summary
callbacks.print_summary()
```

**Output:**

```
[FINALIZED #1]   0.00s -   2.50s | speaker=spk_0
  Text: "Hello world"
  Audio: [audio: high pitch, loud volume]
  Status: prosody=success, emotion=success

[FINALIZED #2]   3.00s -   5.50s | speaker=spk_1
  Text: "How are you today?"
  Audio: [audio: neutral pitch, moderate volume, positive]
  Status: prosody=success, emotion=success

[ERROR - RECOVERABLE] Enrichment completed with partial errors
  Exception: Failed to extract emotion
  Segment: 6.00s - 8.20s

====================================================================
CALLBACK SUMMARY
====================================================================
Segments finalized: 2
Errors encountered: 1
Total duration: 5.50s
Segments with audio_state: 2/2
```

---

### Best Practices

#### 1. Exception Handling

**DO: Let invoke_callback_safely handle exceptions**

```python
class MyCallbacks:
    def on_segment_finalized(self, segment):
        # No try/except needed - invoke_callback_safely catches exceptions
        self.db.insert(segment)  # May raise DatabaseError
        self.update_ui(segment)  # May raise ConnectionError
```

**DON'T: Wrap everything in try/except**

```python
# Not needed - adds unnecessary complexity
class MyCallbacks:
    def on_segment_finalized(self, segment):
        try:
            self.db.insert(segment)
        except Exception as e:
            logger.error(e)  # invoke_callback_safely already does this
```

#### 2. Error Monitoring

**DO: Use on_error for centralized error handling**

```python
class MyCallbacks:
    def __init__(self):
        self.error_count = 0
        self.metrics = PrometheusClient()

    def on_error(self, error):
        self.error_count += 1
        self.metrics.increment("streaming_errors", {
            "context": error.context,
            "recoverable": str(error.recoverable),
        })

        if not error.recoverable:
            self.alert_ops_team(error)
```

#### 3. Stateful Callbacks

**DO: Accumulate state for post-processing**

```python
class StatefulCallbacks:
    def __init__(self):
        self.segments = []
        self.risk_flags = []

    def on_segment_finalized(self, segment):
        self.segments.append(segment)

    def on_semantic_update(self, payload):
        if payload.risk_tags:
            self.risk_flags.append({
                "turn_id": payload.turn.id,
                "tags": payload.risk_tags,
                "timestamp": payload.turn.start,
            })

    def generate_report(self):
        return {
            "total_segments": len(self.segments),
            "total_duration": self.segments[-1].end - self.segments[0].start,
            "risk_flags": self.risk_flags,
        }
```

#### 4. Async Considerations (v2.0 Future)

**Current (v1.9): Synchronous callbacks only**

```python
class MyCallbacks:
    def on_segment_finalized(self, segment):
        # Must be synchronous - blocks streaming pipeline
        self.db.insert(segment)
```

**Future (v2.0): Async callback support planned**

```python
class AsyncCallbacks:
    async def on_segment_finalized(self, segment):
        # Non-blocking database insert
        await self.db.insert_async(segment)
        await self.notify_websocket_clients(segment)
```

**Workaround for v1.9: Use background queue**

```python
import queue
import threading

class QueuedCallbacks:
    def __init__(self):
        self.segment_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def on_segment_finalized(self, segment):
        # Non-blocking enqueue
        self.segment_queue.put(segment)

    def _worker(self):
        while True:
            segment = self.segment_queue.get()
            # Async-like processing in background thread
            self.db.insert(segment)
            self.segment_queue.task_done()
```

---

### Running the Demo

See the complete working example in `examples/streaming/callback_demo.py`:

```bash
# Basic demo with sample transcript
uv run python examples/streaming/callback_demo.py

# With custom audio and prosody enrichment
uv run python examples/streaming/callback_demo.py \
    --audio input_audio/conversation.wav \
    --enable-prosody

# Verbose mode (show full audio_state)
uv run python examples/streaming/callback_demo.py \
    --audio input_audio/conversation.wav \
    --enable-prosody \
    --verbose
```

---

### Async Callback Support (Future: v2.0.0)

Async callback support is planned for v2.0.0 to enable non-blocking downstream processing:

```python
import asyncio
from typing import Protocol

class AsyncStreamCallbacks(Protocol):
    """Async variant for non-blocking callbacks (v2.0.0+)."""

    async def on_segment_finalized(self, segment: StreamSegment) -> None: ...
    async def on_semantic_update(self, payload: SemanticUpdatePayload) -> None: ...
    async def on_error(self, error: StreamingError) -> None: ...

class AsyncProcessor:
    async def on_segment_finalized(self, segment):
        # Non-blocking database insert
        await self.db.insert(segment)
        # Non-blocking WebSocket broadcast
        await self.ws_manager.broadcast(segment)

# Async session wrapper (v2.0.0+)
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

## Event Envelope Specification (v2.0.0)

This section provides the complete protocol specification for WebSocket streaming events. All server-to-client messages use a consistent envelope format that enables reliable event ordering, resumption, and backpressure handling.

**Reference Implementation:** `transcription/streaming_ws.py`

---

### Event Envelope Structure

Every server-to-client message is wrapped in an `EventEnvelope` that provides consistent metadata:

```json
{
  "event_id": 42,
  "stream_id": "str-550e8400-e29b-41d4-a716-446655440000",
  "segment_id": "seg-007",
  "type": "FINALIZED",
  "ts_server": 1736251200123,
  "ts_audio_start": 10.5,
  "ts_audio_end": 14.2,
  "payload": {
    "segment": {
      "start": 10.5,
      "end": 14.2,
      "text": "Hello, how can I help you today?",
      "speaker_id": "spk_0",
      "audio_state": null
    }
  }
}
```

#### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_id` | integer | yes | Monotonically increasing counter per stream |
| `stream_id` | string | yes | Unique stream identifier (format: `str-{uuid4}`) |
| `segment_id` | string | no | Segment identifier (format: `seg-{seq}`), null for non-segment events |
| `type` | string | yes | Event type (see [Event Types](#event-types-specification)) |
| `ts_server` | integer | yes | Server timestamp in Unix epoch milliseconds |
| `ts_audio_start` | float | no | Audio timestamp start in seconds, null for non-audio events |
| `ts_audio_end` | float | no | Audio timestamp end in seconds, null for non-audio events |
| `payload` | object | yes | Event-type-specific payload data |

---

### ID Contracts

The protocol uses three distinct identifier types with specific guarantees:

#### stream_id

| Property | Specification |
|----------|---------------|
| **Format** | `str-{uuid4}` (e.g., `str-550e8400-e29b-41d4-a716-446655440000`) |
| **Scope** | Per WebSocket connection |
| **Uniqueness** | Globally unique across all streams |
| **Lifetime** | Created at `SESSION_STARTED`, immutable for session duration |
| **Generation** | Server-side only, using UUID v4 |

```python
# Example generation (from streaming_ws.py)
stream_id = f"str-{uuid.uuid4()}"
```

#### event_id

| Property | Specification |
|----------|---------------|
| **Format** | Positive integer starting at 1 |
| **Scope** | Per stream (scoped to `stream_id`) |
| **Uniqueness** | Unique within a single stream |
| **Ordering** | Strictly monotonically increasing |
| **Reset** | Never resets within a stream session |
| **Guarantees** | If `event_id=N` is received, all events 1 to N-1 have been sent |

```python
# Example counter increment (from streaming_ws.py)
def _next_event_id(self) -> int:
    """Generate next monotonically increasing event ID."""
    self._event_id_counter += 1
    return self._event_id_counter
```

#### segment_id

| Property | Specification |
|----------|---------------|
| **Format** | `seg-{seq}` where seq is a zero-indexed sequence number (e.g., `seg-0`, `seg-007`) |
| **Scope** | Per stream |
| **Uniqueness** | Unique within a single stream |
| **Stability** | Same `segment_id` used for all PARTIAL events until FINALIZED |
| **Correlation** | Links PARTIAL events to their final FINALIZED event |

```python
# Example generation (from streaming_ws.py)
def _next_segment_id(self) -> str:
    """Generate next segment ID."""
    seg_id = f"seg-{self._segment_seq}"
    self._segment_seq += 1
    return seg_id
```

**Segment ID Lifecycle:**

```
PARTIAL  (segment_id: seg-0)  → text: "Hello"
PARTIAL  (segment_id: seg-0)  → text: "Hello world"
FINALIZED (segment_id: seg-0) → text: "Hello world, how are you?"
PARTIAL  (segment_id: seg-1)  → text: "I'm"
...
```

---

### Event Types Specification

The server emits five event types, each with a specific purpose and payload schema:

#### PARTIAL

Emitted when ASR produces an intermediate transcription result for an in-progress segment.

```json
{
  "event_id": 5,
  "stream_id": "str-abc123",
  "segment_id": "seg-2",
  "type": "PARTIAL",
  "ts_server": 1736251200123,
  "ts_audio_start": 4.5,
  "ts_audio_end": 6.2,
  "payload": {
    "segment": {
      "start": 4.5,
      "end": 6.2,
      "text": "Thank you for",
      "speaker_id": "spk_0"
    },
    "confidence": 0.85
  }
}
```

**Payload Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `segment.start` | float | yes | Segment start time (seconds) |
| `segment.end` | float | yes | Current segment end time (seconds) |
| `segment.text` | string | yes | Current transcribed text (may change) |
| `segment.speaker_id` | string | no | Speaker identifier, null if not available |
| `confidence` | float | no | ASR confidence score (0.0-1.0) |

**Guarantees:**
- Text may grow or change with subsequent PARTIAL events
- Same `segment_id` may emit multiple PARTIAL events
- Always followed by exactly one FINALIZED event with same `segment_id`
- Can be dropped under backpressure (see [Backpressure Contract](#backpressure-contract))

---

#### FINALIZED

Emitted when a segment is complete and will not change.

```json
{
  "event_id": 8,
  "stream_id": "str-abc123",
  "segment_id": "seg-2",
  "type": "FINALIZED",
  "ts_server": 1736251203456,
  "ts_audio_start": 4.5,
  "ts_audio_end": 8.3,
  "payload": {
    "segment": {
      "start": 4.5,
      "end": 8.3,
      "text": "Thank you for calling customer support.",
      "speaker_id": "spk_0",
      "audio_state": {
        "prosody": {
          "pitch": {"level": "neutral", "mean_hz": 185.2},
          "energy": {"level": "moderate", "mean_db": -22.5}
        },
        "emotion": {
          "valence": {"level": "neutral", "score": 0.45}
        },
        "rendering": "[audio: neutral pitch, moderate volume]"
      }
    }
  }
}
```

**Payload Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `segment.start` | float | yes | Final segment start time (seconds) |
| `segment.end` | float | yes | Final segment end time (seconds) |
| `segment.text` | string | yes | Final transcribed text (immutable) |
| `segment.speaker_id` | string | no | Speaker identifier |
| `segment.audio_state` | object | no | Audio enrichment data (if enabled) |

**Guarantees:**
- Text is final and will not change
- Exactly one FINALIZED event per `segment_id`
- `audio_state` populated if enrichment was enabled and succeeded
- Never dropped under backpressure (see [Backpressure Contract](#backpressure-contract))

---

#### SPEAKER_TURN

Emitted when a speaker turn boundary is detected (speaker change or significant pause).

```json
{
  "event_id": 12,
  "stream_id": "str-abc123",
  "segment_id": null,
  "type": "SPEAKER_TURN",
  "ts_server": 1736251210789,
  "ts_audio_start": 0.0,
  "ts_audio_end": 15.5,
  "payload": {
    "turn": {
      "id": "turn-3",
      "speaker_id": "spk_1",
      "start": 0.0,
      "end": 15.5,
      "segment_ids": ["seg-0", "seg-1", "seg-2"],
      "text": "Hello, thank you for calling. How can I help you today?"
    },
    "previous_speaker": "spk_0"
  }
}
```

**Payload Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `turn.id` | string | yes | Turn identifier (format: `turn-{seq}`) |
| `turn.speaker_id` | string | yes | Speaker identifier for this turn |
| `turn.start` | float | yes | Turn start time (seconds) |
| `turn.end` | float | yes | Turn end time (seconds) |
| `turn.segment_ids` | array | yes | List of segment IDs comprising this turn |
| `turn.text` | string | yes | Concatenated text from all segments |
| `previous_speaker` | string | no | Previous speaker ID (null for first turn) |

**Guarantees:**
- Emitted after the FINALIZED event that closed the turn
- Contains references to all segment IDs in the turn
- Never dropped under backpressure

---

#### SEMANTIC_UPDATE

Emitted when semantic annotation completes for a finalized turn.

```json
{
  "event_id": 15,
  "stream_id": "str-abc123",
  "segment_id": null,
  "type": "SEMANTIC_UPDATE",
  "ts_server": 1736251215000,
  "ts_audio_start": 0.0,
  "ts_audio_end": 15.5,
  "payload": {
    "turn_id": "turn-3",
    "keywords": ["customer support", "help", "inquiry"],
    "risk_tags": [],
    "actions": [
      {
        "text": "provide assistance",
        "pattern": "imperative",
        "confidence": 0.78
      }
    ],
    "question_count": 1,
    "intent": "greeting",
    "sentiment": "neutral",
    "context_size": 3
  }
}
```

**Payload Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `turn_id` | string | yes | Reference to the annotated turn |
| `keywords` | array | yes | Extracted keywords/phrases |
| `risk_tags` | array | yes | Risk indicators (e.g., "churn_risk", "escalation") |
| `actions` | array | yes | Detected action items |
| `question_count` | integer | yes | Number of questions in turn |
| `intent` | string | no | Classified intent (if available) |
| `sentiment` | string | no | Sentiment classification |
| `context_size` | integer | yes | Current context window size |

**Guarantees:**
- Only emitted for finalized turns (not segments)
- Emitted after corresponding SPEAKER_TURN event
- May be dropped under extreme backpressure (after PARTIAL events)

---

#### ERROR

Emitted when a recoverable or fatal error occurs during processing.

```json
{
  "event_id": 20,
  "stream_id": "str-abc123",
  "segment_id": "seg-5",
  "type": "ERROR",
  "ts_server": 1736251220000,
  "ts_audio_start": 25.0,
  "ts_audio_end": 27.5,
  "payload": {
    "code": "ASR_TIMEOUT",
    "message": "ASR processing timed out for audio segment",
    "recoverable": true,
    "details": {
      "timeout_ms": 5000,
      "audio_duration_sec": 2.5
    }
  }
}
```

**Payload Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | yes | Error code identifier |
| `message` | string | yes | Human-readable error description |
| `recoverable` | boolean | yes | Whether the client can continue |
| `details` | object | no | Additional error-specific context |

**Standard Error Codes:**

| Code | Recoverable | Description |
|------|-------------|-------------|
| `ASR_TIMEOUT` | yes | ASR processing exceeded timeout |
| `ASR_FAILURE` | yes | ASR engine returned an error |
| `ENRICHMENT_FAILURE` | yes | Audio enrichment failed |
| `SEQUENCE_ERROR` | no | Chunk sequence violation |
| `BUFFER_OVERFLOW` | yes | Event buffer overflow (events dropped) |
| `RESUME_GAP` | yes | Resume failed, gap in event history |
| `SESSION_ERROR` | no | Unrecoverable session error |
| `INVALID_MESSAGE` | yes | Client message parsing failed |

**Guarantees:**
- Never dropped under backpressure
- `recoverable: false` indicates client should close connection

---

### Ordering Guarantees

The protocol provides five specific ordering guarantees that clients can rely on:

#### Guarantee 1: Monotonic event_id

> `event_id` is monotonically increasing per stream

```
event_id=1 → event_id=2 → event_id=3 → ...
```

- Clients will never receive `event_id=N` after receiving `event_id=M` where M > N
- Gaps in `event_id` indicate dropped events (PARTIAL under backpressure)
- No event with `event_id` < previously received `event_id` will ever arrive

#### Guarantee 2: PARTIAL before FINALIZED

> `PARTIAL` events for a segment arrive before its `FINALIZED` event

```
PARTIAL  (seg-0) → PARTIAL  (seg-0) → FINALIZED (seg-0)
```

- Zero or more PARTIAL events precede each FINALIZED
- Same `segment_id` used throughout the segment lifecycle
- Clients can safely replace partial text when FINALIZED arrives

#### Guarantee 3: Monotonic FINALIZED audio timestamps

> `FINALIZED` events are monotonic in `ts_audio_start`

```
FINALIZED (ts_audio_start=0.0) → FINALIZED (ts_audio_start=4.5) → FINALIZED (ts_audio_start=10.2)
```

- Audio timeline progresses forward
- Enables correct transcript ordering even if network reorders packets
- Exception: End-of-stream may finalize buffered audio retroactively

#### Guarantee 4: SPEAKER_TURN after closing FINALIZED

> `SPEAKER_TURN` events arrive after the `FINALIZED` event that closed the turn

```
FINALIZED (seg-2) → SPEAKER_TURN (turn-1, segment_ids=[seg-0, seg-1, seg-2])
```

- SPEAKER_TURN references only finalized segments
- All `segment_ids` in the turn have been finalized
- Enables reliable turn aggregation

#### Guarantee 5: No out-of-order event_id

> No event arrives with `event_id` < previously received `event_id`

- The server guarantees strict ordering at the protocol level
- Network-level reordering is resolved before delivery
- If detected client-side, indicates protocol violation

**Client Implementation Pattern:**

```python
class OrderedEventProcessor:
    def __init__(self):
        self.last_event_id = 0
        self.pending_partials: dict[str, dict] = {}

    def process_event(self, event: dict) -> None:
        event_id = event["event_id"]

        # Validate ordering guarantee #5
        if event_id <= self.last_event_id:
            raise ProtocolError(f"Out-of-order event: {event_id} <= {self.last_event_id}")

        # Detect dropped events (gap in event_id)
        if event_id > self.last_event_id + 1:
            dropped_count = event_id - self.last_event_id - 1
            self.on_events_dropped(dropped_count)

        self.last_event_id = event_id

        # Route by type
        event_type = event["type"]
        if event_type == "PARTIAL":
            self.pending_partials[event["segment_id"]] = event
        elif event_type == "FINALIZED":
            # Clear any pending partials for this segment
            self.pending_partials.pop(event["segment_id"], None)
            self.on_segment_finalized(event)
```

---

### Backpressure Contract

When the server produces events faster than the client can consume them, the backpressure contract governs what happens:

#### Configuration

| Parameter | Default | Configurable | Description |
|-----------|---------|--------------|-------------|
| `buffer_size` | 100 | yes | Maximum events buffered before drop policy activates |
| `drop_policy` | `partial_first` | yes | Strategy for dropping events when buffer full |
| `finalized_drop` | `never` | no | FINALIZED events are never dropped |

#### Drop Priority

When the event buffer reaches `buffer_size`, events are dropped in this priority order:

1. **Drop oldest `PARTIAL` events first** — Partials are superseded by FINALIZED anyway
2. **Drop oldest `SEMANTIC_UPDATE` events second** — Semantic data is supplementary
3. **`FINALIZED` and `ERROR` are never dropped** — Critical for correctness

```
Buffer: [P1, P2, F1, S1, P3, E1, F2, P4]
         ↑                              ↑
       oldest                        newest

On overflow:
  - Drop P1 first (oldest PARTIAL)
  - Then P2 (next PARTIAL)
  - Then S1 (oldest SEMANTIC_UPDATE)
  - F1, E1, F2 are protected
```

#### Backpressure Signals

The server signals backpressure conditions via ERROR events:

```json
{
  "type": "ERROR",
  "payload": {
    "code": "BUFFER_OVERFLOW",
    "message": "Event buffer overflow: 15 PARTIAL events dropped",
    "recoverable": true,
    "details": {
      "dropped_count": 15,
      "dropped_types": {"PARTIAL": 12, "SEMANTIC_UPDATE": 3},
      "buffer_size": 100
    }
  }
}
```

#### Client Mitigation Strategies

1. **Faster consumption** — Process events asynchronously, don't block on I/O
2. **Request larger buffer** — Negotiate higher `buffer_size` at session start
3. **Accept partial loss** — Design UI to handle missing intermediate updates
4. **Implement flow control** — Pause audio submission if consistently dropping

---

### Resume Contract (v2.0 — Best Effort)

The resume protocol enables clients to recover from brief disconnections without losing events:

#### Resume Flow

```
Client                                          Server
   │                                               │
   │  WS DISCONNECT (network blip)                 │
   │× ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ×│
   │                                               │
   │  WS RECONNECT                                 │
   │──────────────────────────────────────────────▶│
   │                                               │
   │  RESUME_SESSION {stream_id, last_event_id}   │
   │──────────────────────────────────────────────▶│
   │                                               │
   │  (Option A: Events in buffer)                 │
   │  Events 43, 44, 45... (replayed)             │
   │◀──────────────────────────────────────────────│
   │                                               │
   │  (Option B: Gap in buffer)                    │
   │  ERROR {code: RESUME_GAP}                     │
   │◀──────────────────────────────────────────────│
   │                                               │
```

#### Resume Request

```json
{
  "type": "RESUME_SESSION",
  "stream_id": "str-abc123",
  "last_event_id": 42
}
```

#### Successful Resume Response

If `last_event_id` is within the server's replay buffer, events are replayed:

```json
{"event_id": 43, "type": "PARTIAL", ...}
{"event_id": 44, "type": "FINALIZED", ...}
{"event_id": 45, "type": "SPEAKER_TURN", ...}
```

#### Resume Gap Error

If `last_event_id` is not in buffer (too old), the server signals a gap:

```json
{
  "event_id": 100,
  "type": "ERROR",
  "payload": {
    "code": "RESUME_GAP",
    "message": "Cannot resume: events 43-99 not in buffer",
    "recoverable": true,
    "details": {
      "missing_from": 43,
      "missing_to": 99,
      "buffer_oldest": 100,
      "recommendation": "re-request audio or accept gap"
    }
  }
}
```

#### Client Gap Handling

When a `RESUME_GAP` occurs, clients should:

1. **Accept the gap** — Continue from current point, UI shows "[gap in transcript]"
2. **Re-request audio window** — If audio is available, resubmit the gap period
3. **Start new session** — If gap is unacceptable, close and restart

```python
def handle_resume_gap(self, error: dict) -> None:
    missing_from = error["details"]["missing_from"]
    missing_to = error["details"]["missing_to"]

    if self.can_resubmit_audio(missing_from, missing_to):
        # Option 2: Re-request the audio window
        self.resubmit_audio_range(missing_from, missing_to)
    else:
        # Option 1: Accept the gap
        self.insert_gap_marker(missing_from, missing_to)
        logger.warning(f"Accepted transcript gap: events {missing_from}-{missing_to}")
```

#### Resume Buffer Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replay_buffer_size` | 1000 | Maximum events kept for replay |
| `replay_buffer_ttl_sec` | 300 | Maximum age of events in replay buffer |

---

### Security Posture (v2.0)

The v2.0 streaming protocol includes a security roadmap for production deployment:

#### v2.0 Skeleton (Current)

| Aspect | Status | Notes |
|--------|--------|-------|
| Authentication | None | Development/testing only |
| Authorization | None | All operations permitted |
| Encryption | TLS recommended | Use `wss://` in production |
| Rate limiting | Basic | 10 streams/IP default |

#### v2.1+ Planned Authentication

**Bearer Token (Header-based):**

```
GET /stream HTTP/1.1
Upgrade: websocket
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

**WebSocket Subprotocol Auth:**

```
GET /stream HTTP/1.1
Upgrade: websocket
Sec-WebSocket-Protocol: transcription.v2, auth.bearer.eyJhbGciOiJIUzI1NiIs...
```

**First-Message Auth (Fallback):**

```json
{
  "type": "AUTHENTICATE",
  "token": "eyJhbGciOiJIUzI1NiIs..."
}
```

#### Rate Limiting

| Limit | Default | Scope | Configurable |
|-------|---------|-------|--------------|
| Streams per IP | 10 | IP address | yes |
| Chunks per second | 50 | Per stream | yes |
| Bytes per second | 1 MB | Per stream | yes |
| Concurrent sessions | 100 | Server-wide | yes |

**Rate Limit Error:**

```json
{
  "type": "ERROR",
  "payload": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded: 10 streams per IP",
    "recoverable": false,
    "details": {
      "limit": 10,
      "current": 11,
      "retry_after_sec": 60
    }
  }
}
```

#### Security Recommendations

1. **Always use TLS** — Deploy behind TLS termination (`wss://`)
2. **Validate origins** — Configure allowed CORS origins
3. **Implement authentication** — Add Bearer token validation before production
4. **Monitor abuse** — Log rate limit violations and unusual patterns
5. **Segment audio storage** — If storing audio, encrypt at rest

---

### JSON Schema Reference

Complete JSON Schema definitions for validation:

#### EventEnvelope Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://slower-whisper.dev/schemas/event-envelope.json",
  "title": "EventEnvelope",
  "description": "Server-to-client event envelope for WebSocket streaming",
  "type": "object",
  "required": ["event_id", "stream_id", "type", "ts_server", "payload"],
  "properties": {
    "event_id": {
      "type": "integer",
      "minimum": 1,
      "description": "Monotonically increasing event ID per stream"
    },
    "stream_id": {
      "type": "string",
      "pattern": "^str-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
      "description": "Unique stream identifier (str-{uuid4})"
    },
    "segment_id": {
      "type": ["string", "null"],
      "pattern": "^seg-\\d+$",
      "description": "Segment identifier (seg-{seq}), null for non-segment events"
    },
    "type": {
      "type": "string",
      "enum": ["PARTIAL", "FINALIZED", "SPEAKER_TURN", "SEMANTIC_UPDATE", "ERROR", "SESSION_STARTED", "SESSION_ENDED", "PONG"],
      "description": "Event type"
    },
    "ts_server": {
      "type": "integer",
      "description": "Server timestamp in Unix epoch milliseconds"
    },
    "ts_audio_start": {
      "type": ["number", "null"],
      "minimum": 0,
      "description": "Audio timestamp start in seconds"
    },
    "ts_audio_end": {
      "type": ["number", "null"],
      "minimum": 0,
      "description": "Audio timestamp end in seconds"
    },
    "payload": {
      "type": "object",
      "description": "Event-type-specific payload"
    }
  },
  "additionalProperties": false
}
```

#### Client Message Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://slower-whisper.dev/schemas/client-message.json",
  "title": "ClientMessage",
  "description": "Client-to-server message for WebSocket streaming",
  "type": "object",
  "required": ["type"],
  "properties": {
    "type": {
      "type": "string",
      "enum": ["START_SESSION", "AUDIO_CHUNK", "END_SESSION", "PING", "RESUME_SESSION"],
      "description": "Client message type"
    }
  },
  "allOf": [
    {
      "if": {"properties": {"type": {"const": "AUDIO_CHUNK"}}},
      "then": {
        "required": ["type", "data", "sequence"],
        "properties": {
          "data": {"type": "string", "description": "Base64-encoded audio data"},
          "sequence": {"type": "integer", "minimum": 1, "description": "Chunk sequence number"}
        }
      }
    },
    {
      "if": {"properties": {"type": {"const": "START_SESSION"}}},
      "then": {
        "properties": {
          "config": {
            "type": "object",
            "properties": {
              "max_gap_sec": {"type": "number", "default": 1.0},
              "enable_prosody": {"type": "boolean", "default": false},
              "enable_emotion": {"type": "boolean", "default": false},
              "sample_rate": {"type": "integer", "default": 16000},
              "audio_format": {"type": "string", "default": "pcm_s16le"}
            }
          }
        }
      }
    },
    {
      "if": {"properties": {"type": {"const": "RESUME_SESSION"}}},
      "then": {
        "required": ["type", "stream_id", "last_event_id"],
        "properties": {
          "stream_id": {"type": "string"},
          "last_event_id": {"type": "integer", "minimum": 0}
        }
      }
    }
  ]
}
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
| v1.9.0 | Event callback API with StreamCallbacks protocol, StreamingError dataclass, invoke_callback_safely helper |
| v2.0.0 | WebSocket protocol, LLM-backed semantics, async callbacks (planned) |
| v2.0.0 (#133) | Event Envelope Specification: ID contracts, ordering guarantees, backpressure contract, resume protocol, security posture |
