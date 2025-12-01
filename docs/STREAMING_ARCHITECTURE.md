# Streaming Architecture (RFC)

Goal: ship a small, predictable streaming surface for slower-whisper 1.3+ without changing the offline JSON contract. This document captures the data model, lifecycle rules, and a minimal WebSocket API.

## Data Model (streaming deltas)
- **Segments**: carry `partial: bool` and `revision: int` (monotonic per segment ID). `partial=true` means text/timestamps may shift; `partial=false` locks the segment.
- **Turns**: same shape as v1.2, plus `partial: bool` and `revision`. Turn grouping can lag behind segments; updates re-use turn IDs.
- **Transcript meta**: add `stream` block with `session_id`, `seq` (global monotonic), `start_at`, and `features` (diarization, analytics, semantics).
- **Versioning**: keep `schema_version: 2` for offline compatibility; add `stream_version: 1` inside `meta.stream` to signal streaming extensions.

## When is something “final”?
- **Segments**: final when VAD closes the window and no further text/time shifts are expected (`partial=false`).
- **Turns**: final when surrounding segments are final and no speaker boundary changes are pending.
- **Speaker analytics**: final when turns are final (talk ratios, longest monologue, interruption flags). Emit interim snapshots while turns are still partial.
- **Semantic annotations**: final after the last turn in the session is finalized (or when a bounded chunk closes).

## Minimal WebSocket API

**Client → Server**
- `start`: session setup `{type:"start", sample_rate, language?, diarization?, analytics?, semantics?, auth?}`
- `media`: audio frames (binary Opus/PCM) with `seq` for ordering.
- `stop`: client-driven end of stream.

**Server → Client events**
- `ready`: acknowledged capabilities and session_id.
- `partial_segment`: `{type:"partial_segment", seq, segment: {..., partial:true, revision}}`
- `segment_finalized`: `{type:"segment_finalized", seq, segment:{..., partial:false, revision}}`
- `turn_updated`: `{type:"turn_updated", seq, turn:{..., partial:bool, revision}}`
- `analytics_snapshot`: `{type:"analytics_snapshot", seq, speakers:[...], turns:[...], completeness:{"turns_final":bool}}`
- `error`: fatal or non-fatal errors.
- `done`: no more updates; includes aggregate metadata (duration, model, device, cache hits).

**State rules**
- Events are idempotent per `(id, revision)`.
- Clients rebuild state by applying events in `seq` order; missing frames should not corrupt committed (partial=false) items.
- Backpressure: if client falls behind, server may coalesce multiple partials into the next event but must always send the latest `revision`.

## Transcript evolution (offline compatibility)
- Keep writing JSON snapshots that are valid v2 transcripts; unfinished items carry `partial:true` and the latest `revision`.
- Add `meta.stream`:
  ```json
  "meta": {
    "stream": {
      "session_id": "stream_123",
      "seq": 42,
      "stream_version": 1,
      "features": {"diarization": true, "analytics": true, "semantics": false}
    }
  }
  ```
- Downstream consumers that ignore `partial` will see only finalized content; streaming-aware clients can surface live updates using revisions.

## Python Type Definitions

The following types are defined in `transcription/streaming.py`:

### Enums

```python
class StreamEventType(Enum):
    READY = "ready"
    PARTIAL_SEGMENT = "partial_segment"
    SEGMENT_FINALIZED = "segment_finalized"
    TURN_UPDATED = "turn_updated"
    ANALYTICS_SNAPSHOT = "analytics_snapshot"
    ERROR = "error"
    DONE = "done"
```

### Configuration

```python
@dataclass(slots=True)
class StreamConfig:
    sample_rate: int = 16000
    language: str | None = None
    enable_diarization: bool = False
    enable_analytics: bool = False
    enable_semantics: bool = False
```

### Streaming Metadata

```python
@dataclass(slots=True)
class StreamMeta:
    session_id: str
    seq: int
    stream_version: int = 1
    features: dict[str, bool] = field(default_factory=dict)
```

### Partial Segments

```python
@dataclass(slots=True)
class PartialSegment:
    id: int
    start: float
    end: float
    text: str
    revision: int
    partial: bool = True
    speaker_id: str | None = None

    def finalize(self) -> Segment:
        """Convert to finalized Segment."""
        ...
```

### Events

```python
@dataclass(slots=True)
class StreamEvent:
    type: StreamEventType
    seq: int
    payload: dict | PartialSegment | Segment | None = None
```

### Session Protocol

```python
class StreamingSessionProtocol(Protocol):
    @property
    def session_id(self) -> str: ...
    @property
    def config(self) -> StreamConfig: ...
    def start(self) -> StreamEvent: ...
    def process_audio(self, audio_bytes: bytes, seq: int) -> list[StreamEvent]: ...
    def stop(self) -> StreamEvent: ...
    def get_transcript(self) -> Transcript: ...
```

## Implementation Status

| Component | Status |
|-----------|--------|
| Type definitions | ✅ Defined in v1.3.1 |
| StreamingSession class | 🚧 NotImplementedError (v2.0) |
| WebSocket server | 🚧 Not started (v2.0) |
| Audio buffering/VAD | 🚧 Not started (v2.0) |
| Event replay/apply | 🚧 NotImplementedError (v2.0) |

## Usage (v2.0+)

```python
from transcription.streaming import StreamingSession, StreamConfig

config = StreamConfig(
    sample_rate=16000,
    enable_diarization=True,
    enable_analytics=True,
)
session = StreamingSession(config)

# Start session
ready_event = session.start()
print(f"Session started: {ready_event.payload}")

# Process audio chunks
for chunk in audio_stream:
    events = session.process_audio(chunk, seq=seq)
    for event in events:
        if event.type == StreamEventType.PARTIAL_SEGMENT:
            print(f"Partial: {event.payload.text}")
        elif event.type == StreamEventType.SEGMENT_FINALIZED:
            print(f"Final: {event.payload.text}")

# Finalize
done_event = session.stop()
transcript = session.get_transcript()
```
