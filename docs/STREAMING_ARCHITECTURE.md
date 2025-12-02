# Streaming Architecture (v0.1)

Goal: ship a small, predictable streaming surface for slower-whisper without changing the offline JSON contract. This document captures the v0.1 data model and state rules for streaming **post-ASR text chunks** (no audio transport yet).

## Data Model

### StreamChunk (input)

```python
class StreamChunk(TypedDict):
    start: float   # seconds
    end: float
    text: str
    speaker_id: str | None
```

Chunks are **post-ASR** for v0.1. Audio/VAD/encoder streaming will layer on later.

### StreamSegment (state)

```python
@dataclass(slots=True)
class StreamSegment:
    start: float
    end: float
    text: str
    speaker_id: str | None = None
```

### StreamEvent (output)

```python
class StreamEventType(Enum):
    PARTIAL_SEGMENT = "partial_segment"  # may change as more chunks arrive
    FINAL_SEGMENT = "final_segment"      # no further changes expected

@dataclass(slots=True)
class StreamEvent:
    type: StreamEventType
    segment: StreamSegment
```

Optional future events: `TURN_BOUNDARY`, `CHUNK_BOUNDARY`.

### Configuration

```python
@dataclass(slots=True)
class StreamConfig:
    max_gap_sec: float = 1.0  # gap threshold to start a new segment
```

## State rules

- Chunks arrive with **monotonic time**. If `chunk.start < current.end`, `ingest_chunk` raises `ValueError`.
- A chunk **extends the current partial segment** when:
  - Same `speaker_id` (including both `None`), and
  - Gap `chunk.start - current.end` is `<= max_gap_sec`.
- Otherwise, the current partial is finalized and a new partial begins.
- `ingest_chunk(chunk)` returns the events produced by that chunk:
  - Zero or one `FINAL_SEGMENT` for the previous segment (if it closed).
  - Exactly one `PARTIAL_SEGMENT` for the current segment state.
- `end_of_stream()` flushes any remaining partial as a `FINAL_SEGMENT` and clears session state.

## Session surface

```python
class StreamingSession:
    def __init__(self, config: StreamConfig | None = None) -> None: ...
    def ingest_chunk(self, chunk: StreamChunk) -> list[StreamEvent]: ...
    def end_of_stream(self) -> list[StreamEvent]: ...
```

Pure Python, no asyncio or sockets. This simulates how downstream consumers will see a live transcription feed.

## Implementation Status

| Component | Status |
|-----------|--------|
| Type definitions | ✅ Implemented in v0.1 |
| StreamingSession state machine | ✅ Implemented in v0.1 |
| WebSocket server | 🚧 Not started (future) |
| Audio buffering/VAD | 🚧 Not started (future) |
| Event replay/apply | 🚧 Not started (future) |

## Usage

```python
from transcription.streaming import StreamChunk, StreamConfig, StreamingSession

session = StreamingSession(StreamConfig(max_gap_sec=1.0))

for chunk in stream_of_chunks:
    events = session.ingest_chunk(chunk)
    for event in events:
        if event.type == StreamEventType.PARTIAL_SEGMENT:
            print("Partial:", event.segment.text)
        elif event.type == StreamEventType.FINAL_SEGMENT:
            print("Final:", event.segment.text)

for event in session.end_of_stream():
    print("Final:", event.segment.text)
```
