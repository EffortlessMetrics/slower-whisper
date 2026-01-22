# Streaming API Reference

**Version:** v2.0.0
**Last Updated:** 2026-01-21

This document provides comprehensive API documentation for slower-whisper's streaming transcription endpoints, including the WebSocket API, REST SSE API, and Python client library.

---

## Table of Contents

1. [Overview](#overview)
2. [WebSocket API](#websocket-api)
3. [REST SSE API](#rest-sse-api)
4. [Python Client](#python-client)
5. [Code Examples](#code-examples)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)

---

## Overview

slower-whisper provides two streaming APIs for real-time transcription:

| API | Protocol | Use Case | Latency |
|-----|----------|----------|---------|
| **WebSocket** | `ws://` / `wss://` | Bidirectional real-time streaming | Lowest |
| **REST SSE** | `POST` with `text/event-stream` | File upload with progressive results | Low |

### When to Use Each API

**WebSocket API** (`/stream`):
- Real-time audio capture from microphone
- Live transcription of ongoing audio streams
- Applications requiring bidirectional communication
- Lowest latency requirements

**REST SSE API** (`/transcribe/stream`):
- Transcribing uploaded audio files with progress updates
- Simpler integration (standard HTTP)
- Progressive display of results in web UIs
- No need for WebSocket infrastructure

### Architecture Overview

```
                                    +-----------------------+
  Client Audio Input                |   slower-whisper      |
         |                          |      Service          |
         v                          |                       |
+------------------+    WebSocket   |  +----------------+   |
| WebSocket Client |--------------->|  | /stream        |   |
|  (Real-time)     |<---------------|  | WebSocket      |   |
+------------------+    Events      |  +----------------+   |
                                    |                       |
+------------------+    HTTP POST   |  +----------------+   |
| HTTP Client      |--------------->|  | /transcribe/   |   |
|  (File Upload)   |<---------------|  | stream (SSE)   |   |
+------------------+    SSE Stream  |  +----------------+   |
                                    |                       |
                                    +-----------------------+
```

---

## WebSocket API

### Connection Endpoint

```
ws://localhost:8000/stream
wss://your-domain.com/stream  (with TLS)
```

### Protocol Overview

The WebSocket API uses JSON messages for bidirectional communication.

```
+--------+                                      +--------+
| Client |                                      | Server |
+--------+                                      +--------+
    |                                                |
    |  CONNECT ws://localhost:8000/stream            |
    |----------------------------------------------->|
    |                                                |
    |                        Connection Accepted     |
    |<-----------------------------------------------|
    |                                                |
    |  START_SESSION {config: {...}}                 |
    |----------------------------------------------->|
    |                                                |
    |                   SESSION_STARTED {session_id} |
    |<-----------------------------------------------|
    |                                                |
    |  AUDIO_CHUNK {data: base64, sequence: 1}       |
    |----------------------------------------------->|
    |                                                |
    |                           PARTIAL {segment}    |
    |<-----------------------------------------------|
    |                                                |
    |  AUDIO_CHUNK {data: base64, sequence: 2}       |
    |----------------------------------------------->|
    |                                                |
    |                         FINALIZED {segment}    |
    |<-----------------------------------------------|
    |                                                |
    |  END_SESSION                                   |
    |----------------------------------------------->|
    |                                                |
    |                       SESSION_ENDED {stats}    |
    |<-----------------------------------------------|
    |                                                |
    |  Connection Close                              |
    |<---------------------------------------------->|
    |                                                |
```

### Message Types

#### Client to Server Messages

| Type | Payload | Description |
|------|---------|-------------|
| `START_SESSION` | `{config: SessionConfig}` | Initialize streaming session |
| `AUDIO_CHUNK` | `{data: string, sequence: int}` | Base64-encoded audio chunk |
| `END_SESSION` | `{}` | Finalize and close session |
| `PING` | `{timestamp: int}` | Heartbeat request |

#### Server to Client Messages

| Type | Payload | Description |
|------|---------|-------------|
| `SESSION_STARTED` | `{session_id: string}` | Session initialized successfully |
| `PARTIAL` | `{segment: Segment}` | Partial (in-progress) segment |
| `FINALIZED` | `{segment: Segment}` | Final segment (will not change) |
| `SPEAKER_TURN` | `{turn: Turn}` | Speaker change detected |
| `SEMANTIC_UPDATE` | `{payload: SemanticPayload}` | Semantic annotation complete |
| `ERROR` | `{code: string, message: string, recoverable: bool}` | Error notification |
| `SESSION_ENDED` | `{stats: SessionStats}` | Session statistics |
| `PONG` | `{timestamp: int, server_timestamp: int}` | Heartbeat response |

### Session Configuration

Configuration is sent with the `START_SESSION` message:

```json
{
  "type": "START_SESSION",
  "config": {
    "max_gap_sec": 1.0,
    "enable_prosody": false,
    "enable_emotion": false,
    "enable_categorical_emotion": false,
    "sample_rate": 16000,
    "audio_format": "pcm_s16le"
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_gap_sec` | float | 1.0 | Gap threshold to finalize segment (seconds) |
| `enable_prosody` | bool | false | Extract prosodic features (pitch, energy, rate) |
| `enable_emotion` | bool | false | Extract dimensional emotion (valence, arousal) |
| `enable_categorical_emotion` | bool | false | Extract categorical emotion labels |
| `sample_rate` | int | 16000 | Expected audio sample rate in Hz |
| `audio_format` | string | "pcm_s16le" | Audio encoding format |

### Audio Format Requirements

- **Format:** PCM 16-bit signed little-endian (`pcm_s16le`)
- **Sample Rate:** 16000 Hz (16 kHz)
- **Channels:** Mono (1 channel)
- **Encoding:** Base64 for transport over WebSocket

### Event Envelope

All server messages use a consistent envelope format:

```json
{
  "event_id": 1,
  "stream_id": "str-a1b2c3d4-...",
  "type": "FINALIZED",
  "ts_server": 1706123456789,
  "segment_id": "seg-0",
  "ts_audio_start": 0.0,
  "ts_audio_end": 2.5,
  "payload": {
    "segment": {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello world",
      "speaker_id": null,
      "audio_state": null
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | int | Monotonically increasing ID per stream |
| `stream_id` | string | Unique stream identifier (`str-{uuid4}`) |
| `type` | string | Message type (see table above) |
| `ts_server` | int | Server timestamp (Unix epoch milliseconds) |
| `segment_id` | string | Segment ID (`seg-{n}`) for segment events |
| `ts_audio_start` | float | Audio timestamp start (seconds) |
| `ts_audio_end` | float | Audio timestamp end (seconds) |
| `payload` | object | Message-specific payload |

### Configuration Endpoint

Get default configuration before connecting:

```
GET /stream/config
```

Response:

```json
{
  "default_config": {
    "max_gap_sec": 1.0,
    "enable_prosody": false,
    "enable_emotion": false,
    "enable_categorical_emotion": false,
    "sample_rate": 16000,
    "audio_format": "pcm_s16le"
  },
  "supported_audio_formats": ["pcm_s16le"],
  "supported_sample_rates": [16000],
  "message_types": {
    "client": ["START_SESSION", "AUDIO_CHUNK", "END_SESSION", "PING"],
    "server": [
      "SESSION_STARTED", "PARTIAL", "FINALIZED", "SPEAKER_TURN",
      "SEMANTIC_UPDATE", "ERROR", "SESSION_ENDED", "PONG"
    ]
  }
}
```

---

## REST SSE API

### Endpoint

```
POST /transcribe/stream
Content-Type: multipart/form-data
Accept: text/event-stream
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | file | required | Audio file to transcribe |
| `model` | string | "large-v3" | Whisper model (tiny, base, small, medium, large-v3) |
| `language` | string | null | Language code (e.g., "en", "es") or null for auto-detect |
| `device` | string | "cpu" | Device for inference ("cuda" or "cpu") |
| `compute_type` | string | null | Precision override (float16, float32, int8) |
| `task` | string | "transcribe" | Task ("transcribe" or "translate" to English) |
| `enable_diarization` | bool | false | Run speaker diarization |
| `diarization_device` | string | "auto" | Device for diarization ("cuda", "cpu", "auto") |
| `min_speakers` | int | null | Minimum expected speakers (hint) |
| `max_speakers` | int | null | Maximum expected speakers (hint) |
| `overlap_threshold` | float | null | Minimum overlap ratio for speaker assignment (0.0-1.0) |
| `word_timestamps` | bool | false | Include word-level timestamps |

### Response Format

The response is a Server-Sent Events (SSE) stream with the following event types:

#### Event: `segment`

Emitted when a new segment is transcribed.

```
event: segment
data: {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world", "speaker": null, "tone": null, "audio_state": null}
```

#### Event: `segment_update`

Emitted when a segment is updated (e.g., with speaker information after diarization).

```
event: segment_update
data: {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world", "speaker": "SPEAKER_00", "tone": null, "audio_state": null}
```

#### Event: `progress`

Emitted periodically with transcription progress.

```
event: progress
data: {"percent": 45}
```

With diarization:

```
event: progress
data: {"percent": 100, "stage": "diarization"}
```

#### Event: `done`

Emitted when transcription is complete.

```
event: done
data: {"total_segments": 10, "language": "en", "file_name": "audio.wav", "speakers": ["SPEAKER_00", "SPEAKER_01"], "meta": {}}
```

#### Event: `error`

Emitted on error.

```
event: error
data: {"code": "transcription_error", "message": "Transcription failed"}
```

### Response Headers

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

---

## Python Client

### Installation

The streaming client is included in the slower-whisper package. It requires the `websockets` package for WebSocket support:

```bash
pip install websockets
```

### Import

```python
from transcription.streaming_client import (
    StreamingClient,
    StreamingConfig,
    StreamEvent,
    EventType,
    create_client,
)
```

### Quick Start

```python
import asyncio
from transcription.streaming_client import create_client

async def main():
    # Create client with default configuration
    client = create_client("ws://localhost:8000/stream")

    async with client:
        # Start session
        await client.start_session(max_gap_sec=1.0)

        # Send audio chunks
        with open("audio.raw", "rb") as f:
            while chunk := f.read(4096):
                await client.send_audio(chunk)

        # End session and get final events
        final_events = await client.end_session()

        for event in final_events:
            if event.type == "FINALIZED":
                print(f"Segment: {event.payload['segment']['text']}")

asyncio.run(main())
```

### StreamingConfig

Configuration options for the client:

```python
from transcription.streaming_client import StreamingConfig

config = StreamingConfig(
    url="ws://localhost:8000/stream",
    max_gap_sec=1.0,
    enable_prosody=False,
    enable_emotion=False,
    enable_categorical_emotion=False,
    sample_rate=16000,
    audio_format="pcm_s16le",
    reconnect_attempts=3,
    reconnect_delay=1.0,
    ping_interval=30.0,
    ping_timeout=10.0,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | str | "ws://localhost:8000/stream" | WebSocket URL |
| `max_gap_sec` | float | 1.0 | Gap threshold for segment finalization |
| `enable_prosody` | bool | False | Extract prosodic features |
| `enable_emotion` | bool | False | Extract dimensional emotion |
| `enable_categorical_emotion` | bool | False | Extract categorical emotion |
| `sample_rate` | int | 16000 | Audio sample rate in Hz |
| `audio_format` | str | "pcm_s16le" | Audio encoding format |
| `reconnect_attempts` | int | 3 | Number of reconnection attempts |
| `reconnect_delay` | float | 1.0 | Delay between reconnection attempts (seconds) |
| `ping_interval` | float | 30.0 | Interval for ping messages (0 to disable) |
| `ping_timeout` | float | 10.0 | Timeout for ping response |

### Async Iteration

Process events as they arrive using async iteration:

```python
import asyncio
from transcription.streaming_client import StreamingClient, StreamingConfig

async def main():
    config = StreamingConfig(url="ws://localhost:8000/stream")
    client = StreamingClient(config=config)

    async with client:
        await client.start_session()

        # Send audio in background
        async def send_audio():
            with open("audio.raw", "rb") as f:
                while chunk := f.read(4096):
                    await client.send_audio(chunk)
                    await asyncio.sleep(0.1)  # Simulate real-time
            await client.end_session()

        asyncio.create_task(send_audio())

        # Process events as they arrive
        async for event in client.events():
            if event.type == "PARTIAL":
                print(f"[PARTIAL] {event.payload['segment']['text']}")
            elif event.type == "FINALIZED":
                print(f"[FINAL] {event.payload['segment']['text']}")
            elif event.type == "SESSION_ENDED":
                print(f"Session ended: {event.payload['stats']}")
                break

asyncio.run(main())
```

### Callback Usage

Use callbacks for event-driven processing:

```python
import asyncio
from transcription.streaming_client import StreamingClient, StreamingConfig, StreamEvent

def on_partial(event: StreamEvent):
    print(f"[PARTIAL] {event.payload['segment']['text']}")

def on_finalized(event: StreamEvent):
    segment = event.payload['segment']
    print(f"[FINAL] {segment['text']}")
    # Persist to database, send to UI, etc.

def on_error(event: StreamEvent):
    print(f"[ERROR] {event.payload['code']}: {event.payload['message']}")
    if not event.is_recoverable:
        print("Fatal error - connection will close")

def on_session_ended(event: StreamEvent):
    stats = event.payload['stats']
    print(f"Session ended: {stats['segments_finalized']} segments in {stats['duration_sec']}s")

async def main():
    client = StreamingClient(
        config=StreamingConfig(url="ws://localhost:8000/stream"),
        on_partial=on_partial,
        on_finalized=on_finalized,
        on_error=on_error,
        on_session_ended=on_session_ended,
    )

    async with client:
        await client.start_session(enable_prosody=True)

        # Send audio
        with open("audio.raw", "rb") as f:
            while chunk := f.read(4096):
                await client.send_audio(chunk)

        await client.end_session()

asyncio.run(main())
```

### Client Properties and Methods

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `state` | ClientState | Current client state |
| `stream_id` | str | Current stream ID (after session start) |
| `is_connected` | bool | Whether client is connected |
| `has_active_session` | bool | Whether session is active |
| `stats` | ClientStats | Client statistics |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `connect()` | None | Establish WebSocket connection |
| `start_session(**config)` | StreamEvent | Start streaming session |
| `send_audio(bytes)` | None | Send audio chunk |
| `end_session()` | list[StreamEvent] | End session, get final events |
| `ping()` | StreamEvent | Send ping, get pong |
| `events()` | AsyncIterator | Async iterator over events |
| `close()` | None | Close connection |

### ClientStats

Statistics tracked by the client:

```python
@dataclass
class ClientStats:
    chunks_sent: int = 0
    bytes_sent: int = 0
    events_received: int = 0
    partials_received: int = 0
    finalized_received: int = 0
    errors_received: int = 0
    reconnect_count: int = 0
    ping_count: int = 0
    pong_count: int = 0
```

---

## Code Examples

### WebSocket Example (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/stream');

let sequence = 0;

ws.onopen = () => {
    console.log('Connected');

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
            console.log(`Session started: ${message.payload.session_id}`);
            startAudioCapture();
            break;

        case 'PARTIAL':
            // Update live caption display
            document.getElementById('live-caption').textContent =
                message.payload.segment.text;
            break;

        case 'FINALIZED':
            // Append to transcript
            const segment = message.payload.segment;
            appendToTranscript(segment);
            console.log(`[${segment.start.toFixed(2)}s] ${segment.text}`);
            break;

        case 'SEMANTIC_UPDATE':
            // Handle semantic annotations
            const payload = message.payload;
            if (payload.risk_tags && payload.risk_tags.length > 0) {
                triggerAlert(payload.risk_tags);
            }
            break;

        case 'ERROR':
            console.error(`Error: ${message.payload.code} - ${message.payload.message}`);
            if (!message.payload.recoverable) {
                ws.close();
            }
            break;

        case 'SESSION_ENDED':
            console.log(`Session ended:`, message.payload.stats);
            break;

        case 'PONG':
            console.log(`Latency: ${Date.now() - message.payload.timestamp}ms`);
            break;
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Connection closed');
};

// Send audio chunk
function sendAudioChunk(audioData) {
    sequence++;
    ws.send(JSON.stringify({
        type: 'AUDIO_CHUNK',
        data: btoa(String.fromCharCode(...new Uint8Array(audioData))),
        sequence: sequence
    }));
}

// End session
function endSession() {
    ws.send(JSON.stringify({ type: 'END_SESSION' }));
}

// Ping for keepalive
function ping() {
    ws.send(JSON.stringify({
        type: 'PING',
        timestamp: Date.now()
    }));
}
```

### REST SSE Example (curl)

```bash
# Stream transcription with progress events
curl -X POST "http://localhost:8000/transcribe/stream" \
    -F "audio=@interview.mp3" \
    -F "model=large-v3" \
    -F "language=en" \
    -F "enable_diarization=true" \
    -H "Accept: text/event-stream" \
    --no-buffer

# Example output:
# event: segment
# data: {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello everyone", ...}
#
# event: progress
# data: {"percent": 25}
#
# event: segment
# data: {"id": 1, "start": 2.8, "end": 5.2, "text": "Welcome to the meeting", ...}
#
# event: progress
# data: {"percent": 50}
#
# ...
#
# event: done
# data: {"total_segments": 42, "language": "en", ...}
```

### REST SSE Example (Python with requests)

```python
import requests

def stream_transcription(audio_path: str):
    """Stream transcription using REST SSE API."""
    url = "http://localhost:8000/transcribe/stream"

    with open(audio_path, "rb") as f:
        files = {"audio": f}
        params = {
            "model": "large-v3",
            "language": "en",
            "enable_diarization": "true",
        }

        # Stream the response
        with requests.post(url, files=files, params=params, stream=True) as response:
            response.raise_for_status()

            current_event = None
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("event:"):
                    current_event = line[7:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    handle_event(current_event, data)
                elif line == "":
                    current_event = None

def handle_event(event_type: str, data: str):
    """Handle SSE event."""
    import json
    payload = json.loads(data)

    if event_type == "segment":
        print(f"[SEGMENT] {payload['text']}")
    elif event_type == "progress":
        print(f"[PROGRESS] {payload['percent']}%")
    elif event_type == "done":
        print(f"[DONE] {payload['total_segments']} segments")
    elif event_type == "error":
        print(f"[ERROR] {payload['code']}: {payload['message']}")

# Usage
stream_transcription("interview.mp3")
```

### Python Async Example

```python
import asyncio
from transcription.streaming_client import (
    StreamingClient,
    StreamingConfig,
    EventType,
)

async def transcribe_realtime(audio_source):
    """
    Transcribe audio in real-time using WebSocket streaming.

    Args:
        audio_source: Async iterator yielding audio chunks
    """
    config = StreamingConfig(
        url="ws://localhost:8000/stream",
        max_gap_sec=1.0,
        enable_prosody=True,
        ping_interval=30.0,
    )

    client = StreamingClient(config=config)

    async with client:
        # Start session
        start_event = await client.start_session()
        print(f"Session started: {start_event.stream_id}")

        # Task to send audio
        async def send_audio():
            async for chunk in audio_source:
                await client.send_audio(chunk)
            await client.end_session()

        # Task to receive events
        async def receive_events():
            segments = []
            async for event in client.events():
                if event.type == EventType.PARTIAL.value:
                    # Show live transcription
                    print(f"\r[LIVE] {event.payload['segment']['text']}", end="")

                elif event.type == EventType.FINALIZED.value:
                    segment = event.payload['segment']
                    segments.append(segment)
                    print(f"\n[FINAL] [{segment['start']:.2f}s] {segment['text']}")

                elif event.type == EventType.ERROR.value:
                    print(f"\n[ERROR] {event.payload['message']}")
                    if not event.is_recoverable:
                        break

                elif event.type == EventType.SESSION_ENDED.value:
                    stats = event.payload['stats']
                    print(f"\n--- Session Statistics ---")
                    print(f"Duration: {stats['duration_sec']:.2f}s")
                    print(f"Segments: {stats['segments_finalized']}")
                    break

            return segments

        # Run send and receive concurrently
        send_task = asyncio.create_task(send_audio())
        segments = await receive_events()
        await send_task

        return segments

# Example with file audio
async def audio_from_file(path: str, chunk_size: int = 4096):
    """Yield audio chunks from a file."""
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk
            await asyncio.sleep(0.1)  # Simulate real-time

async def main():
    segments = await transcribe_realtime(audio_from_file("audio.raw"))
    print(f"\nTranscribed {len(segments)} segments")

asyncio.run(main())
```

### Python Callback Example

```python
import asyncio
from dataclasses import dataclass, field
from transcription.streaming_client import (
    StreamingClient,
    StreamingConfig,
    StreamEvent,
)

@dataclass
class TranscriptAccumulator:
    """Accumulates transcript segments with statistics."""

    segments: list = field(default_factory=list)
    partial_count: int = 0
    error_count: int = 0

    def on_partial(self, event: StreamEvent):
        """Handle partial segment updates."""
        self.partial_count += 1
        text = event.payload['segment']['text']
        print(f"\r[PARTIAL] {text[:50]}...", end="", flush=True)

    def on_finalized(self, event: StreamEvent):
        """Handle finalized segments."""
        segment = event.payload['segment']
        self.segments.append(segment)
        print(f"\n[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")

    def on_error(self, event: StreamEvent):
        """Handle errors."""
        self.error_count += 1
        code = event.payload['code']
        message = event.payload['message']
        recoverable = event.payload.get('recoverable', True)
        print(f"\n[ERROR] {code}: {message} (recoverable: {recoverable})")

    def on_session_ended(self, event: StreamEvent):
        """Handle session end."""
        stats = event.payload['stats']
        print(f"\n{'='*50}")
        print(f"Session Complete")
        print(f"  Duration: {stats['duration_sec']:.2f}s")
        print(f"  Chunks received: {stats['chunks_received']}")
        print(f"  Bytes received: {stats['bytes_received']}")
        print(f"  Segments finalized: {stats['segments_finalized']}")
        print(f"  Partial updates: {self.partial_count}")
        print(f"  Errors: {self.error_count}")

    def get_full_transcript(self) -> str:
        """Get full transcript text."""
        return " ".join(s['text'] for s in self.segments)

async def main():
    # Create accumulator for callbacks
    accumulator = TranscriptAccumulator()

    # Create client with callbacks
    client = StreamingClient(
        config=StreamingConfig(
            url="ws://localhost:8000/stream",
            max_gap_sec=1.0,
        ),
        on_partial=accumulator.on_partial,
        on_finalized=accumulator.on_finalized,
        on_error=accumulator.on_error,
        on_session_ended=accumulator.on_session_ended,
    )

    async with client:
        await client.start_session()

        # Send audio file
        with open("conversation.raw", "rb") as f:
            while chunk := f.read(4096):
                await client.send_audio(chunk)

        await client.end_session()

    # Access accumulated data
    print(f"\nFull transcript ({len(accumulator.segments)} segments):")
    print(accumulator.get_full_transcript())

asyncio.run(main())
```

---

## Error Handling

### WebSocket Errors

The WebSocket API uses structured error events:

```json
{
  "event_id": 5,
  "stream_id": "str-...",
  "type": "ERROR",
  "ts_server": 1706123456789,
  "payload": {
    "code": "invalid_audio_chunk",
    "message": "Invalid base64 audio data",
    "recoverable": true
  }
}
```

#### Error Codes

| Code | Recoverable | Description |
|------|-------------|-------------|
| `invalid_message` | Yes | Failed to parse client message |
| `invalid_message_type` | Yes | Unknown message type |
| `session_already_started` | Yes | Session already active |
| `no_session` | Yes | No active session |
| `invalid_audio_chunk` | Yes | Invalid audio data or sequence |
| `processing_error` | Yes | Error processing audio |
| `session_start_failed` | No | Failed to initialize session |
| `end_session_error` | No | Error ending session |
| `internal_error` | No | Unexpected server error |

### REST SSE Errors

SSE errors are sent as error events:

```
event: error
data: {"code": "transcription_error", "message": "Transcription failed"}
```

HTTP errors use standard status codes:

| Status | Description |
|--------|-------------|
| 400 | Invalid configuration or audio format |
| 413 | File too large |
| 422 | Validation error in parameters |
| 500 | Internal server error |

### Client Reconnection

The Python client supports automatic reconnection:

```python
config = StreamingConfig(
    reconnect_attempts=3,  # Number of attempts
    reconnect_delay=1.0,   # Delay between attempts
)
```

**Note:** Session state is lost on reconnect. The caller must restart the session if needed.

---

## Best Practices

### Audio Preparation

1. **Use correct format:** PCM 16-bit signed little-endian, 16 kHz mono
2. **Reasonable chunk sizes:** 4096-8192 bytes per chunk (0.13-0.26 seconds)
3. **Maintain sequence order:** Chunks must be sent in order with increasing sequence numbers

### Performance

1. **Enable only needed features:** Each enrichment layer adds latency
2. **Use GPU when available:** Set `device=cuda` for faster processing
3. **Monitor backpressure:** PARTIAL events can be dropped; FINALIZED events are never dropped

### Reliability

1. **Handle reconnection:** Implement reconnection logic for production use
2. **Process errors gracefully:** Recoverable errors allow continuation
3. **Use ping/pong:** Keep connections alive in long-running sessions

### Security

1. **Use TLS in production:** `wss://` for WebSocket, HTTPS for REST
2. **Validate input:** Ensure audio data is properly formatted before sending
3. **Set reasonable timeouts:** Prevent resource exhaustion

---

## Related Documentation

- [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) - Internal streaming architecture
- [API_SERVICE.md](API_SERVICE.md) - REST API service setup
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions

---

## Changelog

| Version | Changes |
|---------|---------|
| v2.0.0 | Initial WebSocket API, REST SSE streaming, Python client |

---

**Feedback:** Open an issue on GitHub if you have questions or suggestions for improving this documentation.
