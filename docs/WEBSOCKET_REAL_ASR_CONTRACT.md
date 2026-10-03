# Public WebSocket real-ASR contract

The public `/stream` route uses the process-owned ASR runtime. It does not
construct a model per connection, invoke the legacy append-shaped ASR adapter,
or emit placeholder transcript text.

## Stable input

A stable session negotiates:

- 16 kHz;
- mono;
- signed 16-bit little-endian PCM;
- positive `max_gap_sec`;
- no optional live enrichment or diarization.

Unsupported negotiation returns the non-retryable
`streaming_audio_unsupported` error. It is not reported as runtime outage.
Runtime absence or failed startup remains `runtime_not_ready`.

`AUDIO_CHUNK` contains a base64 string in `data` and an increasing integer
`sequence`. Invalid base64, partial sample frames, oversized chunks, or stale
sequence numbers produce a recoverable `invalid_audio_chunk` event before model
work.

## Classification and boundaries

The route injects a `SpeechClassifier` into each connection. Production uses a
bounded dependency-free RMS classifier; tests may inject deterministic boundary
decisions through `app.state.streaming_speech_classifier_factory`.

Silence is retained only until the configured gap boundary. The retained byte
count is also capped by the incremental core's `max_chunk_bytes`, so continuous
silence cannot grow connection memory without bound. Silence that reaches the
boundary finalizes the active utterance and advances the absolute sample clock.

## Revision events

The controller creates every transcript event through
`WebSocketStreamingSession._create_envelope()`. The session therefore remains
the authority for monotonic `event_id`, immutable `stream_id`, `ts_server`, and
its replay buffer.

`PARTIAL` and `FINALIZED` events carry:

- a stable top-level `segment_id`;
- increasing `payload.revision`;
- complete replacement `payload.text`;
- authoritative `start_sample`, `end_sample`, and `sample_rate`;
- derived second timestamps in both the envelope and `payload.segment`;
- `final` and a canonical `final_reason`.

A later revision replaces the earlier text for the same segment. It is not an
append delta. `FINALIZED` precedes `SESSION_ENDED` and is never cleared to make
room for an error event.

## Runtime ownership

Each hypothesis writes the exact active PCM prefix to a temporary mono 16 kHz
WAV and enters `ASRRuntime.transcribe_file()`. The process runtime remains the
authority for:

- the one selected engine instance;
- device and compute selection;
- inference concurrency;
- readiness and shutdown;
- typed inference and output failure.

The adapter validates that the public sample interval exactly matches the PCM
payload before runtime work.

## Failure truth

`ASRInferenceError`, `ASROutputError`, and unexpected classifier/backend
failures become one non-recoverable `ERROR` envelope. The event contains a
stable reason code and bounded context. Provider exception text remains in the
local exception chain and logs.

No public transcript event may contain `[processing...]`, `[final segment]`, a
raw provider path, or raw unexpected exception text.

Terminal failure clears bounded controller buffers, marks the protocol session
`ERROR`, emits the error through the same envelope authority, and rejects
subsequent audio.

## Deliberate boundary

This contract does not earn optional live diarization, prosody, emotion,
conversation physics, correction detection, or other live enrichment. Those
flags are rejected rather than silently ignored. General decoding and
resampling also remain outside the route; clients must provide the negotiated
PCM format.

The legacy `service_streaming` module remains available for its REST session
management endpoints. Its `/stream` and `/stream/config` routes are excluded
from the mounted router and replaced by `service_runtime_streaming`.

## Acceptance evidence

The `WebSocket Real ASR Contract` workflow proves on Python 3.12 and 3.13:

- direct route ownership with no self-editing workflow;
- exact source formatting and typing;
- one process engine reused by streaming hypotheses;
- stable revision identity and envelope ordering;
- real PCM-to-WAV shape;
- bounded silence and exact sample-span validation;
- unsupported negotiation distinguished from runtime readiness;
- sanitized inference and unexpected failure;
- finalized-before-error and finalized-before-session-ended ordering;
- the actual WebSocket transaction from the installed API wheel outside the
  checkout;
- the full non-heavy repository suite on Python 3.12.
