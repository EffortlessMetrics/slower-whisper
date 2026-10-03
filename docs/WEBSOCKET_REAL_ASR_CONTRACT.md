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
`sequence`. Decoding failures, oversized decoded chunks, or stale sequence
numbers produce a recoverable `invalid_audio_chunk` event before model work.
A network message may end halfway through a 16-bit sample; that byte is retained
and joined with the next message. Only an incomplete sample at `END_SESSION` is
invalid: it produces terminal `streaming_pcm_incomplete`, not padded audio or a
successful `SESSION_ENDED`. Strict base64 validation and pre-decode size limits
remain part of the transport recovery work (#626).

## Classification and boundaries

The route injects a `SpeechClassifier` into each connection. Production uses a
bounded dependency-free RMS classifier. `PCMFrameBuffer` first splits arbitrary
network messages into fixed 320-sample (20 ms) PCM frames. The normalized RMS
threshold is 0.01 and activation requires one positive frame. Tests may inject
deterministic **frame** decisions through
`app.state.streaming_speech_classifier_factory`; a decision sequence is no longer
indexed by network messages. Frame size is server-owned, not negotiated by a
client. Direct controller tests may use smaller explicit `frame_samples`.

Between messages, the framer retains fewer than 640 bytes. On clean EOF it emits
any remaining complete samples once as a short, unpadded classifier frame; it
never inserts silence or drops valid samples to fill a frame. The same PCM
therefore reaches the classifier in the same frames regardless of byte splits.
`RevisionStreamingController.framing_metrics` exposes frame size, accepted bytes,
emitted samples/frame count, and residual bytes. These are framing counters, not
claims that the frames were inferred, delivered, or replayed.

Silence is retained only until the configured gap boundary. The retained byte
count is also capped by the incremental core's `max_chunk_bytes`, so continuous
silence cannot grow connection memory without bound. Silence that reaches the
boundary finalizes the active utterance and advances the absolute sample clock.
A shorter gap is bridged into an active utterance when speech resumes; leading
silence advances the clock without becoming speech. This existing end-hysteresis
policy is now applied to deterministic frames rather than network messages.
Framing invariance does not establish energy-VAD or transcription accuracy.

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

Before promoting the public route, verification on Python 3.12 and 3.13 must cover:

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

### Framing recovery acceptance (#642)

`tests/test_streaming_pcm.py` checks residual bytes, fixed-frame equivalence,
short EOF frames, malformed final samples, input bounds, and terminal cleanup.
`tests/test_streaming_packetization.py` compares the default classifier and real
incremental core under one-message, 20 ms, uneven, random, and odd-byte
partitions with an explicit deterministic ASR backend. It checks all revision
payloads, finalized intervals/reasons, sample accounting, and model-call/work
counts, not only the final concatenated text.

These deterministic checks do not replace installed-wheel, actual FastAPI route,
or real tiny-model acceptance. The new framing path must earn those checks on
the candidate artifact before #642 is closed. One-writer delivery/replay (#626)
and cross-connection resume (#627) remain separate unfinished transactions.
