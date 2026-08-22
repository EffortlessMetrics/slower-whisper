# Incremental ASR contract

The stable raw-audio streaming core treats each VAD utterance as one public
segment whose text may be revised until one explicit finalization boundary.
It does not append complete prefix hypotheses as new text chunks.

## Input boundary

The core accepts boundary-homogeneous audio chunks:

- 16 kHz;
- mono;
- signed 16-bit little-endian PCM;
- a trusted upstream `speech` decision covering the complete chunk;
- server-owned chunk, cadence, and active-utterance bounds.

General decoding and resampling remain outside this seam.

The public `/stream` ASR path accepts this exact PCM contract. Its route-owned
classifier supplies the speech decision, and its bounded silence window turns
the configured gap into explicit `vad_boundary` finality.

## Revision identity

One active utterance receives one stable and unique `segment_id`:

```text
segment_id = seg-00000001
revision 1: "the quick"             final=false
revision 2: "the quick brown"       final=false
revision 3: "the quick brown fox"   final=true
```

A later revision replaces the prior text projection for that segment. It is not
an append-only delta. Revisions increase monotonically. Finalization preserves
the same segment identity and names one reason:

- `vad_boundary`;
- `max_utterance`;
- `end_of_stream`.

The core validates those reasons at runtime rather than relying on static
`Literal` typing alone. Injected segment-ID factories may not return blank,
non-string, or duplicate identifiers.

## Time authority

The core owns integer sample positions. Public intervals are half-open:
`[start_sample, end_sample)`. Seconds are derived only during serialization.
Clearing an internal buffer or splitting continuous speech at the maximum
utterance duration cannot reset or overlap the public clock.

## Bounded work

Inference cadence is determined by received audio samples, not network packet
frequency. Equivalent PCM under different packet fragmentation produces the
same revision sequence, backend calls, and work metrics.

The default cadence uses deterministic geometric backoff. A 30-second
continuous utterance submits prefixes at 1, 2, 4, 8, 16, and 30 seconds. That is
six model calls and 61 seconds of submitted prefix audio, rather than 59 calls
and 914.5 seconds from a fixed half-second cadence. Tests may set the backoff
factor to one when they need a linear schedule.

Active audio never exceeds `max_utterance_samples`. The PCM buffer never exceeds
`max_utterance_samples * bytes_per_sample_frame` bytes. Finalization reuses the
last hypothesis when no new audio arrived, avoiding a duplicate model call at a
VAD boundary.

The metrics receipt records:

- model calls;
- total submitted prefix samples;
- peak active audio bytes;
- revisions emitted;
- segments finalized;
- absolute samples received.

## Failure semantics

Backend inference failure becomes `ASRInferenceError` and moves the session to
`failed`. Invalid non-text output becomes `ASROutputError`. Neither condition is
observationally equivalent to “not enough audio for a hypothesis yet.” A failed
or ended session rejects further audio with `RuntimeNotReadyError`.

Terminal inference failure clears the active PCM buffer and segment state. The
failure reason remains available through bounded public context, while provider
exception text remains available only through local exception chaining.

## Public route ownership

`service_runtime_streaming` now consumes `ASRRevision` directly. It adapts the
process-owned `ASRRuntime`, creates events through the existing session envelope
API, and never invokes the legacy `StreamingASRAdapter` or mock transcript path.
The legacy module remains mounted only for its non-WebSocket session-management
endpoints.

WebSocket event serialization, session envelope IDs, and replay remain owned by
`WebSocketStreamingSession`. Optional live enrichment remains a separate lane
and is rejected by the stable negotiation contract rather than silently
ignored.

See [WEBSOCKET_REAL_ASR_CONTRACT.md](WEBSOCKET_REAL_ASR_CONTRACT.md) for the
installed route transaction.

## Acceptance evidence

The `Incremental ASR Contract` workflow proves on Python 3.12 and 3.13:

- stable unique identity and replacement revisions;
- explicit and runtime-validated finality;
- absolute monotonic sample time;
- packetization-invariant revisions and model work;
- geometrically bounded default prefix work;
- bounded active memory and failure cleanup;
- typed inference and output failure;
- strict PCM input negotiation;
- synchronous and asynchronous backend support;
- randomized packet fragmentation;
- the same semantic transaction from the installed wheel outside the checkout.

The `WebSocket Real ASR Contract` additionally proves that the public installed
route uses this core with the process-owned engine, stable envelope identity,
bounded silence, and sanitized terminal failure.
