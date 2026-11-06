# WebSocket Transport Contract

Stable `/stream` delivery is owned by one bounded writer per accepted connection. The revision-aware ASR controller produces events; it does not write sockets or decide replay truth.

## Authority

`OutboundEventController` is the only normal-path caller of the WebSocket send callable. The route, ASR controller, session model, and optional event producers publish envelopes to that controller.

The controller owns:

- queued event and serialized-byte limits;
- a reserved terminal event/byte slot;
- one sequential writer task;
- partial coalescing by stable `segment_id`;
- pressure gating before more inbound audio is accepted;
- delivered-only replay history;
- explicit delivery and capacity failures;
- transport metrics retained when the connection closes.

Direct `send_json`, `send_text`, or `send_bytes` calls in the stable route are a contract failure.

## Durability

Events are classified by server policy:

- `PARTIAL` is coalescible. A newer queued partial for the same segment replaces the older queued partial. Under hard pressure, an undelivered partial may be dropped.
- Pongs and high-frequency observational updates are transient.
- session lifecycle, finality, errors, semantic/speaker updates, and other non-partial events are durable.

Durable events do not silently disappear. If durable work cannot fit inside the bounded queue, the route fails through `streaming_transport_capacity_exceeded`; the terminal reserve remains available for the final typed error.

## Replay truth

Replay is recorded only after the sole writer successfully sends an event. Event creation, queue admission, and attempted delivery are not delivery evidence.

Durable and transient histories are bounded separately. Transient partial churn therefore cannot evict finality or error history. If durable history is evicted and a cursor predates it, resume fails with `RESUME_GAP` rather than pretending continuity.

The legacy creation-time replay buffer is disabled on the stable route. Cross-connection attachment remains owned by the registry/reconnect contract; this transport slice makes the replay substrate truthful first.

## Inbound bounds

Server limits are not negotiated by the client. `/stream/config` exposes the active public envelope.

Before base64 allocation, `AUDIO_CHUNK` enforces the maximum encoded length implied by `max_decoded_audio_message_bytes`. Decoding is strict and the exact decoded length is checked again. Oversize input returns the typed terminal reason `streaming_audio_message_too_large` and closes with WebSocket code 1009.

The process also owns:

- maximum active sessions;
- maximum session duration;
- queue pressure thresholds;
- send and drain timeouts.

## Acceptance

`WebSocket Transport Contract` runs on Python 3.12 and 3.13 and proves:

- one-writer source ownership;
- coalescing by segment identity;
- durable terminal reserve;
- event and byte bounds;
- inbound pressure;
- delivered-only replay;
- transient churn isolation;
- durable gap detection;
- strict pre-decode audio limits;
- bounded process admission;
- explicit writer failure;
- installed-wheel public route behavior from an unrelated working directory.

This contract establishes bounded same-connection transport. Registry-owned reconnect, attachment epochs/tokens, and the one-reader reference client are accepted separately.