# Runtime service contract

`slower-whisper` v2 service mode owns one configured ASR runtime per process. This contract is narrower than the direct Python and CLI surfaces so readiness, capacity, provenance, and streaming session ownership can remain truthful.

## Process profile

A service process has one ASR profile:

- model identifier and model revision when available;
- selected device and compute type;
- language and task defaults;
- beam, VAD, and word-timestamp settings;
- an ordered record of real backend load attempts;
- a bounded inference concurrency limit.

The runtime is created during FastAPI lifespan startup and closed once during shutdown. REST requests reuse that runtime. A request that asks for a different model, device, or compute profile is rejected before inference; stable v2 service mode does not load an unbounded second model on demand.

Direct Python and CLI calls remain operation-owned. They may create an engine for one operation or reuse one explicitly for batch work. Service ownership does not introduce a process-global singleton outside the application.

## Health semantics

### Liveness

`GET /health/live` proves only that the process and HTTP event loop can respond. It does not load a model or run inference.

### Readiness

`GET /health/ready` returns success only when:

- required package resources are available;
- ffmpeg is available;
- the configured ASR runtime reached its ready state;
- the selected model, device, and compute configuration are known.

A process may remain live while readiness returns `503`. Model-load failure, a missing backend, or a failed runtime is not downgraded to a warning.

### Deep probe

The optional deep probe performs bounded real inference for deployment qualification. It is not part of ordinary liveness or readiness polling. A silent result is a successful empty transcript; an inference failure remains a typed failure.

## Failure mapping

| Domain result | HTTP result |
|---|---:|
| Malformed request, audio, or configuration | `400` or `422` |
| Requested profile differs from the process profile | `409` |
| Backend unavailable, model load failed, or runtime not ready | `503` |
| Inference failed | `500` with `asr_inference_failed` |
| Backend output was invalid | `500` with `asr_output_invalid` |
| Genuine silence | `200` with an empty segment list |

Provider exception text remains chained and logged locally. Remote responses expose stable reason codes and bounded non-sensitive context.

## Provenance

Successful transcripts receive a receipt automatically through canonical generation metadata. The receipt records actual selected runtime values, not only requested configuration:

- package and transcript-schema versions;
- source commit and build identifier when embedded at build time;
- model and optional model revision;
- selected device and compute type;
- ordered runtime attempts;
- normalized configuration hash;
- run identifier and creation time.

The installed package never runs `git` in the caller's current working directory to infer its own source revision. Unknown build provenance is omitted rather than guessed.

## Deployment boundary

The runtime and streaming session registry are currently in process. The supported stable topology is therefore one application worker per service instance, or strict affinity to one process. Multiple workers would create independent models and independent registries; shared multi-worker state is a separate architecture change.

## Acceptance evidence

The `Runtime Contract` workflow exercises the service on Python 3.12 and 3.13, including:

- startup, failure, readiness, and shutdown transitions;
- runtime reuse and bounded concurrency;
- request-profile rejection;
- automatic, cwd-independent receipts;
- typed REST failures and silence semantics;
- installation of the built wheel with API dependencies outside the checkout;
- repeated requests proving one runtime instance serves the process.

Component tests remain useful, but this service capability is accepted only when the installed HTTP transaction passes.
