# Service runtime contract

Stable service mode owns one configured ASR runtime per process. Batch REST
transcription reuses that runtime. Direct Python and CLI operations keep their
operation-owned engine behavior unless an internal runtime explicitly supplies
an engine.

## One process, one resolved profile

A service process resolves one ASR profile before model construction:

- model;
- device (`cpu` or `cuda`);
- compute type;
- language and task defaults;
- beam, VAD, and word-timestamp settings;
- maximum concurrent inference count.

Unresolved service devices are rejected before the profile is stored. Startup
prints the resolved model, device, compute type, and concurrency limit to
stderr. Standard output remains available for script-facing output.

The process runtime is created through FastAPI lifespan, starts once, and closes
once. A failed model initialization leaves the HTTP process live but the runtime
not ready. Ordered backend attempts remain observable after failure.

## Health semantics

### Liveness

`GET /health/live` proves only that the process and HTTP event loop can respond.
It does not load a model or run inference.

### Readiness

`GET /health/ready` succeeds only when:

- ffmpeg is available;
- required installed package resources are present;
- the process-owned runtime is ready.

The response includes both the configured profile and the backend-selected
profile. That distinction preserves a requested CUDA profile and an actual
CPU fallback without rewriting history. Importability, CUDA diagnostics, and
disk pressure remain visible but do not override the state of an already-ready
runtime.

There is no inference-bearing health endpoint in this contract. A deployment
probe that runs real audio must have its own authentication, rate, and capacity
policy before it can share the production inference slot.

## Batch REST semantics

`POST /transcribe` resolves the process runtime before saving or normalizing the
upload. Omitted ASR options inherit the process profile. An explicit model,
device, compute type, language, task, or word-timestamp setting must match that
profile; a mismatch returns `409` before upload or model work. Beam and VAD
settings remain fixed to the process profile and are not request-selectable.

Request-scoped post-processing options such as diarization may vary because
they do not construct a second ASR model. The existing `/transcribe/stream` SSE
endpoint is preserved separately and is not claimed by this batch REST
contract.

The batch endpoint preserves fail-closed domain results across HTTP:

| Domain result | HTTP result |
|---|---:|
| invalid request or configuration | `400` or `422` |
| requested ASR profile differs from the process profile | `409` with `runtime_profile_mismatch` |
| backend unavailable, model load failed, or runtime not ready | `503` |
| inference failed | `500` with `asr_inference_failed` |
| backend output was invalid | `500` with `asr_output_invalid` |
| genuine silence | `200` with an empty segment list |

Provider exception text remains chained and logged locally. Remote responses
contain only stable reason codes and deliberately selected public context.
Validation logs retain bounded location/type information without storing the
submitted invalid value or validation message.

## Concurrency and shutdown

The runtime owns a server-configured semaphore. Batch REST requests share that
boundary and cannot exceed the configured process concurrency. A request
waiting for capacity rechecks runtime state before obtaining the engine.

Shutdown enters `stopping`, rejects queued or new work, waits for active worker
threads to finish, closes the engine once, and reaches `stopped` even when the
close hook raises.

## Provenance

Every successful file, bytes, and batch REST transcription receives one canonical receipt at `meta.receipt`. The receipt records the package version, transcript schema version, actual model/backend/device/compute selection, ordered model-load attempts, canonical configuration hash, run identity, and trusted source/build identity when the artifact contains it.

Runtime code never invokes `git` or consults the caller's current working
directory, `.git` directory, `PATH`, or runtime environment to identify its own
source. Official builds write explicit values to `transcription._build_info`
before constructing the wheel and sdist. Source checkouts and unlabelled local
builds leave those fields unknown; unknown identity is omitted rather than
guessed. The existing `git_commit` receipt field remains for compatibility but
means the source commit embedded in the installed artifact.

The receipt contains only bounded model-load fields and does not retain provider
exception text or caller paths. It validates against the bundled
`receipt-v1.schema.json`; the complete transcript remains valid against the
bundled v2 transcript schema. See [PROVENANCE.md](PROVENANCE.md) for the build
and artifact contract.

## Deliberate boundary

Direct Python and CLI operations keep their current operation-owned engine
construction. The service runtime is not a process-global singleton for library
callers.

The supported deployment topology remains one application worker per instance
while model and streaming session state are in process. Multiple workers create
independent runtimes.

## Acceptance evidence

The `Runtime Lifecycle` workflow proves on Python 3.12 and 3.13:

- one startup and one shutdown;
- failed startup remains live but not ready;
- ordered load attempts remain visible after failure;
- actual backend selection appears in readiness;
- unresolved service profiles fail before storage;
- inference concurrency stays within the configured bound;
- shutdown drains active worker threads and rejects queued work;
- preflight output goes to stderr;
- the built API wheel passes the same lifecycle and readiness transaction from
  outside the checkout.

The `REST Runtime Contract` workflow proves on Python 3.12 and 3.13:

- sequential requests reuse one engine instance;
- overlapping HTTP requests remain within the runtime concurrency bound;
- profile drift is rejected before file save or inference;
- request-scoped diarization does not create a false ASR mismatch;
- startup, inference, and output failures retain typed sanitized HTTP results;
- genuine silence remains a successful empty transcript;
- validation logs exclude submitted values and messages;
- exactly one batch route and the separate SSE route remain mounted;
- the installed API wheel passes reuse, mismatch, silence, and typed-failure
  transactions outside the checkout.

The `Provenance Receipt Contract` and `Artifact Integrity` workflows prove:

- file, bytes, and REST results attach the same canonical receipt;
- selected fallback values and ordered attempts are preserved;
- config hashes are canonical outside per-run volatile fields;
- a fake caller repository and fake `git` executable cannot change identity;
- identical build inputs produce identical metadata bytes;
- wheel and sdist contain the intended generated build module;
- installed receipts and transcripts validate against installed schemas.
