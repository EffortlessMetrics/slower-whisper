# Service runtime lifecycle contract

Stable service mode owns one configured ASR runtime per process. This contract
makes process lifecycle and readiness truthful without changing the ownership
model of direct Python or CLI calls.

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

## Concurrency

The runtime owns a server-configured inference semaphore. Callers using the
runtime boundary cannot exceed that process limit. Queueing, request admission,
and endpoint integration remain separate decisions.

## Deliberate boundary

This change does not yet make REST transcription reuse the process-owned engine.
It also does not attach provenance receipts automatically. Those integrations
remain in issue #623 and require their own cross-surface and installed-artifact
acceptance.

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
- preflight output goes to stderr;
- the built API wheel passes the same lifecycle and readiness transaction from
  outside the checkout.
