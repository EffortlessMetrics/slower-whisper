# AGENTS.md — Repo Agent Guide (slower-whisper)

This file tells AI agents how to work in this repository. If this file conflicts with
code, update this file.

## Quick Reference

- Project guide + invariants: `CLAUDE.md`
- Roadmap + status: `ROADMAP.md`
- Docs index: `docs/INDEX.md`
- Streaming contract: `docs/STREAMING_ARCHITECTURE.md`
- Benchmarks: `docs/BENCHMARKS.md`
- Configuration: `docs/CONFIGURATION.md`
- GPU setup: `docs/GPU_SETUP.md`

## Repo Map (high-signal surfaces)

- Pipeline (batch): `transcription/pipeline.py`
- ASR engine: `transcription/asr_engine.py`
- Streaming (core + ws): `transcription/streaming*.py`, `transcription/streaming_ws.py`
- Service/API (FastAPI): `transcription/service.py`, `transcription/service_*.py`
- Benchmarks: `transcription/benchmarks.py`, `transcription/benchmark_cli.py`
- JSON writers: `transcription/writers.py`
- Device resolution: `transcription/device.py`
- Semantic adapters: `transcription/semantic_adapter.py`
- Public API exports: `transcription/__init__.py`

## Non-Negotiable Invariants

Follow `CLAUDE.md` for the full list. The most important:

1. Device resolution is explicit (`--device auto|cpu|cuda`), preflight banner to stderr.
2. `compute_type` follows resolved device (CUDA->CPU coerces to CPU-valid types).
3. Callback failures never crash the pipeline (safe execution + `on_error`).
4. Streaming end-of-stream finalizes turns (same path as `ingest_chunk`).
5. `transcription.__version__` matches package metadata.
6. Optional deps degrade gracefully (skip with message, never crash).
7. Benchmark runners implement `BenchmarkRunner` and emit JSON receipts.
8. Semantic adapters implement `SemanticAdapter` and use factory `create_adapter`.
9. Streaming uses event envelope (`EventEnvelope`) with monotonic `event_id`.

If a change risks breaking these, stop and escalate.

## Development Commands (Local Gate)

Preferred gate (from `CLAUDE.md`):

```bash
./scripts/ci-local.sh        # full gate
./scripts/ci-local.sh fast   # quick check
nix-clean flake check        # Nix checks
nix-clean run .#verify -- --quick
```

Common tooling (run only what is relevant):

```bash
pytest
pytest -m "not heavy"        # avoid heavy ML model tests
ruff check .
black .
pyright
mypy transcription
```

Notes:
- Devshell exports `LD_LIBRARY_PATH` for wheels; use `nix-clean` to avoid ABI issues.
- Some tests are intentionally skipped via markers (gpu/diarization/heavy).

## Agent Workflow Expectations

- Read before editing. Touch the smallest surface area that satisfies the request.
- Preserve public API stability; if moving code, add shims or re-exports.
- Update docs when behavior or contracts change.
- Add/adjust tests for behavior changes, especially public API and streaming.
- Do not regress versioning: keep package version and `transcription.__version__` aligned.
- Avoid introducing hard deps; optional deps must be guarded and graceful.

## When You Add or Change Features

- Update relevant docs: `README.md`, `docs/*`, or `CLAUDE.md` if invariants shift.
- Update benchmarks/contracts when metrics or schema change.
- Ensure receipts/provenance are emitted for benchmarks and transcripts.

## Gotchas

- Raw `nix` in devshell can fail due to ABI; prefer `nix-clean`.
- Streaming is a protocol: ordering guarantees and backpressure must hold.
- `__init__.py` re-exports are the public API; keep import stability.
