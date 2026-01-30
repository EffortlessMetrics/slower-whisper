# CLAUDE.md — Repo Guide (slower-whisper)

**Status:** v2.0.0-ready — ETL for conversations with streaming, semantic adapters, benchmarks
**Current focus:** v2.0 release prep, issue tracker reconciliation, docs polish
**What it is:** Audio → receipts. Schema-versioned JSON with speakers, timestamps, enrichment, and a stable contract for LLM pipelines.

If this doc disagrees with code, update it.

---

## Local Gate (Actions may be off)

```bash
./scripts/ci-local.sh        # full gate
./scripts/ci-local.sh fast   # quick check
nix-clean flake check        # Nix checks
nix-clean run .#verify -- --quick
```

**Nix caveat:** Devshell exports `LD_LIBRARY_PATH` for wheels (numpy/torch). Raw `nix …` may fail; use `nix-clean` wrapper.

**PR requirement:** Paste gate receipts before merging.

---

## Invariants (don't break)

1. **Device resolution is explicit**
   - `--device auto|cpu|cuda`
   - ASR "auto" uses CTranslate2 CUDA detection
   - Enrichment "auto" uses torch detection
   - Preflight banner → stderr (stdout stays script-friendly)

2. **compute_type follows resolved device**
   - If user didn't pass `--compute-type`, fallback CUDA→CPU coerces to CPU-valid type

3. **Callbacks never crash pipeline**
   - Exceptions caught, routed to `on_error` when possible
   - Pipeline continues

4. **Streaming end-of-stream finalizes turns**
   - Segments via `end_of_stream()` pass through same turn tracking as `ingest_chunk()`

5. **Version must not regress**
   - `transcription.__version__` == package metadata (guardrail test)

6. **Optional deps degrade gracefully**
   - Missing diarization/emotion/integrations → actionable skip/error, not crash

7. **Benchmark runners use consistent interfaces**
   - All runners implement `BenchmarkRunner` protocol
   - Results emit JSON with `receipt` for provenance tracking
   - Missing datasets → skip with clear message, not crash

8. **Semantic adapters are pluggable**
   - All adapters implement `SemanticAdapter` protocol
   - Factory pattern: `create_adapter(provider="local|openai|anthropic|noop")`
   - Cheap triage first (local keywords), expensive reasoning optional

9. **Streaming uses event envelope**
   - All events wrapped in `EventEnvelope` with `event_id`, `stream_id`, `ts_server`
   - `event_id` is monotonically increasing per stream
   - FINALIZED events never dropped; PARTIAL can be dropped under backpressure

10. **slower_whisper maintains faster-whisper API compatibility**
    - `WhisperModel`, `Segment`, `Word`, `TranscriptionInfo` match faster-whisper signatures
    - `Segment` supports both attribute access and tuple unpacking
    - Extensions (`diarize`, `enrich`, `last_transcript`) are additive, not breaking

11. **Post-processing runs in dependency order**
    - `PostProcessor` executes: safety → environment → prosody → turn-taking
    - Turn-level processors (roles, topics) run via `process_turn()` separately
    - Callbacks never crash pipeline (exceptions caught and logged)

---

## Key Surfaces

| Surface | Location |
|---------|----------|
| **faster-whisper compat** | `slower_whisper/` |
| Batch pipeline | `transcription/pipeline.py` |
| ASR engine | `transcription/asr_engine.py` |
| Streaming | `transcription/streaming*.py` |
| Device resolution | `transcription/device.py` |
| JSON I/O | `transcription/writers.py` |
| Benchmark CLI | `transcription/benchmark_cli.py` |
| Benchmarks (datasets) | `transcription/benchmarks.py` |
| Dataset manifests | `benchmarks/datasets/*/manifest.json` |
| Benchmark baselines | `benchmarks/baselines/` |
| Manifest schema | `benchmarks/manifest_schema.json` |
| Semantic adapters | `transcription/semantic_adapter.py` |
| LLM guardrails | `transcription/llm_guardrails.py` |
| WebSocket streaming | `transcription/streaming_ws.py` |
| Streaming client | `transcription/streaming_client.py` |
| Session registry | `transcription/session_registry.py` |
| Post-processing | `transcription/post_process.py` |
| Topic segmentation | `transcription/topic_segmentation.py` |
| Turn-taking policies | `transcription/turn_taking_policy.py` |
| API service | `transcription/service.py` |
| Public API | `transcription/api.py` |

---

## Sharp Edges

- `nix …` inside devshell may fail (`CXXABI_1.3.15 not found`) — use `nix-clean`
- Some tests skipped by markers (gpu/diarization/heavy) — intentional
- `uv sync --extra full --extra dev` for all features

---

## Pointers

| What | Where |
|------|-------|
| Forward plan | [ROADMAP.md](ROADMAP.md) |
| Shipped history | [CHANGELOG.md](CHANGELOG.md) |
| faster-whisper migration | [docs/FASTER_WHISPER_MIGRATION.md](docs/FASTER_WHISPER_MIGRATION.md) |
| Streaming architecture | [docs/STREAMING_ARCHITECTURE.md](docs/STREAMING_ARCHITECTURE.md) |
| GPU setup | [docs/GPU_SETUP.md](docs/GPU_SETUP.md) |
| Type policy | [docs/TYPING_POLICY.md](docs/TYPING_POLICY.md) |
| Configuration | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) |
| Benchmarks | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| Dataset manifests | [docs/DATASET_MANIFEST.md](docs/DATASET_MANIFEST.md) |
| Semantic benchmark | [docs/SEMANTIC_BENCHMARK.md](docs/SEMANTIC_BENCHMARK.md) |
| LLM semantic annotator | [docs/LLM_SEMANTIC_ANNOTATOR.md](docs/LLM_SEMANTIC_ANNOTATOR.md) |
| LibriSpeech dataset setup | [docs/LIBRISPEECH_SETUP.md](docs/LIBRISPEECH_SETUP.md) |
| AMI dataset setup | [docs/AMI_SETUP.md](docs/AMI_SETUP.md) |
| CALLHOME dataset setup | [docs/CALLHOME_SETUP.md](docs/CALLHOME_SETUP.md) |
| IEMOCAP dataset setup | [docs/IEMOCAP_SETUP.md](docs/IEMOCAP_SETUP.md) |
| Documentation index | [docs/INDEX.md](docs/INDEX.md) |
