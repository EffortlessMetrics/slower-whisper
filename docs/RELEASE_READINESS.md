# v2.0 Release Readiness Board

**Last Updated:** 2026-01-26
**Gate:** `./scripts/ci-local.sh` — all PRs require pasted receipts

This is the single source of truth for v2.0 release progress. Each milestone has:
- **Exit criteria** — what "done" means
- **Validation** — exact commands to prove it
- **Remaining work** — what's left (if any)

---

## Executive Summary

| Phase | Status | Blocking? |
|-------|--------|-----------|
| 0. Gate Determinism | ✅ Complete | No |
| 1. API Polish Bundle | ✅ Shipped (#158) | No |
| 2. Benchmark Foundations | ✅ 95% Complete | CLI wiring complete, gate mode future |
| 3. Streaming Protocol | 🔄 30% (skeleton) | Contracts + client needed |
| 4. Semantics | 🔄 50% | OpenAI + local LLM backends |
| 5. Docs & Examples | 📋 Blocked | Awaits phases 3-4 |
| 6. Release Prep | 📋 Blocked | Awaits phases 3-5 |

**Critical path:** Phase 3 (streaming contracts) → Phase 4 (semantics completion) → Phase 5 (docs) → Phase 6 (release)

---

## Phase 0: Gate Determinism

**Status:** ✅ Complete

### Exit Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `./scripts/ci-local.sh fast` completes in bounded time | ✅ | ~2-3 min typical |
| `./scripts/ci-local.sh` (full) completes in bounded time | ✅ | ~5-8 min typical |
| Receipts reflect what ran vs skipped | ✅ | Script outputs pass/fail per check |
| No hangs without stack receipt | ✅ | Uses `set -euo pipefail` |

### Validation

```bash
# Fast gate (bounded)
time ./scripts/ci-local.sh fast

# Full gate (bounded)
time ./scripts/ci-local.sh
```

### What The Gate Checks

1. **Docs sanity** — broken links, forbidden JSON keys, snippet validation
2. **Pre-commit hooks** — ruff lint + format
3. **Type-check** — mypy on transcription/ + strategic test modules
4. **Fast tests** — pytest -m "not slow and not heavy"
5. **Verification suite** — `slower-whisper-verify --quick`
6. **Nix flake check** (full mode only)
7. **Nix verify app** (full mode only)

---

## Phase 1: API Polish Bundle

**Status:** ✅ Shipped (#158)

### Exit Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `transcribe_bytes()` API exists | ✅ | `transcription/api.py` |
| `word_timestamps` REST parameter | ✅ | Config field + CLI flag |
| Word-level timestamps example | ✅ | `examples/word_timestamps/` |
| `Transcript` convenience methods | ✅ | 8 methods: `full_text`, `duration`, etc. |
| CLI + API + REST consistent | ✅ | All surfaces unified |
| Local gate passes | ✅ | Merged with receipts |

### Validation

```bash
# Verify transcribe_bytes exists
uv run python -c "from transcription.api import transcribe_bytes; print('OK')"

# Verify Transcript methods
uv run python -c "
from transcription import Transcript
methods = ['full_text', 'duration', 'word_count', 'speaker_ids',
           'get_segments_by_speaker', 'get_segment_at_time',
           'is_enriched', 'segments_in_range']
for m in methods:
    assert hasattr(Transcript, m), f'Missing {m}'
print('All Transcript methods present')
"

# Verify word_timestamps CLI flag
uv run slower-whisper transcribe --help | grep -q word-timestamps && echo "OK"
```

### Issues Closed

- [x] #70 — `transcribe_bytes()` API
- [x] #71 — `word_timestamps` REST parameter
- [x] #72 — Word-level timestamps example
- [x] #78 — `Transcript` convenience methods

---

## Phase 2: Benchmark Foundations

**Status:** ✅ 95% Complete — all runners shipped, CLI wired

### Exit Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| ASR WER runner | ✅ | `ASRBenchmarkRunner` (#186) |
| Diarization DER runner | ✅ | `DiarizationBenchmarkRunner` (#189) |
| Streaming latency runner | ✅ | `StreamingBenchmarkRunner` (#190) |
| Emotion accuracy runner | ✅ | `EmotionBenchmarkRunner` (#187) |
| Baseline file format | ✅ | `BaselineFile`, `BaselineMetric` classes |
| `benchmark run --track asr` emits JSON | ✅ | CLI wired |
| `benchmark compare` works | ✅ | Baseline comparison functional |
| CI integration (report-only) | ✅ | Phase 2 complete (#99) |
| Gate mode (`--gate`) | 📋 Future | Phase 3 after Actions CI returns |

### Validation

```bash
# Verify benchmark CLI
uv run slower-whisper benchmark --help

# List available tracks
uv run slower-whisper benchmark list

# Show infrastructure status
uv run slower-whisper benchmark status

# List baselines
uv run slower-whisper benchmark baselines
```

### Remaining Work

| Item | Status | Notes |
|------|--------|-------|
| `--gate` flag (regression blocking) | 📋 Future | After GitHub Actions CI returns |
| CALLHOME dataset integration | 📋 Future | AMI + IEMOCAP done |

### Issues Status

- [x] #95 — ASR WER runner (merged #186)
- [x] #137 — Baseline file format (merged #201)
- [x] #97 — Streaming latency harness (merged #190)
- [x] #96 — Diarization DER runner (merged #189)
- [x] #99 — CI integration (Phase 2 complete)
- [x] #94 — Dataset manifest format (complete)
- [x] #57 — CLI subcommand (complete)

---

## Phase 3: Streaming Protocol

**Status:** 🔄 30% — skeleton shipped, contracts pending

### Exit Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Event envelope spec | ✅ Design | Spec in ROADMAP.md; `EventEnvelope` in streaming_ws.py |
| WebSocket endpoint skeleton | ✅ Shipped | `streaming_ws.py` with full protocol |
| Reference Python client | ⏳ Skeleton | `streaming_client.py` incomplete |
| Client contract tests | ⏳ Not started | — |
| REST streaming endpoints | ⏳ Not started | — |
| Streaming API docs | ✅ Partial | `docs/STREAMING_ARCHITECTURE.md` |
| Ordering guarantees proven | ⏳ Not started | Contract tests needed |
| Backpressure/resume protocol | ⏳ Not started | Spec exists, impl pending |

### Validation

```bash
# Verify streaming module imports
uv run python -c "
from transcription.streaming import StreamingSession
from transcription.streaming_ws import EventEnvelope, MessageType
print('Streaming imports OK')
"

# Verify WebSocket message types
uv run python -c "
from transcription.streaming_ws import MessageType
expected = ['AUDIO_CHUNK', 'CONTROL', 'TRANSCRIPT_PARTIAL',
            'TRANSCRIPT_FINAL', 'ERROR', 'SESSION_START', 'SESSION_END']
for m in expected:
    assert hasattr(MessageType, m), f'Missing {m}'
print('All message types present')
"
```

### Remaining Work

| Item | Priority | Blocks |
|------|----------|--------|
| Create issue #133 (event envelope formal spec) | High | Contract tests |
| Create issue #134 (reference client) | High | User adoption |
| Complete `streaming_client.py` | High | Examples |
| Contract test suite for ordering | High | v2.0 release |
| REST streaming endpoints (#85) | Medium | Broad adoption |
| Incremental diarization (#86) | Low | v2.1+ |

### Issues Status

- [ ] #133 — Event envelope spec (issue not yet created)
- [ ] #134 — Reference Python client (issue not yet created)
- [x] #84 — WebSocket endpoint (skeleton shipped)
- [ ] #85 — REST streaming endpoints
- [x] #55 — Streaming API docs (partial)
- [ ] #86 — Incremental diarization (v2.1+)

---

## Phase 4: Semantics

**Status:** 🔄 50% — adapter protocol shipped, cloud LLM pending

### Exit Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| LLM annotation schema + versioning | ✅ Shipped | `SemanticAdapter` protocol, schema v0.1.0 |
| Guardrails (rate limits, cost, PII) | ✅ Shipped | `LLMGuardrails` in llm_guardrails.py |
| Anthropic provider | ✅ Shipped | `AnthropicProvider` (#188) |
| OpenAI provider | ⏳ Stub only | Needs full implementation |
| Local LLM backends (qwen/smollm) | ⏳ Not started | — |
| Golden files + contract tests | ✅ Shipped | `benchmarks/gold/semantic/` |
| Semantic quality benchmark | ✅ Shipped | Metrics compute topic F1 |
| End-to-end semantic pipeline test | ⏳ Partial | Integration needed |

### Validation

```bash
# Verify semantic adapter protocol
uv run python -c "
from transcription.semantic_adapter import SemanticAdapter, NormalizedAnnotation
print('Semantic adapter imports OK')
"

# Verify guardrails
uv run python -c "
from transcription.llm_guardrails import LLMGuardrails, GuardedLLMProvider
print('Guardrails imports OK')
"

# Verify Anthropic provider
uv run python -c "
from transcription.historian.llm_client import AnthropicProvider
print('AnthropicProvider import OK')
"

# Verify gold labels exist
ls benchmarks/gold/semantic/*.json 2>/dev/null && echo "Gold labels exist" || echo "No gold labels"
```

### Remaining Work

| Item | Priority | Blocks |
|------|----------|--------|
| OpenAI provider full implementation | High | Cloud LLM choice |
| Local LLM backend (qwen2.5-7b) | Medium | Offline mode |
| Local LLM backend (smollm) | Medium | Low-resource mode |
| Contract test suite completion | High | v2.0 release |
| End-to-end integration test | High | Confidence |

### Issues Status

- [x] #88 — LLM annotation schema (shipped #201)
- [ ] #90 — Cloud LLM interface (Anthropic done, OpenAI pending)
- [x] #91 — Guardrails (shipped)
- [x] #92 — Golden files + tests (shipped)
- [ ] #89 — Local LLM backend (not started)
- [x] #98 — Semantic quality benchmark (shipped)

---

## Phase 5: Docs & Examples

**Status:** 📋 Blocked on Phases 3-4

### Exit Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| CURRENT_STATUS.md reflects reality | ⏳ Stale | Shows issues as Open that are shipped |
| Examples run without errors | ⏳ Not verified | — |
| STREAMING_API.md complete | ⏳ Partial | v2.0 section incomplete |
| Full API reference | ✅ | API_QUICK_REFERENCE.md |
| Benchmark docs | ✅ | BENCHMARKS.md |
| Semantic docs | ✅ | SEMANTIC_BENCHMARK.md, LLM_SEMANTIC_ANNOTATOR.md |

### Remaining Work

| Item | Priority | Blocks |
|------|----------|--------|
| Update CURRENT_STATUS.md with shipped work | High | User confusion |
| Update ROADMAP.md Quick Status | High | Stale dashboard |
| Complete STREAMING_API.md | Medium | Streaming adoption |
| Verify all examples run | Medium | Onboarding |

---

## Phase 6: Release Prep

**Status:** 📋 Blocked on Phases 3-5

### Exit Criteria

| Criterion | Target | Current |
|-----------|--------|---------|
| All tests passing | ✅ | ✅ |
| ASR WER | < 5% | TBD (run benchmark) |
| Diarization DER | < 15% | TBD (run benchmark) |
| Streaming P95 | < 500ms | TBD (run benchmark) |
| Semantic Topic F1 | > 0.80 | TBD (run benchmark) |
| Coverage targets met | TBD | TBD |
| CHANGELOG.md updated | ⏳ | — |
| Version bump | ⏳ | — |
| Tag created | ⏳ | — |
| Published to PyPI | ⏳ | — |

### Validation (Final Release Gate)

```bash
# Full gate
./scripts/ci-local.sh

# Run all benchmarks
uv run slower-whisper benchmark run --track asr
uv run slower-whisper benchmark run --track diarization
uv run slower-whisper benchmark run --track streaming
uv run slower-whisper benchmark run --track semantic

# Compare against baselines
uv run slower-whisper benchmark compare --track asr
uv run slower-whisper benchmark compare --track diarization
```

---

## Issue Tracker Reconciliation

### Issues to Close (shipped but still Open in GitHub)

| Issue | Shipped In | Action |
|-------|-----------|--------|
| #70 | #158 | Close with receipt |
| #71 | #158 | Close with receipt |
| #72 | #158 | Close with receipt |
| #78 | #158 | Close with receipt |
| #95 | #186 | Close with receipt |
| #96 | #189 | Close with receipt |
| #97 | #190 | Close with receipt |
| #137 | #201 | Close with receipt |
| #88 | #201 | Close with receipt |
| #91 | #201 (guardrails) | Close with receipt |
| #92 | #201 (gold labels) | Close with receipt |
| #98 | #201 (semantic benchmark) | Close with receipt |

### Issues to Create (needed for v2.0)

| Issue | Purpose | DoD |
|-------|---------|-----|
| #133 | Event envelope formal spec | Spec document + contract tests |
| #134 | Reference Python streaming client | Working client + examples + contract tests |

---

## Quick Reference: What To Do Next

### Immediate (unblocks everything)

1. **Close shipped issues** — 12 issues listed above need GitHub closure with receipts
2. **Create #133, #134** — streaming contract issues need to exist
3. **Update CURRENT_STATUS.md** — reconcile with reality

### This Week

4. **Complete streaming client** — `streaming_client.py` needs full implementation
5. **Add streaming contract tests** — prove ordering guarantees
6. **OpenAI semantic adapter** — complete the stub

### Before v2.0.0

7. **Run benchmark suite** — verify targets (WER < 5%, DER < 15%, etc.)
8. **Local LLM backends** — implement qwen/smollm for offline mode
9. **Final docs pass** — all examples run, all docs current
10. **Version bump + tag** — CHANGELOG, version, publish

---

## Validation Commands Cheat Sheet

```bash
# Gate (always run before merge)
./scripts/ci-local.sh fast   # quick
./scripts/ci-local.sh        # full

# Benchmarks
uv run slower-whisper benchmark list
uv run slower-whisper benchmark status
uv run slower-whisper benchmark run --track asr
uv run slower-whisper benchmark compare --track asr

# Module health checks
uv run python -c "from transcription.api import transcribe_bytes"
uv run python -c "from transcription.streaming import StreamingSession"
uv run python -c "from transcription.semantic_adapter import SemanticAdapter"
uv run python -c "from transcription.llm_guardrails import LLMGuardrails"

# Version check
uv run python -c "import transcription; print(transcription.__version__)"
```

---

## Links

- [ROADMAP.md](../ROADMAP.md) — Forward plan
- [CHANGELOG.md](../CHANGELOG.md) — Release history
- [CURRENT_STATUS.md](./CURRENT_STATUS.md) — Issue tracker snapshot
- [STREAMING_ARCHITECTURE.md](./STREAMING_ARCHITECTURE.md) — Streaming protocol spec
- [BENCHMARKS.md](./BENCHMARKS.md) — Benchmark CLI reference
