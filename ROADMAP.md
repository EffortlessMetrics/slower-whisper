# slower-whisper Roadmap

**Current Version:** v1.9.2
**Last Updated:** 2026-01-21
<!-- cspell:ignore backpressure smollm CALLHOME qwen pyannote Libri librispeech rttm RTTM acks goldens -->

Roadmap = forward-looking execution plan.
History lives in [CHANGELOG.md](CHANGELOG.md).
Vision and strategic positioning live in [VISION.md](VISION.md).

---

## Quick Status

| Track | Status | Next Action |
|-------|--------|-------------|
| v1.9.x Closeout | ✅ Complete | — |
| API Polish Bundle | 📋 Ready to Start | Begin #70 |
| Track 1: Benchmarks | 🔄 In Progress | Complete #99 (CI integration) |
| Track 2: Streaming | 📋 Ready to Start | Begin #133 |
| Track 3: Semantics | 📋 Ready to Start | Begin #88 |

---

## Execution Standards

**Quality bar:** Every PR must pass `./scripts/ci-local.sh` with receipts pasted.

### Definition of Done

- **Code** — merged with tests passing
- **Artifacts** — outputs exist (schemas, baselines, receipts)
- **Validation** — `./scripts/ci-local.sh` proves it works
- **Scope** — explicit boundaries documented

### Local Gate (CI may be off)

```bash
./scripts/ci-local.sh        # full gate
./scripts/ci-local.sh fast   # quick check
nix-clean flake check        # Nix checks
```

---

## Recently Shipped

| Version | Highlights |
|---------|------------|
| **Unreleased** | Benchmark evaluation framework (ASR/DER/emotion/streaming), baseline infrastructure (#137), Anthropic LLM provider, parallel audio normalization |
| **v1.9.2** | Version constant fix (`transcription.__version__` now correct) |
| **v1.9.1** | GPU UX (`--device auto` default, preflight banner), CI caching fixes |
| **v1.9.0** | Streaming callbacks (`StreamCallbacks` protocol), safe callback execution |
| **v1.8.0** | Word-level timestamps, word-speaker alignment |
| **v1.7.0** | Streaming enrichment, live semantics, `Config.from_sources()` |

Full history: [CHANGELOG.md](CHANGELOG.md)

---

## Now: v1.x Polish + v2.0 Prerequisites

**Goal:** Clean up v1.x surface, then start v2.0 with solid contracts.

### A) v1.9.x Closeout — ✅ Complete

All v1.9.x deliverables shipped:
- ✅ Callback docs in STREAMING_ARCHITECTURE.md
- ✅ Example callback integration (`examples/streaming/callback_demo.py`)
- ✅ Version constant fix (v1.9.2)

P95 latency harness moves to Track 1 ([#97](https://github.com/EffortlessMetrics/slower-whisper/issues/97)).

### B) API Polish Bundle — 🔄 In Progress

Ship as **one coherent PR** (high adoption value, low risk):

| Issue | Feature | Status |
|-------|---------|--------|
| [#70](https://github.com/EffortlessMetrics/slower-whisper/issues/70) | `transcribe_bytes()` API | ⬜ |
| [#71](https://github.com/EffortlessMetrics/slower-whisper/issues/71) | `word_timestamps` REST parameter | ⬜ |
| [#72](https://github.com/EffortlessMetrics/slower-whisper/issues/72) | Word-level timestamps example | ⬜ |
| [#78](https://github.com/EffortlessMetrics/slower-whisper/issues/78) | `Transcript` convenience methods | ⬜ |

**DoD:** CLI + API + REST + example all consistent. Local gate passes.

### C) Issue Cleanup

Reconcile tracker before v2 work:
- [ ] Close completed issues (#66, #74, #132) with receipts
- [ ] Rewrite partial issues with v2-style DoD (inputs → outputs → validation)

### D) Infrastructure Issues

Verify these exist (create if missing):

| Issue | Purpose |
|-------|---------|
| #135 | Receipt contract (benchmark provenance) |
| #136 | Stable run/event IDs (streaming correlation) |
| #137 | Baseline file format (regression detection) |

<details>
<summary><strong>Receipt Contract Specification (for #135)</strong></summary>

Every transcript and benchmark artifact includes `meta.receipt`:

```json
{
  "meta": {
    "receipt": {
      "tool_version": "1.9.2",
      "schema_version": 2,
      "model": "large-v3",
      "device": "cuda",
      "compute_type": "float16",
      "config_hash": "sha256:abc123...",
      "run_id": "run-20260108-123456-xyz",
      "created_at": "2026-01-08T12:00:00Z",
      "git_commit": "abc1234"
    }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool_version` | string | yes | Package version (from metadata) |
| `schema_version` | int | yes | JSON schema version |
| `model` | string | yes | Whisper model name |
| `device` | string | yes | Resolved device (cuda/cpu) |
| `compute_type` | string | yes | Resolved compute type |
| `config_hash` | string | yes | SHA256 of serialized config |
| `run_id` | string | yes | Unique run identifier (`run-{YYYYMMDD}-{HHMMSS}-{random6}`) |
| `created_at` | string | yes | ISO 8601 timestamp (UTC) |
| `git_commit` | string | no | Git HEAD when run from checkout |

</details>

### Exit Criteria for "Now"

| Criterion | Status |
|-----------|--------|
| API polish PR merged | ⬜ |
| Issue tracker reconciled | ⬜ |
| Infrastructure issues exist (#135–#137) | ⬜ |
| Streaming contract issues exist (#133, #134) | ⬜ |

### Recommended Execution Path

```
API Polish Bundle (#70, #71, #72, #78)
         │
         ▼
Track 1: Benchmarks (#95 → #137 → #97 → #96)
         │
         ▼
Track 2: Streaming (#133 → #134 → #84)
```

**Note:** Complete API polish before benchmarks. Don't mix adoption work and v2 infrastructure.

---

## Next (v2.0): Real-Time + Governance

**Theme:** Streaming is the new mode. Benchmarks are the new gate.

**Design principles:**
- **Benchmarks are artifacts** — JSON schemas with comparison rules, not just scripts
- **Streaming is a protocol** — envelope spec with ordering guarantees and backpressure

**Prerequisite order:** Benchmarks → Streaming → Semantics

---

### Track 1: Benchmark Foundations

**Status:** 🔄 In Progress — evaluation runners implemented, baselines & CI integration remaining

Benchmarks must exist before streaming work can be measured.

| Order | Issue | Deliverable | Status |
|-------|-------|-------------|--------|
| 1 | [#95](https://github.com/EffortlessMetrics/slower-whisper/issues/95) | ASR WER runner (jiwer, smoke dataset) | ✅ `ASRBenchmarkRunner` (#186) |
| 2 | [#137](https://github.com/EffortlessMetrics/slower-whisper/issues/137) | Baseline file format + comparator | ✅ Baseline infrastructure complete |
| 3 | [#97](https://github.com/EffortlessMetrics/slower-whisper/issues/97) | Streaming latency (P50/P95/P99, RTF) | ✅ `StreamingBenchmarkRunner` (#190) |
| 4 | [#96](https://github.com/EffortlessMetrics/slower-whisper/issues/96) | Diarization DER runner (AMI subset) | ✅ `DiarizationBenchmarkRunner` (#189) |
| 5 | [#99](https://github.com/EffortlessMetrics/slower-whisper/issues/99) | CI integration (report-only initially) | ⬜ |

**Also implemented:**

- `EmotionBenchmarkRunner` (#187): Categorical emotion accuracy, F1, confusion matrix

**Supporting:**

- [#94](https://github.com/EffortlessMetrics/slower-whisper/issues/94): Dataset manifest format
- [#57](https://github.com/EffortlessMetrics/slower-whisper/issues/57): CLI `slower-whisper benchmark --track asr|diarization|streaming`

**Remaining work:**

1. CI integration with report-only mode (#99)
2. CLI subcommand wiring for `slower-whisper benchmark` (partially complete)

**Done when:** `slower-whisper benchmark --track asr` emits result JSON + baseline comparison.

<details>
<summary><strong>Dataset Manifest Contract (for #94)</strong></summary>

#### Manifest File Location

`benchmarks/datasets/<track>/<dataset>/manifest.json`

#### Required Fields

```json
{
  "schema_version": 1,
  "id": "librispeech-test-clean",
  "track": "asr",
  "split": "test",
  "samples": [
    {
      "id": "sample-001",
      "audio": "audio/sample-001.wav",
      "sha256": "abc123...",
      "duration_s": 12.5,
      "language": "en",
      "reference_transcript": "the quick brown fox",
      "license": "CC-BY-4.0",
      "source": "librispeech"
    }
  ],
  "meta": {
    "created_at": "2026-01-07T12:00:00Z",
    "total_duration_s": 3600,
    "sample_count": 100
  }
}
```

#### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique dataset identifier |
| `track` | string | yes | Benchmark track (asr, diarization, streaming) |
| `split` | string | yes | Dataset split (train, dev, test, smoke) |
| `samples[].audio` | string | yes | Relative path or URL to audio |
| `samples[].sha256` | string | yes | SHA256 hash of audio file |
| `samples[].reference_transcript` | string | ASR | Ground truth text (ASR track) |
| `samples[].reference_rttm` | string | Diarization | Path to RTTM file (diarization track) |
| `license` | string | yes | License identifier |
| `source` | string | yes | Data provenance |

#### Staging Policy

- **Smoke sets:** Committed to repo (small, fast, always available)
- **Full sets:** Downloaded via script with hash verification
- **Invalid provenance:** Explicitly rejected (no unverified datasets)

#### Local Validation

```bash
slower-whisper benchmark --track asr --dataset smoke --limit 10 --dry-run
```

</details>

<details>
<summary><strong>Benchmark Result Contract (for #95/#96/#97/#98)</strong></summary>

#### Result File Location

`benchmarks/results/<track>/<dataset>-<run_id>.json`

#### Required Schema

```json
{
  "schema_version": 1,
  "track": "asr",
  "dataset": "librispeech-test-clean",
  "created_at": "2026-01-07T12:00:00Z",
  "run_id": "run-20260107-143022-xyz",
  "metrics": {
    "wer": {"value": 0.045, "unit": "ratio"},
    "cer": {"value": 0.012, "unit": "ratio"},
    "rtf": {"value": 0.15, "unit": "ratio"}
  },
  "receipt": {
    "tool_version": "1.9.2",
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "config_hash": "sha256:abc123...",
    "git_commit": "abc1234",
    "dataset_manifest_sha256": "def456..."
  },
  "valid": true,
  "samples": []
}
```

#### Track-Specific Required Metrics

| Track | Required Metrics |
|-------|------------------|
| ASR | `wer`, `cer` |
| Diarization | `der`, `jer`, `speaker_count_accuracy` |
| Streaming | `p50_ms`, `p95_ms`, `p99_ms`, `first_token_ms`, `rtf` |

#### Measurement Invalidation

If semantics drift is discovered after the fact:

```json
{
  "valid": false,
  "invalid_reason": "Baseline used different normalization; see #142",
  "invalidated_at": "2026-01-15T10:00:00Z",
  "superseded_by": "run-20260115-093000-abc"
}
```

This is a first-class state transition, not social drama.

</details>

<details>
<summary><strong>Track 1 Design Notes: Baselines & Regression Policy</strong></summary>

#### Baseline File Format

Location: `benchmarks/baselines/<track>/<dataset>.json`

```json
{
  "schema_version": 1,
  "track": "asr",
  "dataset": "librispeech-test-clean",
  "created_at": "2026-01-07T12:00:00Z",
  "metrics": {
    "wer": {"value": 0.045, "unit": "ratio", "threshold": 0.05},
    "cer": {"value": 0.012, "unit": "ratio", "threshold": null}
  },
  "receipt": {
    "tool_version": "1.9.2",
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16"
  }
}
```

#### Regression Policy

- **Phase 1 (now):** Report-only mode. Benchmark CLI prints comparison vs baseline but never fails.
- **Phase 2 (when Actions return):** Gate mode. Fail if primary metric exceeds threshold.

#### Comparison Rule

```
regression = (current_value - baseline_value) / baseline_value
fail_if: regression > threshold_percent (default: 10%)
```

</details>

### Track 2: Streaming Skeleton

**Status:** ⏳ Blocked on Track 1 (latency measurement must exist)

**Approach:** Protocol-first — define contracts before implementation.

| Order | Issue | Deliverable | Status |
|-------|-------|-------------|--------|
| 1 | #133 | Event envelope spec (IDs, ordering, backpressure) | ⬜ |
| 2 | #134 | Reference Python client + contract tests | ⬜ |
| 3 | [#84](https://github.com/EffortlessMetrics/slower-whisper/issues/84) | WebSocket endpoint (partial/final events) | ⬜ |
| 4 | [#85](https://github.com/EffortlessMetrics/slower-whisper/issues/85) | REST streaming endpoints | ⬜ |
| 5 | [#55](https://github.com/EffortlessMetrics/slower-whisper/issues/55) | Streaming API docs | ⬜ |
| 6 | [#86](https://github.com/EffortlessMetrics/slower-whisper/issues/86) | Incremental diarization hook | ⬜ |

**Prerequisite:** Create #133 and #134 if they don't exist.

**Done when:** Reference client passes contract tests against WS server.

<details>
<summary><strong>Track 2 Design Notes: Event Envelope Specification</strong></summary>

#### Event Envelope (v2 Stable Contract)

All streaming events share this envelope:

```json
{
  "event_id": 42,
  "stream_id": "str-abc123",
  "segment_id": "seg-007",
  "type": "FINALIZED",
  "ts_server": "2026-01-07T12:00:00.123Z",
  "ts_audio_start": 10.5,
  "ts_audio_end": 14.2,
  "payload": { /* type-specific */ }
}
```

#### ID Contracts

| ID | Format | Scope | Guarantees |
|----|--------|-------|------------|
| `stream_id` | `str-{uuid4}` | per connection | Unique across all streams |
| `event_id` | monotonic int | per stream | Never reused within stream |
| `segment_id` | `seg-{seq}` | per stream | Stable reference for partials → finalized |

#### Event Types

| Type | When Emitted | Payload |
|------|--------------|---------|
| `PARTIAL` | ASR partial result | `{text, confidence?}` |
| `FINALIZED` | Segment finalized | `StreamSegment` |
| `SPEAKER_TURN` | Turn boundary | `Turn` dict |
| `SEMANTIC_UPDATE` | Semantic annotation | `SemanticUpdatePayload` |
| `ERROR` | Recoverable error | `StreamingError` |

#### Ordering Guarantees

1. `event_id` is monotonically increasing per stream
2. `PARTIAL` for a segment arrives before its `FINALIZED`
3. `FINALIZED` events are monotonic in `ts_audio_start`
4. `SPEAKER_TURN` arrives after the `FINALIZED` that closed the turn
5. No event arrives with `event_id` < previously received `event_id`

#### Backpressure Contract

| Parameter | Default | Configurable | Description |
|-----------|---------|--------------|-------------|
| `buffer_size` | 100 | yes | Max events buffered before drop policy |
| `drop_policy` | `partial_first` | yes | What to drop when full |
| `finalized_drop` | `never` | no | FINALIZED events block producer, never dropped |

**Drop priority (when buffer full):**
1. Drop oldest `PARTIAL` events first
2. Drop oldest `SEMANTIC_UPDATE` events second
3. `FINALIZED` and `ERROR` are never dropped

#### Resume Contract (v2.0 — best effort)

- Client sends `last_event_id` on reconnect
- Server replays from buffer if `last_event_id` is in buffer
- If `last_event_id` not in buffer: server sends `RESUME_GAP` error with `{missing_from, missing_to}`
- Client must handle gaps gracefully (re-request audio window or accept loss)

#### Security Posture (v2.0)

- No authentication in v2.0 skeleton
- Future: Bearer token header or WS subprotocol auth
- Rate limiting: 10 streams/IP default (configurable)

</details>

### Track 3: Semantics Adapter Skeleton

**Status:** ⏳ Blocked on Track 2 (needs stable Turn/Chunk model)

**Approach:** Contract-first — interfaces before backends.

| Order | Issue | Deliverable | Status |
|-------|-------|-------------|--------|
| 1 | [#88](https://github.com/EffortlessMetrics/slower-whisper/issues/88) | LLM annotation schema + versioning | ⬜ |
| 2 | [#90](https://github.com/EffortlessMetrics/slower-whisper/issues/90) | Cloud LLM interface (OpenAI/Anthropic) | ⬜ |
| 3 | [#91](https://github.com/EffortlessMetrics/slower-whisper/issues/91) | Guardrails (rate limits, cost, PII) | ⬜ |
| 4 | [#92](https://github.com/EffortlessMetrics/slower-whisper/issues/92) | Golden files + contract tests | ⬜ |
| 5 | [#89](https://github.com/EffortlessMetrics/slower-whisper/issues/89) | Local LLM backend (qwen2.5-7b/smollm) | ⬜ |
| 6 | [#98](https://github.com/EffortlessMetrics/slower-whisper/issues/98) | Semantic quality benchmark (Topic F1) | ⬜ |

**Why this order:** Schema + interface + guardrails + golden files must land before any backend to avoid "LLM integration sprawl."

**Done when:** Local backend populates deterministic fields; golden files enforce contracts.

<details>
<summary><strong>Track 3 Design Notes: Semantics Contract</strong></summary>

#### Annotation Schema v0

```json
{
  "annotations": {
    "semantic": {
      "schema_version": "0.1.0",
      "provider": "local",
      "model": "qwen2.5-7b",
      "raw_model_output": { /* provider-specific */ },
      "normalized": {
        "topics": ["pricing", "contract_terms"],
        "intent": "objection",
        "sentiment": "negative",
        "action_items": [],
        "risk_tags": ["churn_risk"]
      },
      "confidence": 0.85,
      "latency_ms": 420
    }
  }
}
```

#### Provider Interface Contract

```python
class SemanticProvider(Protocol):
    """All semantic backends implement this interface."""

    def annotate_chunk(
        self,
        text: str,
        context: ChunkContext
    ) -> SemanticAnnotation:
        """Annotate a single chunk (60-120s of conversation)."""
        ...

    def health_check(self) -> ProviderHealth:
        """Return provider status and quota remaining."""
        ...
```

#### Guardrails Contract

| Guardrail | Default | Configurable | Description |
|-----------|---------|--------------|-------------|
| `rate_limit_rpm` | 60 | yes | Requests per minute |
| `cost_budget_usd` | 1.00 | yes | Max spend per session |
| `pii_warning` | true | yes | Warn if PII detected in chunk |
| `timeout_ms` | 30000 | yes | Per-request timeout |

#### Golden Files Contract

- Location: `tests/fixtures/semantic_golden/`
- Each golden file: input chunk + expected normalized output
- Tests verify: missing provider → graceful skip (not crash)
- Tests verify: deterministic fields match golden output

</details>

### Track 4: v2.0 Cleanup

| Issue | Deliverable | Status |
|-------|-------------|--------|
| [#59](https://github.com/EffortlessMetrics/slower-whisper/issues/59) | Remove deprecated APIs (`--enrich-config`, legacy scripts) | ✅ |
| [#48](https://github.com/EffortlessMetrics/slower-whisper/issues/48) | Expanded benchmark datasets (AMI, CALLHOME, LibriSpeech) | ⬜ |

---

### v2.0 Performance Targets

| Metric | Target | Source |
|--------|--------|--------|
| ASR WER (LibriSpeech) | < 5% | Track 1 |
| Diarization DER (AMI) | < 15% | Track 1 |
| Streaming P95 latency | < 500ms | Track 2 |
| Semantic Topic F1 | > 0.8 | Track 3 |
| Concurrent streams/GPU | > 10 | Track 2 |

---

## Later (v2.1+ / v3)

Placeholders, not commitments.

### v2.1+

| Feature | Issue |
|---------|-------|
| Full incremental diarization for streaming | [#86](https://github.com/EffortlessMetrics/slower-whisper/issues/86) |
| Additional cloud LLM backends | — |
| Domain-specific prompt templates | — |

### v3.0 (2027+)

| Feature | Issue |
|---------|-------|
| Intent detection (prosody + text fusion) | [#60](https://github.com/EffortlessMetrics/slower-whisper/issues/60) |
| Clinical speech domain pack | [#61](https://github.com/EffortlessMetrics/slower-whisper/issues/61) |
| Acoustic scene analysis | [#62](https://github.com/EffortlessMetrics/slower-whisper/issues/62) |
| Discourse structure analysis | [#65](https://github.com/EffortlessMetrics/slower-whisper/issues/65) |
| Domain pack plugin architecture | — |

---

## Dependency Map

```
                    ┌─────────────────────┐
                    │ API Polish Bundle   │
                    │ (#70, #71, #72, #78)│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Track 1:          │
                    │   Benchmarks        │
                    │ (#95 → #137 → #97)  │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Track 2:       │  │  Track 3:       │  │  Track 4:       │
│  Streaming      │  │  Semantics      │  │  Cleanup        │
│ (#133 → #84)    │  │ (#88 → #89)     │  │ (#59, #48)      │
└────────┬────────┘  └────────┬────────┘  └─────────────────┘
         │                    │
         └─────────┬──────────┘
                   ▼
          ┌─────────────────┐
          │    v2.0.0       │
          │    Release      │
          └─────────────────┘
```

**Key constraints:**
- Benchmarks gate streaming (can't claim latency without measuring it)
- Streaming depends on v1.9 callback contracts (✅ shipped)
- Semantics depends on stable Turn/Chunk model from Track 2
- Deprecated APIs removed only after docs point to replacements

---

## Issues to Create

Before starting a track, ensure these issues exist:

| Issue | Track | Purpose |
|-------|-------|---------|
| #133 | 2 | Event envelope spec |
| #134 | 2 | Reference Python client |
| #135 | 1 | Receipt contract |
| #136 | 2 | Stable run/event IDs |
| #137 | 1 | Baseline file format |

Use the contract specs in `<details>` sections above as issue DoD.

---

## Backlog

Community contribution opportunities. See [Backlog milestone](https://github.com/EffortlessMetrics/slower-whisper/milestone/4).

### High Priority

| Issue | Description |
|-------|-------------|
| [#67](https://github.com/EffortlessMetrics/slower-whisper/issues/67) | Security scanning (pip-audit, bandit) |
| [#68](https://github.com/EffortlessMetrics/slower-whisper/issues/68) | Docker vulnerability scanning (Trivy) |
| [#69](https://github.com/EffortlessMetrics/slower-whisper/issues/69) | Parallelize segment processing |
| [#73](https://github.com/EffortlessMetrics/slower-whisper/issues/73) | Diarization standalone example |
| [#63](https://github.com/EffortlessMetrics/slower-whisper/issues/63) | Enable GitHub Discussions |

### Good First Issues

| Issue | Description |
|-------|-------------|
| [#72](https://github.com/EffortlessMetrics/slower-whisper/issues/72) | Word-level timestamps example |
| [#74](https://github.com/EffortlessMetrics/slower-whisper/issues/74) | Export and validation example |
| [#75](https://github.com/EffortlessMetrics/slower-whisper/issues/75) | `--dry-run` CLI option |

---

## How We Execute

### Issues

Every issue needs:
- **Problem** — what's broken or missing
- **DoD** — testable acceptance criteria
- **Files touched** — help reviewers orient
- **Validation** — commands to verify

### PRs

- Keep PRs coherent; stacking is fine
- Include "review map" when > 5 files
- Paste receipts in PR body

### Quality Gate

```bash
./scripts/ci-local.sh        # full gate
./scripts/ci-local.sh fast   # quick check
nix-clean flake check        # Nix checks
```

---

## Versioning

| Version | Theme |
|---------|-------|
| **v1.x** | Stabilize, enrich, polish (current) |
| **v2.x** | Real-time streaming + benchmarks + semantic adapters |
| **v3.x** | Semantic understanding + domain packs |

**Principle:** Each major version adds **layers**, not rewrites. v1.x JSON is forward-compatible with v2.x readers.

### Deprecation Policy

- Announced ≥2 minor versions before removal
- Warnings logged during usage
- Removed only in major versions

**Removed in v2.0.0 (#59):**
- `--enrich-config` (use `--config`)
- `transcribe_pipeline.py` (use `slower-whisper transcribe`)
- `audio_enrich.py` (use `slower-whisper enrich`)
- `slower-whisper-enrich` entry point (use `slower-whisper enrich`)
- `transcription/audio_enrich_cli.py` module

### Stability Guarantees

| Surface | Guarantee |
|---------|-----------|
| JSON Schema v2 | Forward-compatible through v2.x |
| Python API | `transcribe_directory()`, `enrich_directory()` stable |
| CLI | Core subcommands stable (`transcribe`, `enrich`, `export`, `validate`) |

---

## Links

| Document | Purpose |
|----------|---------|
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [VISION.md](VISION.md) | Strategic positioning |
| [CLAUDE.md](CLAUDE.md) | Repo guide + invariants |
| [docs/STREAMING_ARCHITECTURE.md](docs/STREAMING_ARCHITECTURE.md) | Streaming model |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guide |
| [GitHub Issues](https://github.com/EffortlessMetrics/slower-whisper/issues) | Bug reports + features |
