# slower-whisper Roadmap

**Current Version:** v1.9.2
**Last Updated:** 2026-01-07 (restructured Now/Next sequencing)
<!-- cspell:ignore backpressure smollm CALLHOME qwen pyannote Libri librispeech rttm RTTM acks goldens -->

Roadmap = forward-looking execution plan.
History lives in [CHANGELOG.md](CHANGELOG.md).
Vision and strategic positioning live in [VISION.md](VISION.md).

---

## Roadmap Execution Standards

**Primary metric:** DevLT (human attention minutes per trusted change).
**Quality goals:** PR relevancy + design alignment + proof completeness.

### What "Done" Means

- **Implementation** — code merged with tests passing
- **Artifact** — output exists (JSON schema, receipt, baseline file, etc.)
- **Local validation** — command that proves it works (`./scripts/ci-local.sh`)
- **Boundaries** — explicit out-of-scope documented

### When Something Is Wrong

If an item was wrong (measurement drift, doc drift, invalid claim), we:

1. Mark it wrong explicitly
2. Link the fix (PR/commit)
3. Add prevention work (gate/contract) when appropriate

### CI Posture

CI may be rate-limited or off. **Local gate is canonical:**

```bash
./scripts/ci-local.sh        # full gate
./scripts/ci-local.sh fast   # quick check
nix-clean flake check        # Nix checks (use nix-clean wrapper inside devshell)
```

PRs must include receipt output or link to passing local gate.

---

## Snapshot

- **Local-first conversation signal engine** producing structured transcripts + optional enrichment
- **Optional dependencies degrade gracefully** — no hard requirement on diarization, emotion, or integrations
- **CI may be off** — `./scripts/ci-local.sh` is the merge gate; paste receipts in PRs

---

## Recently Shipped (highlights)

- **v1.9.x:** Streaming callbacks + contract tests, GPU UX (`--device auto` + preflight banner), Nix/direnv hardening
- **v1.8.0:** Word-level timestamps and speaker alignment
- **v1.7.0:** Streaming enrichment + live semantics + unified config API

Full details: [CHANGELOG.md](CHANGELOG.md)

---

## Now (1-4 weeks): v1.x Polish + v2.0 Prerequisites

Shrink the surface, then start v2 with clean contracts.

### A) v1.9.x Closeout — Complete

v1.9 deliverables shipped:

- [x] Callback docs in STREAMING_ARCHITECTURE.md
- [x] Example callback integration (`examples/streaming/callback_demo.py`)

The P95 latency harness ships **as part of [#97](https://github.com/EffortlessMetrics/slower-whisper/issues/97)** (Track 1 streaming benchmark). No separate v1.9.x deliverable remains.

### B) API Polish Bundle

High adoption value, low risk. Ship as **one coherent PR** (multiple commits fine):

- [ ] [#71](https://github.com/EffortlessMetrics/slower-whisper/issues/71): `word_timestamps` REST parameter
- [ ] [#72](https://github.com/EffortlessMetrics/slower-whisper/issues/72): Word-level timestamps example
- [ ] [#78](https://github.com/EffortlessMetrics/slower-whisper/issues/78): `Transcript` convenience methods (`duration`, `word_count`, `is_enriched`)
- [ ] [#70](https://github.com/EffortlessMetrics/slower-whisper/issues/70): `transcribe_bytes()` API

**DoD:** CLI + API + REST + example all agree. Paste `./scripts/ci-local.sh` receipts in PR body.

### C) Issue Truth Pass (one-time cleanup)

Before v2 work begins, reconcile the tracker:

- [ ] **Close completed issues** (#66, #74, #132) with final "receipts" comment
- [ ] **Rewrite partial issues** to have v2-style DoD: inputs → outputs → local validation

### D) Create Infrastructure Issues

These are Track 1 prerequisites. Create if they don't exist yet:

| Issue | Title | Why |
|-------|-------|-----|
| #135 | Receipt contract | Benchmarks need provenance to be reproducible |
| #136 | Stable run/event IDs | Streaming needs correlation IDs for debugging |
| #137 | Baseline file format | Regression detection needs a schema |

**Action:** Verify these issues exist in the tracker. If missing, create them with DoD from the spec blocks below.

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
      "run_id": "run-20260107-123456-xyz",
      "created_at": "2026-01-07T12:00:00Z",
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

### Exit Criteria

1. API polish PR merged with receipts
2. Issue tracker reflects reality (done = closed, partial = rewritten DoD)
3. Infrastructure issues (#133–#137) exist with clear DoD from contract specs
4. [#97](https://github.com/EffortlessMetrics/slower-whisper/issues/97) unblocked (latency harness is its deliverable)

### Now → Next Handoff Checklist

Before starting v2 work:

- [ ] Track 1 contracts exist (dataset manifest, result schema, baseline format)
- [ ] Receipt/run_id infrastructure issues created (#135, #136, #137)
- [ ] Streaming contract issues created (#133, #134)
- [ ] API polish bundle merged
- [ ] Issue truth pass complete (done=closed, partial=rewritten)

### Execution Paths

Two coherent sequences after "Now" completes:

| Goal | Sequence |
|------|----------|
| **Move v2 forward** | #97 + #137 → #95 WER → #133 envelope + #134 client → #84 WS skeleton |
| **Improve adoption** | API polish bundle → then benchmarks |

Either path is coherent; mixing both in the same sprint is where velocity dies.

---

## Next (v2.0): Real-Time + Governance

**Theme:** Streaming is the new mode. Benchmarks are the new gate. Semantics is an extension point.

**Design principles:**
- **Benchmarks are artifacts** — JSON schema, baseline schema, comparison rules (not just scripts)
- **Streaming is a protocol** — envelope spec, ordering guarantees, backpressure story (not just endpoints)

**Prerequisite order:** Benchmarks → Streaming → Semantics

### Track 1: Benchmark Foundations (gate for everything else)

Benchmarks must exist before streaming work can be measured.

**Execution order** (make one runner real end-to-end before replicating):

1. [#95](https://github.com/EffortlessMetrics/slower-whisper/issues/95): **ASR WER runner** — smoke dataset first, real WER/CER via jiwer
2. [#137](https://github.com/EffortlessMetrics/slower-whisper/issues/137): **Baseline file format** — schema + local comparator (report-only)
3. [#97](https://github.com/EffortlessMetrics/slower-whisper/issues/97): **Streaming latency benchmark** — P50/P95/P99, first-token latency, RTF
4. [#96](https://github.com/EffortlessMetrics/slower-whisper/issues/96): **Diarization DER runner** — DER/JER on AMI subset (fiddlier, do last)
5. [#99](https://github.com/EffortlessMetrics/slower-whisper/issues/99): **CI integration** — report-only in `ci-local.sh`; gating flips on when Actions return

Supporting issues:

| Issue | Description |
|-------|-------------|
| [#94](https://github.com/EffortlessMetrics/slower-whisper/issues/94) | Dataset manifest format + smoke set in-repo |
| [#57](https://github.com/EffortlessMetrics/slower-whisper/issues/57) | CLI: `slower-whisper benchmark --track asr\|diarization\|streaming` |

**Done when:** `slower-whisper benchmark --track asr` emits result JSON + baseline compare JSON; report-only comparator implemented.

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

Depends on: Track 1 (latency measurement must exist)

**Protocol-first sequencing** (contract before implementation):

1. **#133**: **Event envelope spec** — IDs, ordering guarantees, partial/final semantics, backpressure
2. **#134**: **Reference Python client** — working client + contract tests for ordering/backpressure
3. [#84](https://github.com/EffortlessMetrics/slower-whisper/issues/84): **WebSocket endpoint** — accept connection, emit partial/final events
4. [#85](https://github.com/EffortlessMetrics/slower-whisper/issues/85): **REST streaming endpoints** — `/stream/start`, `/stream/audio`, `/stream/status`
5. [#55](https://github.com/EffortlessMetrics/slower-whisper/issues/55): **Streaming API docs** — update once endpoint is real
6. [#86](https://github.com/EffortlessMetrics/slower-whisper/issues/86): **Incremental diarization hook** — integration point (full impl is v2.1+)

**Action:** Create #133 and #134 if they don't exist yet, using the contract spec below.

**Done when:** Reference client passes contract tests against WS server (ordering + backpressure behavior verified).

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

Depends on: Stable Turn/Chunk model from Track 2

**Contract-first sequencing** (interfaces before backends):

1. [#88](https://github.com/EffortlessMetrics/slower-whisper/issues/88): **LLM annotation schema** — schema slot + versioning
2. [#90](https://github.com/EffortlessMetrics/slower-whisper/issues/90): **Cloud LLM interface** — one provider first (OpenAI or Anthropic)
3. [#91](https://github.com/EffortlessMetrics/slower-whisper/issues/91): **Guardrails** — rate limits, cost controls, PII warnings
4. [#92](https://github.com/EffortlessMetrics/slower-whisper/issues/92): **Tests/fixtures/golden files** — contract tests for "missing deps never break"
5. [#89](https://github.com/EffortlessMetrics/slower-whisper/issues/89): **Local LLM backend** — qwen2.5-7b or smollm (after interface is stable)
6. [#98](https://github.com/EffortlessMetrics/slower-whisper/issues/98): **Semantic quality benchmark** — Topic F1 measurement

> **Why this order:** Without schema + interface + guardrails + golden files first, semantics becomes "LLM integration sprawl."

**Done when:** Schema + interface + guardrails + golden files land before any backend; local backend populates deterministic fields.

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

| Issue | Description |
|-------|-------------|
| [#59](https://github.com/EffortlessMetrics/slower-whisper/issues/59) | Remove deprecated APIs (`--enrich-config`, legacy scripts) |
| [#48](https://github.com/EffortlessMetrics/slower-whisper/issues/48) | Expanded benchmark datasets (AMI, CALLHOME, LibriSpeech) |

### v2.0 Performance Targets

| Metric | Target | Measurement Source |
|--------|--------|-------------------|
| ASR WER (LibriSpeech) | < 5% | Track 1 benchmark |
| Diarization DER (AMI) | < 15% | Track 1 benchmark |
| Streaming P95 latency | < 500ms | Track 2 benchmark |
| Semantic Topic F1 | > 0.8 | Track 3 benchmark |
| Concurrent streams/GPU | > 10 | Track 2 stress test |

---

## Later (v2.1+ / v3)

Intentionally brief — these are placeholders, not commitments.

### v2.1+

- Full incremental diarization for streaming ([#86])
- Additional cloud LLM backends
- Expanded domain-specific prompt templates

### v3.0 (2027+)

- [#60](https://github.com/EffortlessMetrics/slower-whisper/issues/60): Intent detection from prosody+text fusion
- [#61](https://github.com/EffortlessMetrics/slower-whisper/issues/61): Domain pack — clinical speech analysis
- [#62](https://github.com/EffortlessMetrics/slower-whisper/issues/62): Acoustic scene analysis + audio event detection
- [#65](https://github.com/EffortlessMetrics/slower-whisper/issues/65): Discourse structure analysis with topic segmentation
- Domain pack plugin architecture

---

## Dependency Map

```
               ┌─────────────────────────────────────┐
               │  Track 1: Benchmarks (foundations)  │
               └─────────────────┬───────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Track 2: Stream  │  │ Track 3: Semantic│  │ Track 4: Cleanup │
│   (skeleton)     │  │   (adapter)      │  │   (deprecation)  │
└────────┬─────────┘  └────────┬─────────┘  └──────────────────┘
         │                     │
         │   ┌─────────────────┘
         ▼   ▼
    ┌─────────────────────────┐
    │ v2.0.0 Release          │
    │ (real-time + governance)│
    └─────────────────────────┘
```

**Key constraints:**
- Benchmarks gate streaming (can't claim latency without measuring it)
- Streaming endpoints depend on callback contracts (shipped in v1.9)
- Semantics depends on stable Turn/Chunk model
- v2.0 removes deprecated APIs only after docs point to replacements

---

## Issues to Verify/Create

Before starting a track, verify these issues exist. If missing, create them:

| Issue | Track | Description |
|-------|-------|-------------|
| #133 | 2 | Event envelope specification (IDs, ordering, backpressure) |
| #134 | 2 | Reference Python client + contract tests |
| #135 | 1 | Receipt contract specification |
| #136 | 2 | Stable run/event ID format |
| #137 | 1 | Baseline file format specification |

Use the `<details>` contract specs in each track section as the issue DoD.

---

## Backlog (Good Contribution Opportunities)

Items valuable but not scheduled. See [Backlog milestone](https://github.com/EffortlessMetrics/slower-whisper/milestone/4).

**High Priority:**
- [#67](https://github.com/EffortlessMetrics/slower-whisper/issues/67): Security scanning workflow (pip-audit, bandit)
- [#68](https://github.com/EffortlessMetrics/slower-whisper/issues/68): Docker vulnerability scanning (Trivy)
- [#69](https://github.com/EffortlessMetrics/slower-whisper/issues/69): Parallelize segment processing in enrichment
- [#73](https://github.com/EffortlessMetrics/slower-whisper/issues/73): Diarization standalone example
- [#63](https://github.com/EffortlessMetrics/slower-whisper/issues/63): Enable GitHub Discussions

**Good First Issues:**
- [#72](https://github.com/EffortlessMetrics/slower-whisper/issues/72): Word-level timestamps example
- [#74](https://github.com/EffortlessMetrics/slower-whisper/issues/74): Export and validation example
- [#75](https://github.com/EffortlessMetrics/slower-whisper/issues/75): `--dry-run` CLI option

---

## How We Execute

### Issues

Every issue should have:
- **Problem**: What's broken or missing
- **Definition of Done**: Testable acceptance criteria
- **Files likely touched**: Help reviewers orient
- **Local validation**: Commands to verify the fix

### PRs

- Keep PRs coherent; stacking is fine when it doesn't increase review cost
- Include a "review map" when > ~5 files
- Paste local receipts in PR body:
  ```
  ./scripts/ci-local.sh fast
  ./scripts/ci-local.sh
  nix-clean flake check
  ```

### Quality Gate (Actions off)

Since Actions may be disabled, the local gate is canonical:

```bash
./scripts/ci-local.sh        # full gate
./scripts/ci-local.sh fast   # quick check
nix-clean run .#verify -- --quick
```

---

## Versioning Philosophy

- **v1.x** — Stabilize, enrich, polish (current)
- **v2.x** — Real-time streaming + benchmarks + semantic adapters
- **v3.x** — Semantic understanding + domain packs

**Principle:** Each major version adds **layers**, not rewrites.
v1.x JSON is forward-compatible with v2.x readers.

### Deprecation Policy

- **Announcement**: At least 2 minor versions before removal
- **Warning period**: Deprecation warnings logged during usage
- **Removal**: Only in major version bumps (v2.0.0, v3.0.0)

Current deprecations (target removal: v2.0.0):
- `--enrich-config` flag → use `--config`
- `transcribe_pipeline.py` script → use `slower-whisper transcribe`
- `audio_enrich.py` script → use `slower-whisper enrich`

### Backward Compatibility Guarantees

- **JSON Schema v2**: Forward-compatible through v2.x (new optional fields only)
- **Python API**: `transcribe_directory()`, `enrich_directory()` signatures stable through v2.x
- **CLI**: Core subcommands (`transcribe`, `enrich`, `export`, `validate`) stable through v2.x

---

## Links

- [CHANGELOG.md](CHANGELOG.md) — what shipped when
- [VISION.md](VISION.md) — long-term strategic positioning
- [CLAUDE.md](CLAUDE.md) — repo guide + invariants
- [docs/STREAMING_ARCHITECTURE.md](docs/STREAMING_ARCHITECTURE.md) — streaming model + events
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute
- [GitHub Issues](https://github.com/EffortlessMetrics/slower-whisper/issues) — feature requests and discussions
