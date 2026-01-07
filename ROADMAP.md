# slower-whisper Roadmap

**Current Version:** v1.9.2
**Last Updated:** 2026-01-07
<!-- cspell:ignore backpressure smollm CALLHOME qwen pyannote Libri -->

Roadmap = forward-looking execution plan.
History lives in [CHANGELOG.md](CHANGELOG.md).
Vision and strategic positioning live in [VISION.md](VISION.md).

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

## Now (1-4 weeks): v1.x Polish + Enabling Primitives

Work that makes future development cheap and safe.

### A) v1.9.x Closeout

What v1.9 promised. Two items shipped, one remains:

- [x] **Callback docs in STREAMING_ARCHITECTURE.md** — ✅ Shipped
  - `StreamCallbacks` protocol with type signatures, ordering guarantees, error isolation
  - See `docs/STREAMING_ARCHITECTURE.md` §StreamCallbacks Protocol
- [x] **Example callback integration** — ✅ Shipped
  - `examples/streaming/callback_demo.py` — 430-line working demo of all four callbacks
- [ ] **P95 latency harness** (local measurement, not CI-gated yet)
  - Script that prints enrichment latency stats + writes JSON artifact to `benchmarks/results/`

### B) API Polish (high-value v1.x)

- [ ] [#70](https://github.com/EffortlessMetrics/slower-whisper/issues/70): `transcribe_bytes()` — in-memory audio transcription
- [ ] [#71](https://github.com/EffortlessMetrics/slower-whisper/issues/71): `word_timestamps` REST parameter for `/transcribe`
- [ ] [#72](https://github.com/EffortlessMetrics/slower-whisper/issues/72): Word-level timestamps example
- [ ] [#78](https://github.com/EffortlessMetrics/slower-whisper/issues/78): Transcript convenience methods (`duration`, `word_count`, `is_enriched`)

### C) Observability & Provenance (cheap now, priceless later)

These are **missing issues to create** — platform primitives that make benchmarks and debugging actionable:

- [ ] **Receipt contract**: Standard `meta.receipt` fields (tool version, model id, device/compute_type, config hash, schema version)
- [ ] **Stable run/event IDs**: Consistent identifiers for streaming + batch that enable log correlation
- [ ] **Structured logging audit**: Ensure all modules use structured logging with stable event names

<details>
<summary><strong>Receipt Contract Specification</strong></summary>

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

#### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool_version` | string | yes | Package version (from metadata) |
| `schema_version` | int | yes | JSON schema version |
| `model` | string | yes | Whisper model name |
| `device` | string | yes | Resolved device (cuda/cpu) |
| `compute_type` | string | yes | Resolved compute type |
| `config_hash` | string | yes | SHA256 of serialized config |
| `run_id` | string | yes | Unique run identifier |
| `created_at` | string | yes | ISO 8601 timestamp (UTC) |
| `git_commit` | string | no | Git HEAD when run from checkout |

#### Run ID Format

`run-{YYYYMMDD}-{HHMMSS}-{random6}`

Example: `run-20260107-123456-a1b2c3`

</details>

### D) Documentation Hygiene

- [ ] [#132](https://github.com/EffortlessMetrics/slower-whisper/issues/132): Update CLAUDE.md to reflect v1.9.2 reality
- [ ] [#66](https://github.com/EffortlessMetrics/slower-whisper/issues/66): Document export/validate CLI commands

---

## Next (v2.0): Real-Time + Governance

**Theme:** Streaming is the new mode. Benchmarks are the new gate. Semantics is an extension point.

**Prerequisite order:** Benchmarks → Streaming → Semantics

### Track 1: Benchmark Foundations (gate for everything else)

Benchmarks must exist before streaming work can be measured.

| Issue | Description | Walking Skeleton |
|-------|-------------|------------------|
| [#94](https://github.com/EffortlessMetrics/slower-whisper/issues/94) | Dataset collection/curation | Manifest format + smoke set in-repo; large sets fetched on demand |
| [#95](https://github.com/EffortlessMetrics/slower-whisper/issues/95) | ASR WER runner | Smoke dataset first, one real dataset second; JSON/MD output |
| [#96](https://github.com/EffortlessMetrics/slower-whisper/issues/96) | Diarization DER runner | DER measurement on AMI subset |
| [#97](https://github.com/EffortlessMetrics/slower-whisper/issues/97) | Streaming latency benchmark | P95 latency artifact |
| [#57](https://github.com/EffortlessMetrics/slower-whisper/issues/57) | Benchmark CLI runners | `slower-whisper benchmark --track asr\|diarization\|streaming` |
| [#99](https://github.com/EffortlessMetrics/slower-whisper/issues/99) | CI integration | Report now; fail later (when Actions return) |

**Done when:** `slower-whisper benchmark --track asr` outputs reproducible JSON/Markdown + baseline file.

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

| Issue | Description | Walking Skeleton |
|-------|-------------|------------------|
| **NEW** | Event envelope spec | Stable IDs, ordering guarantees, partial/final semantics |
| [#84](https://github.com/EffortlessMetrics/slower-whisper/issues/84) | WebSocket endpoint skeleton | Accept connection, emit partial/final events, no diarization |
| [#85](https://github.com/EffortlessMetrics/slower-whisper/issues/85) | REST streaming endpoints | `/stream/start`, `/stream/audio`, `/stream/status` |
| **NEW** | Reference Python client | Working client + ordering/backpressure contract test |
| [#55](https://github.com/EffortlessMetrics/slower-whisper/issues/55) | Streaming API docs | WebSocket protocol, event flow, client examples |
| [#86](https://github.com/EffortlessMetrics/slower-whisper/issues/86) | Incremental diarization hook | Integration point (full impl is v2.1+) |

**Done when:** Python client connects via WebSocket, receives `PARTIAL` and `FINALIZED` events, latency is measured.

<details>
<summary><strong>Track 2 Design Notes: Event Envelope Specification</strong></summary>

#### Event Envelope (v2 Stable Contract)

All streaming events share this envelope:

```json
{
  "event_id": 42,
  "stream_id": "str-abc123",
  "type": "FINALIZED",
  "ts_server": "2026-01-07T12:00:00.123Z",
  "ts_audio_start": 10.5,
  "ts_audio_end": 14.2,
  "payload": { /* type-specific */ }
}
```

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
2. `FINALIZED` for a segment arrives after all its `PARTIAL` events
3. `SPEAKER_TURN` arrives after the `FINALIZED` that closed the turn
4. `ts_audio_start` of `FINALIZED` events is monotonically increasing

#### Backpressure (v1)

- Server buffers up to N events (default: 100)
- When buffer full: drop `PARTIAL` events first
- `FINALIZED` events are never dropped (blocks producer if necessary)
- Client acks are optional in v1 (mandatory in v2.1+)

</details>

### Track 3: Semantics Adapter Skeleton

Depends on: Stable Turn/Chunk model from Track 2

| Issue | Description | Walking Skeleton |
|-------|-------------|------------------|
| [#88](https://github.com/EffortlessMetrics/slower-whisper/issues/88) | LLM annotation schema | Schema slot + versioning; deterministic baseline annotator |
| [#89](https://github.com/EffortlessMetrics/slower-whisper/issues/89) | Local LLM backend | qwen2.5-7b or smollm adapter |
| [#90](https://github.com/EffortlessMetrics/slower-whisper/issues/90) | Cloud LLM backend interface | OpenAI/Anthropic adapter (one initially) |
| [#91](https://github.com/EffortlessMetrics/slower-whisper/issues/91) | Guardrails | Rate limits, cost controls, PII warnings |
| [#92](https://github.com/EffortlessMetrics/slower-whisper/issues/92) | Tests/fixtures/golden outputs | Contract tests for "missing deps never break pipeline" |
| [#98](https://github.com/EffortlessMetrics/slower-whisper/issues/98) | Semantic quality benchmark | Topic F1 measurement |

**Done when:** `--enable-semantics` with `backend=local` populates `annotations.semantic` using local model.

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

## Missing Issues to Create

The following should be created as issues before v2.0 work begins:

| Title | Track | Description |
|-------|-------|-------------|
| `streaming: Event envelope specification` | 2 | IDs, ordering, partial/final semantics, ack/backpressure |
| `streaming: Reference Python client` | 2 | Working client + contract test |
| `infra: Receipt contract for provenance` | Now | Standard `meta.receipt` fields |
| `infra: Stable run/event IDs` | Now | Correlation IDs for logging |
| `benchmark: Baseline file format` | 1 | JSON schema for regression detection |

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
