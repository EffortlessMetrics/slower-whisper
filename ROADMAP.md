# slower-whisper Roadmap

**Current Version:** v2.0.2
**Last Updated:** 2026-02-17
<!-- cspell:ignore backpressure smollm CALLHOME qwen pyannote Libri librispeech rttm RTTM acks goldens -->

Roadmap = forward-looking execution plan.
History lives in [CHANGELOG.md](CHANGELOG.md).
Vision and strategic positioning live in [VISION.md](VISION.md).

---

## Quick Status

| Area | Status | Next |
|------|--------|------|
| v2.0 core (streaming, benchmarks, semantics, post-processing) | ✅ Shipped | Maintenance |
| Expanded benchmark datasets | ⬜ Open | [#48](https://github.com/EffortlessMetrics/slower-whisper/issues/48) |
| Incremental diarization | ⬜ Open | [#86](https://github.com/EffortlessMetrics/slower-whisper/issues/86) |
| Semantic quality benchmark (Topic F1) | ⬜ Open | [#98](https://github.com/EffortlessMetrics/slower-whisper/issues/98) |

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
| **v2.0.1** | Post-processing orchestration, topic segmentation, turn-taking policies |
| **v2.0.0** | WebSocket streaming API, REST session management, semantic adapter protocol, benchmark evaluation framework (ASR/DER/emotion/streaming), LLM guardrails, baseline infrastructure |
| **v1.9.2** | Version constant fix (`transcription.__version__` now correct) |
| **v1.9.1** | GPU UX (`--device auto` default, preflight banner), CI caching fixes |
| **v1.9.0** | Streaming callbacks (`StreamCallbacks` protocol), safe callback execution |
| **v1.8.0** | Word-level timestamps, word-speaker alignment |
| **v1.7.0** | Streaming enrichment, live semantics, `Config.from_sources()` |

Full history: [CHANGELOG.md](CHANGELOG.md)

---

## Shipped in v2.0

All five development tracks completed and released.

| Track | What Shipped | Key Modules |
|-------|-------------|-------------|
| **Benchmarks** | ASR WER, DER, emotion, streaming latency runners; baseline infrastructure; gate mode (`--gate`, `--threshold`) | `benchmarks.py`, `benchmark_cli.py` |
| **Streaming** | WebSocket + SSE endpoints, event envelope (IDs, ordering, backpressure), Python client, session registry | `streaming_ws.py`, `streaming_client.py`, `session_registry.py` |
| **Semantics** | 5 adapters (local keywords, local LLM, OpenAI, Anthropic, noop), LLM guardrails, golden file contract tests | `semantic_adapter.py`, `llm_guardrails.py` |
| **Cleanup** | Deprecated APIs removed (`--enrich-config`, legacy scripts) | [#59](https://github.com/EffortlessMetrics/slower-whisper/issues/59) |
| **Post-Processing** | `PostProcessor` orchestration, topic segmentation (TF-IDF), turn-taking policies (aggressive/balanced/conservative), domain presets | `post_process.py`, `topic_segmentation.py`, `turn_taking_policy.py` |

### Performance Targets

| Metric | Target | Source |
|--------|--------|--------|
| ASR WER (LibriSpeech) | < 5% | Benchmarks |
| Diarization DER (AMI) | < 15% | Benchmarks |
| Streaming P95 latency | < 500ms | Streaming |
| Semantic Topic F1 | > 0.8 | Semantics |
| Concurrent streams/GPU | > 10 | Streaming |

<details>
<summary><strong>Design Notes: Receipt Contract Specification</strong></summary>

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

<details>
<summary><strong>Design Notes: Dataset Manifest Contract</strong></summary>

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

</details>

<details>
<summary><strong>Design Notes: Benchmark Results & Baselines</strong></summary>

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

```
regression = (current_value - baseline_value) / baseline_value
fail_if: regression > threshold_percent (default: 10%)
```

</details>

<details>
<summary><strong>Design Notes: Event Envelope Specification</strong></summary>

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

<details>
<summary><strong>Design Notes: Semantics Contract</strong></summary>

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

<details>
<summary><strong>Design Notes: Post-Processing Architecture</strong></summary>

#### PostProcessor Execution Order

1. Safety processing (PII + moderation + formatting)
2. Environment classification
3. Extended prosody analysis
4. Turn-taking evaluation

Turn-level processors (roles, topics) run separately via `process_turn()`.

#### Topic Boundary Detection

```python
# Similarity-based boundary detection
similarity = cosine_similarity(prev_window_tfidf, curr_window_tfidf)
should_split = (
    similarity < config.similarity_threshold  # default: 0.35
    or topic_duration >= config.max_topic_duration_sec  # default: 300s
)
```

#### Turn-Taking Confidence Calculation

```python
# Weighted signal aggregation
confidence = (
    silence_strength * policy.silence_weight    # default: 0.40
    + punct_strength * policy.punctuation_weight # default: 0.35
    + prosody_strength * policy.prosody_weight   # default: 0.25
)
```

#### Reason Codes

| Code | Description |
|------|-------------|
| `SILENCE_THRESHOLD` | Silence exceeded configured threshold |
| `TERMINAL_PUNCT` | Period, exclamation, or question mark detected |
| `FALLING_INTONATION` | Prosodic boundary tone falling |
| `COMPLETE_SENTENCE` | Heuristic sentence completion |
| `QUESTION_DETECTED` | Question mark with question semantics |
| `LONG_PAUSE` | Forced end due to max silence |

</details>

---

## Current: v2.0.x Maintenance

**Goal:** Stabilize v2.0, close remaining gaps, expand dataset coverage.

| Item | Issue | Status |
|------|-------|--------|
| Expanded benchmark datasets (AMI, CALLHOME, LibriSpeech full) | [#48](https://github.com/EffortlessMetrics/slower-whisper/issues/48) | ⬜ Open |
| Semantic quality benchmark (Topic F1) | [#98](https://github.com/EffortlessMetrics/slower-whisper/issues/98) | ⬜ Open |
| Incremental diarization hook for streaming | [#86](https://github.com/EffortlessMetrics/slower-whisper/issues/86) | ⬜ Open |
| Real ASR smoke tests (engine + compat layer) | — | ✅ Done |
| Nightly diarization CI with real pyannote | — | ✅ Done |
| Test tier hardening (smoke/heavy/nightly stages) | — | ✅ Done |
| Docs polish (streaming, post-processing, contributing) | — | ✅ Done |

---

## Next: v2.1+

| Feature | Issue | Notes |
|---------|-------|-------|
| Full incremental diarization for streaming | [#86](https://github.com/EffortlessMetrics/slower-whisper/issues/86) | Real-time speaker updates as audio arrives |
| Semantic quality benchmark (Topic F1) | [#98](https://github.com/EffortlessMetrics/slower-whisper/issues/98) | Measure annotation quality against golden sets |
| Expanded benchmark datasets | [#48](https://github.com/EffortlessMetrics/slower-whisper/issues/48) | AMI, CALLHOME, LibriSpeech full evaluation |
| Additional cloud LLM backends | — | Beyond OpenAI/Anthropic |
| Domain-specific prompt templates | — | Vertical-specific semantic prompts |
| Streaming authentication | — | Bearer token or WS subprotocol auth |

## Later: v3.0 (2027+)

| Feature | Issue | Notes |
|---------|-------|-------|
| Intent detection (prosody + text fusion) | [#60](https://github.com/EffortlessMetrics/slower-whisper/issues/60) | Multimodal intent from acoustic + text signals |
| Clinical speech domain pack | [#61](https://github.com/EffortlessMetrics/slower-whisper/issues/61) | Healthcare-specific models and vocabulary |
| Acoustic scene analysis | [#62](https://github.com/EffortlessMetrics/slower-whisper/issues/62) | Environment and noise classification |
| Discourse structure analysis | [#65](https://github.com/EffortlessMetrics/slower-whisper/issues/65) | Conversation structure beyond turns |
| Domain pack plugin architecture | — | Pluggable vertical packs |

---

## Dependency Map (v2.0 — historical reference)

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
                   │
          ┌────────▼────────┐
          │  Track 5:       │
          │  Post-Process   │
          │ (topics, turns) │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │    v2.0.0       │
          │    Release      │
          └─────────────────┘
```

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
| **v1.x** | Stabilize, enrich, polish |
| **v2.x** | Real-time streaming + benchmarks + semantic adapters (current) |
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
