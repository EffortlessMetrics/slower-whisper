# Current Status

<!-- Manually maintained. Related: ./scripts/generate-roadmap-status.py updates ROADMAP.md Quick Status -->

**Last generated:** See git commit timestamp
**Version:** v2.0.0

This is a facts-only status page. Plan and strategy live in [ROADMAP.md](../ROADMAP.md).

> **Note:** Issue status reflects GitHub at generation time. Issues may remain Open while a PR that closes them is pending review.

---

## Issue Tracker Summary

| Category | Open | Closed | Total |
|----------|------|--------|-------|
| [All Issues](https://github.com/EffortlessMetrics/slower-whisper/issues) | — | — | [View](https://github.com/EffortlessMetrics/slower-whisper/issues?q=is%3Aissue) |
| [v2.0 Milestone](https://github.com/EffortlessMetrics/slower-whisper/issues?q=label%3Av2.0) | — | — | [View](https://github.com/EffortlessMetrics/slower-whisper/issues?q=label%3Av2.0) |
| [High Priority](https://github.com/EffortlessMetrics/slower-whisper/issues?q=label%3Apriority%2Fhigh) | — | — | [View](https://github.com/EffortlessMetrics/slower-whisper/issues?q=label%3Apriority%2Fhigh) |

> Note: Counts are live links to GitHub. Check the tracker for current numbers.

---

## Track Status

### API Polish Bundle

**Status:** Shipped in PR #158

| Issue | Title | Status |
|-------|-------|--------|
| [#70](https://github.com/EffortlessMetrics/slower-whisper/issues/70) | `transcribe_bytes()` API | Shipped (#158) |
| [#71](https://github.com/EffortlessMetrics/slower-whisper/issues/71) | `word_timestamps` REST parameter | Shipped (#158) |
| [#72](https://github.com/EffortlessMetrics/slower-whisper/issues/72) | Word-level timestamps example | Shipped (#158) |
| [#78](https://github.com/EffortlessMetrics/slower-whisper/issues/78) | `Transcript` convenience methods | Shipped (#158) |

### Track 1: Benchmarks

**Status:** Mostly shipped

| Order | Issue | Title | Status |
|-------|-------|-------|--------|
| 1 | [#95](https://github.com/EffortlessMetrics/slower-whisper/issues/95) | ASR WER runner | Shipped (#186) |
| 2 | [#137](https://github.com/EffortlessMetrics/slower-whisper/issues/137) | Baseline file format | Shipped (#201) |
| 3 | [#97](https://github.com/EffortlessMetrics/slower-whisper/issues/97) | Streaming latency harness | Shipped (#190) |
| 4 | [#96](https://github.com/EffortlessMetrics/slower-whisper/issues/96) | Diarization DER runner | Shipped (#189) |
| 5 | [#99](https://github.com/EffortlessMetrics/slower-whisper/issues/99) | CI integration | Partial (Phase 2 complete) |

### Track 2: Streaming

**Status:** Partial

| Order | Issue | Title | Status |
|-------|-------|-------|--------|
| 1 | [#222](https://github.com/EffortlessMetrics/slower-whisper/issues/222) | Event envelope spec | Open |
| 2 | [#223](https://github.com/EffortlessMetrics/slower-whisper/issues/223) | Reference Python client | Open |
| 3 | [#84](https://github.com/EffortlessMetrics/slower-whisper/issues/84) | WebSocket endpoint | Partial (skeleton in #201) |
| 4 | [#85](https://github.com/EffortlessMetrics/slower-whisper/issues/85) | REST streaming endpoints | Open |
| 5 | [#55](https://github.com/EffortlessMetrics/slower-whisper/issues/55) | Streaming API docs | Partial (STREAMING_ARCHITECTURE.md) |
| 6 | [#86](https://github.com/EffortlessMetrics/slower-whisper/issues/86) | Incremental diarization | Open |

### Track 3: Semantics

**Status:** Mostly shipped

| Order | Issue | Title | Status |
|-------|-------|-------|--------|
| 1 | [#88](https://github.com/EffortlessMetrics/slower-whisper/issues/88) | LLM annotation schema | Shipped (#201) |
| 2 | [#90](https://github.com/EffortlessMetrics/slower-whisper/issues/90) | Cloud LLM interface | Partial (Anthropic #188, OpenAI pending) |
| 3 | [#91](https://github.com/EffortlessMetrics/slower-whisper/issues/91) | Guardrails | Shipped (#201) |
| 4 | [#92](https://github.com/EffortlessMetrics/slower-whisper/issues/92) | Golden files + tests | Shipped (#201) |
| 5 | [#89](https://github.com/EffortlessMetrics/slower-whisper/issues/89) | Local LLM backend | Open |
| 6 | [#98](https://github.com/EffortlessMetrics/slower-whisper/issues/98) | Semantic quality benchmark | Shipped (#159, #201) |

---

## Infrastructure Issues

| Issue | Purpose | Status |
|-------|---------|--------|
| [#135](https://github.com/EffortlessMetrics/slower-whisper/issues/135) | Receipt contract | Partial (spec designed, classes exist) |
| [#136](https://github.com/EffortlessMetrics/slower-whisper/issues/136) | Stable run/event IDs | Partial (spec designed) |
| [#137](https://github.com/EffortlessMetrics/slower-whisper/issues/137) | Baseline file format | Shipped (#201) |

---

## Regeneration

```bash
# Regenerate ROADMAP Quick Status from GitHub
./scripts/generate-roadmap-status.py --write

# Check if ROADMAP is stale (for CI)
./scripts/generate-roadmap-status.py --check
```

---

## Links

- [ROADMAP.md](../ROADMAP.md) — Plan and strategy
- [CHANGELOG.md](../CHANGELOG.md) — Release history
- [GitHub Issues](https://github.com/EffortlessMetrics/slower-whisper/issues) — Full tracker
