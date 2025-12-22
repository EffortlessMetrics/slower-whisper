# slower-whisper Roadmap

**Current Version:** v1.8.0 (Word-level timestamps and speaker alignment)
**Last Updated:** 2025-12-22
<!-- cspell:ignore pyannote disfluency disfluencies langchain llamaindex Praat
cuda qwen Qwen Smol Neur INTERSPEECH IEMOCAP multimodal mypy -->

This roadmap outlines the evolution from **transcription tool** to **local
conversation intelligence infrastructure**.

See [VISION.md](VISION.md) for strategic positioning and long-term goals.

---

## Versioning Philosophy

- **v1.x** — Stabilize, enrich, and modularize (current focus)
- **v2.x** — Real-time, streaming, and architectural extensibility
- **v3.x** — Semantic understanding and domain specialization

**Principle**: Each major version adds **layers**, not rewrites.
v1.x JSON is forward-compatible with v2.x readers.

---

## v1.0.0 — Production Foundation (SHIPPED ✅)

**Released:** 2025-11-17
**Status:** Stable and supported; superseded by v1.3.0 for analytics, exports, and evaluation

### What Shipped (v1.0.0)

**Core Pipeline (Layer 1):**

- ✅ Whisper transcription via faster-whisper
- ✅ Stable JSON schema v2 with versioning
- ✅ TXT and SRT outputs
- ✅ Configuration system (CLI > file > env > defaults)
- ✅ Python API (`transcribe_directory`, `transcribe_file`)

**Audio Enrichment (Layer 2, basic):**

- ✅ Prosody extraction (pitch, energy, rate, pauses)
- ✅ Emotion recognition (valence/arousal, categorical emotions)
- ✅ LLM-friendly text rendering (`[audio: high pitch, fast speech]`)
- ✅ Speaker-relative baseline normalization

**Infrastructure:**

- ✅ Docker images (CPU, GPU, API service)
- ✅ Kubernetes manifests (deployment, HPA, jobs)
- ✅ Docker Compose (batch, dev, API)
- ✅ REST API service (FastAPI)

**Quality & Testing:**

- ✅ 191 passing tests (57% coverage)
- ✅ BDD scenarios (library + API contracts)
- ✅ Verification CLI (`slower-whisper-verify`)
- ✅ Pre-commit hooks (ruff, mypy)
- ✅ IaC smoke tests (Docker + K8s validation)

**Documentation:**

- ✅ Comprehensive guides (15+ docs)
- ✅ API quick reference
- ✅ Examples (12+ working scripts)
- ✅ BDD/IaC contract documentation

---

## v1.1.0 — Speaker Foundation (SHIPPED ✅)

**Released:** 2025-11-18
**Status:** Stable; superseded by v1.3.0 for analytics/export coverage (diarization remains opt-in/experimental)

### What Shipped (v1.1.0)

- **Speaker diarization + turns (Layer 2):** pyannote backend with overlap-aware
  segment mapping, `speakers[]` table, `turns[]` grouping, and
  `meta.diarization` (`status`, `requested`, `backend`, `error_type`).
- **LLM rendering APIs:** `render_conversation_for_llm`,
  `render_conversation_compact`, and `render_segment` with speaker-label
  mapping, timestamp/audio-cue options, and graceful degradation when
  speakers/turns are absent.
- **Examples, docs, and tests:** Working scripts in `examples/llm_integration/`;
  documentation (`docs/SPEAKER_DIARIZATION.md`, `docs/LLM_PROMPT_PATTERNS.md`,
  `examples/llm_integration/README.md`); 21 new tests covering rendering,
  speaker labels, and edge cases.
- **CLI & DX:** `--enable-diarization` flag (extra dependencies via
  `uv sync --extra diarization`), `--version` flag, and updated help text
  pointing to diarization docs.
- **Schema compatibility:** Speakers/turns remain optional in schema v2 with
  normalized string speaker IDs end-to-end.

### v1.1.x Hardening (short-term priorities)

- Benchmark diarization on AMI subset and synthetic fixtures (DER < 0.25;
  correct speaker counts on >90% synthetic cases).
- Add BDD/fixture coverage for diarization correctness (two-speaker success,
  overlap resilience) and wire into CI smoke tests.
- Improve operational UX: clearer pyannote download/auth errors, progress
  indicators when `--enable-diarization` is used, structured failure reasons
  surfaced via `meta.diarization`.
- Document and monitor regression guardrails in `docs/TESTING_STRATEGY.md` and
  existing diarization trace docs.

---

## v1.2.0 — Speaker Analytics & Evaluation (SHIPPED ✅)

**Released:** 2025-12-01
**Status:** Stable; folded into v1.3.0 alongside exports/adapters/evaluation.

### What Shipped (v1.2.0)

- **Turn metadata (`turns[].metadata`)**: `question_count`,
  `interruption_started_here`, `avg_pause_ms`, and `disfluency_ratio`, populated
  during enrichment and serialized to JSON.
- **Speaker stats (`speaker_stats[]`)**: Talk time, turn counts, average turn
  duration, interruptions initiated/received, question turns, plus prosody and
  sentiment summaries.
- **Analytics controls**: CLI/API flags for turn metadata and speaker stats,
  honoring `skip_existing` so re-runs do not clobber existing analytics.
- **Diarization metadata state machine**: `disabled`, `skipped`, `ok`, `error`
  statuses written to `meta.diarization` even when diarization is turned off.
- **LLM + serialization**: Pattern 6 in `docs/LLM_PROMPT_PATTERNS.md`; JSON
  round-tripping for `turns`, `speaker_stats`, and diarization metadata.

### Research / Benchmarks (post-1.2)

- MVP evaluation harness for speaker-aware summarization (text vs enriched,
  LLM-as-judge).
- Diarization benchmark skeleton (`benchmarks/eval_diarization.py`, dataset in
  `benchmarks/data/diarization/manifest.jsonl`) emitting JSON/Markdown with DER
  and speaker-count checks.
- Role-hint flag and progress UX for long-running analytics.
- Additional BDD/fixture coverage once evaluation datasets are locked.

---

## v1.3.0 — Ecosystem & Evaluation (SHIPPED ✅)

**Released:** 2026-01-15
**Status:** Current version; consolidates ecosystem adapters, exports, and populated benchmarks.

### What Shipped (v1.3.0)

- **Turn-aware chunking:** `chunks[]` added with turn-preserving boundaries and schema/BDD coverage.
- **Exports + validation:** `slower-whisper export` (CSV/HTML/VTT/TextGrid) and `slower-whisper validate` (JSON Schema v2) with CLI/API integration tests.
- **LLM ecosystem adapters:** LangChain and LlamaIndex loaders plus speaker-aware summarization example in `examples/llm_integration/`.
- **Semantic annotation (opt-in):** Keyword-based `SemanticAnnotator` writing to `annotations.semantic` with guardrails and flags.
- **Performance harness:** Throughput probe + `docs/PERFORMANCE.md` to baseline CPU/GPU paths.
- **Benchmarks populated:** ASR WER (`benchmarks/ASR_REPORT.*`), diarization DER (`benchmarks/DIARIZATION_REPORT.*`), and speaker analytics preference check (`benchmarks/SPEAKER_ANALYTICS_MVP.md`).
- **Examples:** Metrics/KPI script and redaction walkthrough for quick adoption.

---

## Type System Hardening (DONE ✅)

**Completed:** 2026-01-XX (part of v1.3.1 maintenance)

### What Was Done

- **Full mypy coverage**: All 39 modules in `transcription/` pass mypy with zero errors (92.9% function-level annotation coverage, 209/225 functions typed).
- **Strategic test typing**: Four test modules (`test_llm_utils.py`, `test_writers.py`, `test_turn_helpers.py`, `test_audio_state_schema.py`) configured for mypy validation.
- **Protocol patterns**: `EmotionRecognizerLike`, `WhisperModelProtocol`, `SemanticAnnotator` protocols enable graceful degradation for optional dependencies.
- **PEP 561 compliance**: `py.typed` marker present for downstream type-checker support.
- **Typing policy documented**: `docs/TYPING_POLICY.md` captures gradual typing strategy, `cast()` usage rationale, and contribution guidelines.

### Typing Configuration

- **mypy mode**: Gradual (not strict) — `disallow_untyped_defs=false`, `check_untyped_defs=true`
- **Pre-commit**: mypy non-blocking (`|| true`) to avoid friction during development
- **CI gate**: mypy runs on `transcription/` and strategic test modules

### Related Files

- `pyproject.toml`: mypy configuration (lines 203-245)
- `pyrightconfig.json`: VS Code/Pylance LSP support
- `transcription/py.typed`: PEP 561 marker
- `docs/TYPING_POLICY.md`: Typing standards and contribution guidelines

---

## v1.7.0 — Streaming Enrichment & Live Semantics (SHIPPED ✅)

**Released:** 2025-12-02
**Status:** Superseded by v1.8.0

### What Shipped (v1.7.0)

- **Streaming Audio Enrichment (`StreamingEnrichmentSession`)**: Real-time audio feature extraction for streaming transcription with prosody and emotion analysis as segments are finalized. Provides low-latency enrichment (~60-220ms) for live applications with graceful error handling and session statistics.
- **Live Semantic Annotation (`LiveSemanticSession`)**: Turn-aware semantic enrichment for streaming conversations with automatic speaker turn detection, keyword extraction, risk flag detection, and action item identification. Maintains rolling context window for conversation coherence.
- **Unified Configuration API (`Config.from_sources()`)**: New classmethod for `TranscriptionConfig` and `EnrichmentConfig` that loads settings from multiple sources with proper precedence (CLI args > config file > environment > defaults). Simplifies programmatic config creation without argparse.
- **Configuration Documentation (`docs/CONFIGURATION.md`)**: Comprehensive guide to configuration management, precedence rules, and usage examples for all configuration sources.
- **Extended Streaming Event Types**: Added `SEMANTIC_UPDATE` event type to `StreamEventType` enum for real-time semantic annotation events.
- **StreamSegment Schema Enhancement**: Added `audio_state` field to `StreamSegment` dataclass for carrying enrichment data through streaming pipeline.

### Streaming Architecture

- **Low-latency enrichment**: ~60-220ms per segment for prosody + emotion extraction
- **Turn-aware semantics**: Automatic speaker turn detection with configurable pause thresholds (default: 1.0s)
- **Event-driven design**: Clean event types (`SEGMENT_FINALIZED`, `SEMANTIC_UPDATE`) for downstream integration
- **Session statistics**: Built-in counters and metrics for monitoring streaming performance

### Examples & Documentation

- `examples/streaming/enrich_stream_from_transcript.py`: Demo streaming enrichment on pre-transcribed data
- `examples/streaming/live_semantics_demo.py`: Demo turn-aware semantic annotation with optional LLM integration
- `docs/CONFIGURATION.md`: Complete configuration management guide
- Updated `docs/API.md`: Quick reference for new streaming and config APIs

## v1.8.0 — Word-Level Alignment (SHIPPED ✅)

**Released:** 2025-12-22
**Status:** Stable; adds granular word-level timestamps and speaker alignment

### What Shipped (v1.8.0)

- **Word-level timestamps**: New `--word-timestamps` CLI flag and `word_timestamps` config option enable per-word timing extraction from faster-whisper. Each word includes start/end timestamps and confidence probability.
- **Word model**: New `Word` dataclass exported from `transcription` package with fields: `word`, `start`, `end`, `probability`, and optional `speaker`.
- **Segment.words field**: Segments now include an optional `words` list containing `Word` objects when word-level timestamps are enabled.
- **Word-level speaker alignment** (`assign_speakers_to_words`): New function for granular speaker assignment at the word level, enabling detection of speaker changes within segments. Segment speaker is derived from the dominant word-level speaker.
- **JSON serialization**: Word-level timestamps are automatically serialized to/from JSON with backward compatibility (old transcripts without words load correctly).

### Future Enhancements (v1.9+)

- Streaming semantics integration quality improvements
- Event callback API for easier downstream integration
- Semantic tuning based on real-world usage
- Enhanced test coverage for streaming semantics

---

## Code Quality & Maintainability (ONGOING 🔧)

**Updated:** 2025-12-22

### Codebase Health Metrics

- **Test coverage**: 713 tests passing
- **Type coverage**: 43/43 source files pass mypy
- **Lint status**: All ruff checks pass
- **Documentation**: 65+ markdown docs

### Recent Improvements (Post-1.8.0)

**Error Handling & Logging:**
- Fixed dead error handler in `audio_io.py` - now captures and logs ffmpeg stderr
- Added debug logging for silent exceptions in `diarization.py` stub mode

**Code Deduplication:**
- Extracted `_extract_audio_descriptors()` utility in `llm_utils.py` (consolidated 3 duplicate implementations)
- Identified additional consolidation opportunities for timestamp formatters and dict conversion helpers

**Documentation:**
- Added docstrings to 10+ undocumented helper functions across `chunking.py`, `turns_enrich.py`, `speaker_stats.py`, `validation.py`
- Improved CLI help text consistency across transcribe/enrich commands

**CLI Improvements:**
- Renamed `--enrich-config` to `--config` for command consistency
- Clarified device flag help text (ASR vs emotion models)
- Added choices validation for device arguments

### Known Technical Debt

**Test Coverage Gaps** (identified via exploration):
- 18 of 41 modules have dedicated test files (~44% module coverage)
- Key untested modules: `api.py`, `pipeline.py`, `cli.py`, `service.py`
- Recommendation: Prioritize test coverage for public API surface

**Code Duplication** (low priority):
- 3 dict conversion helpers (`_to_dict`, `_as_dict`, `turn_to_dict`) with similar logic
- 4 timestamp formatting functions across modules
- Documented in CLAUDE.md; consolidation deferred to avoid breaking changes

**Configuration Complexity**:
- Overlapping `AsrConfig`, `AppConfig`, `TranscriptionConfig` classes
- Legacy backward-compatibility maintained but adds cognitive overhead

---

## v2.0.0 — Streaming & Semantic Depth (PLANNED 🚧)

**Theme:** Real-time streaming, LLM-backed semantic annotation, and larger benchmark coverage.

### Focus Areas

- **Streaming architecture:** Flesh out concrete partial/final message flows in `docs/STREAMING_ARCHITECTURE.md`; ensure `Transcript`/`Chunk` structures support incremental diarization and analytics.
- **LLM semantic annotator:** Design LLM-backed annotations (topic/risk/action) written into `annotations.semantic` without breaking v1.x consumers; define config knobs and safety/guardrail strategy.
- **Benchmarks at scale:** Expand DER/WER datasets beyond fixtures; add streaming + semantic evaluation tracks; harden performance/throughput gates.
- **DX + APIs:** Keep exports/validation contracts stable while introducing streaming endpoints and integration adapters for live flows.

---

## v3.0.0 — Intelligence Layer (2027+)

**Theme:** Semantic understanding + domain specialization.

### Core Features (v3.0.0)

#### 1. Semantic Audio Analysis

- Intent detection from prosody + text
- Discourse structure analysis
- Topic segmentation with acoustic cues

#### 2. Domain Packs

- Clinical speech analysis (therapy, diagnosis)
- Legal transcription (court proceedings)
- Meeting summarization (action items, decisions)

#### 3. Contextual Enrichment

- Background noise classification
- Acoustic scene analysis
- Audio event detection (laughter, applause)

---

## Community & Ecosystem Roadmap

### Documentation & Education

- [ ] Video tutorials and walkthroughs
- [ ] Interactive documentation with live examples
- [ ] Academic paper on acoustic feature rendering for LLMs
- [ ] Conference presentations (PyCon, NeurIPS, INTERSPEECH)

### Community Building

- [ ] Discord/Slack community
- [ ] Monthly community calls
- [ ] Contributor recognition program
- [ ] User showcase gallery

### Research Collaborations

- [ ] Partner with linguistics departments
- [ ] Collaborate with speech therapy researchers
- [ ] Contribute to open speech datasets
- [ ] Publish benchmarks and evaluation metrics

---

## Release Schedule

**Versioning Strategy:**

- **Major (X.0.0)**: Breaking changes, architectural shifts
- **Minor (x.X.0)**: New features, backward-compatible
- **Patch (x.x.X)**: Bug fixes, security patches

**Release Cadence:**

- **Patch releases**: As needed (security, critical bugs)
- **Minor releases**: ~3-4 months
- **Major releases**: ~12-18 months

**Long-Term Support:**

- v1.x receives security updates for 18 months after v2.0.0
- Critical bug fixes for 12 months after LTS period

---

## Contribution Opportunities

### Near Term (post-1.3.0 stabilization)

- [ ] Run pyannote-backed diarization benchmark (HF_TOKEN) and refresh `benchmarks/DIARIZATION_REPORT.*` plus `docs/SPEAKER_DIARIZATION.md`.
- [ ] Broaden CLI/integration smoke tests around `export`, `validate`, and metrics/redaction examples.
- [ ] Doc polish for LangChain/LlamaIndex adapters, chunking semantics, and KPI scripts.

### V2 Prep

- [ ] Flesh out streaming message sequences in `docs/STREAMING_ARCHITECTURE.md` and add lightweight contract tests.
- [ ] Specify LLM-backed `SemanticAnnotator` schema/config (topics/risks/actions) with guardrails.
- [ ] Grow evaluation datasets (ASR/DER/analytics) and automate gates in `make verify-quick`/CI.

### Longer Term

- [ ] Streaming endpoints + adapters once contracts are locked.
- [ ] Additional semantic tasks and domain packs after streaming lands.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.

---

## Feedback & Prioritization

This roadmap is a living document. Priorities shift based on:

- Community feedback and feature requests
- Research developments in speech AI
- Resource availability and contributor interest
- Security and stability needs

**How to influence the roadmap:**

1. Open feature request issues on GitHub
2. Vote on existing issues (👍 reactions)
3. Contribute pull requests
4. Participate in community discussions
5. Share your use cases and needs

---

## Deprecation Policy

**Current Deprecations:** None (v1.1.0 is stable; diarization remains opt-in)

**Future Deprecation Timeline:**

- Deprecation announced: 6 months before removal
- Warning period: Deprecation warnings in code
- Removal: Only in major version bumps

**Example:**

```text
v1.8.0: Deprecate legacy CLI (announce only)
v1.9.0: Legacy CLI works but logs warnings
v2.0.0: Legacy CLI removed (unified CLI only)
```

---

## Long-Term Vision (5+ years)

See [VISION.md](VISION.md) for complete strategic vision.

**Mission:** Make acoustic information accessible to text-based AI systems,
enabling truly multimodal understanding of human communication.

**Goals:**

1. **Universal Acoustic Encoding** — Standard format for representing
   audio-only information
2. **Research Accelerator** — Tool of choice for speech, linguistics, and
   psychology researchers
3. **Production Grade** — Enterprise-ready for commercial transcription and
   analytics
4. **Open Science** — Advance open-source speech AI and contribute to
   academic research
5. **Accessibility** — Enable better tools for hearing-impaired, language
   learners, and assistive technology

---

## Questions or Suggestions?

- **GitHub Issues:** Feature requests and discussions
- **Documentation:** [docs/INDEX.md](docs/INDEX.md)
- **Vision:** [VISION.md](VISION.md)
- **Community:** [Discord/Slack link — coming soon]

**Thank you for being part of the slower-whisper journey!**

---

**Document History:**

- 2025-11-17: Initial roadmap created (v1.0.0 release)
- 2025-11-17: Complete rewrite for layered architecture vision (v1.x focus)
- 2025-11-30: Updated for v1.1.0 release and diarization/LLM rendering
  shipment; added v1.1.x hardening priorities
- 2025-12-01: Updated for v1.2.0 (speaker analytics) and v1.3.0 (exports, evaluation)
