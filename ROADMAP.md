# slower-whisper Roadmap

**Current Version:** v1.1.0 (Diarization + LLM rendering)
**Last Updated:** 2025-11-30
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
**Status:** Stable and supported; superseded by v1.1.0 for diarization/LLM
features

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
**Status:** Stable; diarization is opt-in/experimental

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

## v1.2.0 — Speaker Analytics & Evaluation (**Target: Q2 2026**)

**Theme:** Make speaker data **useful for LLMs** with turn metadata, stats, and
prompt builders built on the v1.1 diarization layer.

**Goal:** Enable speaker-aware summarization, QA, and coaching use cases.

**Status:** *Released when ready. Dates are targets, not commitments.*

### Core Features (v1.2.0)

#### 1. Turn Metadata

**Extend `turns[]` with interaction metadata:**

```json
"turns": [
  {
    "id": "turn_17",
    "speaker_id": "spk_1",
    "segment_ids": [23, 24],
    "start": 123.45,
    "end": 140.10,
    "text": "... concatenated ...",
    "meta": {
      "question_count": 1,              // Detected questions
      "interruption_started_here": false,
      "avg_pause_ms": 320,
      "disfluency_ratio": 0.08
    }
  }
]
```

**Implementation:**

- Detect questions via text heuristics (ends with `?`, starts with wh-words)
- Detect interruptions via turn overlap timing
- Aggregate pauses and disfluencies from segments

#### 2. Speaker Stats

**Per-speaker aggregates for LLM prompts:**

```json
"speaker_stats": [
  {
    "speaker_id": "spk_0",
    "total_talk_time": 512.3,
    "num_turns": 34,
    "avg_turn_duration": 15.1,
    "interruptions_initiated": 4,
    "interruptions_received": 7,
    "question_turns": 9,
    "prosody_summary": {
      "pitch_median_hz": 180.5,
      "energy_median_db": -12.3
    },
    "sentiment_summary": {
      "positive": 0.3,
      "neutral": 0.5,
      "negative": 0.2
    }
  }
]
```

#### 3. Prompt Builder Utilities

**Extend v1.1 rendering with analytics-aware helpers:**

```python
from transcription import load_transcript, to_turn_view, to_speaker_summary

transcript = load_transcript("meeting.json")

# Turn-level dialogue view
prompt_text = to_turn_view(transcript, include_audio_state=True)

# Speaker summary table
speaker_info = to_speaker_summary(transcript)

# Combined prompt
full_prompt = f"{speaker_info}\n\nConversation:\n{prompt_text}"
```

**Output example:**

```text
Speakers:
- Speaker A (spk_0): 65% talk time, 4 interruptions initiated
- Speaker B (spk_1): 35% talk time, 1 interruption initiated

Conversation (turn-level):
[00:15.2–00:22.8] Speaker B: I'm not sure this pricing works...
  [audio: concerned tone, rising pitch]
[00:23.0–00:35.1] Speaker A: Totally understand. Let me walk through...
```

#### 4. MVP Evaluation Harness

**Build minimal testbed** (see `docs/TESTING_STRATEGY.md`):

- **20-50 labeled segments** (not 200+)
- **One task**: "Does speaker labeling improve summarization?"
- **Two conditions**:
  - Baseline: text-only transcript
  - Enriched: text + speaker labels + turn structure
- **Metric**: LLM-as-judge preference (which summary is better?)

**Implementation:**

```bash
# Run evaluation on AMI Meeting Corpus subset
uv run python benchmarks/eval_speaker_utility.py \
  --dataset ami_subset \
  --task summarization \
  --judge gpt-4o-mini
```

**Success criteria:**

- Enriched condition wins >60% of pairwise comparisons
- Clear examples where speaker info helps (e.g., "Agent said X, customer
  objected with Y")

### Developer Experience (v1.2.0)

- `--role-hint` CLI flag for manual speaker role annotation:

  ```bash
  slower-whisper transcribe --enable-diarization \
    --role-hint spk_0=agent,spk_1=customer
  ```

- Progress bars for turn analysis
- Example scripts showing LLM integration

### Deliverables (v1.2.0)

- [ ] Turn metadata builder (extends `transcription/turns.py`)
- [ ] Speaker stats aggregator (`transcription/speaker_stats.py`)
- [ ] Prompt builder utilities (`transcription/llm_utils.py`)
- [ ] MVP evaluation harness (`benchmarks/eval_speaker_utility.py`)
- [ ] BDD scenarios for turn metadata correctness
- [ ] Documentation: `docs/SPEAKER_ANALYTICS.md`
- [ ] Examples: Speaker-aware summarization, QA

---

## v1.3.0 — LLM Ecosystem Integration (**Target: Q3 2026**)

**Theme:** Make slower-whisper **trivial to use** with LangChain, LlamaIndex,
and vector DBs.

**Goal:** Become the standard input layer for conversation-aware LLM
applications.

**Status:** *Released when ready. Dates are targets, not commitments.*

### Core Features (v1.3.0)

#### 1. Intelligent Chunking

**Implement `chunks[]` with turn-aware boundaries:**

```json
"chunks": [
  {
    "id": "chunk_0",
    "start": 0.0,
    "end": 120.0,
    "segment_ids": [0, 1, 2, 3, 4, 5],
    "turn_ids": ["turn_0", "turn_1", "turn_2"],
    "speaker_ids": ["spk_0", "spk_1"],
    "token_count_estimate": 512
  }
]
```

**Strategy:**

- Target 512-1024 tokens OR 60-120 seconds
- **Never split mid-turn** (prefer turn boundaries)
- Fall back to pause boundaries if turns too long
- Include speaker + audio_state aggregates per chunk

#### 2. LangChain Adapter

**Official integration:**

```python
from langchain.document_loaders import SlowerWhisperLoader

loader = SlowerWhisperLoader("meeting.json")
docs = loader.load()  # One Document per chunk

for doc in docs:
    print(doc.page_content)  # Turn-level text
    print(doc.metadata)      # Speaker, timestamps, audio_state summary
```

**Metadata exposed:**

- `speakers`: List of speaker IDs in chunk
- `start_time`, `end_time`: Chunk boundaries
- `prosody_summary`: Aggregated pitch/energy/rate
- `emotion_summary`: Dominant emotion in chunk
- `turn_count`, `question_count`, `interruption_count`

#### 3. LlamaIndex Reader

**Official integration:**

```python
from llama_index import SlowerWhisperReader

documents = SlowerWhisperReader().load_data("meeting.json")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("Where did the customer express frustration?")
```

**Indexing strategy:**

- Chunk-level indexing with acoustic metadata
- Filterable by speaker, emotion, time range
- Retrieval considers both text and `audio_state`

#### 4. Export Formats

**Additional outputs for analysis and interoperability:**

- **WebVTT**: Web video subtitles with speaker labels
- **CSV**: Tabular format (id, start, end, speaker, text, valence, arousal,
  pitch, energy)
- **Annotated HTML**: Web viewing with speaker colors and audio_state tooltips
- **Praat TextGrid**: For phonetics research (tier per speaker, prosody
  annotations)

**Example:**

```bash
# Export to CSV for analysis in Excel/pandas
slower-whisper export meeting.json --format csv --output meeting.csv

# Export to HTML for human review
slower-whisper export meeting.json --format html --output meeting.html
```

#### 5. Schema v2 Finalization

**Lock in core schema contract:**

- Document deprecation policy (`docs/SCHEMA_STABILITY.md`)
- Publish JSON Schema (draft-07) validation file
- Examples for every optional field
- Migration guide for future v3

**Stability guarantee:**

> Within schema v2.x, core fields (`audio`, `meta`, `speakers`, `segments`,
> `turns`, `chunks`) will NOT change meaning. Optional fields may be added but
> never removed without a major version bump.

### Developer Experience (v1.3.0)

- Compact export mode: `--fields text,speaker,audio_state.rendering`
- JSON Schema validation: `slower-whisper validate meeting.json`
- Examples for every integration (LangChain, LlamaIndex, pandas, DuckDB)

### Deliverables (v1.3.0)

- [ ] Chunking implementation (`transcription/chunking.py`)
- [ ] LangChain loader (`integrations/langchain_loader.py`)
- [ ] LlamaIndex reader (`integrations/llamaindex_reader.py`)
- [ ] Export formats (WebVTT, CSV, HTML, Praat TextGrid)
- [ ] JSON Schema validation file (`schema/transcript-v2.json`)
- [ ] Documentation: `docs/LLM_INTEGRATION.md`, `docs/SCHEMA_STABILITY.md`
- [ ] Examples: RAG, summarization, Q&A with LangChain/LlamaIndex

---

## v2.0.0 — Streaming, Semantic Layer & Extensibility (**Target: Q4 2026**)

**Theme:** Real-time processing + optional semantic enrichment (Layer 3) +
plugin architecture.

**Goal:** Production-ready for live use cases + semantic understanding for power
users.

**Status:** *Released when ready. Major version = breaking changes allowed.*

### Core Features (v2.0.0)

#### 1. Streaming Transcription & Enrichment

- Incremental Whisper on sliding windows
- Real-time diarization and prosody extraction
- WebSocket API for live results
- Low-latency mode (<500ms delay)

**Use cases:**

- Live captioning
- Real-time meeting assistance
- Call center agent coaching

#### 2. Semantic Layer (L3) — Optional SLM Integration

**Now that local SLMs are mature (2027), add optional semantic enrichment:**

**Abstraction:**

```python
from transcription import SemanticAnnotator

annotator = SemanticAnnotator(model="qwen2.5-vl-7b", device="cuda")
annotations = annotator.run(transcript, chunk)
```

**Chunk-level outputs (v2.0+ only):**

```json
"chunks": [
  {
    "summary": "Customer voices pricing concern; agent offers alternatives.",
    "semantic_tags": ["objection", "pricing_discussion"],
    "annotations": {
      "llm": [  // Reserved for v2.0+ semantic layer; empty in v1.x
        {
          "type": "interaction_pattern",
          "label": "objection_handling",
          "confidence": 0.85
        }
      ]
    }
  }
]
```

**Note:** These fields are **not populated in v1.x** unless you write custom
plugins.

**Design constraints:**

- **Opt-in only** (`--enable-semantic`)
- **Chunked processing** (60-120s, not per-token)
- **Pluggable backends** (Qwen, SmolLM, custom models)
- **Fully cached** by `(audio_hash, asr_hash, enrichment_hash, slm_model_hash)`

**Evaluation:**

- Compare: text vs text+acoustic vs text+acoustic+semantic
- Measure: downstream task performance (summarization, QA, coaching)

#### 3. Plugin System

**Allow custom enrichment without forking:**

```python
from transcription import register_enrichment_plugin

@register_enrichment_plugin("my_feature")
def extract_custom_feature(segment, audio):
    # Custom logic
    return {"my_metric": 0.85}
```

**Plugin capabilities:**

- Custom feature extractors (Layer 2 or Layer 3)
- Custom output formatters (export to proprietary formats)
- Custom chunking strategies

#### 4. Distributed Processing

- Multi-node job queueing (Celery or similar)
- S3/GCS/Azure storage backends
- Kubernetes operator for auto-scaling

**Use cases:**

- Enterprise batch processing (1000s of hours)
- Multi-tenant SaaS deployments

### Breaking Changes (v1.x → v2.0)

**Schema changes:**

- `chunks[]` becomes required (not optional)
- `audio.id` changes from file path to content hash
- Deprecated fields from v1.x removed

**API changes:**

- Legacy CLI removed (unified CLI only)
- Old config format deprecated

**Migration:**

- Automatic migration tool: `slower-whisper migrate v1-to-v2 transcript.json`
- Compatibility mode: `--schema-version 1` to read old files

### Deliverables (v2.0.0)

- [ ] Streaming ASR module (`transcription/streaming.py`)
- [ ] Semantic annotator interface (`transcription/semantic.py`)
- [ ] Qwen/SmolLM backend implementations
- [ ] Plugin API documentation
- [ ] Cloud storage backends
- [ ] K8s operator (CRDs, controllers)
- [ ] Migration tooling
- [ ] Documentation: Streaming guide, semantic enrichment guide, plugin
  development guide

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

### High Priority (v1.1.x hardening)

- [ ] Diarization benchmarking on AMI subset + synthetic fixtures; publish DER
  and speaker-count results
- [ ] CI/BDD coverage for diarization correctness (two-speaker success,
  overlap resilience)
- [ ] pyannote download/auth UX and progress messaging for
  `--enable-diarization`
- [ ] JSON Schema (draft-07) validation file

### Next Up (v1.2 readiness)

- [ ] Turn metadata builder and speaker stats aggregator
- [ ] Prompt builder utilities aligned with analytics data
- [ ] MVP evaluation harness on AMI/IEMOCAP for speaker utility
- [ ] Speaker-aware examples (summarization, QA)

### Integrations & Research (v1.3+)

- [ ] LangChain adapter
- [ ] LlamaIndex adapter
- [ ] WebVTT / CSV / HTML exporters
- [ ] Praat TextGrid export
- [ ] Custom SER (speech emotion recognition) models
- [ ] Prosody feature experiments
- [ ] Multi-language support
- [ ] Acoustic similarity clustering

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
