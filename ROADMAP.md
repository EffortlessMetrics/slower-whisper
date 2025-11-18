# slower-whisper Roadmap

**Current Version:** v1.0.0 (Production Ready)
**Last Updated:** 2025-11-17

This roadmap outlines the evolution from **transcription tool** to **local conversation intelligence infrastructure**.

See [VISION.md](VISION.md) for strategic positioning and long-term goals.

---

## Versioning Philosophy

- **v1.x** — Stabilize, enrich, and modularize (current focus)
- **v2.x** — Real-time, streaming, and architectural extensibility
- **v3.x** — Semantic understanding and domain specialization

**Principle**: Each major version adds **layers**, not rewrites. v1.x JSON is forward-compatible with v2.x readers.

---

## v1.0.0 — Production Foundation (SHIPPED ✅)

**Released:** 2025-11-17
**Status:** Stable, in production use

### What Shipped

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

## v1.1.0 — Speaker & Schema Lock-In (Q1 2026)

**Theme:** Lock in the **canonical JSON schema v2** and add **speaker diarization**.

**Goal:** Make slower-whisper the standard format for conversation data.

### Core Features

#### 1. Speaker Diarization (Layer 2)

**Implementation:**

- Integrate WhisperX-style diarization (Whisper + pyannote.audio)
- Map diarization segments to Whisper segments via time overlap
- Populate `segment.speaker = {id, confidence, alternatives}`
- Build global `speakers[]` table with cluster IDs and confidence

**Schema additions:**

```json
"speakers": [
  {
    "id": "spk_0",
    "label": "Speaker A",
    "role_hint": null,  // "agent", "customer", "host", etc.
    "cluster_confidence": 0.93,
    "embedding_id": null  // future: cross-session identity
  }
],
"segments": [
  {
    "speaker": {
      "id": "spk_1",
      "confidence": 0.86,
      "alternatives": [{"id": "spk_0", "confidence": 0.10}],
      "source": "pyannote-3.1"
    }
  }
]
```

**Testing:**

- BDD scenario: 2-speaker synthetic audio → correct speaker counts
- Confusion matrix on real labeled data (AMI Meeting Corpus subset)
- LLM-as-judge sanity check (does labeling appear consistent?)

#### 2. Turn Structure (Layer 2)

**Implementation:**

- Build `turns[]` by grouping contiguous segments from same speaker
- Add turn-level metadata:
  - `question_count`
  - `interruption_started_here`
  - `disfluency_ratio`

**Schema additions:**

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
      "question_count": 1,
      "interruption_started_here": false
    }
  }
]
```

#### 3. Speaker Stats (Layer 2)

Per-speaker aggregates for LLM prompts:

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
    "sentiment_summary": {
      "positive": 0.3,
      "neutral": 0.5,
      "negative": 0.2
    }
  }
]
```

#### 4. Schema v2 Finalization

**Lock in:**

- Core fields will NOT change meaning within v2.x
- `audio`, `meta`, `speakers`, `segments`, `turns`, `speaker_stats`
- Deprecation policy documented
- Migration tooling for future v3

**Documentation:**

- Full schema reference with field-by-field contracts
- JSON Schema (draft-07) file for validation
- Examples for every optional field

#### 5. Evaluation Harness

**Build testbed:**

- 50-200 labeled segments (human-annotated or LLM-teacher)
- Tasks:
  - Speaker-aware summarization
  - Responsibility/action item extraction per speaker
  - Conflict/objection detection
- Conditions:
  - Baseline (text only)
  - Text + speaker labels
  - Text + speaker + stats
- Metrics: F1, accuracy, or LLM-as-judge scoring

### Developer Experience

- Progress bars for batch operations
- Better error messages with recovery suggestions
- Dry-run mode for testing configurations
- Verbose logging mode for debugging

### Deliverables

- [ ] Speaker diarization module (`transcription/diarization.py`)
- [ ] Turn builder (`transcription/turns.py`)
- [ ] Speaker stats aggregator
- [ ] Schema v2 JSON Schema file
- [ ] Evaluation harness (`benchmarks/eval_speakers.py`)
- [ ] BDD scenarios for speaker correctness
- [ ] Documentation: Speaker diarization guide

---

## v1.2.0 — Chunking & LLM Integration (Q2 2026)

**Theme:** Make slower-whisper JSON **trivial** to use with LLMs.

### Core Features

#### 1. Intelligent Chunking

**Implementation:**

- Define `chunks[]` that reference `segment_ids[]`
- Chunk boundaries:
  - Target N tokens (512-1024) OR N seconds (60-120)
  - Prefer cut points at turn boundaries, then large pauses
  - Never split mid-turn unless unavoidable

**Schema additions:**

```json
"chunks": [
  {
    "id": "chunk_0",
    "start": 0.0,
    "end": 120.0,
    "segment_ids": [0, 1, 2, 3, 4, 5],
    "summary": null,  // filled by Stage 3
    "semantic_tags": []  // e.g., ["intro", "rapport-building"]
  }
]
```

#### 2. Prompt Builders

**Python utilities:**

- `to_turn_view(transcript)` → turn-annotated text for prompts
- `to_chunk_prompt(chunk, transcript)` → chunk with optional stats
- `to_speaker_summary(transcript)` → speaker table + talk-time ratios

**Example:**

```python
from transcription import load_transcript, to_turn_view

transcript = load_transcript("meeting.json")
prompt_text = to_turn_view(transcript, include_audio_state=True)

# Result:
# Speakers:
# - Speaker A (spk_0), role: agent
# - Speaker B (spk_1), role: customer
#
# Conversation (turn-level):
# [00:15.2–00:22.8] Speaker B: I'm not sure this pricing works...
#   [audio: concerned tone, rising pitch]
# [00:23.0–00:35.1] Speaker A: Totally understand. Let me walk through...
```

#### 3. LangChain / LlamaIndex Adapters

**Create official integrations:**

- `LangChain SlowerWhisperLoader` — document loader with chunks
- `LlamaIndex SlowerWhisperReader` — index builder with metadata
- Both expose `audio_state` as searchable metadata

**Example:**

```python
from langchain.document_loaders import SlowerWhisperLoader

loader = SlowerWhisperLoader("meeting.json")
docs = loader.load()

for doc in docs:
    print(doc.page_content)  # turn text
    print(doc.metadata)      # speaker, audio_state, timestamps
```

#### 4. Export Formats

**Additional outputs:**

- WebVTT (web video subtitles)
- CSV (tabular: id, start, end, speaker, text, valence, arousal)
- Annotated HTML (web viewing with highlights)
- Praat TextGrid (for phonetics research)

### Deliverables

- [ ] Chunker implementation (`transcription/chunking.py`)
- [ ] Prompt builder utilities (`transcription/llm_utils.py`)
- [ ] LangChain adapter (`integrations/langchain_loader.py`)
- [ ] LlamaIndex adapter (`integrations/llamaindex_reader.py`)
- [ ] Export format writers (WebVTT, CSV, HTML)
- [ ] Examples: LLM summarization, QA, RAG
- [ ] Documentation: LLM integration guide

---

## v1.3.0 — Semantic SLM Integration (Q3 2026)

**Theme:** Add optional **local semantic enrichment** (Layer 3) using small multimodal models.

### Core Features

#### 1. SLM Integration Framework

**Abstraction:**

- `SemanticAnnotator` interface:
  - `run(transcript, chunk) -> annotations`
- Backends:
  - Qwen2.5-VL (1-7B)
  - SmolLM (135M-1.7B)
  - Custom text-only models

**Configuration:**

```bash
# Enable semantic enrichment with specific model
uv run slower-whisper enrich \
  --enable-semantic \
  --slm-model qwen2.5-vl-7b \
  --slm-device cuda
```

#### 2. Chunk-Level Annotations

**For each chunk:**

- Generate summary (1-2 sentences)
- Tag decisions, objections, "topic segments"
- Classify interaction type (small talk / content / negotiation / meta)
- Detect sarcasm, irony, uncertainty

**Schema additions:**

```json
"chunks": [
  {
    "summary": "Customer expresses pricing concerns; agent offers alternatives.",
    "semantic_tags": ["objection", "pricing_discussion"],
    "annotations": {
      "llm": [
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

#### 3. Segment-Level Semantic Tags

**Per segment:**

- `is_decision`
- `is_action_item`
- `is_sarcastic`
- `certainty_score`

**Schema additions:**

```json
"segments": [
  {
    "annotations": {
      "llm": [
        {
          "type": "intent",
          "label": "decision",
          "confidence": 0.92
        },
        {
          "type": "certainty",
          "score": 0.35,
          "label": "uncertain"
        }
      ]
    }
  }
]
```

#### 4. Caching & Performance

- Cache semantic passes separately by `(audio_hash, asr_hash, enrichment_hash, slm_model_hash)`
- Process in chunks (60-120s), not per-token
- Optional: distill SLM outputs into small local classifiers for speed

### Deliverables

- [ ] Semantic annotator interface (`transcription/semantic.py`)
- [ ] Qwen2.5-VL backend implementation
- [ ] Caching for semantic passes
- [ ] Evaluation: text vs text+audio_state vs text+audio_state+SLM
- [ ] BDD scenarios for semantic annotation presence
- [ ] Documentation: Semantic enrichment guide

---

## v2.0.0 — Streaming & Extensibility (Q4 2026)

**Theme:** Real-time processing + plugin architecture.

### Core Features

#### 1. Streaming Transcription

- Incremental Whisper on sliding windows
- Websocket API for live results
- Low-latency mode (<500ms delay)

#### 2. Plugin System

- Custom feature extractors
- Custom output formatters
- Pipeline composition DSL

#### 3. Distributed Processing

- Multi-node job queueing
- S3/GCS/Azure storage backends
- Kubernetes operator for auto-scaling

### Deliverables

- [ ] Streaming ASR module
- [ ] Plugin API documentation
- [ ] Cloud storage backends
- [ ] K8s operator (CRDs, controllers)

---

## v3.0.0 — Intelligence Layer (2027+)

**Theme:** Semantic understanding + domain specialization.

### Core Features

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

### High Priority (v1.1)

- [ ] Speaker diarization implementation
- [ ] Turn structure builder
- [ ] Evaluation harness on AMI/IEMOCAP
- [ ] JSON Schema (draft-07) validation file

### Medium Priority (v1.2)

- [ ] LangChain adapter
- [ ] LlamaIndex adapter
- [ ] WebVTT / CSV / HTML exporters
- [ ] Praat TextGrid export

### Research Contributions (v1.3+)

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

**Current Deprecations:** None (v1.0.0 is stable)

**Future Deprecation Timeline:**

- Deprecation announced: 6 months before removal
- Warning period: Deprecation warnings in code
- Removal: Only in major version bumps

**Example:**

```
v1.8.0: Deprecate legacy CLI (announce only)
v1.9.0: Legacy CLI works but logs warnings
v2.0.0: Legacy CLI removed (unified CLI only)
```

---

## Long-Term Vision (5+ years)

See [VISION.md](VISION.md) for complete strategic vision.

**Mission:** Make acoustic information accessible to text-based AI systems, enabling truly multimodal understanding of human communication.

**Goals:**

1. **Universal Acoustic Encoding** — Standard format for representing audio-only information
2. **Research Accelerator** — Tool of choice for speech, linguistics, and psychology researchers
3. **Production Grade** — Enterprise-ready for commercial transcription and analytics
4. **Open Science** — Advance open-source speech AI and contribute to academic research
5. **Accessibility** — Enable better tools for hearing-impaired, language learners, and assistive technology

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
