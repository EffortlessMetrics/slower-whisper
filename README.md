# slower-whisper

## Audio → Receipts

Local-first transcripts with speakers, timestamps, enrichment, and a stable JSON contract for LLM pipelines.

![Python](https://img.shields.io/badge/python-3.11--3.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-production%20ready-success)

---

### What this is

**slower-whisper is ETL for conversations.** It transforms raw audio into schema-versioned structured data that captures not just *what* was said, but *who* said it, *when*, and *how*.

Text-only LLMs can't hear tone, pacing, or hesitation. Omnimodal LLMs burn capacity doing ASR/diarization implicitly. This system pre-computes the acoustic ground truth—timestamps, speaker turns, prosody, emotion—so your LLM can focus on reasoning instead of signal processing.

### Why this exists

**FinOps for LLMs.** Deterministic triage runs first: rule-based keyword extraction, DSP-derived prosody, speaker math. Only the interesting segments need expensive model inference. The cheap layer filters; the expensive layer reasons.

**Truth layer.** LLMs hallucinate tone and pacing. They can't reliably separate speakers or produce stable timestamps. slower-whisper does that with deterministic math (IoU speaker assignment, Praat pitch extraction, librosa energy) and hands the LLM hard facts instead of guesses.

**Local-first.** All processing runs on your hardware. Data never leaves. Model weights are fetched once on first use.

### Key properties

- **Schema-versioned JSON** (v2) with stability tiers and backward compatibility
- **Modular dependencies** — install only what you need (2.5GB base → 8GB+ full)
- **5 semantic adapters** — local keywords, local LLM, OpenAI, Anthropic, or bring your own
- **Streaming built-in** — WebSocket + SSE for real-time pipelines
- **Receipt provenance** — config hash, run IDs, git commit for reproducibility

---

## 5-Minute Quickstart

```bash
# Clone and enter dev shell
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper
nix develop           # or: uv sync --extra full --extra dev

# Transcribe (place audio in raw_audio/, outputs to whisper_json/)
uv run slower-whisper transcribe --root .
```

### Drop-in faster-whisper Replacement

Already using `faster-whisper`? Just change the import:

```python
# Before
from faster_whisper import WhisperModel

# After
from slower_whisper import WhisperModel

model = WhisperModel("base")
segments, info = model.transcribe("audio.wav")

# Same API, plus optional diarization and enrichment
segments, info = model.transcribe("meeting.wav", diarize=True, enrich=True)
transcript = model.last_transcript  # Access enriched data
```

See [Migrating from faster-whisper](docs/FASTER_WHISPER_MIGRATION.md) for details.

---

## Local Gate (Canonical)

CI may be rate-limited; **local gate is canonical**. CI is additive.

```bash
./scripts/ci-local.sh        # full gate
./scripts/ci-local.sh fast   # quick check
```

**Note:** Inside devshell, use `nix-clean` wrapper for nix commands (avoids ABI issues).

**PR requirement:** Paste gate receipts before merging.

---

## Installation

### Nix (Recommended)

```bash
nix develop
uv sync --extra full --extra dev
```

### UV (Fallback)

```bash
# Install system deps (ffmpeg, libsndfile) via apt/brew/choco
uv sync --extra full --extra dev   # contributors
uv sync --extra full               # runtime only
uv sync                            # transcription only (~2.5GB)
```

---

## Output Format

```json
{
  "schema_version": 2,
  "file": "meeting.wav",
  "segments": [{"id": 0, "start": 0.0, "end": 4.2, "text": "Hello."}]
}
```

See [docs/SCHEMA.md](docs/SCHEMA.md) for the complete schema specification and stability contract.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/INDEX.md](docs/INDEX.md) | **Start here** — complete documentation map |
| [docs/FASTER_WHISPER_MIGRATION.md](docs/FASTER_WHISPER_MIGRATION.md) | **Migrating from faster-whisper** |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | First transcription tutorial |
| [docs/SCHEMA.md](docs/SCHEMA.md) | JSON schema reference |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Configuration reference |
| [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | CLI command reference |
| [docs/API_QUICK_REFERENCE.md](docs/API_QUICK_REFERENCE.md) | Python API reference |
| [docs/GPU_SETUP.md](docs/GPU_SETUP.md) | GPU configuration |

| Project File | Description |
|--------------|-------------|
| [ROADMAP.md](ROADMAP.md) | Development roadmap |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) file.

**Privacy:** All processing runs locally. Only model weights are fetched on first use.
