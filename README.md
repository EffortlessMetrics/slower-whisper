# slower-whisper

**Local-first conversation signal engine for LLMs**

![Version](https://img.shields.io/badge/version-1.1.0--dev-blue)
![Tests](https://img.shields.io/badge/tests-267%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-57%25-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-production%20ready-success)

## What is slower-whisper?

slower-whisper transforms audio conversations into **LLM-ready structured data** that captures not just what was said, but **how it was said**.

Unlike traditional transcription tools that output plain text, slower-whisper produces a rich, versioned JSON format with:

- **Timestamped segments** (word-level alignment planned)
- **Speaker diarization** (v1.1 experimental - who spoke when, speaker attribution per segment)
- **Prosodic features** (pitch, energy, speaking rate, pauses)
- **Emotional state** (valence, arousal, categorical emotions)
- **Turn structure** and interaction patterns
- **LLM-friendly text renderings** of acoustic features

**The result**: A text-only LLM can now "hear" key aspects of audio — tone, emphasis, hesitation, excitement — that aren't captured in transcription alone.

## Why slower-whisper?

### The Problem

Standard transcription gives you words, but misses:

- **How** someone said something (sarcasm, uncertainty, stress)
- **Acoustic cues** that change meaning ("I'm fine" said flatly vs. enthusiastically)
- **Interaction patterns** (interruptions, pauses, who dominated the conversation)

Cloud "conversation intelligence" APIs solve this but are:

- **Closed-source** and opaque
- **Cloud-only** (privacy and latency concerns)
- **Text-centric** (limited acoustic feature access)

### The Solution

slower-whisper is a **local-first, open-source conversation signal engine** that:

✅ **Runs entirely locally** (NVIDIA GPU recommended, CPU fallback supported)
✅ **Produces stable, versioned JSON** you can build on
✅ **Modular architecture** — use only the features you need
✅ **Contract-driven** — BDD scenarios guarantee behavioral stability
✅ **LLM-native** — designed for RAG, summarization, analysis, and prompt engineering

## Quick Start

**Recommended:** Run the setup script to detect your environment and get guided setup:

```bash
# Clone the repository
git clone https://github.com/yourusername/slower-whisper.git
cd slower-whisper

# Run guided setup (detects Nix, shows appropriate instructions)
bash scripts/setup-env.sh
```

The script will:
- ✅ Detect if Nix is installed (recommended)
- ✅ Guide you through the best setup for your system
- ⚠️ Warn if using fallback (traditional) setup
- ✅ Check for required dependencies

**Quick path with Nix (recommended):**

```bash
# 1. Install Nix (one-time setup)
sh <(curl -L https://nixos.org/nix/install) --daemon

# 2. Enter dev shell and install dependencies
nix develop
uv sync --extra full --extra diarization --extra dev

# 3. Run transcription
uv run slower-whisper
```

**Fallback path (if Nix unavailable):**

```bash
# 1. Install system deps (ffmpeg, libsndfile) via apt/brew/choco
# 2. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# 3. Install Python deps: uv sync --extra full
# 4. Run: uv run slower-whisper
```

Place your audio files in `raw_audio/` and find transcripts in `whisper_json/`, `transcripts/`.

See detailed instructions below for setup, configuration, and advanced features.

## Architecture Overview

slower-whisper uses a **layered enrichment pipeline** where each layer adds progressively richer conversational context:

### Layer 0 – Ingestion
- Audio normalization (ffmpeg: 16 kHz mono WAV)
- Format detection and chunking
- Audio hashing for caching

### Layer 1 – ASR (Whisper)
- Fast, deterministic transcription via faster-whisper
- Timestamped segments with confidence scores
- Word-level alignment (optional, via WhisperX integration planned)
- **Fully local, GPU-accelerated**

### Layer 2 – Acoustic & Structural Enrichment (local, modular)
Optional enrichment passes that never re-run ASR:

**Speaker Diarization** (v1.1 - Experimental)
- Who spoke when, per segment + global speakers table
- Normalized canonical speaker IDs (`spk_0`, `spk_1`, ...)
- Turn structure grouping contiguous segments by speaker
- Foundation for speaker-relative feature normalization (v1.2)
- **Status:** Functional but experimental; requires HuggingFace token and pyannote.audio dependency

**Prosody Extraction** (current)
- Pitch (mean, range, contour)
- Energy (loudness)
- Speaking rate (syllables/sec)
- Pause statistics

**Emotion Recognition** (current)
- Dimensional: valence (positive/negative), arousal (calm/excited)
- Categorical: happy, sad, angry, frustrated, etc.

**Turn & Interaction Structure** (v1.2 planned)
- Speaker statistics (talk time, turn counts)
- Overlap/interruption detection
- Question/answer linking
- Backchanneling and turn-taking analysis

### Layer 3 – Semantic Enrichment (optional, SLM/MM)
Small local multimodal models for higher-level insights:

- Chunk-level summaries
- Topic segmentation
- Intent classification (decision, objection, risk)
- Sarcasm/irony detection

**Design principle**: L3 is **opt-in**, chunked (60-120s), and never blocks the core pipeline.

### Layer 4 – Task-Specific Outputs
Use the enriched JSON for downstream tasks:

- Meeting notes and action item extraction
- Coaching feedback and QA scoring
- Sentiment trajectory analysis
- RAG/vector search with acoustic context

**Key guarantees:**
- 🔒 **Cacheable & resumable** at every layer
- 🔒 **Versioned JSON schema** with stability contracts
- 🔒 **BDD scenarios** enforce behavioral invariants
- 🔒 **Local-first** — no data leaves your machine

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete design details.

## Requirements

- Windows, Linux, or macOS (any OS with Nix or Python + ffmpeg)
- Python 3.10+ (managed by Nix or system package manager)
- NVIDIA GPU recommended for GPU acceleration (CPU fallback supported)
- **Recommended:** [Nix](https://nixos.org/) for reproducible environments
- **Fallback:** Manual ffmpeg + [uv](https://docs.astral.sh/uv/) installation

## Installation

### Option 1: Nix (Recommended)

**Best for:** Contributors, teams, anyone wanting reproducible environments and local CI.

Nix provides **guaranteed reproducibility** across machines and mirrors your CI environment locally.

```bash
# One-time setup: Install Nix
sh <(curl -L https://nixos.org/nix/install) --daemon

# Enable flakes (required)
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

# Enter dev shell (provides ffmpeg, Python, and all system deps)
nix develop

# Install Python packages
uv sync --extra full --extra diarization --extra dev

# Run local CI checks (same as GitHub Actions)
nix flake check
```

**Benefits:**
- ✅ **Same environment everywhere** - WSL, NixOS, macOS, CI runners
- ✅ **Local CI** - `nix flake check` runs identical tests to GitHub Actions
- ✅ **No system dependency conflicts** - isolated, reproducible builds
- ✅ **Optional direnv integration** - auto-activate on `cd`

See [docs/DEV_ENV_NIX.md](docs/DEV_ENV_NIX.md) for detailed setup, troubleshooting, and direnv integration.

---

### Option 2: Traditional Setup (Fallback)

> ⚠️ **Notice:** This method works but lacks reproducibility guarantees. You may encounter environment-specific issues that don't occur in CI or on other machines. **Consider using Nix (Option 1) for a better development experience.**

**Best for:** Quick testing, environments where Nix can't be installed, one-off usage.

#### Install System Dependencies

**Install ffmpeg:**

- **Linux (Ubuntu/Debian):**

  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
  ```

- **macOS:**

  ```bash
  brew install ffmpeg
  ```

- **Windows (PowerShell, elevated):**

  ```powershell
  choco install ffmpeg -y
  ```

**Install uv (Python package manager):**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or visit [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

#### Install Python Dependencies (Traditional Setup)

> 📝 **Note:** If you're using **Nix (Option 1)**, skip this section and use the commands shown in Option 1 above.

This project uses `pyproject.toml` with dependency groups for flexible installation. Choose the installation that matches your needs:

**Base install (Stage 1: Transcription only, ~2.5GB):**

Just faster-whisper for basic transcription. Lightweight and fast.

```bash
uv sync
```

**Full install (Stage 2: All audio enrichment features, ~4GB additional):**

Includes prosody extraction and emotion recognition capabilities.

```bash
uv sync --extra full
```

**Development install (for contributors):**

Includes all features plus testing, linting, and documentation tools.

```bash
uv sync --extra dev
```

### Understanding Dependency Groups

The project offers modular dependencies based on what you need:

| Group | Size | What's Included | When to Use |
|-------|------|----------------|-------------|
| **base** (default) | ~2.5GB | faster-whisper only | Just need transcription |
| **diarization** | +1.5GB | pyannote.audio for speaker diarization (v1.1 experimental) | Speaker attribution and turn structure |
| **enrich-basic** | +1GB | soundfile, numpy, librosa | Basic prosody features |
| **enrich-prosody** | +36MB | Praat/Parselmouth | Research-grade pitch analysis |
| **emotion** | +4GB | torch, transformers | Emotion recognition |
| **full** | +4GB total | All enrichment features (not including diarization) | Complete audio analysis |
| **dev** | +full + tools | Testing, linting, docs | Contributing to project |

**Install specific groups:**

```bash
# Just prosody (no emotion)
uv sync --extra enrich-prosody

# Just emotion (no prosody)
uv sync --extra emotion

# Speaker diarization (v1.1 experimental)
uv sync --extra diarization

# Basic enrichment only
uv sync --extra enrich-basic
```

### Alternative: pip Installation

If you prefer traditional pip:

```bash
# Base install
pip install -e .

# Full install
pip install -e ".[full]"

# Development install
pip install -e ".[dev]"
```

## Directory layout

By default, the pipeline expects the following structure under a root folder:

```text
root/
  transcribe_pipeline.py
  transcription/
    ...
  raw_audio/
  input_audio/
  transcripts/
  whisper_json/
```

The code will create the directories if they do not exist.

- `raw_audio/` – place your original audio here (`.mp3`, `.m4a`, `.wav`, etc.).
- `input_audio/` – normalized 16 kHz mono WAVs (generated).
- `transcripts/` – `.txt` and `.srt` outputs (generated).
- `whisper_json/` – `.json` structured transcripts (generated).

## Configuration

slower-whisper supports multiple configuration methods with clear precedence rules.

### Configuration Precedence

Settings are loaded in the following order (highest to lowest priority):

```text
1. CLI flags (--model, --device, etc.)
   ↓
2. Config file (--config or --enrich-config)
   ↓
3. Environment variables (SLOWER_WHISPER_*)
   ↓
4. Defaults
```

Each layer only overrides values explicitly set. This allows flexible configuration for different environments.

### Configuration Methods

#### 1. CLI Flags (Highest Priority)

```bash
uv run slower-whisper transcribe --model large-v3 --language en --device cuda
```

#### 2. Configuration Files

```bash
# Use a JSON config file
uv run slower-whisper transcribe --config config/production.json

# Override specific values from config
uv run slower-whisper transcribe --config config/base.json --model large-v3
```

#### 3. Environment Variables

```bash
# Transcription settings (SLOWER_WHISPER_ prefix)
export SLOWER_WHISPER_MODEL=large-v3
export SLOWER_WHISPER_DEVICE=cuda
export SLOWER_WHISPER_LANGUAGE=en

# Enrichment settings (SLOWER_WHISPER_ENRICH_ prefix)
export SLOWER_WHISPER_ENRICH_ENABLE_PROSODY=true
export SLOWER_WHISPER_ENRICH_DEVICE=cuda

uv run slower-whisper transcribe  # Uses env vars
```

#### 4. Python API

```python
from transcription import TranscriptionConfig, EnrichmentConfig

# Load from file
config = TranscriptionConfig.from_file("config.json")

# Load from environment
config = TranscriptionConfig.from_env()

# Create directly
config = TranscriptionConfig(model="large-v3", language="en")
```

### Example Configurations

See [examples/config_examples/](examples/config_examples/) for complete configuration examples:

- **transcription_basic.json**: Lightweight base model setup
- **transcription_production.json**: High-quality production settings
- **transcription_dev_testing.json**: Fast testing with minimal resources
- **enrichment_full.json**: Complete audio analysis (prosody + emotion)
- **enrichment_production.json**: Optimized production enrichment

For detailed configuration documentation, see:

- [examples/config_examples/README.md](examples/config_examples/README.md) - Configuration file examples
- [API_QUICK_REFERENCE.md](docs/API_QUICK_REFERENCE.md) - Complete API reference with all config options

## Usage

The project provides both **Command-Line** and **Python API** interfaces.

### Usage Profiles

Choose your installation and feature set based on your needs:

| Profile | Install Command | Enabled Features | Typical Flags | Model Cache Size |
|---------|----------------|------------------|---------------|------------------|
| **Transcription only** | `uv sync` | ASR (faster-whisper) | `slower-whisper transcribe` | ~2.5GB |
| **Transcription + prosody/emotion** | `uv sync --extra full` | ASR + acoustic features | `--enable-prosody --enable-emotion` | ~6.5GB |
| **Full conversation signals** | `uv sync --extra full --extra diarization` | ASR + acoustic + speaker diarization | `--enable-diarization --min-speakers N --max-speakers M` | ~8GB |

**Notes:**
- All profiles run **entirely locally** (no cloud dependencies at runtime)
- GPU recommended for diarization (pyannote.audio is compute-intensive), CPU fallback supported
- Diarization requires HuggingFace token: `export HF_TOKEN=hf_...`
- See [Model Cache Management](#model-cache-management) for cache location and cleanup

### 5-Minute Quickstart (Transcription Only)

Get your first transcripts in 5 minutes:

```bash
# 1. Clone and install
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper
uv sync

# 2. Add your audio file
mkdir -p raw_audio
cp /path/to/your_audio.wav raw_audio/

# 3. Transcribe!
uv run slower-whisper transcribe

# 4. View results
ls whisper_json/    # Structured JSON with segments
ls transcripts/     # Human-readable TXT and SRT
```

**What you get:**
- `whisper_json/your_audio.json` - Rich structured data with timestamps, confidence scores
- `transcripts/your_audio.txt` - Clean text output
- `transcripts/your_audio.srt` - Subtitle file

**Next steps:**
- Analyze with LLMs: See [LLM Integration](#llm-integration-analyze-conversations) below
- Add prosody/emotion: `uv sync --extra full && slower-whisper enrich`
- Add speaker diarization: See [docs/SPEAKER_DIARIZATION.md](docs/SPEAKER_DIARIZATION.md)

### Unified CLI (Recommended)

The modern CLI uses subcommands for clarity:

#### Stage 1: Transcribe

```bash
# Basic usage with defaults
uv run slower-whisper transcribe

# Customize options
uv run slower-whisper transcribe \
  --root /path/to/project \
  --model large-v3 \
  --language en \
  --device cuda \
  --skip-existing-json
```

#### Stage 2: Enrich

```bash
# Add prosody and emotion features
uv run slower-whisper enrich

# Customize enrichment
uv run slower-whisper enrich \
  --root /path/to/project \
  --enable-prosody \
  --enable-emotion \
  --device cpu
```

#### Speaker Diarization (v1.1 - Experimental)

**Status:** Functional but experimental. Use for testing and feedback; may change in v1.2.

```bash
# Install diarization dependencies (one-time setup)
uv sync --extra diarization

# Set HuggingFace token (after accepting pyannote model license)
export HF_TOKEN=hf_...  # Get from https://huggingface.co/settings/tokens

# Run transcription with diarization enabled
# Note: Diarization auto-selects GPU if available, otherwise CPU
uv run slower-whisper transcribe \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 4

# Or programmatically:
from transcription import transcribe_directory, TranscriptionConfig, DiarizationConfig

config = TranscriptionConfig(
    model="large-v3",
    diarization=DiarizationConfig(
        enabled=True,
        min_speakers=2,
        max_speakers=4
    )
)
transcripts = transcribe_directory("/path/to/project", config)
```

**What you get:**

- `segment.speaker = {"id": "spk_0", "confidence": 0.87}` - Per-segment speaker assignment
- `speakers[]` table - Global speaker metadata (first/last seen, total speech time)
- `turns[]` structure - Contiguous segments grouped by speaker
- `meta.diarization` - Status, backend, and error information

**Requirements:**

- HuggingFace account (free) with accepted pyannote.audio model license
- `HF_TOKEN` environment variable or `~/.cache/huggingface/token`
- GPU recommended (pyannote.audio is resource-intensive)

**Known Limitations (v1.1 experimental):**

- Optimized for 2-4 speakers; performance degrades with >4 speakers
- No support for overlapping speech detection (planned v1.2)
- Speaker names/labels not supported yet (identifiers only)
- See [`docs/SPEAKER_DIARIZATION.md`](docs/SPEAKER_DIARIZATION.md) for detailed design and known issues

#### Model Cache Management

slower-whisper caches large model files (Whisper weights, pyannote diarization models, emotion recognition models) to avoid redownloading on each run.

**Default cache location:**

```text
~/.cache/slower-whisper/
  ├── hf/          # HuggingFace hub cache (HF_HOME)
  ├── torch/       # PyTorch cache (TORCH_HOME)
  ├── whisper/     # Whisper model weights
  ├── emotion/     # Emotion recognition models
  └── diarization/ # Pyannote diarization models
```

**Inspect cache:**

```bash
uv run slower-whisper cache --show
```

**Clear cache:**

```bash
# Clear all caches
uv run slower-whisper cache --clear all

# Clear specific cache
uv run slower-whisper cache --clear whisper
uv run slower-whisper cache --clear emotion
uv run slower-whisper cache --clear diarization
```

**Custom cache location:**

```bash
# Set before running (or add to shell profile)
export SLOWER_WHISPER_CACHE_ROOT=/data/models/slower-whisper
uv run slower-whisper transcribe
```

**Important:** Importing the `transcription` package automatically configures cache paths and sets `HF_HOME`, `TORCH_HOME`, and `HF_HUB_CACHE` environment variables if not already set. To override these defaults, set the environment variables **before** importing or running slower-whisper.

See [`docs/MODEL_CACHE.md`](docs/MODEL_CACHE.md) for detailed cache management documentation, including Docker/CI integration.

#### View help

```bash
uv run slower-whisper --help
uv run slower-whisper transcribe --help
uv run slower-whisper enrich --help
uv run slower-whisper cache --help
```

### Python API

Use the clean programmatic interface:

#### Basic transcription

```python
from transcription import transcribe_directory, TranscriptionConfig

config = TranscriptionConfig(
    model="large-v3",
    language="en",
    device="cuda"
)

transcripts = transcribe_directory("/path/to/project", config)
print(f"Transcribed {len(transcripts)} files")
```

#### Single file transcription

```python
from transcription import transcribe_file, TranscriptionConfig

config = TranscriptionConfig(model="base", language="en")
transcript = transcribe_file(
    audio_path="interview.mp3",
    root="/path/to/project",
    config=config
)

# Access results
for segment in transcript.segments:
    print(f"[{segment.start:.2f}s] {segment.text}")
```

#### Audio enrichment

```python
from transcription import enrich_directory, EnrichmentConfig

config = EnrichmentConfig(
    enable_prosody=True,
    enable_emotion=True,
    device="cpu"
)

enriched = enrich_directory("/path/to/project", config)

# Inspect enriched features
for transcript in enriched:
    for segment in transcript.segments:
        if segment.audio_state:
            print(segment.audio_state["rendering"])
            # e.g., "[audio: high pitch, loud volume, fast speech]"
```

#### Load and save transcripts

```python
from transcription import load_transcript, save_transcript

# Load existing transcript
transcript = load_transcript("transcript.json")

# Modify and save
transcript.segments[0].text = "Corrected text"
save_transcript(transcript, "corrected.json")
```

#### LLM Integration (Analyze Conversations)

slower-whisper includes utilities to render transcripts as LLM-ready text with speaker labels and audio cues:

```python
from transcription import load_transcript, render_conversation_for_llm

# Load transcript with diarization
transcript = load_transcript("whisper_json/support_call.json")

# Render for LLM consumption with speaker labels
context = render_conversation_for_llm(
    transcript,
    mode="turns",  # or "segments"
    include_audio_cues=True,
    speaker_labels={"spk_0": "Agent", "spk_1": "Customer"}
)

# Send to LLM (Claude, GPT, etc.)
# context contains: "[Agent | calm tone] Hello, how can I help you?"
#                   "[Customer | frustrated tone, high pitch] I can't log in..."
```

**Resources:**
- **[`docs/LLM_PROMPT_PATTERNS.md`](docs/LLM_PROMPT_PATTERNS.md)** - Comprehensive guide to LLM prompting with slower-whisper data
- **[`examples/llm_integration/`](examples/llm_integration/)** - Working examples (summarization, coaching, QA scoring)
- **API functions:** `render_conversation_for_llm()`, `render_conversation_compact()`, `render_segment()`

**Common use cases:**
- Meeting summarization with speaker attribution
- Call quality analysis and coaching feedback
- Sentiment tracking per speaker
- Action item extraction with timestamps

See the examples for complete, runnable scripts.

### REST API Service

For web-based deployments, slower-whisper includes an optional FastAPI service wrapper that exposes transcription and enrichment via HTTP endpoints.

**Installation:**

```bash
# Install API dependencies
uv sync --extra api --extra full
```

**Running the service:**

```bash
# Development mode (with auto-reload)
uv run uvicorn transcription.service:app --reload --host 0.0.0.0 --port 8000

# Production mode (4 workers)
uv run uvicorn transcription.service:app --host 0.0.0.0 --port 8000 --workers 4

# Or with Docker
docker build -f Dockerfile.api -t slower-whisper:api .
docker run -p 8000:8000 slower-whisper:api

# Or with Docker Compose
docker-compose -f docker-compose.api.yml up -d
```

**Using the API:**

```bash
# Health check
curl http://localhost:8000/health

# Transcribe audio
curl -X POST -F "audio=@interview.mp3" \
  "http://localhost:8000/transcribe?model=large-v3&language=en"

# Enrich transcript
curl -X POST \
  -F "transcript=@transcript.json" \
  -F "audio=@audio.wav" \
  "http://localhost:8000/enrich?enable_prosody=true&enable_emotion=true"

# Interactive documentation
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

For complete API documentation, examples, and deployment guides, see [API_SERVICE.md](docs/API_SERVICE.md).

### Legacy CLI (Backward Compatibility)

The old CLI still works but is deprecated:

```bash
# Old style (still supported)
uv run python transcribe_pipeline.py
uv run python audio_enrich.py

# New unified style (recommended)
uv run slower-whisper transcribe
uv run slower-whisper enrich

# Use a lighter model and quantized weights
uv run slower-whisper --model medium --compute-type int8_float16

# Skip files that already have JSON output
uv run slower-whisper --skip-existing-json
```

The pipeline prints per-file progress, basic timing statistics (per-file and
overall), and a summary at the end.

### Alternative: Without uv

If you installed with pip, you can run directly:

```bash
# If installed as package
slower-whisper

# Or using python directly
python transcribe_pipeline.py
```

## Model download & privacy

On first use of a given model (e.g. `large-v3`), `faster-whisper` will
download the model weights and cache them locally. This requires one-time
internet access to fetch the weights.

**Your audio and transcripts are not uploaded or sent anywhere by this
pipeline.** All transcription runs locally on your machine; only the model
weights are fetched from the internet on first use.

## JSON Schema v2 — Stable, Versioned Contract

slower-whisper produces a **stable, versioned JSON format** designed for programmatic consumption by LLMs and analytics tools.

### Current Schema (v2.0)

```json
{
  "schema_version": 2,
  "audio": {
    "id": "sha256:abc123...",
    "path": "meeting.wav",
    "duration_sec": 3120.5,
    "sample_rate": 16000,
    "channels": 1
  },
  "meta": {
    "pipeline_version": "1.0.0",
    "language": "en",
    "asr_model": "faster-whisper-large-v3",
    "enrichment": {
      "prosody": { /* config */ },
      "emotion": { /* config */ }
    }
  },
  "speakers": [
    {
      "id": "spk_0",
      "label": "Speaker A",
      "role_hint": "agent",
      "cluster_confidence": 0.93
    }
  ],
  "segments": [
    {
      "id": 23,
      "start": 123.45,
      "end": 129.30,
      "text": "I'm not sure this pricing works for us.",
      "words": [
        {"text": "I'm", "start": 123.45, "end": 123.80},
        {"text": "not", "start": 123.81, "end": 124.12}
      ],
      "speaker": {
        "id": "spk_1",
        "confidence": 0.86
      },
      "audio_state": {
        "prosody": {
          "pitch_mean_hz": 195.2,
          "pitch_contour": "rising",
          "energy_rms": 0.24,
          "speech_rate_wps": 3.1,
          "pause_before_ms": 420
        },
        "emotion": {
          "valence": 0.35,
          "arousal": 0.68,
          "label": "concerned",
          "confidence": 0.78
        },
        "interaction": {
          "is_question": true,
          "is_overlap": false
        }
      },
      "annotations": {
        "llm": []  // Reserved for v2.0+ semantic layer; empty in v1.x
      }
    }
  ],
  "turns": [
    {
      "id": "turn_17",
      "speaker_id": "spk_1",
      "segment_ids": [23, 24],
      "start": 123.45,
      "end": 140.10
    }
  ]
}
```

### Schema Guarantees

- ✅ **Forward compatible** — v2 readers accept v1 transcripts
- ✅ **Backward compatible** — optional fields can be null
- ✅ **Stable core fields** — `segments`, `id`, `start`, `end`, `text` won't change meaning within v2.x
- ✅ **Breaking changes require version bump** — v2 → v3 only for structural changes
- ✅ **Audio state versioned independently** — `AUDIO_STATE_VERSION = "1.0.0"`

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#schema-versioning) for the complete stability contract.

## Audio Feature Enrichment

The project supports optional **Stage 2** audio enrichment that extracts linguistic and emotional features
directly from audio.

### Two-Stage Pipeline

**Stage 1: Transcription** (required)

- Normalize audio to 16 kHz mono WAV
- Transcribe using faster-whisper on NVIDIA GPU
- Output: JSON transcripts with segments

**Stage 2: Audio Enrichment** (optional)

- Extract prosodic features (pitch, energy, speech rate, pauses)
- Extract emotional features (dimensional: valence/arousal/dominance; categorical: emotions)
- Populate `audio_state` field in transcript segments

### What Features Are Extracted

**Prosody Features:**

- **Pitch:** mean frequency (Hz), standard deviation, contour (rising/falling/flat)
- **Energy:** RMS level (dB), variation coefficient
- **Speech Rate:** syllables per second, words per second
- **Pauses:** count, longest duration, density per second

**Emotion Features:**

- **Dimensional:** valence (negative to positive), arousal (calm to excited), dominance (submissive to dominant)
- **Categorical:** primary emotion classification (angry, happy, sad, frustrated, etc.) with confidence scores

All features are automatically categorized into meaningful levels (e.g., "high pitch", "fast speech", "neutral sentiment").

### Usage Example

After transcription, enrich with audio features:

```bash
# Enrich existing transcript with emotion analysis
uv run python examples/emotion_integration.py enrich whisper_json/meeting1.json input_audio/meeting1.wav

# View enriched transcript
cat whisper_json/meeting1.json  # Now includes audio_state in segments

# Analyze emotions across the file
uv run python examples/emotion_integration.py analyze whisper_json/meeting1.json
```

Or use the CLI command:

```bash
# Enrich using the unified CLI (processes all transcripts under --root)
uv run slower-whisper enrich --root .
```

Output JSON will have segments like:

```json
{
  "id": 0,
  "start": 0.0,
  "end": 4.2,
  "text": "Okay, let's get started with today's agenda.",
  "audio_state": {
    "prosody": {
      "pitch": {
        "level": "high",
        "mean_hz": 245.3,
        "std_hz": 32.1,
        "contour": "rising"
      },
      "energy": {
        "level": "loud",
        "db_rms": -8.2
      },
      "rate": {
        "level": "normal",
        "syllables_per_sec": 5.3
      }
    },
    "emotion": {
      "dimensional": {
        "valence": {"level": "positive", "score": 0.72},
        "arousal": {"level": "high", "score": 0.68}
      },
      "categorical": {
        "primary": "happy",
        "confidence": 0.89
      }
    }
  }
}
```

### Installation & Setup

The base pipeline requires only `faster-whisper` and audio tools. Audio enrichment is optional:

```bash
# Install base dependencies (Stage 1: transcription only)
uv sync

# Install audio enrichment dependencies (Stage 2: optional)
uv sync --extra full
```

Or install specific enrichment features:

```bash
# Just prosody analysis (lightweight)
uv sync --extra enrich-prosody

# Just emotion recognition (heavier, requires torch)
uv sync --extra emotion
```

See the **Understanding Dependency Groups** section above for details on what each group includes.

For detailed setup instructions, including model downloads for emotion recognition, see [docs/AUDIO_ENRICHMENT.md](docs/AUDIO_ENRICHMENT.md).

## Extending

The code is structured into modules:

- `transcription.models` – core dataclasses (`Segment`, `Transcript`).
- `transcription.config` – configuration classes (`Paths`, `AsrConfig`, `AppConfig`).
- `transcription.audio_io` – ffmpeg-based normalization.
- `transcription.asr_engine` – faster-whisper wrapper.
- `transcription.writers` – JSON/TXT/SRT writers.
- `transcription.pipeline` – orchestration.
- `transcription.cli` – CLI entrypoint.
- `transcription.enrich` – placeholders for tone and speaker enrichment.

To add tone tagging, diarization, or other analysis, write separate modules
(or expand `transcription.enrich`) that read and modify the JSON or
`Transcript` objects without changing the core pipeline.

## Testing

slower-whisper includes a comprehensive test suite with **267 passing tests** (57% coverage) covering unit tests, integration tests, and BDD scenarios.

For detailed quality thresholds and evaluation criteria, see [docs/TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md).

### Test Suite Overview

The project uses a multi-layered testing strategy:

**Unit Tests (pytest)**: Test individual modules and functions in isolation

- JSON schema validation and I/O operations
- Audio processing utilities
- Configuration parsing and precedence
- Data model validation

**Integration Tests**: End-to-end pipeline validation

- Full transcription workflow
- Audio enrichment with prosody and emotion
- API endpoints and service layer
- CLI command execution

**BDD Scenarios (pytest-bdd + Gherkin)**: User-focused acceptance tests written in natural language

- 15 feature scenarios across transcription and enrichment workflows
- See [tests/features/transcription.feature](tests/features/transcription.feature) and [tests/features/enrichment.feature](tests/features/enrichment.feature)
- Example scenario:

  ```gherkin
  Scenario: Transcribe with custom model configuration
    Given a project with a mono WAV file named "test.wav"
    When I transcribe the project with model "base" and language "en"
    Then a transcript JSON exists for "test.wav"
    And the transcript language is "en"
    And the transcript metadata contains model "base"
  ```

### Test Coverage

Current test coverage: **57% overall**, with high coverage on core modules:

- `transcription/writers.py`: **100%** (JSON/TXT/SRT output)
- `transcription/models.py`: **95%** (data models and schema)
- `transcription/config.py`: **90%** (configuration system)
- `transcription/pipeline.py`: **85%** (transcription orchestration)
- `transcription/prosody.py`: **80%** (prosody extraction)

### BDD/Acceptance Tests

Behavioral acceptance tests are defined using Gherkin syntax and pytest-bdd. These tests represent the **behavioral contract** of slower-whisper - they define guaranteed behaviors that must not break without explicit discussion.

**Library BDD Scenarios** (tests/features/):

```bash
# Run library BDD scenarios (transcription and enrichment)
uv run pytest tests/steps/ -v

# Run only transcription scenarios
uv run pytest tests/steps/test_transcription_steps.py -v

# Run only enrichment scenarios
uv run pytest tests/steps/test_enrichment_steps.py -v
```

**Feature files:**

- `tests/features/transcription.feature` - Core transcription behaviors
- `tests/features/enrichment.feature` - Audio enrichment behaviors

**API Service BDD Scenarios** (features/):

```bash
# Run API service BDD scenarios (black-box REST API tests)
uv run pytest features/ -v -m api

# Run only smoke tests (health, docs)
uv run pytest features/ -v -m "api and smoke"

# Run functional tests (transcribe, enrich endpoints)
uv run pytest features/ -v -m "api and functional"
```

**Feature files:**

- `features/api_service.feature` - REST API endpoint behaviors

**Requirements:**

- Library BDD: Requires `ffmpeg` for audio processing
- API BDD: Requires `httpx` and `uvicorn` (auto-installed with `uv sync --extra dev`)

**Behavioral Contract:** These scenarios define the **guaranteed behaviors** of slower-whisper at both the library level and the REST API level. Breaking these scenarios requires explicit discussion and may trigger a version bump.

### Running Tests

```bash
# Install dev dependencies (includes pytest and other testing tools)
uv sync --extra dev

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=transcription --cov-report=term-missing

# Run BDD scenarios only
uv run pytest tests/steps/ -v

# Run specific test categories
uv run pytest -m "not slow"              # Skip slow tests
uv run pytest -m "not requires_gpu"      # Skip GPU-dependent tests
uv run pytest tests/test_prosody.py      # Run specific test file
```

Tests are not required for running the pipeline but are essential for contributors and those extending the codebase.

## Contributing

Contributions are welcome! Whether you're fixing bugs, adding features, improving documentation, or helping with issues, your help makes this project better.

### Quick Start for Contributors

```bash
# 1. Fork and clone
git clone https://github.com/<your-fork>/slower-whisper.git
cd slower-whisper

# 2. Install dev dependencies
uv sync --extra dev

# 3. Run tests to verify setup
uv run pytest -m "not slow"

# 4. Run linting to check code quality
uv run ruff check .
```

### Development Workflow

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Add or update tests for any behavior changes
   - Follow existing code style and patterns
   - Update documentation if needed

3. **Verify contracts before pushing (REQUIRED)**

   ```bash
   # Run quick verification (minimum before pushing)
   uv run slower-whisper-verify --quick

   # Or use make target
   make verify-quick
   ```

   This verifies:
   - ✅ Code quality (ruff linting and formatting)
   - ✅ Unit tests pass
   - ✅ Library BDD scenarios (behavioral contract)
   - ✅ API BDD scenarios (REST API contract)

   **If this fails, do not push.** Fix the issues first.

4. **Commit and push**

   ```bash
   git add .
   git commit -m "feat: brief description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Submit a PR against the `main` branch
   - Describe what changed and why
   - Link any related issues

### Behavioral Contract Guarantee

This project uses **BDD (Behavior-Driven Development) scenarios as contracts**. These scenarios define guaranteed behaviors that cannot break without explicit discussion and versioning:

**Library BDD Contract** (`tests/features/`):

- Transcription behaviors (audio → JSON/TXT/SRT)
- Audio enrichment behaviors (prosody, emotion extraction)

**API BDD Contract** (`features/`):

- REST API endpoint behaviors
- Health checks, documentation, transcribe/enrich endpoints

**Rules:**

- All BDD scenarios must pass before merging
- Breaking a scenario = breaking the contract = requires versioning discussion
- See `docs/BDD_IAC_LOCKDOWN.md` for versioning policy

### Guidelines

- **Use feature branches** and submit Pull Requests against `main`
- **Add tests** for new functionality or bug fixes
- **Run verification CLI** before pushing: `uv run slower-whisper-verify --quick`
- **For larger changes**, open an issue or discussion first to align on direction
- **Follow code style**: We use ruff for linting and formatting (configured in `pyproject.toml`)
- **Write clear commit messages** using conventional commits format when possible
- **If changing BDD scenarios**, document why the behavioral contract is changing

### Code Quality Tools

```bash
# Format code with ruff
uv run ruff format transcription/ tests/

# Lint with ruff (with auto-fix)
uv run ruff check --fix transcription/ tests/

# Type check with mypy
uv run mypy transcription/
```

### Running Tests (For Contributors)

```bash
# Run all fast tests
uv run pytest -m "not slow"

# Run with coverage
uv run pytest --cov=transcription --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_models.py -v
```

For detailed guidelines, coding standards, and the full development workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Deployment

slower-whisper supports multiple deployment options from local CLI to production Kubernetes clusters. Choose the deployment method that fits your scale and infrastructure.

### Infrastructure as Code (IaC)

slower-whisper treats deployment configurations as **first-class contracts**. All IaC artifacts (Dockerfiles, K8s manifests, Compose files) are:

- ✅ Validated before release
- ✅ Tested in CI/CD pipelines
- ✅ Subject to the same config precedence rules (CLI > file > env > defaults)
- ✅ Smoke tested with verification scripts

**Verification CLI:**

```bash
# Quick verification (code + tests + BDD only)
uv run slower-whisper-verify --quick

# Full verification (includes Docker and K8s)
uv run slower-whisper-verify

# Alternative: run as module
python scripts/verify_all.py --quick
```

**IaC Contract:** All deployment artifacts must build/validate successfully and support consistent configuration across local, Docker, and Kubernetes environments. See `docs/BDD_IAC_LOCKDOWN.md` for the complete contract.

### Deployment Options

#### Local CLI (Simplest)

Run directly on your machine with Python and ffmpeg:

```bash
# Install and run
uv sync
uv run slower-whisper transcribe
```

**Best for**: Development, small-scale processing, testing

---

#### Docker - CPU

Containerized deployment without GPU requirements:

```bash
# Build and run
docker build -t slower-whisper:cpu .
docker run --rm -v $(pwd)/data:/app/data slower-whisper:cpu
```

**Best for**: Reproducible environments, CI/CD, CPU-only servers

---

#### Docker - GPU

GPU-accelerated containerized deployment:

```bash
# Build GPU image
docker build -f Dockerfile.gpu -t slower-whisper:gpu .

# Run with GPU access
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  slower-whisper:gpu \
  slower-whisper transcribe --model large-v3 --device cuda
```

**Best for**: High-throughput processing, production workloads, large models

---

#### Docker Compose - Batch Processing

Multi-service orchestration for continuous batch processing:

```bash
# Start batch processor
docker-compose up -d batch-processor

# Monitor logs
docker-compose logs -f batch-processor

# Stop when done
docker-compose down
```

**Configuration**: Edit `docker-compose.yml` or use `.env` file for settings.

**Best for**: Automated batch workflows, scheduled processing, multi-stage pipelines

---

#### Kubernetes - Production Scale

Cloud-native deployment with autoscaling and resource management:

```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n slower-whisper

# Run batch job
kubectl apply -f k8s/job.yaml
```

**Features**:

- Horizontal pod autoscaling
- GPU node scheduling
- Persistent volume claims for data
- ConfigMaps for environment-specific settings
- CronJobs for scheduled processing

**Best for**: Enterprise deployments, cloud infrastructure, high availability, multi-tenant environments

---

#### FastAPI REST Service

HTTP API for web-based integrations:

```bash
# Install API dependencies
uv sync --extra api --extra full

# Run service
uv run uvicorn transcription.service:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -f Dockerfile.api -t slower-whisper:api .
docker run -p 8000:8000 slower-whisper:api
```

**Endpoints**:

- `POST /transcribe` - Transcribe audio files
- `POST /enrich` - Enrich existing transcripts
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

**Best for**: Web applications, microservices architecture, API-first integrations

---

### Deployment Resources

For detailed deployment guides and infrastructure-as-code configurations:

- **[DOCKER.md](docs/DOCKER.md)** - Complete Docker setup guide with GPU support
- **[k8s/](k8s/)** - Kubernetes manifests and Kustomize configurations
- **[docker-compose.yml](docker-compose.yml)** - Production Docker Compose setup
- **[docker-compose.dev.yml](docker-compose.dev.yml)** - Development environment
- **[API_SERVICE.md](docs/API_SERVICE.md)** - REST API service documentation

### Deployment Configuration

All deployment methods support the same configuration options via:

1. **Environment variables** (`SLOWER_WHISPER_*`)
2. **Config files** (JSON, mounted as volumes in Docker/K8s)
3. **CLI flags** (for one-off overrides)

See the [Configuration](#configuration) section above for precedence rules and examples.

## Troubleshooting

### Common Issues

**uv not found after installation:**

Restart your terminal or source your shell profile:

```bash
source ~/.bashrc  # or ~/.zshrc on macOS
```

**CUDA/GPU not detected:**

Ensure you have NVIDIA drivers installed and CUDA toolkit available. The pipeline will fall back to CPU if CUDA is unavailable, but transcription will be slower.

**ffmpeg not found:**

Add ffmpeg to your system PATH. On Windows, you may need to restart PowerShell after installing.

**Model download fails:**

Check your internet connection. Models are downloaded from Hugging Face on first use and cached locally.

## Additional Resources

- **[Installation Guide](INSTALL.md)**: Detailed setup instructions
- **[LLM Prompt Patterns](docs/LLM_PROMPT_PATTERNS.md)**: Reference prompts for conversation analysis with LLMs
- **[Audio Enrichment Guide](docs/AUDIO_ENRICHMENT.md)**: Deep dive into prosody and emotion features
- **[Speaker Diarization Guide](docs/SPEAKER_DIARIZATION.md)**: Design and implementation details for speaker diarization
- **[Roadmap](ROADMAP.md)**: Future features and development plans
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute to the project
- **[Changelog](CHANGELOG.md)**: Version history and updates
- **[Documentation Index](docs/INDEX.md)**: Complete documentation map

## License

Apache License 2.0 - see LICENSE file for details.
