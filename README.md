# slower-whisper

## Local-first conversation signal engine for LLMs

![Python](https://img.shields.io/badge/python-3.11--3.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-production%20ready-success)

## What is slower-whisper?

slower-whisper transforms audio conversations into **LLM-ready structured data** that captures not just what was said, but **how it was said**.

Unlike traditional transcription tools that output plain text, slower-whisper produces a rich, versioned JSON format with:

- **Timestamped segments** with optional word-level alignment
- **Speaker diarization** (who spoke when, speaker attribution per segment)
- **Prosodic features** (pitch, energy, speaking rate, pauses)
- **Emotional state** (valence, arousal, categorical emotions)
- **Turn structure** and interaction patterns

**The result**: A text-only LLM can now "hear" key aspects of audio — tone, emphasis, hesitation, excitement — that aren't captured in transcription alone.

**Key properties:**

- **Runs entirely locally** (NVIDIA GPU recommended, CPU fallback supported)
- **Produces stable, versioned JSON** you can build on
- **Modular architecture** — use only the features you need
- **Contract-driven** — BDD scenarios guarantee behavioral stability

---

## Quick Start

```bash
# Clone and enter dev shell (Nix recommended)
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper
nix develop           # or: uv sync --extra full --extra dev

# Transcribe
uv run slower-whisper transcribe --root .

# Validate and export
uv run slower-whisper validate whisper_json/sample.json
uv run slower-whisper export whisper_json/sample.json --format html --output sample.html
```

Place audio files in `raw_audio/` and find transcripts in `whisper_json/`.

---

## Local Gate (Contributors)

CI may be rate-limited. **Local gate is canonical:**

```bash
./scripts/ci-local.sh        # full gate
./scripts/ci-local.sh fast   # quick check
nix-clean flake check        # Nix checks (use nix-clean wrapper inside devshell)
```

**PR requirement:** Paste gate receipts before merging.

---

## Installation

### Option 1: Nix (Recommended)

Provides reproducible environments that mirror CI.

```bash
# Install Nix (one-time)
sh <(curl -L https://nixos.org/nix/install) --daemon
mkdir -p ~/.config/nix && echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

# Enter dev shell and install
nix develop
uv sync --extra full --extra dev
```

**Note:** Inside devshell, use `nix-clean` wrapper for nix commands (avoids ABI issues).

### Option 2: UV (Fallback)

```bash
# Install system deps (ffmpeg, libsndfile) via apt/brew/choco
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra full --extra dev   # contributors
uv sync --extra full               # runtime only
uv sync                            # transcription only (~2.5GB)
```

---

## Usage

### CLI

```bash
# Transcribe audio files
uv run slower-whisper transcribe --model large-v3 --language en

# Enrich with prosody and emotion
uv run slower-whisper enrich --enable-prosody --enable-emotion

# With speaker diarization (requires HF_TOKEN)
export HF_TOKEN=hf_...
uv run slower-whisper transcribe --enable-diarization --min-speakers 2

# View help
uv run slower-whisper --help
```

### Python API

```python
from transcription import transcribe_directory, TranscriptionConfig

config = TranscriptionConfig(model="large-v3", language="en", device="cuda")
transcripts = transcribe_directory("/path/to/project", config)

for transcript in transcripts:
    for segment in transcript.segments:
        print(f"[{segment.start:.2f}s] {segment.text}")
```

### REST API

```bash
uv sync --extra api --extra full
uv run uvicorn transcription.service:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/INDEX.md](docs/INDEX.md) | **Start here** — complete documentation map |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | First transcription tutorial |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Configuration reference |
| [docs/API_QUICK_REFERENCE.md](docs/API_QUICK_REFERENCE.md) | Python API reference |
| [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | CLI command reference |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and internals |
| [docs/AUDIO_ENRICHMENT.md](docs/AUDIO_ENRICHMENT.md) | Prosody and emotion features |
| [docs/SPEAKER_DIARIZATION.md](docs/SPEAKER_DIARIZATION.md) | Speaker diarization guide |
| [docs/LLM_PROMPT_PATTERNS.md](docs/LLM_PROMPT_PATTERNS.md) | LLM integration patterns |
| [docs/API_SERVICE.md](docs/API_SERVICE.md) | REST API deployment |
| [docs/DOCKER.md](docs/DOCKER.md) | Docker deployment |
| [docs/GPU_SETUP.md](docs/GPU_SETUP.md) | GPU configuration |

### Project Files

| File | Description |
|------|-------------|
| [ROADMAP.md](ROADMAP.md) | Development roadmap and contracts |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [CLAUDE.md](CLAUDE.md) | Repo guide and invariants |

---

## Output Format

slower-whisper produces **versioned JSON** (schema v2) with stable contracts:

```json
{
  "schema_version": 2,
  "file": "meeting.wav",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Hello, let's get started.",
      "speaker": {"id": "spk_0", "confidence": 0.87},
      "audio_state": {
        "prosody": {"pitch_mean_hz": 195.2, "energy_rms": 0.24},
        "emotion": {"valence": 0.72, "label": "positive"}
      }
    }
  ],
  "speakers": [...],
  "turns": [...]
}
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete schema specification.

---

## Benchmarks

Quality checks on shipped fixtures:

- **ASR WER** — see `benchmarks/ASR_REPORT.md`
- **Diarization DER** — see `benchmarks/DIARIZATION_REPORT.md`
- **Speaker analytics** — see `benchmarks/SPEAKER_ANALYTICS_MVP.md`

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) file for details.

---

## Privacy

**Your audio and transcripts are not uploaded anywhere.** All processing runs locally. Only model weights are fetched from the internet on first use.
