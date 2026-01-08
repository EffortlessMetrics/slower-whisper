# slower-whisper

## Local-first conversation signal engine for LLMs

![Python](https://img.shields.io/badge/python-3.11--3.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-production%20ready-success)

slower-whisper transforms audio conversations into **LLM-ready structured data** that captures not just what was said, but **how it was said**.

Output includes timestamped segments, speaker diarization, prosodic features (pitch, energy, speaking rate), and emotional state. A text-only LLM can now "hear" tone, emphasis, and hesitation that aren't in the transcription alone.

**Key properties:** Runs entirely locally (GPU recommended, CPU fallback). Produces stable, versioned JSON (schema v2). Modular architecture — use only what you need.

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
