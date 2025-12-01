# INSTALL.md

Quick paths for getting slower-whisper running locally. Use Nix for a
reproducible setup that mirrors CI; fall back to a plain Python + ffmpeg
environment if you prefer.

## Prerequisites

- OS: Linux, macOS, or Windows (WSL supported)
- Python: 3.10+ (managed by Nix or your system manager)
- System deps: `ffmpeg` and `libsndfile`
- Python package manager: `uv` (recommended)
- Optional GPU: NVIDIA GPU with recent drivers + CUDA for best performance

## Option A: Nix (recommended)

```bash
# 1) Install Nix (one-time)
sh <(curl -L https://nixos.org/nix/install) --daemon

# 2) Enable flakes (required for this repo)
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

# 3) Enter the dev shell (provides Python, ffmpeg, libsndfile)
nix develop

# 4) Install Python deps via uv (full feature set for contributors)
uv sync --extra full --extra diarization --extra dev

# 5) Sanity-check the install
uv run slower-whisper --version
uv run slower-whisper transcribe --help
```

## Option B: Traditional (no Nix)

```bash
# 1) Install system deps (examples)
# macOS: brew install ffmpeg libsndfile
# Ubuntu: sudo apt-get install ffmpeg libsndfile1
# Windows: choco install ffmpeg

# 2) Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3) Install Python deps
uv sync --extra full

# 4) Verify CLI
uv run slower-whisper --version
uv run slower-whisper transcribe --help
```

## Feature Extras

- Minimal transcription only: `uv sync`
- Prosody/emotion/diarization: `uv sync --extra full`
- Real pyannote diarization requires a HuggingFace token: export `HF_TOKEN=...`
- Optional diarization modes for tests/CI:
  - `export SLOWER_WHISPER_PYANNOTE_MODE=stub` (fake backend, no HF_TOKEN needed)
  - `export SLOWER_WHISPER_PYANNOTE_MODE=missing` (simulate missing dependency)
- API server: `uv sync --extra api`

## Quick Test

```bash
mkdir -p raw_audio
cp path/to/audio.wav raw_audio/
uv run slower-whisper transcribe raw_audio/audio.wav
# Outputs land in whisper_json/ and transcripts/
```

## Common Tasks

- Update deps: `uv sync --upgrade`
- Dev shell with tools: `nix develop` then `uv sync --extra dev`
- Clean install: remove `.venv` then rerun `uv sync --extra dev`

## Troubleshooting

- See `docs/TROUBLESHOOTING.md` for common errors.
- GPU issues: confirm drivers/CUDA, retry inside `nix develop`.
- Missing diarization weights: ensure `HF_TOKEN` is set before running the real pyannote backend.
