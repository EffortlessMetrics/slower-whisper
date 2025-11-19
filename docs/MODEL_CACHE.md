# Model Downloads & Cache Management

## Overview

slower-whisper downloads several large models on first use:

- **Whisper ASR weights** (via faster-whisper) — 150MB to 3GB depending on model size
- **Emotion models** (wav2vec2) — ~2-4GB
- **Pyannote diarization models** (optional, v1.1+) — ~500MB to 2GB

By default, these are cached under a single root directory and **reused across virtualenvs and runs**. This avoids re-downloading multi-GB models every time you recreate a virtualenv or run the pipeline.

## Cache Structure

Default cache root: `~/.cache/slower-whisper`

Directory layout:

```
~/.cache/slower-whisper/
  hf/              # Hugging Face hub cache (pyannote, transformers models)
  torch/           # PyTorch cache (torch hub, etc.)
  whisper/         # Whisper model weights (faster-whisper)
  emotion/         # Emotion recognition models (wav2vec2)
  diarization/     # Speaker diarization models (pyannote)
```

## Customizing Cache Location

Set the `SLOWER_WHISPER_CACHE_ROOT` environment variable:

```bash
export SLOWER_WHISPER_CACHE_ROOT=/mnt/models/slower-whisper
uv run slower-whisper transcribe --root .
```

Or permanently in your shell profile (~/.bashrc, ~/.zshrc):

```bash
export SLOWER_WHISPER_CACHE_ROOT="$HOME/.cache/slower-whisper"
```

You can also override individual cache directories:

```bash
export HF_HOME=/mnt/models/huggingface
export TORCH_HOME=/mnt/models/torch
```

## CLI Commands

### Show Cache Status

Inspect cache locations and sizes:

```bash
uv run slower-whisper cache --show
```

Output:

```
slower-whisper cache locations:
  Root:         /home/user/.cache/slower-whisper
  HF_HOME:      /home/user/.cache/slower-whisper/hf (2.5 GB)
  TORCH_HOME:   /home/user/.cache/slower-whisper/torch (0.0 B)
  Whisper:      /home/user/.cache/slower-whisper/whisper (1.5 GB)
  Emotion:      /home/user/.cache/slower-whisper/emotion (3.2 GB)
  Diarization:  /home/user/.cache/slower-whisper/diarization (1.8 GB)

  Total:        9.0 GB
```

### Clear Caches

Clear specific caches:

```bash
# Clear Whisper models only
uv run slower-whisper cache --clear whisper

# Clear emotion models
uv run slower-whisper cache --clear emotion

# Clear diarization models
uv run slower-whisper cache --clear diarization

# Clear all Hugging Face models
uv run slower-whisper cache --clear hf

# Clear PyTorch cache
uv run slower-whisper cache --clear torch

# Clear everything
uv run slower-whisper cache --clear all
```

**Note**: Clearing a cache will force re-download on next use.

## What About Python Packages?

**Cache infrastructure controls model weights, not Python packages**.

Python packages (torch, pyannote.audio, transformers, etc.) are installed into your virtualenv by `uv sync` or `pip install`. These live in:

- `$VIRTUAL_ENV/lib/python3.X/site-packages` (per-venv)

Model weights are separate and live in the cache root described above.

### First Install

First time installing diarization dependencies:

```bash
uv sync --extra diarization
```

This installs PyTorch + pyannote (~2-3GB download, 10-20 minutes). This happens **per virtualenv**.

### Subsequent Installs

Model weights are cached, so subsequent runs (even in new venvs) only need to install packages, not re-download weights:

```bash
# In new venv
uv sync --extra diarization
# Installs packages (~10-20 minutes)

# Run diarization (uses cached weights, no download)
uv run slower-whisper transcribe --enable-diarization --root .
```

## Disk Space Requirements

Typical cache sizes:

| Component | Size | Notes |
|-----------|------|-------|
| Whisper base | ~150MB | Basic ASR |
| Whisper large-v3 | ~3GB | Production ASR |
| Emotion models | ~3-4GB | wav2vec2 dimensional + categorical |
| Pyannote diarization | ~1-2GB | speaker-diarization-3.1 |
| HF/Torch overhead | ~500MB | Tokenizers, configs, etc. |
| **Total (full install)** | **8-10GB** | All features enabled |

Add 3-5GB for virtualenv packages (torch, pyannote, transformers).

## Cache Behavior

### Automatic Caching

- All model downloads are cached automatically
- Cache is shared across:
  - Multiple virtualenvs
  - Different project directories
  - CLI runs and Python API usage

### Cache Hits

Once downloaded, models load from cache:

- **Whisper**: 1-5 seconds (model load time only)
- **Emotion**: 5-10 seconds (model + GPU init)
- **Diarization**: 10-20 seconds (model + GPU init)

No network access after first download (unless models are updated upstream).

### Model Updates

Models are cached indefinitely until explicitly cleared. To get updated models:

```bash
# Clear specific cache and re-download
uv run slower-whisper cache --clear diarization
uv run slower-whisper transcribe --enable-diarization --root .
```

Or clear all:

```bash
uv run slower-whisper cache --clear all
```

## Troubleshooting

### Models not caching

Check environment variables:

```bash
uv run slower-whisper cache --show
```

Ensure `HF_HOME`, `TORCH_HOME`, and cache subdirectories point to expected locations.

### Permission errors

Ensure cache root is writable:

```bash
ls -ld ~/.cache/slower-whisper
```

If permissions are wrong:

```bash
chmod -R u+w ~/.cache/slower-whisper
```

### Disk space issues

Clear large caches you don't need:

```bash
# Don't use emotion? Clear it
uv run slower-whisper cache --clear emotion

# Only use Whisper? Clear everything else
uv run slower-whisper cache --clear diarization
uv run slower-whisper cache --clear emotion
```

### Multiple users on same machine

Each user should have their own cache root:

```bash
# User 1
export SLOWER_WHISPER_CACHE_ROOT=/home/user1/.cache/slower-whisper

# User 2
export SLOWER_WHISPER_CACHE_ROOT=/home/user2/.cache/slower-whisper
```

Or share a cache (requires shared write permissions):

```bash
export SLOWER_WHISPER_CACHE_ROOT=/mnt/shared/slower-whisper
chmod -R g+w /mnt/shared/slower-whisper
```

## CI/CD and Docker

### Docker Builds

Cache models at build time to avoid runtime downloads:

```dockerfile
FROM python:3.11

# Set cache root to persist across builds
ENV SLOWER_WHISPER_CACHE_ROOT=/models/cache

# Install dependencies
RUN pip install slower-whisper[full]

# Pre-download models (forces cache population)
RUN python -c "from transcription.cache import configure_global_cache_env; configure_global_cache_env()"
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"
RUN python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')"

# Mount cache as volume for runtime
VOLUME /models/cache
```

### CI Caching

Cache model downloads in CI to avoid re-downloading on every build:

```yaml
# GitHub Actions example
- name: Cache model weights
  uses: actions/cache@v3
  with:
    path: ~/.cache/slower-whisper
    key: slower-whisper-models-${{ hashFiles('pyproject.toml') }}
    restore-keys: |
      slower-whisper-models-
```

## Advanced: Programmatic Access

Python API for cache management:

```python
from transcription.cache import CachePaths, configure_global_cache_env

# Configure cache early in your script
paths = configure_global_cache_env()

# Inspect paths
print(f"Whisper cache: {paths.whisper_root}")
print(f"Total cache size: {paths.root}")

# Check if model is cached
if (paths.whisper_root / "models--openai--whisper-large-v3").exists():
    print("Whisper large-v3 is cached")
```

## Summary

- **Model weights**: Cached under `$SLOWER_WHISPER_CACHE_ROOT`, reused across venvs
- **Python packages**: Installed per-venv, not cached by this infrastructure
- **First install**: Heavy (10-20 minutes for packages + weights)
- **Subsequent runs**: Fast (use cached weights, no re-download)
- **CLI**: `slower-whisper cache --show` and `--clear` for management
- **Disk space**: ~10GB for full install (ASR + emotion + diarization)
