# Configuration Guide

This guide covers how to configure slower-whisper for transcription (Stage 1) and audio enrichment (Stage 2). Configuration comes from three sources with a clear precedence hierarchy.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration Sources](#configuration-sources)
4. [Precedence Rules](#precedence-rules)
5. [Common Patterns](#common-patterns)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

## Overview

Configuration in slower-whisper is layered across three sources:

1. **Defaults** - Built-in values in code
2. **Environment Variables** - `SLOWER_WHISPER_*` and `SLOWER_WHISPER_ENRICH_*`
3. **Config Files** - JSON files with `TranscriptionConfig` and `EnrichmentConfig`
4. **CLI Flags** - Command-line arguments (highest priority)

The system merges these intelligently so you can use defaults, override selectively via environment, use a base file configuration, and fine-tune with CLI flags—all at the same time.

## Quick Start

### Transcription (Stage 1)

**Minimal CLI:**
```bash
# Use all defaults (large-v3 model, CUDA, float16 compute)
uv run slower-whisper transcribe

# Override model and device
uv run slower-whisper transcribe --model base --device cpu
```

**With config file:**
```bash
# Create config file
cat > transcription.json <<EOF
{
  "model": "base",
  "device": "cuda",
  "compute_type": "float16",
  "language": "en",
  "skip_existing_json": true
}
EOF

# Use config file
uv run slower-whisper transcribe --config transcription.json
```

**Python API:**
```python
from transcription import TranscriptionConfig, transcribe_directory
from pathlib import Path

# Load from file
config = TranscriptionConfig.from_file("transcription.json")

# Or create programmatically
config = TranscriptionConfig(
    model="base",
    device="cuda",
    language="en"
)

# Transcribe project
transcripts = transcribe_directory(Path("."), config)
```

### Enrichment (Stage 2)

**Minimal CLI:**
```bash
# Use all defaults (prosody + emotion enabled, CPU device)
uv run slower-whisper enrich

# Override device for faster emotion recognition
uv run slower-whisper enrich --device cuda
```

**With config file:**
```bash
cat > enrichment.json <<EOF
{
  "skip_existing": true,
  "enable_prosody": true,
  "enable_emotion": true,
  "device": "cpu"
}
EOF

uv run slower-whisper enrich --enrich-config enrichment.json
```

**Python API:**
```python
from transcription import EnrichmentConfig, enrich_directory
from pathlib import Path

config = EnrichmentConfig(
    enable_prosody=True,
    enable_emotion=True,
    device="cpu"
)

transcripts = enrich_directory(Path("."), config)
```

## Configuration Sources

### 1. Defaults

All configuration classes have sensible defaults built in:

**TranscriptionConfig defaults:**
```python
model = "large-v3"              # Recommended balance of speed/accuracy
device = "cuda"                 # Use GPU if available
compute_type = None             # Auto-derived: int8 for CPU, float16 for CUDA
language = None                 # Auto-detect from audio
task = "transcribe"             # or "translate"
skip_existing_json = True       # Skip files with existing output
vad_min_silence_ms = 500        # Silence threshold for segmentation
beam_size = 5                   # Beam search width
enable_chunking = False         # Disable chunking by default
enable_diarization = False      # Disable diarization by default
```

**EnrichmentConfig defaults:**
```python
skip_existing = True            # Skip already-enriched segments
enable_prosody = True           # Extract pitch, energy, rate
enable_emotion = True           # Extract dimensional emotion
enable_categorical_emotion = False  # Categorical emotion (slower)
enable_turn_metadata = True     # Populate turn structure
enable_speaker_stats = True     # Compute speaker aggregates
device = "cpu"                  # Emotion models on CPU (safer)
pause_threshold = None          # Only split turns on speaker change
```

### 2. Environment Variables

Environment variables override defaults but are overridden by config files and CLI flags.

**Transcription variables (prefix: `SLOWER_WHISPER_`):**

| Variable | Type | Example | Notes |
|----------|------|---------|-------|
| `SLOWER_WHISPER_MODEL` | string | `base`, `large-v3` | Must be valid Whisper model |
| `SLOWER_WHISPER_DEVICE` | string | `cuda`, `cpu` | Device for ASR |
| `SLOWER_WHISPER_COMPUTE_TYPE` | string | `float16`, `int8` | See [compute types](#compute-types) |
| `SLOWER_WHISPER_LANGUAGE` | string | `en`, `es`, `fr` | ISO 639-1 code, or `none` for auto-detect |
| `SLOWER_WHISPER_TASK` | string | `transcribe`, `translate` | Task mode |
| `SLOWER_WHISPER_SKIP_EXISTING_JSON` | bool | `true`, `1`, `yes` | Skip existing outputs |
| `SLOWER_WHISPER_VAD_MIN_SILENCE_MS` | int | `500` | Silence threshold (ms) |
| `SLOWER_WHISPER_BEAM_SIZE` | int | `5` | Beam width |
| `SLOWER_WHISPER_ENABLE_DIARIZATION` | bool | `true`, `false` | Enable speaker diarization (experimental) |
| `SLOWER_WHISPER_DIARIZATION_DEVICE` | string | `auto`, `cuda`, `cpu` | Device for diarization |
| `SLOWER_WHISPER_MIN_SPEAKERS` | int | `2` | Diarization hint (optional) |
| `SLOWER_WHISPER_MAX_SPEAKERS` | int | `5` | Diarization hint (optional) |
| `SLOWER_WHISPER_OVERLAP_THRESHOLD` | float | `0.3` | Segment-to-speaker confidence (0.0-1.0) |
| `SLOWER_WHISPER_ENABLE_CHUNKING` | bool | `true`, `false` | Enable turn-aware chunking |
| `SLOWER_WHISPER_CHUNK_TARGET_DURATION_S` | float | `30.0` | Soft target chunk size (seconds) |
| `SLOWER_WHISPER_CHUNK_MAX_DURATION_S` | float | `45.0` | Hard max chunk size (seconds) |
| `SLOWER_WHISPER_CHUNK_TARGET_TOKENS` | int | `400` | Max tokens before split |
| `SLOWER_WHISPER_CHUNK_PAUSE_SPLIT_THRESHOLD_S` | float | `1.5` | Split on pauses >= threshold (seconds) |

**Enrichment variables (prefix: `SLOWER_WHISPER_ENRICH_`):**

| Variable | Type | Example | Notes |
|----------|------|---------|-------|
| `SLOWER_WHISPER_ENRICH_SKIP_EXISTING` | bool | `true`, `1`, `yes` | Skip enriched segments |
| `SLOWER_WHISPER_ENRICH_ENABLE_PROSODY` | bool | `true`, `false` | Enable prosody extraction |
| `SLOWER_WHISPER_ENRICH_ENABLE_EMOTION` | bool | `true`, `false` | Enable emotion extraction |
| `SLOWER_WHISPER_ENRICH_ENABLE_CATEGORICAL_EMOTION` | bool | `true`, `false` | Enable categorical emotion (slower) |
| `SLOWER_WHISPER_ENRICH_ENABLE_TURN_METADATA` | bool | `true`, `false` | Enable turn structure |
| `SLOWER_WHISPER_ENRICH_ENABLE_SPEAKER_STATS` | bool | `true`, `false` | Enable speaker stats |
| `SLOWER_WHISPER_ENRICH_DEVICE` | string | `cuda`, `cpu` | Device for emotion models |
| `SLOWER_WHISPER_ENRICH_PAUSE_THRESHOLD` | float | `2.0` | Split turns on pauses >= threshold (seconds) |
| `SLOWER_WHISPER_ENRICH_DIMENSIONAL_MODEL_NAME` | string | model ID | Override default emotion model |
| `SLOWER_WHISPER_ENRICH_CATEGORICAL_MODEL_NAME` | string | model ID | Override default categorical model |

**Boolean environment variable formats:**
- True: `true`, `1`, `yes`, `on`
- False: `false`, `0`, `no`, `off`
- Null (for optional fields): `none`, `null`, `""` (empty string)

**Example:**
```bash
# Set up transcription with custom model and device
export SLOWER_WHISPER_MODEL=base
export SLOWER_WHISPER_DEVICE=cpu
export SLOWER_WHISPER_LANGUAGE=en
export SLOWER_WHISPER_ENABLE_DIARIZATION=true

# Set up enrichment to use GPU
export SLOWER_WHISPER_ENRICH_DEVICE=cuda
export SLOWER_WHISPER_ENRICH_ENABLE_EMOTION=true

# Run command (will use env settings)
uv run slower-whisper transcribe
```

### 3. Config Files

Config files are JSON documents that specify a subset of configuration fields. Only fields present in the file override defaults; missing fields use their defaults.

**Transcription config file** (`transcription.json`):
```json
{
  "model": "base",
  "device": "cuda",
  "compute_type": "float16",
  "language": "en",
  "task": "transcribe",
  "skip_existing_json": true,
  "vad_min_silence_ms": 500,
  "beam_size": 5,
  "enable_diarization": true,
  "diarization_device": "auto",
  "min_speakers": 2,
  "max_speakers": 5,
  "overlap_threshold": 0.3
}
```

**Enrichment config file** (`enrichment.json`):
```json
{
  "skip_existing": true,
  "enable_prosody": true,
  "enable_emotion": true,
  "enable_categorical_emotion": false,
  "enable_turn_metadata": true,
  "enable_speaker_stats": true,
  "device": "cpu",
  "pause_threshold": 2.0,
  "dimensional_model_name": null,
  "categorical_model_name": null
}
```

**Loading config files:**
```bash
# Transcription
uv run slower-whisper transcribe --config ./configs/transcription.json

# Enrichment
uv run slower-whisper enrich --enrich-config ./configs/enrichment.json
```

**Python API:**
```python
from transcription import TranscriptionConfig, EnrichmentConfig

# Load from file
trans_cfg = TranscriptionConfig.from_file("transcription.json")
enrich_cfg = EnrichmentConfig.from_file("enrichment.json")
```

### 4. CLI Flags

CLI flags have the highest precedence and override all other sources.

**Transcription flags:**
```bash
uv run slower-whisper transcribe \
  --model base \
  --device cuda \
  --compute-type float16 \
  --language en \
  --task transcribe \
  --skip-existing-json \
  --vad-min-silence-ms 500 \
  --beam-size 5 \
  --enable-diarization \
  --diarization-device auto \
  --min-speakers 2 \
  --max-speakers 5 \
  --overlap-threshold 0.3
```

**Enrichment flags:**
```bash
uv run slower-whisper enrich \
  --skip-existing \
  --enable-prosody \
  --enable-emotion \
  --device cuda \
  --pause-threshold 2.0
```

## Precedence Rules

Configuration is applied in this order (each level overrides previous):

1. **Defaults** - Built into code
2. **Environment Variables** - `SLOWER_WHISPER_*` family
3. **Config File** - JSON file via `--config` or `--enrich-config`
4. **CLI Flags** - Command-line arguments

### Key Principle: Explicit Settings Only

Only **explicitly set** values override lower-priority sources. If a config file doesn't mention a field, that field isn't overridden even if it equals the default.

**Example 1: File overrides env correctly**
```bash
# Environment sets device=cpu
export SLOWER_WHISPER_DEVICE=cpu

# Config file explicitly sets device=cuda (happens to equal default)
echo '{"device": "cuda"}' > config.json

# Command uses no device flag
uv run slower-whisper transcribe --config config.json

# Result: device=cuda (file overrides env, even though cuda is default)
```

**Example 2: Env preserved when not in file**
```bash
# Environment sets device=cpu
export SLOWER_WHISPER_DEVICE=cpu

# Config file omits device (so device defaults to cuda in parsing)
echo '{"model": "base"}' > config.json

# Command uses no device flag
uv run slower-whisper transcribe --config config.json

# Result: device=cpu (env preserved because file didn't mention it)
```

**Example 3: CLI overrides all**
```bash
# Environment sets device=cpu
export SLOWER_WHISPER_DEVICE=cpu

# Config file sets device=cuda
echo '{"device": "cuda"}' > config.json

# CLI flag sets device=cpu
uv run slower-whisper transcribe --config config.json --device cpu

# Result: device=cpu (CLI flag wins)
```

### Auto-Derivation of Compute Type

If `compute_type` is not explicitly set via CLI, file, or environment, it's auto-derived from the final device:
- **CPU** → `int8` (optimal for CPU inference)
- **CUDA** → `float16` (uses Tensor Cores on modern GPUs)

If any of CLI/file/env explicitly sets `compute_type`, that value is used as-is (no auto-derivation).

## Common Patterns

### Pattern 1: Development Workflow

Use environment variables for your local setup, config files for project-specific settings, CLI flags for one-off experiments.

**Step 1: Set up your local environment** (`~/.bashrc` or `.env`):
```bash
# Use default small model for fast iteration
export SLOWER_WHISPER_MODEL=base
export SLOWER_WHISPER_DEVICE=cpu  # Use CPU if no GPU
export SLOWER_WHISPER_SKIP_EXISTING_JSON=false  # Always re-transcribe
```

**Step 2: Create project config** (`myproject/transcription.json`):
```json
{
  "language": "en",
  "task": "transcribe",
  "vad_min_silence_ms": 300
}
```

**Step 3: Run with defaults + project settings:**
```bash
cd myproject
uv run slower-whisper transcribe --config transcription.json
```

**Step 4: One-off experiment (use large model with GPU):**
```bash
uv run slower-whisper transcribe --config transcription.json \
  --model large-v3 --device cuda
```

### Pattern 2: Production Deployment

Lock down all settings via config files, no environment variables.

**`production/transcription.json`:**
```json
{
  "model": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "language": "en",
  "task": "transcribe",
  "skip_existing_json": true,
  "vad_min_silence_ms": 500,
  "beam_size": 5
}
```

**`production/enrichment.json`:**
```json
{
  "skip_existing": true,
  "enable_prosody": true,
  "enable_emotion": true,
  "enable_turn_metadata": true,
  "enable_speaker_stats": true,
  "device": "cuda",
  "pause_threshold": 1.5
}
```

**Deployment script** (`deploy.sh`):
```bash
#!/bin/bash
set -e

ROOT=${PROJECT_ROOT:-"."}
TRANS_CONFIG="${ROOT}/production/transcription.json"
ENRICH_CONFIG="${ROOT}/production/enrichment.json"

echo "Transcribing audio files..."
uv run slower-whisper transcribe --root "$ROOT" --config "$TRANS_CONFIG"

echo "Enriching transcripts..."
uv run slower-whisper enrich --root "$ROOT" --enrich-config "$ENRICH_CONFIG"

echo "Done!"
```

### Pattern 3: Multi-Environment Setup

Different configs for dev/staging/production.

**Directory structure:**
```
configs/
├── base.json              # Shared settings
├── dev.json               # Development overrides
├── staging.json           # Staging overrides
└── production.json        # Production overrides
```

**`configs/base.json`:**
```json
{
  "language": "en",
  "task": "transcribe",
  "skip_existing_json": true
}
```

**`configs/dev.json`:**
```json
{
  "model": "base",
  "device": "cpu",
  "skip_existing_json": false
}
```

**`configs/staging.json`:**
```json
{
  "model": "small",
  "device": "cuda",
  "skip_existing_json": true
}
```

**`configs/production.json`:**
```json
{
  "model": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "skip_existing_json": true
}
```

**Script to run with environment:**
```bash
#!/bin/bash
ENV=${1:-dev}
CONFIG="configs/${ENV}.json"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

uv run slower-whisper transcribe --config "$CONFIG"
```

**Usage:**
```bash
./run.sh dev      # Use dev config (base model, CPU)
./run.sh staging  # Use staging config (small model, CUDA)
./run.sh production  # Use production config (large model, CUDA)
```

## Advanced Topics

### Compute Types

Faster-whisper supports multiple compute types for different hardware and accuracy trade-offs.

**Available types:**
| Type | Accuracy | Speed | Memory | Best For |
|------|----------|-------|--------|----------|
| `float32` | Highest | Slowest | High | Accuracy-critical, strong GPU memory |
| `float16` | Very High | Fast | Medium | Most CUDA GPUs (RTX, A100, etc.) |
| `int8` | High | Fast | Low | CPU inference, low-memory GPU |
| `int8_float16` | High | Very Fast | Low | Mixed precision (int8 compute, float16 accum) |
| `int8_float32` | High | Very Fast | Medium | Mixed precision (int8 compute, float32 accum) |
| `int16` | High | Medium | Low | Older GPUs, CPU |

**Selection guide:**
```bash
# For NVIDIA RTX 30/40 series (recommended)
uv run slower-whisper transcribe --device cuda --compute-type float16

# For CPU-only (recommended)
uv run slower-whisper transcribe --device cpu --compute-type int8

# For memory-constrained GPU
uv run slower-whisper transcribe --device cuda --compute-type int8

# For maximum accuracy
uv run slower-whisper transcribe --device cuda --compute-type float32

# Let it auto-select (recommended)
uv run slower-whisper transcribe --device cuda
# Auto-selects: float16 for CUDA, int8 for CPU
```

### Diarization Configuration

Diarization (speaker identification) requires additional setup and is experimental.

**Enable diarization:**
```bash
# Requires: uv sync --extra diarization
# Requires: HF_TOKEN environment variable

export HF_TOKEN="hf_xxxxx"  # From huggingface.co/settings/tokens

uv run slower-whisper transcribe \
  --enable-diarization \
  --diarization-device auto \
  --min-speakers 2 \
  --max-speakers 5 \
  --overlap-threshold 0.3
```

**Config file with diarization:**
```json
{
  "enable_diarization": true,
  "diarization_device": "auto",
  "min_speakers": 2,
  "max_speakers": 5,
  "overlap_threshold": 0.3
}
```

**Diarization hints:**
- `min_speakers` / `max_speakers`: Constrains speaker count (optional)
- `overlap_threshold` (0.0-1.0): Minimum segment-to-speaker confidence (default: 0.3)
  - Higher → more conservative speaker assignment
  - Lower → more aggressive
- `diarization_device`: Device for pyannote.audio
  - `auto`: Let system choose (GPU if available)
  - `cuda`: Force GPU
  - `cpu`: Force CPU

See [docs/SPEAKER_DIARIZATION.md](SPEAKER_DIARIZATION.md) for complete setup.

### Chunking Configuration

Chunking splits transcripts into turn-aware chunks for RAG/vector databases.

**Enable chunking:**
```bash
uv run slower-whisper transcribe \
  --enable-chunking \
  --chunk-target-duration-s 30.0 \
  --chunk-max-duration-s 45.0 \
  --chunk-target-tokens 400 \
  --chunk-pause-split-threshold-s 1.5
```

**Config file with chunking:**
```json
{
  "enable_chunking": true,
  "chunk_target_duration_s": 30.0,
  "chunk_max_duration_s": 45.0,
  "chunk_target_tokens": 400,
  "chunk_pause_split_threshold_s": 1.5
}
```

**Chunking parameters:**
- `chunk_target_duration_s`: Soft target (30s = good for most cases)
- `chunk_max_duration_s`: Hard limit, never exceed (45s recommended)
- `chunk_target_tokens`: Approx max tokens before split (400-500 typical)
- `chunk_pause_split_threshold_s`: Pause >= this triggers split (1.5s recommended)

### Audio Enrichment Configuration

Control which audio features to extract.

**Extract only prosody (fast):**
```bash
uv run slower-whisper enrich \
  --enable-prosody \
  --no-enable-emotion \
  --no-enable-categorical-emotion
```

**Extract all features (slow):**
```bash
uv run slower-whisper enrich \
  --enable-prosody \
  --enable-emotion \
  --enable-categorical-emotion \
  --enable-turn-metadata \
  --enable-speaker-stats
```

**Config file:**
```json
{
  "enable_prosody": true,
  "enable_emotion": true,
  "enable_categorical_emotion": false,
  "enable_turn_metadata": true,
  "enable_speaker_stats": true,
  "device": "cuda",
  "pause_threshold": 2.0
}
```

**Turn metadata:**
- Questions asked (detected by prosody)
- Pauses within turns
- Disfluency markers (uh, um, er)
- Automatic when `enable_turn_metadata=true`

**Speaker stats:**
- Talk time per speaker
- Turn count
- Average pause duration
- Interruption frequency
- Automatic when `enable_speaker_stats=true`

**Pause threshold:**
- Splits turns when silence >= threshold (seconds)
- `None` (default): Only split on speaker change
- `2.0`: Also split on 2-second pauses within same speaker
- Useful for long monologues

### Skipping & Caching

Control when to re-process files.

**Transcription skip logic:**
```bash
# Skip files with existing JSON output (default)
uv run slower-whisper transcribe --skip-existing-json

# Re-transcribe everything
uv run slower-whisper transcribe --no-skip-existing-json
```

**Enrichment skip logic:**
```bash
# Skip segments that already have audio_state (default)
uv run slower-whisper enrich --skip-existing

# Re-enrich everything
uv run slower-whisper enrich --no-skip-existing
```

## Troubleshooting

### "Invalid model name" error

**Symptom:**
```
ConfigurationError: Invalid model name 'medium-v2'. Must be one of: base, large-v3, ...
```

**Solution:**
Check the allowed model names. Valid Whisper models in faster-whisper:
```
tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo
```

**Fix:**
```bash
# Correct spelling
uv run slower-whisper transcribe --model large-v3  # Not "medium-v2"
```

### "Invalid compute_type" error

**Symptom:**
```
ConfigurationError: Invalid compute_type 'fp16'. Must be one of: float16, int8, ...
```

**Solution:**
Compute types are case-sensitive and must be exact:
- `float32`, `float16` (not `fp32`, `fp16`)
- `int8`, `int16` (not `int8_mixed`)
- `int8_float16`, `int8_float32` (mixed precision)

**Fix:**
```bash
uv run slower-whisper transcribe --compute-type float16  # Not "fp16"
```

### Device not found (CUDA/GPU)

**Symptom:**
```
RuntimeError: CUDA device not available; falling back to CPU
```

**Solution:**
Check GPU availability:
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"  # Should be True

# Check GPU
python -c "import torch; print(torch.cuda.get_device_name())"
```

**Fix:**
```bash
# Use CPU explicitly if GPU unavailable
uv run slower-whisper transcribe --device cpu --compute-type int8
```

### Config file not found

**Symptom:**
```
FileNotFoundError: Configuration file not found: config.json
```

**Solution:**
Verify the config file path:
```bash
# Check path
ls -la ./config.json

# Use absolute path if needed
uv run slower-whisper transcribe --config /full/path/to/config.json
```

### Environment variable not being used

**Symptom:**
Config file or CLI flag value used, but environment variable ignored.

**Solution:**
Remember the precedence: CLI flags > config file > env vars > defaults.

Check that you're not setting a higher-priority value:
```bash
# Set environment
export SLOWER_WHISPER_DEVICE=cpu

# Check what's running (if you pass --device flag, it overrides env)
uv run slower-whisper transcribe --device cuda  # Overrides env!

# To use env, don't pass the flag
uv run slower-whisper transcribe  # Uses env: device=cpu
```

Also verify the environment variable name is correct:
```bash
# Correct prefix and format
export SLOWER_WHISPER_DEVICE=cuda
export SLOWER_WHISPER_MODEL=base
export SLOWER_WHISPER_COMPUTE_TYPE=float16  # Must match _exactly_

# These won't work (wrong prefix/format)
export WHISPER_DEVICE=cuda  # Wrong prefix
export SLOWER_WHISPER_device=cuda  # Case-sensitive (must be uppercase)
```

### Boolean environment variables not parsing

**Symptom:**
```
ValueError: Invalid SLOWER_WHISPER_SKIP_EXISTING_JSON: maybe.
Must be true/false, 1/0, yes/no, or on/off
```

**Solution:**
Use only recognized boolean values:
```bash
# Valid
export SLOWER_WHISPER_SKIP_EXISTING_JSON=true      # or: false
export SLOWER_WHISPER_SKIP_EXISTING_JSON=1         # or: 0
export SLOWER_WHISPER_SKIP_EXISTING_JSON=yes       # or: no
export SLOWER_WHISPER_SKIP_EXISTING_JSON=on        # or: off

# Invalid
export SLOWER_WHISPER_SKIP_EXISTING_JSON=maybe     # Not recognized
export SLOWER_WHISPER_SKIP_EXISTING_JSON=True      # Case-sensitive (must be lowercase)
```

### Diarization errors

**Symptom:**
```
ConfigurationError: Diarization requires HF_TOKEN environment variable
```

**Solution:**
Set up HuggingFace token for model access:
```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_xxxxx"

# Then enable diarization
uv run slower-whisper transcribe --enable-diarization
```

**Symptom:**
```
ValueError: min_speakers (5) cannot be greater than max_speakers (3).
```

**Solution:**
Ensure min_speakers <= max_speakers:
```bash
# Correct
uv run slower-whisper transcribe \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 5

# Wrong
uv run slower-whisper transcribe \
  --enable-diarization \
  --min-speakers 5 \
  --max-speakers 3  # Error: min > max
```

### JSON config file syntax errors

**Symptom:**
```
json.JSONDecodeError: Expecting value: line 2 column 1 (char 1)
```

**Solution:**
Validate JSON syntax:
```bash
# Use a JSON validator
python -m json.tool config.json

# Common issues:
# 1. Missing commas between fields
{
  "model": "base"
  "device": "cuda"  # ERROR: needs comma after "base"
}

# 2. Trailing comma
{
  "model": "base",
  "device": "cuda",  # ERROR: no comma after last field
}

# 3. Unquoted keys
{
  model: "base",  # ERROR: keys must be quoted
  "device": "cuda"
}
```

### Out-of-memory errors

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**Solution:**
Reduce memory usage:
```bash
# Use smaller model
uv run slower-whisper transcribe --model base

# Use CPU
uv run slower-whisper transcribe --device cpu

# Use lower precision (int8 uses less memory than float16)
uv run slower-whisper transcribe --compute-type int8

# Skip GPU-based enrichment
uv run slower-whisper enrich --device cpu
```

### Configuration merging not working as expected

**Symptom:**
Environment variable value used instead of config file value.

**Solution:**
Remember: config file values must be **explicitly set** to override env vars. If a field isn't in the config file, the env var value is preserved.

**Example:**
```bash
# If env sets device=cpu
export SLOWER_WHISPER_DEVICE=cpu

# And config file is
{
  "model": "base"
  # device NOT mentioned
}

# Then device=cpu (from env) NOT cuda (default)

# To override, explicitly set in config:
{
  "model": "base",
  "device": "cuda"  # Now explicitly overrides env
}
```

## See Also

- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Complete CLI option reference
- [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) - Detailed env var documentation
- [SPEAKER_DIARIZATION.md](SPEAKER_DIARIZATION.md) - Diarization setup guide
- [AUDIO_ENRICHMENT.md](AUDIO_ENRICHMENT.md) - Audio enrichment features
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
