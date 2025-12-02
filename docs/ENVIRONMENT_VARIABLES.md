# Environment Variables Reference

This document provides a comprehensive reference for all environment variables used by slower-whisper.

## Overview

slower-whisper recognizes environment variables across multiple domains:

- **Transcription** (`SLOWER_WHISPER_*`): ASR and model configuration
- **Enrichment** (`SLOWER_WHISPER_ENRICH_*`): Audio feature extraction settings
- **Diarization** (`HF_TOKEN`, `SLOWER_WHISPER_PYANNOTE_MODE`): Speaker identification
- **Cache** (`SLOWER_WHISPER_CACHE_ROOT`, `HF_HOME`, `TORCH_HOME`): Model and data directories
- **API** (`SLOWER_WHISPER_API_MODE`, `SLOWER_WHISPER_API_URL`): Service endpoint configuration

All environment variables are **optional**. If not set, sensible defaults are used.

---

## Transcription Configuration

These variables control Stage 1 transcription settings (ASR and model behavior).

**Prefix**: `SLOWER_WHISPER_`

### Model Selection

#### `SLOWER_WHISPER_MODEL`

- **Type**: string
- **Allowed values**: `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo`
- **Default**: `large-v3`
- **Description**: Whisper model size to use for transcription. Larger models are more accurate but slower and require more memory.
- **Example**: `export SLOWER_WHISPER_MODEL=medium`

#### `SLOWER_WHISPER_DEVICE`

- **Type**: string
- **Allowed values**: `cuda`, `cpu`
- **Default**: `cuda` (falls back to `cpu` if CUDA unavailable)
- **Description**: Compute device for ASR. Use `cuda` for NVIDIA GPUs (much faster) or `cpu` for CPU-only systems.
- **Example**: `export SLOWER_WHISPER_DEVICE=cpu`

#### `SLOWER_WHISPER_COMPUTE_TYPE`

- **Type**: string (case-insensitive)
- **Allowed values**: `float16`, `float32`, `int16`, `int8`, `int8_float16`, `int8_float32`
- **Default**: Auto-selected (`float16` for CUDA, `int8` for CPU)
- **Description**: Precision/quantization for model computation. Lower precision = faster but less accurate.
- **Example**: `export SLOWER_WHISPER_COMPUTE_TYPE=int8`

### Language and Task

#### `SLOWER_WHISPER_LANGUAGE`

- **Type**: string
- **Allowed values**: BCP-47 language codes (e.g., `en`, `es`, `fr`)
- **Default**: `None` (auto-detect from audio)
- **Description**: Language code for transcription. If not set, Whisper auto-detects the language.
- **Example**: `export SLOWER_WHISPER_LANGUAGE=en`

#### `SLOWER_WHISPER_TASK`

- **Type**: string
- **Allowed values**: `transcribe`, `translate`
- **Default**: `transcribe`
- **Description**: Transcription task. Set to `translate` to transcribe non-English audio and translate to English.
- **Example**: `export SLOWER_WHISPER_TASK=transcribe`

### Advanced ASR Options

#### `SLOWER_WHISPER_VAD_MIN_SILENCE_MS`

- **Type**: integer
- **Constraints**: Must be non-negative
- **Default**: `500`
- **Description**: Minimum silence duration (milliseconds) to trigger VAD (voice activity detection). Shorter silence = more segments.
- **Example**: `export SLOWER_WHISPER_VAD_MIN_SILENCE_MS=300`

#### `SLOWER_WHISPER_BEAM_SIZE`

- **Type**: integer
- **Constraints**: Must be positive (≥ 1)
- **Default**: `5`
- **Description**: Beam search width for decoding. Larger values = more accurate but slower. Typical range: 1-10.
- **Example**: `export SLOWER_WHISPER_BEAM_SIZE=10`

### Behavior

#### `SLOWER_WHISPER_SKIP_EXISTING_JSON`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `true`
- **Description**: If `true`, skip transcribing files that already have JSON output. Useful for resuming interrupted runs.
- **Example**: `export SLOWER_WHISPER_SKIP_EXISTING_JSON=false`

---

## Enrichment Configuration

These variables control Stage 2 audio enrichment (prosody, emotion, turn analysis).

**Prefix**: `SLOWER_WHISPER_ENRICH_`

### Feature Toggles

#### `SLOWER_WHISPER_ENRICH_ENABLE_PROSODY`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `true`
- **Description**: Extract prosodic features (pitch, energy, speech rate) from audio.
- **Example**: `export SLOWER_WHISPER_ENRICH_ENABLE_PROSODY=true`

#### `SLOWER_WHISPER_ENRICH_ENABLE_EMOTION`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `true`
- **Description**: Extract dimensional emotion features (valence, arousal) using pre-trained models.
- **Example**: `export SLOWER_WHISPER_ENRICH_ENABLE_EMOTION=true`

#### `SLOWER_WHISPER_ENRICH_ENABLE_CATEGORICAL_EMOTION`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `false`
- **Description**: Extract categorical emotion labels (happy, sad, angry, neutral, etc.). Requires additional model downloads.
- **Example**: `export SLOWER_WHISPER_ENRICH_ENABLE_CATEGORICAL_EMOTION=true`

#### `SLOWER_WHISPER_ENRICH_ENABLE_TURN_METADATA`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `true`
- **Description**: Build turn-level metadata (sequence of speaker turns with aggregated features).
- **Example**: `export SLOWER_WHISPER_ENRICH_ENABLE_TURN_METADATA=true`

#### `SLOWER_WHISPER_ENRICH_ENABLE_SPEAKER_STATS`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `true`
- **Description**: Compute per-speaker statistics (total talk time, segment counts, average confidence).
- **Example**: `export SLOWER_WHISPER_ENRICH_ENABLE_SPEAKER_STATS=true`

#### `SLOWER_WHISPER_ENRICH_ENABLE_SEMANTIC_ANNOTATOR`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `false`
- **Description**: Enable semantic annotation (summaries, intent tags) if a custom annotator is provided.
- **Example**: `export SLOWER_WHISPER_ENRICH_ENABLE_SEMANTIC_ANNOTATOR=false`

### Device and Performance

#### `SLOWER_WHISPER_ENRICH_DEVICE`

- **Type**: string
- **Allowed values**: `cuda`, `cpu`
- **Default**: `cpu`
- **Description**: Compute device for enrichment (emotion models, diarization). Use `cuda` if available.
- **Example**: `export SLOWER_WHISPER_ENRICH_DEVICE=cuda`

### Model Configuration

#### `SLOWER_WHISPER_ENRICH_DIMENSIONAL_MODEL_NAME`

- **Type**: string (HuggingFace model identifier)
- **Allowed values**: Model IDs like `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- **Default**: `null` (uses built-in default)
- **Description**: Custom HuggingFace model for dimensional emotion extraction (valence/arousal).
- **Example**: `export SLOWER_WHISPER_ENRICH_DIMENSIONAL_MODEL_NAME=audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`

#### `SLOWER_WHISPER_ENRICH_CATEGORICAL_MODEL_NAME`

- **Type**: string (HuggingFace model identifier)
- **Allowed values**: Model IDs like `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- **Default**: `null` (uses built-in default)
- **Description**: Custom HuggingFace model for categorical emotion extraction (emotion labels).
- **Example**: `export SLOWER_WHISPER_ENRICH_CATEGORICAL_MODEL_NAME=ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`

### Behavior and Tuning

#### `SLOWER_WHISPER_ENRICH_SKIP_EXISTING`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `true`
- **Description**: If `true`, skip enriching transcripts that already have `audio_state` populated.
- **Example**: `export SLOWER_WHISPER_ENRICH_SKIP_EXISTING=false`

#### `SLOWER_WHISPER_ENRICH_PAUSE_THRESHOLD`

- **Type**: float (seconds)
- **Constraints**: Must be non-negative
- **Default**: `null` (uses adaptive thresholds)
- **Description**: Minimum pause duration (seconds) to split speaker turns. If `null`, uses speaker-relative analysis.
- **Example**: `export SLOWER_WHISPER_ENRICH_PAUSE_THRESHOLD=0.5`

---

## Diarization Configuration

These variables control speaker diarization (who spoke when).

### Hugging Face Access

#### `HF_TOKEN`

- **Type**: string (Hugging Face API token)
- **Constraints**: Required only for real diarization (with pyannote.audio)
- **Default**: `null`
- **Description**: Hugging Face user access token needed to download pyannote.audio speaker diarization models. Get one at https://huggingface.co/settings/tokens.
- **Obtain token**: Visit https://huggingface.co/settings/tokens and create a "read" token
- **Accept license**: Accept the license at https://huggingface.co/pyannote/speaker-diarization-3.1
- **Example**: `export HF_TOKEN=hf_xyzabc123...`

### Diarization Mode Control

#### `SLOWER_WHISPER_PYANNOTE_MODE`

- **Type**: string
- **Allowed values**: `auto`, `stub`, `missing`
- **Default**: `auto`
- **Description**: Controls diarization backend behavior:
  - `auto`: Try real pyannote.audio; fall back to stub if unavailable
  - `stub`: Force stub mode (generates 2-speaker pattern for testing)
  - `missing`: Treat pyannote as completely unavailable; raise errors if diarization requested
- **Example**: `export SLOWER_WHISPER_PYANNOTE_MODE=stub`

### Transcription Diarization Options

#### `SLOWER_WHISPER_ENABLE_DIARIZATION`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `false`
- **Description**: Enable speaker diarization (requires diarization dependencies and `HF_TOKEN`).
- **Example**: `export SLOWER_WHISPER_ENABLE_DIARIZATION=true`

#### `SLOWER_WHISPER_DIARIZATION_DEVICE`

- **Type**: string
- **Allowed values**: `cuda`, `cpu`, `auto`
- **Default**: `auto`
- **Description**: Device for diarization model. `auto` selects best available device.
- **Example**: `export SLOWER_WHISPER_DIARIZATION_DEVICE=cuda`

#### `SLOWER_WHISPER_MIN_SPEAKERS`

- **Type**: integer (or `null`)
- **Constraints**: Positive integer or `null`
- **Default**: `null`
- **Description**: Minimum number of speakers to detect. If `null`, auto-detect.
- **Example**: `export SLOWER_WHISPER_MIN_SPEAKERS=1`

#### `SLOWER_WHISPER_MAX_SPEAKERS`

- **Type**: integer (or `null`)
- **Constraints**: Positive integer or `null`
- **Default**: `null`
- **Description**: Maximum number of speakers to detect. If `null`, auto-detect.
- **Example**: `export SLOWER_WHISPER_MAX_SPEAKERS=4`

#### `SLOWER_WHISPER_OVERLAP_THRESHOLD`

- **Type**: float
- **Constraints**: Must be between 0.0 and 1.0
- **Default**: `0.3`
- **Description**: Overlap threshold for speaker segments. Higher = stricter separation between speakers.
- **Example**: `export SLOWER_WHISPER_OVERLAP_THRESHOLD=0.5`

---

## Chunking Configuration

These variables control optional audio chunking for very long files (v1.3+ feature).

**Prefix**: `SLOWER_WHISPER_`

#### `SLOWER_WHISPER_ENABLE_CHUNKING`

- **Type**: boolean
- **Allowed values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Default**: `false`
- **Description**: Enable chunking of audio into smaller pieces for transcription. Useful for very long files.
- **Example**: `export SLOWER_WHISPER_ENABLE_CHUNKING=true`

#### `SLOWER_WHISPER_CHUNK_TARGET_DURATION_S`

- **Type**: float (seconds)
- **Constraints**: Must be positive
- **Default**: `30.0`
- **Description**: Target duration for audio chunks (seconds). Algorithm tries to split audio near this duration.
- **Example**: `export SLOWER_WHISPER_CHUNK_TARGET_DURATION_S=45.0`

#### `SLOWER_WHISPER_CHUNK_MAX_DURATION_S`

- **Type**: float (seconds)
- **Constraints**: Must be positive
- **Default**: `45.0`
- **Description**: Maximum chunk duration (seconds). Hard limit to prevent chunks from getting too large. Must be ≥ `CHUNK_TARGET_DURATION_S`.
- **Example**: `export SLOWER_WHISPER_CHUNK_MAX_DURATION_S=60.0`

#### `SLOWER_WHISPER_CHUNK_TARGET_TOKENS`

- **Type**: integer (tokens)
- **Constraints**: Must be positive
- **Default**: `400`
- **Description**: Target token count per chunk for Whisper's context window. Larger = more context but slower.
- **Example**: `export SLOWER_WHISPER_CHUNK_TARGET_TOKENS=500`

#### `SLOWER_WHISPER_CHUNK_PAUSE_SPLIT_THRESHOLD_S`

- **Type**: float (seconds)
- **Constraints**: Must be non-negative
- **Default**: `1.5`
- **Description**: Pause duration threshold (seconds) to prefer splitting chunks at silence. Helps preserve natural speech boundaries.
- **Example**: `export SLOWER_WHISPER_CHUNK_PAUSE_SPLIT_THRESHOLD_S=2.0`

---

## Cache Configuration

These variables control where all model downloads and cached data are stored.

### Root Cache Directory

#### `SLOWER_WHISPER_CACHE_ROOT`

- **Type**: string (file path)
- **Constraints**: Must be a writable directory
- **Default**: `~/.cache/slower-whisper`
- **Description**: Root directory for all slower-whisper caches (models, datasets, benchmarks). All other cache paths are derived from this unless explicitly overridden.
- **Example**: `export SLOWER_WHISPER_CACHE_ROOT=/mnt/large_disk/models`

### Hugging Face Cache

#### `HF_HOME`

- **Type**: string (file path)
- **Constraints**: Must be writable
- **Default**: `$SLOWER_WHISPER_CACHE_ROOT/hf`
- **Description**: Hugging Face cache directory for model downloads (Whisper, pyannote, emotion models). Set explicitly to override the default.
- **Example**: `export HF_HOME=/custom/hf/cache`

### PyTorch Cache

#### `TORCH_HOME`

- **Type**: string (file path)
- **Constraints**: Must be writable
- **Default**: `$SLOWER_WHISPER_CACHE_ROOT/torch`
- **Description**: PyTorch cache directory for torch module downloads. Set explicitly to override the default.
- **Example**: `export TORCH_HOME=/custom/torch/cache`

---

## API Service Configuration

These variables control the slower-whisper API service (when running as a server).

#### `SLOWER_WHISPER_API_MODE`

- **Type**: string
- **Allowed values**: `stub`, `real`, `live`
- **Default**: `stub`
- **Description**: API service mode for testing:
  - `stub`: Mock API responses (offline testing)
  - `real`/`live`: Connect to a real running API service
- **Example**: `export SLOWER_WHISPER_API_MODE=stub`

#### `SLOWER_WHISPER_API_URL`

- **Type**: string (URL)
- **Constraints**: Must be valid URL
- **Default**: `http://localhost:8765`
- **Description**: Base URL for the API service. Used when making requests to a running server.
- **Example**: `export SLOWER_WHISPER_API_URL=http://api.example.com:8765`

---

## Usage Examples

### Example 1: CPU-based Transcription (No GPU)

```bash
export SLOWER_WHISPER_MODEL=base          # Smaller model for faster CPU inference
export SLOWER_WHISPER_DEVICE=cpu
export SLOWER_WHISPER_COMPUTE_TYPE=int8   # Quantized for CPU efficiency
export SLOWER_WHISPER_LANGUAGE=en
```

### Example 2: GPU-accelerated with Full Enrichment

```bash
export SLOWER_WHISPER_MODEL=large-v3
export SLOWER_WHISPER_DEVICE=cuda
export SLOWER_WHISPER_COMPUTE_TYPE=float16

export SLOWER_WHISPER_ENRICH_DEVICE=cuda
export SLOWER_WHISPER_ENRICH_ENABLE_PROSODY=true
export SLOWER_WHISPER_ENRICH_ENABLE_EMOTION=true
export SLOWER_WHISPER_ENRICH_ENABLE_CATEGORICAL_EMOTION=true
```

### Example 3: Speaker Diarization

```bash
export HF_TOKEN=hf_your_token_here
export SLOWER_WHISPER_ENABLE_DIARIZATION=true
export SLOWER_WHISPER_DIARIZATION_DEVICE=cuda
export SLOWER_WHISPER_MIN_SPEAKERS=2
export SLOWER_WHISPER_MAX_SPEAKERS=4
```

### Example 4: Long Audio with Chunking

```bash
export SLOWER_WHISPER_ENABLE_CHUNKING=true
export SLOWER_WHISPER_CHUNK_TARGET_DURATION_S=30
export SLOWER_WHISPER_CHUNK_MAX_DURATION_S=45
export SLOWER_WHISPER_CHUNK_PAUSE_SPLIT_THRESHOLD_S=1.5
```

### Example 5: Custom Cache Location

```bash
export SLOWER_WHISPER_CACHE_ROOT=/mnt/fast_ssd/models
# HF_HOME and TORCH_HOME are now automatically:
#   /mnt/fast_ssd/models/hf
#   /mnt/fast_ssd/models/torch
```

---

## Environment Variable Precedence

When loading configuration, variables are applied in this order (later overrides earlier):

1. **Default values** (hardcoded in code)
2. **Configuration file** (`--config config.json`)
3. **Environment variables** (`SLOWER_WHISPER_*`)
4. **CLI arguments** (`--model`, `--device`, etc.)

This means:
- CLI arguments take highest priority
- Environment variables override config files
- Config files override defaults
- Defaults are used only if nothing else is specified

---

## Validation

All environment variables are validated when loaded:

- **Type checking**: Values are checked to match expected types
- **Range validation**: Numeric values are checked for allowed ranges
- **Enum validation**: String values are checked against allowed values
- **Dependency validation**: Some variables depend on others being set (e.g., `HF_TOKEN` for real diarization)

If a variable contains an invalid value, an error is raised with a descriptive message indicating what went wrong.

---

## Common Issues

### CUDA Not Found

If `SLOWER_WHISPER_DEVICE=cuda` fails:

```bash
# Fall back to CPU
export SLOWER_WHISPER_DEVICE=cpu
export SLOWER_WHISPER_COMPUTE_TYPE=int8
```

### Diarization Requires Token

If diarization fails with "HF_TOKEN required":

```bash
export HF_TOKEN=hf_your_token_here
# Then accept the license at:
# https://huggingface.co/pyannote/speaker-diarization-3.1
```

### Out of Memory

If GPU runs out of memory:

```bash
export SLOWER_WHISPER_MODEL=base       # Use smaller model
export SLOWER_WHISPER_COMPUTE_TYPE=int8  # Use quantization
export SLOWER_WHISPER_ENRICH_DEVICE=cpu  # Run enrichment on CPU
```

### Models Not Caching

If models keep re-downloading:

```bash
export SLOWER_WHISPER_CACHE_ROOT=/path/with/space
# Ensure directory has write permissions: chmod 755 /path/with/space
```

---

## Programmatic Access

To load configuration from environment variables in Python code:

```python
from transcription.config import TranscriptionConfig, EnrichmentConfig

# Load transcription settings from environment
transcribe_config = TranscriptionConfig.from_env()

# Load enrichment settings from environment
enrich_config = EnrichmentConfig.from_env()

# Customize prefix if needed
custom_config = TranscriptionConfig.from_env(prefix="CUSTOM_PREFIX_")
```

---

## See Also

- **[Configuration Guide](./CONFIGURATION.md)**: Using config files and programmatic configuration
- **[Architecture Documentation](./ARCHITECTURE.md)**: Design and module overview
- **[README](../README.md)**: User guide and quickstart
