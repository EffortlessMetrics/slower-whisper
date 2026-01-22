# Configuration Examples

This directory contains sample configuration files for slower-whisper's transcription and enrichment pipelines.

## Configuration Precedence

Configuration values are loaded in the following order (highest to lowest priority):

```
1. CLI flags (highest priority)
   ↓
2. Config file (--config)
   ↓
3. Environment variables (SLOWER_WHISPER_*)
   ↓
4. Defaults (lowest priority)
```

Each layer overrides only the values explicitly set in lower-priority layers.

## Using Configuration Files

### Transcription Configuration

```bash
# Use a config file
uv run slower-whisper transcribe --config examples/config_examples/transcription_basic.json

# Override specific values from config file
uv run slower-whisper transcribe \
  --config examples/config_examples/transcription_basic.json \
  --model large-v3 \
  --language auto

# Load from Python
from transcription import TranscriptionConfig

config = TranscriptionConfig.from_file("transcription_basic.json")
```

### Enrichment Configuration

```bash
# Use a config file
uv run slower-whisper enrich --config examples/config_examples/enrichment_full.json

# Override specific values
uv run slower-whisper enrich \
  --config examples/config_examples/enrichment_full.json \
  --device cpu \
  --no-enable-categorical-emotion

# Load from Python
from transcription import EnrichmentConfig

config = EnrichmentConfig.from_file("enrichment_full.json")
```

## Environment Variables

### Transcription Environment Variables

Set these with the `SLOWER_WHISPER_` prefix:

```bash
export SLOWER_WHISPER_MODEL=large-v3
export SLOWER_WHISPER_DEVICE=cuda
export SLOWER_WHISPER_COMPUTE_TYPE=float16
export SLOWER_WHISPER_LANGUAGE=en
export SLOWER_WHISPER_TASK=transcribe
export SLOWER_WHISPER_SKIP_EXISTING_JSON=true
export SLOWER_WHISPER_VAD_MIN_SILENCE_MS=500
export SLOWER_WHISPER_BEAM_SIZE=5
```

### Enrichment Environment Variables

Set these with the `SLOWER_WHISPER_ENRICH_` prefix:

```bash
export SLOWER_WHISPER_ENRICH_SKIP_EXISTING=true
export SLOWER_WHISPER_ENRICH_ENABLE_PROSODY=true
export SLOWER_WHISPER_ENRICH_ENABLE_EMOTION=true
export SLOWER_WHISPER_ENRICH_ENABLE_CATEGORICAL_EMOTION=false
export SLOWER_WHISPER_ENRICH_DEVICE=cuda
export SLOWER_WHISPER_ENRICH_DIMENSIONAL_MODEL_NAME=audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
```

## Configuration Files

### Transcription Configs

- **`transcription_basic.json`**: Lightweight setup using base model
  - Use case: Quick transcription, limited resources
  - Model: base (~150MB)
  - Device: cuda
  - Language: English

- **`transcription_full_enrichment.json`**: Best quality for enrichment workflow
  - Use case: Production transcription followed by audio enrichment
  - Model: large-v3 (~3GB)
  - Device: cuda
  - Language: auto-detect

- **`transcription_production.json`**: Production deployment settings
  - Use case: High-quality production transcription
  - Model: large-v3
  - Device: cuda
  - Advanced: Tighter VAD, larger beam size for accuracy

- **`transcription_dev_testing.json`**: Fast testing and development
  - Use case: CI/CD, rapid iteration, no GPU
  - Model: tiny (~75MB)
  - Device: cpu
  - Optimization: Minimal beam size, quantized weights

### Enrichment Configs

- **`enrichment_basic.json`**: Prosody-only (lightweight)
  - Use case: Basic pitch/energy/rate analysis
  - Prosody: enabled
  - Emotion: disabled
  - Device: cpu

- **`enrichment_full.json`**: Complete audio analysis
  - Use case: Full emotional and prosodic analysis
  - Prosody: enabled
  - Emotion: both dimensional and categorical
  - Device: cuda (for faster emotion inference)

- **`enrichment_production.json`**: Optimized production enrichment
  - Use case: Production deployment with emotion analysis
  - Prosody: enabled
  - Emotion: dimensional only (faster than categorical)
  - Device: cuda
  - Custom model: explicit dimensional model specification

- **`enrichment_dev_testing.json`**: Fast development testing
  - Use case: Testing enrichment logic without GPU
  - Prosody: enabled
  - Emotion: disabled (faster)
  - Device: cpu
  - Skip existing: false (always re-process)

## Precedence Examples

### Example 1: CLI flags override config file

```bash
# Config file says model=base, but CLI overrides to large-v3
uv run slower-whisper transcribe \
  --config transcription_basic.json \
  --model large-v3
# Result: Uses large-v3 model, other settings from config file
```

### Example 2: Config file overrides environment

```bash
# Environment says device=cpu
export SLOWER_WHISPER_DEVICE=cpu

# Config file says device=cuda
uv run slower-whisper transcribe --config transcription_production.json
# Result: Uses cuda (config file wins over environment)
```

### Example 3: Environment overrides defaults

```bash
# No config file, no CLI flags
export SLOWER_WHISPER_MODEL=medium

uv run slower-whisper transcribe
# Result: Uses medium model (environment wins over defaults)
```

### Example 4: Layered precedence

```bash
# Set environment variable
export SLOWER_WHISPER_DEVICE=cpu
export SLOWER_WHISPER_MODEL=base

# Use config file (has device=cuda, model not set)
# Override model with CLI flag
uv run slower-whisper transcribe \
  --config transcription_production.json \
  --model large-v3

# Result:
# - model: large-v3 (from CLI flag - highest priority)
# - device: cuda (from config file - overrides environment)
# - language: en (from config file)
# - other settings: from config file or defaults
```

## Creating Custom Configs

### Transcription Config Template

```json
{
  "model": "large-v3",              // Whisper model: tiny, base, small, medium, large, large-v2, large-v3
  "device": "cuda",                 // Device: cuda or cpu
  "compute_type": "float16",        // Precision: float16, float32, int8, int8_float16
  "language": "en",                 // Language code or null for auto-detect
  "task": "transcribe",             // Task: transcribe or translate
  "skip_existing_json": true,       // Skip files with existing JSON output
  "vad_min_silence_ms": 500,        // VAD silence threshold (ms)
  "beam_size": 5                    // Beam search size (1-10, higher = more accurate but slower)
}
```

### Enrichment Config Template

```json
{
  "skip_existing": true,                     // Skip segments with audio_state already populated
  "enable_prosody": true,                    // Extract pitch, energy, rate, pauses
  "enable_emotion": true,                    // Extract dimensional emotion (valence/arousal/dominance)
  "enable_categorical_emotion": false,       // Extract categorical emotion (slower)
  "device": "cpu",                           // Device: cpu or cuda
  "dimensional_model_name": null,            // Override default dimensional model
  "categorical_model_name": null             // Override default categorical model
}
```

## Common Patterns

### Development Workflow

```bash
# Fast iteration with tiny model
uv run slower-whisper transcribe \
  --config examples/config_examples/transcription_dev_testing.json

# Test enrichment without emotion models
uv run slower-whisper enrich \
  --config examples/config_examples/enrichment_dev_testing.json
```

### Production Deployment

```bash
# Stage 1: High-quality transcription
uv run slower-whisper transcribe \
  --config examples/config_examples/transcription_production.json \
  --root /data/production

# Stage 2: Production enrichment
uv run slower-whisper enrich \
  --config examples/config_examples/enrichment_production.json \
  --root /data/production
```

### Multi-Language Support

```bash
# Auto-detect language
uv run slower-whisper transcribe --language none

# Force Spanish
uv run slower-whisper transcribe --language es

# Use config but override language
uv run slower-whisper transcribe \
  --config transcription_basic.json \
  --language fr
```

### Resource-Constrained Environments

```bash
# CPU-only with minimal model
uv run slower-whisper transcribe \
  --model tiny \
  --device cpu \
  --compute-type int8

# Prosody-only enrichment (no GPU needed)
uv run slower-whisper enrich \
  --enable-prosody \
  --no-enable-emotion \
  --device cpu
```

## Validation

All configuration files in this directory have been validated and can be loaded without errors:

```bash
# Validate transcription config
python -c "
from transcription import TranscriptionConfig
cfg = TranscriptionConfig.from_file('examples/config_examples/transcription_basic.json')
print(f'✓ Valid: model={cfg.model}, device={cfg.device}')
"

# Validate enrichment config
python -c "
from transcription import EnrichmentConfig
cfg = EnrichmentConfig.from_file('examples/config_examples/enrichment_full.json')
print(f'✓ Valid: prosody={cfg.enable_prosody}, emotion={cfg.enable_emotion}')
"
```

## See Also

- [API Quick Reference](../../docs/API_QUICK_REFERENCE.md) - Configuration field documentation
- [README](../../README.md) - Main usage guide
- [CLAUDE.md](../../CLAUDE.md) - Developer documentation
