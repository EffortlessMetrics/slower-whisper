# Configuration Examples - Quick Index

## Quick Start

1. **New to configuration?** Start with [README.md](README.md) - Configuration Precedence section
2. **Need field reference?** See [../../docs/API_QUICK_REFERENCE.md](../../docs/API_QUICK_REFERENCE.md) - Configuration section
3. **Want to see it in action?** Run `python config_demo.py`

## Configuration Files in This Directory

### Transcription Configurations

| File | Model | Device | Use Case | When to Use |
|------|-------|--------|----------|-------------|
| [transcription_basic.json](transcription_basic.json) | base | cuda | Quick transcription | Limited resources, fast iteration |
| [transcription_full_enrichment.json](transcription_full_enrichment.json) | large-v3 | cuda | Best quality | Production transcription for enrichment |
| [transcription_production.json](transcription_production.json) | large-v3 | cuda | Production | High-quality production deployment |
| [transcription_dev_testing.json](transcription_dev_testing.json) | tiny | cpu | Development | CI/CD, testing, no GPU |

### Enrichment Configurations

| File | Prosody | Emotion | Categorical | Device | Use Case |
|------|---------|---------|-------------|--------|----------|
| [enrichment_basic.json](enrichment_basic.json) | ✓ | ✗ | ✗ | cpu | Lightweight prosody only |
| [enrichment_full.json](enrichment_full.json) | ✓ | ✓ | ✓ | cuda | Complete audio analysis |
| [enrichment_production.json](enrichment_production.json) | ✓ | ✓ | ✗ | cuda | Optimized production |
| [enrichment_dev_testing.json](enrichment_dev_testing.json) | ✓ | ✗ | ✗ | cpu | Fast development testing |

## How to Use

### With CLI

```bash
# Transcription
uv run slower-whisper transcribe --config examples/config_examples/transcription_production.json

# Enrichment
uv run slower-whisper enrich --config examples/config_examples/enrichment_full.json

# Override specific values
uv run slower-whisper transcribe \
  --config examples/config_examples/transcription_basic.json \
  --model large-v3
```

### With Python API

```python
from transcription import TranscriptionConfig, EnrichmentConfig, transcribe_directory

# Load config and use
config = TranscriptionConfig.from_file("examples/config_examples/transcription_production.json")
transcripts = transcribe_directory("/path/to/project", config)
```

## Configuration Precedence

Remember the order:

```
CLI flags > Config file > Environment variables > Defaults
```

Example:
```bash
# This uses:
# - model: base (from CLI - highest priority)
# - device: cuda (from config file)
# - beam_size: from config file or default
uv run slower-whisper transcribe \
  --config transcription_production.json \
  --model base
```

## Interactive Demo

Run the demo script to see all configuration methods:

```bash
# Show all demos
python config_demo.py

# Show specific demo
python config_demo.py --demo defaults      # Show default values
python config_demo.py --demo file          # Load from files
python config_demo.py --demo env           # Load from environment
python config_demo.py --demo precedence    # See precedence in action
python config_demo.py --demo create        # Create custom configs
```

## Common Scenarios

### Scenario 1: Development Workflow

```bash
# Fast iteration with minimal resources
uv run slower-whisper transcribe \
  --config examples/config_examples/transcription_dev_testing.json

uv run slower-whisper enrich \
  --config examples/config_examples/enrichment_dev_testing.json
```

### Scenario 2: Production Deployment

```bash
# High-quality transcription + optimized enrichment
uv run slower-whisper transcribe \
  --config examples/config_examples/transcription_production.json \
  --root /data/production

uv run slower-whisper enrich \
  --config examples/config_examples/enrichment_production.json \
  --root /data/production
```

### Scenario 3: Multi-Language Processing

```bash
# Use production config but override language
uv run slower-whisper transcribe \
  --config examples/config_examples/transcription_production.json \
  --language es  # Spanish
```

### Scenario 4: Resource-Constrained Environment

```bash
# CPU-only with minimal model
uv run slower-whisper transcribe \
  --config examples/config_examples/transcription_dev_testing.json

# Prosody-only enrichment (no GPU/torch needed)
uv run slower-whisper enrich \
  --enable-prosody \
  --no-enable-emotion \
  --device cpu
```

## Environment Variables

Quick reference for all environment variables:

### Transcription
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

### Enrichment
```bash
export SLOWER_WHISPER_ENRICH_SKIP_EXISTING=true
export SLOWER_WHISPER_ENRICH_ENABLE_PROSODY=true
export SLOWER_WHISPER_ENRICH_ENABLE_EMOTION=true
export SLOWER_WHISPER_ENRICH_ENABLE_CATEGORICAL_EMOTION=false
export SLOWER_WHISPER_ENRICH_DEVICE=cuda
```

## Documentation Links

- **Main README**: [../../README.md](../../README.md) - Configuration section
- **API Reference**: [../../docs/API_QUICK_REFERENCE.md](../../docs/API_QUICK_REFERENCE.md) - Complete field documentation
- **This Directory**: [README.md](README.md) - Detailed configuration guide
- **Demo Script**: [config_demo.py](config_demo.py) - Interactive demonstrations

## Creating Custom Configurations

1. Copy an existing config that's closest to your needs
2. Modify the fields you want to change
3. Save with a descriptive name (e.g., `my_custom_config.json`)
4. Use with `--config` flag

Example:
```bash
# Copy and customize
cp transcription_production.json my_config.json
# Edit my_config.json to change model to "medium"

# Use it
uv run slower-whisper transcribe --config my_config.json
```

## Validation

Validate your config files before using:

```bash
# Validate transcription config
python -c "
from transcription import TranscriptionConfig
cfg = TranscriptionConfig.from_file('my_config.json')
print(f'✓ Valid: {cfg.model} on {cfg.device}')
"

# Validate enrichment config
python -c "
from transcription import EnrichmentConfig
cfg = EnrichmentConfig.from_file('my_enrich_config.json')
print(f'✓ Valid: prosody={cfg.enable_prosody}, emotion={cfg.enable_emotion}')
"
```

## Need Help?

- Configuration not working? Check [README.md](README.md) for precedence rules
- Need field details? See [../../docs/API_QUICK_REFERENCE.md](../../docs/API_QUICK_REFERENCE.md)
- Want examples? Run `python config_demo.py`
