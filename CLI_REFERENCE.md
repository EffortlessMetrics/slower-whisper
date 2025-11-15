# CLI Reference

This document describes the command-line interfaces (CLIs) provided by the `slower-whisper` package.

---

## Overview

The package provides two primary CLI commands:

1. **`slower-whisper`** - Main transcription pipeline (Stage 1)
2. **`slower-whisper-enrich`** - Audio enrichment pipeline (Stage 2)

Both commands are installed as executable scripts when you install the package with `pip` or `uv`.

---

## Installation & Entry Points

### Entry Point Configuration

The package defines the following entry points in `pyproject.toml`:

```toml
[project.scripts]
slower-whisper = "transcription.cli:main"
slower-whisper-enrich = "transcription.audio_enrich_cli:main"
```

### How to Install

**Option 1: Using uv (recommended)**

```bash
# Basic install (transcription only)
uv sync

# Full install (with audio enrichment)
uv sync --extra full

# Development install
uv sync --extra dev
```

**Option 2: Using pip**

```bash
# Basic install
pip install -e .

# Full install
pip install -e ".[full]"

# Development install
pip install -e ".[dev]"
```

### Verifying Installation

After installation, verify the commands are available:

```bash
# Check if commands are in PATH
which slower-whisper
which slower-whisper-enrich

# Or on Windows:
where slower-whisper
where slower-whisper-enrich

# Test help output
slower-whisper --help
slower-whisper-enrich --help
```

---

## Command 1: `slower-whisper`

**Purpose:** Transcribe audio files using faster-whisper (Stage 1 pipeline)

**Entry Point:** `transcription.cli:main`

**Module:** `/home/steven/code/Python/slower-whisper/transcription/cli.py`

### Usage

```bash
slower-whisper [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--root` | Path | `.` (current directory) | Root directory containing `raw_audio/` etc. |
| `--model` | str | `large-v3` | Whisper model name |
| `--device` | str | `cuda` | Device for inference (`cuda` or `cpu`) |
| `--compute-type` | str | `float16` | Compute type for faster-whisper (e.g., `float16`, `int8_float16`) |
| `--language` | str | `None` (auto-detect) | Force language code (e.g., `en`, `es`) |
| `--task` | str | `transcribe` | Task to perform (`transcribe` or `translate`) |
| `--vad-min-silence-ms` | int | `500` | Minimum silence duration in ms to split segments |
| `--beam-size` | int | `5` | Beam size for decoding |
| `--skip-existing-json` | flag | `False` | Skip transcription for files that already have JSON output |

### Examples

**Basic transcription (auto-detect language):**
```bash
slower-whisper
```

**Force English and use CPU:**
```bash
slower-whisper --language en --device cpu
```

**Use lighter model with quantization:**
```bash
slower-whisper --model medium --compute-type int8_float16
```

**Skip already transcribed files:**
```bash
slower-whisper --skip-existing-json
```

**Custom root directory:**
```bash
slower-whisper --root /path/to/project
```

**Translate to English instead of transcribe:**
```bash
slower-whisper --task translate --language es
```

### Directory Structure

The command expects/creates this structure under `--root`:

```
root/
  raw_audio/          # Input: Place your audio files here
  input_audio/        # Generated: Normalized 16kHz mono WAV files
  transcripts/        # Generated: .txt and .srt files
  whisper_json/       # Generated: Structured JSON transcripts
```

### Output Files

For each audio file (e.g., `meeting.mp3`), the pipeline generates:

1. **`input_audio/meeting.wav`** - Normalized audio (16kHz mono)
2. **`transcripts/meeting.txt`** - Plain text transcript with timestamps
3. **`transcripts/meeting.srt`** - Subtitle file (SRT format)
4. **`whisper_json/meeting.json`** - Structured JSON (canonical format)

### Alternative Invocations

If you haven't installed the package, you can run directly:

```bash
# Using the convenience script
python transcribe_pipeline.py [OPTIONS]

# Or using the module directly
python -m transcription.cli [OPTIONS]

# Or with uv run
uv run slower-whisper [OPTIONS]
```

---

## Command 2: `slower-whisper-enrich`

**Purpose:** Enrich existing transcripts with audio features (Stage 2 pipeline)

**Entry Point:** `transcription.audio_enrich_cli:main`

**Module:** `/home/steven/code/Python/slower-whisper/transcription/audio_enrich_cli.py`

### Usage

```bash
slower-whisper-enrich [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--root` | Path | `.` (current directory) | Root directory containing `whisper_json/` and `input_audio/` |
| `--skip-existing` | flag | `False` | Skip JSON files that already have `audio_state` |
| `--enable-prosody` | flag | `True` | Enable prosody feature extraction |
| `--no-enable-prosody` | flag | - | Disable prosody feature extraction |
| `--enable-emotion` | flag | `True` | Enable dimensional emotion analysis |
| `--no-enable-emotion` | flag | - | Disable dimensional emotion analysis |
| `--enable-categorical-emotion` | flag | `False` | Enable categorical emotion classification (slower) |
| `--device` | str | `cuda` | Device for emotion models (`cuda` or `cpu`) |
| `--file` | str | `None` | Enrich a single JSON file instead of batch processing |

### Examples

**Batch enrich all transcripts (default: prosody + dimensional emotion):**
```bash
slower-whisper-enrich
```

**Enrich single file:**
```bash
slower-whisper-enrich --file whisper_json/meeting1.json
```

**Enable categorical emotion (slower but more detailed):**
```bash
slower-whisper-enrich --enable-categorical-emotion
```

**Prosody only (no emotion):**
```bash
slower-whisper-enrich --no-enable-emotion
```

**Emotion only (no prosody):**
```bash
slower-whisper-enrich --no-enable-prosody
```

**Skip already enriched files:**
```bash
slower-whisper-enrich --skip-existing
```

**Use CPU instead of GPU:**
```bash
slower-whisper-enrich --device cpu
```

### Prerequisites

Before running `slower-whisper-enrich`:

1. Run `slower-whisper` to generate transcripts
2. Install enrichment dependencies:
   ```bash
   uv sync --extra full
   # or
   pip install -e ".[full]"
   ```

### Directory Structure

The command expects this structure under `--root`:

```
root/
  whisper_json/       # Input: JSON transcripts from Stage 1
  input_audio/        # Input: WAV files from Stage 1
```

### Output

Enrichment updates the JSON files in-place, adding `audio_state` to each segment:

**Before enrichment:**
```json
{
  "id": 0,
  "start": 0.0,
  "end": 4.2,
  "text": "Okay, let's get started.",
  "audio_state": null
}
```

**After enrichment:**
```json
{
  "id": 0,
  "start": 0.0,
  "end": 4.2,
  "text": "Okay, let's get started.",
  "audio_state": {
    "prosody": {
      "pitch": {"level": "high", "mean_hz": 245.3, "std_hz": 32.1, "contour": "rising"},
      "energy": {"level": "loud", "db_rms": -8.2},
      "rate": {"level": "normal", "syllables_per_sec": 5.3}
    },
    "emotion": {
      "dimensional": {
        "valence": {"level": "positive", "score": 0.72},
        "arousal": {"level": "high", "score": 0.68}
      }
    }
  }
}
```

### Alternative Invocations

If you haven't installed the package, you can run directly:

```bash
# Using the convenience script
python audio_enrich.py [OPTIONS]

# Or using the module directly
python -m transcription.audio_enrich_cli [OPTIONS]

# Or with uv run
uv run slower-whisper-enrich [OPTIONS]
```

---

## Legacy Scripts

The repository also contains convenience wrapper scripts at the root level:

### `transcribe_pipeline.py`

Wrapper script that delegates to `transcription.cli:main`.

**Usage:**
```bash
python transcribe_pipeline.py [OPTIONS]
```

**Source:**
```python
from transcription.cli import main

if __name__ == "__main__":
    main()
```

### `audio_enrich.py`

Wrapper script that delegates to `transcription.audio_enrich_cli:main`.

**Usage:**
```bash
python audio_enrich.py [OPTIONS]
```

**Source:**
```python
from transcription.audio_enrich_cli import main

if __name__ == "__main__":
    main()
```

These scripts are useful for:
- Running without installing the package
- Development and testing
- Users who prefer `python script.py` over `command` invocation

---

## Complete Workflow Example

Here's a typical workflow using both commands:

```bash
# Step 1: Install the package
uv sync --extra full

# Step 2: Prepare your audio
mkdir -p raw_audio
cp /path/to/meeting.mp3 raw_audio/

# Step 3: Transcribe (Stage 1)
slower-whisper --language en --skip-existing-json

# Step 4: Enrich with audio features (Stage 2)
slower-whisper-enrich --enable-categorical-emotion

# Step 5: View results
cat whisper_json/meeting.json
cat transcripts/meeting.txt
```

---

## Development & Testing

### Running from Source (without installation)

```bash
# Transcription
python transcribe_pipeline.py --help
python -m transcription.cli --help

# Enrichment
python audio_enrich.py --help
python -m transcription.audio_enrich_cli --help
```

### Using uv run (recommended for development)

```bash
# Transcription
uv run slower-whisper --help
uv run python transcribe_pipeline.py --help

# Enrichment
uv run slower-whisper-enrich --help
uv run python audio_enrich.py --help
```

### Testing Entry Points

Verify the entry points are correctly configured:

```bash
# Check entry point syntax
python3 -c "from transcription.cli import main; print('✓ cli:main accessible')"
python3 -c "from transcription.audio_enrich_cli import main; print('✓ audio_enrich_cli:main accessible')"

# Simulate package installation (development mode)
pip install -e .

# Test installed commands
slower-whisper --help
slower-whisper-enrich --help
```

---

## Troubleshooting

### Command not found after installation

**Problem:** `slower-whisper: command not found`

**Solution:**
```bash
# Ensure the package is installed
pip list | grep slower-whisper

# Check if the script is in your PATH
which slower-whisper

# If using virtual environment, make sure it's activated
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Entry point doesn't work

**Problem:** Entry point exists but throws import errors

**Solution:**
```bash
# Reinstall the package
pip install -e . --force-reinstall

# Or with uv
uv sync --reinstall-package slower-whisper
```

### Dependencies missing

**Problem:** `ModuleNotFoundError: No module named 'faster_whisper'`

**Solution:**
```bash
# For transcription
uv sync
# or
pip install -e .

# For enrichment
uv sync --extra full
# or
pip install -e ".[full]"
```

---

## Additional Resources

- **[README.md](README.md)** - Project overview and quick start
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Detailed getting started guide
- **[docs/AUDIO_ENRICHMENT.md](docs/AUDIO_ENRICHMENT.md)** - Audio enrichment features
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **[pyproject.toml](pyproject.toml)** - Package configuration and dependencies
