# slower-whisper (ffmpeg + faster-whisper)

This project provides a small, structured codebase for running a fully local
transcription pipeline on a machine with an NVIDIA GPU.

It:

- Normalizes audio in `raw_audio/` to 16 kHz mono WAV in `input_audio/` using `ffmpeg`.
- Transcribes using Whisper via `faster-whisper` on CUDA.
- Writes:
  - `transcripts/<name>.txt` – timestamped text.
  - `transcripts/<name>.srt` – subtitles.
  - `whisper_json/<name>.json` – structured JSON for analysis.

JSON is the canonical output format and is designed to be extended later
with tone, speaker, and other annotations.

## Quick Start

Get started in 3 steps:

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Install dependencies (choose your level)
uv sync              # Basic transcription only (~2.5GB)
# or: uv sync --extra full  # Full audio enrichment (~6.5GB total)

# 3. Run transcription
uv run slower-whisper
```

Place your audio files in `raw_audio/` and find transcripts in `whisper_json/`, `transcripts/`.

See detailed instructions below for setup, configuration, and advanced features.

## Requirements

- Windows, Linux, or macOS (any OS with Python and ffmpeg).
- Python 3.10+.
- NVIDIA GPU with a recent CUDA-capable driver (for GPU acceleration).
- `ffmpeg` on PATH.
- [uv](https://docs.astral.sh/uv/) for package management (recommended).

### Installing System Dependencies

**Install ffmpeg:**

- **Windows (PowerShell, elevated):**
  ```powershell
  choco install ffmpeg -y
  ```
- **macOS:**
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```

**Install uv (recommended package manager):**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or visit [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Installing Python Dependencies

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
| **enrich-basic** | +1GB | soundfile, numpy, librosa | Basic prosody features |
| **enrich-prosody** | +36MB | Praat/Parselmouth | Research-grade pitch analysis |
| **emotion** | +4GB | torch, transformers | Emotion recognition |
| **full** | +4GB total | All enrichment features | Complete audio analysis |
| **dev** | +full + tools | Testing, linting, docs | Contributing to project |

**Install specific groups:**

```bash
# Just prosody (no emotion)
uv sync --extra enrich-prosody

# Just emotion (no prosody)
uv sync --extra emotion

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

## Usage

### Quick Start

From the project directory:

```bash
uv run slower-whisper
```

Or using the direct script:

```bash
uv run python transcribe_pipeline.py
```

This uses defaults:

- root: current directory
- model: `large-v3`
- device: `cuda`
- compute type: `float16`
- VAD min silence: 500 ms
- language: auto-detect
- task: transcribe

### Command-Line Options

You can override defaults with CLI options:

```bash
# Force English and skip files already transcribed
uv run slower-whisper --language en --skip-existing-json

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

## JSON schema

Each JSON file looks like:

```json
{
  "schema_version": 2,
  "file": "meeting1.wav",
  "language": "en",
  "meta": {
    "generated_at": "2025-11-15T03:21:00Z",
    "audio_file": "meeting1.wav",
    "audio_duration_sec": 3120.5,
    "model_name": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "beam_size": 5,
    "vad_min_silence_ms": 500,
    "language_hint": "en",
    "task": "transcribe",
    "pipeline_version": "1.0.0",
    "root": "C:/transcription_toolkit"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Okay, let's get started with today's agenda.",
      "speaker": null,
      "tone": null,
      "audio_state": null
    }
  ]
}
```

This schema is stable and is intended to be the basis for future tooling:

- Tone tagging: populate `tone`.
- Speaker diarization: populate `speaker`.
- Audio enrichment: populate `audio_state` with prosody and emotion features.
- Search and analysis: operate over `segments[]`.
- Run-level analysis and reproducibility: read `meta`.

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
# Enrich using the CLI tool
uv run slower-whisper-enrich whisper_json/meeting1.json input_audio/meeting1.wav
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

## Running Tests

The project includes a comprehensive test suite for validating the JSON schema, SRT formatting, and audio enrichment features.

### Running Tests with uv

```bash
# Install dev dependencies (includes pytest and other testing tools)
uv sync --extra dev

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=transcription --cov-report=term-missing

# Run specific test categories
uv run pytest -m "not slow"              # Skip slow tests
uv run pytest -m "not requires_gpu"      # Skip GPU-dependent tests
uv run pytest tests/test_prosody.py      # Run specific test file
```

### Test Organization

The test suite covers:

- **JSON schema validation**: Ensures `write_json` produces the documented structure
- **SRT formatting**: Validates timestamp formatting and subtitle generation
- **Audio enrichment**: Tests prosody and emotion extraction features
- **Integration tests**: End-to-end pipeline validation

Tests are not required for running the pipeline but are useful if you're contributing or extending the code.

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/slower-whisper.git
cd slower-whisper

# Install with dev dependencies
uv sync --extra dev

# Run tests to verify setup
uv run pytest
```

### Code Quality Tools

The project uses modern Python tooling for code quality:

```bash
# Format code with black
uv run black transcription/ tests/

# Sort imports with isort
uv run isort transcription/ tests/

# Lint with ruff
uv run ruff check transcription/ tests/

# Type check with mypy
uv run mypy transcription/

# Run all quality checks
uv run black . && uv run isort . && uv run ruff check . && uv run mypy transcription/
```

### Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Run tests and quality checks
4. Submit a pull request

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
- **[Audio Enrichment Guide](docs/AUDIO_ENRICHMENT.md)**: Deep dive into prosody and emotion features
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute to the project
- **[Changelog](CHANGELOG.md)**: Version history and updates

## License

Apache License 2.0 - see LICENSE file for details.
