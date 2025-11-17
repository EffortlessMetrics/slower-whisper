# slower-whisper (ffmpeg + faster-whisper)

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Tests](https://img.shields.io/badge/tests-191%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-57%25-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-production%20ready-success)

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

## Configuration

slower-whisper supports multiple configuration methods with clear precedence rules.

### Configuration Precedence

Settings are loaded in the following order (highest to lowest priority):

```text
1. CLI flags (--model, --device, etc.)
   ↓
2. Config file (--config or --enrich-config)
   ↓
3. Environment variables (SLOWER_WHISPER_*)
   ↓
4. Defaults
```

Each layer only overrides values explicitly set. This allows flexible configuration for different environments.

### Configuration Methods

#### 1. CLI Flags (Highest Priority)

```bash
uv run slower-whisper transcribe --model large-v3 --language en --device cuda
```

#### 2. Configuration Files

```bash
# Use a JSON config file
uv run slower-whisper transcribe --config config/production.json

# Override specific values from config
uv run slower-whisper transcribe --config config/base.json --model large-v3
```

#### 3. Environment Variables

```bash
# Transcription settings (SLOWER_WHISPER_ prefix)
export SLOWER_WHISPER_MODEL=large-v3
export SLOWER_WHISPER_DEVICE=cuda
export SLOWER_WHISPER_LANGUAGE=en

# Enrichment settings (SLOWER_WHISPER_ENRICH_ prefix)
export SLOWER_WHISPER_ENRICH_ENABLE_PROSODY=true
export SLOWER_WHISPER_ENRICH_DEVICE=cuda

uv run slower-whisper transcribe  # Uses env vars
```

#### 4. Python API

```python
from transcription import TranscriptionConfig, EnrichmentConfig

# Load from file
config = TranscriptionConfig.from_file("config.json")

# Load from environment
config = TranscriptionConfig.from_env()

# Create directly
config = TranscriptionConfig(model="large-v3", language="en")
```

### Example Configurations

See [examples/config_examples/](examples/config_examples/) for complete configuration examples:

- **transcription_basic.json**: Lightweight base model setup
- **transcription_production.json**: High-quality production settings
- **transcription_dev_testing.json**: Fast testing with minimal resources
- **enrichment_full.json**: Complete audio analysis (prosody + emotion)
- **enrichment_production.json**: Optimized production enrichment

For detailed configuration documentation, see:

- [examples/config_examples/README.md](examples/config_examples/README.md) - Configuration file examples
- [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md) - Complete API reference with all config options

## Usage

The project provides both **Command-Line** and **Python API** interfaces.

### Unified CLI (Recommended)

The modern CLI uses subcommands for clarity:

#### Stage 1: Transcribe

```bash
# Basic usage with defaults
uv run slower-whisper transcribe

# Customize options
uv run slower-whisper transcribe \
  --root /path/to/project \
  --model large-v3 \
  --language en \
  --device cuda \
  --skip-existing-json
```

#### Stage 2: Enrich

```bash
# Add prosody and emotion features
uv run slower-whisper enrich

# Customize enrichment
uv run slower-whisper enrich \
  --root /path/to/project \
  --enable-prosody \
  --enable-emotion \
  --device cpu
```

#### View help

```bash
uv run slower-whisper --help
uv run slower-whisper transcribe --help
uv run slower-whisper enrich --help
```

### Python API

Use the clean programmatic interface:

#### Basic transcription

```python
from transcription import transcribe_directory, TranscriptionConfig

config = TranscriptionConfig(
    model="large-v3",
    language="en",
    device="cuda"
)

transcripts = transcribe_directory("/path/to/project", config)
print(f"Transcribed {len(transcripts)} files")
```

#### Single file transcription

```python
from transcription import transcribe_file, TranscriptionConfig

config = TranscriptionConfig(model="base", language="en")
transcript = transcribe_file(
    audio_path="interview.mp3",
    root="/path/to/project",
    config=config
)

# Access results
for segment in transcript.segments:
    print(f"[{segment.start:.2f}s] {segment.text}")
```

#### Audio enrichment

```python
from transcription import enrich_directory, EnrichmentConfig

config = EnrichmentConfig(
    enable_prosody=True,
    enable_emotion=True,
    device="cpu"
)

enriched = enrich_directory("/path/to/project", config)

# Inspect enriched features
for transcript in enriched:
    for segment in transcript.segments:
        if segment.audio_state:
            print(segment.audio_state["rendering"])
            # e.g., "[audio: high pitch, loud volume, fast speech]"
```

#### Load and save transcripts

```python
from transcription import load_transcript, save_transcript

# Load existing transcript
transcript = load_transcript("transcript.json")

# Modify and save
transcript.segments[0].text = "Corrected text"
save_transcript(transcript, "corrected.json")
```

### REST API Service

For web-based deployments, slower-whisper includes an optional FastAPI service wrapper that exposes transcription and enrichment via HTTP endpoints.

**Installation:**

```bash
# Install API dependencies
uv sync --extra api --extra full
```

**Running the service:**

```bash
# Development mode (with auto-reload)
uv run uvicorn transcription.service:app --reload --host 0.0.0.0 --port 8000

# Production mode (4 workers)
uv run uvicorn transcription.service:app --host 0.0.0.0 --port 8000 --workers 4

# Or with Docker
docker build -f Dockerfile.api -t slower-whisper:api .
docker run -p 8000:8000 slower-whisper:api

# Or with Docker Compose
docker-compose -f docker-compose.api.yml up -d
```

**Using the API:**

```bash
# Health check
curl http://localhost:8000/health

# Transcribe audio
curl -X POST -F "audio=@interview.mp3" \
  "http://localhost:8000/transcribe?model=large-v3&language=en"

# Enrich transcript
curl -X POST \
  -F "transcript=@transcript.json" \
  -F "audio=@audio.wav" \
  "http://localhost:8000/enrich?enable_prosody=true&enable_emotion=true"

# Interactive documentation
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

For complete API documentation, examples, and deployment guides, see [API_SERVICE.md](API_SERVICE.md).

### Legacy CLI (Backward Compatibility)

The old CLI still works but is deprecated:

```bash
# Old style (still supported)
uv run python transcribe_pipeline.py
uv run python audio_enrich.py

# New unified style (recommended)
uv run slower-whisper transcribe
uv run slower-whisper enrich

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

## Testing

slower-whisper includes a comprehensive test suite with **200+ passing tests** covering unit tests, integration tests, and BDD scenarios.

### Test Suite Overview

The project uses a multi-layered testing strategy:

**Unit Tests (pytest)**: Test individual modules and functions in isolation

- JSON schema validation and I/O operations
- Audio processing utilities
- Configuration parsing and precedence
- Data model validation

**Integration Tests**: End-to-end pipeline validation

- Full transcription workflow
- Audio enrichment with prosody and emotion
- API endpoints and service layer
- CLI command execution

**BDD Scenarios (pytest-bdd + Gherkin)**: User-focused acceptance tests written in natural language

- 15 feature scenarios across transcription and enrichment workflows
- See [tests/features/transcription.feature](tests/features/transcription.feature) and [tests/features/enrichment.feature](tests/features/enrichment.feature)
- Example scenario:

  ```gherkin
  Scenario: Transcribe with custom model configuration
    Given a project with a mono WAV file named "test.wav"
    When I transcribe the project with model "base" and language "en"
    Then a transcript JSON exists for "test.wav"
    And the transcript language is "en"
    And the transcript metadata contains model "base"
  ```

### Test Coverage

Current test coverage: **56% overall**, with high coverage on core modules:

- `transcription/writers.py`: **100%** (JSON/TXT/SRT output)
- `transcription/models.py`: **95%** (data models and schema)
- `transcription/config.py`: **90%** (configuration system)
- `transcription/pipeline.py`: **85%** (transcription orchestration)
- `transcription/prosody.py`: **80%** (prosody extraction)

### Running Tests

```bash
# Install dev dependencies (includes pytest and other testing tools)
uv sync --extra dev

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=transcription --cov-report=term-missing

# Run BDD scenarios only
uv run pytest tests/features/

# Run specific test categories
uv run pytest -m "not slow"              # Skip slow tests
uv run pytest -m "not requires_gpu"      # Skip GPU-dependent tests
uv run pytest tests/test_prosody.py      # Run specific test file
```

Tests are not required for running the pipeline but are essential for contributors and those extending the codebase.

## Contributing

Contributions are welcome! Whether you're fixing bugs, adding features, improving documentation, or helping with issues, your help makes this project better.

### Quick Start for Contributors

```bash
# 1. Fork and clone
git clone https://github.com/<your-fork>/slower-whisper.git
cd slower-whisper

# 2. Install dev dependencies
uv sync --extra dev

# 3. Run tests to verify setup
uv run pytest -m "not slow"

# 4. Run linting to check code quality
uv run ruff check .
```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Add or update tests for any behavior changes
   - Follow existing code style and patterns
   - Update documentation if needed

3. **Run quality checks**
   ```bash
   # Format and lint
   uv run ruff format .
   uv run ruff check .

   # Type check (optional but recommended)
   uv run mypy transcription/

   # Run tests
   uv run pytest
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: brief description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Submit a PR against the `main` branch
   - Describe what changed and why
   - Link any related issues

### Guidelines

- **Use feature branches** and submit Pull Requests against `main`
- **Add tests** for new functionality or bug fixes
- **Run quality checks** before pushing (`uv run pytest && uv run ruff check .`)
- **For larger changes**, open an issue or discussion first to align on direction
- **Follow code style**: We use ruff for linting and formatting (configured in `pyproject.toml`)
- **Write clear commit messages** using conventional commits format when possible

### Code Quality Tools

```bash
# Format code with ruff
uv run ruff format transcription/ tests/

# Lint with ruff (with auto-fix)
uv run ruff check --fix transcription/ tests/

# Type check with mypy
uv run mypy transcription/
```

### Running Tests

```bash
# Run all fast tests
uv run pytest -m "not slow"

# Run with coverage
uv run pytest --cov=transcription --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_models.py -v
```

For detailed guidelines, coding standards, and the full development workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Deployment

slower-whisper supports multiple deployment options from local CLI to production Kubernetes clusters. Choose the deployment method that fits your scale and infrastructure.

### Deployment Options

#### Local CLI (Simplest)

Run directly on your machine with Python and ffmpeg:

```bash
# Install and run
uv sync
uv run slower-whisper transcribe
```

**Best for**: Development, small-scale processing, testing

---

#### Docker - CPU

Containerized deployment without GPU requirements:

```bash
# Build and run
docker build -t slower-whisper:cpu .
docker run --rm -v $(pwd)/data:/app/data slower-whisper:cpu
```

**Best for**: Reproducible environments, CI/CD, CPU-only servers

---

#### Docker - GPU

GPU-accelerated containerized deployment:

```bash
# Build GPU image
docker build -f Dockerfile.gpu -t slower-whisper:gpu .

# Run with GPU access
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  slower-whisper:gpu \
  slower-whisper transcribe --model large-v3 --device cuda
```

**Best for**: High-throughput processing, production workloads, large models

---

#### Docker Compose - Batch Processing

Multi-service orchestration for continuous batch processing:

```bash
# Start batch processor
docker-compose up -d batch-processor

# Monitor logs
docker-compose logs -f batch-processor

# Stop when done
docker-compose down
```

**Configuration**: Edit `docker-compose.yml` or use `.env` file for settings.

**Best for**: Automated batch workflows, scheduled processing, multi-stage pipelines

---

#### Kubernetes - Production Scale

Cloud-native deployment with autoscaling and resource management:

```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n slower-whisper

# Run batch job
kubectl apply -f k8s/job.yaml
```

**Features**:

- Horizontal pod autoscaling
- GPU node scheduling
- Persistent volume claims for data
- ConfigMaps for environment-specific settings
- CronJobs for scheduled processing

**Best for**: Enterprise deployments, cloud infrastructure, high availability, multi-tenant environments

---

#### FastAPI REST Service

HTTP API for web-based integrations:

```bash
# Install API dependencies
uv sync --extra api --extra full

# Run service
uv run uvicorn transcription.service:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -f Dockerfile.api -t slower-whisper:api .
docker run -p 8000:8000 slower-whisper:api
```

**Endpoints**:

- `POST /transcribe` - Transcribe audio files
- `POST /enrich` - Enrich existing transcripts
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

**Best for**: Web applications, microservices architecture, API-first integrations

---

### Deployment Resources

For detailed deployment guides and infrastructure-as-code configurations:

- **[DOCKER.md](DOCKER.md)** - Complete Docker setup guide with GPU support
- **[k8s/](k8s/)** - Kubernetes manifests and Kustomize configurations
- **[docker-compose.yml](docker-compose.yml)** - Production Docker Compose setup
- **[docker-compose.dev.yml](docker-compose.dev.yml)** - Development environment
- **[API_SERVICE.md](API_SERVICE.md)** - REST API service documentation

### Deployment Configuration

All deployment methods support the same configuration options via:

1. **Environment variables** (`SLOWER_WHISPER_*`)
2. **Config files** (JSON, mounted as volumes in Docker/K8s)
3. **CLI flags** (for one-off overrides)

See the [Configuration](#configuration) section above for precedence rules and examples.

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
- **[Roadmap](ROADMAP.md)**: Future features and development plans
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute to the project
- **[Changelog](CHANGELOG.md)**: Version history and updates
- **[Documentation Index](docs/INDEX.md)**: Complete documentation map

## License

Apache License 2.0 - see LICENSE file for details.
