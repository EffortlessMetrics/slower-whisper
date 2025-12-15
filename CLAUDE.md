# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**slower-whisper** is a **local-first conversation signal engine** that turns audio into **LLM-ready structured data**.

Unlike traditional transcription tools that output plain text, slower-whisper produces a rich, versioned JSON format capturing:

- **What was said** (text, segment-level timestamps; word-level alignment planned)
- **Who said it** (speaker diarization, turn structure) — *v1.1 priority*
- **How it was said** (prosody, emotion, interaction patterns)
- **Semantic context** (chunk summaries, intent tags) — *v1.3 optional*

### Positioning

slower-whisper is **infrastructure, not a product**:

- **Not** "another transcription app" or meeting bot
- **Not** a cloud API or SaaS platform
- **Is** local-first, open-source conversation intelligence infrastructure
- **Is** "OpenTelemetry for audio conversations"

See [VISION.md](VISION.md) for strategic positioning and [ROADMAP.md](ROADMAP.md) for development timeline.

### Layered Architecture (L0-L4)

- **L0: Ingestion** — audio normalization, chunking, hashing
- **L1: ASR (Whisper)** — fast, deterministic transcription via faster-whisper
- **L2: Acoustic Enrichment** — speaker diarization, prosody, emotion, turns (modular, cacheable)
- **L3: Semantic Enrichment** — optional local SLM annotations (chunk summaries, intent tags)
- **L4: Task Outputs** — meeting notes, QA, coaching (consumer applications)

**Key principle**: Each layer adds value without blocking earlier layers. All layers independently cacheable and resumable.

## Development Environment (Nix-First)

**Recommended approach:** Use **Nix** for reproducible development and CI environments.

### Nix Setup (Recommended)

```bash
# Enter Nix dev shell (provides ffmpeg, Python, system deps)
nix develop

# Install Python dependencies
uv sync --extra full --extra diarization --extra dev

# Run local CI checks (mirrors GitHub Actions)
nix flake check
```

**Why Nix?**
- ✅ **Reproducible builds** - Same environment on WSL, NixOS, macOS, CI
- ✅ **Local CI** - `nix flake check` runs identical tests to GitHub Actions
- ✅ **No "works on my machine"** - Isolated, versioned system dependencies
- ✅ **Optional direnv** - Auto-activate shell on `cd` into repo

See [docs/DEV_ENV_NIX.md](docs/DEV_ENV_NIX.md) for complete setup.

### Fallback: Traditional Setup

> ⚠️ **Only use this if Nix installation is blocked.** Traditional setup works but lacks reproducibility guarantees.

Install system dependencies manually (ffmpeg, libsndfile) via apt/brew/choco, then use uv for Python packages.

## Package Management

This project uses **uv** (Astral's fast Python package manager) with `pyproject.toml`:

```bash
# Install base dependencies (transcription only)
uv sync

# Install with audio enrichment features
uv sync --extra full

# Install development dependencies (includes testing, linting, docs)
uv sync --extra dev
```

**Dependency groups:**
- **base** (default): faster-whisper only (~2.5GB)
- **enrich-basic**: soundfile, numpy, librosa (~1GB additional)
- **enrich-prosody**: adds praat-parselmouth for research-grade pitch analysis
- **emotion**: torch, transformers for emotion recognition (~4GB additional)
- **full**: all enrichment features (prosody + emotion)
- **dev**: full + testing/linting/docs tools

Alternative pip installation: `pip install -e ".[dev]"`

## Development Priorities (v1.1 Focus)

**Current Status**: v1.0.0 shipped (production-ready transcription + basic enrichment)

**Next Milestone**: v1.1.0 — Speaker & Schema Lock-In (Q1 2026)

### High-Priority Features

1. **Speaker Diarization (L2)**
   - Integrate WhisperX-style diarization (Whisper + pyannote.audio)
   - Populate `segment.speaker = {id, confidence, alternatives}`
   - Build global `speakers[]` table
   - Module: `transcription/diarization.py`

2. **Turn Structure (L2)**
   - Group contiguous segments by speaker into `turns[]`
   - Add turn-level metadata (question_count, interruptions)
   - Module: `transcription/turns.py`

3. **Speaker Stats (L2)**
   - Per-speaker aggregates (talk_time, num_turns, interruptions)
   - Sentiment summaries per speaker
   - Module: extends `transcription/turns.py`

4. **Schema v2 Finalization**
   - Lock in core fields contract (no meaning changes within v2.x)
   - Create JSON Schema (draft-07) validation file
   - Document deprecation policy

5. **Evaluation Harness**
   - Build testbed with 50-200 labeled segments
   - Task-level tests: speaker-aware summarization, action items, conflict detection
   - LLM-as-judge scoring: text vs text+speaker vs text+speaker+stats
   - Module: `benchmarks/eval_speakers.py`

### Design Constraints

**Schema Stability**:
- Core fields (`audio`, `meta`, `speakers`, `segments`, `turns`) are **locked** in v2
- Only add optional fields; never remove or rename core fields
- Breaking changes require v3 and migration tooling

**Modularity**:
- L2 features are opt-in via config flags (`--enable-diarization`)
- Never re-run ASR when enrichment changes
- Each feature caches independently by hash

**Local-First**:
- No cloud dependencies at runtime
- All models run locally (pyannote, wav2vec2, etc.)
- Only model weights download (one-time, HuggingFace cache)

### When Building v1.1 Features

**DO**:
- Add new optional fields to `Segment` or `Transcript` dataclasses
- Write BDD scenarios for new behaviors
- Update schema documentation
- Provide LLM-friendly "views" (e.g., `to_turn_view()`)
- Cache new passes separately by config hash

**DON'T**:
- Change meaning of existing fields
- Remove or rename core schema fields
- Add cloud API dependencies
- Break backward compatibility with v1.0 JSON

**Test Rigor**:
- BDD scenario for correct speaker counts
- Synthetic 2-speaker audio → validate diarization
- Real-world confusion matrix (AMI Meeting Corpus subset)
- LLM-as-judge sanity check (speaker labeling consistency)

See [ROADMAP.md](ROADMAP.md) for complete v1.1 spec and deliverables.

## Common Commands

### Building and Running

```bash
# Run transcription pipeline
uv run slower-whisper transcribe

# Run with specific options
uv run slower-whisper transcribe --model large-v3 --language en --device cuda

# Run audio enrichment (Stage 2)
uv run slower-whisper enrich

# Legacy entry points (still work)
uv run python transcribe_pipeline.py
uv run python audio_enrich.py
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=transcription --cov-report=term-missing

# Run specific test categories
uv run pytest -m "not slow"              # Skip slow tests
uv run pytest -m "not requires_gpu"      # Skip GPU tests
uv run pytest tests/test_prosody.py      # Run specific file

# Run single test
uv run pytest tests/test_models.py::test_segment_creation -v
```

### Code Quality

```bash
# Format code with ruff
uv run ruff format transcription/ tests/

# Lint with ruff
uv run ruff check transcription/ tests/

# Auto-fix linting issues
uv run ruff check --fix transcription/ tests/

# Type check with mypy
uv run mypy transcription/

# Run pre-commit hooks manually
uv run pre-commit run --all-files
```

### Pre-commit Hooks

Pre-commit hooks are configured in `.pre-commit-config.yaml`:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files
```

Hooks run automatically on `git commit`:
- ruff (linter with auto-fix)
- ruff-format (formatter)
- trailing-whitespace, end-of-file-fixer
- check-yaml, check-json, check-toml
- mypy (type checking, non-blocking)

## Code Architecture

### Two-Stage Pipeline Design

The project is architected around a **strict separation** between transcription (Stage 1) and audio enrichment (Stage 2):

**Stage 1: Transcription Pipeline** (`transcription/pipeline.py`)
1. Audio normalization (ffmpeg: 16 kHz mono WAV)
2. Transcription (faster-whisper on GPU)
3. Output: JSON, TXT, SRT files

**Stage 2: Audio Enrichment** (optional, `transcription/audio_enrichment.py`)
1. Load existing transcripts
2. Extract prosodic features from WAV
3. Extract emotional features from WAV
4. Populate `audio_state` field in JSON
5. Generate text rendering: `[audio: high pitch, loud volume, fast speech]`

### Core Modules

**Data Models** (`transcription/models.py`)
- `Segment`: Single transcribed segment with optional `audio_state` field
- `Transcript`: Complete transcript with metadata
- `SCHEMA_VERSION = 2`: Current JSON schema version
- `AUDIO_STATE_VERSION = "1.0.0"`: Audio feature schema version

**Configuration** (`transcription/config.py`)
- `TranscriptionConfig`: Public API for Stage 1 settings
- `EnrichmentConfig`: Public API for Stage 2 settings
- `Paths`, `AsrConfig`, `AppConfig`: Legacy internal configs (backward compatibility)

**Public API** (`transcription/api.py`)
- `transcribe_directory()`: Transcribe all audio files in a project
- `transcribe_file()`: Transcribe single audio file
- `enrich_directory()`: Enrich all transcripts with audio features
- `enrich_transcript()`: Enrich single transcript
- `load_transcript()`: Load transcript from JSON
- `save_transcript()`: Save transcript to JSON

**Audio Processing**
- `audio_io.py`: ffmpeg-based audio normalization
- `audio_utils.py`: Memory-efficient WAV segment extraction
- `asr_engine.py`: faster-whisper wrapper

**Audio Enrichment**
- `prosody.py`: Pitch, energy, speech rate, pause extraction (Praat/Parselmouth, librosa)
- `emotion.py`: Emotion recognition using pre-trained wav2vec2 models
- `audio_rendering.py`: Convert numeric features to text annotations
- `audio_enrichment.py`: Orchestrates all extractors, handles baselines and errors

**I/O and CLI**
- `writers.py`: JSON/TXT/SRT output with schema versioning
- `cli.py`: Modern unified CLI with subcommands (`transcribe`, `enrich`)
- `cli_legacy_transcribe.py`: Backward-compatible legacy interface

### JSON Schema (Version 2)

**Transcript structure:**
```json
{
  "schema_version": 2,
  "file_name": "audio.wav",
  "language": "en",
  "meta": {
    "generated_at": "2025-11-15T...",
    "model_name": "large-v3",
    "device": "cuda",
    "audio_enrichment": { /* optional enrichment metadata */ }
  },
  "segments": [ /* array of Segment objects */ ]
}
```

**Segment with audio_state:**
```json
{
  "id": 0,
  "start": 0.0,
  "end": 4.2,
  "text": "Hello world",
  "speaker": null,
  "tone": null,
  "audio_state": {
    "prosody": {
      "pitch": {"level": "high", "mean_hz": 245.3, "contour": "rising"},
      "energy": {"level": "loud", "db_rms": -8.2},
      "rate": {"level": "fast", "syllables_per_sec": 6.3}
    },
    "emotion": {
      "valence": {"level": "positive", "score": 0.72},
      "arousal": {"level": "high", "score": 0.68}
    },
    "rendering": "[audio: high pitch, loud volume, fast speech]",
    "extraction_status": {"prosody": "success", "emotion_dimensional": "success"}
  }
}
```

### Schema Versioning and Compatibility

**Forward Compatibility (v1 → v2):**
- v2 readers accept v1 transcripts (missing `audio_state` treated as `null`)
- `load_transcript_from_json()` handles both versions transparently

**Backward Compatibility:**
- v1 transcripts remain valid
- New optional fields (`audio_state`) don't break v1 consumers

**Stability Contract:**
- Core fields (`file_name`, `language`, `segments`, `id`, `start`, `end`, `text`) are stable within v2
- Optional fields (`speaker`, `tone`, `audio_state`) may be `null`
- Adding new optional fields does NOT bump schema version
- Breaking changes (removing/renaming core fields) require version bump (v3)

### Directory Layout

```
project_root/
  raw_audio/         # Original audio files (input)
  input_audio/       # Normalized 16kHz mono WAV (generated)
  transcripts/       # TXT and SRT outputs (generated)
  whisper_json/      # JSON structured transcripts (generated)
  transcription/     # Python package
  tests/             # Test suite
  examples/          # Example scripts
  docs/              # Documentation
```

## Important Design Patterns

### Audio State Is Audio-Only

**Critical**: All features in `audio_state` must be extracted from raw audio waveform, NOT inferred from text. This is the core distinction from semantic/NLP analysis.

Examples:
- **Allowed**: Pitch (Hz), energy (dB), speech rate (syllables/sec), pauses (ms), emotion from audio model
- **Not allowed**: Sentiment from text, topic detection, speaker name inference from content

### Speaker-Relative Normalization

Prosody features use **speaker-relative baselines** rather than absolute thresholds:
- Baseline computed from median of 20 sampled segments
- "high pitch" = above speaker's median, not an absolute Hz threshold
- Supports multi-speaker scenarios (future: per-speaker baselines with diarization)

### Graceful Degradation

Audio enrichment continues even if individual features fail:
- Each feature extraction wrapped in try/except
- `extraction_status` tracks success/failure per feature
- Partial enrichment supported (e.g., prosody succeeds, emotion fails)
- Errors logged but don't halt processing

### Lazy Model Loading

Emotion recognition models are large (~4GB) and GPU-intensive:
- Models loaded only when first needed
- Singleton pattern with lazy initialization
- GPU/CPU device selection at load time

### Dict Conversion Helpers

Three helpers exist for converting objects to JSON-serializable dicts:

**`turn_to_dict()` in `turn_helpers.py`** (canonical for Turn objects):
- **This is the canonical way to convert Turn objects to dicts**
- Handles Turn dataclasses, plain dicts, and objects with `to_dict()` methods
- Returns plain dicts unchanged (or shallow copy if `copy=True`)
- Calls `obj.to_dict()` if available
- Falls back to `asdict()` for dataclasses
- **Raises TypeError** for unsupported types
- Exported from `transcription/__init__.py` for public use
- Use this instead of reinventing Turn conversion logic in new code

**`_to_dict()` in `writers.py`** (lenient):
- Used for JSON serialization in `write_json()`
- Returns plain dicts unchanged
- Calls `obj.to_dict()` if available
- Falls back to `asdict()` for dataclasses
- Returns unsupported types unchanged (doesn't raise)
- Use when you need graceful handling of mixed types

**`_as_dict()` in `llm_utils.py`** (strict):
- Used for LLM rendering functions (`to_turn_view()`, `render_conversation_compact()`)
- Returns plain dicts unchanged
- Calls `obj.to_dict()` if available
- Falls back to `asdict()` for dataclasses
- **Raises TypeError** for unsupported types
- Use when you need guaranteed dict output

**Model `to_dict()` methods**:
- `DiarizationMeta`, `TurnMeta`, `Turn`, `Chunk`, `SpeakerStats` all implement `to_dict()`
- These are called by the helpers above during serialization
- All have docstrings explaining their purpose

**Contract for `transcript.meta`**:
- Must be a dict, None, or object convertible via `_to_dict()`
- If meta can't be converted to dict, `write_json()` logs a warning and uses `{}`
- Nested objects (like `DiarizationMeta`) are converted via their `to_dict()` methods

### Optional Dependency Pattern

For heavy optional dependencies (emotion, diarization), we use a protocol + factory pattern:

```python
# 1. Define protocol for the interface
class EmotionRecognizerLike(Protocol):
    def extract_emotion_dimensional(self, audio, sr) -> dict: ...

# 2. Try/except import with availability flag
EMOTION_AVAILABLE = False
try:
    import torch
    from transformers import ...
    EMOTION_AVAILABLE = True
except Exception:
    torch = cast(Any, None)

# 3. Dummy implementation for graceful degradation
class DummyEmotionRecognizer:
    def extract_emotion_dimensional(self, audio, sr) -> dict:
        return {"valence": {"level": "neutral", "score": 0.5}, ...}

# 4. Factory function returns appropriate implementation
def get_emotion_recognizer() -> EmotionRecognizerLike:
    if EMOTION_AVAILABLE:
        return EmotionRecognizer()
    logger.warning("Emotion dependencies unavailable; using dummy recognizer")
    return DummyEmotionRecognizer()
```

This pattern is used in:
- `emotion.py`: `EmotionRecognizerLike` + `get_emotion_recognizer()`
- `asr_engine.py`: `WhisperModelProtocol` + availability check
- `semantic.py`: `SemanticAnnotator` + `NoOpSemanticAnnotator`
- `diarization.py`: Environment-based stub mode (`SLOWER_WHISPER_PYANNOTE_MODE=stub`)

## Testing Philosophy

**Test organization:**
- `tests/test_models.py`: Data model tests
- `tests/test_prosody.py`: Prosody extraction tests
- `tests/test_emotion.py`: Emotion recognition tests
- `tests/test_audio_enrichment.py`: Integration tests for enrichment
- `tests/test_api_integration.py`: Public API integration tests
- `tests/test_writers.py`: JSON I/O and schema validation

**Test markers (pytest):**
- `@pytest.mark.slow`: Skip with `-m "not slow"`
- `@pytest.mark.heavy`: Heavy ML model tests (emotion, diarization) - skipped in fast CI
- `@pytest.mark.requires_gpu`: Skip with `-m "not requires_gpu"`
- `@pytest.mark.requires_enrich`: Requires enrichment dependencies
- `@pytest.mark.requires_diarization`: Requires pyannote.audio diarization extra

**Running subsets:**
```bash
# Fast tests only
uv run pytest -m "not slow and not requires_gpu"

# Unit tests only (no integration)
uv run pytest tests/test_models.py tests/test_prosody.py
```

## Type System Baseline (v1.3.1+)

### Coverage Summary

- **Package modules**: 39/39 pass mypy (92.9% function-level coverage)
- **Strategic tests**: 4 test modules configured for type checking
- **PEP 561**: `py.typed` marker present for downstream consumers

### Typed Modules (Full Coverage)

All modules in `transcription/` pass mypy:
- **Core**: `api.py`, `models.py`, `pipeline.py`, `writers.py`
- **CLI**: `cli.py`, `audio_enrich_cli.py`, `service.py`
- **Audio**: `audio_enrichment.py`, `prosody.py`, `emotion.py`, `diarization.py`
- **LLM**: `llm_utils.py`, `semantic.py`, `chunking.py`
- **Config**: `config.py`, `cache.py`
- **Utils**: `turn_helpers.py`, `turns.py`, `exporters.py`, `validation.py`

### Strategic Test Modules

These test modules are also type-checked in CI:
- `tests/test_llm_utils.py` (96.9% typed)
- `tests/test_writers.py` (95.5% typed)
- `tests/test_turn_helpers.py` (93.3% typed)
- `tests/test_audio_state_schema.py` (fixtures mostly untyped by design)

### Protocol Patterns

For optional dependencies, use the protocol + factory pattern:

```python
# In transcription/emotion.py
class EmotionRecognizerLike(Protocol):
    def extract_emotion_dimensional(self, audio: NDArray, sr: int) -> dict: ...

def get_emotion_recognizer() -> EmotionRecognizerLike:
    if EMOTION_AVAILABLE:
        return EmotionRecognizer()
    return DummyEmotionRecognizer()
```

Similar patterns in: `asr_engine.py` (WhisperModelProtocol), `semantic.py` (SemanticAnnotator), `diarization.py` (stub mode).

### Type Checking Commands

```bash
# Run mypy locally (same as CI)
uv run mypy transcription/ tests/test_llm_utils.py tests/test_writers.py tests/test_turn_helpers.py tests/test_audio_state_schema.py

# Quick verification (includes type checking)
uv run slower-whisper-verify --quick
```

### Guidelines for New Code

- **DO**: Add type annotations to new functions and classes
- **DO**: Use protocols for optional dependency interfaces
- **DO**: Document `cast()` usage with comments explaining why
- **DON'T**: Add `# type: ignore` without explanation
- **DON'T**: Use `Any` in return types of public API functions

See `docs/TYPING_POLICY.md` for complete typing standards.

## Code Style and Quality

**Tools configured:**
- **ruff**: Fast linter and formatter (replaces black + isort + flake8)
- **mypy**: Type checking (configured to warn but not fail in pre-commit)
- **pytest**: Testing with coverage reporting

**ruff configuration** (pyproject.toml):
- Line length: 100 characters
- Target: Python 3.11+
- Enabled rules: pycodestyle, pyflakes, isort, flake8-bugbear, flake8-comprehensions, pyupgrade

**mypy configuration**:
- Type checking enabled for `transcription/` package
- Third-party libraries (faster-whisper, librosa, transformers) ignored
- Configured for gradual typing (not strict mode)

## Key Files to Know

**Entry points:**
- `transcription/cli.py`: Main CLI with subcommands (recommended)
- `transcription/api.py`: Public Python API
- `transcribe_pipeline.py`: Legacy script entry point

**Core implementation:**
- `transcription/models.py`: Data models (Segment, Transcript)
- `transcription/pipeline.py`: Stage 1 orchestration
- `transcription/audio_enrichment.py`: Stage 2 orchestration
- `transcription/writers.py`: JSON I/O with versioning

**Configuration:**
- `pyproject.toml`: Project metadata, dependencies, tool configs
- `.pre-commit-config.yaml`: Pre-commit hooks
- `uv.lock`: Locked dependency versions

**Documentation:**
- `README.md`: User-facing documentation
- `docs/ARCHITECTURE.md`: Detailed implementation summary
- `CHANGELOG.md`: Version history

## Common Development Tasks

### Adding a New Audio Feature

1. Implement extractor in new or existing module (e.g., `transcription/voice_quality.py`)
2. Follow pattern from `prosody.py` or `emotion.py`:
   - Extract from audio segment (not text)
   - Return structured dict with numeric values + categorized levels
   - Handle errors gracefully (return None or partial data)
3. Integrate in `audio_enrichment.py`:
   - Add to `enrich_segment()` function
   - Update `extraction_status` tracking
   - Add config flag if optional
4. Update `audio_rendering.py` to render new features as text
5. Add tests in `tests/test_audio_enrichment.py`
6. Update `AUDIO_STATE_VERSION` if structure changes

### Modifying the JSON Schema

**For backward-compatible changes** (adding optional fields):
1. Add field to `Segment` or `Transcript` dataclass with default `None`
2. Update `write_json()` in `writers.py` to serialize new field
3. Update `load_transcript_from_json()` to handle missing field
4. Add tests for forward/backward compatibility
5. **Do not** bump `SCHEMA_VERSION`

**For breaking changes** (removing/renaming core fields):
1. Increment `SCHEMA_VERSION` in `models.py`
2. Update all writers and readers
3. Provide migration tool or compatibility layer
4. Document migration path in CHANGELOG
5. Support old version for at least one major release

### Running Benchmarks

```bash
# Run audio enrichment benchmark
uv run python benchmarks/benchmark_audio_enrich.py

# View benchmark results
cat benchmarks/results/benchmark_*.json
```

## External Dependencies

**Required (Stage 1):**
- `faster-whisper`: Whisper transcription engine
- `ffmpeg`: Audio normalization (system dependency, not Python package)

**Optional (Stage 2):**
- `librosa`: Audio analysis (energy, basic pitch)
- `praat-parselmouth`: Research-grade pitch extraction
- `soundfile`: WAV file I/O
- `torch`: PyTorch for neural models
- `transformers`: HuggingFace models for emotion recognition

**Development:**
- `pytest`, `pytest-cov`: Testing
- `ruff`: Linting and formatting
- `mypy`: Type checking
- `pre-commit`: Git hooks

## Model Downloads

**Emotion recognition models** (downloaded on first use):
- Dimensional: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- Categorical: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`

Models cached in `~/.cache/huggingface/` (Linux/macOS) or `%USERPROFILE%\.cache\huggingface\` (Windows).

**Whisper models** (downloaded on first use):
- Model size determines download: base (~150MB), medium (~1.5GB), large-v3 (~3GB)
- Cached by faster-whisper in `~/.cache/huggingface/hub/`

## Troubleshooting

**Import errors for enrichment modules:**
- Install enrichment dependencies: `uv sync --extra full`
- For prosody only: `uv sync --extra enrich-prosody`
- For emotion only: `uv sync --extra emotion`

**CUDA/GPU not detected:**
- Ensure NVIDIA drivers installed
- Check `torch.cuda.is_available()` in Python
- Pipeline falls back to CPU (slower but functional)

**Tests failing with "requires enrichment dependencies":**
- Install dev dependencies: `uv sync --extra dev`
- Or skip those tests: `uv run pytest -m "not requires_enrich"`

**Pre-commit hooks failing:**
- Run manually to see errors: `uv run pre-commit run --all-files`
- Auto-fix with ruff: `uv run ruff check --fix .`
- Format with ruff: `uv run ruff format .`

## Privacy and Security

**No data leaves your machine:**
- Transcription runs locally (faster-whisper)
- Audio enrichment runs locally
- Only model weights downloaded from internet (HuggingFace, on first use)
- No telemetry, no uploads

**Model sources:**
- Whisper models: OpenAI via HuggingFace
- Emotion models: Pre-trained academic models on HuggingFace
- All models publicly available and widely used
