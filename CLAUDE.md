# CLAUDE.md — Repo/Agent Guide (slower-whisper)

This file is the **single source of truth** for how to work in this repository:
- what the project is (today),
- how to run the canonical local gate (CI is currently off),
- where the important surfaces are,
- and what invariants we don't break.

If this doc disagrees with the code, the doc is wrong — update it.

---

## Snapshot

**Current series:** v1.9.x (shipped streaming callbacks + hardening)
**Current package version:** 1.9.2 (read from `pyproject.toml`)
**Last updated:** 2026-01-07
**Runtime version:** `transcription.__version__` is single-sourced from package metadata (guardrail test enforces this)

slower-whisper produces a **rich, versioned JSON transcript** capturing:

- **What was said:** text + segment timestamps + optional word timestamps/alignment
- **Who said it:** optional diarization → speakers + turns + speaker stats
- **How it was said:** prosody + emotion + interaction patterns
- **What it means (light):** keyword-based semantics today; deeper LLM semantics is v2+

Key posture:
- Local-first, reproducible, modular.
- Optional dependencies must degrade gracefully.
- Backward compatibility for JSON schema is intentional.

**Canonical docs (for details beyond this guide):**
- `ROADMAP.md` — Now/Next/Later execution plan + v2 track dependencies
- `CHANGELOG.md` — what shipped when
- `docs/STREAMING_ARCHITECTURE.md` — streaming model, events, callbacks
- `docs/GPU_SETUP.md` — GPU and Docker guidance
- `docs/TYPING_POLICY.md` — type system standards

---

## Repository posture (Actions are off)

GitHub Actions may be disabled / rate-limited. Assume CI is unavailable.

**Rule:** before merging anything, run the local gate and paste receipts in the PR.

### Canonical local gate

From repo root:

```bash
# In devshell
./scripts/ci-local.sh        # full
./scripts/ci-local.sh fast   # quick
```

What it does (high level):

* `uv sync` (expected extras)
* ruff + formatting
* mypy
* pytest (fast or full)
* `slower-whisper-verify --quick`
* Nix checks (via nix-clean) in full mode

---

## Environment

### Nix + direnv

This repo uses a Nix flake devshell, commonly loaded via direnv.

**Important:** the devshell intentionally exports `LD_LIBRARY_PATH` for Python wheels
(numpy/torch/ffmpeg). That breaks raw `nix` inside the shell due to libstdc++ ABI mismatch.

Use **nix-clean** for all Nix invocations:

```bash
nix-clean flake check
nix-clean run .#verify -- --quick
nix-clean run .#ci -- fast
```

direnv entry is hardened via a "clean flake entry" wrapper to avoid poisoning
direnv's internal `nix print-dev-env` call when re-entering a shell.

### Python deps

Use `uv` (lockfile is committed).

Typical first-time setup:

```bash
uv sync --extra full --extra dev
# optional extras:
# uv sync --extra diarization
# uv sync --extra integrations
```

---

## Where things live

### Core pipeline

* `transcription/pipeline.py` — orchestration for batch runs
* `transcription/asr_engine.py` — ASR engine plumbing (faster-whisper / CTranslate2)
* `transcription/writers.py` — JSON + exports writer utilities
* `transcription/validation.py` — schema validation and checks

### Streaming

* `transcription/streaming.py` — base streaming session + events
* `transcription/streaming_enrich.py` — streaming enrichment session (user-facing)
* `transcription/streaming_semantic.py` — live semantics (turn-aware)
* `transcription/streaming_callbacks.py` — `StreamCallbacks` protocol + safe invocation

Docs:

* `docs/STREAMING_ARCHITECTURE.md` — authoritative streaming model + event flow

### Device selection & GPU UX

* `transcription/device.py` — device resolution + compute_type resolution + preflight banner
* `docs/GPU_SETUP.md` — practical local + docker GPU guidance

### Requirements generation

* `uv.lock` — source of truth lock
* `requirements*.txt` — generated artifacts for pip/Docker fallback
* `scripts/regenerate-requirements.sh` — regenerate from `uv.lock`

---

## Behavioral invariants (don't break these)

### 1) Device resolution is explicit and explainable

* CLI supports `--device auto|cpu|cuda`
* Auto detection for ASR uses **CTranslate2** CUDA availability (not torch)
* Enrichment "auto" uses **torch** backend
* A preflight banner prints to **stderr** to keep stdout clean for scripts

### 2) compute_type follows the *resolved* device (unless user explicitly set it)

If user didn't pass `--compute-type`, and we fall back from CUDA → CPU, then
`compute_type` must be coerced to a valid CPU compute type (e.g., int8).

### 3) Callbacks never crash the pipeline

All callback invocations must be isolated:

* exceptions are caught
* errors are routed to `on_error` when possible
* pipeline continues

### 4) Streaming end-of-stream must finalize turns consistently

Segments finalized via `end_of_stream()` must also be passed through the same
turn tracking logic as segments finalized during `ingest_chunk()`.

### 5) Versioning must not regress

* `transcription.__version__` must equal `importlib.metadata.version("slower-whisper")` when installed
* guardrail test exists (`tests/test_versioning.py`)

### 6) Optional dependencies must degrade gracefully

No hard requirement on:

* diarization stack
* emotion models
* integrations (langchain/llamaindex)

If missing, fail with actionable messages or skip relevant tests.

---

## Local validation recipes (copy/paste receipts)

When posting PR receipts, prefer this format.

```text
Local receipts:

./scripts/ci-local.sh fast
./scripts/ci-local.sh
nix-clean flake check
nix-clean run .#verify -- --quick
```

If you only ran fast mode, say so explicitly.

---

## Issue / PR expectations (for humans + agents)

### Issues

Every issue should have:

* **Problem**
* **Definition of done** (testable)
* **Files likely touched**
* **Local validation command(s)**

### PRs

* Keep PRs coherent; stacking is fine when it doesn't increase review cost.
* Include a "review map" when > ~5 files.
* Paste local receipts in the PR body or a comment.
* If a PR changes behavior, add or update tests.

---

## Release process (when you cut a tag)

1. Ensure version bump is correct:

* only bump `pyproject.toml` (runtime pulls from metadata)
* update `CHANGELOG.md`

2. Run gate:

```bash
./scripts/ci-local.sh
```

3. Tag and push:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

4. Optional verification:

* confirm GHCR images build on tag (if Docker workflow enabled)
* confirm provenance attestation succeeds (OIDC permissions required)

---

## Known sharp edges

* Raw `nix ...` may fail inside devshell due to `LD_LIBRARY_PATH` being set for wheels.
  Use `nix-clean`.
* If you see `CXXABI_1.3.15 not found`, you ran `nix` unwrapped in the wheel-runtime env.
* Some tests are intentionally skipped based on markers (gpu/diarization/heavy). That's OK.
  The point is determinism and correct behavior when deps exist.

---

## Code Architecture

### Layered Architecture (L0-L4)

- **L0: Ingestion** — audio normalization, chunking, hashing
- **L1: ASR (Whisper)** — fast, deterministic transcription via faster-whisper
- **L2: Acoustic Enrichment** — speaker diarization, prosody, emotion, turns (modular, cacheable)
- **L3: Semantic Enrichment** — optional local SLM annotations (chunk summaries, intent tags)
- **L4: Task Outputs** — meeting notes, QA, coaching (consumer applications)

**Key principle**: Each layer adds value without blocking earlier layers. All layers independently cacheable and resumable.

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
  "file": "audio.wav",
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

> **Note:** The dataclass field is `Transcript.file_name` but serializes to `"file"` in JSON.

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

**Forward Compatibility (v1 -> v2):**
- v2 readers accept v1 transcripts (missing `audio_state` treated as `null`)
- `load_transcript_from_json()` handles both versions transparently

**Backward Compatibility:**
- v1 transcripts remain valid
- New optional fields (`audio_state`) don't break v1 consumers

**Stability Contract:**

- Core fields (`file`, `language`, `segments`, `id`, `start`, `end`, `text`) are stable within v2
- Optional fields (`speaker`, `tone`, `audio_state`) may be `null`
- Adding new optional fields does NOT bump schema version
- Breaking changes (removing/renaming core fields) require version bump (v3)

> **Source of truth:** `transcription/writers.py` (serialization) and `transcription/models.py` (dataclasses).

---

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

### Audio Rendering Utilities

**`_extract_audio_descriptors()` in `llm_utils.py`**:
- Parses `[audio: high pitch, loud volume]` format rendering strings
- Returns list of descriptor strings (e.g., `["high pitch", "loud volume"]`)
- Used by `render_segment()`, `_render_turn_dict()`, and `_collect_audio_descriptors()`
- **Use this instead of reimplementing the `[audio:...]` parsing pattern**

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

---

## Testing

**Test organization:**
- `tests/test_models.py`: Data model tests
- `tests/test_prosody.py`: Prosody extraction tests
- `tests/test_emotion.py`: Emotion recognition tests
- `tests/test_audio_enrichment.py`: Integration tests for enrichment
- `tests/test_api_integration.py`: Public API integration tests
- `tests/test_writers.py`: JSON I/O and schema validation
- `tests/test_streaming*.py`: Streaming pipeline tests

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

---

## Type System

### Coverage Summary

- **Package modules**: 39/39 pass mypy (92.9% function-level coverage)
- **Strategic tests**: 4 test modules configured for type checking
- **PEP 561**: `py.typed` marker present for downstream consumers

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

---

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

---

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

---

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

---

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
