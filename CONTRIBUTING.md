# Contributing to slower-whisper

Thank you for your interest in contributing! This guide will help you set up your development environment and understand the project workflow.

Whether you're fixing a bug, adding a feature, or improving documentation, this guide has you covered. We welcome contributions of all kinds and skill levels!

---

## Table of Contents

- [Who This Project Is For](#who-this-project-is-for)
- [Quick Start for Contributors](#quick-start-for-contributors)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Code Style and Linting](#code-style-and-linting)
- [Making Changes](#making-changes)
- [Adding New Features](#adding-new-features)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Development Tips](#development-tips)
- [GPU/CUDA Development](#gpucuda-development)
- [Getting Help](#getting-help)
- [Frequently Asked Questions](#frequently-asked-questions)

---

## Who This Project Is For

Before contributing, it's helpful to understand who slower-whisper serves and how we prioritize features:

### Primary Users (v1.x Focus)

**1. Infrastructure / Platform Engineers**
- Building internal conversation processing systems
- On-prem transcription stacks (compliance, security)
- Multi-tenant platforms with conversation analytics

**What they need:**
- ✅ Stability, contracts, Docker/K8s deployment
- ✅ Versioned JSON schema (build on it without fear)
- ✅ BDD/IaC guarantees (infrastructure-grade quality)

**2. Research Labs (Linguistics, HCI, Psychology, UX)**
- Analyzing conversations for academic research
- Prosody and emotion studies
- Speaker interaction patterns

**What they need:**
- ✅ Accurate prosody/emotion features
- ✅ Reproducible pipelines (same input → same output)
- ✅ Export to research tools (Praat, ELAN)

### Secondary Users (v1.2+)

**3. LLM Application Developers**
- Building conversation-aware LLM apps
- RAG/vector search with acoustic metadata
- Meeting summarization and action-item extraction

**What they need:**
- ✅ LangChain/LlamaIndex adapters
- ✅ Prompt builder utilities
- ✅ Easy chunking and formatting

### Design Priorities

When contributing, keep these priorities in mind:

1. **Local-first always** — No cloud dependencies at runtime
2. **Contracts over convenience** — Schema stability beats new features
3. **Infrastructure users first** — Reliability over novelty
4. **BDD scenarios are contracts** — Breaking them requires discussion

**This is NOT:**
- A consumer transcription app
- A SaaS platform or cloud API
- A "meeting notes" product

**This IS:**
- Infrastructure for building conversation-aware systems
- "OpenTelemetry for audio conversations"
- A stable foundation for production use

See [VISION.md](VISION.md) for detailed positioning.

---

## Quick Start for Contributors

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/slower-whisper.git
cd slower-whisper

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Install all dependencies including dev tools
uv sync --extra dev

# 4. Set up pre-commit hooks (recommended)
uv run pre-commit install

# 5. Run tests to verify everything works
uv run pytest

# 6. Start developing!
```

---

## Development Environment Setup

### Prerequisites

Before you begin, ensure you have:

1. **Python 3.10 or later** - Check with `python --version`
2. **ffmpeg** - Required for audio processing
   - Ubuntu/Debian: `sudo apt-get install ffmpeg libsndfile1`
   - macOS: `brew install ffmpeg`
   - Windows: `choco install ffmpeg`
3. **Git** - For version control
4. **uv** - Fast Python package manager (installed in next step)

### Installing uv

uv is a fast, modern Python package manager that handles virtual environments and dependencies. It's **much faster** than pip and makes development smoother.

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

Verify installation:
```bash
uv --version
```

### Setting Up Your Development Environment

The project uses `pyproject.toml` with dependency groups. Use `uv sync` to install everything:

**Full development setup (recommended for contributors):**
```bash
# Install ALL dependencies including dev tools, testing, linting, etc.
uv sync --extra dev

# This installs:
# - Base transcription dependencies (faster-whisper)
# - Audio enrichment dependencies (librosa, parselmouth, torch, transformers)
# - Testing tools (pytest, pytest-cov, pytest-xdist, pytest-mock)
# - Linting tools (ruff, black, isort, mypy)
# - Pre-commit hooks
# - Documentation tools (sphinx)
# - Build tools
```

**Minimal setup (if you have disk space constraints):**
```bash
# Base install only (transcription features)
uv sync

# Then manually install dev tools
uv pip install pytest pytest-cov ruff mypy pre-commit
```

**Verify your installation:**
```bash
# Run tests to ensure everything is working
uv run pytest

# Should show all tests passing (191 tests)
```

### Alternative: Using pip or poetry

While we recommend uv, you can also use traditional tools:

**Using pip:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Or: venv\Scripts\activate  # Windows

pip install -e ".[dev]"  # Editable install with dev dependencies
```

**Using poetry:**
```bash
poetry install --with dev
poetry shell
```

---

## Project Structure

```
slower-whisper/
├── transcription/          # Core package
│   ├── models.py          # Data models (Segment, Transcript)
│   ├── config.py          # Configuration classes
│   ├── audio_io.py        # Audio normalization (ffmpeg)
│   ├── asr_engine.py      # Whisper wrapper
│   ├── writers.py         # JSON/TXT/SRT output
│   ├── pipeline.py        # Orchestration
│   ├── cli.py             # Stage 1 CLI
│   ├── prosody.py         # Prosody extraction
│   ├── emotion.py         # Emotion recognition
│   ├── audio_enrichment.py # Stage 2 orchestrator
│   ├── audio_enrich_cli.py # Stage 2 CLI
│   ├── audio_rendering.py  # Text rendering
│   └── audio_utils.py      # Audio utilities
├── tests/                  # Test suite
├── examples/               # Example scripts
├── docs/                   # Documentation
├── transcribe_pipeline.py  # Stage 1 entry point
└── audio_enrich.py        # Stage 2 entry point
```

---

## Running Tests

All test commands should be run with `uv run` to ensure you're using the virtual environment:

### Basic Test Commands

```bash
# Run all tests (most common during development)
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=transcription --cov-report=html
# Then open htmlcov/index.html in your browser

# Run specific test file
uv run pytest tests/test_prosody.py -v

# Run specific test function
uv run pytest tests/test_prosody.py::test_syllable_counting -v

# Run tests matching a keyword
uv run pytest -k "prosody" -v
```

### Test Categories and Markers

The project uses pytest markers to categorize tests:

```bash
# Run only fast unit tests (default in CI)
uv run pytest -m "not slow and not heavy"

# Run integration tests only
uv run pytest -m "integration"

# Run slow tests (downloads models, processes files)
uv run pytest -m "slow"

# Run heavy tests (requires emotion models, ~4GB download)
uv run pytest -m "heavy"

# Skip slow tests during development
uv run pytest -m "not slow"
```

### Test Suite Overview

The project has 191 tests across multiple categories:

| Category | Count | Description |
|----------|-------|-------------|
| Audio Enrichment | 19 | Prosody + emotion extraction |
| Audio Rendering | 12 | Text formatting with audio features |
| Prosody | 12 | Pitch, energy, rate extraction |
| Integration | 8 | End-to-end pipeline tests |
| Writers | 6 | JSON, TXT, SRT output |
| BDD Scenarios | 15+ | Behavioral acceptance tests |
| Other | 119+ | Unit tests across all modules |

**Expected result:** All tests should pass unless you're actively developing new features.

### Continuous Testing During Development

For rapid feedback during development, use pytest's watch mode or run specific tests:

```bash
# Run tests related to your changes only
uv run pytest tests/test_prosody.py -v

# Use pytest-watch for auto-rerun on file changes (optional)
uv pip install pytest-watch
uv run ptw -- tests/test_prosody.py
```

---

## Code Style and Linting

This project uses **Ruff** for both linting and formatting. Ruff is extremely fast and handles everything in one tool.

### Running Linters and Formatters

All linting and formatting should be done with `uv run`:

```bash
# Check code style (shows issues without fixing)
uv run ruff check .

# Auto-fix code style issues
uv run ruff check --fix .

# Check code formatting
uv run ruff format --check .

# Auto-format code
uv run ruff format .

# Run type checker (optional, currently non-strict)
uv run mypy transcription tests

# Run all checks at once (what CI runs)
uv run ruff check . && uv run ruff format --check . && uv run mypy transcription tests
```

### Pre-commit Hooks (Recommended)

Pre-commit hooks automatically run linters before each commit, catching issues early:

```bash
# Install pre-commit hooks (one-time setup)
uv run pre-commit install

# Now hooks run automatically on git commit

# Run manually on all files
uv run pre-commit run --all-files

# Skip hooks for a specific commit (use sparingly)
git commit --no-verify -m "message"
```

**What pre-commit checks:**
- Ruff linting with auto-fix
- Ruff formatting
- Type checking with mypy (warning only)
- Trailing whitespace
- End-of-file fixers
- YAML/JSON/TOML validation
- Large file detection
- Merge conflict markers
- Debug statement detection

### Code Style Guidelines

We follow PEP 8 with some project-specific conventions:

**1. Line Length:** 100 characters (enforced by Ruff)

**2. Type Hints:** Use them for function signatures (helpful but not strictly required)
```python
# Good
def extract_pitch(audio: np.ndarray, sr: int) -> float:
    ...

# Acceptable for simple cases
def helper_function(value):
    ...
```

**3. Docstrings:** Add for all public functions and classes
```python
def extract_prosody(
    audio: np.ndarray,
    sr: int,
    text: str,
    speaker_baseline: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Extract prosodic features from audio segment.

    Args:
        audio: Mono audio signal (float32)
        sr: Sample rate in Hz
        text: Transcribed text
        speaker_baseline: Optional baseline statistics for normalization

    Returns:
        Dictionary with pitch, energy, rate, and pause features

    Example:
        >>> audio = load_audio("test.wav")
        >>> features = extract_prosody(audio, 16000, "Hello world")
        >>> print(features["pitch_hz_mean"])
        150.5
    """
    # Implementation...
```

**4. Error Handling:** Prefer graceful degradation over hard failures
```python
# Good - graceful fallback
try:
    pitch = extract_pitch_praat(audio, sr)
except ImportError:
    logger.warning("Praat not available, using librosa fallback")
    pitch = extract_pitch_librosa(audio, sr)

# Avoid - hard failure for optional features
pitch = extract_pitch_praat(audio, sr)  # Crashes if praat not installed
```

**5. Logging:** Use Python's logging module, not print()
```python
import logging
logger = logging.getLogger(__name__)

# Good
logger.info(f"Processing {len(segments)} segments")
logger.debug(f"Extracted pitch: {pitch_hz}")

# Bad
print(f"Processing {len(segments)} segments")
```

**6. Import Organization:** Ruff handles this automatically
```python
# Standard library
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party
import numpy as np
import torch
from transformers import pipeline

# Local
from transcription.models import Segment, Transcript
from transcription.config import Config
```

---

## Making Changes

### Development Workflow

Follow these steps when contributing:

**1. Create a feature branch**
```bash
# Update main branch first
git checkout main
git pull origin main

# Create your feature branch
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/issue-description
```

**Branch naming conventions:**
- `feature/add-voice-quality-extraction` - New features
- `fix/pitch-extraction-short-audio` - Bug fixes
- `docs/update-installation-guide` - Documentation updates
- `refactor/simplify-audio-loading` - Code refactoring
- `test/add-emotion-tests` - Test additions

**2. Make your changes**

Edit code, add features, fix bugs, etc.

**3. Add tests**

For new features or bug fixes, add tests:

```bash
# Create or update test file
# tests/test_your_feature.py

def test_your_new_feature():
    """Test that the new feature works as expected."""
    result = your_new_function(test_input)
    assert result == expected_output
```

See the [Testing Guidelines](#testing-guidelines) section below for more details.

**4. Run tests locally**

```bash
# Run all tests
uv run pytest

# Run only tests related to your changes
uv run pytest tests/test_your_feature.py -v

# Check coverage
uv run pytest --cov=transcription --cov-report=term-missing
```

**5. Verify contracts (REQUIRED before pushing)**

Before pushing your changes, **you must run the verification CLI** to ensure you haven't broken the behavioral or deployment contracts:

```bash
# Quick verification (code quality + tests + BDD + API BDD)
# This is the MINIMUM required before pushing
uv run slower-whisper-verify --quick

# Full verification (includes Docker and K8s validation)
# Recommended before creating a PR
uv run slower-whisper-verify
```

**What this verifies:**
- ✅ Code quality (ruff linting and formatting)
- ✅ Unit tests pass
- ✅ Library BDD scenarios (behavioral contract)
- ✅ API BDD scenarios (REST API contract)
- ✅ Docker images build correctly (full mode only)
- ✅ Kubernetes manifests are valid (full mode only)

If any step fails, **do not push**. Fix the issues first. This verification ensures you're not breaking the project's **behavioral contracts** (BDD scenarios) or **deployment contracts** (Docker/K8s).

**6. Format and lint your code**

```bash
# Auto-format code
uv run ruff format .

# Auto-fix linting issues
uv run ruff check --fix .

# Or let pre-commit handle it
uv run pre-commit run --all-files
```

**7. Update documentation**

If your changes affect usage:
- Update relevant files in `docs/`
- Update README.md if adding user-facing features
- Add examples if helpful
- Update docstrings

**8. Commit your changes**

```bash
# Stage your changes
git add .

# Commit with a clear message (pre-commit hooks will run automatically)
git commit -m "Add pause density feature to prosody extraction"

# If pre-commit fixes issues, review and commit again
git add .
git commit -m "Add pause density feature to prosody extraction"
```

**9. Push and create a pull request**

```bash
# Push your branch
git push origin feature/your-feature-name

# Then create a PR on GitHub
# The CI will automatically run tests
```

### Commit Message Guidelines

Write clear, descriptive commit messages that explain **what** and **why**:

**Good commit messages:**
```bash
git commit -m "Add pause density calculation to prosody features

Pause density helps identify speaking rhythm patterns. This adds
calculation of pause count per second to the prosody extraction."

git commit -m "Fix pitch extraction for audio segments under 0.1s

Short audio segments caused division by zero in pitch extraction.
Now returns None for segments that are too short."

git commit -m "Update installation docs to use uv sync instead of pip"
```

**Avoid vague messages:**
```bash
# Too vague
git commit -m "fix bug"
git commit -m "updates"
git commit -m "changes"

# Better
git commit -m "Fix division by zero in pitch extraction"
git commit -m "Update installation instructions"
git commit -m "Refactor audio loading to reduce memory usage"
```

**Format:**
- First line: Short summary (50-72 characters)
- Blank line
- Detailed explanation if needed (wrap at 72 characters)
- Reference issues: `Fixes #123` or `Closes #456`

---

## Adding New Features

### Stage 1: Transcription Pipeline

If adding transcription features:
1. Update `transcription/asr_engine.py` or `transcription/pipeline.py`
2. Update `transcription/models.py` if schema changes
3. Add tests to `tests/test_integration.py`
4. Update `docs/QUICKSTART.md` if user-facing

### Stage 2: Audio Enrichment

If adding audio features:
1. Add extraction logic to appropriate module:
   - Prosody → `transcription/prosody.py`
   - Emotion → `transcription/emotion.py`
   - New category → Create new module
2. Update `transcription/audio_enrichment.py` to integrate
3. Update `transcription/audio_rendering.py` for text rendering
4. Add tests to `tests/test_audio_enrichment.py`
5. Update `docs/AUDIO_ENRICHMENT.md`

### Example: Adding a New Feature

```python
# 1. Add extraction function (e.g., in transcription/prosody.py)
def extract_voice_quality(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Extract jitter, shimmer, HNR."""
    # Implementation
    return {
        "jitter": jitter_value,
        "shimmer": shimmer_value,
        "hnr": hnr_value
    }

# 2. Integrate in audio_enrichment.py
def enrich_segment_audio(...):
    # ... existing code ...

    if config.enable_voice_quality:
        voice_quality = extract_voice_quality(audio, sr)
        audio_state["voice_quality"] = voice_quality

# 3. Add tests (tests/test_audio_enrichment.py)
def test_voice_quality_extraction():
    audio = generate_test_audio()
    result = extract_voice_quality(audio, 16000)
    assert "jitter" in result
    assert "shimmer" in result
    assert "hnr" in result

# 4. Update docs (docs/AUDIO_ENRICHMENT.md)
# Add section explaining the new feature
```

---

## Testing Guidelines

### Writing Tests

1. **Use pytest fixtures** - See `tests/conftest.py`
2. **Test happy path + edge cases**
3. **Use descriptive test names**

```python
# Good test names
def test_extract_prosody_with_baseline_normalization():
    """Test that prosody features are normalized to speaker baseline."""
    # ...

def test_emotion_extraction_handles_short_audio():
    """Test emotion extraction works on audio < 0.5 seconds."""
    # ...
```

### Test Coverage

Aim for:
- **New features:** 100% coverage
- **Bug fixes:** Add regression test
- **Existing code:** Don't reduce coverage

Check coverage:
```bash
pytest tests/ --cov=transcription --cov-report=term-missing
```

---

## Documentation

### When to Update Documentation

Update docs when:
- Adding new features
- Changing CLI arguments
- Modifying JSON schema
- Changing installation process
- Fixing bugs that affect usage

### Which Files to Update

| Change | Update Files |
|--------|-------------|
| New feature | README.md, docs/AUDIO_ENRICHMENT.md or docs/PROSODY.md |
| CLI change | README.md, docs/QUICKSTART.md |
| Installation | README.md, CONTRIBUTING.md |
| Troubleshooting | docs/TROUBLESHOOTING.md (if exists) |
| Examples | examples/README.md, add example script |

### Documentation Style

- **Clear and concise**
- **Include code examples**
- **Show expected output**
- **Link between related docs**

---

## Pull Request Process

### Before Submitting a PR

Run through this checklist before creating your pull request:

**Required (Hard gates - PR will be rejected without these):**
- [ ] **Verification CLI passes**: `uv run slower-whisper-verify --quick`
  - This verifies code quality, tests, and behavioral contracts (BDD scenarios)
  - If this fails, **do not create a PR**
- [ ] Tests added for new features or bug fixes
- [ ] Branch is up-to-date with main
- [ ] No BDD scenarios broken (verification CLI checks this)

**Strongly Recommended:**
- [ ] Full verification passes: `uv run slower-whisper-verify`
  - Includes Docker and K8s validation
  - Required for releases, recommended for PRs
- [ ] Documentation updated (README, docs/, docstrings)
- [ ] Examples added if applicable
- [ ] Commit messages are clear and descriptive
- [ ] Pre-commit hooks installed and passing

**Behavioral Contract Awareness:**
If your changes affect any BDD scenarios (`tests/features/` or `features/`), you **must**:
- [ ] Document why the behavioral contract is changing
- [ ] Discuss versioning impact (major/minor/patch)
- [ ] Update `CHANGELOG.md` with contract changes
- [ ] Consider if this requires a deprecation period

### Creating a Pull Request

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a PR on GitHub** with this template:

   ```markdown
   ## Description

   Brief description of what this PR does and why.

   Fixes #<issue_number> (if applicable)

   ## Type of Change

   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update
   - [ ] Code refactoring
   - [ ] Performance improvement

   ## Changes Made

   - Change 1
   - Change 2
   - Change 3

   ## Testing

   Describe how you tested your changes:

   - [ ] Added new tests
   - [ ] All existing tests pass
   - [ ] Tested manually with: [describe test scenario]

   ## Screenshots/Examples (if applicable)

   Include code examples, output samples, or screenshots if relevant.

   ## Checklist

   - [ ] My code follows the project's style guidelines
   - [ ] I have performed a self-review of my code
   - [ ] I have commented my code, particularly in hard-to-understand areas
   - [ ] I have made corresponding changes to the documentation
   - [ ] My changes generate no new warnings
   - [ ] I have added tests that prove my fix is effective or that my feature works
   - [ ] New and existing unit tests pass locally with my changes
   - [ ] Any dependent changes have been merged and published
   ```

### What Happens Next

1. **Automated CI checks run** - Your PR will automatically trigger:
   - Linting (ruff check)
   - Formatting check (ruff format)
   - Type checking (mypy)
   - Tests on Python 3.10, 3.11, 3.12
   - Integration tests

2. **Review by maintainer** - A project maintainer will:
   - Review your code
   - Test functionality
   - Provide feedback or request changes

3. **Address feedback** - If changes are requested:
   ```bash
   # Make changes
   git add .
   git commit -m "Address review feedback"
   git push origin feature/your-feature-name
   # CI will re-run automatically
   ```

4. **Merge** - Once approved and all checks pass:
   - Maintainer will merge your PR
   - Your contribution will be included in the next release!

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

**On every PR:**
- Lint check (ruff)
- Format check (ruff format)
- Type check (mypy, continue-on-error)
- Unit tests on multiple Python versions
- Integration tests

**On main branch:**
- All of the above
- Heavy tests (emotion models, optional)
- Code coverage reporting

**Expected behavior:**
- All required checks must pass before merge
- Type check failures are warnings only
- Heavy tests are optional (run on main branch only)

---

## Development Tips

### Running the Pipeline Locally

Always use `uv run` to ensure you're using the project's virtual environment:

```bash
# Create required directories
mkdir -p raw_audio input_audio transcripts whisper_json

# Add some test audio files to raw_audio/

# Stage 1: Transcribe using base model
uv run slower-whisper --model base --language en

# Stage 2: Enrich with prosody only (fast, no GPU needed)
uv run slower-whisper-enrich --no-enable-emotion

# Stage 2: Full enrichment with emotion (requires GPU)
uv run slower-whisper-enrich --enable-categorical-emotion

# You can also use the Python scripts directly:
uv run python transcribe_pipeline.py --help
uv run python audio_enrich.py --help
```

### Working with Different Dependency Sets

The project supports multiple dependency configurations:

```bash
# Minimal install (transcription only, ~2.5GB)
uv sync

# Add prosody features (~3.5GB)
uv sync --extra enrich-prosody

# Add emotion recognition (~6.5GB)
uv sync --extra emotion

# Full installation (everything)
uv sync --extra full

# Development (full + dev tools)
uv sync --extra dev

# Install additional extras after initial sync
uv pip install -e ".[emotion]"
```

### Generating Test Audio

```python
import numpy as np
import soundfile as sf

# Generate 2 seconds of test audio (sine wave)
sr = 16000
duration = 2.0
frequency = 440  # A4 note
t = np.linspace(0, duration, int(sr * duration))
audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

sf.write("test_audio.wav", audio, sr)
```

### Debugging

**Enable debug logging:**
```python
# Add to code for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in specific module
logger = logging.getLogger(__name__)
logger.debug(f"Extracted pitch: {pitch_hz}")
logger.debug(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
```

**Use Python debugger:**
```bash
# Run with debugger
uv run python -m pdb transcription/cli.py

# Or use ipdb (better interface)
uv pip install ipdb
# Add to code: import ipdb; ipdb.set_trace()
```

**Debug tests:**
```bash
# Run single test with verbose output
uv run pytest tests/test_prosody.py::test_syllable_counting -vv

# Print debug output in tests
uv run pytest tests/test_prosody.py -s

# Drop into debugger on failure
uv run pytest tests/test_prosody.py --pdb
```

### Common Development Tasks

**Update dependencies:**
```bash
# Update all dependencies to latest compatible versions
uv sync --upgrade

# Update specific package
uv pip install --upgrade numpy

# Re-sync after changing pyproject.toml
uv sync
```

**Clean up environment:**
```bash
# Remove virtual environment and reinstall
rm -rf .venv
uv sync --extra dev

# Clear pytest cache
rm -rf .pytest_cache __pycache__

# Clear coverage data
rm -rf .coverage htmlcov/
```

**Build distribution packages:**
```bash
# Build wheel and source distribution
uv run python -m build

# Install locally from source
uv pip install -e .
```

### Troubleshooting

**Tests failing:**
```bash
# Clear cache and rerun
uv run pytest --cache-clear

# Run with verbose output to see details
uv run pytest -vv

# Check for conflicting dependencies
uv pip list
```

**Import errors:**
```bash
# Make sure you're using uv run
uv run python script.py

# Or verify virtual environment is activated
which python  # Should show .venv/bin/python

# Reinstall dependencies
uv sync --reinstall
```

**Pre-commit hooks failing:**
```bash
# Update hooks to latest version
uv run pre-commit autoupdate

# Run hooks manually to debug
uv run pre-commit run --all-files --verbose

# Skip hooks temporarily (not recommended)
git commit --no-verify -m "message"
```

**GPU/CUDA issues:**
```bash
# Check if CUDA is available
uv run python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode for testing
uv run slower-whisper-enrich --device cpu

# Check CUDA version
nvidia-smi
```

**Memory issues:**
```bash
# Use smaller Whisper model
uv run slower-whisper --model tiny

# Process fewer files at once
# Limit batch size in code

# Monitor memory usage
uv pip install psutil
uv run python -c "import psutil; print(psutil.virtual_memory())"
```

---

## GPU/CUDA Development

### Testing GPU Code

If you have NVIDIA GPU:
```bash
# Use GPU
python audio_enrich.py --device cuda

# Force CPU (for testing fallback)
python audio_enrich.py --device cpu
```

### GPU Requirements

- **Emotion models:** ~2-4 GB VRAM
- **Faster-whisper:** 2-6 GB VRAM (model-dependent)
- **Both together:** 4-8 GB VRAM total

---

## Getting Help

### Resources

- **Documentation:** See `docs/` directory for detailed guides
- **Examples:** See `examples/` directory for working code samples
- **Tests:** See `tests/` directory for usage examples
- **Issues:** Check [GitHub Issues](https://github.com/steven/slower-whisper/issues) for known problems

### Asking Questions

When asking for help, provide context:

1. **Describe your goal:** What are you trying to accomplish?
2. **Show your attempt:** Share the code or commands you tried
3. **Include error messages:** Full traceback, not just the last line
4. **Mention your environment:**
   - Operating System (Windows/Linux/macOS)
   - Python version (`python --version`)
   - GPU details (if applicable)
   - Dependency versions (`uv pip list`)

**Good question example:**
```
I'm trying to add a new prosody feature for voice quality. I added
a function in prosody.py but getting:

  ImportError: cannot import name 'extract_voice_quality'

I'm running Python 3.11 on Ubuntu 22.04 with the dev dependencies
installed via `uv sync --extra dev`. Here's my code: [paste code]
```

---

## Frequently Asked Questions

### For New Contributors

**Q: I'm new to open source. Where should I start?**

A: Great! Here's a beginner-friendly path:
1. Start by reading the README and trying the tool yourself
2. Look for issues labeled "good first issue" or "documentation"
3. Fix typos or improve documentation (great first contribution!)
4. Then move on to small bug fixes or test additions

**Q: Do I need to know machine learning to contribute?**

A: No! Many contributions don't require ML knowledge:
- Documentation improvements
- Bug fixes in file I/O or data processing
- Test additions
- Code refactoring
- Installation/setup improvements

**Q: How do I find something to work on?**

A:
1. Check the [GitHub Issues](https://github.com/steven/slower-whisper/issues)
2. Look for "good first issue" or "help wanted" labels
3. Read the code and find areas that could be improved
4. Fix something that bothered you while using the tool

**Q: What if I break something?**

A: Don't worry! That's what tests and reviews are for:
- Tests will catch most issues before merge
- Maintainers review all PRs
- You can always ask for help
- Breaking things in a PR is a learning opportunity, not a problem

### Technical Questions

**Q: Should I use `uv run` or activate the virtual environment?**

A: Either works, but `uv run` is recommended:
```bash
# Recommended: uv run automatically uses the right environment
uv run pytest

# Also works: activate first, then run commands
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pytest
```

**Q: How do I run only the tests for my changes?**

A: Use pytest's file/function selection:
```bash
# Run specific test file
uv run pytest tests/test_prosody.py

# Run specific test function
uv run pytest tests/test_prosody.py::test_syllable_counting

# Run tests matching a keyword
uv run pytest -k "pitch"
```

**Q: My pre-commit hooks are slow. Can I skip them?**

A: Pre-commit hooks save time by catching issues early. But for rapid iteration:
```bash
# Skip for one commit (not recommended)
git commit --no-verify -m "WIP: testing"

# Or only run on changed files (faster)
uv run pre-commit run

# Run hooks manually instead of on every commit
pre-commit uninstall  # Remove hooks
uv run pre-commit run --all-files  # Run manually when ready
```

**Q: How do I add a dependency?**

A: Add it to the appropriate section in `pyproject.toml`:
```bash
# 1. Edit pyproject.toml and add your dependency
# 2. Re-sync to install it
uv sync

# 3. If it's a core dependency, add to main dependencies
# If it's optional, add to appropriate [project.optional-dependencies]
# If it's dev-only, add to dev group
```

**Q: Tests pass locally but fail in CI. Why?**

A: Common reasons:
- Different Python version (CI tests 3.10, 3.11, 3.12)
- Missing test markers (slow/heavy tests)
- File paths (use `Path` from pathlib)
- Platform differences (use `os.path.join` or `pathlib`)

Debug with:
```bash
# Test on multiple Python versions locally with tox
uv pip install tox
uv run tox

# Or use uv to test different Python versions
uv python install 3.10
uv run --python 3.10 pytest
```

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful** - Treat others with respect and consideration
- **Be constructive** - Provide helpful feedback and suggestions
- **Be collaborative** - Work together to improve the project
- **Be patient** - Remember that everyone has different skill levels and backgrounds
- **Be open-minded** - Consider other perspectives and approaches

If you encounter unacceptable behavior, please report it to the project maintainers.

---

## Release Process

(For maintainers only)

### Version Numbering

slower-whisper uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR** (x.0.0): Breaking changes to public API or JSON schema
- **MINOR** (0.x.0): New features, backward-compatible changes
- **PATCH** (0.0.x): Bug fixes, no new features

Additionally, the project maintains:
- **SCHEMA_VERSION** in `transcription/models.py`: Incremented only for breaking JSON schema changes
- **AUDIO_STATE_VERSION** in `transcription/models.py`: Incremented for changes to the `audio_state` field structure

### Pre-Release Checklist

Before creating a release, ensure:

1. **All tests pass**
   ```bash
   # Run full test suite including slow and heavy tests
   uv run pytest
   uv run pytest -m heavy
   uv run pytest -m slow

   # Check code coverage (aim for >80%)
   uv run pytest --cov=transcription --cov-report=term-missing
   ```

2. **Code quality checks pass**
   ```bash
   # Run all linters and formatters
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy transcription tests

   # Run pre-commit on all files
   uv run pre-commit run --all-files
   ```

3. **Examples and documentation are up-to-date**
   ```bash
   # Verify all example scripts work
   uv run python examples/basic_transcription.py
   uv run python examples/enrichment_workflow.py

   # Check that documentation reflects current features
   # Review README.md, docs/ARCHITECTURE.md, etc.
   ```

4. **CHANGELOG.md is updated**
   - Move "Unreleased" changes to new version section
   - Add release date
   - Ensure all changes are documented

### Creating a Release

**Step 1: Update version numbers**

```bash
# Edit pyproject.toml
# Change version = "x.y.z" to new version

# If JSON schema changed, update in transcription/models.py:
# SCHEMA_VERSION = 3  # Only if breaking changes to core fields

# If audio_state structure changed, update in transcription/models.py:
# AUDIO_STATE_VERSION = "2.0.0"  # Major.Minor.Patch
```

**Step 2: Update CHANGELOG.md**

```markdown
## [x.y.z] - YYYY-MM-DD

### Added
- New feature 1 with usage example
- New feature 2

### Changed
- Modified behavior description
- Updated dependency version

### Fixed
- Bug fix with issue reference (#123)
- Performance improvement details

### Deprecated
- Feature being phased out (will be removed in version X.Y.Z)

### Removed
- Deprecated feature that was previously warned about

### Security
- Security vulnerability fix (if applicable)
```

**Step 3: Commit version changes**

```bash
# Commit version bump and changelog
git add pyproject.toml CHANGELOG.md transcription/models.py
git commit -m "Bump version to x.y.z"

# Push to main (or release branch)
git push origin main
```

**Step 4: Create and push git tag**

```bash
# Create annotated tag
git tag -a vx.y.z -m "Release version x.y.z

Major changes:
- Feature 1
- Feature 2
- Bug fix 3"

# Push tag to GitHub
git push origin vx.y.z
```

**Step 5: Build distribution packages**

```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
uv run python -m build

# Verify the build
ls dist/
# Should show:
# slower_whisper-x.y.z-py3-none-any.whl
# slower-whisper-x.y.z.tar.gz
```

**Step 6: Test the distribution (optional but recommended)**

```bash
# Create a fresh virtual environment
python -m venv test_env
source test_env/bin/activate

# Install from the built wheel
pip install dist/slower_whisper-x.y.z-py3-none-any.whl

# Test that it works
slower-whisper --help
python -c "from transcription import api; print(api.__version__)"

# Deactivate and remove test environment
deactivate
rm -rf test_env
```

**Step 7: Publish to PyPI (if applicable)**

```bash
# Install/upgrade twine
uv pip install --upgrade twine

# Check the distribution
uv run twine check dist/*

# Upload to TestPyPI first (recommended)
uv run twine upload --repository testpypi dist/*
# Then test install: pip install --index-url https://test.pypi.org/simple/ slower-whisper

# Upload to PyPI (production)
uv run twine upload dist/*
# You'll be prompted for PyPI credentials or API token
```

**Step 8: Create GitHub Release**

1. Go to https://github.com/EffotlessMetrics/slower-whisper/releases/new
2. Select the tag you just pushed (vx.y.z)
3. Title: "Release x.y.z" or "Version x.y.z - Feature Name"
4. Description: Copy relevant sections from CHANGELOG.md
5. Attach distribution files (optional):
   - Upload `dist/slower_whisper-x.y.z-py3-none-any.whl`
   - Upload `dist/slower-whisper-x.y.z.tar.gz`
6. Check "Set as the latest release"
7. Click "Publish release"

**Step 9: Post-release tasks**

```bash
# Add "Unreleased" section back to CHANGELOG.md
```markdown
## [Unreleased]

### Added

### Changed

### Fixed
```

# Commit and push
git add CHANGELOG.md
git commit -m "Prepare CHANGELOG for next release"
git push origin main
```

### Release Types

**Patch Release (x.y.Z)** - Bug fixes only
- No new features
- No breaking changes
- Backward compatible
- Example: 1.0.0 → 1.0.1

**Minor Release (x.Y.0)** - New features
- New features added
- Backward compatible
- No breaking changes
- May include bug fixes
- Example: 1.0.1 → 1.1.0

**Major Release (X.0.0)** - Breaking changes
- Breaking changes to public API
- Breaking changes to JSON schema
- May remove deprecated features
- May include new features and bug fixes
- Example: 1.1.0 → 2.0.0

### Schema Versioning Guidelines

**SCHEMA_VERSION** (in `transcription/models.py`):
- Increment ONLY for breaking changes to core JSON schema fields
- Core fields: `file_name`, `language`, `segments`, segment `id`/`start`/`end`/`text`
- Adding optional fields does NOT require version bump
- Document migration path in CHANGELOG

**AUDIO_STATE_VERSION** (in `transcription/models.py`):
- Follow semantic versioning for `audio_state` field structure
- MAJOR: Breaking changes (removing/renaming fields)
- MINOR: New optional fields or feature additions
- PATCH: Bug fixes, no structure changes
- Example: "1.0.0" → "1.1.0" when adding new prosody features

### Hotfix Releases

For critical bugs in production:

1. Create hotfix branch from tagged release:
   ```bash
   git checkout -b hotfix/x.y.z vx.y.z-1
   ```

2. Fix the bug and add tests

3. Update version to x.y.Z (increment patch)

4. Follow normal release process

5. Merge hotfix back to main:
   ```bash
   git checkout main
   git merge hotfix/x.y.z
   git push origin main
   ```

### Release Announcement

After publishing:
- Announce on project discussion/forum
- Update documentation website (if applicable)
- Notify users of breaking changes (major releases)
- Share on social media/relevant communities (optional)

---

## Additional Resources

### Useful Documentation

- [uv documentation](https://docs.astral.sh/uv/) - Package manager
- [Ruff documentation](https://docs.astral.sh/ruff/) - Linter and formatter
- [pytest documentation](https://docs.pytest.org/) - Testing framework
- [pre-commit documentation](https://pre-commit.com/) - Git hooks

### Project-Specific Docs

- `README.md` - Project overview and installation
- `docs/QUICKSTART.md` - User quickstart guide
- `docs/AUDIO_ENRICHMENT.md` - Audio enrichment features (if exists)
- `examples/` - Working code examples
- `.github/workflows/ci.yml` - CI/CD pipeline configuration

### External Resources

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Transcription engine
- [librosa](https://librosa.org/) - Audio analysis library
- [Praat-parselmouth](https://parselmouth.readthedocs.io/) - Prosody extraction

---

## Quick Reference Commands

Here's a handy cheat sheet for common commands:

```bash
# Setup
uv sync --extra dev              # Install all dependencies
uv run pre-commit install        # Set up pre-commit hooks

# Development
uv run pytest                    # Run all tests
uv run pytest -m "not slow"      # Run fast tests only
uv run ruff check --fix .        # Lint and fix
uv run ruff format .             # Format code
uv run mypy transcription tests  # Type check

# Running the pipeline
uv run slower-whisper            # Stage 1: Transcribe
uv run slower-whisper-enrich     # Stage 2: Enrich

# Debugging
uv run pytest -vv                # Verbose test output
uv run pytest --pdb              # Debug on failure
uv run pytest -s                 # Show print statements

# Maintenance
uv sync --upgrade                # Update dependencies
uv run pre-commit run --all-files # Run all hooks manually
rm -rf .venv && uv sync --extra dev # Clean reinstall
```

---

## Thank You!

Your contributions make this project better. Whether it's:

- Reporting bugs
- Fixing typos
- Improving documentation
- Adding tests
- Implementing new features
- Reviewing pull requests
- Answering questions

**All contributions are valued and appreciated!**

We're excited to see what you'll contribute. Don't hesitate to ask questions, and welcome to the community!

Happy coding!
