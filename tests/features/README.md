# BDD Feature Tests

This directory contains Behavior-Driven Development (BDD) feature files written in Gherkin syntax for pytest-bdd.

## Overview

The BDD tests provide high-level, user-focused scenarios that test the slower-whisper pipeline through its public API. These tests are complementary to the unit tests and focus on real-world workflows.

## Feature Files

### transcription.feature
Tests for Stage 1 (Transcription) pipeline:
- Basic transcription workflows
- Single file and batch processing
- Output format verification (JSON, TXT, SRT)
- Configuration options (model, language, skip_existing)
- Multi-file handling

### enrichment.feature
Tests for Stage 2 (Audio Enrichment) pipeline:
- Prosody feature extraction
- Emotion recognition
- Full enrichment workflows
- Directory and single-file enrichment
- Error handling and graceful degradation
- Speaker-relative baseline computation

## Running BDD Tests

### Prerequisites

1. Install dev dependencies including pytest-bdd:
```bash
uv sync --extra dev
```

2. Install system dependencies:
- **ffmpeg**: Required for audio normalization
- **CUDA** (optional): For GPU acceleration

### Run all BDD tests

```bash
# Run all BDD tests
uv run pytest tests/steps/

# Run with verbose output
uv run pytest tests/steps/ -v

# Run specific feature
uv run pytest tests/steps/test_transcription_steps.py -v
uv run pytest tests/steps/test_enrichment_steps.py -v
```

### Run specific scenarios

```bash
# Run by test name (scenario gets converted to test function name)
uv run pytest tests/steps/ -k "simple_audio_file" -v

# Run transcription scenarios only
uv run pytest tests/steps/test_transcription_steps.py -v

# Run enrichment scenarios only
uv run pytest tests/steps/test_enrichment_steps.py -v
```

### Skip tests requiring optional dependencies

```bash
# Skip GPU tests
uv run pytest tests/steps/ -m "not requires_gpu"

# Skip enrichment tests (if enrichment deps not installed)
uv run pytest tests/steps/ -m "not requires_enrich"

# Skip slow tests
uv run pytest tests/steps/ -m "not slow"
```

## Test Markers

BDD tests use the same pytest markers as the rest of the test suite:

- `slow`: Tests that take significant time (can skip with `-m "not slow"`)
- `requires_gpu`: Tests requiring NVIDIA GPU
- `requires_enrich`: Tests requiring enrichment dependencies (librosa, torch, etc.)
- `integration`: Integration tests (all BDD tests are integration tests)

## Writing New Scenarios

### 1. Add scenario to feature file

Edit `transcription.feature` or `enrichment.feature`:

```gherkin
Scenario: Your new scenario
  Given some precondition
  When some action occurs
  Then verify the result
```

### 2. Implement missing steps

If pytest-bdd reports missing steps, add them to the appropriate step definition file:

- `tests/steps/test_transcription_steps.py` for transcription steps
- `tests/steps/test_enrichment_steps.py` for enrichment steps

### 3. Use the public API

Step definitions should use the public API from `transcription.api`:

```python
from transcription import (
    transcribe_directory,
    transcribe_file,
    enrich_directory,
    enrich_transcript,
    TranscriptionConfig,
    EnrichmentConfig,
)
```

### 4. Share state between steps

Use fixtures to share state:

```python
@pytest.fixture
def test_state():
    return {
        "project_root": None,
        "transcripts": [],
        "config": None,
    }

@when("I transcribe the project")
def transcribe_project(test_state):
    config = TranscriptionConfig(model="base")
    transcripts = transcribe_directory(test_state["project_root"], config)
    test_state["transcripts"] = transcripts
```

## Design Principles

1. **Public API Only**: BDD tests use only the public API, not internal modules
2. **User-Focused**: Scenarios reflect real user workflows, not implementation details
3. **Readable**: Gherkin scenarios are readable by non-programmers
4. **Isolated**: Each scenario is independent and uses temporary directories
5. **Fast Defaults**: Use smallest model ("base") and CPU for speed
6. **Graceful Skips**: Skip tests gracefully when optional dependencies unavailable

## Test Data

BDD tests create synthetic test data using:
- Silent WAV files (numpy zeros)
- Minimal audio duration (1-2 seconds)
- Temporary directories (pytest tmp_path)

No real audio files are required or committed to the repository.

## Troubleshooting

### Tests fail with "ffmpeg not found"
Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

### Tests fail with "enrichment dependencies not available"
Install enrichment dependencies:
```bash
uv sync --extra full
```

Or skip enrichment tests:
```bash
uv run pytest tests/steps/ -m "not requires_enrich"
```

### Tests are slow
Use fast model and skip slow tests:
```bash
uv run pytest tests/steps/ -m "not slow"
```

## Benefits of BDD Testing

1. **Documentation**: Feature files serve as executable documentation
2. **Stakeholder Communication**: Non-technical stakeholders can read scenarios
3. **Regression Testing**: Ensures public API stability across releases
4. **Workflow Coverage**: Tests realistic multi-step workflows
5. **Acceptance Criteria**: Scenarios define "done" for features

## Resources

- [pytest-bdd documentation](https://pytest-bdd.readthedocs.io/)
- [Gherkin syntax reference](https://cucumber.io/docs/gherkin/reference/)
- [BDD best practices](https://cucumber.io/docs/bdd/)
