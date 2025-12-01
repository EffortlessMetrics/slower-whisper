# Typing Policy

This document describes the typing standards and strategy for slower-whisper.

## Overview

slower-whisper uses **gradual typing** — a pragmatic approach that maximizes type safety where it matters most (public APIs, data models) while allowing flexibility in test code and internal utilities.

**Current Status (v1.3.1+):**
- 39/39 package modules pass mypy
- 92.9% function-level annotation coverage (209/225 functions)
- 4 strategic test modules are type-checked
- PEP 561 compliant (`py.typed` marker present)

## Configuration

### mypy (pyproject.toml)

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false      # Gradual: allows untyped defs
disallow_incomplete_defs = false   # Allows partial annotations
check_untyped_defs = true          # BUT validates their usage
no_implicit_optional = true
strict_equality = true
```

### Pre-commit

mypy runs in **non-blocking** mode (`|| true`) to avoid friction during rapid development. CI enforces the same checks.

### VS Code / Pylance

`pyrightconfig.json` provides IDE support with `typeCheckingMode: "standard"`.

## Typed Boundaries

### Fully Typed (Hard Contract)

These modules have complete type coverage and are part of the public API contract:

| Module | Purpose |
|--------|---------|
| `api.py` | Public API functions |
| `models.py` | Data models (Segment, Transcript, Turn) |
| `config.py` | Configuration dataclasses |
| `llm_utils.py` | LLM rendering functions |
| `turn_helpers.py` | Turn conversion utilities |
| `exporters.py` | Export format functions |
| `validation.py` | Schema validation |
| `types_audio.py` | TypedDict definitions for audio state |

### Protocol-Based (Optional Dependencies)

These modules use protocols to handle optional dependencies gracefully:

| Module | Protocol | Purpose |
|--------|----------|---------|
| `emotion.py` | `EmotionRecognizerLike` | ML emotion recognition |
| `asr_engine.py` | `WhisperModelProtocol` | faster-whisper wrapper |
| `semantic.py` | `SemanticAnnotator` | Semantic tagging |
| `diarization.py` | Environment-based stub | Speaker diarization |

### Test Modules (Strategic)

Only these test modules are type-checked:

- `test_llm_utils.py` — validates LLM rendering API
- `test_writers.py` — validates JSON I/O contracts
- `test_turn_helpers.py` — validates Turn conversion
- `test_audio_state_schema.py` — validates audio state structure

Other test files are intentionally untyped (pytest fixtures, dynamic generation).

## Patterns

### Protocol + Factory for Optional Dependencies

```python
from typing import Protocol, cast, Any

# 1. Define protocol
class EmotionRecognizerLike(Protocol):
    def extract_emotion_dimensional(
        self, audio: NDArray[np.floating[Any]], sr: int
    ) -> dict[str, dict[str, Any]]: ...

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
        return {"valence": {"level": "neutral", "score": 0.5}}

# 4. Factory returns appropriate implementation
def get_emotion_recognizer() -> EmotionRecognizerLike:
    if EMOTION_AVAILABLE:
        return EmotionRecognizer()
    return DummyEmotionRecognizer()
```

### TypedDict for Runtime Schema

```python
from typing import TypedDict

class PitchState(TypedDict, total=False):
    mean_hz: float
    level: str
    contour: str
    std_hz: float
```

### Documenting cast() Usage

Always explain why `cast()` is needed:

```python
# cast() needed because faster-whisper types are incomplete
model = cast(WhisperModelProtocol, WhisperModel(model_size))
```

## Guidelines

### DO

- Add type annotations to all new functions and classes
- Use protocols for optional dependency interfaces
- Use TypedDict for structured dict schemas
- Document cast() with comments
- Run mypy before committing: `uv run mypy transcription/`

### DON'T

- Add `# type: ignore` without a comment explaining why
- Use `Any` in return types of public API functions
- Remove type annotations during refactoring
- Skip mypy errors without understanding them

### Acceptable Any Usage

- Third-party library types that lack stubs (faster-whisper, parselmouth)
- Internal dict structures that are transformed before reaching public API
- Test fixtures and mock objects

## Verification

```bash
# Local (same as CI)
uv run mypy transcription/ \
  tests/test_llm_utils.py \
  tests/test_writers.py \
  tests/test_turn_helpers.py \
  tests/test_audio_state_schema.py

# Quick verification (includes type checking)
uv run slower-whisper-verify --quick
```

## Roadmap

### Current (v1.3.1)
- Gradual typing with 92.9% coverage
- Non-blocking pre-commit
- CI enforcement on core modules

### Future (v2.0+)
- Strict mode on public API modules
- py.typed + inline annotations for downstream consumers
- Type stub generation for external tools
