# Release Checklist for slower-whisper v2.0.0

This document provides a step-by-step checklist for releasing slower-whisper v2.0.0.

---

## Overview

v2.0.0 is a major release introducing:
- WebSocket streaming API with resume protocol
- REST session management endpoints
- Reference Python streaming client
- Benchmark infrastructure (ASR, diarization, emotion, streaming)
- Semantic adapter protocol with LLM providers
- LLM guardrails (rate limits, cost tracking, PII detection)

---

## Pre-Release: Code and Quality

### 1. Tests and Linting

- [ ] **Run fast tests**: All pass
  ```bash
  uv run pytest -m "not slow and not requires_gpu and not requires_diarization and not api" -q
  ```

- [ ] **Lint code**: All checks pass
  ```bash
  uv run ruff check transcription/ tests/ examples/
  uv run ruff format --check transcription/ tests/ examples/
  ```

- [ ] **Type check**:
  ```bash
  uv run mypy transcription/
  ```

- [ ] **Full local gate**:
  ```bash
  ./scripts/ci-local.sh
  ```

- [ ] **Full test suite** (if GPU + diarization available):
  ```bash
  export HF_TOKEN=your_token_here
  uv run pytest -v
  ```

### 2. Version Bump

**Location:** `pyproject.toml` (single source of truth)

The version is single-sourced from `pyproject.toml` via `importlib.metadata`. Only one change is needed:

- [ ] **Update version in `pyproject.toml`**:
  ```toml
  version = "2.0.0"
  ```

- [ ] **Verify version constant**:
  ```bash
  uv run python -c "import transcription; print(transcription.__version__)"
  # Should output: 2.0.0
  ```

- [ ] **Verify CLI version flag**:
  ```bash
  uv run slower-whisper --version
  # Should output: slower-whisper 2.0.0
  ```

### 3. Documentation Verification

- [ ] **CHANGELOG.md updated**:
  - [x] v2.0.0 section with all features
  - [ ] **Update release date**: Verify `## [2.0.0] - 2026-01-26` is correct date

- [ ] **ROADMAP.md updated**:
  - [ ] Version updated to v2.0.0
  - [ ] Track 1, 2, 3 marked as complete
  - [ ] Recently Shipped section updated

- [ ] **README.md current**:
  - [ ] Version references updated
  - [ ] Streaming section added/updated
  - [ ] LLM providers section updated

- [ ] **docs/STREAMING_API.md complete**:
  - [ ] WebSocket protocol documented
  - [ ] REST endpoints documented
  - [ ] Python client documented
  - [ ] Code examples working

### 4. Module Health Checks

- [ ] **Verify all new exports**:
  ```bash
  uv run python -c "
  # Streaming exports
  from transcription import (
      WebSocketStreamingSession,
      WebSocketSessionConfig,
      EventEnvelope,
      ClientMessageType,
      ServerMessageType,
      SessionState,
      SessionStats,
  )
  print('Streaming exports OK')
  "
  ```

- [ ] **Verify streaming client**:
  ```bash
  uv run python -c "
  from transcription import (
      StreamingClient,
      StreamingConfig,
      ClientState,
      ClientStats,
      create_client,
  )
  print('Streaming client exports OK')
  "
  ```

- [ ] **Verify semantic adapter**:
  ```bash
  uv run python -c "
  from transcription import (
      SemanticAdapter,
      SemanticAnnotation,
      NormalizedAnnotation,
      ActionItem,
      ChunkContext,
      ProviderHealth,
      create_adapter,
      SEMANTIC_SCHEMA_VERSION,
  )
  print('Semantic adapter exports OK')
  "
  ```

- [ ] **Verify LLM guardrails**:
  ```bash
  uv run python -c "
  from transcription import (
      LLMGuardrails,
      GuardedLLMProvider,
      GuardrailStats,
      PIIMatch,
      RateLimitExceeded,
      CostBudgetExceeded,
      RequestTimeout,
      create_guarded_provider,
  )
  print('LLM guardrails exports OK')
  "
  ```

### 5. Benchmark Verification

- [ ] **Run benchmark suite**:
  ```bash
  # List available tracks
  uv run slower-whisper benchmark list

  # Check infrastructure status
  uv run slower-whisper benchmark status

  # Run ASR benchmark (requires dataset)
  uv run slower-whisper benchmark run --track asr --dataset smoke

  # Compare against baselines
  uv run slower-whisper benchmark compare --track asr
  ```

- [ ] **Verify benchmark targets** (if applicable):
  - [ ] ASR WER < 5% (LibriSpeech test-clean)
  - [ ] Diarization DER < 15% (AMI)
  - [ ] Streaming P95 < 500ms
  - [ ] Semantic Topic F1 > 0.80

---

## Pre-Release: Manual Testing

### 6. Streaming API Verification

- [ ] **Test WebSocket streaming** (requires running server):
  ```bash
  # Start server
  uv run slower-whisper api --host 0.0.0.0 --port 8000

  # In another terminal, test with Python client
  uv run python -c "
  import asyncio
  from transcription.streaming_client import create_client

  async def test():
      client = create_client('ws://localhost:8000/stream')
      async with client:
          await client.start_session()
          print(f'Session started: {client.stream_id}')
          # Send test audio if available

  asyncio.run(test())
  "
  ```

- [ ] **Test REST session management**:
  ```bash
  # Start server (if not running)
  uv run slower-whisper api --host 0.0.0.0 --port 8000 &

  # Create session
  curl -X POST "http://localhost:8000/stream/sessions" \
       -H "Content-Type: application/json"

  # List sessions
  curl "http://localhost:8000/stream/sessions"

  # Get session status
  curl "http://localhost:8000/stream/sessions/{session_id}"
  ```

### 7. Fresh Install Test

- [ ] **Test clean installation** (in a separate directory):
  ```bash
  # Clone to temp directory
  cd /tmp
  git clone https://github.com/EffortlessMetrics/slower-whisper.git slower-whisper-test
  cd slower-whisper-test

  # Install and test
  uv sync
  uv run slower-whisper --version
  uv run slower-whisper transcribe --help

  # Test with full extras
  uv sync --extra full
  uv run python -c "from transcription import StreamingClient; print('OK')"
  ```

### 8. Wheel Build Test

- [ ] **Build wheel**:
  ```bash
  uv build
  # or: python -m build
  ```

- [ ] **Test wheel installation**:
  ```bash
  # Create fresh venv
  python -m venv /tmp/test-wheel-venv
  source /tmp/test-wheel-venv/bin/activate

  # Install wheel
  pip install dist/slower_whisper-2.0.0-py3-none-any.whl

  # Verify
  python -c "import transcription; print(transcription.__version__)"
  slower-whisper --version

  deactivate
  ```

---

## Release Process

### 9. Git and GitHub

- [ ] **Commit all changes**:
  ```bash
  git status
  git add pyproject.toml CHANGELOG.md ROADMAP.md
  git commit -m "$(cat <<'EOF'
  chore: prepare v2.0.0 release

  - Bump version to 2.0.0
  - Update CHANGELOG.md with release notes
  - Update ROADMAP.md with shipped status
  EOF
  )"
  ```

- [ ] **Tag the release**:
  ```bash
  git tag -a v2.0.0 -m "$(cat <<'EOF'
  Release v2.0.0: Streaming Protocol + Benchmarks + Semantic Adapters

  Major features:
  - WebSocket streaming API with resume protocol
  - REST session management endpoints
  - Reference Python streaming client
  - Benchmark infrastructure (ASR, diarization, emotion, streaming)
  - Semantic adapter protocol with LLM providers
  - LLM guardrails (rate limits, cost tracking, PII detection)

  Breaking changes:
  - Removed deprecated --enrich-config flag (use --config)
  - Removed legacy entry points and scripts

  See CHANGELOG.md for complete details.
  EOF
  )"
  ```

- [ ] **Push to main and tags**:
  ```bash
  git push origin main
  git push origin v2.0.0
  ```

### 10. GitHub Release

- [ ] **Create GitHub Release** at https://github.com/EffortlessMetrics/slower-whisper/releases/new:
  - **Tag version**: v2.0.0
  - **Release title**: "v2.0.0 - Streaming Protocol + Benchmarks + Semantic Adapters"
  - **Description**: Use template below

```markdown
# slower-whisper v2.0.0

This release introduces **real-time streaming**, **benchmark infrastructure**, and **semantic analysis** capabilities.

## Highlights

### WebSocket Streaming API
- **Real-time transcription** with bidirectional WebSocket communication
- **Resume protocol** for recovering from network interruptions
- **Backpressure handling** with guaranteed delivery of finalized events
- **Reference Python client** with async iteration and callbacks

[Streaming API Documentation](docs/STREAMING_API.md)

### REST Session Management
- Create and manage streaming sessions via REST endpoints
- Monitor session status and statistics
- Force close sessions when needed

### Benchmark Infrastructure
- **ASR evaluation**: WER/CER computation with jiwer
- **Diarization evaluation**: DER/JER metrics with pyannote.metrics
- **Emotion evaluation**: Accuracy, F1, confusion matrix
- **Streaming evaluation**: P50/P95/P99 latency, RTF
- **Baseline comparison**: Regression detection with configurable thresholds

[Benchmark Documentation](docs/BENCHMARKS.md)

### Semantic Adapter Protocol
- Pluggable semantic analysis backends
- OpenAI and Anthropic LLM providers
- Guardrails for rate limits, cost tracking, and PII detection

## Installation

```bash
# Basic installation
pip install slower-whisper

# With all extras
pip install slower-whisper[full]

# Development
uv sync --extra dev
```

## Quick Start (Streaming)

```python
import asyncio
from transcription import create_client

async def main():
    client = create_client("ws://localhost:8000/stream")
    async with client:
        await client.start_session()
        # Send audio chunks...
        await client.end_session()

asyncio.run(main())
```

## Breaking Changes

- Removed deprecated `--enrich-config` flag (use `--config`)
- Removed legacy entry points (`slower-whisper-enrich`)
- Removed legacy scripts (`transcribe_pipeline.py`, `audio_enrich.py`)

## Documentation

- [CHANGELOG.md](CHANGELOG.md) - Complete release notes
- [docs/STREAMING_API.md](docs/STREAMING_API.md) - Streaming API reference
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) - Benchmark CLI reference
- [docs/SEMANTIC_BENCHMARK.md](docs/SEMANTIC_BENCHMARK.md) - Semantic evaluation

**Full Changelog**: https://github.com/EffortlessMetrics/slower-whisper/compare/v1.9.2...v2.0.0
```

### 11. PyPI Publication

- [ ] **Build the package**:
  ```bash
  rm -rf dist/
  uv build
  ```

- [ ] **Check the package**:
  ```bash
  twine check dist/*
  ```

- [ ] **Upload to Test PyPI** (recommended first):
  ```bash
  twine upload --repository testpypi dist/*

  # Test install
  pip install --index-url https://test.pypi.org/simple/ \
              --extra-index-url https://pypi.org/simple/ \
              slower-whisper
  ```

- [ ] **Upload to PyPI**:
  ```bash
  twine upload dist/*
  ```

- [ ] **Verify installation**:
  ```bash
  pip install slower-whisper==2.0.0
  slower-whisper --version  # Should show 2.0.0
  ```

---

## Post-Release

### 12. Announcement and Communication

- [ ] **Update README badges** (if applicable):
  - Version badge to 2.0.0
  - PyPI badge updated

- [ ] **Announce release** (choose platforms):
  - [ ] GitHub Discussions
  - [ ] Twitter/X
  - [ ] Reddit (r/LocalLLaMA, r/MachineLearning)
  - [ ] HuggingFace community

- [ ] **Close related GitHub issues**:
  - Link issues to release in comments
  - Add "Released in v2.0.0" label

### 13. Prepare for v2.0.1

- [ ] **CHANGELOG.md prepared**:
  - [Unreleased] section reset

- [ ] **ROADMAP.md updated**:
  - v2.0.0 marked as released
  - v2.1 planning started

---

## Rollback Plan

If critical issues are discovered post-release:

1. **Hotfix branch**:
   ```bash
   git checkout -b hotfix/v2.0.1 v2.0.0
   # Make fixes
   git commit -m "fix: critical issue"
   ```

2. **Tag and release v2.0.1**:
   ```bash
   git tag -a v2.0.1 -m "Hotfix: ..."
   git push origin hotfix/v2.0.1
   git push origin v2.0.1
   ```

3. **Update GitHub Release and PyPI** with v2.0.1

---

## Release Sign-Off

- [ ] All checklist items completed
- [ ] Local gate passes
- [ ] Manual testing passed
- [ ] Documentation accurate
- [ ] GitHub Release published
- [ ] PyPI package published

**Released by**: _________________
**Date**: _________________
**Notes**: _________________

---

## Quick Reference: Release Commands

```bash
# Pre-release verification
./scripts/ci-local.sh
uv run pytest -m "not slow" -q
uv run mypy transcription/

# Version check
uv run python -c "import transcription; print(transcription.__version__)"

# Build
uv build
twine check dist/*

# Tag and push
git tag -a v2.0.0 -m "Release v2.0.0"
git push origin main
git push origin v2.0.0

# Publish
twine upload --repository testpypi dist/*  # Test first
twine upload dist/*                         # Production
```
