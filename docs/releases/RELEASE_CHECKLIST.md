# Release Checklist for slower-whisper v1.1.0

This document provides a step-by-step checklist for releasing slower-whisper v1.1.0.

---

## Pre-Release: Code and Quality

### 1. Tests and Linting

- [x] **Run fast tests**: All pass (268 passed, 8 skipped, 19 xfailed)
  ```bash
  uv run pytest -m "not slow and not requires_gpu and not requires_diarization and not api" -q
  ```

- [x] **Lint code**: All checks pass
  ```bash
  uv run ruff check transcription/ tests/ examples/
  uv run ruff format transcription/ tests/ examples/
  ```

- [ ] **Type check** (optional, non-blocking):
  ```bash
  uv run mypy transcription/
  ```

- [ ] **Full test suite** (if GPU + diarization available):
  ```bash
  export HF_TOKEN=your_token_here
  uv run pytest -v
  ```

### 2. Version and Metadata

- [x] **Update version number** in:
  - [x] `pyproject.toml` → `version = "1.1.0"`
  - [x] `transcription/__init__.py` → `__version__ = "1.1.0"`

- [x] **Update pyproject.toml metadata**:
  - [x] Description reflects LLM integration and diarization
  - [x] Keywords include "diarization", "llm-integration", "conversation-intelligence"
  - [x] Classifiers include Python 3.13
  - [x] Entry points defined (`slower-whisper`, `slower-whisper-verify`)

- [x] **Verify CLI version flag**:
  ```bash
  uv run slower-whisper --version  # Should output: slower-whisper 1.1.0
  ```

### 3. Documentation

- [ ] **CHANGELOG.md finalized**:
  - [x] v1.1.0 section complete with all features
  - [ ] **Update release date**: Change `## [1.1.0] - TBD` to `## [1.1.0] - YYYY-MM-DD` (today's date)

- [x] **CHANGELOG.md content verified**:
  - [x] v1.1.0 section with release date
  - [x] Added: Speaker diarization, LLM integration API, examples
  - [x] Improved: CLI help, README structure
  - [x] Fixed: Speaker type consistency, linting

- [x] **README.md polished**:
  - [x] 5-minute quickstart section
  - [x] LLM integration section with examples
  - [x] Cross-links to docs and examples

- [x] **docs/INDEX.md updated**:
  - [x] LLM integration flow enhanced
  - [x] Links to new examples

- [x] **Examples documented**:
  - [x] `examples/llm_integration/README.md` exists
  - [x] `examples/llm_integration/summarize_with_diarization.py` is documented

---

## Pre-Release: Manual Testing

### 4. Example Script Verification

- [ ] **Test LLM integration example** (requires ANTHROPIC_API_KEY):
  ```bash
  # Create a test transcript (or use existing)
  mkdir -p raw_audio whisper_json
  cp test_audio.wav raw_audio/
  uv run slower-whisper transcribe --enable-diarization

  # Test the example
  export ANTHROPIC_API_KEY=your_key_here
  python examples/llm_integration/summarize_with_diarization.py whisper_json/test_audio.json
  ```

- [ ] **Verify rendering API** (quick Python test):
  ```python
  from transcription import load_transcript, render_conversation_for_llm

  transcript = load_transcript("whisper_json/test_audio.json")
  context = render_conversation_for_llm(
      transcript,
      mode="turns",
      speaker_labels={"spk_0": "Agent", "spk_1": "Customer"}
  )
  print(context)  # Should show labeled speakers and turns
  ```

### 5. Fresh Install Test

- [ ] **Test clean installation** (in a separate directory):
  ```bash
  # Clone to temp directory
  git clone https://github.com/yourusername/slower-whisper.git /tmp/slower-whisper-test
  cd /tmp/slower-whisper-test

  # Install and test basic transcription
  uv sync
  uv run slower-whisper --version
  uv run slower-whisper transcribe --help

  # Test with diarization extra
  uv sync --extra diarization
  uv run slower-whisper transcribe --enable-diarization --help
  ```

---

## Release Process

### 6. Git and GitHub

- [ ] **Commit all changes**:
  ```bash
  git status
  git add -A
  git commit -m "Release v1.1.0: Speaker diarization and LLM integration"
  ```

- [ ] **Tag the release**:
  ```bash
  git tag -a v1.1.0 -m "Release v1.1.0: Speaker diarization and LLM integration

  Major features:
  - Experimental speaker diarization with pyannote.audio
  - First-class LLM integration API (render_conversation_for_llm)
  - Speaker label mapping for human-readable output
  - Working examples in examples/llm_integration/
  - Comprehensive documentation updates

  See CHANGELOG.md for complete details."
  ```

- [ ] **Push to main and tags**:
  ```bash
  git push origin main
  git push origin v1.1.0
  ```

### 7. GitHub Release

- [ ] **Create GitHub Release** at https://github.com/yourusername/slower-whisper/releases/new:
  - **Tag version**: v1.1.0
  - **Release title**: "v1.1.0 - Speaker Diarization and LLM Integration"
  - **Description** (template below):

```markdown
# slower-whisper v1.1.0

This release adds **experimental speaker diarization** and **first-class LLM integration** to slower-whisper.

## 🎯 Highlights

### Speaker Diarization (Experimental)
- **Who spoke when**: Automatic speaker identification using pyannote.audio
- **Speaker table**: Global speakers[] with aggregated stats
- **Turn structure**: Contiguous segments grouped by speaker
- **Backend-agnostic IDs**: Normalized speaker IDs (spk_0, spk_1, ...)

📖 [Setup Guide](docs/SPEAKER_DIARIZATION.md)

### LLM Integration API
- **`render_conversation_for_llm()`**: Convert transcripts to LLM-ready text
  - Speaker labels: Map spk_0 → "Agent", "Customer", etc.
  - Audio cues: Include prosody/emotion annotations
  - Flexible modes: "turns" (speaker-grouped) or "segments"
- **`render_conversation_compact()`**: Token-efficient format for constrained contexts
- **Working examples**: `examples/llm_integration/` with Claude API integration

📖 [Prompt Patterns Guide](docs/LLM_PROMPT_PATTERNS.md)
💻 [Example Scripts](examples/llm_integration/)

## 📦 Installation

```bash
# Basic transcription only
uv sync

# Full features (diarization + prosody + emotion)
uv sync --extra full --extra diarization

# Diarization requires HuggingFace token:
export HF_TOKEN=hf_...
```

## 🚀 Quick Start (LLM Integration)

```python
from transcription import load_transcript, render_conversation_for_llm

transcript = load_transcript("whisper_json/call.json")
context = render_conversation_for_llm(
    transcript,
    mode="turns",
    speaker_labels={"spk_0": "Agent", "spk_1": "Customer"}
)

# Send to Claude, GPT, or any LLM...
```

## 📚 Documentation

- [CHANGELOG.md](CHANGELOG.md) - Complete release notes
- [README.md](README.md) - 5-minute quickstart added
- [docs/LLM_PROMPT_PATTERNS.md](docs/LLM_PROMPT_PATTERNS.md) - Reference prompts
- [docs/SPEAKER_DIARIZATION.md](docs/SPEAKER_DIARIZATION.md) - Diarization setup
- [examples/llm_integration/](examples/llm_integration/) - Working code

## ⚠️ Breaking Changes

None. This release is backward compatible with v1.0.0.

## 🐛 Bug Fixes

- Fixed speaker type consistency (all fields now use string IDs)
- Fixed linting issues with module-level imports

## 🙏 Acknowledgments

Thanks to all contributors and users who provided feedback on v1.0!

---

**Full Changelog**: https://github.com/yourusername/slower-whisper/compare/v1.0.0...v1.1.0
```

- [ ] **Attach artifacts** (optional):
  - Built wheel (see step 8)
  - Example outputs

### 8. PyPI Publication (Optional)

If publishing to PyPI:

- [ ] **Build the package**:
  ```bash
  uv build
  # or: python -m build
  ```

- [ ] **Check the package**:
  ```bash
  twine check dist/*
  ```

- [ ] **Upload to Test PyPI** (recommended first):
  ```bash
  twine upload --repository testpypi dist/*
  # Test install: pip install --index-url https://test.pypi.org/simple/ slower-whisper
  ```

- [ ] **Upload to PyPI**:
  ```bash
  twine upload dist/*
  ```

- [ ] **Verify installation**:
  ```bash
  pip install slower-whisper[full,diarization]
  slower-whisper --version  # Should show 1.1.0
  ```

---

## Post-Release

### 9. Announcement and Communication

- [ ] **Update README badges** (if applicable):
  - Version badge → 1.1.0
  - Test count (if changed)

- [ ] **Announce release** (choose platforms):
  - [ ] GitHub Discussions
  - [ ] Twitter/X
  - [ ] Reddit (r/LocalLLaMA, r/MachineLearning)
  - [ ] HuggingFace community
  - [ ] Project blog/website

- [ ] **Close related GitHub issues**:
  - Link issues to release in comments
  - Add "Released in v1.1.0" label

### 10. Prepare for v1.2

- [ ] **Update CHANGELOG.md**:
  ```markdown
  ## [Unreleased]

  ### Added
  - TBD

  ### Changed
  - TBD

  ### Fixed
  - TBD
  ```

- [ ] **Review ROADMAP.md**:
  - Mark v1.1 as complete
  - Prioritize v1.2 features

---

## Rollback Plan

If critical issues are discovered post-release:

1. **Hotfix branch**:
   ```bash
   git checkout -b hotfix/v1.1.1 v1.1.0
   # Make fixes
   git commit -m "fix: critical issue in diarization"
   ```

2. **Tag and release v1.1.1**:
   ```bash
   git tag -a v1.1.1 -m "Hotfix: ..."
   git push origin hotfix/v1.1.1
   git push origin v1.1.1
   ```

3. **Update GitHub Release and PyPI** with v1.1.1

---

## Release Sign-Off

- [ ] All checklist items completed
- [ ] Manual testing passed
- [ ] Documentation accurate
- [ ] GitHub Release published
- [ ] (Optional) PyPI package published

**Released by**: _________________
**Date**: _________________
**Notes**: _________________
