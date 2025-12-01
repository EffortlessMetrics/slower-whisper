# CI Checks & Nix Integration

This document explains how the Nix-based CI works and how to run checks locally.

The goal: **one command to run the same CI locally as in GitHub Actions**.

---

## Quick Start

```bash
# Enter Nix dev shell
nix develop

# Install Python dependencies (first time)
uv sync --extra full --extra diarization --extra dev

# Run full CI suite
nix run .#ci

# Run fast checks only (lint, format, typecheck, fast tests)
nix run .#ci -- fast
```

---

## Overview

The Nix + uv CI is structured in two layers:

### 1. Pure Offline Checks (Hermetic)

`nix flake check` runs lint + format checks using ruff from nixpkgs. These are:
- ✅ Pure/hermetic (no network, no writable state)
- ✅ Fast (~10 seconds)
- ✅ Safe for pre-commit hooks

```bash
nix flake check
```

### 2. Full CI Suite (Online, via Apps)

`nix run .#ci` orchestrates all checks via a shell script running inside the Nix dev environment. This includes:
- Lint + format (using uv-installed ruff)
- Type-check (mypy)
- Fast tests (pytest, no slow/heavy markers)
- Integration tests (full mode only)
- BDD scenarios (full mode only)
- Verification suite (full mode only, requires HF_TOKEN)
- Dogfood smoke (full mode only)

```bash
# Fast mode (lint, format, typecheck, fast tests)
nix run .#ci -- fast

# Full mode (all checks)
nix run .#ci -- full  # or just: nix run .#ci
```

---

## Architecture

CI is defined in two places:

- **`flake.nix`**:
  - `devShells`: Reproducible dev environment (Python, uv, ffmpeg, etc.)
  - `checks`: Pure lint + format checks (offline)
  - `apps.ci`: CI orchestrator script
  - `apps.dogfood` / `apps.verify`: Convenience wrappers

- **`.github/workflows/ci-nix.yml`**:
  - `nix-pure`: Runs `nix flake check` (fast, hermetic)
  - `nix-ci-fast`: Runs `nix run .#ci -- fast` (~5-10 minutes)
  - `nix-ci-full`: Runs `nix run .#ci -- full` (~30-60 minutes)

The CI script:
1. Checks if `.venv` exists (runs `uv sync` if not)
2. Runs each check in sequence
3. Tracks failures and emits a summary
4. Returns exit code 0 (all passed) or 1 (any failed)

---

## CI Modes

| Mode   | What it runs                                         | Duration | When to use                     |
| ------ | ---------------------------------------------------- | -------- | ------------------------------- |
| `fast` | Lint, format, typecheck, fast tests                  | ~5 min   | During development, pre-commit  |
| `full` | Fast mode + integration, BDD, verify, dogfood        | ~30 min  | Before PR, before release       |

### Verify workflow split

- PRs run `.github/workflows/verify.yml` in quick mode via `nix develop --command uv run slower-whisper-verify --quick`.
- Nightly schedule/manual dispatch runs the same workflow in full mode (`uv run slower-whisper-verify`), which includes Docker + K8s checks and benefits from `HF_TOKEN` when running real pyannote.
- Both paths share the Nix dev shell, so local runs match CI output.

### Fast Mode

```bash
nix run .#ci -- fast
```

Runs:
1. **Lint** (ruff check)
2. **Format** (ruff format --check)
3. **Type-check** (mypy, warnings allowed)
4. **Fast tests** (pytest -m "not slow and not heavy")

Skips: integration tests, BDD, verify, dogfood.

### Full Mode

```bash
nix run .#ci  # or: nix run .#ci -- full
```

Runs everything in fast mode, plus:
5. **Integration tests** (pytest *integration*.py)
6. **BDD library** (pytest tests/steps/)
7. **BDD API** (pytest features/)
8. **Verify** (slower-whisper-verify --quick, requires HF_TOKEN)
9. **Dogfood smoke** (synthetic audio, no LLM)

---

## Environment Setup

### Nix Dev Shell

The dev shell provides:
- Python 3.12
- uv (package manager)
- ffmpeg, libsndfile, portaudio (audio processing)
- ruff, mypy (code quality)
- git, jq, curl (utilities)

Environment variables set automatically:
- `UV_PYTHON`: Points to Nix's Python
- `UV_PROJECT_ENVIRONMENT`: Uses `.venv` in project root
- `UV_CACHE_DIR`: Uses `.cache/uv` in project root
- `SLOWER_WHISPER_CACHE_ROOT`: Uses `~/.cache/slower-whisper`

Enter the shell:

```bash
nix develop
```

### Python Dependencies

Inside the dev shell:

```bash
# Full install (recommended for CI/development)
uv sync --extra full --extra diarization --extra dev

# Minimal install (ASR only, no enrichment)
uv sync --extra dev
```

---

## HF_TOKEN, pyannote mode, & HuggingFace Cache

Some checks (verify, dogfood) pull pyannote diarization models from HuggingFace.
Set `HF_TOKEN` when running the real backend; the lightweight stub backend works
without it.

### Local Setup

1. Create a HuggingFace account and token: https://huggingface.co/settings/tokens
2. Accept terms for pyannote models:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
3. Export the token:

```bash
export HF_TOKEN=hf_...
# Optional: force stub/missing diarization in local runs
export SLOWER_WHISPER_PYANNOTE_MODE=stub  # or "missing" to simulate absence
```

Then run CI:

```bash
nix run .#ci  # full mode will use HF_TOKEN
```

Without `HF_TOKEN`, the verify check is skipped (other checks still run).

### CI (GitHub Actions)

The workflow uses the `HF_TOKEN` secret from GitHub repository settings. It also caches:
- `~/.cache/huggingface` (model downloads)
- `.venv` (Python packages)
- `.cache/uv` (uv cache)

---

## Workflow Comparison

### Traditional uv-only (manual)

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy transcription/
uv run pytest -m "not slow and not heavy"
# ... (manually run each check)
```

### Nix + uv (automated)

```bash
nix run .#ci -- fast
```

**Advantages**:
- ✅ One command runs all checks
- ✅ Same environment as CI (reproducible)
- ✅ Structured output (✓/✗ per check)
- ✅ Failure summary at the end
- ✅ Can run offline (`nix flake check`)

---

## Local Development Workflow

### Day-to-day coding

```bash
# Enter shell once per session
nix develop

# Make changes, run specific tools
uv run pytest tests/test_models.py
uv run ruff check transcription/
uv run mypy transcription/api.py

# Before committing
nix run .#ci -- fast
```

### Before opening a PR

```bash
# Full suite (with HF_TOKEN if you have it)
export HF_TOKEN=hf_...
nix run .#ci
```

### Before tagging a release

```bash
# Full suite + pure checks
nix flake check
nix run .#ci
```

---

## Troubleshooting

### "Could not find Python"

The dev shell sets `UV_PYTHON` automatically. If you see this error outside `nix develop`, enter the shell first:

```bash
nix develop
nix run .#ci -- fast
```

### "No module named 'pytest'"

Run `uv sync` to install Python dependencies:

```bash
uv sync --extra dev
```

### "HF_TOKEN not set" (verify skipped)

This is expected if you don't have a HuggingFace token. The verify check is optional in fast mode. To enable it:

1. Get a token from https://huggingface.co/settings/tokens
2. Accept pyannote model terms
3. Export the token:

```bash
export HF_TOKEN=hf_...
nix run .#ci
```

### Test failures unrelated to code changes

Some tests may fail due to missing dependencies or environment issues. Check:

1. **Missing Python packages**: Run `uv sync --extra full --extra diarization --extra dev`
2. **Missing system packages**: The Nix dev shell provides all required system deps
3. **Stale .venv**: Delete `.venv` and re-run `uv sync`

### Nix flake check fails but uv ruff passes

The pure checks use ruff from nixpkgs (0.7.x), which may have different defaults than uv's ruff. For consistency, use `nix run .#ci -- fast` which uses the same ruff as CI.

---

## FAQ

**Q: Do I need to use Nix?**
**A:** For contributors and CI-adjacent work, yes. For casual users or quick experiments, traditional `uv sync` + manual testing still works.

**Q: Can I run individual checks?**
**A:** Yes, inside `nix develop`:

```bash
uv run ruff check .
uv run pytest -m "not slow"
```

**Q: Why is CI split into fast/full modes?**
**A:** Fast mode (~5 min) catches most issues during development. Full mode (~30 min) runs expensive checks (model downloads, integration tests) that are only needed before PR/release.

**Q: Does nix run .#ci download models every time?**
**A:** No. Models are cached in `~/.cache/huggingface`. Only the first run downloads them (~3-5GB).

**Q: Can I skip specific checks?**
**A:** Not currently. The CI script runs all checks in the selected mode. You can run checks manually via `uv run` if needed.

---

## Summary

| Command                   | What it does                          | Use case                 |
| ------------------------- | ------------------------------------- | ------------------------ |
| `nix flake check`         | Pure lint + format (offline)          | Pre-commit, quick sanity |
| `nix run .#ci -- fast`    | Lint, format, typecheck, fast tests   | During development       |
| `nix run .#ci`            | Full test suite                       | Before PR, before tag    |
| `nix run .#verify`        | Run verification spine                | Dogfooding, smoke test   |
| `nix run .#dogfood`       | Run dogfood workflow                  | Testing with real audio  |

**Bottom line**: `nix run .#ci -- fast` is your main local CI button. It mirrors what GitHub Actions runs, ensuring "if it passes locally, it'll pass in CI."
