---
description: Quality pass to make the current branch PR-ready with fixes, gate verification, and readiness report
argument-hint: [optional: "minimal churn", "run full gate", "skip heavy tests", "prep for issue #N"]
allowed-tools: Bash, Read, Edit, Write, Glob, Grep, Task
---

# PR Cleanup

Make this branch PR-ready through a quality-focused cleanup pass: identify issues, apply fixes, verify the gate passes, and report readiness.

**User context:** $ARGUMENTS

## Process

### 1. Gather context

First, understand the current state:

```bash
# Branch and status
git branch --show-current
git status --short

# Base branch
git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo main

# What changed (replace BASE with actual)
git log --oneline origin/BASE..HEAD
git diff --stat origin/BASE..HEAD
```

### 2. Explore and gate (parallel)

Launch these concurrently:

**Explore agents** — understand what this branch touches:

- **Changed files**: Group by transcription core, CLI, tests, docs
- **Semantic vs mechanical**: What changes behavior vs formatting/imports
- **Interface touchpoints**: Public API, CLI flags, JSON schema fields, config options
- **Risk patterns**: Device handling, callbacks, streaming state, optional deps

Reference the repo invariants from CLAUDE.md:
- Device resolution explicit: `--device auto|cpu|cuda`
- compute_type follows resolved device
- Callbacks catch exceptions → route to `on_error`
- Streaming `end_of_stream()` finalizes turns
- Version matches package metadata
- Optional deps degrade gracefully

**Gate** — run the local CI gate:

```bash
./scripts/ci-local.sh        # full gate
# or
./scripts/ci-local.sh fast   # quick check
```

The gate covers:
1. Docs sanity: Internal markdown links, snippet correctness
2. Pre-commit: ruff lint + format, other hooks
3. Type checking: mypy on transcription/ and strategic test files
4. Fast tests: pytest -m "not slow and not heavy"
5. Verification suite: slower-whisper-verify --quick
6. Nix checks: flake check + verify app (full mode)

Capture failures for targeted fixes.

### 3. Plan fixes

Prioritize based on exploration and gate findings:

**Quick wins** (apply now):
- Format/lint issues (auto-fixable)
- Type annotation gaps on changed code
- Broken doc links
- Missing test coverage on new code paths

**Scope boundaries** (apply carefully):
- Test failures that reveal bugs in the change
- Interface clarity improvements on touched code

**Defer** (note for follow-up):
- Pre-existing issues unrelated to this change
- Refactors that expand scope
- Performance optimizations

### 4. Apply fixes

Execute fixes in order of impact:

```bash
# Auto-fix formatting
uv run ruff check --fix transcription/ tests/
uv run ruff format transcription/ tests/

# Re-run pre-commit to catch remaining issues
uv run pre-commit run --all-files
```

For manual fixes:
- Address type errors flagged by mypy
- Fix failing tests (understand root cause first)
- Update docs if behavior changed

Commit fixes with clear messages:
- `fix: resolve type errors in streaming callbacks`
- `style: apply ruff formatting`
- `test: add missing assertion for edge case`

### 5. Verify gate passes

Re-run the full gate:

```bash
./scripts/ci-local.sh
```

All checks must pass before marking ready.

### 6. Report readiness

Summarize for the user:

- **Issues found and fixed**: What was improved
- **Interface impact**: API/CLI/schema changes
- **Evidence**: Gate results, test counts
- **Remaining items**: Deferred issues with reasoning
- **PR readiness**: Ready or blocked by X

If ready, suggest running `/pr-create` with relevant context.

---

## Repo-specific guidance

### Common cleanup patterns

| Issue | Fix approach |
|-------|--------------|
| Ruff lint errors | `uv run ruff check --fix` then manual review |
| Ruff format drift | `uv run ruff format` |
| Type errors (mypy) | Add annotations to changed code; ignore pre-existing |
| Broken doc links | Update paths or remove stale references |
| Missing test coverage | Add focused tests for new behavior |

### Invariant checks

Verify these still hold after changes:

```bash
# Device resolution test
uv run pytest tests/test_device.py -v

# Callback safety test
uv run pytest tests/ -k callback -v

# Version consistency
uv run python -c "from transcription import __version__; import importlib.metadata; assert __version__ == importlib.metadata.version('slower-whisper')"
```

### Test markers

Understand what's skipped and why:
- `slow`: Long-running tests (full audio processing)
- `heavy`: Requires ML models (emotion, diarization)
- `requires_gpu`: CUDA-specific tests
- `requires_enrich`: Needs enrichment extras
- `requires_diarization`: Needs pyannote.audio

---

Now: gather context → explore + gate (parallel) → plan fixes → apply → verify → report.
