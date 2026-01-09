---
description: Quality pass to make the current branch PR-ready with fixes, gate verification, and readiness report
argument-hint: [optional: "minimal churn", "run full gate", "skip heavy tests", "prep for issue #N"]
allowed-tools: >
  Bash(git status:*), Bash(git branch:*), Bash(git rev-parse:*), Bash(git symbolic-ref:*), Bash(git remote:*),
  Bash(git merge-base:*), Bash(git log:*), Bash(git show:*), Bash(git diff:*),
  Bash(git add:*), Bash(git restore:*), Bash(git checkout:*), Bash(git stash:*), Bash(git commit:*),
  Bash(ls:*), Bash(find:*), Bash(rg:*), Bash(sed:*), Bash(awk:*), Bash(wc:*),
  Bash(mkdir:*), Bash(cat:*), Bash(tee:*), Bash(date:*),
  Bash(pytest:*), Bash(ruff:*), Bash(mypy:*),
  Bash(uv:*), Bash(pip-audit:*),
  Bash(./scripts/*:*)
---

# PR Cleanup

Make this branch PR-ready through a quality-focused cleanup pass: identify issues, apply fixes, verify the gate passes, and report readiness.

**User context:** $ARGUMENTS

## Context (auto-collected)

- Branch: !`git branch --show-current`
- Status: !`git status --porcelain=v1 -b`
- Base branch: !`git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo main`

- Receipts directory:
  !`BR=$(git branch --show-current | tr '/ ' '__'); TS=$(date +"%Y%m%d-%H%M%S"); DIR="target/pr-cleanup/${TS}-${BR}"; mkdir -p "$DIR"; echo "$DIR" | tee target/pr-cleanup/LAST_DIR`

- Baseline summary:
  !`DIR=$(cat target/pr-cleanup/LAST_DIR); BASE=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo main); MB=$(git merge-base HEAD origin/$BASE 2>/dev/null || git merge-base HEAD $BASE); git diff --name-only $MB..HEAD > "$DIR/files_before.txt"; git diff --numstat $MB..HEAD > "$DIR/numstat_before.txt"; awk '{add=$1; del=$2; if(add=="-"||del=="-"){next} A+=add; D+=del} END{printf "%d files | +%d -%d\n", NR, A, D}' "$DIR/numstat_before.txt"`

---

## Process

### 1. Explore the change surface

Use the **Explore** agent to understand what this branch touches:

- **Changed files**: Group by transcription core, CLI, tests, docs
- **Semantic vs mechanical**: What changes behavior vs formatting/imports
- **Interface touchpoints**: Public API, CLI flags, JSON schema fields, config options
- **Risk patterns**: Device handling, callbacks, streaming state, optional deps

Reference the repo invariants (from CLAUDE.md):
- Device resolution explicit: `--device auto|cpu|cuda`
- compute_type follows resolved device
- Callbacks catch exceptions → route to `on_error`
- Streaming `end_of_stream()` finalizes turns
- Version matches package metadata
- Optional deps degrade gracefully

Save findings to `<receipts>/explore.md`.

### 2. Run the gate (identify issues)

Execute the local CI gate:

```bash
./scripts/ci-local.sh 2>&1 | tee <receipts>/gate_initial.log
```

The gate runs:
1. **Docs sanity**: Internal markdown links, snippet correctness
2. **Pre-commit**: ruff lint + format, other hooks
3. **Type checking**: mypy on transcription/ and strategic test files
4. **Fast tests**: pytest -m "not slow and not heavy"
5. **Verification suite**: slower-whisper-verify --quick
6. **Nix checks**: flake check + verify app (full mode)

Capture failures for targeted fixes.

### 3. Plan fixes

Use the **Plan** agent to prioritize:

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

Save plan to `<receipts>/fix_plan.md`.

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
./scripts/ci-local.sh 2>&1 | tee <receipts>/gate_final.log
```

All checks must pass before marking ready.

### 6. Produce readiness report

Write `<receipts>/cleanup_report.md` with:

```markdown
## Cleanup summary

What was improved and why. Focus on:
- Issues found and fixed
- Quality signals strengthened
- Scope explicitly deferred

## Interface impact

- **Public API**: unchanged | additive | breaking
- **CLI surface**: unchanged | changed
- **JSON schema**: unchanged | changed

## What changed during cleanup

Key files touched, grouped by:
- Formatting/lint fixes
- Type annotation improvements
- Test additions/fixes
- Doc updates

## Evidence

- Initial gate: X failures (see gate_initial.log)
- Final gate: ✓ all passing (see gate_final.log)
- Test summary: N passed, M skipped

## Remaining items

Issues noted but deferred (with reasoning).

## PR readiness

**Ready** | **Blocked by X**

If ready, suggest running `/pr-create` with context.
```

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
| Import sorting | Handled by ruff (I rules) |

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

## Example cleanup session

```
Explore: Branch touches streaming_transcriber.py, adds callback parameter
  - Semantic: new on_segment callback
  - Mechanical: import reordering
  - Risk: callback exception handling

Initial gate: 2 failures
  - mypy: missing return type on callback wrapper
  - test: assertion missing in test_streaming_basic

Fixes applied:
  1. Added return type annotation → streaming_transcriber.py:52
  2. Added assertion → tests/test_streaming.py:78
  3. Ran ruff format (no changes needed)

Final gate: ✓ all passing

Ready for /pr-create "Closes #42"
```

---

Now: explore → run gate → plan fixes → apply → verify → report.
