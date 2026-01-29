---
description: Quality pass to make the current branch PR-ready with fixes, gate verification, and readiness report
argument-hint: [optional: "minimal churn", "full gate", "skip heavy", "issue #N"]
---

# PR Cleanup

Make this branch PR-ready: find issues, fix them, verify the gate passes.

**User context:** $ARGUMENTS

## What to do

1. **Understand the branch**: What changed, what's the intent, what might break
2. **Run the gate**: `./scripts/ci-local.sh` to find issues
3. **Fix what's broken**: Prioritize by impact—lint/format first, then types, then tests
4. **Verify fixes**: Re-run gate until clean
5. **Report readiness**: Summarize what was fixed and current status

## Fix priorities

**Apply now:**
- Lint/format issues (`uv run ruff check --fix && uv run ruff format`)
- Type errors on changed code
- Failing tests caused by this branch
- Broken doc links

**Defer:**
- Pre-existing issues unrelated to this change
- Refactors that expand scope
- Performance work

## Commit guidance

Use clear, scoped messages:
- `fix: resolve type errors in X`
- `style: apply ruff formatting`
- `test: add missing assertion for Y`

## Readiness report

When done, summarize:
- What was found and fixed
- Gate status (passing/failing)
- Interface impact (API/CLI/schema changes)
- Any deferred items
- Ready for `/pr-create` or blocked by X

## Repo context

Gate runs: docs sanity → pre-commit → mypy → fast tests → verify suite

Key invariants from CLAUDE.md:
- Device resolution explicit
- compute_type follows resolved device
- Callbacks catch exceptions → on_error
- Streaming end_of_stream finalizes turns
- Optional deps degrade gracefully

Test markers: `slow`, `heavy`, `requires_gpu`, `requires_enrich`, `requires_diarization`
