---
description: Create a PR from the current branch with narrative summary, quality signals, and verification receipts
argument-hint: [optional: "Issue #N", "draft", "ready", "base=main"]
allowed-tools: >
  Bash(git status:*), Bash(git branch:*), Bash(git rev-parse:*), Bash(git symbolic-ref:*), Bash(git remote:*),
  Bash(git merge-base:*), Bash(git log:*), Bash(git show:*), Bash(git diff:*),
  Bash(ls:*), Bash(find:*), Bash(rg:*), Bash(sed:*), Bash(awk:*), Bash(wc:*),
  Bash(mkdir:*), Bash(cat:*), Bash(tee:*), Bash(date:*),
  Bash(gh:*),
  Bash(pytest:*), Bash(ruff:*), Bash(mypy:*),
  Bash(uv:*), Bash(pip-audit:*),
  Bash(./scripts/*:*)
---

# Create PR

Create a pull request from the current branch, crafting a narrative that helps reviewers understand intent, changes, and evidence.

**User context:** $ARGUMENTS

## Context (auto-collected)

- Branch: !`git branch --show-current`
- Status: !`git status --porcelain=v1 -b`
- Base branch: !`git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo main`

- Receipts directory:
  !`BR=$(git branch --show-current | tr '/ ' '__'); TS=$(date +"%Y%m%d-%H%M%S"); DIR="target/pr-create/${TS}-${BR}"; mkdir -p "$DIR"; echo "$DIR" | tee target/pr-create/LAST_DIR`

- Change summary:
  !`DIR=$(cat target/pr-create/LAST_DIR); BASE=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo main); MB=$(git merge-base HEAD origin/$BASE 2>/dev/null || git merge-base HEAD $BASE); git diff --name-only $MB..HEAD > "$DIR/files.txt"; git diff --numstat $MB..HEAD > "$DIR/diff_numstat.txt"; git log --oneline $MB..HEAD > "$DIR/commits.txt"; awk '{add=$1; del=$2; if(add=="-"||del=="-"){next} A+=add; D+=del} END{printf "%d files | +%d -%d\n", NR, A, D}' "$DIR/diff_numstat.txt"`

---

## Process

### 1. Understand the change

Use the **Explore** agent to map what changed:

- **Semantic hotspots**: Where does behavior change? (vs mechanical: formatting, imports, renames)
- **Interface touchpoints**: Public API, CLI flags, config options, JSON schema fields
- **Risk patterns**: Device resolution, callback handling, streaming state, optional dependency paths
- **Key invariants to verify** (from CLAUDE.md):
  - Device resolution remains explicit (`--device auto|cpu|cuda`)
  - compute_type follows resolved device
  - Callbacks handle exceptions gracefully
  - Streaming `end_of_stream()` finalizes turns correctly
  - Version string matches package metadata

Save findings to `<receipts>/explore_findings.md`.

### 2. Run verification

Execute the local gate and capture results:

```bash
# Full gate (preferred)
./scripts/ci-local.sh 2>&1 | tee <receipts>/gate.log

# Or fast mode if full already passed
./scripts/ci-local.sh fast 2>&1 | tee <receipts>/gate_fast.log
```

The gate covers:
- Docs sanity (internal links, snippets)
- Pre-commit (ruff lint + format)
- Type checking (mypy on transcription/ + strategic test files)
- Fast tests (pytest -m "not slow and not heavy")
- Verification suite (slower-whisper-verify --quick)
- Nix checks (full mode only)

If tests fail, understand and fix before proceeding.

### 3. Compose the PR

Write `<receipts>/pr_title.txt` and `<receipts>/pr_body.md`:

**Title**: Concise, imperative (e.g., "feat: add streaming callback for segment events")

**Body structure**:

```markdown
## Summary

1-3 paragraphs: what changed, why, and what should be true after merge.
Reference issues with "Closes #N" or "Relates to #N".

## What changed

Narrative explanation at system level. Group by:
- Core behavior changes
- API/CLI surface changes
- Test/docs updates

Avoid file-by-file lists; focus on conceptual changes.

## Interface impact

- **Public API**: unchanged | additive | breaking
- **CLI surface**: unchanged | new flags | changed behavior
- **JSON schema**: unchanged | new fields | changed semantics
- **Config options**: unchanged | new | changed

## How to review

Suggest a review path:
1. Start with X to understand the core change
2. Then look at Y for the interface contract
3. Z contains the tests that verify the behavior

Highlight the semantic hotspots vs mechanical changes.

## Evidence

What verification ran and what it proves:
- Gate: `./scripts/ci-local.sh` ✓ (see gate.log)
- Tests: N tests passed, M skipped (markers: heavy, requires_gpu)
- Coverage: unchanged | +X% on touched files

Reproduction command for reviewers:
```
uv run pytest tests/test_<specific>.py -v
```

## Risk assessment

- Blast radius: what could break
- Rollback: how to revert if needed
- Monitoring: what to watch post-merge

## Follow-ups

Explicit deferrals or future work this enables.
```

### 4. Create the PR

```bash
# Default: draft PR
gh pr create --draft --title "$(cat <receipts>/pr_title.txt)" --body-file <receipts>/pr_body.md

# If user specified "ready": create as ready for review
gh pr create --title "$(cat <receipts>/pr_title.txt)" --body-file <receipts>/pr_body.md
```

Save the command to `<receipts>/pr_cmd.txt` and the resulting URL to `<receipts>/pr_url.txt`.

---

## Quality signals to highlight

When composing the PR, emphasize these signals reviewers care about:

| Signal | What to show |
|--------|--------------|
| Interface stability | API/CLI/schema changes with before/after |
| Verification depth | Gate results, test coverage on changed code |
| Risk surface | Device/concurrency/IO/deps changes |
| Maintainability | Boundary clarity, complexity changes |

---

## Example PR body (abbreviated)

```markdown
## Summary

Adds `on_segment` callback to streaming transcription, enabling real-time segment
notifications without polling. Closes #42.

The callback receives finalized segments only (after VAD confirms end-of-speech),
maintaining the invariant that streaming `end_of_stream()` handles turn finalization.

## What changed

- **StreamingTranscriber**: New `on_segment: Callable[[Segment], None]` parameter
- **Callback safety**: Exceptions caught and routed to `on_error`, pipeline continues
- **Tests**: 3 new tests covering callback invocation, error handling, async context

## Interface impact

- **Public API**: additive (new optional parameter)
- **CLI surface**: unchanged (callback is programmatic only)
- **JSON schema**: unchanged

## How to review

1. `streaming_transcriber.py:45-78` - callback integration point
2. `tests/test_streaming_callbacks.py` - behavior specification
3. `docs/STREAMING_ARCHITECTURE.md` - updated sequence diagram

## Evidence

- Gate: `./scripts/ci-local.sh` ✓
- Tests: 142 passed, 8 skipped (heavy)
- Callback error path verified with fault injection test
```

---

Now: explore the change, run verification, compose the PR narrative, and create it.
