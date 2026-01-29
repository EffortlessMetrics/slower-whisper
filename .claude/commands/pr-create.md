---
description: Create a PR from the current branch with narrative summary, quality signals, and verification receipts
argument-hint: [optional: "Issue #N", "draft", "ready", "base=main"]
allowed-tools: Bash, Read, Glob, Grep, Task, Write
---

# Create PR

Create a pull request from the current branch, crafting a narrative that helps reviewers understand intent, changes, and evidence.

**User context:** $ARGUMENTS

## Process

### 1. Gather context

First, collect the essential git state:

```bash
# Current branch and status
git branch --show-current
git status --short

# Determine base branch
git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo main

# Commits since base (replace BASE with actual base branch)
git log --oneline origin/BASE..HEAD

# Changed files summary
git diff --stat origin/BASE..HEAD
```

Check if the branch is pushed and up-to-date with remote.

### 2. Understand the change

Use **Explore** agents to map the change surface. Focus on:

- **Semantic hotspots**: Where does behavior change? (vs mechanical: formatting, imports, renames)
- **Interface touchpoints**: Public API, CLI flags, config options, JSON schema fields
- **Risk patterns**: Device resolution, callback handling, streaming state, optional dependency paths

Reference CLAUDE.md for key invariants that should be verified.

### 3. Run verification (if not already done)

Execute the local gate:

```bash
./scripts/ci-local.sh        # full gate
# or
./scripts/ci-local.sh fast   # quick check
```

If the gate fails, understand and fix before proceeding.

### 4. Compose the PR

**Title**: Concise, imperative, under 70 chars (e.g., "feat: add streaming callback for segment events")

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

## How to review

Suggest a review path:
1. Start with X to understand the core change
2. Then look at Y for the interface contract
3. Z contains the tests that verify the behavior

## Test plan

- [ ] Gate verification: `./scripts/ci-local.sh`
- [ ] Specific tests to run or verify
- [ ] Manual verification steps if applicable
```

### 5. Create the PR

Use `gh pr create` with the composed title and body. Use a HEREDOC for the body:

```bash
gh pr create --title "the title" --body "$(cat <<'EOF'
## Summary
...

## Test plan
...
EOF
)"
```

- Default to `--draft` unless user specified "ready"
- If user specified a base branch, use `--base <branch>`

Report the PR URL when complete.

---

## Quality signals

When composing the PR, emphasize:

| Signal | What to show |
|--------|--------------|
| Interface stability | API/CLI/schema changes with before/after |
| Verification depth | Gate results, test coverage on changed code |
| Risk surface | Device/concurrency/IO/deps changes |
| Maintainability | Boundary clarity, complexity changes |
