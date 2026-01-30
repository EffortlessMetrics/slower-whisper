---
description: Create a PR from the current branch with narrative summary, quality signals, and verification receipts
argument-hint: [optional: "Issue #N", "draft", "ready", "base=branch"]
---

# Create PR

Create a pull request from the current branch with a clear narrative for reviewers.

**User context:** $ARGUMENTS

## What to do

1. **Gather git state**: branch name, commits since base, changed files, push status
2. **Understand the changes**: Use Explore agents to map semantic vs mechanical changes, interface impacts, and risk areas
3. **Verify quality**: Run `./scripts/ci-local.sh` (or fast mode) if not already passing
4. **Compose the PR**: Write a title and body that tells reviewers what changed, why, and how to review it
5. **Create the PR**: Use `gh pr create` with the composed content

## PR body structure

```markdown
## Summary
What changed and why. Reference issues with "Closes #N" or "Relates to #N".

## What changed
Narrative at system level, grouped by:
- Core behavior changes
- API/CLI surface changes
- Test/docs updates

## Interface impact
- **Public API**: unchanged | additive | breaking
- **CLI surface**: unchanged | new flags | changed behavior

## How to review
Suggested review path highlighting semantic hotspots.

## Test plan
- [ ] Gate verification
- [ ] Specific tests or manual steps
```

## Guidelines

- Title: imperative, under 70 chars (e.g., "feat: add streaming callback")
- Default to `--draft` unless user specified "ready"
- Use HEREDOC for the body to preserve formatting
- Report the PR URL when complete

## Repo context

Reference CLAUDE.md for invariants that should hold:
- Device resolution explicit
- compute_type follows resolved device
- Callbacks never crash pipeline
- Streaming end_of_stream finalizes turns
- Version matches package metadata
