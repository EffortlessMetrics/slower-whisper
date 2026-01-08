# PR Fact Bundles

This directory contains deterministic fact bundles generated for PRs.

## What is a Fact Bundle?

A fact bundle is a JSON snapshot of all PR data gathered from GitHub, plus computed metrics. It contains:

- **Metadata**: PR title, URL, author, timestamps, labels
- **Scope**: Files changed, insertions/deletions, key files, blast radius
- **Commits**: All commits with messages, timestamps, and detected patterns
- **Sessions**: Work sessions derived from commit bursts, with LB/UB time estimates
- **Comments**: Issue comments and review comments
- **Reviews**: PR reviews with state
- **Check Runs**: CI check runs with timing (when available)
- **Receipt Paths**: Paths to receipts found in PR body/comments
- **Estimation**: Bounded time estimates (wall clock, active work, DevLT split)
- **Machine Time**: Check run timing for machine cost estimation

## Why Fact Bundles?

1. **Deterministic**: Generated without LLM, reproducible
2. **Sealed Input**: The LLM analyzers receive this as their only input
3. **Audit Trail**: Can regenerate dossiers from bundles
4. **Cost Tracking**: Precomputed bounded estimates

## Generating a Bundle

```bash
# Generate bundle for PR #123
python scripts/generate-pr-ledger.py --pr 123 --dump-bundle

# Output: docs/audit/pr/_bundles/123.json
```

## Bundle Schema

See the `FactBundle` dataclass in `transcription/historian/bundle.py` for the full schema.

Key fields:
- `bundle_version`: Schema version (current: 1)
- `generated_at`: When the bundle was created
- `pr_number`: GitHub PR number
- `metadata`: Core PR metadata
- `scope`: Change scope information
- `commits`: List of commits with patterns
- `sessions`: Work sessions with bounded estimates
- `estimation`: Overall bounded estimation
- `machine_time`: Check run timing data

## See Also

- [PR_DOSSIER_SCHEMA.md](../PR_DOSSIER_SCHEMA.md) - Schema for analyzed dossiers
- [PR_ANALYSIS_WORKFLOW.md](../PR_ANALYSIS_WORKFLOW.md) - Full analysis workflow
