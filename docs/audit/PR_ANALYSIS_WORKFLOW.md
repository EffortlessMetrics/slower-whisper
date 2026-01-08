# PR Analysis Workflow

This document describes how to use Claude Code to analyze PRs and generate ledgers/dossiers.

## Quick Start

```bash
# 1. Gather PR data and generate analysis prompt
python scripts/generate-pr-ledger.py --pr 123

# 2. Feed the output to Claude Code for analysis
# (copy/paste or pipe)

# 3. Claude produces the complete dossier
```

Or directly in Claude Code:

```
Analyze PR #123 and generate a complete ledger dossier
```

---

## Workflow Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  gather-pr-data │ ──▶ │  LLM Analysis    │ ──▶ │  Dossier JSON   │
│  (script)       │     │  (Claude)        │     │  + Ledger MD    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
   - PR metadata           - Intent/goal            - docs/audit/pr/N.json
   - Commits               - Findings               - PR description update
   - Comments/reviews      - Friction events
   - File changes          - DevLT estimate
   - CI checks             - Reflection
   - Computed metrics      - Factory delta
```

---

## Step 1: Gather PR Data

The script `scripts/generate-pr-ledger.py` fetches all relevant data from GitHub:

```bash
# Default: output analysis prompt for LLM
python scripts/generate-pr-ledger.py --pr 123

# Raw JSON data (for inspection)
python scripts/generate-pr-ledger.py --pr 123 --raw

# Include full diff (large PRs)
python scripts/generate-pr-ledger.py --pr 123 --include-diff

# Save to file
python scripts/generate-pr-ledger.py --pr 123 --output analysis-prompt.md
```

### Data Gathered

| Category | Data |
|----------|------|
| Metadata | title, URL, state, author, dates, labels |
| Scope | files changed, insertions/deletions, directories |
| Commits | messages, timestamps, SHAs |
| Comments | PR comments, review comments, review states |
| CI | check runs, pass/fail status |
| Analysis | issue refs, active work estimate, wall-clock time |

### Friction Signal Detection

The script automatically detects friction signals:

- **Fix commits:** Messages containing "fix", "bug", "correct", "resolve"
- **Revert commits:** Messages containing "revert", "undo", "rollback"
- **WIP commits:** Messages containing "WIP", "work in progress"
- **Test commits:** Messages containing "test", "spec", "coverage"
- **Doc commits:** Messages containing "doc", "readme", "comment"

---

## Step 2: LLM Analysis

When Claude Code receives the analysis prompt, it should:

### 2.1 Analyze Intent

From PR title, description, and commit messages:

- **Goal:** What does this PR accomplish? (1-2 sentences)
- **Type:** feature | hardening | mechanization | perf/bench | refactor
- **Phase:** Infer from labels, branch names, or description

### 2.2 Identify Findings

Look for issues discovered during the PR:

- Fix commits → something was wrong initially
- Review comments requesting changes → caught in review
- Multiple iterations on same file → friction
- Reverts → significant problem

Categorize by failure mode (see [FAILURE_MODES.md](FAILURE_MODES.md)):
- measurement drift
- doc drift
- packaging drift
- test flake
- dependency hazard
- process mismatch

### 2.3 Extract Friction Events

For each "what went wrong" event:

| Field | Source |
|-------|--------|
| Event | Commit message, review comment |
| Detected by | gate (CI), review, post-merge |
| Disposition | fixed here, deferred to #X |
| Prevention | New gate, test, doc, or "none needed" |

### 2.4 Assess Evidence

Based on file categories:

- **Tests added/modified:** Check tests/ files in changes
- **Docs updated:** Check docs/ and *.md files
- **Benchmarks:** Check benchmarks/ or perf-related files
- **Schema changes:** Check schema files

### 2.5 Estimate DevLT

Use the computed metrics as baseline:

| Metric | Heuristic |
|--------|-----------|
| Active work | From commit bursts (provided) |
| Author attention | commits × complexity factor |
| Review attention | comments × review depth factor |

Map to bands: `10-20m` | `20-45m` | `45-90m` | `>90m`

### 2.6 Generate Reflection

**What went well:**
- Clean commit history
- Good test coverage
- Quick review turnaround
- No CI failures

**What could be better:**
- Earlier design discussion
- More atomic commits
- Better PR description
- Missing tests for edge cases

### 2.7 Identify Factory Delta

What was added to improve future development:
- New validation gates
- New schema contracts
- New tests covering edge cases
- Prevention issues to file

---

## Step 3: Output

### Dossier JSON

Save to `docs/audit/pr/<number>.json`:

```json
{
  "schema_version": 2,
  "pr_number": 123,
  "title": "...",
  "intent": { "goal": "...", "type": "feature", ... },
  "scope": { ... },
  "findings": [ ... ],
  "evidence": { ... },
  "process": { "friction_events": [...], ... },
  "cost": { ... },
  "reflection": { "went_well": [...], "could_be_better": [...] },
  "outcome": "shipped",
  "factory_delta": { ... },
  "followups": [ ... ]
}
```

### Ledger Markdown

Update PR description with ledger (see [PR_LEDGER_TEMPLATE.md](PR_LEDGER_TEMPLATE.md)).

---

## Subagent Decomposition (Optional)

For complex PRs, decompose analysis into specialized passes:

### Diff Scout
```
Analyze the PR diff and produce:
- Change surface map (which modules/directories affected)
- Hotspots (files with most changes)
- Semantic summary (what's actually different)
```

### Evidence Auditor
```
Check the PR for evidence:
- Are receipts present? (gate output, test results)
- Do claims in description match artifacts?
- Any contradictions between claimed and actual changes?
```

### Docs/Schema Auditor
```
Verify documentation integrity:
- Do doc changes match code changes?
- Are API examples still valid?
- Any stale references or wrong keys?
```

### Perf Integrity Auditor
```
If performance-related PR:
- Are metrics absolute (not just percentages)?
- Is baseline commit pinned?
- Are semantics unchanged between measurements?
- Any drift indicators?
```

### Process/Friction Miner
```
Extract friction events:
- Find all fix/correction commits
- Find review comments requesting changes
- Identify iteration patterns
- Document what went wrong and how it was resolved
```

### Design Alignment Auditor
```
Check design alignment:
- Does implementation match issue requirements?
- Any scope creep or drift from plan?
- Are constraints from ADR/design docs respected?
```

Each subagent produces 5-15 lines. Main analysis synthesizes these.

---

## Example Usage

### Manual Flow

```bash
# 1. Generate analysis prompt
python scripts/generate-pr-ledger.py --pr 42 > /tmp/pr42-analysis.md

# 2. In Claude Code, paste the content and ask:
#    "Analyze this PR data and generate a complete dossier"

# 3. Save the dossier output
#    Copy JSON to docs/audit/pr/42.json
#    Update PR description with ledger markdown
```

### Claude Code Direct

```
User: Analyze PR #42 and generate a ledger

Claude: [Runs the script, analyzes output, produces dossier]
```

---

## Backfill Workflow

For backfilling exhibits from existing PRs:

1. **Select exhibit candidates** (see scoring rubric in [EXHIBITS.md](EXHIBITS.md))
2. **Generate analysis for each:**
   ```bash
   for pr in 10 15 23 42; do
     python scripts/generate-pr-ledger.py --pr $pr --output /tmp/pr-$pr.md
   done
   ```
3. **Analyze each with Claude Code**
4. **Save dossiers** to `docs/audit/pr/`
5. **Update EXHIBITS.md** with entries
6. **Update PR descriptions** with ledgers (optional for closed PRs)

---

## See Also

- [PR_DOSSIER_SCHEMA.md](PR_DOSSIER_SCHEMA.md) — JSON schema (v2)
- [PR_LEDGER_TEMPLATE.md](PR_LEDGER_TEMPLATE.md) — Markdown template
- [EXHIBITS.md](EXHIBITS.md) — Exhibit catalog and selection criteria
- [FAILURE_MODES.md](FAILURE_MODES.md) — Failure taxonomy
