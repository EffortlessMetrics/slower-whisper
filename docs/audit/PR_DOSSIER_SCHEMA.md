# PR Dossier Schema

For significant PRs requiring deep analysis, we generate structured dossiers. This schema defines the fields captured.

## File Location

`docs/audit/pr/<pr-number>.json`

## Schema (v2)

```json
{
  "schema_version": 2,
  "pr_number": 123,
  "title": "feat: add streaming latency benchmark",
  "url": "https://github.com/EffortlessMetrics/slower-whisper/pull/123",
  "created_at": "2026-01-07T12:00:00Z",
  "merged_at": "2026-01-08T14:30:00Z",

  "intent": {
    "goal": "Add P95 latency benchmark for streaming transcription",
    "issues": ["#97"],
    "phase": "v2.0-track1",
    "type": "perf/bench",
    "out_of_scope": ["full dataset support", "CI gating mode"]
  },

  "scope": {
    "files_changed": 12,
    "insertions": 450,
    "deletions": 120,
    "top_directories": [
      "transcription/benchmarks/",
      "tests/benchmarks/"
    ],
    "key_files": [
      "transcription/benchmarks/streaming_latency.py",
      "tests/benchmarks/test_streaming_latency.py"
    ],
    "blast_radius": "medium"
  },

  "findings": [
    {
      "type": "measurement drift",
      "description": "Initial baseline used wrong normalization",
      "detected_by": "manual review",
      "disposition": "fixed in same PR",
      "commit": "abc1234",
      "prevention_added": "#142 (baseline semantic validation)"
    }
  ],

  "evidence": {
    "local_gate": {
      "passed": true,
      "command": "./scripts/ci-local.sh fast",
      "receipt_path": "receipts/pr-123-gate.txt"
    },
    "tests": {
      "added": 5,
      "modified": 2,
      "path": "tests/benchmarks/test_streaming_latency.py",
      "coverage_delta": null
    },
    "typing": {
      "mypy_passed": true,
      "ruff_passed": true
    },
    "benchmarks": {
      "results_path": "benchmarks/results/streaming/smoke-run-123.json",
      "metrics": {
        "p50_ms": 42.3,
        "p95_ms": 78.1,
        "p99_ms": 112.5
      },
      "baseline_commit": "def5678",
      "semantics_unchanged": true
    },
    "docs_updated": true,
    "schema_validated": true
  },

  "process": {
    "friction_events": [
      {
        "event": "Baseline used different audio normalization",
        "detected_by": "manual review of benchmark config",
        "disposition": "fixed here",
        "prevention": "Added config hash validation to benchmark runner"
      }
    ],
    "design_alignment": {
      "drifted": false,
      "notes": null
    },
    "measurement_integrity": {
      "valid": true,
      "invalidation_reason": null,
      "correction_link": null
    }
  },

  "cost": {
    "wall_clock": {
      "days": 1.1,
      "created_at": "2026-01-07T12:00:00Z",
      "merged_at": "2026-01-08T14:30:00Z"
    },
    "active_work": {
      "hours": 4.5,
      "method": "commit bursts (gaps ≤45min = same session)"
    },
    "devlt": {
      "author_minutes": 45,
      "author_band": "45-90m",
      "review_minutes": 15,
      "review_band": "10-20m",
      "method": "manual tracking"
    },
    "machine_spend": {
      "estimate_usd": 3.20,
      "band": "$2-$10",
      "method": "claude session cost",
      "notes": "Includes 2 retry loops"
    }
  },

  "reflection": {
    "went_well": [
      "Benchmark design caught edge case early",
      "Config validation prevented measurement drift"
    ],
    "could_be_better": [
      "Should have pinned baseline commit from start",
      "Documentation could have been written earlier"
    ]
  },

  "outcome": "shipped",

  "factory_delta": {
    "gates_added": ["baseline semantic validation"],
    "contracts_added": ["streaming result schema"],
    "prevention_issues": ["#142"]
  },

  "followups": [
    {
      "issue": "#143",
      "description": "Add full dataset benchmark mode"
    }
  ]
}
```

## Field Definitions

### Core Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | int | yes | Dossier schema version (current: 2) |
| `pr_number` | int | yes | GitHub PR number |
| `title` | string | yes | PR title |
| `url` | string | yes | PR URL |
| `created_at` | string | yes | ISO 8601 timestamp |
| `merged_at` | string | no | ISO 8601 timestamp (null if not merged) |

### Intent

| Field | Type | Description |
|-------|------|-------------|
| `goal` | string | 1-2 sentence description of what this PR aims to do |
| `issues` | string[] | Linked issue numbers |
| `phase` | string | Roadmap phase (v1.9, v2.0-track1, etc.) |
| `type` | string | `feature` \| `hardening` \| `mechanization` \| `perf/bench` \| `refactor` |
| `out_of_scope` | string[] | Explicit exclusions |

### Scope

| Field | Type | Description |
|-------|------|-------------|
| `files_changed` | int | Total files modified |
| `insertions` | int | Lines added |
| `deletions` | int | Lines removed |
| `top_directories` | string[] | Most affected paths |
| `key_files` | string[] | Critical files to review |
| `blast_radius` | string | Impact estimate (`low` \| `medium` \| `high`) |

### Findings

Array of issues discovered during PR:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Failure mode type (see [FAILURE_MODES.md](FAILURE_MODES.md)) |
| `description` | string | What was found |
| `detected_by` | string | How it was caught (gate / review / post-merge) |
| `disposition` | string | `fixed here` \| `fixed in #X` \| `deferred to #X` \| `invalid` |
| `commit` | string | Fix commit hash |
| `prevention_added` | string | Issue/PR for prevention work |

### Evidence

#### Local Gate

| Field | Type | Description |
|-------|------|-------------|
| `passed` | bool | Did the gate pass? |
| `command` | string | Command run |
| `receipt_path` | string | Path to receipt file |

#### Tests

| Field | Type | Description |
|-------|------|-------------|
| `added` | int | Tests added |
| `modified` | int | Tests modified |
| `path` | string | Primary test file |
| `coverage_delta` | float \| null | Coverage change (if tracked) |

#### Typing

| Field | Type | Description |
|-------|------|-------------|
| `mypy_passed` | bool | mypy clean |
| `ruff_passed` | bool | ruff clean |

#### Benchmarks

| Field | Type | Description |
|-------|------|-------------|
| `results_path` | string | Path to results file |
| `metrics` | object | `{p50_ms, p95_ms, p99_ms}` or similar |
| `baseline_commit` | string | Commit used for baseline |
| `semantics_unchanged` | bool | True if comparison is valid |

#### Other

| Field | Type | Description |
|-------|------|-------------|
| `docs_updated` | bool | Were docs updated? |
| `schema_validated` | bool | Schema validation passed? |

### Process

#### Friction Events

Array of problems encountered:

| Field | Type | Description |
|-------|------|-------------|
| `event` | string | What broke |
| `detected_by` | string | How it was caught |
| `disposition` | string | How it was resolved |
| `prevention` | string | What was added to prevent recurrence |

#### Design Alignment

| Field | Type | Description |
|-------|------|-------------|
| `drifted` | bool | Did design drift from plan? |
| `notes` | string \| null | Details if drifted |

#### Measurement Integrity

| Field | Type | Description |
|-------|------|-------------|
| `valid` | bool | Are measurements trustworthy? |
| `invalidation_reason` | string \| null | Why invalid (if applicable) |
| `correction_link` | string \| null | Link to correction PR/issue |

### Cost

#### Wall Clock

| Field | Type | Description |
|-------|------|-------------|
| `days` | float | Total days from create to merge |
| `created_at` | string | PR creation timestamp |
| `merged_at` | string | PR merge timestamp |

#### Active Work

| Field | Type | Description |
|-------|------|-------------|
| `hours` | float | Estimated active work hours |
| `method` | string | How estimated (e.g., "commit bursts") |

#### DevLT (Developer Time)

| Field | Type | Description |
|-------|------|-------------|
| `author_minutes` | int | Author attention time |
| `author_band` | string | `10-20m` \| `20-45m` \| `45-90m` \| `>90m` |
| `review_minutes` | int | Reviewer attention time |
| `review_band` | string | Same bands as author |
| `method` | string | How estimated |

#### Machine Spend

| Field | Type | Description |
|-------|------|-------------|
| `estimate_usd` | float \| null | Estimated cost |
| `band` | string | `$0-$2` \| `$2-$10` \| `$10-$25` \| `unknown` |
| `method` | string | How estimated |
| `notes` | string | Context |

### Reflection

| Field | Type | Description |
|-------|------|-------------|
| `went_well` | string[] | 2-5 things that worked well |
| `could_be_better` | string[] | 2-5 actionable improvements |

### Outcome

One of: `shipped` | `deferred` | `rejected` | `invalid_measurement`

### Factory Delta

What was added to improve future development:

| Field | Type | Description |
|-------|------|-------------|
| `gates_added` | string[] | New validation gates |
| `contracts_added` | string[] | New schema/API contracts |
| `prevention_issues` | string[] | Issues created for prevention work |

### Follow-ups

| Field | Type | Description |
|-------|------|-------------|
| `issue` | string | Issue number |
| `description` | string | What needs to be done |

---

## Migration from v1

Schema v2 adds:
- `intent.goal` and `intent.type`
- `scope.key_files`
- `findings[].detected_by`
- `evidence` restructured with nested objects
- `process` section (friction_events, design_alignment, measurement_integrity)
- `cost` section (wall_clock, active_work, expanded devlt, machine_spend with band)
- `reflection` section (went_well, could_be_better)
- `followups` array

v1 dossiers remain valid but should be migrated when edited.

---

## Usage

Dossiers are generated manually for PRs worth analyzing in depth. Not every PR needs one.

Good candidates for dossiers:
- PRs that revealed new failure modes
- Large architectural changes
- Performance-sensitive changes
- PRs with significant iteration/rework

To generate a dossier:
1. Use `python scripts/generate-pr-ledger.py --pr <number>` for a template
2. Fill in fields from PR metadata and your notes
3. Save to `docs/audit/pr/<number>.json`

---

## See Also

- [PR_LEDGER_TEMPLATE.md](PR_LEDGER_TEMPLATE.md) — Markdown template for PR comments
- [EXHIBITS.md](EXHIBITS.md) — Annotated PR examples
- [FAILURE_MODES.md](FAILURE_MODES.md) — Failure taxonomy
