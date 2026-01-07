# PR Dossier Schema

For significant PRs requiring deep analysis, we generate structured dossiers. This schema defines the fields captured.

## File Location

`docs/audit/pr/<pr-number>.json`

## Schema

```json
{
  "schema_version": 1,
  "pr_number": 123,
  "title": "feat: add streaming latency benchmark",
  "url": "https://github.com/EffortlessMetrics/slower-whisper/pull/123",
  "created_at": "2026-01-07T12:00:00Z",
  "merged_at": "2026-01-08T14:30:00Z",

  "intent": {
    "issues": ["#97"],
    "phase": "v2.0-track1",
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
    "blast_radius": "medium"
  },

  "findings": [
    {
      "type": "measurement drift",
      "description": "Initial baseline used wrong normalization",
      "disposition": "fixed in same PR",
      "commit": "abc1234",
      "prevention_added": "#142 (baseline semantic validation)"
    }
  ],

  "evidence": {
    "tests_added": true,
    "tests_path": "tests/benchmarks/test_streaming_latency.py",
    "local_gate_receipt": "receipts/pr-123-gate.txt",
    "benchmark_results": "benchmarks/results/streaming/smoke-run-123.json",
    "docs_updated": true
  },

  "devlt": {
    "author_minutes": 45,
    "review_minutes": 15,
    "notes": "Multiple iterations on baseline format"
  },

  "machine_cost": {
    "estimate_usd": 3.20,
    "method": "claude session cost",
    "notes": "Includes 2 retry loops"
  },

  "outcome": "shipped",
  "factory_delta": {
    "gates_added": ["baseline semantic validation"],
    "contracts_added": ["streaming result schema"],
    "prevention_issues": ["#142"]
  }
}
```

## Field Definitions

### Core Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | int | yes | Dossier schema version |
| `pr_number` | int | yes | GitHub PR number |
| `title` | string | yes | PR title |
| `url` | string | yes | PR URL |
| `created_at` | string | yes | ISO 8601 timestamp |
| `merged_at` | string | no | ISO 8601 timestamp (null if not merged) |

### Intent

| Field | Type | Description |
|-------|------|-------------|
| `issues` | string[] | Linked issue numbers |
| `phase` | string | Roadmap phase (v1.9, v2.0-track1, etc.) |
| `out_of_scope` | string[] | Explicit exclusions |

### Scope

| Field | Type | Description |
|-------|------|-------------|
| `files_changed` | int | Total files modified |
| `insertions` | int | Lines added |
| `deletions` | int | Lines removed |
| `top_directories` | string[] | Most affected paths |
| `blast_radius` | string | Impact estimate (low/medium/high) |

### Findings

Array of issues discovered during PR:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Failure mode type (see FAILURE_MODES.md) |
| `description` | string | What was found |
| `disposition` | string | fixed here / fixed in #X / still open / invalid |
| `commit` | string | Fix commit hash |
| `prevention_added` | string | Issue/PR for prevention work |

### Evidence

| Field | Type | Description |
|-------|------|-------------|
| `tests_added` | bool | Were tests added/updated? |
| `tests_path` | string | Path to test file |
| `local_gate_receipt` | string | Path to gate receipt |
| `benchmark_results` | string | Path to benchmark output |
| `docs_updated` | bool | Were docs updated? |

### DevLT

| Field | Type | Description |
|-------|------|-------------|
| `author_minutes` | int | Author attention time |
| `review_minutes` | int | Reviewer attention time |
| `notes` | string | Context for time spent |

### Machine Cost

| Field | Type | Description |
|-------|------|-------------|
| `estimate_usd` | float | Estimated cost |
| `method` | string | How estimated (claude session cost, api billing, etc.) |
| `notes` | string | Context |

### Outcome

`shipped` | `deferred` | `rejected` | `invalid_measurement`

### Factory Delta

What was added to improve future development:

| Field | Type | Description |
|-------|------|-------------|
| `gates_added` | string[] | New validation gates |
| `contracts_added` | string[] | New schema/API contracts |
| `prevention_issues` | string[] | Issues created for prevention work |

## Usage

Dossiers are generated manually for PRs worth analyzing in depth. Not every PR needs one.

Good candidates for dossiers:
- PRs that revealed new failure modes
- Large architectural changes
- Performance-sensitive changes
- PRs with significant iteration/rework

To generate a dossier:
1. Copy the template from this doc
2. Fill in fields from PR metadata and your notes
3. Save to `docs/audit/pr/<number>.json`
