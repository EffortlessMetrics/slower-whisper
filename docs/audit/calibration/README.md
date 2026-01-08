# DevLT Estimation Calibration

This directory contains infrastructure for tracking and validating the accuracy of DevLT (Developer Lead Time) estimations against human-reported or known values.

## Purpose

The estimation model in `transcription/historian/estimation.py` produces bounded estimates (lower bound, upper bound) for developer time. The control-plane model (decision-weighted) uses `transcription/historian/analyzers/decision_extractor.py` to identify material decisions and sum their time bounds.

Calibration allows us to:

1. **Validate accuracy**: Check if human-reported times fall within our estimated bounds
2. **Track drift**: Monitor estimation performance over time
3. **Improve the model**: Identify systematic biases and adjust decision weights
4. **Tune decision extractor**: Adjust time bands per decision type based on real data

## How It Works

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PR merged     │───>│ Human records    │───>│ Add to samples  │
│                 │    │ actual DevLT     │    │ .json           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
                                                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ check-calibration│<──│ Compare actual   │<───│ Run historian   │
│ .py outputs     │    │ vs estimated     │    │ pipeline        │
│ tuning advice   │    │ bounds           │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

1. After a PR is analyzed, we record the estimated bounds (from decision extractor or session proxy)
2. Authors/reviewers provide their actual time (when available)
3. The calibration script compares estimates against actuals
4. Metrics and actionable tuning advice are output

## Coverage Levels

Estimates have different coverage levels that affect accuracy:

| Coverage | Description | Expected Accuracy |
|----------|-------------|-------------------|
| `github_only` | PR metadata, commits, comments, reviews, checks | Lower - missing IDE/session context |
| `github_plus_agent_logs` | GitHub + Claude Code session logs | Higher - full decision context |

Always record the coverage level when adding samples to enable stratified analysis.

## Calibration Data Format

The `samples.json` file contains an array of calibration samples. See `calibration.schema.json` for the formal schema.

### Example Entry

```json
{
  "pr_number": 129,
  "title": "ci(nix): avoid libstdc++/nix ABI collisions; add local CI gate",
  "human_reported": {
    "lb_minutes": 35,
    "ub_minutes": 50,
    "point_estimate_minutes": 42,
    "confidence": "med",
    "source": "author"
  },
  "estimated": {
    "lb_minutes": 30,
    "ub_minutes": 75,
    "method": "decision-weighted-v1",
    "coverage": "github_only",
    "decision_count": 6
  },
  "decisions_summary": [
    {"type": "debug", "description": "Resolve ABI collision between pip wheels and nix"},
    {"type": "design", "description": "Create nix-clean wrapper pattern"},
    {"type": "quality", "description": "Add ci-local.sh as local gate"}
  ],
  "recorded_at": "2026-01-07",
  "notes": "Multi-session debugging across 5 commits, some oscillation"
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pr_number` | int | yes | GitHub PR number |
| `title` | string | no | PR title for reference |
| `human_reported.lb_minutes` | int | yes* | Lower bound of human estimate |
| `human_reported.ub_minutes` | int | yes* | Upper bound of human estimate |
| `human_reported.point_estimate_minutes` | int | no | Single best estimate if given |
| `human_reported.confidence` | string | no | `high`, `med`, `low` - how confident in the estimate |
| `human_reported.source` | string | no | `author`, `reviewer`, `both`, `reconstructed` |
| `estimated.lb_minutes` | int | yes | Lower bound from estimation model |
| `estimated.ub_minutes` | int | yes | Upper bound from estimation model |
| `estimated.method` | string | no | `decision-weighted-v1`, `session-proxy`, etc. |
| `estimated.coverage` | string | no | `github_only`, `github_plus_agent_logs` |
| `estimated.decision_count` | int | no | Number of decisions extracted |
| `decisions_summary` | array | no | Key decisions for reference |
| `recorded_at` | string | no | ISO date when sample was recorded |
| `notes` | string | no | Additional context, friction points, etc. |

*At least one of `lb_minutes/ub_minutes` or `point_estimate_minutes` is required.

## Running Calibration Checks

```bash
# Run with default samples.json
python scripts/check-calibration.py

# Specify a different samples file
python scripts/check-calibration.py --samples-path path/to/samples.json

# Verbose output (show each sample)
python scripts/check-calibration.py --verbose

# JSON output for automation
python scripts/check-calibration.py --json

# Show tuning recommendations
python scripts/check-calibration.py --tuning-advice
```

## Metrics Computed

The calibration script reports:

- **Coverage rate**: % of human-reported values that fall within [estimated_lb, estimated_ub]
- **Mean absolute error (MAE)**: Average |estimated_midpoint - human_midpoint|
- **Bias**: Average (estimated_midpoint - human_midpoint), positive = overestimate
- **Bound width ratio**: Ratio of estimated width to human width (>1 = too wide, <1 = too narrow)
- **Per-decision-type accuracy**: Error breakdown by decision type (scope, design, quality, debug, publish)

## Adding Calibration Data

When completing a PR:

1. **Track time as you work** (start/stop or estimate at PR close)
2. **Run the estimation**: `python scripts/generate-pr-ledger.py --pr <N> --dump-bundle`
3. **Add to samples.json** with both human and estimated bounds
4. **Run calibration**: `python scripts/check-calibration.py --verbose`

### Tips for Human Estimates

- Give a range, not just a point estimate
- Note confidence level:
  - `high`: Tracked with timer or clear session boundaries
  - `med`: Good estimate based on calendar/memory
  - `low`: Reconstructed from vague memory
- Include interruptions/context-switching only if it was PR-related
- Exclude pre-PR research that would have happened anyway

## Tuning the Decision Extractor

The calibration script outputs actionable tuning advice based on error patterns:

### Decision Time Bands (current defaults)

```python
DECISION_TIME_BANDS = {
    "scope": (2, 8),    # Deferring, cutting, splitting
    "design": (6, 20),  # Schema, API, contract choices
    "quality": (4, 15), # Gate decisions, accepting limitations
    "debug": (8, 30),   # Fixing failures, root cause analysis
    "publish": (3, 12), # Release decisions, documentation
}
```

### Tuning Process

1. Run `python scripts/check-calibration.py --tuning-advice`
2. Review per-type error breakdown
3. If a type consistently over/underestimates:
   - Overestimates: Reduce the (min, max) bounds
   - Underestimates: Increase the (min, max) bounds
4. Update `transcription/historian/analyzers/decision_extractor.py`
5. Re-run calibration to verify improvement

### Expected Calibration Targets

| Metric | Good | Acceptable | Needs Work |
|--------|------|------------|------------|
| Coverage rate | >= 85% | 70-85% | < 70% |
| MAE | <= 10 min | 10-20 min | > 20 min |
| Bias | +/- 5 min | +/- 15 min | > +/- 15 min |

## See Also

- [PR_DOSSIER_SCHEMA.md](../PR_DOSSIER_SCHEMA.md) - Dossier format including DevLT fields
- `transcription/historian/estimation.py` - Estimation model implementation
- `transcription/historian/analyzers/decision_extractor.py` - Decision extraction
- `scripts/generate-pr-ledger.py` - PR analysis script
- `scripts/check-calibration.py` - Calibration analysis
