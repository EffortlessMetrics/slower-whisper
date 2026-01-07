# DevLT Estimation Calibration

This directory contains infrastructure for tracking and validating the accuracy of DevLT (Developer Lead Time) estimations against human-reported or known values.

## Purpose

The estimation model in `transcription/historian/estimation.py` produces bounded estimates (lower bound, upper bound) for developer time. Calibration allows us to:

1. **Validate accuracy**: Check if human-reported times fall within our estimated bounds
2. **Track drift**: Monitor estimation performance over time
3. **Improve the model**: Identify systematic biases and adjust constants

## How It Works

1. After a PR is analyzed, we record the estimated bounds
2. Authors/reviewers provide their actual time (when available)
3. The calibration script compares estimates against actuals
4. Metrics are computed to assess model performance

## Calibration Data Format

The `samples.json` file contains an array of calibration samples:

```json
[
  {
    "pr_number": 123,
    "human_reported_minutes": 45,
    "estimated_lb": 30,
    "estimated_ub": 60,
    "source": "author",
    "recorded_at": "2026-01-07",
    "notes": "Author self-report via PR comment"
  }
]
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pr_number` | int | yes | GitHub PR number |
| `human_reported_minutes` | int \| null | no | Actual time reported (null if unknown) |
| `estimated_lb` | int | yes | Lower bound estimate (minutes) |
| `estimated_ub` | int | yes | Upper bound estimate (minutes) |
| `source` | string | no | Who reported: `author`, `reviewer`, `both`, or `unknown` |
| `recorded_at` | string | no | ISO date when sample was recorded |
| `notes` | string | no | Additional context |

## Running Calibration Checks

```bash
# Run with default samples.json
python scripts/check-calibration.py

# Specify a different samples file
python scripts/check-calibration.py --samples-path path/to/samples.json

# Verbose output (show each sample)
python scripts/check-calibration.py --verbose
```

## Metrics Computed

The calibration script reports:

- **Coverage rate**: % of human-reported values that fall within [lb, ub]
- **Mean absolute error (MAE)**: Average |estimated_midpoint - actual|
- **Bias**: Average (estimated_midpoint - actual), positive = overestimate
- **Bound width**: Average (ub - lb), measures estimate precision

## Adding Calibration Data

When completing a PR:

1. Note your actual time spent (author or reviewer)
2. Run the estimation on the PR: `python scripts/generate-pr-ledger.py --pr <N> --dump-bundle`
3. Add a sample to `samples.json` with the estimated bounds and your actual time
4. Run `python scripts/check-calibration.py` to update metrics

## Improving the Model

If calibration reveals systematic issues:

1. **Low coverage rate** (< 80%): Bounds may be too tight
   - Review constants in `estimation.py`
   - Consider widening `session_slack_minutes` or adjusting floor values

2. **High MAE**: Midpoint estimates are inaccurate
   - Analyze which PR types have highest errors
   - Consider type-specific adjustments

3. **Consistent bias**: Model systematically over/underestimates
   - Positive bias: reduce floor values or session slack
   - Negative bias: increase floor values or add minimum bounds

## See Also

- [PR_DOSSIER_SCHEMA.md](../PR_DOSSIER_SCHEMA.md) - Dossier format including DevLT fields
- `transcription/historian/estimation.py` - Estimation model implementation
- `scripts/generate-pr-ledger.py` - PR analysis script
