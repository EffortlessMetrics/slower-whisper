# Benchmark Baselines

This directory contains baseline files for regression testing benchmark results.

## Directory Structure

```
baselines/
├── asr/
│   └── librispeech.json
├── diarization/
│   └── ami.json
├── streaming/
│   └── librispeech.json
├── semantic/
│   └── ami.json
└── emotion/
    └── iemocap.json
```

## File Format

Each baseline file follows this schema:

```json
{
  "schema_version": 1,
  "track": "asr",
  "dataset": "librispeech",
  "created_at": "2026-01-21T00:00:00Z",
  "metrics": {
    "wer": {
      "value": 4.5,
      "unit": "%",
      "threshold": 0.10
    }
  },
  "receipt": {
    "tool_version": "1.9.2",
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16"
  }
}
```

### Fields

| Field | Description |
|-------|-------------|
| `schema_version` | Baseline file schema version (currently 1) |
| `track` | Benchmark track (asr, diarization, streaming, semantic, emotion) |
| `dataset` | Dataset name |
| `created_at` | ISO 8601 timestamp when baseline was created |
| `metrics` | Dict of metric name -> {value, unit, threshold} |
| `receipt` | Provenance info (tool version, model, device, etc.) |

### Metrics

- `value`: The baseline metric value
- `unit`: Unit of measurement (%, ms, x, etc.)
- `threshold`: Maximum allowed regression as a decimal (0.10 = 10%)

## Usage

### Create a New Baseline

```bash
slower-whisper benchmark save-baseline --track asr --dataset librispeech
```

With custom regression thresholds:

```bash
slower-whisper benchmark save-baseline --track asr --dataset librispeech \
  --threshold wer=0.05 --threshold cer=0.10
```

### Compare Against Baseline

Report mode (never fails):

```bash
slower-whisper benchmark compare --track asr --dataset librispeech
```

Gate mode (fails on regression):

```bash
slower-whisper benchmark compare --track asr --dataset librispeech --gate
```

### List All Baselines

```bash
slower-whisper benchmark baselines
```

## Regression Policy

**Phase 1 (current):** Report-only mode. Comparison prints results but never fails the CLI.

**Phase 2 (future):** Gate mode. Use `--gate` flag to fail if regression exceeds threshold.

### Regression Formula

```
regression = (current_value - baseline_value) / baseline_value
```

A metric **fails** if `regression > threshold`.

## Updating Baselines

When intentionally changing model behavior (new model version, config changes):

1. Run the benchmark to verify expected improvements/changes
2. Save a new baseline: `slower-whisper benchmark save-baseline --track <track>`
3. Commit the updated baseline file with a message explaining the change

## Notes

- Baseline files are committed to the repository
- Each track+dataset combination has one baseline file
- Metrics with `None` values are not included in baselines
- Missing metrics in baseline are skipped during comparison
