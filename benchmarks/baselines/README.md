# Benchmark Baselines

This directory contains baseline files for regression testing benchmark results.

## Directory Structure

```
baselines/
├── asr/
│   ├── librispeech.json    # Full LibriSpeech dataset baseline
│   └── smoke.json          # CI smoke test baseline (always available)
├── diarization/
│   ├── ami.json            # Full AMI Meeting Corpus baseline
│   └── smoke.json          # CI smoke test baseline (always available)
├── streaming/
│   └── librispeech.json
├── semantic/
│   └── ami.json
└── emotion/
    └── iemocap.json
```

## Smoke vs Full Baselines

| Baseline Type | Purpose | Threshold | CI Usage |
|--------------|---------|-----------|----------|
| **smoke** | Quick validation, always available | High (50%) | Default for PRs |
| **full** | Production quality gates | Low (5-15%) | Manual/scheduled |

Smoke baselines use high thresholds to avoid flaky CI failures while still catching catastrophic regressions.

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

## Regression Policy Details

### What's a Regression?

A regression is detected when:

```
regression_ratio = (current - baseline) / baseline
```

For most metrics (WER, CER, DER, latency), **higher is worse**, so:
- `regression_ratio > 0` means performance degraded
- `regression_ratio > threshold` triggers a failure in gate mode

### What's Noise?

Expected variance that should not trigger failures:

| Source | Typical Variance | Mitigation |
|--------|------------------|------------|
| Hardware | 1-5% | Run on consistent CI hardware |
| Model loading | <1% | Use warm cache |
| Floating point | <0.1% | Usually negligible |
| Batch size | 1-3% | Fix batch size in config |
| Random seeds | 0-2% | Fix seeds when possible |

### Threshold Guidelines

| Track | Metric | Smoke Threshold | Production Threshold |
|-------|--------|-----------------|---------------------|
| ASR | WER | 50% | 10% |
| ASR | CER | 50% | 15% |
| Diarization | DER | 50% | 15% |
| Diarization | JER | 50% | 20% |
| Streaming | latency_p50 | 100% | 25% |
| Streaming | RTF | 100% | 20% |
| Semantic | topic_f1 | 50% | 15% |

**Rationale:**
- Smoke thresholds are high to avoid CI flakiness on limited samples
- Production thresholds are tighter to catch real regressions
- Different metrics have different sensitivity to changes

### When to Update Baselines

1. **Model upgrade**: New Whisper version improves accuracy
2. **Config change**: Changed default beam size, etc.
3. **Bug fix**: Fixed bug that was artificially improving metrics
4. **Dataset change**: Updated reference annotations
5. **Calibration**: After significant code refactoring

Always document the reason for baseline updates in commit messages.

## Manifest Integration

Dataset manifests can include `expected_baseline` fields that define the
expected performance for that dataset:

```json
{
  "expected_baseline": {
    "wer": {
      "value": 5.0,
      "tolerance": 5.0,
      "unit": "%",
      "notes": "TTS audio is clean; WER should be low."
    }
  }
}
```

This provides self-documenting datasets where expected performance is
defined alongside the data itself.

## CI Integration

The benchmark CI workflow (`.github/workflows/benchmark.yml`) uses baselines as follows:

1. **PR runs**: Report-only mode, smoke datasets, no failures
2. **Manual dispatch**: Optional gate mode with full datasets
3. **Scheduled runs**: Full evaluation against production baselines

See [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md) for CI configuration details.
