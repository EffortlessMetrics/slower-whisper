# Benchmark CLI Reference

This document describes the benchmark CLI for evaluating slower-whisper quality on standard datasets.

## Overview

The benchmark CLI provides a standardized way to evaluate slower-whisper across multiple dimensions:

- **ASR (Automatic Speech Recognition)**: Measure transcription accuracy using WER/CER metrics
- **Diarization**: Evaluate speaker segmentation using DER metrics
- **Streaming**: Test latency and throughput for real-time transcription
- **Semantic**: Assess quality of LLM-based annotations (summaries, actions, etc.)
- **Emotion**: Validate emotion recognition accuracy

## Quick Start

```bash
# List available tracks and datasets
slower-whisper benchmark list

# Check which datasets are staged
slower-whisper benchmark status

# Run ASR evaluation on LibriSpeech (quick test with 10 samples)
slower-whisper benchmark run --track asr --dataset librispeech --limit 10

# Run full diarization evaluation on AMI test set
slower-whisper benchmark run --track diarization --dataset ami --split test --output results.json
```

## Available Benchmark Tracks

### ASR Track

Evaluates transcription accuracy using Word Error Rate (WER) and Character Error Rate (CER).

| Metric | Description |
|--------|-------------|
| `wer`  | Word Error Rate (lower is better) |
| `cer`  | Character Error Rate (lower is better) |
| `rtf`  | Real-time factor (processing speed) |

**Supported datasets:** LibriSpeech, CommonVoice (smoke test)

```bash
# Quick WER check on dev-clean
slower-whisper benchmark run --track asr --dataset librispeech --split dev-clean --limit 50

# Full test-clean evaluation
slower-whisper benchmark run --track asr --dataset librispeech --split test-clean

# Quick smoke test with CommonVoice subset
slower-whisper benchmark run --track asr --dataset commonvoice_en_smoke --limit 10
```

### Diarization Track

Evaluates speaker diarization quality using Diarization Error Rate (DER).

| Metric | Description |
|--------|-------------|
| `der`  | Diarization Error Rate (lower is better) |
| `jer`  | Jaccard Error Rate (lower is better) |
| `speaker_count_accuracy` | How often speaker count is correct |

**Supported datasets:** AMI, LibriCSS

```bash
# Quick DER check on AMI test meetings
slower-whisper benchmark run --track diarization --dataset ami --split test --limit 5

# Full AMI test set evaluation
slower-whisper benchmark run --track diarization --dataset ami --split test -o ami_der.json
```

### Streaming Track

Evaluates latency and throughput for streaming transcription scenarios.

| Metric | Description |
|--------|-------------|
| `latency_p50` | Median time to first token (ms) |
| `latency_p99` | 99th percentile latency (ms) |
| `throughput` | Tokens per second |
| `rtf` | Real-time factor |

**Supported datasets:** LibriSpeech, AMI

```bash
# Measure streaming performance
slower-whisper benchmark run --track streaming --dataset librispeech --limit 20 -v
```

### Semantic Track

Evaluates LLM-based semantic annotation quality. Supports two evaluation modes:

**Mode: `tags` (Deterministic, CI-friendly)**

Compares extracted annotations against gold labels using F1/precision/recall metrics.

| Metric | Description |
|--------|-------------|
| `topic_f1` | Topic extraction F1 (micro-averaged) |
| `risk_f1` | Risk detection F1 (type matching) |
| `risk_f1_weighted` | Severity-weighted risk F1 |
| `action_accuracy` | Action extraction accuracy |

```bash
# Deterministic evaluation against gold labels
slower-whisper benchmark run --track semantic --mode tags --dataset ami --split test

# Quick smoke test
slower-whisper benchmark run --track semantic --mode tags --dataset ami --limit 10
```

**Mode: `summary` (LLM-as-Judge)**

Uses Claude to evaluate generated summaries against reference summaries.

| Metric | Description |
|--------|-------------|
| `faithfulness` | Factual accuracy of generated content (0-10) |
| `coverage` | Completeness of key information (0-10) |
| `clarity` | Readability and coherence (0-10) |

```bash
# Evaluate summary quality (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-ant-xxx slower-whisper benchmark run \
  --track semantic --mode summary --dataset ami --split test --limit 5
```

**Supported datasets:** AMI (requires reference summaries or gold labels)

**Note:** `--mode summary` requires `ANTHROPIC_API_KEY` environment variable. `--mode tags` requires gold label files.

For detailed documentation including gold label format, metric definitions, and labeling workflow, see [SEMANTIC_BENCHMARK.md](SEMANTIC_BENCHMARK.md).

### Emotion Track

Evaluates emotion recognition accuracy against ground truth labels.

| Metric | Description |
|--------|-------------|
| `accuracy` | Classification accuracy (%) |
| `f1_weighted` | Weighted F1 score |
| `confusion_matrix` | Per-class confusion matrix |

**Supported datasets:** IEMOCAP

```bash
# Evaluate emotion recognition
slower-whisper benchmark run --track emotion --dataset iemocap --limit 100
```

## Supported Datasets

### LibriSpeech

A large corpus of read English speech for ASR evaluation.

| Split | Description | Samples | Duration |
|-------|-------------|---------|----------|
| `dev-clean` | Clean development set | ~2,703 | 5.4 hours |
| `test-clean` | Clean test set | ~2,620 | 5.4 hours |
| `dev-other` | Challenging development set | ~2,864 | 5.3 hours |
| `test-other` | Challenging test set | ~2,939 | 5.1 hours |

**Setup:**
```bash
# Download (example for dev-clean)
wget https://www.openslr.org/resources/12/dev-clean.tar.gz

# Extract to benchmarks directory
BENCH_ROOT=$(slower-whisper benchmark status | grep "Benchmarks root" | cut -d: -f2 | xargs)
tar -xzf dev-clean.tar.gz -C "$BENCH_ROOT/librispeech/"
```

See the `iter_librispeech()` docstring in `transcription/benchmarks.py` for complete setup instructions.

### AMI Meeting Corpus

Multi-party meeting recordings for diarization and summarization evaluation.

| Split | Description | Meetings |
|-------|-------------|----------|
| `train` | Training meetings | ~136 |
| `dev` | Development meetings | ~18 |
| `test` | Test meetings | ~16 |

**Setup:**
See `docs/AMI_SETUP.md` for complete instructions.

```bash
# Quick verification
BENCH_ROOT=~/.cache/slower-whisper/benchmarks
mkdir -p "$BENCH_ROOT/ami/audio"
mkdir -p "$BENCH_ROOT/ami/annotations"
mkdir -p "$BENCH_ROOT/ami/splits"

# Download meeting audio and create annotations
# See docs/AMI_SETUP.md for details
```

### IEMOCAP

Interactive emotional dyadic motion capture database for emotion recognition.

| Session | Description |
|---------|-------------|
| Session1 | First session pair |
| Session2 | Second session pair |
| Session3 | Third session pair |
| Session4 | Fourth session pair |
| Session5 | Fifth session pair |

**Setup:**
See `docs/IEMOCAP_SETUP.md` for complete instructions.

### LibriCSS

LibriSpeech Continuous Speech Separation for overlapping speech evaluation.

**Setup:**
See `docs/LIBRICSS_SETUP.md` (when available).

## CLI Reference

### benchmark list

List all available benchmark tracks and their supported datasets.

```bash
slower-whisper benchmark list
```

**Output:**
```
Available Benchmark Tracks:
------------------------------------------------------------

  asr: ASR (Automatic Speech Recognition)
     Evaluate transcription accuracy using WER/CER metrics
     Datasets: librispeech
     Metrics: wer, cer, rtf

  diarization: Speaker Diarization
     Evaluate speaker segmentation using DER metrics
     Datasets: ami, libricss
     Metrics: der, jer, speaker_count_accuracy

  ...

Available Datasets:
------------------------------------------------------------

  ami: [not staged]
     AMI Meeting Corpus for diarization and summarization evaluation
     Path: /home/user/.cache/slower-whisper/benchmarks/ami
     Tasks: diarization, summarization, action_items
     Setup: See docs/AMI_SETUP.md

  librispeech: [available]
     ...
```

### benchmark status

Show the current benchmark infrastructure status.

```bash
slower-whisper benchmark status
```

**Output:**
```
Benchmark Infrastructure Status
============================================================

Benchmarks root: /home/user/.cache/slower-whisper/benchmarks
  Exists: True

Datasets: 1/4 available
  [OK] librispeech
  [MISSING] ami
  [MISSING] iemocap
  [MISSING] libricss
```

### benchmark run

Run a benchmark evaluation.

```bash
slower-whisper benchmark run --track TRACK [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--track` | `-t` | Evaluation track (required) |
| `--dataset` | `-d` | Dataset to use (default: track-specific) |
| `--split` | `-s` | Dataset split (default: test) |
| `--limit` | `-n` | Limit number of samples |
| `--output` | `-o` | Save results to JSON file |
| `--verbose` | `-v` | Show detailed progress |

**Examples:**

```bash
# Quick smoke test (10 samples)
slower-whisper benchmark run -t asr -d librispeech -n 10 -v

# Full evaluation with saved results
slower-whisper benchmark run \
  --track diarization \
  --dataset ami \
  --split test \
  --output results/ami_der_$(date +%Y%m%d).json

# Semantic evaluation (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-ant-xxx slower-whisper benchmark run \
  --track semantic \
  --dataset ami \
  --limit 5 \
  --verbose
```

### benchmark baselines

List all stored baselines.

```bash
slower-whisper benchmark baselines
```

**Output:**
```
Stored Baselines
============================================================

Baselines directory: /home/user/.cache/slower-whisper/benchmarks/baselines

Found 5 baseline(s):

  [asr] librispeech
    Created: 2026-01-21T00:00:00Z
    Version: 1.9.2
    Metrics: wer, cer

  [diarization] ami
    Created: 2026-01-21T00:00:00Z
    Version: 1.9.2
    Metrics: der, jer, speaker_count_accuracy
  ...
```

### benchmark save-baseline

Run a benchmark and save results as a baseline for future comparisons.

```bash
slower-whisper benchmark save-baseline --track TRACK [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--track` | `-t` | Evaluation track (required) |
| `--dataset` | `-d` | Dataset to use (default: track-specific) |
| `--split` | `-s` | Dataset split (default: test) |
| `--limit` | `-n` | Limit number of samples |
| `--output` | `-o` | Also save full benchmark results JSON |
| `--verbose` | `-v` | Show detailed progress |
| `--threshold` | | Set regression threshold (e.g., `--threshold wer=0.05`) |

**Examples:**

```bash
# Create ASR baseline with default thresholds (10%)
slower-whisper benchmark save-baseline --track asr --dataset librispeech

# Create with custom thresholds
slower-whisper benchmark save-baseline --track asr --dataset librispeech \
  --threshold wer=0.05 --threshold cer=0.10

# Quick baseline from subset
slower-whisper benchmark save-baseline --track diarization --dataset ami --limit 10
```

### benchmark compare

Run a benchmark and compare results against a stored baseline.

```bash
slower-whisper benchmark compare --track TRACK [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--track` | `-t` | Evaluation track (required) |
| `--dataset` | `-d` | Dataset to use (default: track-specific) |
| `--split` | `-s` | Dataset split (default: test) |
| `--limit` | `-n` | Limit number of samples |
| `--output` | `-o` | Save comparison results JSON |
| `--verbose` | `-v` | Show detailed progress |
| `--gate` | | Exit with error if regression exceeds threshold |

**Examples:**

```bash
# Report-only comparison (never fails)
slower-whisper benchmark compare --track asr --dataset librispeech

# Gate mode (fails on regression)
slower-whisper benchmark compare --track asr --dataset librispeech --gate

# Quick check with limited samples
slower-whisper benchmark compare --track diarization --dataset ami --limit 10 -v
```

**Output:**
```
Benchmark Comparison: asr / librispeech
============================================================
Mode: report (informational only)
Baseline created: 2026-01-21T00:00:00Z

------------------------------------------------------------
Metric               Current      Baseline     Regression   Status
------------------------------------------------------------
wer                  4.8000       4.5000       +6.7%        ✓ PASS (≤10%)
cer                  1.3000       1.2000       +8.3%        ✓ PASS (≤15%)
------------------------------------------------------------

Overall: PASSED
```

## Baselines

Baselines provide a reference point for detecting performance regressions.

### Directory Structure

```
benchmarks/baselines/
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

### Baseline File Format

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
    },
    "cer": {
      "value": 1.2,
      "unit": "%",
      "threshold": 0.15
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

### Regression Policy

- **Report mode (default):** Comparison prints results but never fails
- **Gate mode (`--gate`):** Exit with error if any metric exceeds its threshold

**Regression formula:**
```
regression = (current_value - baseline_value) / baseline_value
```

A metric **fails** if `regression > threshold`.

### Updating Baselines

When intentionally changing model behavior:

1. Run the benchmark to verify expected changes
2. Save a new baseline: `slower-whisper benchmark save-baseline --track <track>`
3. Commit the updated baseline file with a descriptive message

## Output Format

Benchmark results are saved as JSON with the following structure:

```json
{
  "track": "asr",
  "dataset": "librispeech",
  "split": "dev-clean",
  "samples_evaluated": 100,
  "samples_failed": 2,
  "metrics": [
    {
      "name": "wer",
      "value": 4.23,
      "unit": "%",
      "description": "Word Error Rate (lower is better)"
    },
    {
      "name": "cer",
      "value": 1.87,
      "unit": "%",
      "description": "Character Error Rate (lower is better)"
    }
  ],
  "timestamp": "2025-01-15T10:30:00",
  "config": {},
  "errors": [
    "sample_123: Audio file not found"
  ],
  "system_info": {
    "platform": "Linux-6.6.0-x86_64-with-glibc2.39",
    "python_version": "3.12.0",
    "cuda_available": true,
    "cuda_device": "NVIDIA GeForce RTX 4090"
  }
}
```

## CI Integration

### GitHub Actions

Add benchmark evaluation to your CI pipeline:

```yaml
# .github/workflows/benchmark.yml
name: Benchmarks

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --extra full --extra diarization

      - name: Download LibriSpeech dev-clean
        run: |
          mkdir -p ~/.cache/slower-whisper/benchmarks/librispeech
          wget -q https://www.openslr.org/resources/12/dev-clean.tar.gz
          tar -xzf dev-clean.tar.gz -C ~/.cache/slower-whisper/benchmarks/librispeech/

      - name: Run ASR benchmark
        run: |
          uv run slower-whisper benchmark run \
            --track asr \
            --dataset librispeech \
            --split dev-clean \
            --limit 50 \
            --output benchmark_results.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark_results.json

      - name: Check WER threshold
        run: |
          WER=$(jq '.metrics[] | select(.name=="wer") | .value' benchmark_results.json)
          if (( $(echo "$WER > 10.0" | bc -l) )); then
            echo "WER $WER% exceeds threshold of 10%"
            exit 1
          fi
          echo "WER $WER% is within threshold"
```

### Performance Gates

Set up automatic quality gates:

```bash
#!/bin/bash
# scripts/benchmark_gate.sh

set -e

# Run benchmark
slower-whisper benchmark run \
  --track asr \
  --dataset librispeech \
  --split dev-clean \
  --limit 100 \
  --output /tmp/benchmark.json

# Check thresholds
WER=$(jq '.metrics[] | select(.name=="wer") | .value' /tmp/benchmark.json)

if (( $(echo "$WER > 5.0" | bc -l) )); then
  echo "FAIL: WER $WER% exceeds 5% threshold"
  exit 1
fi

echo "PASS: WER $WER% is acceptable"
```

## Comparison and Tracking

### Comparing Results

Use `jq` to compare benchmark runs:

```bash
# Compare two runs
jq -s '
  {
    before: .[0].metrics | map({(.name): .value}) | add,
    after: .[1].metrics | map({(.name): .value}) | add
  }
' baseline.json improved.json

# Track over time
jq -s '
  map({
    date: .timestamp[:10],
    wer: (.metrics[] | select(.name=="wer") | .value)
  })
' results/*.json
```

### Visualization

Export metrics for plotting:

```bash
# Extract metrics for CSV
for f in results/*.json; do
  date=$(jq -r '.timestamp[:10]' "$f")
  wer=$(jq '.metrics[] | select(.name=="wer") | .value' "$f")
  echo "$date,$wer"
done > metrics.csv
```

## Troubleshooting

### Dataset not found

```
Error: AMI Meeting Corpus not found at /home/user/.cache/slower-whisper/benchmarks/ami
```

**Solution:** Stage the dataset following the setup documentation:
```bash
slower-whisper benchmark status  # Check which datasets are missing
# Follow setup docs for the missing dataset
```

### No samples found

```
Error: No samples found for split 'test'
```

**Solution:** Verify the split exists in your staged dataset:
```bash
ls ~/.cache/slower-whisper/benchmarks/librispeech/LibriSpeech/
# Should show: dev-clean, test-clean, etc.
```

### Slow performance

For large evaluations, consider:

1. Use `--limit N` for initial testing
2. Enable GPU if available (`--device cuda`)
3. Run overnight for full evaluations
4. Cache intermediate results

### Memory issues

For memory-constrained systems:

```bash
# Reduce batch size via environment
BATCH_SIZE=4 slower-whisper benchmark run --track asr --limit 50

# Or process fewer samples
slower-whisper benchmark run --track asr --limit 10
```

## Architecture

The benchmark CLI is built on:

1. **`transcription/benchmarks.py`**: Dataset iterators and sample loading
2. **`transcription/benchmark_cli.py`**: CLI scaffolding and runner framework
3. **`benchmarks/*.py`**: Detailed evaluation harnesses (e.g., `eval_summaries.py`)

### Extending with New Tracks

To add a new benchmark track:

1. Add track definition to `BENCHMARK_TRACKS` in `benchmark_cli.py`
2. Create a runner class extending `BenchmarkRunner`
3. Implement `get_samples()`, `evaluate_sample()`, and `aggregate_metrics()`
4. Register the runner in `get_benchmark_runner()`

Example:

```python
class CustomBenchmarkRunner(BenchmarkRunner):
    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        # Load samples from your dataset
        ...

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        # Run evaluation on single sample
        ...

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        # Compute summary metrics
        ...
```

## Related Documentation

- [Dataset Manifest Infrastructure](DATASET_MANIFEST.md) - Manifest schema, smoke datasets, fetch script
- [Semantic Benchmark Reference](SEMANTIC_BENCHMARK.md) - Gold labels, metrics, and labeling workflow
- [AMI Setup Guide](AMI_SETUP.md) - Setting up AMI corpus
- [IEMOCAP Setup Guide](IEMOCAP_SETUP.md) - Setting up IEMOCAP
- [LibriSpeech Quickstart](LIBRISPEECH_QUICKSTART.md) - ASR evaluation setup
- [Benchmark Evaluation Quickstart](BENCHMARK_EVALUATION_QUICKSTART.md) - Claude-driven evaluation loop
- [Getting Started with Evaluation](GETTING_STARTED_EVALUATION.md) - 5-step evaluation quickstart
- [LLM Semantic Annotator](LLM_SEMANTIC_ANNOTATOR.md) - Semantic annotation design document
