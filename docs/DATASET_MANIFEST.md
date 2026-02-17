# Dataset Manifest Infrastructure

This document describes the benchmark dataset manifest infrastructure for reproducible benchmarks.

## Overview

The manifest infrastructure provides:

1. **Standardized schema** for describing benchmark datasets
2. **Smoke datasets** committed to the repo for CI (always available)
3. **Fetch script** for downloading datasets by name
4. **Baseline store** for regression testing
5. **CI integration** using manifests

## Quick Start

```bash
# List all datasets
python scripts/fetch_datasets.py list

# Verify smoke datasets
python scripts/fetch_datasets.py fetch --smoke

# Run ASR benchmark on smoke dataset
slower-whisper benchmark run --track asr --dataset smoke

# Compare against baseline
slower-whisper benchmark compare --track asr --dataset smoke
```

## Directory Structure

```
benchmarks/
├── manifest_schema.json           # JSON Schema for manifest validation
├── baselines/                     # Baseline results for regression testing
│   ├── asr/
│   │   ├── librispeech.json       # Full dataset baseline
│   │   └── smoke.json             # Smoke test baseline
│   └── diarization/
│       ├── ami.json
│       └── smoke.json
├── datasets/                      # Dataset manifests
│   ├── asr/
│   │   ├── smoke/                 # CI smoke tests (always available)
│   │   │   └── manifest.json
│   │   ├── librispeech-test-clean/
│   │   │   └── manifest.json
│   │   └── commonvoice_en_smoke/
│   │       ├── manifest.json
│   │       └── selection.csv
│   └── diarization/
│       ├── smoke/
│       │   └── manifest.json
│       ├── ami-headset/
│       │   └── manifest.json
│       └── callhome-english/
│           └── manifest.json
└── data/                          # Committed smoke test data
    ├── asr/audio/
    │   ├── call_center_narrowband.wav
    │   ├── team_sync_meeting.wav
    │   └── status_update_clean.wav
    └── diarization/
        ├── synthetic_2speaker.wav
        ├── synthetic_2speaker.rttm
        └── ...
```

## Manifest Schema

All manifests follow `benchmarks/manifest_schema.json` (JSON Schema draft 2020-12).

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Always 1 |
| `id` | string | Unique dataset identifier |
| `track` | string | Benchmark track (asr, diarization, etc.) |
| `split` | string | Dataset split (smoke, test, dev, train) |
| `source` | object | Dataset provenance |
| `license` | object | License information |
| `meta` | object | Aggregate statistics |

### Sample Definition

For smoke datasets, samples are explicitly defined:

```json
{
  "samples": [
    {
      "id": "sample_001",
      "audio": "../../../data/asr/audio/sample.wav",
      "sha256": "abc123...",
      "duration_s": 6.5,
      "language": "en",
      "reference_transcript": "Ground truth text.",
      "license": "MIT",
      "source": "synthetic-tts"
    }
  ]
}
```

### Sample Discovery

For large datasets, samples are discovered dynamically:

```json
{
  "samples": [],
  "sample_discovery": {
    "method": "filesystem",
    "pattern": "**/*.flac",
    "transcript_pattern": "**/*.trans.txt",
    "transcript_format": "librispeech"
  }
}
```

Or using a splits file:

```json
{
  "samples": [],
  "sample_discovery": {
    "method": "splits_file",
    "splits_file": "splits/test.txt",
    "audio_pattern": "audio/{meeting_id}.wav",
    "rttm_pattern": "rttm/{meeting_id}.rttm"
  }
}
```

### Expected Baseline

Smoke datasets can define expected performance:

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

## Smoke Datasets

Smoke datasets are minimal datasets committed to the repository for quick CI validation.

### Characteristics

- Small (few seconds of audio)
- Always available (no download required)
- Synthetic or permissively licensed
- Deterministic results
- Fast to run (typically <60 seconds)

### Available Smoke Datasets

| Dataset | Track | Samples | Duration | Description |
|---------|-------|---------|----------|-------------|
| asr-smoke | ASR | 3 | 21s | Synthetic TTS audio |
| diarization-smoke | Diarization | 3 | 39.47s | Synthetic dual-voice speech |
| diarization-smoke-tones | Diarization | 3 | 34.1s | Legacy deterministic tone fixtures |

### Using Smoke Datasets

```bash
# Run benchmark
slower-whisper benchmark run --track asr --dataset smoke

# Compare with baseline
slower-whisper benchmark compare --track asr --dataset smoke --gate
```

## Fetch Script

The `scripts/fetch_datasets.py` script provides unified dataset management.

### Commands

```bash
# List all datasets
python scripts/fetch_datasets.py list
python scripts/fetch_datasets.py list --smoke-only
python scripts/fetch_datasets.py list --track asr

# Fetch datasets
python scripts/fetch_datasets.py fetch --dataset librispeech-test-clean
python scripts/fetch_datasets.py fetch --smoke
python scripts/fetch_datasets.py fetch --track asr
python scripts/fetch_datasets.py fetch --dataset ami-headset --force

# Verify integrity
python scripts/fetch_datasets.py verify --dataset asr-smoke

# Show license info
python scripts/fetch_datasets.py license --dataset commonvoice_en_smoke

# Validate manifests
python scripts/fetch_datasets.py validate

# Show status
python scripts/fetch_datasets.py status
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLOWER_WHISPER_BENCHMARKS` | `~/.cache/slower-whisper/benchmarks` | Cache directory for downloaded datasets |

## Baselines

Baselines are reference results for detecting regressions.

### File Format

```json
{
  "schema_version": 1,
  "track": "asr",
  "dataset": "smoke",
  "created_at": "2026-01-26T00:00:00Z",
  "metrics": {
    "wer": {
      "value": 5.0,
      "unit": "%",
      "threshold": 0.50
    }
  },
  "receipt": {
    "tool_version": "1.9.2",
    "model": "base",
    "device": "cpu"
  }
}
```

### Regression Detection

```
regression = (current - baseline) / baseline
```

A metric **fails** if `regression > threshold`.

### Smoke vs Full Baselines

| Type | Threshold | Purpose |
|------|-----------|---------|
| Smoke | 50% | Catch catastrophic regressions |
| Full | 5-15% | Production quality gates |

Smoke thresholds are high to avoid flaky CI failures on limited samples.

## CI Integration

The benchmark CI workflow (`.github/workflows/benchmark.yml`) uses this infrastructure.

### PR Workflow

1. Runs smoke datasets (always available, no download)
2. Compares against smoke baselines
3. Posts PR comment with results
4. Never fails in report mode

### Manual Dispatch Options

- `dataset`: Choose smoke, librispeech, or commonvoice_en_smoke
- `gate_mode`: Fail CI on regression
- `save_baseline`: Save new baseline
- `limit`: Sample limit

### Example Workflow Run

```yaml
# Run smoke tests (default for PRs)
uv run slower-whisper benchmark run --track asr --dataset smoke

# Compare with smoke baseline
uv run slower-whisper benchmark compare --track asr --dataset smoke

# Upload results as artifacts
```

## Adding New Datasets

### 1. Create Manifest

Create `benchmarks/datasets/<track>/<dataset-name>/manifest.json`:

```json
{
  "schema_version": 1,
  "id": "my-dataset",
  "track": "asr",
  "split": "test",
  "description": "My custom dataset",
  "source": {
    "name": "My Dataset",
    "url": "https://example.com/dataset"
  },
  "license": {
    "id": "CC-BY-4.0",
    "name": "Creative Commons Attribution 4.0",
    "url": "https://creativecommons.org/licenses/by/4.0/"
  },
  "download": {
    "url": "https://example.com/dataset.tar.gz",
    "sha256": "...",
    "format": "tar.gz"
  },
  "meta": {
    "created_at": "2026-01-26T00:00:00Z",
    "sample_count": 100
  }
}
```

### 2. Validate Manifest

```bash
python scripts/fetch_datasets.py validate
```

### 3. Add Iterator (if needed)

For custom sample loading, add an iterator in `transcription/benchmarks.py`:

```python
def iter_my_dataset(limit: int | None = None) -> Iterable[EvalSample]:
    """Iterate over my dataset samples."""
    # Load samples from manifest or filesystem
    ...
```

### 4. Register in Benchmark Runner

Update the appropriate runner in `transcription/benchmark_cli.py`:

```python
def get_samples(self, limit: int | None = None) -> list[EvalSample]:
    if self.dataset == "my-dataset":
        return list(iter_my_dataset(limit=limit))
    ...
```

### 5. Create Baseline

```bash
slower-whisper benchmark save-baseline --track asr --dataset my-dataset
```

## Troubleshooting

### Dataset not found

```bash
# Check dataset status
python scripts/fetch_datasets.py list

# Verify smoke datasets
python scripts/fetch_datasets.py verify --dataset asr-smoke
```

### Manifest validation errors

```bash
# Validate all manifests
python scripts/fetch_datasets.py validate

# Check schema
cat benchmarks/manifest_schema.json
```

### Baseline comparison fails

```bash
# List available baselines
slower-whisper benchmark baselines

# Create new baseline
slower-whisper benchmark save-baseline --track asr --dataset smoke
```

## Related Documentation

- [Benchmark CLI](BENCHMARKS.md)
- [Baselines README](../benchmarks/baselines/README.md)
- [Datasets README](../benchmarks/datasets/README.md)
