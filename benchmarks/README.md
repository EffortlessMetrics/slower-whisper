# Benchmark Suite

This directory contains benchmark assets and tooling used by `slower-whisper` quality evaluation.

Canonical interface:

```bash
slower-whisper benchmark ...
```

The benchmark system is track-based and supports baseline comparison for regression gates.

## What Lives Here

| Path | Purpose |
|------|---------|
| `datasets/` | Dataset manifest definitions by track |
| `baselines/` | Stored baseline metrics per `track/dataset` |
| `gold/semantic/` | Gold labels for deterministic semantic evaluation |
| `data/` | Small committed fixtures used for smoke/evaluation |
| `*.py` scripts | Focused benchmark runners/report helpers |

## Supported Tracks

| Track | Primary Metrics | Typical Datasets |
|-------|------------------|------------------|
| `asr` | `wer`, `cer`, `rtf` | `smoke`, `librispeech`, `commonvoice_en_smoke` |
| `diarization` | `der`, `jer`, `speaker_count_accuracy` | `smoke`, `ami`, `callhome`, `libricss` |
| `streaming` | `latency_p50`, `latency_p99`, `throughput`, `rtf` | `librispeech`, `ami` |
| `semantic` | tags mode: `topic_f1`, `risk_f1`, `action_*`; summary mode: quality scores | `ami` |
| `emotion` | `accuracy`, `f1_weighted`, confusion metrics | `iemocap` |

## Quick Start

```bash
# Show tracks and datasets
uv run slower-whisper benchmark list

# Show benchmark infra status and staged data
uv run slower-whisper benchmark status

# Run ASR smoke benchmark
uv run slower-whisper benchmark run --track asr --dataset smoke

# Run deterministic semantic tags benchmark
uv run slower-whisper benchmark run --track semantic --dataset ami --mode tags --limit 10
```

## CLI Action Map

| Action | Command |
|--------|---------|
| List tracks/datasets | `slower-whisper benchmark list` |
| Show infra status | `slower-whisper benchmark status` |
| Run evaluation | `slower-whisper benchmark run --track <track> --dataset <dataset>` |
| Save baseline | `slower-whisper benchmark save-baseline --track <track> --dataset <dataset>` |
| Compare vs baseline | `slower-whisper benchmark compare --track <track> --dataset <dataset>` |
| List saved baselines | `slower-whisper benchmark baselines` |

## Baselines and Regression Gates

Create a baseline:

```bash
uv run slower-whisper benchmark save-baseline --track asr --dataset smoke
```

Compare with baseline:

```bash
# Report mode (always exits 0)
uv run slower-whisper benchmark compare --track asr --dataset smoke

# Gate mode (fails when regression exceeds threshold)
uv run slower-whisper benchmark compare --track asr --dataset smoke --gate
```

Run + gate in one command:

```bash
uv run slower-whisper benchmark run --track asr --dataset smoke --gate
```

Override thresholds:

```bash
uv run slower-whisper benchmark run --track asr --dataset smoke --gate \
  --threshold wer=0.05 --threshold cer=0.10
```

## Dataset Staging

Most non-smoke datasets are staged outside the repository cache root.

Use setup helpers:

```bash
uv run python scripts/setup_benchmark_datasets.py status
uv run python scripts/setup_benchmark_datasets.py setup librispeech-test-clean
```

See dataset docs:

- [docs/LIBRISPEECH_SETUP.md](../docs/LIBRISPEECH_SETUP.md)
- [docs/AMI_SETUP.md](../docs/AMI_SETUP.md)
- [docs/CALLHOME_SETUP.md](../docs/CALLHOME_SETUP.md)
- [docs/IEMOCAP_SETUP.md](../docs/IEMOCAP_SETUP.md)
- [docs/LIBRICSS_SETUP.md](../docs/LIBRICSS_SETUP.md)

## Legacy/Focused Scripts

The CLI is recommended for standard evaluation and CI gates. These scripts remain useful for focused workflows:

- `eval_asr.py`
- `eval_diarization.py`
- `eval_emotion.py`
- `eval_asr_diarization.py`
- `eval_speaker_utility.py`
- `eval_summaries.py`
- `compare_results.py`
- `results_reporter.py`
- `throughput.py`

## Related Docs

- [docs/BENCHMARKS.md](../docs/BENCHMARKS.md)
- [docs/SEMANTIC_BENCHMARK.md](../docs/SEMANTIC_BENCHMARK.md)
- [benchmarks/datasets/README.md](datasets/README.md)
- [benchmarks/baselines/README.md](baselines/README.md)
- [benchmarks/gold/semantic/README.md](gold/semantic/README.md)
