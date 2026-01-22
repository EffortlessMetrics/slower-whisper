# Common Voice Setup Guide

Common Voice is Mozilla's crowdsourced speech dataset. This guide explains how to set up the Common Voice EN smoke slice for ASR regression testing.

## Overview

The Common Voice smoke slice provides **messy real-world audio** to complement clean LibriSpeech benchmarks:

- **Accent variance**: US, UK, Indian, Australian, and other accents
- **Mic variance**: Various recording devices and environments
- **Background noise**: Real-world conditions
- **Transcript variance**: Punctuation, casing, and minor transcription errors

We use a **fixed 15-clip slice** (not random sampling) to ensure deterministic regression testing.

## License Compliance

**IMPORTANT:** Common Voice has specific terms you must follow.

### Audio License (CC0-1.0)

Audio clips are released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain).

### Mozilla Terms

By using Common Voice data, you agree to:

1. **Do NOT attempt to identify speakers** - This is forbidden and could expose personal information
2. **Do NOT redistribute the dataset** - Direct others to the official source
3. **Respect privacy** - Treat audio as potentially containing personal information

See: https://commonvoice.mozilla.org/terms

## Prerequisites

- Python 3.10+
- Hugging Face account
- ~500 MB disk space (for smoke slice)

## Setup Steps

### 1. Accept Terms on Hugging Face

Visit the dataset page and accept the terms:
https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0

### 2. Login to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
```

### 3. Stage the Smoke Slice

```bash
python benchmarks/scripts/stage_commonvoice.py
```

Options:
- `--output-dir PATH` - Custom cache directory
- `--verify` - Verify existing files without downloading
- `--update-manifest` - Update manifest.json with SHA256 hashes
- `-v` - Verbose output

### 4. Verify Installation

```bash
python benchmarks/scripts/stage_commonvoice.py --verify
```

## Usage

### Run Benchmark

```bash
# Dry run (validates manifest, no audio required)
slower-whisper benchmark --track asr --dataset commonvoice_en_smoke --dry-run

# Full run (requires staged audio)
slower-whisper benchmark --track asr --dataset commonvoice_en_smoke

# Compare against baseline
slower-whisper benchmark compare --track asr --dataset commonvoice_en_smoke
```

### CI Integration

The CI workflow validates the manifest on every PR. Full benchmark runs are available via `workflow_dispatch`:

```yaml
# .github/workflows/benchmark.yml
workflow_dispatch:
  inputs:
    dataset:
      options:
        - commonvoice_en_smoke
```

## Text Normalization

Common Voice transcripts include punctuation and casing. For fair WER comparison, we normalize both reference and hypothesis:

| Transform | Applied |
|-----------|---------|
| Lowercase | Yes |
| Remove punctuation | Yes |
| Expand contractions | No |
| Unicode normalization | NFKC |

Example:
- Raw: `"Hello, world! I'm here."`
- Normalized: `"hello world im here"`

This normalization is applied automatically by the benchmark runner.

## Troubleshooting

### "Repository not found" error

You haven't accepted the terms. Visit:
https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0

### "Invalid token" error

Re-authenticate:
```bash
huggingface-cli logout
huggingface-cli login
```

### Clips not found in dataset

The selection.csv references specific clip IDs from CV 17.0. If clips are missing, the dataset version may have changed. File an issue.

### Slow download

Common Voice is large. The staging script only downloads selected clips, but initial dataset indexing can take time.

## Related Documentation

- [Benchmark CLI](BENCHMARKS.md)
- [LibriSpeech setup](LIBRISPEECH_SETUP.md) (if exists)
- [Dataset manifest format](../benchmarks/datasets/README.md)
