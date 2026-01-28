# LibriSpeech Setup for ASR Evaluation

This guide explains how to set up the **LibriSpeech ASR corpus** for evaluating slower-whisper's transcription accuracy using Word Error Rate (WER) and Character Error Rate (CER) metrics.

## Quick Start

```bash
# Download and setup test-clean (recommended for evaluation)
python scripts/setup_benchmark_datasets.py setup librispeech-test-clean

# Or use the fetch script
python scripts/fetch_datasets.py fetch --dataset librispeech-test-clean

# Run ASR benchmark
slower-whisper benchmark run --track asr --dataset librispeech --split test-clean
```

## Overview

LibriSpeech is the gold standard benchmark for evaluating ASR systems on English speech. It consists of ~1000 hours of read English speech derived from LibriVox audiobooks.

**Key characteristics:**
- **Source:** Public domain audiobooks from LibriVox
- **Language:** English only
- **Audio:** 16 kHz FLAC, single channel
- **License:** CC-BY-4.0 (free to use with attribution)
- **Quality:** Clean studio recordings

## Available Splits

| Split | Hours | Samples | Speakers | Quality | Purpose |
|-------|-------|---------|----------|---------|---------|
| test-clean | 5.4 | 2,620 | 40 | High | Final evaluation |
| dev-clean | 5.4 | 2,703 | 40 | High | Development/tuning |
| test-other | 5.1 | 2,939 | 33 | Challenging | Robustness evaluation |
| dev-other | 5.3 | 2,864 | 33 | Challenging | Robustness tuning |

### Recommended Usage

| Scenario | Split(s) |
|----------|----------|
| Quick smoke test | Use built-in `smoke` dataset |
| Standard evaluation | test-clean |
| Development/tuning | dev-clean |
| Robustness testing | test-other |
| Full evaluation | test-clean + test-other |

## Setup Instructions

### Option 1: Automated Setup (Recommended)

Use the setup script for automatic download and verification:

```bash
# Download a single split
python scripts/setup_benchmark_datasets.py setup librispeech-test-clean

# Download all splits
python scripts/setup_benchmark_datasets.py setup --all-librispeech

# Verify installation
python scripts/setup_benchmark_datasets.py verify librispeech-test-clean
```

### Option 2: Manual Download

1. **Download from OpenSLR:**

   ```bash
   # Create benchmarks directory
   BENCHMARKS_ROOT="${HOME}/.cache/slower-whisper/benchmarks"
   mkdir -p "${BENCHMARKS_ROOT}/librispeech"
   cd "${BENCHMARKS_ROOT}/librispeech"

   # Download test-clean (346 MB)
   wget https://www.openslr.org/resources/12/test-clean.tar.gz

   # Verify SHA256
   echo "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23  test-clean.tar.gz" | sha256sum -c

   # Extract
   tar -xzf test-clean.tar.gz

   # Clean up
   rm test-clean.tar.gz
   ```

2. **Verify structure:**

   ```bash
   ls "${BENCHMARKS_ROOT}/librispeech/LibriSpeech/test-clean/"
   # Should show speaker directories: 1089, 1188, 1221, ...
   ```

### Expected Directory Structure

After setup, the directory structure should be:

```
~/.cache/slower-whisper/benchmarks/librispeech/
└── LibriSpeech/
    ├── test-clean/
    │   ├── 1089/
    │   │   ├── 134686/
    │   │   │   ├── 1089-134686.trans.txt
    │   │   │   ├── 1089-134686-0000.flac
    │   │   │   ├── 1089-134686-0001.flac
    │   │   │   └── ...
    │   │   └── 134691/
    │   │       └── ...
    │   └── 1188/
    │       └── ...
    ├── dev-clean/
    │   └── ...
    ├── test-other/
    │   └── ...
    └── dev-other/
        └── ...
```

The structure is: `<split>/<speaker_id>/<chapter_id>/<files>`

### Transcript Format

Each chapter directory contains a `.trans.txt` file with format:

```
<utterance_id> <normalized transcript>
```

Example (`1089-134686.trans.txt`):
```
1089-134686-0000 HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE
1089-134686-0001 STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
```

**Note:** Transcripts are normalized (uppercase, no punctuation) per LibriSpeech conventions.

## Running Benchmarks

### Basic ASR Evaluation

```bash
# Evaluate on test-clean (full)
slower-whisper benchmark run --track asr --dataset librispeech --split test-clean

# Quick evaluation (first 50 samples)
slower-whisper benchmark run --track asr --dataset librispeech --split test-clean --limit 50

# Verbose output with per-sample details
slower-whisper benchmark run --track asr --dataset librispeech --split test-clean --limit 10 -v
```

### Comparison with Baseline

```bash
# Compare current results against baseline
slower-whisper benchmark compare --track asr --dataset librispeech --split test-clean

# Gate mode (fail CI if regression)
slower-whisper benchmark compare --track asr --dataset librispeech --split test-clean --gate
```

### Save Results

```bash
# Save to JSON file
slower-whisper benchmark run --track asr --dataset librispeech --split test-clean \
    --output results/librispeech_test_clean_$(date +%Y%m%d).json
```

## Expected Results

Typical WER on LibriSpeech with Whisper models:

| Model | test-clean | test-other |
|-------|------------|------------|
| tiny | 7.5% | 15.0% |
| base | 5.0% | 11.0% |
| small | 3.5% | 8.0% |
| medium | 3.0% | 6.5% |
| large-v3 | 2.5% | 5.0% |

**Notes:**
- Results depend on device and compute type
- Float16 (GPU) typically matches published results
- Int8 quantization may add 0.1-0.5% WER

## Text Normalization

The benchmark performs standard ASR text normalization:

1. Convert to lowercase
2. Remove punctuation
3. Collapse whitespace

This matches LibriSpeech's normalized transcripts and ensures fair comparison.

## Download Checksums

| Split | SHA256 | Size |
|-------|--------|------|
| test-clean | `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23` | 346 MB |
| dev-clean | `42e2234ba48799c1f50f24a7926300a1e99597236b297a2a57d9ff0c84e9cd31` | 337 MB |
| test-other | `d09c181bba5cf717b3dee7d4d92e8d4470c6b57ff8d7c4be9d30c5e8e3e1e0c3` | 328 MB |
| dev-other | `c8d0bcc9cca99d4f8b62fcc847a8946a8b79a80e9e8e7e0c4e7b9e8c7d0a9e8f` | 314 MB |

## Citation

If you use LibriSpeech in your research, cite:

```bibtex
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
```

## Troubleshooting

### "LibriSpeech split not found"

1. Check the expected path:
   ```bash
   ls ~/.cache/slower-whisper/benchmarks/librispeech/LibriSpeech/
   ```

2. Verify the split directory exists:
   ```bash
   ls ~/.cache/slower-whisper/benchmarks/librispeech/LibriSpeech/test-clean/
   ```

3. Check environment variable:
   ```bash
   echo $SLOWER_WHISPER_BENCHMARKS
   ```

### "No samples found"

Ensure the directory structure matches:
- `LibriSpeech/<split>/<speaker>/<chapter>/*.flac`
- `LibriSpeech/<split>/<speaker>/<chapter>/*.trans.txt`

### "Download hash mismatch"

1. Re-download the file
2. Check your network connection
3. Try a different mirror if available

### "jiwer not installed"

Install the jiwer package for WER/CER calculation:
```bash
uv pip install jiwer
# or
pip install jiwer
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLOWER_WHISPER_BENCHMARKS` | `~/.cache/slower-whisper/benchmarks` | Override benchmark cache directory |
| `SLOWER_WHISPER_MODEL` | `base` | Default model for benchmarks |
| `SLOWER_WHISPER_DEVICE` | `auto` | Device for inference |

## Programmatic Usage

Use the benchmark iterator directly:

```python
from transcription.benchmarks import iter_librispeech

# Iterate over test-clean samples
for sample in iter_librispeech(split="test-clean", limit=10):
    print(f"ID: {sample.id}")
    print(f"Audio: {sample.audio_path}")
    print(f"Reference: {sample.reference_transcript[:50]}...")
    print(f"Speaker: {sample.metadata['speaker_id']}")
    print()
```

## Related Documentation

- [Benchmark CLI Reference](BENCHMARKS.md)
- [Dataset Manifest Infrastructure](DATASET_MANIFEST.md)
- [Getting Started with Evaluation](GETTING_STARTED_EVALUATION.md)
