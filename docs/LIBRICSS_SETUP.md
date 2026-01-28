# LibriCSS Dataset Setup for Diarization Evaluation

This guide explains how to set up the **LibriCSS (LibriSpeech Continuous Speech Separation)** dataset for evaluating slower-whisper's diarization performance, especially in overlapping speech scenarios.

## Quick Reference

**Expected root:** `~/.cache/slower-whisper/benchmarks/libricss/` (or `SLOWER_WHISPER_BENCHMARKS/libricss/`)

**Iterator:** `transcription.benchmarks.iter_libricss()`

## Overview

LibriCSS is commonly used to test diarization under realistic overlap conditions. It is derived from LibriSpeech and includes multi-speaker sessions with overlapping segments.

**Primary evaluation task:** Speaker diarization (DER/JER)
**Audio format:** WAV (recommended) or other standard audio formats
**Reference format:** RTTM for speaker turns

## Setup Instructions

### Step 1: Download LibriCSS

LibriCSS is distributed separately from LibriSpeech. Follow the official dataset instructions for download and licensing.

```
https://github.com/chenzhuo1011/libri_css
```

> Note: LibriCSS licensing and redistribution terms follow the upstream sources. Always review the official documentation before use.

### Step 2: Stage the Data

slower-whisper expects LibriCSS to be staged under your benchmarks cache:

```bash
# Default location
~/.cache/slower-whisper/benchmarks/libricss/

# Or set a custom location
export SLOWER_WHISPER_BENCHMARKS=/path/to/benchmarks
```

**Required directory structure:**

```
benchmarks/libricss/
├── audio/
│   ├── <recording_id>.wav
│   └── ...
├── rttm/
│   ├── <recording_id>.rttm
│   └── ...
└── splits/
    ├── test.txt
    ├── dev.txt
    └── train.txt
```

**Key conventions:**
- `recording_id` must match between audio and RTTM (e.g., `meeting_001.wav` + `meeting_001.rttm`).
- Split files are optional. If missing, all audio files in `audio/` are used.

### Step 3: Provide RTTM Reference Annotations

Diarization metrics require reference speaker turns in RTTM format. The iterator will parse RTTM files when present.

**RTTM format:**
```
SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
```

If RTTM files are missing, benchmarks will still run but diarization metrics will be reported as `None`.

### Step 4: Verify Setup

```python
from transcription.benchmarks import iter_libricss

for sample in iter_libricss(split="test", limit=3):
    print(sample.id, sample.audio_path)
```

You should see valid paths for the first few samples.

## Troubleshooting

**"LibriCSS not found"**
- Confirm the dataset is staged under `SLOWER_WHISPER_BENCHMARKS/libricss/`
- Check that the `audio/` directory exists and contains files

**"No reference speakers"**
- Ensure RTTM files exist under `rttm/` with matching `recording_id`
- Verify RTTM formatting and speaker IDs

## Related Documentation

- [BENCHMARKS.md](BENCHMARKS.md) - Benchmark CLI usage
- [SPEAKER_DIARIZATION.md](SPEAKER_DIARIZATION.md) - Diarization design and metrics
- [CALLHOME_SETUP.md](CALLHOME_SETUP.md) - RTTM-based diarization setup pattern
