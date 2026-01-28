# CALLHOME American English Setup for Diarization Evaluation

This guide explains how to set up the **CALLHOME American English** corpus for evaluating slower-whisper's speaker diarization capabilities on telephone speech.

## Quick Reference

**Manifest:** [`benchmarks/datasets/diarization/callhome-english/manifest.json`](../benchmarks/datasets/diarization/callhome-english/manifest.json)

```bash
# Check dataset status
python scripts/setup_benchmark_datasets.py status
```

## Overview

CALLHOME American English is a standard benchmark for:
- **Speaker diarization** (2-speaker telephone conversations)
- **Speaker verification** and identification
- **Conversational speech recognition**

**Dataset characteristics:**
- **Size:** ~500 MB (20 test conversations)
- **Audio:** 8 kHz narrowband telephone speech
- **Speakers:** 2 per conversation (family/friends)
- **Style:** Casual, spontaneous conversation
- **License:** LDC User Agreement (commercial license)

## Why CALLHOME?

CALLHOME complements AMI by providing:

| Aspect | CALLHOME | AMI |
|--------|----------|-----|
| Domain | Telephone | Meeting room |
| Audio quality | Narrowband (8kHz) | Wideband (16kHz) |
| Speakers per file | 2 | 4 |
| Speech style | Casual/personal | Professional |
| Overlap | Minimal | Moderate |
| Challenge | Low bandwidth, casual speech | Multiple speakers, overlaps |

Using both datasets provides a comprehensive evaluation across different diarization scenarios.

## License Requirements

CALLHOME is distributed by the Linguistic Data Consortium (LDC) and requires:

1. **LDC Membership** (institutional) - Free access
2. **Individual Purchase** - ~$250 for non-members
3. **Research Agreement** - Required for all users

**Important:** CALLHOME cannot be freely redistributed. Each user must obtain their own copy.

### Checking Institutional Access

Many universities and research institutions have LDC membership. Check with:
- Your university library
- Research computing department
- Linguistics department
- Speech/NLP research group

### Obtaining Access

1. Visit the LDC catalog page:
   ```
   https://catalog.ldc.upenn.edu/LDC97S42
   ```

2. If your institution is an LDC member:
   - Log in with institutional credentials
   - Download directly

3. If not an LDC member:
   - Create an account
   - Purchase access (~$250)

## Setup Instructions

### Step 1: Download from LDC

After obtaining access, download the CALLHOME American English corpus.

The download includes:
- Audio files in NIST SPHERE format (`.sph`)
- Transcriptions with speaker labels
- Documentation

### Step 2: Convert Audio Format

CALLHOME audio is in NIST SPHERE format. Convert to WAV for processing:

```bash
# Install sox if needed
# Ubuntu/Debian: sudo apt-get install sox
# macOS: brew install sox

# Convert all .sph files to 16-bit mono WAV
cd /path/to/callhome_download

for sph_file in *.sph; do
    wav_file="${sph_file%.sph}.wav"
    sox "$sph_file" -r 8000 -c 1 -b 16 "$wav_file"
done
```

Alternatively, use ffmpeg:

```bash
for sph_file in *.sph; do
    wav_file="${sph_file%.sph}.wav"
    ffmpeg -i "$sph_file" -ar 8000 -ac 1 -acodec pcm_s16le "$wav_file"
done
```

### Step 3: Create RTTM Reference Files

CALLHOME transcriptions include speaker labels. Convert to RTTM format:

```python
#!/usr/bin/env python3
"""Convert CALLHOME transcriptions to RTTM format."""

import re
from pathlib import Path

def parse_callhome_transcript(trans_file: Path) -> list[tuple[str, float, float, str]]:
    """Parse CALLHOME transcript to extract speaker turns.

    CALLHOME transcript format (simplified):
        speaker_id start_time end_time
        transcript text

    Returns:
        List of (speaker_id, start, end, text) tuples
    """
    turns = []
    with open(trans_file, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Parse based on CALLHOME format
    # Adjust parsing logic based on actual transcript format
    current_speaker = None
    current_start = None
    current_end = None

    for line in lines:
        # Example parsing - adjust based on actual format
        match = re.match(r'(\w+)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', line)
        if match:
            speaker, start, end = match.groups()
            turns.append((speaker, float(start), float(end), ""))

    return turns


def write_rttm(turns: list[tuple[str, float, float, str]], output_file: Path, file_id: str):
    """Write turns to RTTM format.

    RTTM format:
        SPEAKER file_id 1 start duration <NA> <NA> speaker_id <NA> <NA>
    """
    with open(output_file, 'w') as f:
        for speaker, start, end, _ in turns:
            duration = end - start
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")


# Usage
# for trans_file in Path("transcripts").glob("*.txt"):
#     turns = parse_callhome_transcript(trans_file)
#     file_id = trans_file.stem
#     write_rttm(turns, Path("rttm") / f"{file_id}.rttm", file_id)
```

**Note:** The exact parsing logic depends on the CALLHOME transcript format version. Adjust the script based on your downloaded files.

### Step 4: Stage the Dataset

Create the expected directory structure:

```bash
BENCHMARKS_ROOT="${HOME}/.cache/slower-whisper/benchmarks"
CALLHOME_DIR="${BENCHMARKS_ROOT}/diarization/callhome-english"

mkdir -p "${CALLHOME_DIR}/audio"
mkdir -p "${CALLHOME_DIR}/rttm"
mkdir -p "${CALLHOME_DIR}/splits"

# Copy converted audio files
cp /path/to/converted/*.wav "${CALLHOME_DIR}/audio/"

# Copy generated RTTM files
cp /path/to/rttm/*.rttm "${CALLHOME_DIR}/rttm/"
```

### Step 5: Create Split Files

Define the test split (standard CALLHOME evaluation set):

```bash
# Create test split with standard evaluation conversations
cat > "${CALLHOME_DIR}/splits/test.txt" << 'EOF'
4074
4093
4247
4290
4315
4325
4335
4374
4397
4415
4465
4479
4577
4609
4613
4652
4680
4711
4734
4782
EOF
```

The exact call IDs depend on your copy. The standard CALLHOME test set includes 20 conversations.

### Step 6: Verify Setup

```bash
# Check dataset status
python scripts/setup_benchmark_datasets.py status

# Verify with the benchmark CLI
slower-whisper benchmark list
```

Expected directory structure:
```
~/.cache/slower-whisper/benchmarks/diarization/callhome-english/
├── audio/
│   ├── 4074.wav
│   ├── 4093.wav
│   └── ...
├── rttm/
│   ├── 4074.rttm
│   ├── 4093.rttm
│   └── ...
└── splits/
    └── test.txt
```

## Running Evaluations

Once CALLHOME is set up:

```bash
# Run diarization benchmark
slower-whisper benchmark run --track diarization --dataset callhome-english

# Compare with baseline
slower-whisper benchmark compare --track diarization --dataset callhome-english

# Quick check with limited samples
slower-whisper benchmark run --track diarization --dataset callhome-english --limit 5 -v
```

## Expected Results

Typical DER (Diarization Error Rate) on CALLHOME:

| System | DER |
|--------|-----|
| State-of-the-art (2024) | 5-8% |
| pyannote.audio 3.x | 8-12% |
| Baseline systems | 15-25% |

CALLHOME is considered easier than AMI due to:
- Only 2 speakers (simpler speaker assignment)
- Minimal overlap
- Clear turn-taking

However, the narrowband audio (8kHz) can challenge ASR systems trained on wideband data.

## Citation

If you use CALLHOME in your research, cite:

```bibtex
@misc{callhome1997,
  title={CALLHOME American English Speech},
  author={{Linguistic Data Consortium}},
  year={1997},
  publisher={Linguistic Data Consortium},
  note={LDC97S42}
}
```

## Troubleshooting

### "CALLHOME not found"

1. Check the expected path:
   ```bash
   ls ~/.cache/slower-whisper/benchmarks/diarization/callhome-english/
   ```

2. Verify environment variable:
   ```bash
   echo $SLOWER_WHISPER_BENCHMARKS
   ```

### "Invalid audio format"

Ensure audio files are:
- WAV format (not SPHERE)
- 8000 Hz sample rate
- Mono channel
- 16-bit PCM

Convert with:
```bash
sox input.sph -r 8000 -c 1 -b 16 output.wav
```

### "RTTM parse error"

Check RTTM format:
```
SPEAKER file_id 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
```

Each line should have exactly 10 space-separated fields.

### "No reference speakers"

The benchmark requires RTTM reference files. Ensure:
1. RTTM files exist in `rttm/` directory
2. File names match audio files (e.g., `4074.wav` -> `4074.rttm`)
3. RTTM format is valid

## Alternatives

If CALLHOME is not accessible, consider:

1. **AMI Meeting Corpus** (CC-BY-4.0)
   - Free to use with attribution
   - More challenging (4 speakers, overlaps)
   - See `docs/AMI_SETUP.md`

2. **VoxConverse** (CC-BY-4.0)
   - YouTube-sourced conversations
   - Free to download
   - Variable speaker counts

3. **DIHARD Challenge data**
   - Multi-domain diarization
   - Requires registration

## Related Documentation

- [Benchmark CLI](BENCHMARKS.md)
- [AMI Setup](AMI_SETUP.md)
- [Dataset Manifest Infrastructure](DATASET_MANIFEST.md)
