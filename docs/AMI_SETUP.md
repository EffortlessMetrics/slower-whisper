# AMI Meeting Corpus Setup for Evaluation

This guide explains how to set up the **AMI Meeting Corpus** for evaluating slower-whisper's diarization and LLM-based summarization capabilities.

## Quick Reference

**Manifest:** [`benchmarks/datasets/diarization/ami-headset/manifest.json`](../benchmarks/datasets/diarization/ami-headset/manifest.json)

```bash
# Check dataset status
python scripts/download_datasets.py --dataset ami-headset --verify
```

## Overview

The AMI corpus is a widely-used benchmark for:
- **Speaker diarization** (DER calculation)
- **Meeting summarization** (extractive and abstractive)
- **Action item detection**
- **Multi-party conversation understanding**

**Dataset size:** ~100 hours of meeting recordings (test set: ~9 hours, 16 meetings)
**License:** Creative Commons Attribution 4.0
**Citation:** Required for academic use (see below)

## Why AMI?

- **Gold standard** for meeting analysis research
- **Multiple modalities:** Headset, lapel, array microphone recordings
- **Rich annotations:** Speaker turns, topics, abstractive summaries, decisions
- **Real scenarios:** Project meetings with defined roles (PM, UI designer, etc.)

## Setup Instructions

### Step 1: Accept Terms and Download

AMI requires manual download due to licensing and user agreement.

1. **Visit the AMI corpus website:**
   ```
   https://groups.inf.ed.ac.uk/ami/corpus/
   ```

2. **Review the license:**
   - AMI is released under CC BY 4.0
   - Academic use requires citation (see below)
   - Commercial use is permitted with attribution

3. **Download the corpus:**

   **Option A: Full corpus (recommended for comprehensive evaluation)**
   ```bash
   # Download AMI annotations and audio
   # Visit: https://groups.inf.ed.ac.uk/ami/download/

   # For headset mix (cleanest audio, ~20GB):
   wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav
   # ... repeat for all meetings in your target split
   ```

   **Option B: Subset for quick evaluation**
   ```bash
   # Download just test set meetings (ES prefix)
   # Typical test set: ~16 meetings, ~5GB
   ```

   **Option C: Using HuggingFace Datasets (if available)**
   ```python
   from datasets import load_dataset

   # Note: HF AMI dataset may not include all annotation types
   ami = load_dataset("edinburghcstr/ami", "headset-mix")
   ```

### Step 2: Stage the Data

slower-whisper expects AMI to be organized under your benchmarks cache:

```bash
# Default location
~/.cache/slower-whisper/benchmarks/diarization/ami-headset/

# Or set custom location
export SLOWER_WHISPER_BENCHMARKS=/path/to/benchmarks
```

**Required directory structure:**

```
benchmarks/diarization/ami-headset/
├── audio/
│   ├── ES2002a.Mix-Headset.wav
│   ├── ES2002b.Mix-Headset.wav
│   ├── ES2002c.Mix-Headset.wav
│   └── ...
├── rttm/
│   ├── ES2002a.rttm
│   ├── ES2002b.rttm
│   └── ...
└── splits/
    ├── train.txt
    ├── dev.txt
    └── test.txt
```

**Create the structure:**

```bash
# Create directories
BENCHMARKS_ROOT="${HOME}/.cache/slower-whisper/benchmarks"
mkdir -p "${BENCHMARKS_ROOT}/diarization/ami-headset/audio"
mkdir -p "${BENCHMARKS_ROOT}/diarization/ami-headset/rttm"
mkdir -p "${BENCHMARKS_ROOT}/diarization/ami-headset/splits"

# Move downloaded audio
mv path/to/downloaded/wavs/*.wav "${BENCHMARKS_ROOT}/diarization/ami-headset/audio/"
```

### Step 3: Prepare Annotations

AMI's raw annotations are in XML format. Convert them to slower-whisper's JSON format:

**Annotation JSON format:**

```json
{
  "transcript": "So let's start the meeting. Today we'll discuss...",
  "summary": "The team discussed project timeline and decided to...",
  "speakers": [
    {
      "speaker_id": "PM",
      "start": 0.0,
      "end": 5.23,
      "text": "So let's start the meeting."
    },
    {
      "speaker_id": "UI",
      "start": 5.45,
      "end": 10.12,
      "text": "Today we'll discuss the interface redesign."
    }
  ],
  "metadata": {
    "scenario": "project_meeting",
    "roles": {
      "PM": "Project Manager",
      "UI": "UI Designer",
      "ME": "Marketing Expert",
      "ID": "Industrial Designer"
    },
    "duration": 600.5
  }
}
```

**Conversion script (example):**

```python
# scripts/convert_ami_annotations.py
import json
from pathlib import Path
import xml.etree.ElementTree as ET

def convert_ami_xml_to_json(xml_path: Path, output_path: Path):
    """Convert AMI XML annotations to slower-whisper JSON format."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract speakers, transcript, summary
    # (Implementation depends on AMI XML schema)

    annotation = {
        "transcript": "...",
        "summary": "...",
        "speakers": [...],
        "metadata": {...}
    }

    output_path.write_text(json.dumps(annotation, indent=2))

# Usage
ami_root = Path.home() / ".cache/slower-whisper/benchmarks/ami"
for xml_file in Path("ami_annotations").glob("*.xml"):
    json_file = ami_root / "annotations" / f"{xml_file.stem}.json"
    convert_ami_xml_to_json(xml_file, json_file)
```

**If you don't need full annotations:**

For quick diarization testing, you can skip annotation conversion. The evaluation harness will use slower-whisper's automatic diarization and compare structural properties (speaker count, turn alternation).

### Step 4: Create Split Files

Define train/dev/test splits:

```bash
# Test split (example)
cat > ~/.cache/slower-whisper/benchmarks/ami/splits/test.txt <<EOF
ES2002a
ES2002b
ES2002c
ES2002d
ES2003a
ES2003b
ES2003c
ES2003d
ES2004a
ES2004b
ES2004c
ES2004d
ES2005a
ES2005b
ES2005c
ES2005d
EOF
```

Standard AMI splits:
- **Train:** ~140 meetings
- **Dev:** ~18 meetings
- **Test:** ~16 meetings

### Step 5: Verify Setup

Test that the benchmark infrastructure can find AMI using the dataset helper script:

```bash
# Check dataset status
python scripts/download_datasets.py --dataset ami-headset --verify

# List all datasets
python scripts/download_datasets.py --list
```

**Expected output:**
```
============================================================
Dataset: ami-headset
============================================================
This dataset requires manual download
  Instructions: https://groups.inf.ed.ac.uk/ami/download/
  Dataset found at: /home/user/.cache/slower-whisper/benchmarks/diarization/ami-headset
  Verifying...
  Verification OK
```

You can also verify programmatically:

```bash
uv run python -c "
from pathlib import Path
import os

cache_dir = os.environ.get('SLOWER_WHISPER_BENCHMARKS', Path.home() / '.cache/slower-whisper/benchmarks')
ami_dir = Path(cache_dir) / 'diarization' / 'ami-headset'
audio_dir = ami_dir / 'audio'

if audio_dir.exists():
    wavs = list(audio_dir.glob('*.wav'))
    print(f'AMI available: True')
    print(f'AMI path: {ami_dir}')
    print(f'Audio files: {len(wavs)}')
else:
    print('AMI available: False')
    print(f'Expected at: {ami_dir}')
"
```

## Running Evaluations

Once AMI is set up, you can run benchmark evaluations:

```bash
# Diarization evaluation (DER)
uv run python benchmarks/eval_diarization.py --dataset ami --split test --n 16

# Summary evaluation (Claude-as-judge)
uv run python benchmarks/eval_summaries.py --dataset ami --split test --n 10

# Full evaluation suite
uv run python benchmarks/eval_all.py --dataset ami
```

Results are saved to `benchmarks/results/ami_<task>_<date>.json`.

## Citation

If you use AMI in your research or evaluation reports, please cite:

```bibtex
@inproceedings{carletta2005ami,
  title={The AMI meeting corpus: A pre-announcement},
  author={Carletta, Jean and Ashby, Simone and Bourban, Sebastien and Flynn, Mike and Guillemot, Mael and Hain, Thomas and Kadlec, Jaroslav and Karaiskos, Vasilis and Kraaij, Wessel and Kronenthal, Melissa and others},
  booktitle={International workshop on machine learning for multimodal interaction},
  pages={28--39},
  year={2005},
  organization={Springer}
}
```

## Troubleshooting

**"AMI Meeting Corpus not found"**
- Verify path: `ls ~/.cache/slower-whisper/benchmarks/diarization/ami-headset/`
- Check environment variable: `echo $SLOWER_WHISPER_BENCHMARKS`
- Ensure `audio/` and `rttm/` subdirectories exist
- Use `python scripts/download_datasets.py --dataset ami-headset --verify` for detailed status

**"AMI directory structure invalid"**
- Verify structure matches the template above
- Ensure WAV files are in `audio/`, not nested deeper

**Audio format issues**
- AMI uses `.wav` format at 16kHz (compatible with slower-whisper)
- If you have `.sph` or other formats, convert:
  ```bash
  ffmpeg -i input.sph -ar 16000 -ac 1 output.wav
  ```

**Missing annotations**
- For diarization-only eval, annotations are optional
- For summary eval, you need `annotations/*.json` with `summary` field
- If unavailable, you can evaluate against baseline metrics (segment count, speaker distribution)

## Alternative: Minimal AMI Subset

For quick testing without full AMI setup:

1. **Download 2-3 sample meetings:**
   ```bash
   wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav
   wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002b/audio/ES2002b.Mix-Headset.wav
   ```

2. **Place in benchmarks/ami/audio/**

3. **Run quick eval:**
   ```bash
   uv run python benchmarks/eval_diarization.py --dataset ami --n 2
   ```

This is sufficient for smoke testing the evaluation harness before investing in full corpus setup.

## Related Resources

- **Dataset manifest:** [`benchmarks/datasets/diarization/ami-headset/manifest.json`](../benchmarks/datasets/diarization/ami-headset/manifest.json)
- **Baseline metrics:** [`benchmarks/baselines/diarization/ami.json`](../benchmarks/baselines/diarization/ami.json)
- **Download helper:** `python scripts/download_datasets.py --help`

## Next Steps

- See [`IEMOCAP_SETUP.md`](IEMOCAP_SETUP.md) for emotion evaluation setup
- See [`benchmarks/datasets/README.md`](../benchmarks/datasets/README.md) for dataset manifest documentation
- See [`benchmarks/README.md`](../benchmarks/README.md) for evaluation harness documentation
