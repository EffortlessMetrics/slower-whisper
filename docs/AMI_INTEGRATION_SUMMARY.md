# AMI Meeting Corpus Integration Summary

This document provides a comprehensive overview of the AMI Meeting Corpus integration for slower-whisper evaluation.

## Current Status

✅ **IMPLEMENTED** - AMI integration is fully functional as of v1.0.0

### What's Working

1. **Dataset Iterator** (`transcription/benchmarks.py`)
   - `iter_ami_meetings()` function loads AMI samples
   - Supports train/dev/test splits via `splits/*.txt` files
   - Handles reference transcripts, summaries, and speaker diarization
   - Returns `EvalSample` dataclass with all metadata

2. **Directory Structure** (`.cache/slower-whisper/benchmarks/ami/`)
   - `audio/` - WAV audio files
   - `annotations/` - JSON reference annotations
   - `splits/` - Optional train/dev/test split manifests

3. **Annotation Schema** (see `AMI_ANNOTATION_SCHEMA.md`)
   - Standardized JSON format with `transcript`, `summary`, `speakers`, `metadata`
   - Compatible with WER, DER, and LLM-as-judge evaluation

4. **Evaluation Scripts**
   - `benchmarks/eval_asr_diarization.py` - Local WER/DER evaluation
   - `benchmarks/eval_summaries.py` - LLM-as-judge summary evaluation

5. **Utility Scripts**
   - `verify_ami_setup.sh` - Verify AMI directory structure
   - `scripts/fix_ami_annotations.py` - Convert legacy annotation formats

## File Locations

### Source Code

```
transcription/
├── benchmarks.py              # Dataset iterators (iter_ami_meetings)
├── cache.py                   # CachePaths with benchmarks_root

benchmarks/
├── eval_asr_diarization.py    # WER/DER evaluation
├── eval_summaries.py          # LLM-based summary evaluation

scripts/
├── fix_ami_annotations.py     # Annotation format converter
```

### Documentation

```
docs/
├── AMI_SETUP.md               # Complete setup instructions
├── AMI_DIRECTORY_LAYOUT.md    # Directory structure reference
├── AMI_ANNOTATION_SCHEMA.md   # JSON schema specification
├── AMI_INTEGRATION_SUMMARY.md # This file

examples/
├── ami_annotation_example.json # Complete annotation example
```

### Runtime Data

```
~/.cache/slower-whisper/benchmarks/ami/
├── audio/
│   ├── ES2002a.wav            # Mix-Headset audio files
│   ├── ES2002b.wav
│   └── ...
├── annotations/
│   ├── ES2002a.json           # Reference annotations
│   ├── ES2002b.json
│   └── ...
└── splits/                     # Optional
    ├── train.txt              # Meeting IDs for train split
    ├── dev.txt                # Meeting IDs for dev split
    └── test.txt               # Meeting IDs for test split
```

## Usage Examples

### 1. Verify AMI Setup

```bash
# Run verification script
./verify_ami_setup.sh

# Expected output:
# ✓ Checking directories...
#   Audio files:      16
#   Annotation files: 16
# ✓ Testing dataset iteration...
#   Found 16 samples
# ✓ AMI setup looks good!
```

### 2. Iterate Over AMI Samples (Python)

```python
from transcription.benchmarks import iter_ami_meetings

# Iterate over test split
for sample in iter_ami_meetings(split="test", limit=5):
    print(f"Meeting: {sample.id}")
    print(f"  Audio: {sample.audio_path}")
    print(f"  Speakers: {len(sample.reference_speakers) if sample.reference_speakers else 0}")
    print(f"  Has summary: {bool(sample.reference_summary)}")
```

### 3. Run DER/WER Evaluation

```bash
# Evaluate 2 AMI samples (local, no LLM)
uv run python benchmarks/eval_asr_diarization.py --dataset ami --n 2

# Output: benchmarks/results/asr_diar_ami.json
# Contains: WER, DER, per-sample metrics
```

### 4. Run Summary Evaluation

```bash
# Requires: ANTHROPIC_API_KEY environment variable

# Evaluate 5 AMI summaries with Claude-as-judge
uv run python benchmarks/eval_summaries.py --dataset ami --split test --n 5

# Output: benchmarks/results/ami_summaries_<timestamp>.json
# Contains: faithfulness, coverage, clarity scores
```

### 5. Fix Legacy Annotation Files

```bash
# Convert old annotation format to new schema
uv run python scripts/fix_ami_annotations.py

# This converts:
#   "reference_summary" → "summary"
#   "reference_transcript" → "transcript"
#   Moves top-level fields to "metadata"
```

## Annotation JSON Schema

### Minimal Example (Diarization Only)

```json
{
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [{"start": 0.0, "end": 10.0}]
    }
  ]
}
```

### Full Example (WER + DER + Summary)

```json
{
  "transcript": "Good morning. Let's review the agenda. Sounds good.",
  "summary": "Team discussed agenda and agreed on priorities.",
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [
        {"start": 0.0, "end": 5.0},
        {"start": 10.0, "end": 15.0}
      ]
    },
    {
      "id": "SPEAKER_01",
      "segments": [
        {"start": 5.5, "end": 9.5}
      ]
    }
  ],
  "metadata": {
    "scenario": "project_meeting",
    "duration": 15.0,
    "num_speakers": 2
  }
}
```

See `AMI_ANNOTATION_SCHEMA.md` for complete specification.

## How to Add AMI Data

### Option A: Download Official AMI Corpus

1. Visit: https://groups.inf.ed.ac.uk/ami/corpus/
2. Accept license (CC BY 4.0)
3. Download Mix-Headset audio files (cleanest quality)
4. Stage in `~/.cache/slower-whisper/benchmarks/ami/audio/`
5. Create annotation JSONs (see conversion scripts below)

### Option B: Use HuggingFace Dataset (Experimental)

```python
from datasets import load_dataset

# Note: May not include all annotation types
ami = load_dataset("edinburghcstr/ami", "headset-mix")
```

### Option C: Use Synthetic/Test Samples

For quick testing, create minimal synthetic samples:

```bash
# Already included in repository
ls ~/.cache/slower-whisper/benchmarks/ami/audio/
# TEST001.wav (2-speaker synthetic)
# TEST002.wav (1-speaker synthetic)
```

## Converting Official AMI Annotations

AMI provides annotations in XML format. Use this pattern to convert:

```python
# Example converter (adapt to AMI XML schema)
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_ami_xml(xml_path: Path, output_path: Path):
    """Convert AMI XML to slower-whisper JSON format."""
    tree = ET.parse(xml_path)

    # Extract transcript, speakers, etc. from XML
    # (Implementation depends on AMI XML structure)

    annotation = {
        "transcript": "...",
        "summary": "...",  # May need manual creation
        "speakers": [
            {
                "id": "SPEAKER_00",
                "segments": [{"start": 0.0, "end": 5.0}]
            }
        ],
        "metadata": {
            "scenario": "...",
            "duration": 600.0,
        }
    }

    output_path.write_text(json.dumps(annotation, indent=2))
```

**Note**: AMI summaries are abstractive and may need manual extraction from AMI's abstract files.

## Common Issues and Solutions

### Issue: "AMI Meeting Corpus not found"

**Cause**: AMI directory doesn't exist or is in wrong location

**Fix**:
```bash
# Check expected location
BENCH_ROOT=$(uv run python -c "from transcription.benchmarks import get_benchmarks_root; print(get_benchmarks_root())")
echo "Expected: $BENCH_ROOT/ami"

# Create structure
mkdir -p "$BENCH_ROOT/ami/audio"
mkdir -p "$BENCH_ROOT/ami/annotations"
```

### Issue: "No samples found" but files exist

**Cause**: Filename mismatch between audio and annotations

**Fix**:
```bash
# Audio files must match annotation files
# Audio:       ES2002a.wav
# Annotation:  ES2002a.json  (same stem)

# Check for mismatches
cd ~/.cache/slower-whisper/benchmarks/ami
diff <(ls audio/*.wav | sed 's|audio/||; s|\.wav||' | sort) \
     <(ls annotations/*.json | sed 's|annotations/||; s|\.json||' | sort)
```

### Issue: "Missing reference_summary" but JSON has `summary` field

**Cause**: Using old field names (`reference_summary` instead of `summary`)

**Fix**:
```bash
# Run annotation fixer
uv run python scripts/fix_ami_annotations.py
```

### Issue: WER/DER evaluation fails with missing dependencies

**Cause**: Missing `jiwer` or `pyannote.audio`

**Fix**:
```bash
# Install evaluation dependencies
uv sync --extra full --extra diarization
uv pip install jiwer

# Set HuggingFace token (required for pyannote models)
export HF_TOKEN=hf_...
```

## Evaluation Metrics

### WER (Word Error Rate)
- **Range**: 0.0 (perfect) to 1.0+ (completely wrong)
- **Computed by**: `jiwer` package
- **Requires**: `transcript` field in annotation JSON

### DER (Diarization Error Rate)
- **Range**: 0.0 (perfect) to 1.0+ (completely wrong)
- **Computed by**: `pyannote.metrics`
- **Requires**: `speakers` field in annotation JSON

### Summary Quality (LLM-as-judge)
- **Dimensions**: Faithfulness, Coverage, Clarity (0-10 each)
- **Computed by**: Claude 3.5 Sonnet via API
- **Requires**: `summary` field in annotation JSON + `ANTHROPIC_API_KEY`

## Next Steps for v1.1

Current AMI integration is **complete and functional**. Future enhancements:

### Planned Improvements

1. **Automatic AMI Downloader**
   - Script to download AMI corpus with license acceptance
   - Automatic XML → JSON conversion
   - Progress tracking and resumable downloads

2. **Expanded Test Corpus**
   - Add 10-20 real AMI meetings to evaluation set
   - Cover diverse scenarios (design, brainstorming, decision-making)
   - Include both scenario and non-scenario meetings

3. **Turn-Level Evaluation**
   - Evaluate turn detection accuracy (not just segment-level DER)
   - Measure speaker attribution quality for action items
   - Test interrupt/overlap handling

4. **Benchmark Baselines**
   - Establish v1.0 baseline metrics (WER, DER, summary scores)
   - Track regression across releases
   - Compare slower-whisper vs WhisperX vs raw Whisper

## References

### Official AMI Resources

- **Website**: https://groups.inf.ed.ac.uk/ami/corpus/
- **License**: Creative Commons Attribution 4.0
- **Citation**:
  ```bibtex
  @inproceedings{carletta2005ami,
    title={The AMI meeting corpus: A pre-announcement},
    author={Carletta, Jean and others},
    booktitle={International workshop on machine learning for multimodal interaction},
    pages={28--39},
    year={2005},
    organization={Springer}
  }
  ```

### slower-whisper Documentation

- [AMI_SETUP.md](AMI_SETUP.md) - Setup guide
- [AMI_DIRECTORY_LAYOUT.md](AMI_DIRECTORY_LAYOUT.md) - Directory structure
- [AMI_ANNOTATION_SCHEMA.md](AMI_ANNOTATION_SCHEMA.md) - JSON schema
- [BENCHMARK_EVALUATION_QUICKSTART.md](BENCHMARK_EVALUATION_QUICKSTART.md) - Evaluation workflow

### Related Tools

- **faster-whisper**: https://github.com/guillaumekln/faster-whisper
- **WhisperX**: https://github.com/m-bain/whisperX (diarization reference)
- **pyannote.audio**: https://github.com/pyannote/pyannote-audio (diarization)
- **jiwer**: https://github.com/jitsi/jiwer (WER calculation)

## Conclusion

AMI Meeting Corpus integration is **production-ready** for v1.0 evaluation workflows. The infrastructure supports:

✅ Standardized dataset iteration
✅ Multi-metric evaluation (WER, DER, LLM-judge)
✅ Flexible annotation schema
✅ Clear documentation and examples
✅ Verification and conversion utilities

All components are tested and documented for immediate use.
