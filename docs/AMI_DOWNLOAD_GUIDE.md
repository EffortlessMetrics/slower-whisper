# AMI Meeting Corpus Download Guide

Quick reference for downloading and staging real AMI Meeting Corpus data for evaluation.

## Quick Start (Recommended Test Set)

Download the 16 test meetings used in AMI's standard evaluation:

```bash
# Set target directory
BENCH_ROOT="${HOME}/.cache/slower-whisper/benchmarks/ami"
mkdir -p "${BENCH_ROOT}/audio"
mkdir -p "${BENCH_ROOT}/annotations"
mkdir -p "${BENCH_ROOT}/splits"

# Download test set meetings (ES prefix)
cd "${BENCH_ROOT}/audio"

# ES2002 series (4 meetings from same team)
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002b/audio/ES2002b.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002c/audio/ES2002c.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002d/audio/ES2002d.Mix-Headset.wav

# ES2003 series
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2003a/audio/ES2003a.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2003b/audio/ES2003b.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2003c/audio/ES2003c.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2003d/audio/ES2003d.Mix-Headset.wav

# ES2004 series
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004b/audio/ES2004b.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004c/audio/ES2004c.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004d/audio/ES2004d.Mix-Headset.wav

# ES2005 series
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2005a/audio/ES2005a.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2005b/audio/ES2005b.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2005c/audio/ES2005c.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2005d/audio/ES2005d.Mix-Headset.wav
```

**Total size**: ~5GB for 16 meetings

## Create Test Split File

```bash
cat > "${BENCH_ROOT}/splits/test.txt" <<'EOF'
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

## Download Annotations (Manual Step Required)

AMI provides annotations in multiple formats. You'll need to:

### Option A: Use Official AMI Annotations

1. **Download annotation files from AMI website**:
   ```bash
   # Visit: https://groups.inf.ed.ac.uk/ami/download/
   # Download: AMI Corpus Annotations (da-extseg, abstractive summaries, etc.)
   ```

2. **Convert to slower-whisper JSON format**:
   - AMI provides:
     - **Words.xml**: Word-level alignments with speaker IDs
     - **Segments.xml**: Speaker segments with timestamps
     - **Abstractive-summaries.xml**: Meeting summaries

   - You'll need to write a converter (see template below)

### Option B: Create Manual Annotations

For quick testing, create minimal annotations:

```bash
# Example: ES2002a.json
cat > "${BENCH_ROOT}/annotations/ES2002a.json" <<'EOF'
{
  "summary": "The project team held their first design meeting to discuss the remote control redesign. The project manager outlined the budget constraints, the industrial designer presented technical requirements, the user interface designer proposed a simplified button layout, and the marketing expert emphasized targeting a younger demographic. Key decisions included focusing on TV-only functionality to reduce costs.",
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [
        {"start": 0.0, "end": 45.2},
        {"start": 67.3, "end": 112.5}
      ]
    },
    {
      "id": "SPEAKER_01",
      "segments": [
        {"start": 45.5, "end": 67.0},
        {"start": 112.8, "end": 156.3}
      ]
    }
  ],
  "metadata": {
    "scenario": "project_meeting",
    "num_speakers": 4,
    "roles": {
      "PM": "Project Manager",
      "UI": "User Interface Designer",
      "ID": "Industrial Designer",
      "ME": "Marketing Expert"
    }
  }
}
EOF
```

**Note**: Creating accurate speaker segments manually is time-consuming. For production evaluation, use automated tools or official annotations.

## AMI Annotation Converter Template

```python
#!/usr/bin/env python3
"""Convert AMI XML annotations to slower-whisper JSON format.

This is a template - adapt to AMI's actual XML schema.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict


def parse_ami_segments(segments_xml: Path) -> list[dict]:
    """Parse AMI Segments.xml file to extract speaker turns.

    Args:
        segments_xml: Path to Segments.xml file

    Returns:
        List of speaker dicts with segments
    """
    tree = ET.parse(segments_xml)
    root = tree.getroot()

    # Group segments by speaker
    speaker_segments = defaultdict(list)

    for seg in root.findall(".//segment"):
        speaker_id = seg.get("speaker")
        start = float(seg.get("starttime"))
        end = float(seg.get("endtime"))

        speaker_segments[speaker_id].append({
            "start": start,
            "end": end
        })

    # Convert to list format
    speakers = []
    for speaker_id, segments in sorted(speaker_segments.items()):
        speakers.append({
            "id": speaker_id,
            "segments": sorted(segments, key=lambda s: s["start"])
        })

    return speakers


def parse_ami_summary(summary_xml: Path) -> str | None:
    """Parse AMI abstractive summary file.

    Args:
        summary_xml: Path to abstractive summary XML

    Returns:
        Summary text or None
    """
    if not summary_xml.exists():
        return None

    tree = ET.parse(summary_xml)
    root = tree.getroot()

    # Extract summary text (adapt to actual AMI schema)
    summary_elem = root.find(".//summary")
    if summary_elem is not None:
        return summary_elem.text.strip()

    return None


def parse_ami_transcript(words_xml: Path) -> str | None:
    """Parse AMI Words.xml to reconstruct transcript.

    Args:
        words_xml: Path to Words.xml file

    Returns:
        Full transcript text
    """
    if not words_xml.exists():
        return None

    tree = ET.parse(words_xml)
    root = tree.getroot()

    words = []
    for word in root.findall(".//w"):
        words.append(word.text or "")

    return " ".join(words).strip()


def convert_ami_meeting(
    meeting_id: str,
    ami_root: Path,
    output_path: Path,
) -> None:
    """Convert all AMI annotations for a meeting to slower-whisper format.

    Args:
        meeting_id: Meeting ID (e.g., "ES2002a")
        ami_root: Root directory of AMI corpus
        output_path: Output JSON path
    """
    # Locate AMI annotation files
    meeting_dir = ami_root / meeting_id
    segments_xml = meeting_dir / "annotations" / "Segments.xml"
    words_xml = meeting_dir / "annotations" / "Words.xml"
    summary_xml = meeting_dir / "annotations" / "Abstractive-summary.xml"

    # Parse annotations
    speakers = parse_ami_segments(segments_xml) if segments_xml.exists() else []
    transcript = parse_ami_transcript(words_xml) if words_xml.exists() else None
    summary = parse_ami_summary(summary_xml) if summary_xml.exists() else None

    # Build annotation JSON
    annotation = {
        "transcript": transcript,
        "summary": summary,
        "speakers": speakers,
        "metadata": {
            "meeting_id": meeting_id,
            "source": "ami_corpus"
        }
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(annotation, indent=2))
    print(f"Converted {meeting_id} -> {output_path}")


if __name__ == "__main__":
    # Example usage
    ami_corpus_root = Path("/path/to/ami/corpus")
    output_dir = Path.home() / ".cache/slower-whisper/benchmarks/ami/annotations"

    test_meetings = [
        "ES2002a", "ES2002b", "ES2002c", "ES2002d",
        "ES2003a", "ES2003b", "ES2003c", "ES2003d",
        "ES2004a", "ES2004b", "ES2004c", "ES2004d",
        "ES2005a", "ES2005b", "ES2005c", "ES2005d",
    ]

    for meeting_id in test_meetings:
        output_path = output_dir / f"{meeting_id}.json"
        try:
            convert_ami_meeting(meeting_id, ami_corpus_root, output_path)
        except Exception as e:
            print(f"Error converting {meeting_id}: {e}")
```

**Note**: This is a template. You'll need to adapt it to AMI's actual XML schema structure.

## Alternative: HuggingFace Dataset (Experimental)

HuggingFace hosts an AMI dataset, but it may not include all annotation types:

```python
from datasets import load_dataset
from pathlib import Path
import json

# Load AMI from HuggingFace
ami = load_dataset("edinburghcstr/ami", "headset-mix", split="test")

# Convert to slower-whisper format
output_dir = Path.home() / ".cache/slower-whisper/benchmarks/ami"

for sample in ami:
    meeting_id = sample["meeting_id"]

    # Save audio
    audio_path = output_dir / "audio" / f"{meeting_id}.wav"
    # (Save sample['audio'] to WAV file)

    # Create annotation
    annotation = {
        "transcript": sample.get("text"),  # May not be available
        "summary": sample.get("summary"),  # May not be available
        "speakers": [],  # May need to extract from diarization data
        "metadata": {}
    }

    annotation_path = output_dir / "annotations" / f"{meeting_id}.json"
    annotation_path.write_text(json.dumps(annotation, indent=2))
```

**Limitation**: HF dataset may not include speaker segments or summaries. Verify before use.

## Verify Download

After downloading, verify setup:

```bash
# Check files
ls -lh ~/.cache/slower-whisper/benchmarks/ami/audio/
ls -lh ~/.cache/slower-whisper/benchmarks/ami/annotations/

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

## Citation Requirement

If you use AMI in any publications or reports, cite:

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

## Next Steps

After downloading:

1. ✅ Verify setup: `./verify_ami_setup.sh`
2. ✅ Run test evaluation: `uv run python benchmarks/eval_asr_diarization.py --dataset ami --n 2`
3. ✅ Check results: `cat benchmarks/results/asr_diar_ami.json`
4. ✅ Expand to full test set (16 meetings)

## Troubleshooting

### Download fails with 404

AMI mirror may be down or URLs changed. Alternative mirrors:

- Main site: https://groups.inf.ed.ac.uk/ami/corpus/
- Mirror list: Check AMI website for current mirrors

### Slow downloads

AMI files are large (~300MB per meeting). Use parallel downloads:

```bash
# Download in parallel (GNU parallel)
cat "${BENCH_ROOT}/splits/test.txt" | parallel -j 4 \
  'wget -P ${BENCH_ROOT}/audio https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{}/audio/{}.Mix-Headset.wav'
```

### Out of disk space

Test set requires ~5GB. If space-constrained, start with 2-3 meetings:

```bash
# Minimal test set (ES2002a-c)
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002b/audio/ES2002b.Mix-Headset.wav
wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002c/audio/ES2002c.Mix-Headset.wav
```

## See Also

- [AMI_SETUP.md](AMI_SETUP.md) - Complete setup guide
- [AMI_ANNOTATION_SCHEMA.md](AMI_ANNOTATION_SCHEMA.md) - JSON format specification
- [AMI_INTEGRATION_SUMMARY.md](AMI_INTEGRATION_SUMMARY.md) - Integration overview
