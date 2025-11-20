# AMI Directory Layout Verification

This document shows the **exact** directory structure that `iter_ami_meetings()` expects. Use this as a checklist when staging AMI data.

---

## Expected Structure

```
~/.cache/slower-whisper/benchmarks/ami/
├── audio/
│   ├── ES2002a.wav
│   ├── ES2002b.wav
│   ├── ES2005a.wav
│   └── ...
├── annotations/
│   ├── ES2002a.json
│   ├── ES2002b.json
│   ├── ES2005a.json
│   └── ...
└── splits/ (optional)
    ├── train.txt
    ├── dev.txt
    └── test.txt
```

---

## Required Files

### Audio Files (`audio/*.wav`)

- **Format:** 16 kHz, mono WAV (or any format ffmpeg can handle)
- **Naming:** `{meeting_id}.wav` (e.g., `ES2002a.wav`)
- **Location:** `benchmarks_root/ami/audio/`

### Annotation Files (`annotations/*.json`)

- **Naming:** `{meeting_id}.json` (must match audio filename)
- **Location:** `benchmarks_root/ami/annotations/`

**JSON structure:**
```json
{
  "transcript": "Optional reference transcript (not used for summaries)",
  "summary": "This is the reference summary for evaluation...",
  "speakers": [
    {"id": "PM", "name": "Project Manager", "turns": 12},
    {"id": "ID", "name": "Industrial Designer", "turns": 8}
  ],
  "metadata": {
    "scenario": "Design remote control",
    "duration_minutes": 15.3,
    "num_speakers": 4
  }
}
```

**Required fields for summary evaluation:**
- `summary` (string): Reference summary for LLM-as-judge comparison

**Optional fields:**
- `transcript` (string): Reference transcript (for WER/ASR evaluation)
- `speakers` (list): Speaker metadata (for diarization evaluation)
- `metadata` (dict): Any additional metadata

### Split Files (`splits/*.txt`, optional)

If you want to use train/dev/test splits:

```
# splits/test.txt
ES2002a
ES2002b
ES2005a
```

One meeting ID per line, no `.wav` extension.

If no split file exists, `iter_ami_meetings()` will use all audio files in `audio/`.

---

## Minimum Viable Setup (for testing)

To run your first evaluation, you only need:

1. **One audio file:**
   ```
   ~/.cache/slower-whisper/benchmarks/ami/audio/ES2002a.wav
   ```

2. **One annotation file:**
   ```
   ~/.cache/slower-whisper/benchmarks/ami/annotations/ES2002a.json
   ```

   Content:
   ```json
   {
     "summary": "The team discussed the functional design of a new remote control. The industrial designer presented technical constraints, the UI designer proposed a simple button layout, and the marketing expert emphasized the need for a young, trendy appearance. They agreed to focus on TV-only functionality to keep costs down. Action items: ID to research battery options, UI to create mockups."
   }
   ```

3. **Run:**
   ```bash
   uv run python benchmarks/eval_summaries.py --dataset ami --n 1
   ```

That's it! If this works, you can expand to 5-10 meetings.

---

## Verification Commands

### Check directory structure
```bash
# Get benchmarks root path
BENCH_ROOT=$(uv run python -c "from transcription.benchmarks import get_benchmarks_root; print(get_benchmarks_root())")

# Check structure
ls -la "$BENCH_ROOT/ami/"
ls -la "$BENCH_ROOT/ami/audio/"
ls -la "$BENCH_ROOT/ami/annotations/"
```

### Count files
```bash
# Count audio files
ls "$BENCH_ROOT/ami/audio/"*.wav | wc -l

# Count annotation files
ls "$BENCH_ROOT/ami/annotations/"*.json | wc -l

# Should match!
```

### Test iteration
```bash
# Test that iter_ami_meetings() finds your files
uv run python -c "
from transcription.benchmarks import iter_ami_meetings
samples = list(iter_ami_meetings(limit=5))
print(f'Found {len(samples)} samples')
for s in samples:
    has_summary = '✓' if s.reference_summary else '✗'
    print(f'  {s.id}: audio={s.audio_path.exists()}, summary={has_summary}')
"
```

Expected output:
```
Found 2 samples
  ES2002a: audio=True, summary=✓
  ES2002b: audio=True, summary=✓
```

### Validate JSON format
```bash
# Check annotation JSON structure
jq 'keys' "$BENCH_ROOT/ami/annotations/ES2002a.json"

# Should show: ["summary"] or ["summary", "transcript", "speakers", "metadata"]
```

---

## Common Issues

### "No samples found"

**Symptom:** `iter_ami_meetings()` returns empty list

**Causes:**
1. Wrong directory structure (missing `audio/` or `annotations/` subdirectories)
2. Files not named correctly (must be `{id}.wav` and `{id}.json`)
3. Audio/annotation filename mismatch

**Fix:**
```bash
# Check structure
ls -la "$BENCH_ROOT/ami/"
# Should show: audio/ annotations/ (and optionally splits/)

# Check filenames match
ls "$BENCH_ROOT/ami/audio/" | sed 's/.wav$//' | sort > /tmp/audio_ids.txt
ls "$BENCH_ROOT/ami/annotations/" | sed 's/.json$//' | sort > /tmp/anno_ids.txt
diff /tmp/audio_ids.txt /tmp/anno_ids.txt
# Should be empty (no differences)
```

### "Missing reference summary"

**Symptom:** Evaluation skips meetings or shows `reference_summary=None`

**Cause:** JSON missing `"summary"` field

**Fix:**
```bash
# Check which files are missing summaries
for f in "$BENCH_ROOT/ami/annotations/"*.json; do
    if ! jq -e '.summary' "$f" > /dev/null 2>&1; then
        echo "Missing summary: $(basename $f)"
    fi
done
```

### "Audio file not found"

**Symptom:** `FileNotFoundError` during transcription

**Cause:** Audio file missing or wrong format

**Fix:**
```bash
# Verify audio files exist
ls -lh "$BENCH_ROOT/ami/audio/"*.wav

# Test with ffmpeg
ffmpeg -i "$BENCH_ROOT/ami/audio/ES2002a.wav" 2>&1 | grep Duration
```

---

## Quick Sanity Check Script

Save this as `verify_ami_setup.sh`:

```bash
#!/bin/bash
set -e

BENCH_ROOT=$(uv run python -c "from transcription.benchmarks import get_benchmarks_root; print(get_benchmarks_root())")
AMI_ROOT="$BENCH_ROOT/ami"

echo "Checking AMI setup at: $AMI_ROOT"
echo

# Check directories
echo "✓ Checking directories..."
test -d "$AMI_ROOT/audio" || { echo "✗ Missing audio/ directory"; exit 1; }
test -d "$AMI_ROOT/annotations" || { echo "✗ Missing annotations/ directory"; exit 1; }

# Count files
N_AUDIO=$(ls "$AMI_ROOT/audio/"*.wav 2>/dev/null | wc -l)
N_ANNOT=$(ls "$AMI_ROOT/annotations/"*.json 2>/dev/null | wc -l)

echo "  Audio files:      $N_AUDIO"
echo "  Annotation files: $N_ANNOT"

if [ "$N_AUDIO" -eq 0 ]; then
    echo "✗ No audio files found!"
    exit 1
fi

if [ "$N_ANNOT" -eq 0 ]; then
    echo "✗ No annotation files found!"
    exit 1
fi

# Test iteration
echo
echo "✓ Testing dataset iteration..."
uv run python -c "
from transcription.benchmarks import iter_ami_meetings
samples = list(iter_ami_meetings(limit=5))
if not samples:
    print('✗ No samples found!')
    exit(1)
print(f'  Found {len(samples)} samples:')
for s in samples:
    has_summary = '✓' if s.reference_summary else '✗'
    print(f'    {s.id}: summary={has_summary}')
"

echo
echo "✓ AMI setup looks good!"
echo "  Ready to run: uv run python benchmarks/eval_summaries.py --dataset ami --n $N_AUDIO"
```

Run with:
```bash
chmod +x verify_ami_setup.sh
./verify_ami_setup.sh
```

---

## Next Steps

Once verification passes:

1. ✅ `./verify_ami_setup.sh` succeeds
2. ✅ Run first evaluation: `uv run python benchmarks/eval_summaries.py --dataset ami --n 2`
3. ✅ Check results JSON exists and looks sane
4. ✅ Expand to 5-10 meetings for baseline
5. ✅ Start iteration loop (see [EVALUATION_LOOP_QUICKREF.md](EVALUATION_LOOP_QUICKREF.md))

---

For full AMI setup instructions, see [AMI_SETUP.md](AMI_SETUP.md).
