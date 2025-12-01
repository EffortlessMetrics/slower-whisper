# Dogfooding Setup Guide

This guide walks through systematic testing of slower-whisper v1.1.0 with sample datasets before public release.

---

## Pre-Flight: Model Cache Status

**Current status:**
```bash
uv run python scripts/check_model_cache.py
```

**What will be downloaded (one-time):**
- Whisper large-v3: ~3.1 GB
- Pyannote diarization models: ~140 MB
- Total: ~3.2 GB

**Already cached:**
- Emotion models: ~3.7 GB ✓

All models cached permanently at `~/.cache/huggingface/` after first download.

---

## Sample Dataset: Mini Speaker Diarization

**Source:** [Kaggle - Mini Speaker Diarization](https://www.kaggle.com/datasets/wiradkp/mini-speech-diarization)

**Why this dataset:**
- True 2-speaker conversations (student + professor)
- Clean, structured audio
- Small enough for rapid iteration
- Pre-mixed test file for validation

### Download (Option 1: Kaggle CLI)

```bash
# Install kaggle CLI if needed
pip install kaggle

# Configure API credentials (one-time)
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/kaggle.json
# 4. chmod 600 ~/.kaggle/kaggle.json

# Download dataset
mkdir -p data/mini_diarization
kaggle datasets download -d wiradkp/mini-speech-diarization \
  -p data/mini_diarization --unzip
```

### Download (Option 2: Manual)

1. Go to https://www.kaggle.com/datasets/wiradkp/mini-speech-diarization
2. Click "Download" (requires Kaggle account)
3. Unzip to `data/mini_diarization/`

### Verify Download

```bash
ls data/mini_diarization/dataset/
# Expected: raw/ train/ valid/ test/

ls data/mini_diarization/dataset/test/
# Expected: test.wav
```

---

## Dogfood Workflow

### 1. Prepare Test Audio

```bash
# Copy sample into slower-whisper's expected layout
mkdir -p raw_audio
cp data/mini_diarization/dataset/test/test.wav raw_audio/mini_diarization_test.wav
```

### 2. Baseline: Transcription Only

```bash
# Run basic transcription (no diarization)
uv run slower-whisper transcribe

# Verify outputs
ls whisper_json/mini_diarization_test.json
ls transcripts/mini_diarization_test.txt
ls transcripts/mini_diarization_test.srt

# Read transcript
head -20 transcripts/mini_diarization_test.txt
```

**Expected:** Clean text transcript, no speaker labels.

### 3. Enable Diarization

```bash
# Set HuggingFace token (required for pyannote models)
export HF_TOKEN=hf_...   # Your token from https://huggingface.co/settings/tokens

# Accept pyannote model terms (one-time):
# 1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1
# 2. Click "Agree and access repository"
# 3. Go to https://huggingface.co/pyannote/segmentation-3.0
# 4. Click "Agree and access repository"

# Run transcription with diarization
uv run slower-whisper transcribe \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 2
```

**What happens:**
- First run: Downloads pyannote models (~140 MB, 1-2 min)
- Subsequent runs: Uses cached models (no download)

### 4. Inspect Diarization Results

```bash
# Check diarization metadata
jq '.meta.diarization' whisper_json/mini_diarization_test.json

# Expected:
# {
#   "status": "success",
#   "requested": true,
#   "backend": "pyannote.audio",
#   ...
# }

# Check speaker table
jq '.speakers' whisper_json/mini_diarization_test.json

# Expected: Array with 2 speakers (spk_0, spk_1)

# Check first 10 segments with speaker labels
jq '.segments[0:10] | map({id, start, end, speaker: .speaker.id, text})' \
  whisper_json/mini_diarization_test.json

# Expected: Segments labeled with "spk_0" or "spk_1"

# Check turn structure
jq '.turns[0:5]' whisper_json/mini_diarization_test.json

# Expected: Turns alternating between speakers
```

**Quality checks:**
- ✓ `.meta.diarization.status == "success"`
- ✓ `.speakers` length is 2 (or close; not 1 or 5+)
- ✓ `.segments[*].speaker.id` mostly alternates between `"spk_0"` and `"spk_1"`
- ✓ `.turns` show A-B-A-B pattern (conversational structure)

Don't expect perfection; "generally correct" is the bar for v1.1.

### 5. Test LLM Integration

```bash
# Set LLM API key
export ANTHROPIC_API_KEY=sk-ant-...   # Or adapt script for other providers

# Run summarization example
python examples/llm_integration/summarize_with_diarization.py \
  whisper_json/mini_diarization_test.json
```

**What to observe:**
- ✓ Uses "Student" and "Professor" labels (not `spk_0`/`spk_1`)
- ✓ Summary correctly attributes who said what
- ✓ Audio cues helpful, not distracting
- ⚠️ If verbose: Try `include_audio_cues=False` in script

### 6. Generate Human-Readable Summary

```bash
# Use the stats script to get quick quality metrics
uv run python scripts/diarization_stats.py whisper_json/mini_diarization_test.json
```

(We'll create this script next)

---

## Capture Findings

Create `DOGFOOD_NOTES.md` with observations:

```markdown
# Dogfood Notes – v1.1.0

## Sample: mini_diarization_test.wav

### Date: YYYY-MM-DD

### Diarization Quality
- [ ] Speaker count stable at 2? (check .speakers length)
- [ ] Most segments correctly attributed? (spot-check 10 random segments)
- [ ] Turn structure makes sense? (check .turns for A-B-A-B pattern)
- [ ] No phantom speakers? (no spk_2, spk_3, etc.)

### LLM Rendering
- [ ] Speaker labels clear? (Student/Professor vs spk_0/spk_1)
- [ ] Summary accurate? (doesn't misattribute dialogue)
- [ ] Audio cues helpful? (or too verbose?)
- [ ] Timestamps useful? (for temporal context)

### Issues Found
- (List any specific problems: timestamps, mislabeled segments, etc.)

### Notes
- (General observations, edge cases, suggestions for 1.1.1)
```

---

## Troubleshooting

### "HuggingFace token invalid"
```bash
# Check token validity
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/whoami-v2

# If invalid, regenerate at https://huggingface.co/settings/tokens
```

### "pyannote model access denied"
```bash
# Accept model terms:
# 1. https://huggingface.co/pyannote/speaker-diarization-3.1 → "Agree and access"
# 2. https://huggingface.co/pyannote/segmentation-3.0 → "Agree and access"
```

### "CUDA out of memory"
```bash
# Use CPU instead of GPU
uv run slower-whisper transcribe \
  --enable-diarization \
  --device cpu
```

### "Downloads taking forever"
```bash
# Check cache location
echo $HF_HOME
# Default: ~/.cache/huggingface/

# Monitor download progress
du -sh ~/.cache/huggingface/hub/
```

---

## Next Steps

After dogfooding 1-2 samples:

1. **If no major issues:** Publish GitHub Release for v1.1.0 as-is
2. **If quick wins found:** Create 1.1.1 with small fixes, then publish
3. **If systemic issues:** Triage into v1.2 roadmap

See `PRE_RELEASE_TEST_PLAN.md` for complete validation scenarios.
