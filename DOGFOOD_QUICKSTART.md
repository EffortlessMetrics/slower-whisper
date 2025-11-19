# Dogfooding Quickstart

Ultra-concise guide to test v1.1.0 before public release.

---

## Prerequisites (One-Time Setup)

```bash
# 1. Install dependencies
uv sync --extra diarization

# 2. Get HuggingFace token
# Go to: https://huggingface.co/settings/tokens
# Create token with "read" access
export HF_TOKEN=hf_...

# 3. Accept pyannote model terms (one-time)
# Visit and click "Agree and access repository":
# - https://huggingface.co/pyannote/speaker-diarization-3.1
# - https://huggingface.co/pyannote/segmentation-3.0

# 4. Check what will be downloaded (~3.2 GB first time)
uv run python scripts/check_model_cache.py
```

---

## Get Sample Data

**Option 1: Kaggle CLI (recommended)**
```bash
# Install kaggle CLI
pip install kaggle

# Configure credentials (one-time)
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/kaggle.json
# 4. chmod 600 ~/.kaggle/kaggle.json

# Download Mini Speaker Diarization dataset
mkdir -p data/mini_diarization
kaggle datasets download -d wiradkp/mini-speech-diarization \
  -p data/mini_diarization --unzip

# Copy test file to slower-whisper layout
mkdir -p raw_audio
cp data/mini_diarization/dataset/test/test.wav \
  raw_audio/mini_diarization_test.wav
```

**Option 2: Manual download**
1. Go to https://www.kaggle.com/datasets/wiradkp/mini-speech-diarization
2. Click "Download" (requires Kaggle account)
3. Unzip to `data/mini_diarization/`
4. Copy `dataset/test/test.wav` → `raw_audio/mini_diarization_test.wav`

---

## Run Complete Test (One Command)

```bash
# Run full dogfood workflow
./scripts/dogfood.sh raw_audio/mini_diarization_test.wav

# What it does:
# 1. Checks model cache status
# 2. Transcribes with diarization (downloads models on first run)
# 3. Generates human-readable stats
# 4. Shows JSON preview
# 5. Optionally tests LLM integration (if ANTHROPIC_API_KEY set)
```

**First run:** Takes 5-10 min (downloads ~3.2 GB models, transcribes, diarizes)
**Subsequent runs:** ~1-2 min (models cached, only transcription/diarization)

---

## Manual Step-by-Step (Alternative)

If you prefer to run each step manually:

```bash
# 1. Transcribe with diarization
export HF_TOKEN=hf_...
uv run slower-whisper transcribe \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 2

# 2. Check results
uv run python scripts/diarization_stats.py \
  whisper_json/mini_diarization_test.json

# 3. Inspect JSON
jq '.meta.diarization' whisper_json/mini_diarization_test.json
jq '.speakers' whisper_json/mini_diarization_test.json
jq '.turns[0:5]' whisper_json/mini_diarization_test.json

# 4. Test LLM integration (optional)
export ANTHROPIC_API_KEY=sk-ant-...
python examples/llm_integration/summarize_with_diarization.py \
  whisper_json/mini_diarization_test.json
```

---

## What to Look For

### Diarization Quality
- ✓ `.meta.diarization.status == "success"`
- ✓ `.speakers` length is 2 (student + professor)
- ✓ Segments mostly labeled correctly (spot-check 10 random)
- ✓ Turns alternate A-B-A-B (conversational flow)

### LLM Rendering
- ✓ Uses "Student" and "Professor" labels (not `spk_0`/`spk_1`)
- ✓ Summary correctly attributes who said what
- ✓ Audio cues helpful, not overwhelming

### Common Issues (Expected for v1.1 "Experimental")
- ⚠️ A few short segments (<1s) may be mislabeled
- ⚠️ Noisy segments might split single speaker into two
- ⚠️ Perfect accuracy not expected; "generally correct" is the bar

---

## Record Findings

Open `DOGFOOD_NOTES.md` and fill in observations:

```bash
# Copy stats output for reference
uv run python scripts/diarization_stats.py \
  whisper_json/mini_diarization_test.json \
  >> DOGFOOD_NOTES.md

# Add your notes manually
vim DOGFOOD_NOTES.md  # or your editor
```

**Key questions:**
1. Did diarization work "well enough" for 2-speaker case?
2. Is LLM output coherent and useful?
3. Any quick wins to fix before public release?
4. Any blockers that prevent shipping 1.1.0?

---

## Troubleshooting

**"HuggingFace token invalid"**
```bash
# Verify token
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/whoami-v2
```

**"pyannote model access denied"**
- Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
- Accept terms: https://huggingface.co/pyannote/segmentation-3.0

**"CUDA out of memory"**
```bash
# Use CPU instead
uv run slower-whisper transcribe \
  --enable-diarization \
  --device cpu
```

**"Downloads too slow"**
- Models cached after first download (~3.2 GB)
- Location: `~/.cache/huggingface/hub/`
- Subsequent runs use cache (no re-download)

---

## Next Steps After Dogfooding

Based on findings in `DOGFOOD_NOTES.md`:

**Option A: Ship 1.1.0 as-is**
- No major issues found
- Create GitHub Release with `GITHUB_RELEASE_v1.1.0.md` content
- Optional: Publish to PyPI

**Option B: Quick polish → 1.1.1**
- Found small fixable issues (defaults, docs, error messages)
- Create 1.1.1 with fixes
- Then publish 1.1.1 instead

**Option C: Deeper work needed**
- Systemic issues found
- Triage into v1.2 roadmap
- Keep v1.1.0 as internal baseline

---

## Files Reference

| File | Purpose |
|------|---------|
| `DOGFOOD_SETUP.md` | Detailed setup guide with troubleshooting |
| `DOGFOOD_QUICKSTART.md` | This file - ultra-concise quick reference |
| `DOGFOOD_NOTES.md` | Template for recording findings |
| `scripts/dogfood.sh` | One-command complete test workflow |
| `scripts/check_model_cache.py` | Check what models are cached |
| `scripts/diarization_stats.py` | Human-readable stats from JSON |
| `PRE_RELEASE_TEST_PLAN.md` | Complete validation scenarios (more thorough) |

---

## Estimated Time

- **Setup (first time):** 15-20 min (install, tokens, model downloads)
- **Per sample test:** 5-10 min (first run), 1-2 min (cached)
- **Recording findings:** 5-10 min per sample
- **Total for 2-3 samples:** ~1 hour including setup

After dogfooding, you'll know whether 1.1.0 is ready to ship publicly or needs polish.
