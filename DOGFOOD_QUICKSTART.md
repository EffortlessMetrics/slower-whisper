# Dogfooding Quickstart

Ultra-concise guide to test v1.1.0 before public release.

---

## One-Command Dogfood Workflow

```bash
# Option 1: Quick synthetic test (no downloads, 30 seconds)
uv run slower-whisper-dogfood --sample synthetic

# Option 2: Real dataset test (requires HF_TOKEN)
export HF_TOKEN=hf_...
uv run slower-whisper-dogfood --sample mini-diarization

# Option 3: Custom audio file
uv run slower-whisper-dogfood --file raw_audio/my.wav
```

---

## Prerequisites (One-Time Setup)

```bash
# 1. Install dependencies
uv sync --extra diarization

# 2. Get HuggingFace token (for real datasets, not needed for synthetic)
# Visit: https://huggingface.co/settings/tokens
export HF_TOKEN=hf_...

# 3. Accept pyannote model terms (required for diarization)
# Visit these URLs while logged in to HuggingFace:
# - https://huggingface.co/pyannote/speaker-diarization-3.1
# - https://huggingface.co/pyannote/segmentation-3.0
# Click "Agree and access repository" on each page

# 4. (Optional) Set Anthropic API key for LLM integration test
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## What the Dogfood Workflow Does

The `slower-whisper-dogfood` command automatically:

1. ✅ **Checks cache** - Shows what models are cached vs. what will download
2. ✅ **Prepares sample** - Generates synthetic or copies cached dataset
3. ✅ **Transcribes with diarization** - Runs full pipeline
4. ✅ **Analyzes quality** - Computes speaker stats, turn structure, labeling coverage
5. ✅ **Tests LLM integration** (optional) - Validates JSON → LLM rendering

Time estimate:
- **Synthetic sample**: 30-60 seconds (no downloads)
- **Real dataset (first time)**: 5-10 min (downloads models once)
- **Real dataset (cached)**: 1-2 min (models already cached)

---

## CLI Options

```bash
# Sample datasets
slower-whisper-dogfood --sample synthetic           # Generated 2-speaker audio
slower-whisper-dogfood --sample mini-diarization    # Real student/professor conversation

# Custom files
slower-whisper-dogfood --file raw_audio/my.wav

# Skip steps
slower-whisper-dogfood --sample synthetic --skip-transcribe  # Use existing JSON
slower-whisper-dogfood --sample synthetic --skip-llm         # Skip LLM test

# Save results
slower-whisper-dogfood --sample synthetic --save-results dogfood_results/run.json
```

---

## Alternative: Shell Script (Same Functionality)

If you prefer shell scripts:

```bash
# Same as slower-whisper-dogfood --sample synthetic
./scripts/dogfood.sh --sample synthetic

# Same as slower-whisper-dogfood --file raw_audio/my.wav
./scripts/dogfood.sh --file raw_audio/my.wav
```

---

## Sample Dataset Management

```bash
# List available datasets
slower-whisper samples list

# Generate synthetic audio (no downloads needed)
slower-whisper samples generate

# Copy dataset to project (after manual download)
slower-whisper samples copy mini_diarization

# View cache status
slower-whisper cache --show

# Clear sample cache
slower-whisper cache --clear samples
```

---

## Understanding the Output

### Step 1: Cache Status

Shows what's already cached vs. what will download:

```
HF            /home/user/.cache/slower-whisper/hf
              (76.8 KB)
Emotion       /home/user/.cache/slower-whisper/emotion
              (3.6 GB)  ← Already cached
Diarization   /home/user/.cache/slower-whisper/diarization
              (0.0 B)   ← Will download ~3.2 GB on first run
```

### Step 2: Transcription

Runs the pipeline with diarization enabled:

```
✓ Transcription complete
```

### Step 3: Diarization Statistics

Quality metrics for quick assessment:

```
=== Speakers ===
Total speakers: 2
1. spk_0 - Speech time: 6.0s, Talk %: 47.6%, Segments: 2
2. spk_1 - Speech time: 6.6s, Talk %: 52.4%, Segments: 2

=== Labeling Coverage ===
Labeled:   4/4 segments
Coverage:  100.0%

=== Turn Structure ===
Total turns: 4
First 4 turns: spk_0 → spk_1 → spk_0 → spk_1

=== Quality Checks ===
✓ Speaker count reasonable (2)
✓ Good labeling coverage (100.0%)
✓ Good turn alternation (0 repeats)
```

**Interpretation:**
- ✅ All checks passing = ready to ship
- ⚠ Some checks failing = needs investigation
- ❌ Most checks failing = systemic issues

### Step 4: LLM Integration (Optional)

Tests the full chain: audio → JSON → LLM-ready format:

```
Speaker spk_0:
- "Hello world" (0.0-3.0s)

Speaker spk_1:
- "Hello back" (3.2-6.2s)
```

---

## Recording Findings

After running dogfood, record observations in `DOGFOOD_NOTES.md`:

```markdown
## Test: synthetic_2speaker - 2025-11-18

### Quick Summary
✅ Diarization works
✅ 2 speakers detected correctly
✅ Turn alternation looks good

### Detailed Observations
- Speaker labels: spk_0, spk_1 (expected)
- Talk time: 47.6% / 52.4% (balanced, expected for A-B-A-B pattern)
- All segments labeled (100% coverage)
- LLM rendering coherent

### Decision
✅ Ship v1.1.0 as-is - diarization working well
```

---

## Decision Framework

**Ship 1.1.0 now** if:
- ✓ Diarization works "well enough" (2 speakers detected, mostly correct)
- ✓ Quality checks mostly passing (>80%)
- ✓ LLM output coherent and useful
- ✓ No critical UX bugs

**Polish → 1.1.1** if:
- Quick wins found (doc tweaks, default changes)
- Small fixable issues that improve UX
- Nothing systemic

**Defer to 1.2** if:
- Diarization systematically fails (wrong speaker counts, random labels)
- Quality checks mostly failing (<60%)
- LLM rendering confusing
- Missing critical features

---

## Troubleshooting

**"Error: HF_TOKEN not set"**
- Required for pyannote models (diarization)
- Get token: https://huggingface.co/settings/tokens
- `export HF_TOKEN=hf_...`

**"Failed to download pyannote models"**
- Accept model terms (see Prerequisites above)
- Ensure HF_TOKEN is valid (not expired)
- Check internet connection

**"No such file or directory: mini_diarization_test.wav"**
- Sample dataset not downloaded yet
- For now, use synthetic: `--sample synthetic`
- Or download manually per DOGFOOD_SETUP.md

**Synthetic audio but no diarization models?**
- Use `--sample synthetic` for quick testing without downloads
- Transcription works, but diarization requires pyannote models

---

## Next Steps

1. Run dogfood on 1-2 samples (synthetic + real dataset)
2. Record findings in DOGFOOD_NOTES.md
3. Make release decision (ship / polish / defer)
4. If shipping: Create GitHub Release with GITHUB_RELEASE_v1.1.0.md

---

**Ready to test?**

```bash
# Quick 30-second test
uv run slower-whisper-dogfood --sample synthetic
```

All infrastructure in place. Just run it and see how v1.1.0 performs! 🚀
