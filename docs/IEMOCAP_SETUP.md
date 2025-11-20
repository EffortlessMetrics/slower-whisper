# IEMOCAP Dataset Setup for Emotion Evaluation

This guide explains how to set up the **IEMOCAP (Interactive Emotional Dyadic Motion Capture) database** for evaluating slower-whisper's emotion recognition capabilities.

## Overview

IEMOCAP is a widely-used benchmark for:
- **Emotion recognition** (categorical and dimensional)
- **Prosody analysis** (pitch, energy, speech rate validation)
- **Affective computing** research

**Dataset size:** ~12 hours of audiovisual data from 10 actors (5 sessions)
**Utterances:** ~10,000 emotion-labeled speech segments
**License:** Requires registration and signed EULA (academic/research use only)
**Citation:** Required for all uses (see below)

## Why IEMOCAP?

- **Gold standard** for speech emotion recognition (SER) research
- **Multi-modal:** Audio, video, motion capture, facial expressions
- **Rich annotations:**
  - Categorical emotions (angry, happy, sad, neutral, excited, frustrated, fearful, surprised, disgusted, other)
  - Dimensional ratings (valence, arousal, dominance) from multiple annotators
  - Transcripts with word-level alignment
- **Naturalistic speech:** Both scripted and improvised dialogues
- **High inter-annotator agreement:** Multiple evaluators per utterance

## Setup Instructions

### Step 1: Accept Terms and Download

IEMOCAP requires manual download with signed EULA. **This dataset is NOT publicly redistributable.**

1. **Visit the IEMOCAP download page:**
   ```
   https://sail.usc.edu/iemocap/
   ```

2. **Request access:**
   - Click "Request Database"
   - Fill out the registration form (requires academic/research affiliation)
   - Sign and submit the End User License Agreement (EULA)
   - Wait for approval (typically 1-3 business days)

3. **Download the corpus:**
   Once approved, you'll receive download instructions via email.

   ```bash
   # You'll receive a download link and credentials
   # The full corpus is ~12GB compressed, ~25GB extracted

   # Download IEMOCAP_full_release.tar.gz
   wget --user=<username> --password=<password> <download_url>

   # Extract
   tar -xzf IEMOCAP_full_release.tar.gz
   ```

### Step 2: Stage the Data

slower-whisper expects IEMOCAP to be organized under your benchmarks cache:

```bash
# Default location
~/.cache/slower-whisper/benchmarks/iemocap/

# Or set custom location
export SLOWER_WHISPER_BENCHMARKS=/path/to/benchmarks
```

**Required directory structure:**

```
benchmarks/iemocap/
├── Session1/
│   ├── sentences/
│   │   └── wav/
│   │       ├── Ses01F_impro01/
│   │       │   ├── Ses01F_impro01_F000.wav
│   │       │   ├── Ses01F_impro01_F001.wav
│   │       │   └── ...
│   │       ├── Ses01F_impro02/
│   │       └── ...
│   └── dialog/
│       └── EmoEvaluation/
│           ├── Ses01F_impro01.txt
│           ├── Ses01F_impro02.txt
│           └── ...
├── Session2/
├── Session3/
├── Session4/
└── Session5/
```

**Copy IEMOCAP to benchmarks directory:**

```bash
# Create benchmarks directory
BENCHMARKS_ROOT="${HOME}/.cache/slower-whisper/benchmarks"
mkdir -p "${BENCHMARKS_ROOT}/iemocap"

# Copy extracted IEMOCAP
# Assuming you extracted to ~/Downloads/IEMOCAP_full_release
cp -r ~/Downloads/IEMOCAP_full_release/Session* "${BENCHMARKS_ROOT}/iemocap/"

# Verify structure
ls -la "${BENCHMARKS_ROOT}/iemocap/"
# Should see: Session1/ Session2/ Session3/ Session4/ Session5/
```

### Step 3: Understand the Annotation Format

IEMOCAP emotion annotations are stored in text files under `dialog/EmoEvaluation/`.

**Annotation file format** (`Ses01F_impro01.txt`):

```
[START_TIME - END_TIME]	TURN	EMOTION	[VAL, ACT, DOM]

Example line:
[6.2901 - 8.2357]	Ses01F_impro01_F000	ang	[2.5000, 2.5000, 2.5000]
```

**Field descriptions:**
- **START_TIME - END_TIME:** Segment boundaries in seconds
- **TURN:** Utterance ID (matches .wav filename stem)
- **EMOTION:** 3-letter categorical emotion code
- **[VAL, ACT, DOM]:** Dimensional ratings (valence, activation/arousal, dominance) on scale 1-5

**Emotion codes mapping:**

| Code | Full Label    | Description |
|------|--------------|-------------|
| `ang` | angry        | Anger, irritation, hostility |
| `hap` | happy        | Happiness, joy, contentment |
| `sad` | sad          | Sadness, sorrow, melancholy |
| `neu` | neutral      | Neutral affect, no strong emotion |
| `exc` | excited      | Excitement, enthusiasm, high arousal positive |
| `fru` | frustrated   | Frustration, annoyance |
| `fea` | fearful      | Fear, anxiety, worry |
| `sur` | surprised    | Surprise, astonishment |
| `dis` | disgusted    | Disgust, revulsion |
| `oth` | other        | Emotions not fitting above categories |

**Dimensional ratings (1-5 scale):**
- **Valence (VAL):** 1 = very negative, 5 = very positive
- **Activation/Arousal (ACT):** 1 = very calm, 5 = very excited
- **Dominance (DOM):** 1 = very submissive, 5 = very dominant

### Step 4: Label Mapping to slower-whisper Schema

slower-whisper's emotion models output different scales and labels. Here's the mapping:

**Categorical emotions:**
The `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` model outputs:
- `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

**IEMOCAP to model mapping:**
```python
IEMOCAP_TO_MODEL_CATEGORICAL = {
    "ang": "angry",
    "hap": "happy",
    "sad": "sad",
    "neu": "neutral",
    "exc": "happy",  # Excitement maps to happy (high arousal positive)
    "fru": "angry",  # Frustration maps to angry (high arousal negative)
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",
    "oth": None,  # Skip "other" category
}
```

**Dimensional emotions:**
The `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` model outputs:
- **Valence:** 0.0 (very negative) to 1.0 (very positive)
- **Arousal:** 0.0 (very calm) to 1.0 (very excited)
- **Dominance:** 0.0 (very submissive) to 1.0 (very dominant)

**IEMOCAP dimensional mapping (1-5 scale to 0-1 scale):**
```python
def iemocap_to_model_dimensional(val: float, act: float, dom: float) -> dict:
    """Convert IEMOCAP 1-5 scale to model 0-1 scale."""
    return {
        "valence": (val - 1.0) / 4.0,  # 1→0.0, 5→1.0
        "arousal": (act - 1.0) / 4.0,
        "dominance": (dom - 1.0) / 4.0,
    }
```

### Step 5: Preprocessing Considerations

**Audio preprocessing:**
- IEMOCAP audio is already 16kHz mono WAV (compatible with slower-whisper)
- Segment lengths vary (0.5s to 10s, median ~2-3s)
- No additional normalization needed

**Annotation filtering:**
- Some utterances have multiple emotion labels from different annotators
- The EmoEvaluation files contain the **majority vote** categorical label
- For dimensional ratings, use the **mean** of all annotator ratings
- Skip utterances labeled `oth` (other) for categorical evaluation

**Train/dev/test splits:**
Standard IEMOCAP evaluation uses:
- **Leave-one-session-out cross-validation** (5 folds)
- Or: **Session 1-4 train, Session 5 test**

For slower-whisper evaluation, we recommend:
```python
# Session-based split to avoid speaker leakage
TRAIN_SESSIONS = ["Session1", "Session2", "Session3"]
DEV_SESSIONS = ["Session4"]
TEST_SESSIONS = ["Session5"]
```

### Step 6: Verify Setup

Test that the benchmark infrastructure can find IEMOCAP:

```bash
# Check if IEMOCAP is detected
uv run python -c "
from transcription.benchmarks import list_available_benchmarks
benchmarks = list_available_benchmarks()
print('IEMOCAP available:', benchmarks['iemocap']['available'])
print('IEMOCAP path:', benchmarks['iemocap']['path'])
"

# List first 5 clips from Session1
uv run python -c "
from transcription.benchmarks import iter_iemocap_clips
for i, sample in enumerate(iter_iemocap_clips(session='Session1', limit=5)):
    print(f'{i+1}. {sample.id}')
    print(f'   Audio: {sample.audio_path}')
    print(f'   Emotions: {sample.reference_emotions}')
    print()
"
```

**Expected output:**
```
IEMOCAP available: True
IEMOCAP path: /home/user/.cache/slower-whisper/benchmarks/iemocap
1. Ses01F_impro01_F000
   Audio: /home/user/.cache/slower-whisper/benchmarks/iemocap/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
   Emotions: ['angry']

2. Ses01F_impro01_F001
   Audio: /home/user/.cache/slower-whisper/benchmarks/iemocap/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F001.wav
   Emotions: ['neutral']

...
```

## Running Evaluations

Once IEMOCAP is set up, you can run emotion recognition evaluations:

```bash
# Categorical emotion evaluation (accuracy, F1)
uv run python benchmarks/eval_emotion.py --dataset iemocap --session Session5 --mode categorical

# Dimensional emotion evaluation (MAE, correlation)
uv run python benchmarks/eval_emotion.py --dataset iemocap --session Session5 --mode dimensional

# Full evaluation across all sessions
uv run python benchmarks/eval_emotion.py --dataset iemocap --mode both

# Quick sanity check (10 samples)
uv run python benchmarks/eval_emotion.py --dataset iemocap --limit 10
```

Results are saved to `benchmarks/results/iemocap_emotion_<date>.json`.

## Citation

If you use IEMOCAP in your research or evaluation reports, please cite:

```bibtex
@article{busso2008iemocap,
  title={IEMOCAP: Interactive emotional dyadic motion capture database},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
  journal={Language resources and evaluation},
  volume={42},
  number={4},
  pages={335--359},
  year={2008},
  publisher={Springer}
}
```

## Troubleshooting

**"IEMOCAP dataset not found"**
- Verify path: `ls ~/.cache/slower-whisper/benchmarks/iemocap/`
- Check environment variable: `echo $SLOWER_WHISPER_BENCHMARKS`
- Ensure Session directories exist with proper capitalization

**"No emotion annotations found"**
- Verify `dialog/EmoEvaluation/*.txt` files exist in each session
- Check file permissions (should be readable)
- Ensure annotation files match the expected format

**"Audio file not found" errors**
- Verify WAV files exist under `sentences/wav/<dialog_id>/*.wav`
- Check that dialog IDs in annotation files match subdirectory names
- IEMOCAP uses specific naming: `Ses01F_impro01` not `ses01f_impro01`

**Emotion model dependencies missing**
```bash
# Install emotion recognition dependencies
uv sync --extra emotion

# Or for full enrichment features
uv sync --extra full
```

**CUDA out of memory**
- Reduce batch size (process clips one at a time)
- Use `--device cpu` flag
- Emotion models are ~1.5GB each, ensure sufficient GPU memory

## Dataset Statistics

**IEMOCAP full corpus:**
- **Total utterances:** 10,039
- **Total duration:** ~12 hours
- **Sessions:** 5 (2 actors per session)
- **Speakers:** 10 total (5 male, 5 female)
- **Dialogues:** 151 total
  - 79 scripted scenarios
  - 72 improvised scenarios

**Emotion distribution (approximate):**
- Neutral: ~36%
- Frustrated: ~19%
- Excited: ~15%
- Sad: ~13%
- Angry: ~11%
- Happy: ~5%
- Other: ~1%
- Fear, disgust, surprise: <1% each

**Important notes:**
- Highly imbalanced (neutral dominates)
- Frustrated + excited are IEMOCAP-specific (not in many other datasets)
- Happy is rare (consider merging with excited for evaluation)

## Alternative: Minimal IEMOCAP Subset

For quick testing without full IEMOCAP access, you can use publicly available alternatives:

1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - Publicly available, no registration required
   - ~1500 utterances, 8 emotions
   - More acted/exaggerated than IEMOCAP

2. **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
   - Publicly available
   - ~7400 utterances, 6 emotions
   - Multiple actors, good diversity

3. **Synthetic test set** (for smoke testing only)
   ```bash
   # Use slower-whisper's built-in synthetic audio generator
   uv run python benchmarks/generate_synthetic_emotion_clips.py
   ```

However, **IEMOCAP remains the gold standard** for emotion recognition evaluation. For publication-quality results, you must use the full corpus.

## Next Steps

- See `benchmarks/eval_emotion.py` for the evaluation harness implementation
- See `docs/AUDIO_ENRICHMENT.md` for emotion feature extraction details
- See `docs/PROSODY.md` for prosody analysis (complementary to emotion)
- See `benchmarks/README.md` for full evaluation suite documentation
