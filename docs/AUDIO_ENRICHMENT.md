# Audio Feature Enrichment

This document describes the Stage 2 audio enrichment pipeline that extracts prosodic and emotional features from transcribed audio segments.

## Overview

Audio enrichment is an optional second stage that runs after transcription. It analyzes the actual audio waveforms to extract features that provide deeper insight into the speech characteristics beyond the transcribed text alone.

### Why Audio-Only Features?

Text-based analysis (sentiment analysis, topic detection) tells you *what* was said. Audio-only analysis tells you *how* it was said:

- **Prosody** reveals pacing, emphasis, and intonation patterns
- **Emotion** captures affective state independent of semantic content
- **Voice quality** indicates confidence, stress, or fatigue

These features work on audio directly without requiring language-specific NLP models, making them robust across languages.

### Two-Stage Architecture

```
Raw Audio Files
    ↓
Stage 1: Transcription (faster-whisper)
    ├─ Normalize audio (ffmpeg)
    ├─ Transcribe with ASR
    └─ Output: JSON with segments
    ↓
Stage 2: Audio Enrichment (optional)
    ├─ Load segment audio from original files
    ├─ Extract prosodic features
    ├─ Extract emotional features
    └─ Populate audio_state field in JSON
    ↓
Enriched Transcripts
    └─ JSON with segments containing audio_state
```

## Semantic Enrichment (text-based)

Semantic enrichment is an opt-in text pass that runs alongside audio features when `enable_semantic_annotator=True` (CLI flag: `--enable-semantics`). It tags transcripts with lightweight, schema-stable signals under `annotations.semantic`:

- `keywords`: matched escalation/churn lexicon tokens
- `risk_tags`: canonical buckets (`escalation`, `churn_risk`)
- `actions`: action items with `text`, `speaker_id`, and `segment_ids`

Even when no signals are detected, the semantic block is emitted with empty lists to keep consumers predictable.

Example output:

```jsonc
"annotations": {
  "semantic": {
    "keywords": ["manager", "switch"],
    "risk_tags": ["escalation", "churn_risk"],
    "actions": [
      {"text": "I'll send the quote tomorrow.", "speaker_id": "spk_1", "segment_ids": [4]}
    ]
  }
}
```

## Extracted Features

### Prosody Features

Prosodic analysis captures the acoustic characteristics of speech:

#### Pitch (Fundamental Frequency)
- **mean_hz:** Average pitch in Hertz
- **std_hz:** Pitch variation (standard deviation)
- **contour:** Rising, falling, or flat pitch pattern
- **level:** Categorical pitch level (very_low, low, neutral, high, very_high)

Typical ranges:
- Very low: < 80 Hz (deep bass voices)
- Low: 80-120 Hz
- Neutral: 120-180 Hz (typical male/female average)
- High: 180-250 Hz
- Very high: > 250 Hz

#### Energy (Intensity)
- **db_rms:** RMS energy in decibels
- **variation:** How consistent the energy is (low/moderate/high)
- **level:** Categorical energy level (very_quiet, quiet, normal, loud, very_loud)

Typical ranges:
- Very quiet: < -35 dB RMS
- Quiet: -35 to -25 dB RMS
- Normal: -25 to -15 dB RMS (typical conversational speech)
- Loud: -15 to -8 dB RMS
- Very loud: > -8 dB RMS

#### Speech Rate
- **syllables_per_sec:** Articulation rate (syllables uttered per second)
- **words_per_sec:** Word rate (words uttered per second)
- **level:** Categorical rate (very_slow, slow, normal, fast, very_fast)

Typical ranges:
- Very slow: < 3.0 syl/sec
- Slow: 3.0-4.5 syl/sec
- Normal: 4.5-5.5 syl/sec (typical conversation)
- Fast: 5.5-7.0 syl/sec
- Very fast: > 7.0 syl/sec

#### Pauses
- **count:** Number of silence periods detected
- **longest_ms:** Duration of the longest pause in milliseconds
- **density_per_sec:** Pauses per second
- **density:** Categorical pause frequency (very_sparse, sparse, moderate, frequent, very_frequent)

### Emotion Features

Emotional analysis uses pre-trained wav2vec2 models for robust cross-lingual emotion recognition:

#### Dimensional Emotion (3-dimensional space)

Represents emotion as a point in a 3D space:

- **Valence:** How positive (happy, content) vs. negative (sad, angry) is the emotion?
  - Range: 0.0 (very negative) to 1.0 (very positive)
  - Levels: very_negative, negative, neutral, positive, very_positive

- **Arousal:** How excited/energetic vs. calm/passive is the emotion?
  - Range: 0.0 (very calm) to 1.0 (very excited)
  - Levels: very_low, low, medium, high, very_high

- **Dominance:** How assertive/dominant vs. submissive/passive is the emotion?
  - Range: 0.0 (very submissive) to 1.0 (very dominant)
  - Levels: very_submissive, submissive, neutral, dominant, very_dominant

#### Categorical Emotion

Discrete emotion classification:

- **primary:** Top predicted emotion (angry, disgusted, fearful, happy, neutral, sad, surprised, etc.)
- **confidence:** Confidence score for primary emotion (0.0-1.0)
- **secondary:** Second-most likely emotion
- **secondary_confidence:** Confidence for secondary emotion
- **all_scores:** Dictionary of all emotion probabilities

### Speaker Baselines

When processing multiple segments from the same speaker, features can be normalized to speaker baselines:

```python
{
    'pitch_median': 180.0,      # Speaker's typical pitch
    'pitch_std': 28.5,           # Speaker's pitch variation
    'energy_median': -15.2,      # Speaker's typical loudness
    'energy_std': 3.1,
    'rate_median': 5.3,          # Speaker's typical speech rate
    'rate_std': 0.8
}
```

When baselines are available, prosodic features are normalized relative to the speaker's typical patterns, giving relative categories (very_low, low, neutral, high, very_high) instead of absolute categories.

## Installation & Setup

### Prerequisites

- Python 3.9+
- Audio files already transcribed (Stage 1 complete)
- NVIDIA GPU recommended (emotion models are fast on GPU but work on CPU)

### Step 1: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

Or, if you already have Stage 1 set up and only want to add enrichment:

```bash
pip install soundfile librosa praat-parselmouth transformers torch
```

### Step 2: Model Downloads

Emotion recognition models are downloaded automatically on first use via Hugging Face. The models require internet access on first run:

```bash
# Dimensional emotion model (~1.2 GB)
# audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim

# Categorical emotion model (~1.2 GB)
# ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
```

Model files are cached in `~/.cache/huggingface/hub/` by default.

### Step 3: Verify Installation

```bash
# Quick test of emotion extraction
python -c "
from transcription.emotion import extract_emotion_dimensional, extract_emotion_categorical
import numpy as np

# Generate test audio (1 second of noise)
audio = np.random.randn(16000).astype(np.float32) * 0.1
sr = 16000

try:
    dim = extract_emotion_dimensional(audio, sr)
    cat = extract_emotion_categorical(audio, sr)
    print('Installation successful!')
    print(f'Dimensional: {dim}')
    print(f'Categorical: {cat}')
except Exception as e:
    print(f'Error: {e}')
"
```

## Usage Examples

### Basic Usage: Enrich a Transcript

```bash
# Enrich an existing transcript with emotion analysis
python examples/emotion_integration.py enrich whisper_json/meeting1.json input_audio/meeting1.wav

# Output: whisper_json/meeting1_enriched.json
```

### Analyze Emotions in a Transcript

```bash
# View emotion statistics across the entire file
python examples/emotion_integration.py analyze whisper_json/meeting1_enriched.json
```

Output:
```
============================================================
EMOTION ANALYSIS SUMMARY
============================================================

File: meeting1.wav
Duration: 3120.5s
Segments: 45

Categorical Emotion Distribution:
  neutral        ████████████████            28 ( 62.2%)
  frustrated     ███████                     12 ( 26.7%)
  happy          ███                          3 (  6.7%)
  angry          ▌                            2 (  4.4%)

Dimensional Averages:
  Average Valence: 0.512 (positive)
  Average Arousal: 0.438 (low energy)
============================================================
```

### Extract Prosody from Segments

```python
import soundfile as sf
from transcription.prosody import extract_prosody

# Load audio segment
audio, sr = sf.read('input_audio/meeting1.wav', start=10, stop=20)

# Extract prosody
prosody = extract_prosody(
    audio, sr,
    text="This is the transcribed text",
    start_time=10.0,
    end_time=20.0
)

print(prosody)
# {
#   'pitch': {'level': 'high', 'mean_hz': 245.3, ...},
#   'energy': {'level': 'loud', 'db_rms': -8.2, ...},
#   'rate': {'level': 'normal', 'syllables_per_sec': 5.3},
#   'pauses': {'count': 2, 'longest_ms': 320, ...}
# }
```

### Extract Emotion from Segments

```python
import soundfile as sf
from transcription.emotion import extract_emotion_dimensional, extract_emotion_categorical

# Load audio segment
audio, sr = sf.read('input_audio/meeting1.wav', start=10, stop=20)

# Dimensional emotion
dim_emotion = extract_emotion_dimensional(audio, sr)
print(dim_emotion)
# {
#   'valence': {'level': 'positive', 'score': 0.72},
#   'arousal': {'level': 'high', 'score': 0.68},
#   'dominance': {'level': 'neutral', 'score': 0.51}
# }

# Categorical emotion
cat_emotion = extract_emotion_categorical(audio, sr)
print(cat_emotion)
# {
#   'categorical': {
#     'primary': 'happy',
#     'confidence': 0.89,
#     'secondary': 'neutral',
#     'secondary_confidence': 0.08,
#     'all_scores': {...}
#   }
# }
```

### Process All Segments with Speaker Baseline

```python
import json
import soundfile as sf
import numpy as np
from pathlib import Path
from transcription.prosody import extract_prosody, compute_speaker_baseline

# Load transcript
with open('whisper_json/meeting1.json') as f:
    transcript = json.load(f)

# Collect segments for baseline computation
segments_data = []
audio_path = Path('input_audio/meeting1.wav')
audio_full, sr = sf.read(audio_path)

# Only use longer segments for baseline (> 1 second)
for seg in transcript['segments']:
    if seg['end'] - seg['start'] > 1.0:
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        audio_seg = audio_full[start_sample:end_sample]
        segments_data.append({
            'audio': audio_seg,
            'sr': sr,
            'text': seg['text']
        })

# Compute baseline for this speaker
baseline = compute_speaker_baseline(segments_data)
print(f"Speaker baseline: {baseline}")

# Now process all segments with normalization
for seg in transcript['segments']:
    start_sample = int(seg['start'] * sr)
    end_sample = int(seg['end'] * sr)
    audio_seg = audio_full[start_sample:end_sample]

    # Extract with baseline normalization
    prosody = extract_prosody(
        audio_seg, sr,
        text=seg['text'],
        speaker_baseline=baseline,
        start_time=seg['start'],
        end_time=seg['end']
    )

    seg['audio_state'] = {'prosody': prosody}

# Save enriched transcript
with open('whisper_json/meeting1_enriched.json', 'w') as f:
    json.dump(transcript, f, indent=2)
```

## Configuration Options

### Prosody Extraction

```python
extract_prosody(
    audio,                      # numpy array, mono
    sr,                         # sample rate (Hz)
    text,                       # transcribed text
    speaker_baseline=None,      # optional baseline dict for normalization
    start_time=0.0,            # for logging
    end_time=None              # for logging
)
```

### Emotion Extraction

Both functions work with numpy audio arrays at 16 kHz (resampling is automatic):

```python
# Dimensional emotion (valence/arousal/dominance)
extract_emotion_dimensional(audio, sr)

# Categorical emotion (discrete emotions)
extract_emotion_categorical(audio, sr)
```

### Silence Detection Thresholds

When detecting pauses in `detect_pauses()`:

```python
from transcription.prosody import detect_pauses

pauses = detect_pauses(
    audio,
    sr,
    silence_threshold=-40,      # dB below which is silence
    min_pause_duration=0.15     # minimum pause duration (seconds)
)
```

## JSON Schema Reference

### Segment with Audio State

```json
{
  "id": 0,
  "start": 10.5,
  "end": 14.2,
  "text": "This is a sample utterance.",
  "speaker": null,
  "tone": null,
  "audio_state": {
    "prosody": {
      "pitch": {
        "level": "high",
        "mean_hz": 245.3,
        "std_hz": 32.1,
        "variation": "moderate",
        "contour": "rising"
      },
      "energy": {
        "level": "loud",
        "db_rms": -8.2,
        "variation": "low"
      },
      "rate": {
        "level": "normal",
        "syllables_per_sec": 5.3,
        "words_per_sec": 2.8
      },
      "pauses": {
        "count": 1,
        "longest_ms": 250,
        "density": "sparse"
      }
    },
    "emotion": {
      "dimensional": {
        "valence": {
          "level": "positive",
          "score": 0.72
        },
        "arousal": {
          "level": "high",
          "score": 0.68
        },
        "dominance": {
          "level": "neutral",
          "score": 0.51
        }
      },
      "categorical": {
        "primary": "happy",
        "confidence": 0.89,
        "secondary": "neutral",
        "secondary_confidence": 0.08,
        "all_scores": {
          "angry": 0.01,
          "disgusted": 0.00,
          "fearful": 0.00,
          "happy": 0.89,
          "neutral": 0.08,
          "sad": 0.01,
          "surprised": 0.01
        }
      }
    }
  }
}
```

## Troubleshooting

### Issue: Models Download Fails

**Problem:** `OSError: Can't load 'transformers' for models ...`

**Solution:**
- Ensure internet connection is active
- Check Hugging Face API: `https://huggingface.co`
- Set custom cache directory:
  ```bash
  export HF_HOME=/path/to/cache
  python examples/emotion_integration.py enrich ...
  ```

### Issue: Out of Memory (OOM)

**Problem:** `RuntimeError: CUDA out of memory` or `MemoryError`

**Solution:**
- Ensure GPU has sufficient VRAM (emotion models need ~2-3 GB)
- Use CPU fallback:
  ```python
  # Models will automatically use CPU if CUDA fails
  # But you can force CPU:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Before importing emotion module
  ```

### Issue: Audio Segments Too Short

**Problem:** Warning: `Audio segment (0.3s) is shorter than recommended minimum (0.5s)`

**Solution:**
- This is expected for short speech segments
- Emotion models still work but may be less reliable on audio < 0.5s
- Prosody extraction works on any length but needs at least a few frames

### Issue: Pitch Extraction Returns None

**Problem:** Prosody features show `mean_hz: null`

**Solution:**
- Check if `parselmouth` is installed: `pip install praat-parselmouth`
- Verify audio format (should be mono, 16-bit PCM or float32)
- Check if audio is too noisy or not speech (e.g., silence, music)

### Issue: Emotion Scores Don't Match Text Sentiment

**Problem:** Audio has high arousal but text is neutral

**Solution:**
- This is expected! Audio captures *how* something is said, not *what* is said
- A person can say "I'm fine" (neutral text) with frustrated tone (high arousal)
- Combining audio and text features gives complete understanding

## Performance Characteristics

Approximate processing times on NVIDIA GPU:

- **Prosody extraction:** ~0.1x real-time (10 seconds of audio in ~1 second)
- **Dimensional emotion:** ~0.5x real-time (10 seconds of audio in ~20 seconds, first load slower)
- **Categorical emotion:** ~0.5x real-time (10 seconds of audio in ~20 seconds, first load slower)
- **Model loading:** ~3-5 seconds per model on first call

## References

### Emotion Recognition Models

- **Dimensional (MSP-Podcast):**
  Fei, Z., Sap, M., & Mihalcea, R. (2022). "Unsupervised and Supervised Learning of Speech-based Emotion Dimensions."
  Model: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`

- **Categorical (XLSR multilingual):**
  Ghosh, S., Laksana, E., Venkatesan, S., et al. (2020). "Multilingual and Code-Switching ASR Challenges for Low Resource Indian Languages."
  Model: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`

### Prosody Feature Extraction

- Parselmouth: Python interface to Praat speech analysis software
- Librosa: Audio analysis library for Python
