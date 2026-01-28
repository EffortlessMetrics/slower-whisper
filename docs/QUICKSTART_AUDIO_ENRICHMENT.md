# Audio Enrichment Quickstart

Get audio features (prosody & emotion) from your transcribed audio in 5 steps.

## Prerequisites

- Audio files already transcribed to JSON (Stage 1 complete)
- Audio files in `input_audio/` directory
- Python 3.12+

## 5-Step Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs audio processing, emotion recognition, and prosody extraction libraries.

### Step 2: Verify Installation (30 seconds)

```bash
python -c "
from transcription.emotion import extract_emotion_dimensional
import numpy as np

# Test with dummy audio
audio = np.random.randn(16000).astype(np.float32) * 0.1
result = extract_emotion_dimensional(audio, 16000)
print('Installation OK!')
print(result)
"
```

### Step 3: Enrich Your First Transcript

Replace `meeting1` with your actual filename:

```bash
python examples/emotion_integration.py enrich whisper_json/meeting1.json input_audio/meeting1.wav
```

This creates an enriched JSON with `audio_state` fields in each segment.

### Step 4: View the Results

```bash
# Pretty-print your enriched JSON
python -c "
import json
with open('whisper_json/meeting1.json') as f:
    data = json.load(f)
    print(json.dumps(data['segments'][0]['audio_state'], indent=2))
"
```

### Step 5: Analyze Emotions (Optional)

```bash
python examples/emotion_integration.py analyze whisper_json/meeting1.json
```

This shows a summary of emotions and prosody across the file.

## Expected Output

After enrichment, each segment will have an `audio_state` field:

```json
{
  "id": 0,
  "start": 0.0,
  "end": 4.2,
  "text": "Okay, let's get started with today's agenda.",
  "audio_state": {
    "emotion": {
      "dimensional": {
        "valence": {"level": "positive", "score": 0.68},
        "arousal": {"level": "medium", "score": 0.55},
        "dominance": {"level": "neutral", "score": 0.50}
      },
      "categorical": {
        "primary": "happy",
        "confidence": 0.82,
        "secondary": "neutral",
        "secondary_confidence": 0.12
      }
    }
  }
}
```

## Minimal Script Example

```python
#!/usr/bin/env python3
"""Minimal audio enrichment example."""

import json
import soundfile as sf
from pathlib import Path
from transcription.emotion import (
    extract_emotion_dimensional,
    extract_emotion_categorical
)

# Load transcript
with open('whisper_json/meeting1.json') as f:
    transcript = json.load(f)

# Load audio
audio_path = Path('input_audio/meeting1.wav')
audio_full, sr = sf.read(audio_path)

# Enrich first segment with emotion
seg = transcript['segments'][0]
start_sample = int(seg['start'] * sr)
end_sample = int(seg['end'] * sr)
audio_seg = audio_full[start_sample:end_sample]

dim_emotion = extract_emotion_dimensional(audio_seg, sr)
cat_emotion = extract_emotion_categorical(audio_seg, sr)

seg['audio_state'] = {
    'emotion': {
        'dimensional': dim_emotion,
        'categorical': cat_emotion['categorical']
    }
}

# Save
with open('whisper_json/meeting1_enriched.json', 'w') as f:
    json.dump(transcript, f, indent=2)

print("Done! Check meeting1_enriched.json")
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Import error: `No module named transcription` | Run from repo root directory |
| `FileNotFoundError` | Check paths to JSON and audio files |
| `CUDA out of memory` | Models use GPU; ensure 3+ GB VRAM available |
| Models download fails | Check internet connection; models auto-download on first run |
| Short audio segments < 0.5s | Still work but emotion may be less reliable |

## Next Steps

- See [docs/AUDIO_ENRICHMENT.md](docs/AUDIO_ENRICHMENT.md) for full documentation
- Customize emotion/prosody thresholds in your scripts
- Combine with text analysis for richer insights
- Export enriched JSON for downstream analysis

## Features Extracted

- **Emotion (Dimensional):** valence, arousal, dominance
- **Emotion (Categorical):** primary emotion with confidence
- **Prosody (Optional):** pitch, energy, speech rate, pauses
- **Speaker Baselines (Advanced):** normalize features to speaker patterns

That's it! You now have audio features enriching your transcripts.
