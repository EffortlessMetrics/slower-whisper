# Prosody Module - Quick Reference Card

## Installation
```bash
pip install praat-parselmouth librosa numpy
```

## Basic Usage
```python
from transcription.prosody import extract_prosody

features = extract_prosody(audio, sr, text)
# Returns: {'pitch': {...}, 'energy': {...}, 'rate': {...}, 'pauses': {...}}
```

## Function Signatures

### Main Functions
```python
extract_prosody(
    audio: np.ndarray,           # Mono audio (float32)
    sr: int,                     # Sample rate (Hz)
    text: str,                   # Transcribed text
    speaker_baseline: dict = None # Optional baseline
) -> dict

compute_speaker_baseline(
    segments_data: List[dict]    # [{'audio': ..., 'sr': ..., 'text': ...}]
) -> dict                        # Returns baseline stats
```

### Helper Functions
```python
extract_pitch_features(audio, sr) -> dict
extract_energy_features(audio, sr) -> dict
extract_speech_rate(audio, sr, text, duration) -> dict
extract_pause_features(audio, sr, duration) -> dict
count_syllables(text) -> int
detect_pauses(audio, sr, threshold=-40, min_duration=0.15) -> List[Tuple]
categorize_value(value, thresholds) -> str
normalize_to_baseline(value, median, std=None) -> str
```

## Output Schema
```json
{
  "pitch": {
    "level": "high",              // Category
    "mean_hz": 245.3,             // Average pitch
    "std_hz": 32.1,               // Pitch variation
    "variation": "moderate",      // Low/moderate/high
    "contour": "rising"           // Rising/falling/flat
  },
  "energy": {
    "level": "loud",              // Category
    "db_rms": -8.2,               // Loudness in dB
    "variation": "low"            // Low/moderate/high
  },
  "rate": {
    "level": "fast",              // Category
    "syllables_per_sec": 6.3,    // Speech rate
    "words_per_sec": 3.1          // Word rate
  },
  "pauses": {
    "count": 2,                   // Number of pauses
    "longest_ms": 320,            // Longest pause
    "density": "sparse"           // Pause frequency
  }
}
```

## Categories (5-level)
- **Pitch:** very_low, low, neutral, high, very_high
- **Energy:** very_quiet, quiet, normal, loud, very_loud
- **Rate:** very_slow, slow, normal, fast, very_fast
- **Pauses:** very_sparse, sparse, moderate, frequent, very_frequent

## Thresholds (Customizable)
```python
PITCH_THRESHOLDS = {
    'very_low': 80, 'low': 120, 'neutral': 180,
    'high': 250, 'very_high': inf
}

ENERGY_THRESHOLDS = {
    'very_quiet': -35, 'quiet': -25, 'normal': -15,
    'loud': -8, 'very_loud': inf
}

RATE_THRESHOLDS = {
    'very_slow': 3.0, 'slow': 4.5, 'normal': 5.5,
    'fast': 7.0, 'very_fast': inf
}

PAUSE_DENSITY_THRESHOLDS = {
    'very_sparse': 0.5, 'sparse': 1.0, 'moderate': 1.5,
    'frequent': 2.5, 'very_frequent': inf
}
```

## Common Patterns

### Extract from segment
```python
segment_audio = audio[int(start * sr):int(end * sr)]
features = extract_prosody(segment_audio, sr, text)
```

### With speaker baseline
```python
# Build baseline
segments = [{'audio': a1, 'sr': 16000, 'text': t1}, ...]
baseline = compute_speaker_baseline(segments)

# Extract normalized
features = extract_prosody(audio, sr, text, speaker_baseline=baseline)
```

### Access specific features
```python
pitch_level = features['pitch']['level']          # "high"
pitch_hz = features['pitch']['mean_hz']           # 245.3
energy_db = features['energy']['db_rms']          # -8.2
rate_syl = features['rate']['syllables_per_sec']  # 6.3
pause_count = features['pauses']['count']         # 2
```

## Error Handling
- Returns valid structure with `None` values on failure
- Falls back to numpy if parselmouth/librosa unavailable
- Never crashes - safe for production pipelines

## Performance
- **Processing time:** 50-200ms per segment (2-5 sec audio)
- **Bottleneck:** Pitch extraction
- **Parallelizable:** Yes (across segments)

## Examples
```bash
# Quick start
python3 examples/prosody_quickstart.py

# Full demo
python3 examples/prosody_demo.py

# Tests
python3 -m pytest tests/test_prosody.py -v
```

## Documentation
- Full docs: `/docs/PROSODY.md`
- Summary: `/PROSODY_IMPLEMENTATION_SUMMARY.md`
- Module: `/transcription/prosody.py` (712 lines)

## Files
- **Module:** `/transcription/prosody.py`
- **Tests:** `/tests/test_prosody.py`
- **Demo:** `/examples/prosody_demo.py`
- **Quick Start:** `/examples/prosody_quickstart.py`
- **Docs:** `/docs/PROSODY.md`
