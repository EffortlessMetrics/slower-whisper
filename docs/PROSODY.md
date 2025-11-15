# Prosody Feature Extraction Module

## Overview

The prosody module (`transcription/prosody.py`) provides comprehensive extraction and analysis of prosodic features from audio segments. Prosody refers to the rhythm, stress, and intonation patterns in speech that convey meaning beyond the literal words.

## Features

### 1. Pitch Analysis
- **Mean fundamental frequency (F0)** in Hz
- **Standard deviation** to measure pitch variation
- **Pitch contour** classification (rising/falling/flat)
- **Min/max pitch** values
- Uses Praat's autocorrelation method via Parselmouth

### 2. Energy/Intensity Analysis
- **RMS (Root Mean Square)** energy levels
- **dB RMS** for standardized loudness measurements
- **Energy variation** (coefficient of variation)
- Frame-based analysis using librosa

### 3. Speech Rate Analysis
- **Syllables per second** estimation
- **Words per second** calculation
- Heuristic syllable counting based on vowel patterns

### 4. Pause Detection
- **Pause count** in segment
- **Longest pause** duration in milliseconds
- **Total pause duration**
- **Pause density** (pauses per second)
- Configurable silence threshold and minimum duration

### 5. Categorical Mapping
All numeric features are mapped to 5-level categories:
- `very_low` / `very_quiet` / `very_slow` / `very_sparse`
- `low` / `quiet` / `slow` / `sparse`
- `neutral` / `normal` / `normal` / `moderate`
- `high` / `loud` / `fast` / `frequent`
- `very_high` / `very_loud` / `very_fast` / `very_frequent`

### 6. Speaker Baseline Normalization
Features can be normalized relative to a speaker's baseline statistics:
- Compute baseline from multiple segments
- Z-score normalization when std available
- Ratio-based normalization as fallback
- Returns speaker-relative categories

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- `numpy>=1.24.0`
- `praat-parselmouth>=0.4.0` - For pitch extraction
- `librosa>=0.10.0` - For energy and audio processing

## Usage

### Basic Usage

```python
import numpy as np
from transcription.prosody import extract_prosody

# Your audio data
audio = np.array([...])  # Mono audio as float32
sr = 16000  # Sample rate in Hz
text = "This is the transcribed text."

# Extract prosody features
features = extract_prosody(audio, sr, text)

print(features)
```

### Output Format

The output is a structured dictionary matching this JSON schema:

```json
{
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
    "level": "fast",
    "syllables_per_sec": 6.3,
    "words_per_sec": 3.1
  },
  "pauses": {
    "count": 2,
    "longest_ms": 320,
    "density": "sparse"
  }
}
```

### Speaker Baseline Normalization

```python
from transcription.prosody import extract_prosody, compute_speaker_baseline

# Collect multiple segments from the same speaker
segments = [
    {'audio': audio1, 'sr': 16000, 'text': "First segment."},
    {'audio': audio2, 'sr': 16000, 'text': "Second segment."},
    # ... more segments
]

# Compute baseline statistics
baseline = compute_speaker_baseline(segments)
# Returns: {'pitch_median': 180.0, 'pitch_std': 28.5, ...}

# Extract features normalized to this speaker
features = extract_prosody(audio_new, 16000, text_new, speaker_baseline=baseline)

# Now features['pitch']['level'] is relative to this speaker's typical pitch
```

### Advanced Configuration

```python
from transcription.prosody import (
    extract_prosody,
    PITCH_THRESHOLDS,
    ENERGY_THRESHOLDS,
    RATE_THRESHOLDS
)

# Customize thresholds for your domain
PITCH_THRESHOLDS['high'] = 300  # Adjust high pitch threshold

# Extract with custom parameters
features = extract_prosody(
    audio,
    sr,
    text,
    speaker_baseline=baseline,
    start_time=0.0,  # For logging
    end_time=2.5     # For logging
)
```

## API Reference

### Main Functions

#### `extract_prosody(audio, sr, text, speaker_baseline=None, start_time=0.0, end_time=None)`

Main entry point for prosody extraction.

**Parameters:**
- `audio` (np.ndarray): Mono audio signal as float32
- `sr` (int): Sample rate in Hz
- `text` (str): Transcribed text for this segment
- `speaker_baseline` (dict, optional): Speaker baseline statistics
- `start_time` (float, optional): Start time for logging
- `end_time` (float, optional): End time for logging

**Returns:**
- `dict`: Structured prosody features

#### `compute_speaker_baseline(segments_data)`

Compute baseline statistics across multiple segments.

**Parameters:**
- `segments_data` (List[dict]): List of dicts with `audio`, `sr`, `text`

**Returns:**
- `dict`: Baseline statistics (median and std for each feature)

### Helper Functions

#### `extract_pitch_features(audio, sr)`
Extract pitch-related features using Parselmouth.

#### `extract_energy_features(audio, sr)`
Extract energy/intensity features using librosa.

#### `extract_speech_rate(audio, sr, text, duration)`
Extract speech rate features.

#### `extract_pause_features(audio, sr, duration)`
Extract pause-related features.

#### `count_syllables(text)`
Estimate syllable count using heuristic.

#### `detect_pauses(audio, sr, silence_threshold=-40, min_pause_duration=0.15)`
Detect pauses (silence regions) in audio.

#### `categorize_value(value, thresholds)`
Map numeric value to categorical level.

#### `normalize_to_baseline(value, baseline_median, baseline_std=None)`
Normalize value relative to speaker baseline.

## Threshold Constants

You can customize these thresholds for your specific use case:

```python
PITCH_THRESHOLDS = {
    'very_low': 80,    # Hz
    'low': 120,
    'neutral': 180,
    'high': 250,
    'very_high': float('inf')
}

ENERGY_THRESHOLDS = {
    'very_quiet': -35,  # dB RMS
    'quiet': -25,
    'normal': -15,
    'loud': -8,
    'very_loud': float('inf')
}

RATE_THRESHOLDS = {
    'very_slow': 3.0,   # syllables per second
    'slow': 4.5,
    'normal': 5.5,
    'fast': 7.0,
    'very_fast': float('inf')
}

PAUSE_DENSITY_THRESHOLDS = {
    'very_sparse': 0.5,  # pauses per second
    'sparse': 1.0,
    'moderate': 1.5,
    'frequent': 2.5,
    'very_frequent': float('inf')
}
```

## Error Handling

The module is designed to handle errors gracefully:

1. **Missing Dependencies**: Falls back to numpy-based calculations if librosa/parselmouth unavailable
2. **Invalid Audio**: Returns default structure with `None` values
3. **Extraction Failures**: Logs errors and returns partial results
4. **Empty Text**: Handles gracefully with zero rates

All functions return valid structures even on failure, ensuring pipeline robustness.

## Examples

See `/examples/prosody_demo.py` for comprehensive demonstrations:

```bash
cd /home/steven/code/Python/slower-whisper
python3 examples/prosody_demo.py
```

The demo includes:
1. Basic prosody extraction
2. Speaker baseline normalization
3. Different speech styles (calm, excited, questions)
4. JSON schema compliance

## Testing

Run the test suite:

```bash
python3 -m pytest tests/test_prosody.py -v
```

Tests cover:
- Syllable counting
- Categorical value mapping
- Baseline normalization
- Basic extraction
- Edge cases (empty audio, silence, noise)
- Speaker baseline computation

## Integration with Transcription Pipeline

### Example Integration

```python
from transcription.pipeline import TranscriptionPipeline
from transcription.prosody import extract_prosody, compute_speaker_baseline
import soundfile as sf

# Load audio
audio, sr = sf.read("recording.wav")

# Run transcription
pipeline = TranscriptionPipeline()
transcript = pipeline.transcribe(audio, sr)

# Extract prosody for each segment
for segment in transcript.segments:
    segment_audio = audio[int(segment.start * sr):int(segment.end * sr)]

    prosody = extract_prosody(
        segment_audio,
        sr,
        segment.text
    )

    # Attach to segment (requires extending Segment model)
    segment.prosody = prosody

# Save enriched transcript
# ...
```

### With Speaker Diarization

```python
# Group segments by speaker
speakers = {}
for segment in transcript.segments:
    speaker = segment.speaker or "unknown"
    if speaker not in speakers:
        speakers[speaker] = []
    speakers[speaker].append(segment)

# Compute baseline per speaker
baselines = {}
for speaker, segments in speakers.items():
    segment_data = [
        {
            'audio': audio[int(s.start * sr):int(s.end * sr)],
            'sr': sr,
            'text': s.text
        }
        for s in segments
    ]
    baselines[speaker] = compute_speaker_baseline(segment_data)

# Extract normalized features
for segment in transcript.segments:
    speaker = segment.speaker or "unknown"
    segment_audio = audio[int(segment.start * sr):int(segment.end * sr)]

    prosody = extract_prosody(
        segment_audio,
        sr,
        segment.text,
        speaker_baseline=baselines.get(speaker)
    )

    segment.prosody = prosody
```

## Performance Considerations

- **Pitch extraction**: Most computationally expensive (Praat autocorrelation)
- **Processing time**: ~50-200ms per segment on modern hardware
- **Memory**: Minimal - processes one segment at a time
- **Batch processing**: Can parallelize across segments

## Limitations

1. **Syllable counting**: Heuristic-based, not 100% accurate
2. **Pitch range**: Optimized for human speech (75-600 Hz)
3. **Language**: English-optimized (syllable counting)
4. **Audio quality**: Assumes reasonable SNR (>10 dB)
5. **Segment length**: Works best with 0.5-10 second segments

## Future Enhancements

- [ ] Voice quality features (jitter, shimmer, HNR)
- [ ] Formant analysis (F1, F2, F3)
- [ ] Speaking style classification
- [ ] Emotion detection from prosody
- [ ] Multi-language syllable counting
- [ ] GPU acceleration for batch processing
- [ ] Integration with advanced NLP for better syllable counting

## References

- **Parselmouth**: Praat acoustic analysis in Python
- **Librosa**: Audio feature extraction
- **Praat**: Speech analysis software (Boersma & Weenink)
- Prosodic feature research papers and phonetics literature

## License

Part of the slower-whisper project. See main LICENSE file.

## Support

For issues or questions:
1. Check the examples and tests
2. Review this documentation
3. Open an issue in the repository
4. Check dependencies are installed correctly
