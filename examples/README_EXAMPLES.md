# Complete Workflow Examples

This directory contains comprehensive examples demonstrating the complete pipeline for transcribing, enriching, and analyzing audio files.

## Overview

The example scripts show how to:

1. **Transcribe** audio files using the transcription pipeline
2. **Enrich** transcripts with audio features (prosody, emotion, etc.)
3. **Query** enriched transcripts to find specific patterns
4. **Generate reports** in multiple formats (text, CSV, JSON)

## Scripts

### 1. `complete_workflow.py`

A comprehensive example showing the full analysis pipeline from raw transcripts to enriched reports.

**Key Classes:**
- `TranscriptAnalyzer`: Main analysis class for enriched transcripts

**Key Methods:**
- `analyze_audio_characteristics()`: Get statistics on pitch, energy, speech rate, pauses
- `find_excited_moments()`: Locate high-energy, high-pitch, fast-speech segments
- `find_calm_moments()`: Locate low-energy, low-pitch, slow-speech segments
- `find_emotional_segments()`: Find segments with strong emotional indicators
- `find_hesitant_segments()`: Identify segments with hesitation patterns (pauses, stuttering)
- `create_annotated_transcript()`: Generate a readable transcript with inline audio annotations
- `export_to_csv()`: Export all segment data to CSV spreadsheet format

**Usage:**

```bash
# Basic analysis - print comprehensive report
python examples/complete_workflow.py /path/to/enriched.json

# With audio file for full feature extraction
python examples/complete_workflow.py /path/to/enriched.json --audio /path/to/audio.wav

# Generate annotated transcript
python examples/complete_workflow.py /path/to/enriched.json \
  --output-annotated enriched_with_annotations.txt

# Export to CSV for further analysis
python examples/complete_workflow.py /path/to/enriched.json \
  --output-csv transcript_data.csv

# All options
python examples/complete_workflow.py /path/to/enriched.json \
  --audio /path/to/audio.wav \
  --output-annotated out.txt \
  --output-csv out.csv
```

**Example Output:**

```
================================================================================
TRANSCRIPT ANALYSIS REPORT
================================================================================

File: meeting.wav
Language: en

--- Audio Characteristics ---
Total segments: 145
Total duration: 892.5s
Segments with audio features: 145

Pitch Analysis:
  Range: 80.5 - 250.3 Hz
  Mean: 165.2 Hz
  Distribution: {'high': 42, 'medium': 68, 'low': 35}
  Contours: {'rising': 45, 'falling': 52, 'flat': 48}

Energy Analysis:
  Range: -32.1 - -8.5 dB
  Mean: -18.3 dB
  Distribution: {'high': 51, 'medium': 62, 'low': 32}

Speech Rate Analysis:
  Range: 2.1 - 5.8 syl/sec
  Mean: 3.9 syl/sec
  Distribution: {'fast': 38, 'normal': 71, 'slow': 36}

--- Excited Moments (23) ---
  [45.2s] Yeah absolutely! This is a fantastic opportunity...
  [102.5s] I'm really excited about this direction...
  ... and 21 more

--- Calm Moments (18) ---
  [234.1s] Let me think about that for a moment...
  [456.8s] So the key insight is...
  ... and 16 more

--- Emotionally Expressive Segments (45) ---
  [78.3s] joy: That's wonderful news!...
  [156.2s] surprise: Wait, really?...
  ... and 43 more

--- Hesitant Segments (12) ---
  [201.5s] (4 pauses) Well, I think that... um... might work...
  ... and 11 more
```

### 2. `query_audio_features.py`

Specialized query utilities for finding specific patterns in enriched transcripts.

**Key Classes:**
- `AudioFeatureQuery`: Query interface for audio features

**Key Methods:**
- `get_summary_statistics()`: Comprehensive statistics (mean, stdev, percentiles, etc.)
- `find_excited_moments()`: Find high-energy segments (percentile-based)
- `find_calm_moments()`: Find low-energy segments (percentile-based)
- `find_emotional_segments()`: Find emotionally significant segments
- `find_hesitant_segments()`: Find segments with hesitation patterns
- `find_high_pitch_moments()`: All high-pitch segments
- `find_low_pitch_moments()`: All low-pitch segments
- `get_emotion_distribution()`: Distribution of categorical emotions
- `get_pitch_contour_distribution()`: Distribution of pitch patterns (rising/falling/flat)
- `analyze_temporal_trends()`: Trend analysis over time using sliding window

**Usage:**

```bash
# Print comprehensive summary
python examples/query_audio_features.py /path/to/enriched.json --summary

# Find excited moments
python examples/query_audio_features.py /path/to/enriched.json --excited

# Find calm moments
python examples/query_audio_features.py /path/to/enriched.json --calm

# Find emotional segments
python examples/query_audio_features.py /path/to/enriched.json --emotional

# Find hesitant segments
python examples/query_audio_features.py /path/to/enriched.json --hesitant

# Run all analyses
python examples/query_audio_features.py /path/to/enriched.json --all
```

**Example Output:**

```
================================================================================
AUDIO FEATURE SUMMARY REPORT
================================================================================

File: meeting.wav
Total segments: 145
Segments with audio features: 145

--- Pitch ---
  count           :     145.0 Hz
  mean            :     165.2 Hz
  stdev           :      38.5 Hz
  min             :      80.5 Hz
  max             :     250.3 Hz
  median          :     162.1 Hz
  q25             :     138.2 Hz
  q75             :     190.5 Hz

--- Energy ---
  count           :     145.0 dB
  mean            :     -18.3 dB
  stdev           :       6.2 dB
  min             :     -32.1 dB
  max             :      -8.5 dB

--- Speech Rate ---
  count           :     145.0 syl/sec
  mean            :       3.9 syl/sec
  stdev           :       0.9 syl/sec
  min             :       2.1 syl/sec
  max             :       5.8 syl/sec

--- Emotion Distribution ---
  neutral        : ████████████        85 ( 58.6%)
  joy            : ███████              24 ( 16.6%)
  confidence     : ████                 14 ( 9.7%)
  surprise       : ██                    8 ( 5.5%)
  concern        : █                     8 ( 5.5%)

--- Pitch Contour Distribution ---
  flat           : ████████             48 ( 33.1%)
  rising         : ████████             45 ( 31.0%)
  falling        : ███████              52 ( 35.9%)
```

## Complete Workflow Example

Here's a typical workflow:

```bash
#!/bin/bash

# 1. Transcribe audio files (creates JSON transcripts)
python -m transcription.cli /path/to/audio/

# 2. Enrich transcripts with audio features
# (This would be done by your transcription engine or separately)

# 3. Analyze the enriched transcript
python examples/complete_workflow.py output/transcripts/audio.json \
  --audio /path/to/audio/audio.wav \
  --output-annotated output/analysis/audio_annotated.txt \
  --output-csv output/analysis/audio_data.csv

# 4. Run specific queries
python examples/query_audio_features.py output/transcripts/audio.json --all
```

## Audio Features Explained

### Pitch
- **Level**: low, medium, high (relative to speaker baseline)
- **Mean Hz**: Average fundamental frequency
- **Contour**: rising (question), falling (statement), flat (monotone)

### Energy
- **Level**: low, medium, high (loudness)
- **dB RMS**: Energy in decibels (root mean square)

### Speech Rate
- **Level**: slow, normal, fast
- **Syllables/sec**: Actual speech rate

### Pauses
- **Count**: Number of silent intervals
- **Density**: Pauses per second

### Emotion (Dimensional)
- **Valence**: Positive (>0.5) or negative (<0.5)
- **Arousal**: High energy (>0.5) or low energy (<0.5)

### Emotion (Categorical)
- **Primary**: joy, sadness, anger, fear, surprise, neutral, etc.
- **Confidence**: Probability of the classification (0-1)

## Data Formats

### JSON Output (Enriched Transcript)
```json
{
  "schema_version": 2,
  "file": "audio.wav",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, how are you?",
      "audio_state": {
        "pitch": {
          "level": "high",
          "mean_hz": 185.5,
          "contour": "rising"
        },
        "energy": {
          "level": "high",
          "db_rms": -15.2
        },
        "rate": {
          "level": "fast",
          "syllables_per_sec": 4.2
        },
        "pauses": {
          "count": 0,
          "density": 0.0
        },
        "emotion": {
          "categorical": {
            "primary": "joy",
            "confidence": 0.85
          },
          "dimensional": {
            "valence": {"score": 0.75},
            "arousal": {"score": 0.68}
          }
        }
      }
    }
  ]
}
```

### CSV Output
Each row represents a segment with all extracted features:

```
segment_id,start_time,end_time,duration,text,pitch_level,pitch_mean_hz,...
0,0.000,2.500,2.500,"Hello, how are you?",high,185.5,rising,...
1,2.500,5.100,2.600,"I'm doing well, thanks!",medium,165.2,flat,...
```

### Annotated Text Output
```
# Annotated Transcript: audio.wav
# Language: en
# Generated from enriched transcript

[  0.00s -   2.50s]
Hello, how are you?
  >> Pitch: high (rising) | Energy: high (-15.2 dB) | Rate: fast (4.2 syl/sec) | Emotion: joy (0.85)

[  2.50s -   5.10s]
I'm doing well, thanks!
  >> Pitch: medium (flat) | Energy: medium (-18.5 dB) | Rate: normal (3.8 syl/sec) | Emotion: joy (0.72)
```

## Integration with Pipeline

These examples work with the existing pipeline:

1. **Transcription Pipeline** (`transcription.pipeline`): Creates initial JSON transcripts
2. **Audio Enrichment** (optional module): Adds audio features to segments
3. **Analysis Examples** (this directory): Queries and reports on enriched data

## Requirements

Standard library only, plus project dependencies:
- `pathlib`, `json`, `csv`, `logging`, `collections`, `statistics`
- Project modules: `transcription.models`, `transcription.writers`, `transcription.audio_utils`

## Tips for Usage

1. **Memory Efficiency**: Large transcripts are loaded into memory. For files >10,000 segments, consider processing in batches.

2. **Audio Features**: Queries work best with fully enriched transcripts. Missing features are gracefully skipped.

3. **Percentile Thresholds**: Adjust percentile values in `find_excited_moments()` and `find_calm_moments()` for different sensitivity levels.

4. **Custom Queries**: Extend `AudioFeatureQuery` class to add domain-specific queries.

5. **Report Generation**: Use the CSV export with spreadsheet tools like Excel, Google Sheets, or data analysis tools like Pandas.

## Examples Structure

```
examples/
├── README_EXAMPLES.md           (this file)
├── complete_workflow.py         (comprehensive analysis)
├── query_audio_features.py      (specialized queries)
├── prosody_demo.py              (prosody features demo)
├── prosody_quickstart.py        (quick start)
└── emotion_integration.py       (emotion analysis demo)
```

## See Also

- `transcription.pipeline`: Core transcription pipeline
- `transcription.models`: Data models (Transcript, Segment)
- `transcription.writers`: Output writers (JSON, TXT, SRT)
- `transcription.audio_utils`: Audio segment extraction
