# Quick Start Guide

This document provides quick examples for getting started with the example scripts.

## Two Main Scripts

1. **complete_workflow.py** - Comprehensive analysis with CSV export
2. **query_audio_features.py** - Statistical queries and pattern finding

## Installation

All dependencies are already part of the project. Just ensure you have an enriched JSON transcript.

## Complete Workflow Examples

### Example 1: Load and Analyze

```python
from pathlib import Path
from examples.complete_workflow import TranscriptAnalyzer

# Load transcript
analyzer = TranscriptAnalyzer(Path("enriched.json"))

# Get statistics
stats = analyzer.analyze_audio_characteristics()
print(f"Pitch: {stats['pitch']['min']:.1f}-{stats['pitch']['max']:.1f} Hz")
print(f"Energy: {stats['energy']['min']:.1f}-{stats['energy']['max']:.1f} dB")

# Find patterns
excited = analyzer.find_excited_moments()
calm = analyzer.find_calm_moments()
emotional = analyzer.find_emotional_segments()
hesitant = analyzer.find_hesitant_segments()

print(f"Excited: {len(excited)}, Calm: {len(calm)}, Emotional: {len(emotional)}, Hesitant: {len(hesitant)}")
```

### Example 2: Export Data

```python
# Generate annotated transcript (human-readable with annotations)
analyzer.create_annotated_transcript(Path("transcript_annotated.txt"))

# Export to CSV (for spreadsheet analysis)
analyzer.export_to_csv(Path("transcript_data.csv"))
```

### Example 3: Command Line

```bash
# Full analysis and reports
python examples/complete_workflow.py enriched.json \
  --audio audio.wav \
  --output-annotated annotated.txt \
  --output-csv data.csv
```

## Query Audio Features Examples

### Example 1: Get Statistics

```python
from pathlib import Path
from examples.query_audio_features import AudioFeatureQuery

query = AudioFeatureQuery(Path("enriched.json"))

# Get detailed statistics
stats = query.get_summary_statistics()
print(f"Pitch: {stats['pitch']['mean']:.1f} +/- {stats['pitch']['stdev']:.1f} Hz")
print(f"Range: {stats['pitch']['q25']:.1f} - {stats['pitch']['q75']:.1f} Hz (25-75%)")
```

### Example 2: Find Patterns

```python
# Find excited moments (percentile-based)
excited = query.find_excited_moments(
    high_energy_percentile=0.75,
    high_pitch_percentile=0.75,
    fast_rate_percentile=0.75
)

# Find calm moments (percentile-based)
calm = query.find_calm_moments(
    low_energy_percentile=0.25,
    low_pitch_percentile=0.25,
    slow_rate_percentile=0.25
)

# Find emotional and hesitant segments
emotional = query.find_emotional_segments()
hesitant = query.find_hesitant_segments(min_pause_count=2)

print(f"Excited: {len(excited)}, Calm: {len(calm)}")
print(f"Emotional: {len(emotional)}, Hesitant: {len(hesitant)}")
```

### Example 3: Distributions

```python
# Get emotion distribution
emotions = query.get_emotion_distribution()
for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {count}")

# Get pitch contour distribution
contours = query.get_pitch_contour_distribution()
print(f"Pitch contours: {contours}")
```

### Example 4: Command Line

```bash
# Get comprehensive summary
python examples/query_audio_features.py enriched.json --summary

# Run all analyses
python examples/query_audio_features.py enriched.json --all

# Find specific patterns
python examples/query_audio_features.py enriched.json --excited
python examples/query_audio_features.py enriched.json --calm
python examples/query_audio_features.py enriched.json --emotional
python examples/query_audio_features.py enriched.json --hesitant
```

## Output Examples

### Console Report (Summary)

```
================================================================================
AUDIO FEATURE SUMMARY REPORT
================================================================================

File: meeting.json
Total segments: 145

--- Pitch ---
  mean            :     165.2 Hz
  stdev           :      38.5 Hz
  range           :      80.5 - 250.3 Hz

--- Emotion Distribution ---
  neutral        : ████████████        85 ( 58.6%)
  joy            : ███████              24 ( 16.6%)
  confidence     : ████                 14 (  9.7%)
```

### Annotated Transcript

```
[  0.00s -   2.50s]
Hello, how are you?
  >> Pitch: high (rising) | Energy: high (-15.2 dB) | Rate: fast (4.2 syl/sec) | Emotion: joy (0.85)

[  2.50s -   5.10s]
I'm doing well, thanks!
  >> Pitch: medium (flat) | Energy: medium (-18.5 dB) | Rate: normal (3.8 syl/sec)
```

### CSV Data

```csv
segment_id,start_time,end_time,text,pitch_level,pitch_mean_hz,energy_level,emotion_category
0,0.000,2.500,"Hello, how are you?",high,185.5,high,joy
1,2.500,5.100,"I'm doing well, thanks!",medium,165.2,medium,joy
```

## Common Use Cases

### 1. Find High-Energy Moments for Editing

```python
analyzer = TranscriptAnalyzer("enriched.json")
excited = analyzer.find_excited_moments()

# Export timestamps for video editing
with open("edit_points.txt", "w") as f:
    for seg in excited:
        f.write(f"{seg.start:.2f} - {seg.end:.2f}: {seg.text}\n")
```

### 2. Find Hesitations for Presentation Coaching

```python
analyzer = TranscriptAnalyzer("enriched.json")
hesitant = analyzer.find_hesitant_segments(min_pause_count=3)

for seg in hesitant:
    pauses = seg.audio_state['pauses']['count']
    print(f"Minute {seg.start/60:.0f}: {pauses} pauses - {seg.text}")
```

### 3. Emotion-Based Content Analysis

```python
query = AudioFeatureQuery("enriched.json")
emotions = query.get_emotion_distribution()
emotional = query.find_emotional_segments()

print(f"Emotional segments: {len(emotional)}/{len(query.transcript.segments)}")
for emotion, count in emotions.items():
    pct = count / len(query.transcript.segments) * 100
    print(f"  {emotion}: {pct:.1f}%")
```

### 4. Speaker Baseline Comparison

```python
query = AudioFeatureQuery("enriched.json")
stats = query.get_summary_statistics()

# Find unusually high-pitched moments
threshold = stats['pitch']['mean'] + stats['pitch']['stdev']
high_pitch = query.find_high_pitch_moments(percentile=0.9)
print(f"Unusual pitch: {len(high_pitch)} segments")
```

### 5. Temporal Trend Analysis

```python
query = AudioFeatureQuery("enriched.json")
trends = query.analyze_temporal_trends(window_size=10)

# Check if energy increases over time
energy_trend = trends['energy_trend']
if energy_trend[-1] > energy_trend[0]:
    print("Energy increases over time (getting more excited)")
else:
    print("Energy decreases over time (getting tired)")
```

## File Locations

- **complete_workflow.py**: `/home/steven/code/Python/slower-whisper/examples/complete_workflow.py` (625 lines)
- **query_audio_features.py**: `/home/steven/code/Python/slower-whisper/examples/query_audio_features.py` (667 lines)
- **README_EXAMPLES.md**: `/home/steven/code/Python/slower-whisper/examples/README_EXAMPLES.md` (Full documentation)
- **QUICK_START.md**: This file

## Key Classes and Methods

### TranscriptAnalyzer (complete_workflow.py)

```
__init__(json_path, audio_path=None)
├── analyze_audio_characteristics() -> Dict
├── find_excited_moments() -> List[Segment]
├── find_calm_moments() -> List[Segment]
├── find_emotional_segments() -> List[Segment]
├── find_hesitant_segments() -> List[Segment]
├── create_annotated_transcript(output_path) -> None
└── export_to_csv(output_path) -> None
```

### AudioFeatureQuery (query_audio_features.py)

```
__init__(json_path)
├── get_summary_statistics() -> Dict
├── find_excited_moments(...) -> List[Tuple]
├── find_calm_moments(...) -> List[Tuple]
├── find_emotional_segments(...) -> List[Tuple]
├── find_hesitant_segments(...) -> List[Tuple]
├── find_high_pitch_moments(percentile) -> List[Segment]
├── find_low_pitch_moments(percentile) -> List[Segment]
├── get_emotion_distribution() -> Dict
├── get_pitch_contour_distribution() -> Dict
└── analyze_temporal_trends(window_size) -> Dict
```

## Tips & Tricks

1. **Memory Usage**: Process large transcripts (>10K segments) in batches if memory is limited
2. **Percentile Tuning**: Adjust percentile thresholds for different sensitivity (0.5 = median, 0.75 = upper quartile)
3. **Batch Processing**: Loop over multiple JSON files to analyze full datasets
4. **CSV Integration**: Open CSV output in Excel/Sheets for further analysis
5. **Custom Queries**: Extend classes to add domain-specific queries

## Next Steps

1. Read `README_EXAMPLES.md` for comprehensive documentation
2. Read `EXAMPLE_SCRIPTS_SUMMARY.md` for detailed feature descriptions
3. Run the command-line examples above
4. Integrate into your own analysis pipelines
5. Customize thresholds and queries for your use case

## Troubleshooting

**"Missing audio features"**: Ensure transcript is enriched with audio features (audio_state field)

**"Empty results"**: Check percentile thresholds - try 0.5 (median) instead of 0.75

**"Memory error on large files"**: Process transcripts in smaller batches using loops

**"No pauses found"**: Verify audio features include pause data, adjust min_pause_count threshold

## More Information

See `README_EXAMPLES.md` in the examples directory for:
- Complete method signatures
- Detailed parameter descriptions
- JSON schema examples
- Integration patterns
- Data format specifications
