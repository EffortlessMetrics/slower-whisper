# Examples Directory Index

This directory contains comprehensive examples demonstrating the complete audio transcription and analysis workflow.

## New Scripts (Created)

### 1. complete_workflow.py (625 lines, 24KB)
**Purpose**: Complete end-to-end analysis pipeline with multiple output formats

**Key Class**: `TranscriptAnalyzer`

**Core Methods**:
- `analyze_audio_characteristics()` - Get statistics on all audio features
- `find_excited_moments()` - High-energy segments
- `find_calm_moments()` - Low-energy segments
- `find_emotional_segments()` - Strong emotional indicators
- `find_hesitant_segments()` - Hesitation/pause patterns
- `create_annotated_transcript()` - Human-readable with annotations
- `export_to_csv()` - Spreadsheet export

**Best For**: Quick analysis, multiple output formats, end-to-end workflows

**Example Usage**:
```bash
python examples/complete_workflow.py enriched.json --output-csv data.csv
```

---

### 2. query_audio_features.py (667 lines, 23KB)
**Purpose**: Statistical query interface for detailed analysis

**Key Class**: `AudioFeatureQuery`

**Core Methods**:
- `get_summary_statistics()` - Detailed stats with percentiles
- `find_excited_moments()` - Percentile-based excited segments
- `find_calm_moments()` - Percentile-based calm segments
- `find_emotional_segments()` - Emotion-based queries
- `find_hesitant_segments()` - Hesitation patterns
- `find_high_pitch_moments()` - Pitch-specific queries
- `find_low_pitch_moments()` - Pitch-specific queries
- `get_emotion_distribution()` - Emotion statistics
- `get_pitch_contour_distribution()` - Intonation patterns
- `analyze_temporal_trends()` - Time-based trend analysis

**Best For**: Statistical analysis, percentile-based queries, trend analysis

**Example Usage**:
```bash
python examples/query_audio_features.py enriched.json --all
```

---

## Documentation (Created)

### 3. README_EXAMPLES.md (361 lines, 12KB)
Complete documentation covering:
- Overview of both scripts
- Detailed method descriptions
- Command-line usage
- Audio features explanation
- Data format specifications
- Integration guidelines

**Start here for**: Comprehensive reference

---

### 4. QUICK_START.md (307 lines, 9.2KB)
Quick reference guide with:
- Short code examples
- Common use cases
- Command-line snippets
- Expected output formats
- Tips and tricks

**Start here for**: Getting started quickly

---

### 5. INDEX.md (This file)
Navigation guide for all examples and documentation.

---

## Existing Scripts

### 6. prosody_demo.py (249 lines)
Comprehensive demonstration of prosody feature extraction capabilities.

### 7. prosody_quickstart.py (70 lines)
Quick introduction to prosody features.

### 8. emotion_integration.py (201 lines)
Shows how to enrich transcripts with emotion analysis.

### 9. word_timestamps_example.py (v1.8.0+)
**Purpose**: Demonstrates word-level timestamp extraction for precise timing

**Key Features**:
- Enable `word_timestamps=True` in TranscriptionConfig
- Access per-word timing (start, end) and confidence scores
- Generate word-level SRT subtitles (karaoke-style)
- Filter low-confidence words for quality review
- Search for specific words with timestamps
- Analyze speaking rate and word duration statistics

**Best For**: Subtitle generation, forced alignment, word-level search, quality assurance

**Example Usage**:
```bash
python examples/word_timestamps_example.py interview.wav ./output
python examples/word_timestamps_example.py podcast.mp3 --device cpu --search hello
```

---

## Quick Navigation

### I want to...

**Get started quickly**
1. Read: `QUICK_START.md` (this directory)
2. Run: `python complete_workflow.py enriched.json`

**Understand all features**
1. Read: `README_EXAMPLES.md` (this directory)
2. Read: `../EXAMPLE_SCRIPTS_SUMMARY.md` (detailed comparison)

**Do statistical analysis**
1. Run: `python query_audio_features.py enriched.json --summary`
2. Customize: Adjust percentile thresholds in code

**Export data**
1. Run: `python complete_workflow.py enriched.json --output-csv data.csv`
2. Import: Open CSV in Excel/Sheets/Pandas

**Find excited moments**
```bash
# Using complete_workflow
python complete_workflow.py enriched.json --output-annotated out.txt

# Using query_audio_features
python query_audio_features.py enriched.json --excited
```

**Find calm moments**
```bash
python query_audio_features.py enriched.json --calm
```

**Find emotional content**
```bash
python query_audio_features.py enriched.json --emotional
```

**Find hesitations**
```bash
python query_audio_features.py enriched.json --hesitant
```

**Get detailed statistics**
```bash
python query_audio_features.py enriched.json --summary
```

**Get word-level timestamps**
```bash
# Transcribe with word timestamps enabled
python word_timestamps_example.py audio.wav ./output

# Generate word-level SRT for karaoke subtitles
# (output saved to output/audio_word_level.srt)
```

---

## File Organization

```
examples/
├── INDEX.md                          (this file)
├── README_EXAMPLES.md                (full documentation)
├── QUICK_START.md                    (quick reference)
├── complete_workflow.py              (625 lines)
├── query_audio_features.py           (667 lines)
├── word_timestamps_example.py        (v1.8.0+ word-level timing)
├── prosody_demo.py                   (existing)
├── prosody_quickstart.py             (existing)
├── emotion_integration.py            (existing)
└── __pycache__/

../
├── EXAMPLE_SCRIPTS_SUMMARY.md        (detailed feature comparison)
└── (other project files)
```

---

## Feature Comparison

| Feature | complete_workflow.py | query_audio_features.py |
|---------|---------------------|------------------------|
| **Simple API** | Yes | Yes |
| **Multiple Outputs** | CSV, Text, Annotated | Console only |
| **Percentile Analysis** | No | Yes |
| **Summary Statistics** | Basic | Full (with stdev, percentiles) |
| **Excited Moments** | Yes | Yes (percentile-based) |
| **Calm Moments** | Yes | Yes (percentile-based) |
| **Emotional Segments** | Yes | Yes |
| **Hesitant Segments** | Yes | Yes |
| **Emotion Distribution** | No | Yes |
| **Pitch Contour Analysis** | No | Yes |
| **Temporal Trends** | No | Yes |
| **High/Low Pitch Queries** | No | Yes |
| **Lines of Code** | 625 | 667 |

---

## Integration Examples

### Example 1: Batch Processing All Transcripts

```python
from pathlib import Path
from examples.complete_workflow import TranscriptAnalyzer

for json_file in Path("output/transcripts").glob("*.json"):
    analyzer = TranscriptAnalyzer(json_file)
    csv_path = json_file.parent.parent / "analysis" / f"{json_file.stem}.csv"
    analyzer.export_to_csv(csv_path)
    print(f"Exported {json_file.name}")
```

### Example 2: Find High-Energy Moments for Editing

```python
from pathlib import Path
from examples.complete_workflow import TranscriptAnalyzer

analyzer = TranscriptAnalyzer(Path("enriched.json"))
excited = analyzer.find_excited_moments()

with open("edit_points.txt", "w") as f:
    for seg in excited:
        f.write(f"{seg.start:.2f} - {seg.end:.2f}: {seg.text}\n")
```

### Example 3: Statistical Analysis

```python
from pathlib import Path
from examples.query_audio_features import AudioFeatureQuery

query = AudioFeatureQuery(Path("enriched.json"))
stats = query.get_summary_statistics()

print(f"Pitch: {stats['pitch']['mean']:.1f} +/- {stats['pitch']['stdev']:.1f} Hz")
print(f"Emotion: {query.get_emotion_distribution()}")
```

---

## Audio Features Explained

### Prosody Features

**Pitch**
- `level`: low, medium, or high (relative to speaker baseline)
- `mean_hz`: Average fundamental frequency in Hertz
- `contour`: rising, falling, or flat (intonation pattern)

**Energy**
- `level`: low, medium, or high (loudness)
- `db_rms`: RMS energy in decibels

**Speech Rate**
- `level`: slow, normal, or fast
- `syllables_per_sec`: Actual speech rate

**Pauses**
- `count`: Number of silent intervals
- `density`: Pauses per second

### Emotion Features

**Categorical**
- `primary`: Emotion category (joy, sadness, anger, fear, surprise, neutral, etc.)
- `confidence`: Probability (0-1)

**Dimensional**
- `valence`: Positive (>0.5) to negative (<0.5)
- `arousal`: High energy (>0.5) to low energy (<0.5)

---

## Common Use Cases

### 1. Content Analysis
Find exciting, calm, and emotional moments for content marking.

### 2. Speaker Coaching
Identify hesitation patterns and areas for improvement.

### 3. Video Editing
Export high-energy segments for highlight reels.

### 4. Research
Export full data to CSV for statistical analysis.

### 5. Quality Assessment
Check emotion distribution and energy levels across content.

---

## Getting Help

1. **Quick start?** → Read `QUICK_START.md`
2. **Full reference?** → Read `README_EXAMPLES.md`
3. **Feature details?** → Read `../EXAMPLE_SCRIPTS_SUMMARY.md`
4. **Source code?** → Read the `.py` files (well-documented)

---

## Output Formats

### Console Reports
- Text-based summaries
- Formatted statistics
- Example segments from each category

### CSV Export
- All segment data
- 18+ columns of features
- Ready for spreadsheet/data analysis

### Annotated Transcripts
- Human-readable format
- Inline audio annotations
- Perfect for review and editing

---

## Command-Line Quick Reference

```bash
# Complete workflow - full analysis
python examples/complete_workflow.py enriched.json

# With outputs
python examples/complete_workflow.py enriched.json \
  --audio audio.wav \
  --output-annotated transcript.txt \
  --output-csv data.csv

# Query audio features - summary
python examples/query_audio_features.py enriched.json --summary

# Find specific patterns
python examples/query_audio_features.py enriched.json --excited
python examples/query_audio_features.py enriched.json --calm
python examples/query_audio_features.py enriched.json --emotional
python examples/query_audio_features.py enriched.json --hesitant

# Run all analyses
python examples/query_audio_features.py enriched.json --all
```

---

## File Sizes

| File | Type | Size | Lines |
|------|------|------|-------|
| complete_workflow.py | Script | 24 KB | 625 |
| query_audio_features.py | Script | 23 KB | 667 |
| README_EXAMPLES.md | Docs | 12 KB | 361 |
| QUICK_START.md | Docs | 9.2 KB | 307 |
| EXAMPLE_SCRIPTS_SUMMARY.md | Docs | 16 KB | 500+ |

**Total**: 88+ KB of code and documentation

---

## Next Steps

1. Choose your use case from "I want to..." section above
2. Run the suggested command
3. Consult the relevant documentation
4. Customize thresholds and queries as needed

---

## Technical Details

- **Python Version**: 3.7+
- **Dependencies**: Standard library only (+ project modules)
- **Input**: Enriched JSON transcripts
- **Output**: CSV, TXT, console, programmatic access

---

## Summary

Two comprehensive example scripts demonstrating:
- Complete transcription-to-analysis workflow
- Statistical query and analysis capabilities
- Multiple output formats (CSV, text, annotated, console)
- Support for all audio features (prosody, emotion)
- Full command-line and programmatic interfaces
- Extensive documentation and examples

Ready to analyze your transcripts!
