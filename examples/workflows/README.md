# End-to-End Workflow Examples

This directory contains complete workflow examples for common audio transcription and analysis use cases. Each workflow demonstrates the full pipeline from raw audio input to enriched, analyzed output.

## Overview

All workflows follow a two-stage pipeline:

**Stage 1: Transcription**
- Normalize audio to 16 kHz mono WAV
- Transcribe using faster-whisper on GPU
- Generate JSON transcripts with segments

**Stage 2: Audio Enrichment** (optional but recommended)
- Extract prosodic features (pitch, energy, speech rate, pauses)
- Extract emotional features (valence, arousal, categorical emotions)
- Populate `audio_state` field in segments

## Available Workflows

### 1. Meeting Transcription and Analysis
**Use Case:** Zoom/Teams meeting recordings
**Script:** `meeting_transcription.py`
**Best For:** Business meetings, stand-ups, conference calls

**Features:**
- Searchable transcripts with timestamps
- Emotional markers for engagement tracking
- Speaker turn identification
- Key moment detection (decisions, action items, heated discussions)

**Typical Output:**
- Full transcript with emotional annotations
- Summary of engagement levels throughout meeting
- Highlighted moments of excitement, concern, or agreement
- CSV export for trend analysis

[→ See detailed documentation](#meeting-transcription-and-analysis)

---

### 2. Podcast Processing
**Use Case:** Multi-speaker podcast episodes
**Script:** `podcast_processing.py`
**Best For:** Podcast production, show notes generation, content repurposing

**Features:**
- Multi-speaker support (when combined with diarization)
- Enhanced show notes with emotional context
- Chapter/topic segmentation
- Quote extraction based on engagement markers

**Typical Output:**
- Formatted show notes with timestamps
- Highlight reel suggestions (exciting moments)
- Guest emotion tracking
- Social media quote suggestions

[→ See detailed documentation](#podcast-processing)

---

### 3. Interview Analysis
**Use Case:** Research interviews, job interviews, user testing
**Script:** `interview_analysis.py`
**Best For:** Qualitative research, UX research, HR interviews

**Features:**
- Emotional shift detection
- Hesitation and uncertainty markers
- Key moment identification
- Response pattern analysis

**Typical Output:**
- Interview timeline with emotional shifts
- Hesitation markers (pauses, speech rate changes)
- High-confidence vs. uncertain responses
- Emotional journey visualization data

[→ See detailed documentation](#interview-analysis)

---

### 4. Call Center QA
**Use Case:** Customer service call analysis
**Script:** `call_center_qa.py`
**Best For:** Quality assurance, agent training, customer sentiment tracking

**Features:**
- Frustration detection
- Satisfaction markers
- Call escalation predictors
- Agent performance metrics

**Typical Output:**
- Call quality scores
- Frustration/satisfaction timeline
- Agent empathy markers
- Training opportunity identification

[→ See detailed documentation](#call-center-qa)

---

### 5. Research Data Processing
**Use Case:** Batch processing many recordings
**Script:** `batch_research_processing.py`
**Best For:** Academic research, large-scale studies, corpus analysis

**Features:**
- Batch processing with progress tracking
- Consistent metadata extraction
- Aggregated statistics across recordings
- CSV export for statistical analysis

**Typical Output:**
- Individual enriched transcripts
- Aggregated CSV with all segments
- Cross-recording statistics
- Quality control reports

[→ See detailed documentation](#research-data-processing)

---

## Quick Start

### Prerequisites

```bash
# Install base dependencies
pip install faster-whisper>=1.0.0

# Install enrichment dependencies (recommended)
pip install -r requirements.txt
```

### Basic Usage Pattern

All workflows follow a similar pattern:

```bash
# 1. Place audio in raw_audio/
cp your_recording.mp3 raw_audio/

# 2. Run workflow script
python examples/workflows/<workflow_name>.py --audio raw_audio/your_recording.mp3

# Or use pre-configured settings
python examples/workflows/<workflow_name>.py --config examples/workflows/configs/<workflow>_config.json
```

### Using Configuration Files

Each workflow has a template configuration file in `configs/`:

```bash
# Copy template
cp examples/workflows/configs/meeting_template.json my_config.json

# Edit settings
nano my_config.json

# Run with config
python examples/workflows/meeting_transcription.py --config my_config.json
```

---

## Workflow Details

## Meeting Transcription and Analysis

### Overview
Transform meeting recordings into searchable, analyzed transcripts with emotional context.

### Input Requirements
- Audio format: MP3, WAV, M4A, or any ffmpeg-compatible format
- Typical length: 30-120 minutes
- Content: Business meetings, team calls, conference sessions

### Configuration

Template: `configs/meeting_template.json`

```json
{
  "transcription": {
    "model": "large-v3",
    "language": "en",
    "device": "cuda",
    "compute_type": "float16"
  },
  "enrichment": {
    "enable_prosody": true,
    "enable_emotion_dimensional": true,
    "enable_emotion_categorical": true,
    "device": "cuda"
  },
  "analysis": {
    "detect_key_moments": true,
    "engagement_tracking": true,
    "export_formats": ["txt", "csv", "annotated"]
  }
}
```

### Usage Example

```bash
# Basic usage
python examples/workflows/meeting_transcription.py \
  --audio raw_audio/team_meeting.mp3 \
  --output outputs/team_meeting

# With configuration
python examples/workflows/meeting_transcription.py \
  --config configs/meeting_config.json \
  --audio raw_audio/team_meeting.mp3

# Skip existing files (resume processing)
python examples/workflows/meeting_transcription.py \
  --audio raw_audio/team_meeting.mp3 \
  --skip-existing
```

### Output Files

```
outputs/team_meeting/
├── team_meeting.json              # Full enriched transcript
├── team_meeting.txt               # Plain text with timestamps
├── team_meeting_annotated.txt     # Text with emotion annotations
├── team_meeting.csv               # Spreadsheet-ready data
└── team_meeting_analysis.txt      # Summary report
```

### Example Output Snippet

```
[  12.50s -   18.30s]
I think we should definitely prioritize the customer feedback feature.
  >> Pitch: high (rising) | Energy: loud (-8.2 dB) | Rate: fast (6.2 syl/sec) | Emotion: confident (87%)

[  18.50s -   24.10s]
Yeah, I'm not sure... maybe we could look at the timeline first?
  >> Pitch: low (flat) | Energy: quiet (-18.5 dB) | Rate: slow (3.8 syl/sec) | Pauses: 3 | Emotion: uncertain (72%)
```

### Common Pitfalls

1. **Poor audio quality**: Use headset recordings when possible
2. **Background noise**: Pre-process with noise reduction if needed
3. **Multiple simultaneous speakers**: May result in overlapping segments
4. **Very long meetings**: Consider splitting into chunks for faster processing

### Performance Considerations

| Meeting Length | Processing Time* | GPU Memory | Storage |
|---------------|------------------|------------|---------|
| 30 min        | ~5-8 min         | ~4 GB      | ~2 MB   |
| 60 min        | ~10-15 min       | ~4 GB      | ~4 MB   |
| 120 min       | ~20-30 min       | ~4 GB      | ~8 MB   |

*On NVIDIA RTX 3080, with large-v3 model

---

## Podcast Processing

### Overview
Process podcast episodes with enhanced show notes, emotional context, and content highlights.

### Input Requirements
- Audio format: MP3, WAV, M4A
- Typical length: 30-180 minutes
- Content: Conversational podcasts, interviews, panel discussions

### Configuration

Template: `configs/podcast_template.json`

```json
{
  "transcription": {
    "model": "large-v3",
    "language": "auto",
    "device": "cuda"
  },
  "enrichment": {
    "enable_prosody": true,
    "enable_emotion_categorical": true,
    "track_engagement": true
  },
  "podcast_specific": {
    "generate_show_notes": true,
    "detect_highlights": true,
    "engagement_threshold": 0.7,
    "min_highlight_duration": 10.0,
    "chapter_markers": true
  }
}
```

### Usage Example

```bash
# Process podcast episode
python examples/workflows/podcast_processing.py \
  --audio raw_audio/episode_042.mp3 \
  --title "Episode 42: The Future of AI" \
  --output outputs/podcast_ep42

# Generate highlight reel timestamps
python examples/workflows/podcast_processing.py \
  --audio raw_audio/episode_042.mp3 \
  --highlights-only \
  --min-duration 15
```

### Output Files

```
outputs/podcast_ep42/
├── episode_042.json                # Full enriched transcript
├── episode_042_show_notes.md       # Formatted show notes
├── episode_042_highlights.txt      # Timestamp highlights
├── episode_042_quotes.txt          # Pull quote suggestions
└── episode_042_chapters.json       # Chapter markers
```

### Example Show Notes Output

```markdown
# Episode 42: The Future of AI

**Duration:** 62:34

## Highlights

### 🔥 [12:34] - Exciting Discussion on AGI Timeline
High energy segment with rapid exchange of ideas.
Notable emotion: excited, confident

### 💡 [28:15] - Key Insight on Model Training
Thoughtful, measured discussion with technical depth.
Notable emotion: analytical, focused

### 😊 [45:22] - Humorous Anecdote
Lighthearted moment with positive energy.
Notable emotion: happy, amused

## Full Timestamps

[00:00] Introduction and welcome
[02:15] Guest background and expertise
[12:34] Discussion on AGI development timeline
[28:15] Deep dive into model training techniques
...
```

### Common Pitfalls

1. **Music/intro segments**: May produce poor transcripts (trim before processing)
2. **Cross-talk**: Works best with turn-based conversation
3. **Sound effects**: Can interfere with emotion detection
4. **Ads/sponsorships**: Consider marking and excluding from analysis

### Performance Considerations

- **Recommendation**: Process overnight for multi-hour episodes
- **Optimization**: Use `medium` model for drafts, `large-v3` for final
- **Storage**: Plan for ~5-10 MB per hour of audio

---

## Interview Analysis

### Overview
Analyze research interviews, job interviews, or user testing sessions with emotional journey tracking.

### Input Requirements
- Audio format: Any ffmpeg-compatible format
- Typical length: 15-60 minutes
- Content: One-on-one or panel interviews

### Configuration

Template: `configs/interview_template.json`

```json
{
  "transcription": {
    "model": "large-v3",
    "language": "en",
    "vad_min_silence_ms": 500
  },
  "enrichment": {
    "enable_prosody": true,
    "enable_emotion_dimensional": true,
    "enable_emotion_categorical": true,
    "track_hesitation": true
  },
  "analysis": {
    "detect_emotional_shifts": true,
    "track_confidence_levels": true,
    "identify_key_moments": true,
    "export_timeline": true
  }
}
```

### Usage Example

```bash
# Analyze interview
python examples/workflows/interview_analysis.py \
  --audio raw_audio/user_interview_05.wav \
  --output outputs/interview_05 \
  --report

# Batch process multiple interviews
python examples/workflows/interview_analysis.py \
  --batch raw_audio/interviews/*.wav \
  --output outputs/interviews \
  --aggregate
```

### Output Files

```
outputs/interview_05/
├── user_interview_05.json          # Enriched transcript
├── user_interview_05_timeline.csv  # Emotional journey data
├── user_interview_05_analysis.txt  # Analysis report
├── user_interview_05_moments.json  # Key moments
└── user_interview_05_shifts.json   # Emotional shift events
```

### Example Analysis Report

```
INTERVIEW ANALYSIS REPORT
================================================================================

Interview: user_interview_05.wav
Duration: 42:15
Segments: 127

--- Emotional Journey ---

Overall Sentiment: Positive (valence: 0.62)
Engagement Level: Moderate-High (arousal: 0.58)

Emotional Shifts Detected: 8

  [05:23] Neutral → Frustrated
    Trigger: Discussion of current workflow pain points
    Confidence: 0.84

  [18:45] Frustrated → Excited
    Trigger: Introduction of new feature concept
    Confidence: 0.91

  [32:10] Excited → Uncertain
    Trigger: Questions about implementation timeline
    Confidence: 0.76

--- Confidence Indicators ---

High Confidence Segments: 45 (35%)
- Fast speech rate, low pauses, rising pitch contour

Uncertain/Hesitant Segments: 23 (18%)
- Slow speech rate, multiple pauses, flat pitch

--- Key Moments ---

🎯 [18:45] Peak Excitement
  "This would completely solve the problem we've been having!"
  Emotion: excited (0.91) | Energy: high | Rate: fast

⚠️  [32:10] Concern Expressed
  "I'm... not sure how that would work with our current setup..."
  Emotion: uncertain (0.76) | Pauses: 4 | Rate: slow

💡 [38:20] Insight Shared
  "Oh, I see! That makes a lot of sense now."
  Emotion: satisfied (0.87) | Energy: moderate-high
```

### Common Pitfalls

1. **Interview environment noise**: Use quiet rooms when possible
2. **Interviewer bias**: Long interviewer segments may skew analysis
3. **Cultural differences**: Emotion models may vary across cultures
4. **Technical jargon**: May affect transcription accuracy

### Performance Considerations

- **Typical processing**: 2-3x real-time with enrichment
- **Recommended**: Process same day for timely insights
- **Batch mode**: Can process 10-20 interviews overnight

---

## Call Center QA

### Overview
Analyze customer service calls for quality assurance, agent performance, and customer sentiment.

### Input Requirements
- Audio format: MP3, WAV (common call recording formats)
- Typical length: 2-30 minutes
- Content: Customer service interactions

### Configuration

Template: `configs/call_center_template.json`

```json
{
  "transcription": {
    "model": "large-v3",
    "language": "en",
    "vad_min_silence_ms": 400
  },
  "enrichment": {
    "enable_prosody": true,
    "enable_emotion_categorical": true,
    "enable_emotion_dimensional": true
  },
  "qa_metrics": {
    "detect_frustration": true,
    "detect_satisfaction": true,
    "track_escalation_risk": true,
    "measure_agent_empathy": true,
    "call_quality_score": true
  },
  "thresholds": {
    "frustration_arousal": 0.7,
    "frustration_valence": 0.3,
    "satisfaction_valence": 0.7,
    "empathy_indicators": ["calm", "reassuring", "understanding"]
  }
}
```

### Usage Example

```bash
# Analyze single call
python examples/workflows/call_center_qa.py \
  --audio raw_audio/call_12345.wav \
  --output outputs/qa/call_12345 \
  --agent-id "agent_042"

# Batch QA analysis
python examples/workflows/call_center_qa.py \
  --batch raw_audio/calls/*.wav \
  --output outputs/qa_batch \
  --generate-report \
  --aggregate-by-agent

# Training opportunity detection
python examples/workflows/call_center_qa.py \
  --audio raw_audio/call_12345.wav \
  --training-mode \
  --highlight-opportunities
```

### Output Files

```
outputs/qa/call_12345/
├── call_12345.json                 # Enriched transcript
├── call_12345_qa_score.json        # Quality metrics
├── call_12345_timeline.csv         # Sentiment timeline
├── call_12345_report.txt           # QA report
└── call_12345_training.txt         # Training opportunities
```

### Example QA Report

```
CALL CENTER QA REPORT
================================================================================

Call ID: call_12345
Agent: agent_042
Duration: 8:42
Date: 2025-11-15

--- Quality Metrics ---

Overall Call Quality Score: 7.8/10

Customer Satisfaction Indicators:
  ✓ Issue resolved: Yes
  ✓ Final sentiment: Positive (0.72)
  ✓ Escalation: None
  ⚠ Frustration detected: Yes (early in call)

Agent Performance:
  ✓ Response time: Good (avg 2.3s)
  ✓ Empathy markers: 12 instances
  ✓ Professional tone: Maintained
  ⚠ Active listening: Room for improvement

--- Sentiment Timeline ---

[00:00 - 01:30] Customer: Frustrated (0.82)
  Issue: Initial complaint about service outage
  Emotion: angry, frustrated
  → Training Note: Customer was highly frustrated at call start

[01:30 - 03:45] Agent: Calm, Empathetic
  Response: Acknowledgment and apology
  Markers: slow speech, low pitch, measured pace
  → Good example of de-escalation technique

[03:45 - 06:20] Mixed: Problem-Solving
  Customer sentiment improving
  Frustration → Neutral → Slightly Positive
  → Agent effectively addressed concerns

[06:20 - 08:42] Customer: Satisfied (0.75)
  Resolution achieved
  Emotion: satisfied, relieved
  → Successful call resolution

--- Training Opportunities ---

🎯 Strength: De-escalation
  [01:45] Excellent use of empathy and acknowledgment
  "I completely understand how frustrating that must be..."

⚠️  Development Area: Active Listening
  [04:23] Customer repeated concern - may indicate agent missed cue
  "Like I said before, the connection keeps dropping..."
  → Recommendation: Practice summarization techniques

💡 Coaching Suggestion:
  Agent handled frustrated customer well. Consider using this as
  training example for new agents on de-escalation techniques.

--- Recommendations ---

1. ✅ Maintain current empathy approach
2. 📚 Training: Active listening and summarization
3. 🔄 Follow-up: Customer satisfaction survey recommended
```

### Common Pitfalls

1. **Phone line quality**: Compression artifacts affect emotion detection
2. **Hold music**: Remove or mark hold periods before processing
3. **IVR segments**: Exclude automated segments from analysis
4. **Background noise**: Call center environments can be noisy

### Performance Considerations

| Batch Size | Processing Time* | Recommended Hardware |
|-----------|------------------|---------------------|
| 10 calls  | ~30 min          | RTX 3060+           |
| 100 calls | ~5 hours         | RTX 3080+           |
| 1000 calls| Overnight        | Multiple GPUs       |

*Average 5-minute calls with full enrichment

### Use Case Variations

**Agent Training:**
```bash
# Focus on training opportunities
python examples/workflows/call_center_qa.py \
  --audio raw_audio/call_12345.wav \
  --focus training \
  --highlight-best-practices
```

**Trend Analysis:**
```bash
# Aggregate across time period
python examples/workflows/call_center_qa.py \
  --batch raw_audio/calls_november/*.wav \
  --aggregate-by date \
  --trend-report
```

---

## Research Data Processing

### Overview
Batch process large collections of audio recordings for research studies, corpus analysis, or archival projects.

### Input Requirements
- Audio format: Any ffmpeg-compatible format
- Typical corpus size: 10-1000+ files
- Content: Research interviews, field recordings, oral histories

### Configuration

Template: `configs/research_batch_template.json`

```json
{
  "transcription": {
    "model": "large-v3",
    "language": "auto",
    "device": "cuda",
    "compute_type": "float16",
    "skip_existing": true
  },
  "enrichment": {
    "enable_prosody": true,
    "enable_emotion_dimensional": true,
    "enable_emotion_categorical": false,
    "batch_size": 10
  },
  "batch_settings": {
    "parallel_workers": 1,
    "progress_tracking": true,
    "error_handling": "continue",
    "checkpoint_interval": 10
  },
  "output": {
    "individual_transcripts": true,
    "aggregated_csv": true,
    "statistics_report": true,
    "quality_control_report": true
  },
  "metadata": {
    "extract_from_filename": true,
    "filename_pattern": "{study_id}_{participant_id}_{session}.wav",
    "custom_fields": ["study_id", "participant_id", "session"]
  }
}
```

### Usage Example

```bash
# Process all files in directory
python examples/workflows/batch_research_processing.py \
  --input-dir raw_audio/study_2025 \
  --output-dir outputs/study_2025_processed \
  --config configs/research_config.json

# Resume interrupted batch
python examples/workflows/batch_research_processing.py \
  --input-dir raw_audio/study_2025 \
  --output-dir outputs/study_2025_processed \
  --resume

# Process with metadata extraction
python examples/workflows/batch_research_processing.py \
  --input-dir raw_audio/study_2025 \
  --output-dir outputs/study_2025_processed \
  --extract-metadata \
  --pattern "{condition}_{subject_id}_{trial}.wav"

# Quality control only (no reprocessing)
python examples/workflows/batch_research_processing.py \
  --input-dir outputs/study_2025_processed \
  --qc-only
```

### Output Structure

```
outputs/study_2025_processed/
├── transcripts/
│   ├── control_001_1.json
│   ├── control_001_2.json
│   ├── treatment_002_1.json
│   └── ...
├── aggregated/
│   ├── all_segments.csv           # All segments from all files
│   ├── file_metadata.csv          # File-level statistics
│   └── participant_summary.csv    # Aggregated by participant
├── reports/
│   ├── processing_log.txt         # Detailed processing log
│   ├── statistics_report.txt      # Corpus statistics
│   ├── quality_control.txt        # QC findings
│   └── error_log.txt              # Any errors encountered
└── checkpoints/
    └── batch_state.json           # Resume state
```

### Example Statistics Report

```
RESEARCH CORPUS PROCESSING REPORT
================================================================================

Corpus: Study 2025 - Emotional Response to Stimuli
Processing Date: 2025-11-15
Total Files: 245

--- Processing Summary ---

Successfully Processed: 242 (98.8%)
Failed: 3 (1.2%)
  - audio_corrupt_045.wav: Audio file corrupted
  - missing_data_128.wav: File not found
  - incomplete_237.wav: Duration too short (<10s)

Total Audio Duration: 182.5 hours
Total Processing Time: 36.8 hours (5.0x real-time)
Average Processing Speed: 4.96x real-time

--- Corpus Statistics ---

Participants: 82
Sessions per Participant: 2-3 (avg: 2.95)
Average Session Duration: 44.8 minutes

Language Distribution:
  English: 238 (97.1%)
  Spanish: 4 (1.6%)
  Mixed: 3 (1.2%)

--- Audio Quality Metrics ---

Average SNR: 24.3 dB (Good)
Audio Quality Distribution:
  Excellent (>30 dB): 89 files (36.3%)
  Good (20-30 dB): 128 files (52.2%)
  Fair (15-20 dB): 22 files (9.0%)
  Poor (<15 dB): 6 files (2.4%)

Recommended for Re-recording: 6 files (see qc_report.txt)

--- Enrichment Statistics ---

Segments with Prosody Features: 45,239 (98.5%)
Segments with Emotion Features: 44,892 (97.8%)

Average Features per Segment:
  - Pitch analysis: 98.5%
  - Energy analysis: 98.2%
  - Speech rate: 97.9%
  - Pause detection: 96.1%
  - Emotion (dimensional): 97.8%

--- Prosody Distributions ---

Pitch (mean across corpus):
  Mean: 187.3 Hz
  Range: 95.2 - 412.8 Hz
  Distribution:
    Low (< 150 Hz): 28.3%
    Medium (150-250 Hz): 54.2%
    High (> 250 Hz): 17.5%

Speech Rate (mean across corpus):
  Mean: 4.8 syllables/sec
  Range: 2.1 - 8.9 syllables/sec
  Distribution:
    Slow (< 4.0): 32.1%
    Normal (4.0-6.0): 52.8%
    Fast (> 6.0): 15.1%

--- Emotion Distributions ---

Valence (mean across corpus):
  Mean: 0.52 (Slightly Positive)
  Distribution:
    Negative (< 0.4): 23.4%
    Neutral (0.4-0.6): 48.9%
    Positive (> 0.6): 27.7%

Arousal (mean across corpus):
  Mean: 0.48 (Moderate)
  Distribution:
    Low (< 0.4): 31.2%
    Moderate (0.4-0.6): 42.3%
    High (> 0.6): 26.5%

--- Participant-Level Analysis ---

Participants with Complete Data: 79 (96.3%)
Average Sessions per Participant: 2.95

Metadata Fields Extracted:
  ✓ Study condition (control/treatment)
  ✓ Participant ID
  ✓ Session number
  ✓ Recording date (from filename)

--- Export Files Generated ---

1. all_segments.csv
   - 45,892 rows (segments)
   - 24 columns (features)
   - Size: 12.3 MB

2. file_metadata.csv
   - 242 rows (files)
   - 18 columns (metadata + summary stats)
   - Size: 156 KB

3. participant_summary.csv
   - 82 rows (participants)
   - 31 columns (aggregated features)
   - Size: 42 KB

--- Recommendations ---

✓ Corpus quality is excellent (98.8% success rate)
✓ Audio quality is good (88.5% good or better)
⚠ Review 6 poor-quality recordings for potential re-recording
✓ Metadata extraction successful for all files
✓ Ready for statistical analysis

--- Next Steps ---

1. Review quality control report for specific issues
2. Import CSVs into statistical software (R, SPSS, Python)
3. Consider re-recording 6 poor-quality files if critical
4. Validate metadata extraction for accuracy
```

### CSV Export Format

**all_segments.csv:**
```csv
file_id,study_condition,participant_id,session,segment_id,start_time,end_time,duration,text,pitch_level,pitch_mean_hz,pitch_contour,energy_level,energy_db_rms,rate_level,syllables_per_sec,pause_count,pause_density,emotion_valence,emotion_arousal,word_count
control_001_1,control,001,1,0,0.000,4.230,4.230,"Welcome and thank you for participating.",medium,198.4,rising,moderate,-12.3,normal,4.8,0,0.0,0.62,0.45,6
control_001_1,control,001,1,1,4.450,9.120,4.670,"Please make yourself comfortable.",low,165.2,flat,quiet,-18.7,slow,3.9,1,0.21,0.58,0.32,4
...
```

### Common Pitfalls

1. **Inconsistent file naming**: Affects metadata extraction
2. **Mixed audio quality**: Some files may fail enrichment
3. **Storage space**: Large corpora can exceed available disk
4. **Processing interruptions**: Always use checkpoint/resume features

### Performance Considerations

| Corpus Size | Est. Processing Time* | Storage Required |
|------------|---------------------|------------------|
| 10 files   | 1-2 hours           | ~50 MB           |
| 50 files   | 5-8 hours           | ~250 MB          |
| 100 files  | 10-16 hours         | ~500 MB          |
| 500 files  | 2-3 days            | ~2.5 GB          |
| 1000 files | 4-6 days            | ~5 GB            |

*Assuming 30-min average duration, large-v3 model, full enrichment, single GPU

### Optimization Tips

1. **Use skip_existing**: Resume interrupted processing without reprocessing
2. **Batch processing**: Process overnight or over weekend
3. **Quality filtering**: Pre-filter obviously corrupted files
4. **Parallel processing**: Use multiple GPUs if available
5. **Checkpointing**: Enable checkpoints every 10-20 files

### Statistical Analysis Integration

The CSV exports are designed for direct import into statistical software:

**R Example:**
```r
# Load data
segments <- read.csv("outputs/study_2025_processed/aggregated/all_segments.csv")
metadata <- read.csv("outputs/study_2025_processed/aggregated/file_metadata.csv")

# Analyze emotion by condition
library(dplyr)
emotion_by_condition <- segments %>%
  group_by(study_condition) %>%
  summarise(
    mean_valence = mean(emotion_valence, na.rm = TRUE),
    mean_arousal = mean(emotion_arousal, na.rm = TRUE),
    mean_pitch = mean(pitch_mean_hz, na.rm = TRUE)
  )
```

**Python/Pandas Example:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
segments = pd.read_csv("outputs/study_2025_processed/aggregated/all_segments.csv")

# Analyze pitch over time
participant_001 = segments[segments['participant_id'] == '001']
plt.plot(participant_001['start_time'], participant_001['pitch_mean_hz'])
plt.xlabel('Time (s)')
plt.ylabel('Pitch (Hz)')
plt.title('Pitch Contour - Participant 001')
plt.show()
```

---

## Performance Optimization Guide

### Hardware Recommendations

**Minimum:**
- GPU: NVIDIA GTX 1660 (6 GB VRAM)
- CPU: 4 cores
- RAM: 16 GB
- Storage: 100 GB free

**Recommended:**
- GPU: NVIDIA RTX 3080 (10 GB VRAM)
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 500 GB SSD

**Optimal (Batch Processing):**
- GPU: NVIDIA RTX 4090 or multiple GPUs
- CPU: 16+ cores
- RAM: 64 GB
- Storage: 1+ TB NVMe SSD

### Model Selection

| Model | Speed | Accuracy | VRAM | Best For |
|-------|-------|----------|------|----------|
| tiny | 32x | Low | 1 GB | Testing only |
| base | 16x | Moderate | 1 GB | Drafts |
| small | 6x | Good | 2 GB | Quick processing |
| medium | 2x | Very Good | 5 GB | Balanced |
| large-v3 | 1x | Excellent | 10 GB | Production |

### Compute Type Selection

| Type | Speed | Quality | VRAM |
|------|-------|---------|------|
| int8 | Fastest | Good | Lowest |
| int8_float16 | Fast | Very Good | Low |
| float16 | Medium | Excellent | Medium |
| float32 | Slow | Excellent | High |

### Processing Speed Estimates

**Single file (30-min audio, large-v3, full enrichment, RTX 3080):**
- Transcription: ~6 minutes
- Enrichment: ~3 minutes
- Total: ~9 minutes (3.3x real-time)

**Optimization techniques:**
1. Use `--skip-existing` to avoid reprocessing
2. Process similar-length files together
3. Use SSD for temporary files
4. Close other GPU applications
5. Use compute_type int8_float16 for 2x speedup with minimal quality loss

---

## Troubleshooting

### Common Issues

**1. Out of memory errors**
```
Error: CUDA out of memory
```
**Solution:** Use smaller model or int8 compute type:
```bash
--model medium --compute-type int8_float16
```

**2. Audio file not found**
```
Error: Audio file not found: raw_audio/meeting.mp3
```
**Solution:** Check file path and ensure ffmpeg can read the format:
```bash
ffmpeg -i raw_audio/meeting.mp3
```

**3. Poor transcription quality**
```
Warning: Low confidence scores detected
```
**Solutions:**
- Check audio quality (SNR should be >15 dB)
- Try `--language en` instead of auto-detect
- Use larger model (`large-v3`)
- Pre-process audio with noise reduction

**4. Emotion features missing**
```
Warning: No emotion features extracted
```
**Solutions:**
- Ensure enrichment dependencies installed: `pip install -r requirements.txt`
- Check segment duration (needs >0.5s)
- Verify audio file is accessible

### Getting Help

- Check logs in `outputs/<name>/processing.log`
- Review error messages in console output
- See main documentation: `/docs/`
- Open issue on GitHub with log files

---

## Configuration Reference

### Complete Configuration Schema

```json
{
  "transcription": {
    "model": "large-v3 | medium | small | base | tiny",
    "language": "en | auto | es | fr | de | ...",
    "device": "cuda | cpu",
    "compute_type": "float16 | int8_float16 | int8 | float32",
    "beam_size": 5,
    "vad_min_silence_ms": 500,
    "skip_existing": true
  },
  "enrichment": {
    "enable_prosody": true,
    "enable_emotion_dimensional": true,
    "enable_emotion_categorical": true,
    "device": "cuda | cpu",
    "batch_size": 10
  },
  "output": {
    "formats": ["json", "txt", "csv", "annotated"],
    "include_metadata": true,
    "pretty_print": true
  },
  "workflow_specific": {
    "// See individual workflow configs for specific options": null
  }
}
```

### Environment Variables

```bash
# Override default GPU device
export CUDA_VISIBLE_DEVICES=0

# Set model cache directory
export HF_HOME=/path/to/model/cache

# Enable debug logging
export LOG_LEVEL=DEBUG
```

---

## Additional Resources

- **Main Documentation:** `/docs/AUDIO_ENRICHMENT.md`
- **Quick Start:** `/QUICKSTART_AUDIO_ENRICHMENT.md`
- **API Reference:** `/docs/API.md`
- **Examples:** `/examples/`

---

## License

See main project LICENSE file.
