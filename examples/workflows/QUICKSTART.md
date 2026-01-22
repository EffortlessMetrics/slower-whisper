# Workflow Quick Start Guide

Get started with the most common workflow examples in under 5 minutes.

## Prerequisites

1. **Complete base transcription** for your audio files:
   ```bash
   slower-whisper transcribe
   ```

2. **Enrich with audio features** (recommended):
   ```bash
   slower-whisper enrich
   ```

3. Your enriched transcripts should now be in `whisper_json/`

## Quick Examples

### 1. Meeting Analysis (5 minutes)

**Scenario:** You recorded a team meeting and want to find key moments and engagement levels.

```bash
# Process the meeting
python examples/workflows/meeting_transcription.py \
  --audio raw_audio/team_meeting.mp3 \
  --output outputs/meetings

# Outputs:
# - team_meeting_annotated.txt (readable transcript with emotions)
# - team_meeting_analysis.txt (summary report)
# - team_meeting.csv (spreadsheet data)
# - team_meeting_key_moments.json (important moments)
```

**What you get:**
- Searchable transcript with emotional markers
- Key moments: decisions, action items, heated discussions
- Engagement timeline showing energy levels over time

---

### 2. Podcast Highlights (3 minutes)

**Scenario:** You have a podcast episode and want to generate show notes and find clips to share.

```bash
# Quick highlights extraction
python examples/workflows/podcast_processing.py \
  --audio raw_audio/episode_042.mp3 \
  --title "Episode 42: Future of AI" \
  --highlights-only

# Or full processing with show notes
python examples/workflows/podcast_processing.py \
  --audio raw_audio/episode_042.mp3 \
  --title "Episode 42: Future of AI" \
  --output outputs/podcasts

# Outputs:
# - episode_042_highlights.txt (shareable moments)
# - episode_042_show_notes.md (formatted show notes)
# - episode_042_quotes.txt (social media quotes)
# - episode_042_chapters.json (chapter markers)
```

**What you get:**
- Highlight reel timestamps for exciting moments
- Show notes with emotional context
- Pull quotes for social media
- Chapter markers for podcast players

---

### 3. Interview Insights (10 minutes)

**Scenario:** You conducted user research interviews and want to find patterns and key insights.

```bash
# Analyze single interview
python examples/workflows/interview_analysis.py \
  --audio raw_audio/user_interview_05.wav \
  --output outputs/interviews \
  --report

# Batch process multiple interviews
python examples/workflows/interview_analysis.py \
  --batch raw_audio/interviews/*.wav \
  --output outputs/interviews

# Outputs per interview:
# - user_interview_05_analysis.txt (comprehensive report)
# - user_interview_05_timeline.csv (emotional journey)
# - user_interview_05_shifts.json (emotion changes)
# - user_interview_05_moments.json (key insights)
```

**What you get:**
- Emotional journey through the interview
- Confidence vs. uncertainty indicators
- Key insights and "aha" moments
- Pain points and positive reactions

---

### 4. Call Center QA (5 minutes per call)

**Scenario:** You need to review customer service calls for quality and training opportunities.

```bash
# Analyze single call
python examples/workflows/call_center_qa.py \
  --audio raw_audio/call_12345.wav \
  --agent-id agent_042 \
  --output outputs/qa

# Batch process with agent summary
python examples/workflows/call_center_qa.py \
  --batch raw_audio/calls/*.wav \
  --output outputs/qa \
  --aggregate-by-agent

# Outputs:
# - call_12345_qa_report.txt (detailed QA report)
# - call_12345_qa_metrics.json (quality scores)
# - call_12345_timeline.csv (sentiment timeline)
# - agent_performance_summary.txt (batch only)
```

**What you get:**
- Quality score (0-10)
- Frustration/satisfaction detection
- Agent strengths and training opportunities
- Sentiment timeline of the call

---

### 5. Research Batch Processing (30+ minutes for large datasets)

**Scenario:** You have 100+ audio files from a research study and need to process them all.

```bash
# Process entire directory
python examples/workflows/batch_research_processing.py \
  --input-dir whisper_json \
  --output-dir outputs/study_2025 \
  --pattern "{condition}_{subject_id}_{session}.json"

# Outputs:
# - aggregated/all_segments.csv (all segments, all files)
# - aggregated/file_metadata.csv (per-file statistics)
# - aggregated/participant_summary.csv (per-participant data)
# - reports/statistics_report.txt (corpus overview)
# - reports/quality_control.txt (QC findings)
```

**What you get:**
- CSV files ready for R/Python/SPSS analysis
- Corpus-wide statistics
- Quality control report
- Metadata extraction from filenames

---

## Common Patterns

### Pattern 1: Full Pipeline (from audio to analysis)

```bash
# 1. Transcribe
slower-whisper transcribe

# 2. Enrich with audio features
slower-whisper enrich

# 3. Run workflow
python examples/workflows/meeting_transcription.py \
  --audio raw_audio/meeting.mp3
```

### Pattern 2: Using Configuration Files

```bash
# 1. Copy template
cp examples/workflows/configs/meeting_template.json my_meeting_config.json

# 2. Edit settings (use your text editor)
nano my_meeting_config.json

# 3. Run with config
python examples/workflows/meeting_transcription.py \
  --config my_meeting_config.json \
  --audio raw_audio/meeting.mp3
```

### Pattern 3: Batch Processing

```bash
# Process all MP3 files in a directory
python examples/workflows/podcast_processing.py \
  --batch raw_audio/podcasts/*.mp3 \
  --output outputs/all_podcasts
```

### Pattern 4: Resume Interrupted Processing

```bash
# The batch processor saves checkpoints every 10 files
# If interrupted, just re-run - it will skip already processed files
python examples/workflows/batch_research_processing.py \
  --input-dir whisper_json \
  --output-dir outputs/study_2025
```

---

## Troubleshooting Quick Fixes

### "Transcript not found" Error

**Problem:** Workflow can't find the JSON transcript.

**Solution:**
```bash
# Make sure you've run transcription first
slower-whisper transcribe

# Check if JSON exists
ls whisper_json/

# If JSON exists but workflow still can't find it, specify full path
python examples/workflows/meeting_transcription.py \
  --audio raw_audio/meeting.mp3 \
  --json whisper_json/meeting.json
```

### "No audio features" Warning

**Problem:** Transcript exists but has no emotional/prosodic data.

**Solution:**
```bash
# Run enrichment on the transcript
slower-whisper enrich --file whisper_json/meeting.json
```

### Performance Too Slow

**Problem:** Processing takes too long.

**Solution:**
```bash
# Use faster model for drafts
slower-whisper transcribe --model medium

# Use int8 compute type
slower-whisper transcribe --compute-type int8_float16

# For podcasts: use highlights-only mode
python examples/workflows/podcast_processing.py \
  --audio episode.mp3 \
  --highlights-only
```

---

## Next Steps

1. **Try the examples** above with your own audio files
2. **Review the outputs** in `outputs/` directories
3. **Customize configurations** in `configs/` templates
4. **Read full documentation** in `README.md` for advanced features
5. **Check output examples** in `outputs/` for format reference

## Need Help?

- Full documentation: `README.md`
- Configuration examples: `configs/`
- Output examples: `outputs/`
- Main project docs: `../../docs/`

## Pro Tips

1. **Start small:** Test with one short file before batch processing
2. **Use configs:** Save time by creating reusable configuration files
3. **Check examples:** Review output examples to understand formats
4. **Customize thresholds:** Adjust detection thresholds based on your needs
5. **Export to CSV:** Use CSV exports for statistical analysis in R/Python/Excel

---

**Happy analyzing!** 🎯
