# Stage 2: Enrichment Pipeline Design

## Overview

Stage 2 operates on the structured JSON output from Stage 1 (transcription). It enriches transcripts with:
- **Tone analysis**: Emotional/tonal labels per segment
- **Speaker diarization**: Speaker attribution per segment
- **Summaries & analytics**: Aggregate insights

**Key principle**: Stage 2 never re-transcribes audio. It consumes `whisper_json/*.json` as input.

---

## 1. Tone Analysis

### 1.1 Tone Taxonomy

We use a focused set of tone labels optimized for conversational audio:

**Primary Tones** (mutually exclusive):
- `neutral` - Calm, matter-of-fact delivery
- `positive` - Enthusiastic, happy, excited, encouraging
- `negative` - Frustrated, angry, disappointed, critical
- `questioning` - Curious, probing, seeking clarification
- `uncertain` - Hesitant, doubtful, tentative
- `emphatic` - Strong emphasis, passionate, assertive

**Confidence Score**: 0.0 to 1.0 indicating model confidence

**Rationale**:
- Small enough to be learnable and consistent
- Large enough to capture meaningful emotional variation
- Avoids overly granular categories (e.g., "slightly annoyed" vs "moderately annoyed")
- Works for business meetings, podcasts, interviews, personal audio

### 1.2 Tone Analysis Approach

**LLM-based classification** (preferred):
- Use Claude API (or local LLM) for nuanced understanding
- Batch segments for efficiency (e.g., 5-10 segments at a time with context)
- Prompt engineering for consistent classification
- Fallback to `neutral` if model unavailable

**Input**: Segment text + optional surrounding context
**Output**: Tone label + confidence score

### 1.3 Tone Enrichment Flow

```
whisper_json/file.json
    ↓
Read → Transcript object
    ↓
annotate_tone(transcript, config)
    ↓ (calls LLM API per segment/batch)
Updated Transcript (segment.tone populated)
    ↓
Write → whisper_json/file.json (updated in-place)
    ↓
Optional: Generate tone report (MD/HTML)
```

**Versioning**:
- When tone is added, update `meta.enrichments.tone_version = "1.0"`
- Track `meta.enrichments.tone_model` (e.g., "claude-sonnet-4.5")
- Track `meta.enrichments.tone_timestamp`

---

## 2. Speaker Diarization

### 2.1 Approach

Use **pyannote.audio** for speaker diarization:
- Pre-trained models available
- CUDA support for GPU acceleration
- Outputs: speaker turns with timestamps

### 2.2 Diarization Flow

```
input_audio/file.wav + whisper_json/file.json
    ↓
Run pyannote diarization → speaker turns [(start, end, speaker_id), ...]
    ↓
Map turns to segments via timestamp overlap
    ↓
Update Transcript (segment.speaker populated)
    ↓
Write → whisper_json/file.json (updated in-place)
```

**Versioning**:
- `meta.enrichments.speaker_version = "1.0"`
- `meta.enrichments.diarization_model`
- `meta.enrichments.diarization_timestamp`

### 2.3 Speaker Labels

Format: `SPEAKER_00`, `SPEAKER_01`, etc.

Optional: User can provide mapping JSON:
```json
{
  "SPEAKER_00": "Alice",
  "SPEAKER_01": "Bob"
}
```

---

## 3. Architecture

### 3.1 Directory Structure

```
transcription/
  enrichment/               # New package for Stage 2
    __init__.py
    tone/
      __init__.py
      analyzer.py           # ToneAnalyzer class
      prompts.py            # LLM prompts
      config.py             # ToneConfig
    speaker/
      __init__.py
      diarizer.py           # SpeakerDiarizer class
      config.py             # DiarizationConfig
    analytics/
      __init__.py
      indexer.py            # Generate index.json/csv
      reports.py            # Tone reports, summaries

tone_enrich.py              # CLI for tone enrichment
speaker_enrich.py           # CLI for speaker diarization
generate_index.py           # CLI for index generation
```

### 3.2 Enrichment Metadata Schema

Add to `Transcript.meta`:

```json
{
  "meta": {
    ...existing ASR metadata...
    "enrichments": {
      "tone_version": "1.0",
      "tone_model": "claude-sonnet-4.5",
      "tone_timestamp": "2025-11-15T10:30:00Z",
      "speaker_version": "1.0",
      "diarization_model": "pyannote/speaker-diarization-3.1",
      "diarization_timestamp": "2025-11-15T10:35:00Z"
    }
  }
}
```

### 3.3 Configuration

**ToneConfig**:
- `api_provider` (e.g., "anthropic", "openai", "local")
- `model_name`
- `api_key` (from env var)
- `batch_size` (segments per API call)
- `max_retries`
- `confidence_threshold`

**DiarizationConfig**:
- `model_name` (pyannote model)
- `device` (cuda/cpu)
- `num_speakers` (optional hint)
- `min_speakers`, `max_speakers`

---

## 4. Index & Analytics

### 4.1 Transcript Index

`transcripts_index.json`:
```json
{
  "generated_at": "2025-11-15T10:00:00Z",
  "total_files": 42,
  "total_duration_sec": 25200.5,
  "transcripts": [
    {
      "file_name": "meeting1.wav",
      "json_path": "whisper_json/meeting1.json",
      "duration_sec": 1800.5,
      "language": "en",
      "segment_count": 245,
      "has_tone": true,
      "has_speakers": true,
      "transcribed_at": "2025-11-14T08:00:00Z"
    },
    ...
  ]
}
```

Also generate `transcripts_index.csv` for Excel analysis.

### 4.2 Tone Analytics Report

`reports/tone_analysis_YYYYMMDD.md`:

```markdown
# Tone Analysis Report

**Generated**: 2025-11-15 10:00:00
**Files analyzed**: 42
**Total segments**: 8,432

## Overall Tone Distribution

- Neutral: 45.2% (3,811 segments)
- Positive: 28.1% (2,369 segments)
- Negative: 12.3% (1,037 segments)
- Questioning: 8.4% (708 segments)
- Uncertain: 4.2% (354 segments)
- Emphatic: 1.8% (153 segments)

## Per-File Summary

### meeting1.wav
- Duration: 30m 0s
- Segments: 245
- Tone breakdown: Neutral 50%, Positive 30%, ...

...
```

---

## 5. CLI Design

### tone_enrich.py

```bash
# Enrich all transcripts
python tone_enrich.py

# Enrich specific file
python tone_enrich.py --file meeting1.json

# Enrich with options
python tone_enrich.py --model claude-sonnet-4.5 --batch-size 10 --report

# Skip already enriched
python tone_enrich.py --skip-existing
```

### speaker_enrich.py

```bash
# Diarize all
python speaker_enrich.py

# Diarize specific file
python speaker_enrich.py --file meeting1.json

# With speaker count hint
python speaker_enrich.py --num-speakers 3

# With custom labels
python speaker_enrich.py --speaker-map speakers.json
```

### generate_index.py

```bash
# Generate both JSON and CSV indices
python generate_index.py

# Output to specific location
python generate_index.py --output-dir reports/
```

---

## 6. Error Handling & Robustness

- **Per-file isolation**: One file's enrichment failure doesn't stop batch
- **Partial enrichment**: If tone fails on segment 50/100, save first 49 and continue
- **Retry logic**: Exponential backoff for API calls (tone analysis)
- **Validation**: Verify JSON schema after enrichment
- **Backup**: Optional `--backup` flag to copy original JSON before enrichment

---

## 7. Testing Strategy

### Unit Tests
- `test_tone_analyzer.py`: Mock LLM responses, verify tone assignment
- `test_diarizer.py`: Mock pyannote output, verify speaker mapping
- `test_indexer.py`: Verify index generation from sample transcripts

### Integration Tests
- `test_full_enrichment.py`: End-to-end with sample audio

### Test Fixtures
- `tests/fixtures/sample_transcript.json`
- `tests/fixtures/sample_audio.wav`

---

## 8. Implementation Phases

1. **Phase 2A**: Tone analysis
   - Implement `ToneAnalyzer`
   - CLI (`tone_enrich.py`)
   - Tests

2. **Phase 2B**: Speaker diarization
   - Implement `SpeakerDiarizer`
   - CLI (`speaker_enrich.py`)
   - Tests

3. **Phase 2C**: Polish & analytics
   - VTT writer
   - Index generator
   - Tone analytics reports
   - Tests

4. **Phase 2D**: Documentation
   - Update README
   - Usage examples
   - Troubleshooting guide
