# slower-whisper (ffmpeg + faster-whisper)

A fully local, GPU-accelerated transcription and analysis toolkit for audio files.

## Overview

This project provides a complete two-stage pipeline for audio transcription and enrichment:

**Stage 1: Transcription**
- Normalizes audio in `raw_audio/` to 16 kHz mono WAV using `ffmpeg`
- Transcribes using Whisper via `faster-whisper` on CUDA
- Outputs: TXT, SRT, VTT subtitles + structured JSON

**Stage 2: Enrichment** (optional)
- Tone analysis: LLM-powered emotional tone classification per segment
- Speaker diarization: Automatic speaker identification and attribution
- Analytics: Transcript indices, tone reports, speaker reports

All processing runs **100% locally**. Your audio never leaves your machine (except for optional LLM API calls for tone analysis).

## Requirements

**Stage 1 (Transcription)**:
- Windows, Linux, or macOS with Python 3.9+
- NVIDIA GPU with CUDA support (recommended) or CPU
- `ffmpeg` on PATH
- Python packages: `faster-whisper`

**Stage 2 (Enrichment)** - optional:
- For tone analysis: Anthropic API key (or OpenAI API key)
- For speaker diarization: HuggingFace token + `pyannote.audio`

### Installation

**Install core dependencies (Stage 1)**:
```bash
pip install faster-whisper
```

**Install enrichment dependencies (Stage 2)** - only if needed:
```bash
# For tone analysis
pip install anthropic  # or: pip install openai

# For speaker diarization
pip install pyannote.audio
```

**Install ffmpeg**:

Windows (PowerShell, elevated):
```powershell
choco install ffmpeg -y
```

macOS:
```bash
brew install ffmpeg
```

Linux (Ubuntu/Debian):
```bash
sudo apt update && sudo apt install ffmpeg
```

## Directory Layout

```text
root/
  # Core pipeline
  transcribe_pipeline.py      # Stage 1: Audio → Transcripts
  tone_enrich.py              # Stage 2: Tone analysis
  speaker_enrich.py           # Stage 2: Speaker diarization
  generate_index.py           # Analytics & reporting

  # Package
  transcription/
    models.py                 # Domain models (Segment, Transcript)
    config.py                 # Configuration
    asr_engine.py             # Whisper wrapper
    audio_io.py               # FFmpeg normalization
    writers.py                # JSON/TXT/SRT/VTT writers
    pipeline.py               # Orchestration
    cli.py                    # CLI
    enrichment/               # Stage 2 modules
      tone/                   # Tone analysis
      speaker/                # Speaker diarization
      analytics/              # Reports & indices

  # Data directories (created automatically)
  raw_audio/                  # Your original audio files
  input_audio/                # Normalized 16 kHz mono WAVs
  transcripts/                # .txt, .srt, .vtt outputs
  whisper_json/               # Structured JSON (canonical format)

  # Optional outputs
  transcripts_index.json      # Index of all transcripts
  transcripts_index.csv       # CSV index (for Excel)
  tone_analysis.md            # Tone distribution report
  speaker_analysis.md         # Speaker diarization report
```

Directories are created automatically as needed.

## Quick Start

### Stage 1: Transcription

1. **Place audio files** in `raw_audio/` (supports .mp3, .m4a, .wav, etc.)

2. **Run transcription**:
   ```bash
   python transcribe_pipeline.py --language en
   ```

3. **Outputs** are created in:
   - `transcripts/` - Human-readable TXT, SRT, VTT
   - `whisper_json/` - Structured JSON (for Stage 2)

**CLI Options**:
```bash
# Basic usage with defaults
python transcribe_pipeline.py

# Force language and skip already-transcribed files
python transcribe_pipeline.py --language en --skip-existing-json

# Use lighter model with quantization (faster, less accurate)
python transcribe_pipeline.py --model medium --compute-type int8_float16

# Use CPU instead of GPU
python transcribe_pipeline.py --device cpu

# All options
python transcribe_pipeline.py \
  --root /path/to/data \
  --model large-v3 \
  --device cuda \
  --language en \
  --skip-existing-json
```

### Stage 2: Enrichment

#### Tone Analysis

Analyzes emotional tone of each segment using LLM (Claude or GPT).

**Setup**:
```bash
export ANTHROPIC_API_KEY="your-key-here"  # or OPENAI_API_KEY
```

**Run**:
```bash
# Enrich all transcripts with tone
python tone_enrich.py

# Enrich specific file
python tone_enrich.py --file meeting1.json

# Skip already-enriched files
python tone_enrich.py --skip-existing

# Use different model/provider
python tone_enrich.py --provider openai --model gpt-4

# Test without API (assigns all "neutral")
python tone_enrich.py --provider mock
```

**Tone labels**: neutral, positive, negative, questioning, uncertain, emphatic

#### Speaker Diarization

Identifies and labels speakers using pyannote.audio.

**Setup**:
```bash
export HF_TOKEN="your-huggingface-token"  # Get from: https://huggingface.co/settings/tokens
```

**Run**:
```bash
# Diarize all transcripts
python speaker_enrich.py

# Diarize specific file
python speaker_enrich.py --file meeting1.json

# If you know the number of speakers
python speaker_enrich.py --num-speakers 3

# Provide custom speaker names
python speaker_enrich.py --speaker-map speakers.json
```

**Speaker mapping** (`speakers.json`):
```json
{
  "SPEAKER_00": "Alice",
  "SPEAKER_01": "Bob",
  "SPEAKER_02": "Charlie"
}
```

#### Analytics & Reports

Generate indices and reports from enriched transcripts.

```bash
# Generate everything (indices + reports)
python generate_index.py

# Just indices
python generate_index.py --index-only

# Just reports
python generate_index.py --tone-report --speaker-report

# Custom output directory
python generate_index.py --output-dir reports/
```

**Outputs**:
- `transcripts_index.json` - JSON index of all transcripts
- `transcripts_index.csv` - CSV for Excel/spreadsheet analysis
- `tone_analysis.md` - Tone distribution report
- `speaker_analysis.md` - Speaker time analysis

## Model download & privacy

On first use of a given model (e.g. `large-v3`), `faster-whisper` will
download the model weights and cache them locally. This requires one-time
internet access to fetch the weights.

**Your audio and transcripts are not uploaded or sent anywhere by this
pipeline.** All transcription runs locally on your machine; only the model
weights are fetched from the internet on first use.

## JSON schema

Each JSON file looks like:

```json
{
  "schema_version": 1,
  "file": "meeting1.wav",
  "language": "en",
  "meta": {
    "generated_at": "2025-11-15T03:21:00Z",
    "audio_file": "meeting1.wav",
    "audio_duration_sec": 3120.5,
    "model_name": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "beam_size": 5,
    "vad_min_silence_ms": 500,
    "language_hint": "en",
    "task": "transcribe",
    "pipeline_version": "1.0.0",
    "root": "C:/transcription_toolkit"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Okay, let's get started with today's agenda.",
      "speaker": null,
      "tone": null
    }
  ]
}
```

This schema is stable and is intended to be the basis for future tooling:

- Tone tagging: populate `tone`.
- Speaker diarization: populate `speaker`.
- Search and analysis: operate over `segments[]`.
- Run-level analysis and reproducibility: read `meta`.

## Typical Workflows

### Basic Transcription
```bash
# 1. Put audio in raw_audio/
# 2. Transcribe
python transcribe_pipeline.py --language en --skip-existing-json
# 3. Find outputs in transcripts/ and whisper_json/
```

### Full Enrichment Pipeline
```bash
# 1. Transcribe
python transcribe_pipeline.py --language en

# 2. Add tone analysis
export ANTHROPIC_API_KEY="your-key"
python tone_enrich.py --skip-existing

# 3. Add speaker labels
export HF_TOKEN="your-token"
python speaker_enrich.py --skip-existing

# 4. Generate reports
python generate_index.py

# Results:
# - whisper_json/*.json (enriched with tone + speakers)
# - transcripts/*.vtt (subtitles with speaker/tone metadata)
# - tone_analysis.md, speaker_analysis.md (insights)
```

### Re-enrich Existing Transcripts
If you already transcribed files but want to add enrichment later:

```bash
# Your JSON files are already in whisper_json/
# Just run enrichment directly:
python tone_enrich.py
python speaker_enrich.py
python generate_index.py
```

The enrichment updates JSON files in-place and adds metadata to `meta.enrichments`.

## Architecture

### Two-Stage Design

**Stage 1** (transcription) runs once per audio file and is computationally expensive. It creates the canonical JSON.

**Stage 2** (enrichment) operates on JSON only and can be run multiple times with different parameters without re-transcribing.

This separation means:
- ✅ Transcribe once, analyze many times
- ✅ Experiment with tone/speaker models without reprocessing audio
- ✅ Add new enrichment types (summaries, topics, etc.) without touching Stage 1

### Modules

```
transcription/
├── models.py          # Segment, Transcript dataclasses
├── config.py          # Configuration classes
├── audio_io.py        # FFmpeg normalization
├── asr_engine.py      # Whisper wrapper
├── writers.py         # JSON/TXT/SRT/VTT output
├── pipeline.py        # Stage 1 orchestration
├── cli.py             # CLI entrypoint
└── enrichment/        # Stage 2 modules
    ├── tone/          # LLM-based tone analysis
    ├── speaker/       # Pyannote diarization
    └── analytics/     # Indexing and reports
```

## Testing

Run tests with pytest:

```bash
pip install pytest
pytest tests/
```

Tests cover:
- JSON schema validation
- SRT/VTT timestamp formatting
- Tone analyzer (mock mode)
- Index generation

## Troubleshooting

### "CUDA out of memory"
- Use a smaller model: `--model medium` or `--model small`
- Use quantization: `--compute-type int8_float16`
- Reduce batch size in tone enrichment: `--batch-size 5`

### "ffmpeg not found"
- Ensure ffmpeg is installed and on PATH
- Test: `ffmpeg -version`

### "Anthropic API key not found"
```bash
export ANTHROPIC_API_KEY="your-key-here"  # Linux/Mac
# or
set ANTHROPIC_API_KEY=your-key-here       # Windows CMD
```

### "HuggingFace token required"
- Get token from: https://huggingface.co/settings/tokens
- Accept pyannote model conditions: https://huggingface.co/pyannote/speaker-diarization-3.1
```bash
export HF_TOKEN="your-token"
```

### Slow transcription
- Ensure GPU is being used: check for `device: cuda` in logs
- Check RTF (Real-Time Factor) in output. Target: < 0.3x for large-v3
- Consider smaller model for faster processing

### Speaker diarization accuracy
- Use `--num-speakers N` if you know the count
- Provide speaker mapping after initial run for better labels
- Ensure audio quality is good (clear voices, low background noise)

## Privacy & Security

**Local Processing**:
- Stage 1 (transcription) runs 100% locally
- Audio files never leave your machine
- Only model weights are downloaded once from internet

**API Usage (Stage 2 Tone Analysis)**:
- Sends **text segments only** to LLM API (not audio)
- To avoid API: use `--provider mock` or skip tone enrichment
- Alternative: Use local LLM (requires custom integration)

**No Telemetry**:
- This toolkit sends no usage data, analytics, or telemetry
- All data stays on your machine

## Extending

Want to add new enrichment types? Follow this pattern:

1. Create new module in `transcription/enrichment/`
2. Read JSON → `Transcript` objects
3. Enrich segments or add to `meta.enrichments`
4. Write updated JSON back
5. Create CLI script

Examples:
- **Summarization**: Read JSON, send to LLM for summary, add to `meta.summary`
- **Topic tagging**: Classify segments by topic, add `segment.topic`
- **Search index**: Build full-text search index from JSON
- **Translation**: Add `segment.translation` for multilingual support

The `Transcript` and `Segment` models are designed to be extended without breaking existing code.
