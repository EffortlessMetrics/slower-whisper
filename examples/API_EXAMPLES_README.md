# API Examples

This directory contains example scripts demonstrating real-world usage of the slower-whisper transcription API.

## Quick Start Examples

### 1. Basic Transcription (`basic_transcription.py`)
The simplest way to transcribe all audio files in a directory.

```bash
# Transcribe all files in raw_audio/ directory
python examples/basic_transcription.py /path/to/project

# With custom options
python examples/basic_transcription.py /path/to/project --language en --device cuda
```

**What it does:**
- Processes all audio files in `raw_audio/` subdirectory
- Auto-creates output directories (input_audio, whisper_json, transcripts)
- Saves JSON, TXT, and SRT formats
- Skips already-transcribed files by default

**Best for:** Batch processing multiple audio files

---

### 2. Single File Transcription (`single_file.py`)
Process a single audio file with detailed output.

```bash
# Basic usage
python examples/single_file.py interview.mp3 ./output

# Show all segments with timestamps
python examples/single_file.py podcast.wav ./output --show-segments --language en

# Translate to English
python examples/single_file.py speech.m4a ./output --task translate
```

**What it does:**
- Transcribes one audio file at a time
- Shows detailed statistics (duration, word count, segments)
- Displays transcript preview
- Optionally shows full segment-by-segment breakdown

**Best for:** Quick transcription of individual files, testing

---

### 3. Custom Configuration (`custom_config.py`)
Advanced configuration with presets and interactive mode.

```bash
# Use a preset
python examples/custom_config.py /path/to/project --preset balanced

# Interactive configuration builder
python examples/custom_config.py /path/to/project --custom

# List available presets
python examples/custom_config.py /path/to/project --list-presets
```

**Available Presets:**
- `fast_draft` - Fastest processing (tiny model)
- `balanced` - Good speed/quality trade-off (medium model)
- `high_quality` - Best quality (large-v3, thorough beam search)
- `cpu_fallback` - CPU-only with optimized settings
- `multilingual` - Auto-detect language, translate to English
- `english_only` - Optimized for English content

**Best for:** Fine-tuning quality vs. speed, resource-constrained environments

---

### 4. Two-Stage Enrichment Workflow (`enrichment_workflow.py`)
Complete pipeline: transcribe + enrich with audio features.

```bash
# Full workflow (transcribe + enrich)
python examples/enrichment_workflow.py /path/to/project --transcribe --enrich

# Transcription only (Stage 1)
python examples/enrichment_workflow.py /path/to/project --transcribe --language en

# Enrichment only (Stage 2) - requires existing transcripts
python examples/enrichment_workflow.py /path/to/project --enrich --enable-categorical

# Custom workflow
python examples/enrichment_workflow.py /path/to/project --transcribe --enrich \
    --model large-v3 --language en --enable-categorical --enrich-device cuda
```

**What it adds:**
- **Prosody features:** pitch, energy, speech rate, pauses
- **Dimensional emotion:** valence (positive/negative), arousal (calm/excited)
- **Categorical emotion:** happy, sad, angry, neutral, etc. (optional)

**Best for:** Analysis requiring audio features beyond text (emotion detection, prosody analysis)

---

## Advanced Examples

For more complex use cases, see:
- `complete_workflow.py` - Query and analyze enriched transcripts
- `prosody_quickstart.py` - Direct prosody feature extraction
- `query_audio_features.py` - Advanced querying of audio features
- `emotion_integration.py` - Emotion detection workflows

---

## Directory Structure

Your project should follow this structure:

```
project_root/
├── raw_audio/          # Place your audio files here (MP3, WAV, M4A, etc.)
├── input_audio/        # Auto-created: normalized WAV files
├── whisper_json/       # Auto-created: JSON transcripts (with enrichment)
└── transcripts/        # Auto-created: TXT and SRT outputs
```

**Supported audio formats:** MP3, WAV, M4A, FLAC, OGG, OPUS (anything ffmpeg can read)

---

## Configuration Reference

### TranscriptionConfig (Stage 1)

```python
from transcription import TranscriptionConfig

config = TranscriptionConfig(
    model="large-v3",           # Model: tiny, base, small, medium, large-v3
    device="cuda",              # Device: cuda or cpu
    compute_type="float16",     # float16 (GPU) or int8 (CPU)
    language="en",              # Language code or None for auto-detect
    task="transcribe",          # transcribe or translate
    beam_size=5,                # Beam search width (1-10)
    vad_min_silence_ms=500,     # VAD silence threshold (ms)
    skip_existing_json=True,    # Skip already-transcribed files
)
```

### EnrichmentConfig (Stage 2)

```python
from transcription import EnrichmentConfig

config = EnrichmentConfig(
    enable_prosody=True,                  # Extract prosody features
    enable_emotion=True,                  # Extract dimensional emotion
    enable_categorical_emotion=False,     # Extract categorical emotion
    device="cpu",                         # Device: cpu or cuda
    skip_existing=True,                   # Skip already-enriched files
)
```

---

## API Functions

```python
from transcription import (
    transcribe_directory,
    transcribe_file,
    enrich_directory,
    enrich_transcript,
    load_transcript,
    save_transcript,
)

# Transcribe all files in a directory
transcripts = transcribe_directory("/path/to/project", config)

# Transcribe a single file
transcript = transcribe_file("audio.mp3", "/path/to/output", config)

# Enrich all transcripts in a directory
enriched = enrich_directory("/path/to/project", enrich_config)

# Enrich a single transcript
transcript = load_transcript("transcript.json")
enriched = enrich_transcript(transcript, "audio.wav", enrich_config)
save_transcript(enriched, "enriched.json")
```

---

## Transcript Data Model

```python
# Access transcript data
transcript.file_name      # "interview.wav"
transcript.language       # "en"
transcript.segments       # List of Segment objects
transcript.meta           # Metadata (model, device, etc.)

# Access segment data
segment.id                # 0, 1, 2, ...
segment.start             # 0.0 (seconds)
segment.end               # 5.2 (seconds)
segment.text              # "Hello, world!"
segment.audio_state       # Optional: enriched features

# Access enriched features (if enriched)
segment.audio_state["pitch"]     # Pitch features
segment.audio_state["energy"]    # Energy features
segment.audio_state["rate"]      # Speech rate features
segment.audio_state["pauses"]    # Pause information
segment.audio_state["emotion"]   # Emotion features
```

---

## Tips & Best Practices

### Performance Optimization
- Use `large-v3` model for best quality
- Use GPU (`device="cuda"`) for 10-20x speedup
- Use smaller models (`medium`, `small`) for faster processing
- Set `skip_existing_json=True` to avoid re-transcribing

### Language Detection
- Let Whisper auto-detect language (don't set `language`)
- Provide language hint (`language="en"`) for better accuracy
- Use `task="translate"` to translate to English

### Enrichment
- Run enrichment on CPU (`device="cpu"`) to free GPU for transcription
- Enable categorical emotion only if needed (slower)
- Skip enrichment if you only need text transcripts

### Batch Processing
1. Transcribe all files first: `--transcribe`
2. Review transcripts
3. Enrich later if needed: `--enrich`

---

## Requirements

Install the package with enrichment dependencies:

```bash
# Basic installation
pip install -e .

# With enrichment support
pip install -e ".[enrichment]"

# Or manually install enrichment dependencies
pip install praat-parselmouth librosa transformers torch
```

---

## Troubleshooting

### "No audio files found"
- Ensure files are in `project_root/raw_audio/`
- Check file extensions (.mp3, .wav, .m4a, etc.)

### "CUDA out of memory"
- Use smaller model (`--model medium`)
- Use CPU (`--device cpu`)
- Process files one at a time

### "Model not found"
- First run downloads the model from Hugging Face
- Check internet connection
- Models are cached in `~/.cache/huggingface/`

### Import errors for enrichment
- Install enrichment dependencies: `pip install -e ".[enrichment]"`
- Check that `transformers` and `torch` are installed

---

## Next Steps

After running these examples:
1. Load transcripts: `transcript = load_transcript("output.json")`
2. Analyze features with `complete_workflow.py`
3. Export to CSV with `query_audio_features.py`
4. Build custom analysis pipelines

For more details, see the main documentation in `docs/`.
