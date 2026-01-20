# Quickstart Guide

Get started with slower-whisper in 5 minutes. This guide covers both Stage 1 (transcription) and Stage 2 (audio enrichment).

---

## Prerequisites

- **Python 3.11+**
- **ffmpeg** installed and on PATH
- **NVIDIA GPU** (recommended but optional)
  - Required for fast transcription and emotion recognition
  - CPU fallback available

---

## Installation

### Step 1: Install ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows (with Chocolatey):**
```powershell
choco install ffmpeg -y
```

### Step 2: Set Up Python Environment

**Option 1: Using Nix (recommended)**
```bash
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper
nix develop
uv sync --extra full --extra dev
```

**Option 2: Using uv (fast)**
```bash
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper

# Install system deps (ffmpeg, libsndfile) via apt/brew/choco
uv sync --extra full --extra dev   # contributors
# or
uv sync --extra full               # runtime only
```

**Option 3: Using pip (traditional)**
```bash
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate on Windows

pip install -e ".[full,dev]"
```

### Step 3: Verify Installation

```bash
# Check ffmpeg
ffmpeg -version

# Check slower-whisper CLI
slower-whisper --version

# Optional: Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Stage 1: Transcription

### Your First Transcription

**Step 1: Prepare Audio**
```bash
# Create directory structure (automatic on first run)
mkdir -p raw_audio

# Add your audio file
cp /path/to/your/audio.mp3 raw_audio/
# Supported formats: mp3, m4a, wav, flac, ogg, etc.
```

**Step 2: Run Transcription**
```bash
# Basic transcription (auto-detect language)
slower-whisper transcribe --root .

# Or specify language
slower-whisper transcribe --root . --language en
```

**Step 3: Check Output**
```bash
# View text transcript
cat transcripts/audio.txt

# View structured JSON
cat whisper_json/audio.json | python -m json.tool

# View SRT subtitles
cat transcripts/audio.srt
```

### Understanding the Output

**Text output (transcripts/audio.txt):**
```
[0.0s -> 4.2s] Okay, let's get started with today's agenda.
[4.2s -> 8.5s] First item is the quarterly review.
```

**JSON output (whisper_json/audio.json):**
```json
{
  "schema_version": 2,
  "file": "audio.wav",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Okay, let's get started with today's agenda.",
      "speaker": null,
      "tone": null,
      "audio_state": null
    }
  ]
}
```

**SRT output (transcripts/audio.srt):**
```
1
00:00:00,000 --> 00:00:04,200
Okay, let's get started with today's agenda.

2
00:00:04,200 --> 00:00:08,500
First item is the quarterly review.
```

---

## Stage 2: Audio Enrichment

### Adding Audio Features

After transcription, optionally enrich transcripts with prosody and emotion features.

**Step 1: Install Enrichment Dependencies**

All dependencies are in `requirements.txt`. If you installed everything in the Installation step, you're ready. Otherwise:

```bash
# Using uv
uv pip install soundfile librosa praat-parselmouth transformers torch

# Using pip
pip install soundfile librosa praat-parselmouth transformers torch
```

**Step 2: Run Audio Enrichment**

```bash
# Fast mode (prosody only, ~30 seconds per minute of audio)
slower-whisper enrich --root . --no-enable-emotion

# Standard mode (prosody + dimensional emotion, ~2-3 minutes per minute of audio)
slower-whisper enrich --root .

# Full mode (prosody + dimensional + categorical emotion, ~5-10 minutes per minute of audio)
slower-whisper enrich --root . --enable-categorical-emotion
```

**Step 3: View Enriched Output**

```bash
# Check first segment
cat whisper_json/audio.json | python -m json.tool | head -50
```

### Understanding Enriched Output

After enrichment, each segment includes `audio_state`:

```json
{
  "id": 0,
  "start": 0.0,
  "end": 4.2,
  "text": "Okay, let's get started!",
  "audio_state": {
    "prosody": {
      "pitch": {
        "level": "high",
        "mean_hz": 245.3,
        "contour": "rising"
      },
      "energy": {
        "level": "loud",
        "db_rms": -8.2
      },
      "rate": {
        "level": "fast",
        "syllables_per_sec": 6.3
      },
      "pauses": {
        "count": 0,
        "density": "sparse"
      }
    },
    "emotion": {
      "valence": {"level": "positive", "score": 0.72},
      "arousal": {"level": "high", "score": 0.68}
    },
    "rendering": "[audio: high pitch, loud volume, fast speech, excited tone]"
  }
}
```

**Key Fields:**
- **prosody** - Pitch, energy (loudness), speech rate, pauses
- **emotion** - Valence (positive/negative), arousal (energy level), optionally categorical (happy, sad, etc.)
- **rendering** - LLM-friendly text annotation summarizing acoustic features

---

## Analyzing Results

### Using Example Scripts

```bash
# Comprehensive analysis
python examples/complete_workflow.py whisper_json/audio.json

# Export to CSV for spreadsheet analysis
python examples/complete_workflow.py whisper_json/audio.json --output-csv analysis.csv

# Find excited moments
python examples/query_audio_features.py whisper_json/audio.json --excited

# Find calm moments
python examples/query_audio_features.py whisper_json/audio.json --calm

# Full statistical summary
python examples/query_audio_features.py whisper_json/audio.json --summary
```

### Example Analysis Output

```
================================================================================
TRANSCRIPT ANALYSIS REPORT
================================================================================

File: audio.wav
Language: en
Total segments: 45
Total duration: 892.5s

--- Pitch Analysis ---
  Range: 120.5 - 280.3 Hz
  Mean: 185.2 Hz
  Distribution: {'high': 18, 'medium': 21, 'low': 6}

--- Emotion Distribution ---
  neutral        : ████████████        28 ( 62.2%)
  excited        : ███████              12 ( 26.7%)
  happy          : ███                   3 (  6.7%)
  concerned      : ▌                     2 (  4.4%)

--- Excited Moments (12) ---
  [  45.2s] Yeah absolutely! This is a fantastic opportunity...
  [ 102.5s] I'm really excited about this direction...
  ... and 10 more
```

---

## Common Workflows

### Workflow 1: Quick Transcription

```bash
# Transcribe with defaults
slower-whisper transcribe --root . --language en

# That's it! Check transcripts/ directory
```

### Workflow 2: Full Pipeline

```bash
# Stage 1: Transcribe
slower-whisper transcribe --root . --language en

# Stage 2: Enrich with prosody only (fast)
slower-whisper enrich --root . --no-enable-emotion

# Analyze
python examples/complete_workflow.py whisper_json/your_audio.json
```

### Workflow 3: Batch Processing

```bash
# Place multiple audio files in raw_audio/
# Transcribe all
slower-whisper transcribe --root . --language en

# Enrich all (skip already enriched)
slower-whisper enrich --root . --skip-existing

# Analyze each
for file in whisper_json/*.json; do
  python examples/query_audio_features.py "$file" --summary > "${file%.json}_report.txt"
done
```

### Workflow 4: Incremental Processing

```bash
# Day 1: Transcribe everything
slower-whisper transcribe --root . --language en

# Day 2: Add new audio files and skip existing
slower-whisper transcribe --root . --language en --skip-existing-json

# Day 3: Enrich everything (skip already enriched)
slower-whisper enrich --root . --skip-existing
```

---

## Command Reference

### Stage 1: slower-whisper transcribe

```bash
# Basic options
slower-whisper transcribe
  --root DIR              # Root directory (default: current)
  --model MODEL           # Model size: tiny, base, small, medium, large-v3
  --language LANG         # Language code (e.g., en, es, fr) or auto-detect
  --device DEVICE         # auto, cuda, or cpu
  --compute-type TYPE     # float16, float32, int8, int8_float16 (auto by default)
  --skip-existing-json    # Skip files with existing JSON output (default: true)
  --progress              # Show progress indicator

# Examples
slower-whisper transcribe --root . --model medium --language en
slower-whisper transcribe --root . --device cpu
slower-whisper transcribe --root /path/to/project --progress
```

### Stage 2: slower-whisper enrich

```bash
# Basic options
slower-whisper enrich
  --root DIR                    # Root directory (default: current)
  --skip-existing               # Skip files with existing audio_state (default: true)
  --device DEVICE               # auto, cuda, or cpu
  --enable-prosody              # Extract prosody (default: true)
  --no-enable-prosody           # Skip prosody
  --enable-emotion              # Extract dimensional emotion (default: true)
  --no-enable-emotion           # Skip emotion
  --enable-categorical-emotion  # Extract categorical emotion (default: false)
  --enable-semantics            # Run keyword-based semantic annotation
  --progress                    # Show progress indicator

# Examples
slower-whisper enrich --root . --no-enable-emotion   # Prosody only
slower-whisper enrich --root . --enable-categorical-emotion  # Full analysis
slower-whisper enrich --root . --device cpu          # CPU mode
```

---

## Performance Tips

### Transcription Speed
- **Use GPU:** 5-10x faster than CPU
- **Choose model wisely:**
  - `base`: Fast, decent quality
  - `medium`: Good balance
  - `large-v3`: Best quality, slowest
- **VAD threshold:** Adjust `--vad-min-silence-ms` for better segmentation

### Enrichment Speed
- **Prosody only:** Very fast (~0.1x realtime on CPU)
- **Dimensional emotion:** Moderate (faster on GPU)
- **Categorical emotion:** Slower (much faster on GPU)
- **Skip existing:** Use `--skip-existing` to resume interrupted processing

### Memory Optimization
- **Large files:** Process incrementally with `--skip-existing`
- **Limited VRAM:** Use smaller Whisper model or `--device cpu`
- **Many files:** Process in batches

---

## Troubleshooting

### "Command 'ffmpeg' not found"
**Solution:** Install ffmpeg (see Installation section above)

### "CUDA out of memory"
**Solution:**
```bash
# Use smaller model
slower-whisper transcribe --root . --model medium

# Or use CPU
slower-whisper transcribe --root . --device cpu
slower-whisper enrich --root . --device cpu
```

### "No module named 'transformers'"
**Solution:**
```bash
# Install enrichment dependencies
uv pip install transformers torch
# or
pip install transformers torch
```

### "Model download fails"
**Solution:**
- Check internet connection
- Models download on first use (Whisper: ~3GB, Emotion: ~2.5GB total)
- If behind proxy, configure: `export HF_HOME=/custom/cache`

### "Audio segments too short for emotion"
**Note:** This is expected for very short utterances (< 0.5s)
- Emotion models work best on segments > 0.5 seconds
- Prosody extraction works on any length
- This doesn't affect transcription quality

### "Results seem inaccurate"
**Solutions:**
- **Transcription:** Try different model size or language setting
- **Prosody:** Check audio quality (low background noise helps)
- **Emotion:** Accuracy varies by domain; pre-trained models may not match your specific use case

---

## Next Steps

### Learn More
- **[Audio Enrichment Guide](AUDIO_ENRICHMENT.md)** - Comprehensive Stage 2 documentation
- **[Prosody Details](PROSODY.md)** - Deep dive into prosody features
- **[Prosody Reference](PROSODY_REFERENCE.md)** - Quick feature lookup
- **[Examples](../examples/)** - Example analysis scripts

### Advanced Usage
- Speaker diarization (future feature)
- Custom emotion models
- Integration with RAG pipelines
- LLM context enrichment

### Contributing
See **[CONTRIBUTING.md](../CONTRIBUTING.md)** for:
- Development environment setup
- Running tests
- Adding new features

---

## Summary

**Minimal workflow:**
```bash
# 1. Install
uv sync --extra full

# 2. Transcribe
slower-whisper transcribe --root . --language en

# 3. Done! Check transcripts/ and whisper_json/
```

**Full workflow:**
```bash
# 1. Install
uv sync --extra full

# 2. Transcribe
slower-whisper transcribe --root . --language en

# 3. Enrich
slower-whisper enrich --root .

# 4. Analyze
python examples/complete_workflow.py whisper_json/your_audio.json

# 5. Done! Check outputs
```

---

**Ready to go!** Place audio in `raw_audio/` and run `slower-whisper transcribe --root .`.

For questions or issues, see the main **[README](../README.md)** or open an issue.
