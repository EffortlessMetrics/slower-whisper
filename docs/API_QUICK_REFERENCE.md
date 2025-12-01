# API Quick Reference

## Import

```python
from transcription import (
    # Functions
    transcribe_directory,
    transcribe_file,
    enrich_directory,
    enrich_transcript,
    load_transcript,
    save_transcript,
    # Config
    TranscriptionConfig,
    EnrichmentConfig,
    # Models
    Transcript,
    Segment,
)
```

## Transcription

### Directory (Batch)

```python
config = TranscriptionConfig(
    model="large-v3",
    language="en",
    device="cuda"
)

transcripts = transcribe_directory("/path/to/project", config)
```

### Single File

```python
config = TranscriptionConfig(model="base", language="en")

transcript = transcribe_file(
    audio_path="interview.mp3",
    root="/path/to/project",
    config=config
)
```

## Enrichment

### Directory (Batch)

```python
config = EnrichmentConfig(
    enable_prosody=True,
    enable_emotion=True,
    device="cpu"
)

enriched = enrich_directory("/path/to/project", config)
```

### Single Transcript

```python
config = EnrichmentConfig(enable_prosody=True)

enriched = enrich_transcript(
    transcript=transcript,
    audio_path="audio.wav",
    config=config
)
```

## I/O

```python
# Load
transcript = load_transcript("output.json")

# Save
save_transcript(transcript, "output.json")
```

## Configuration

### Configuration Precedence

Settings are loaded in order of priority:

```
1. CLI flags (highest priority)
   ↓
2. Config file (--config or --enrich-config)
   ↓
3. Environment variables (SLOWER_WHISPER_*)
   ↓
4. Defaults (lowest priority)
```

### TranscriptionConfig

```python
TranscriptionConfig(
    model="large-v3",          # Whisper model name
    device="cuda",             # "cuda" or "cpu"
    compute_type=None,         # Precision (None = auto: float16 on CUDA, int8 on CPU)
    language=None,             # None = auto-detect
    task="transcribe",         # or "translate"
    skip_existing_json=True,   # Skip already transcribed
    vad_min_silence_ms=500,    # VAD silence threshold
    beam_size=5,               # Beam search size
)
```

**Field Details:**

| Field | Type | Default | Description | CLI Flag | Env Var |
|-------|------|---------|-------------|----------|---------|
| `model` | `str` | `"large-v3"` | Whisper model: tiny, base, small, medium, large, large-v2, large-v3 | `--model` | `SLOWER_WHISPER_MODEL` |
| `device` | `str` | `"cuda"` | Computation device: "cuda" or "cpu" | `--device` | `SLOWER_WHISPER_DEVICE` |
| `compute_type` | `str \| None` | `None` (auto) | Precision: auto picks float16 on CUDA, int8 on CPU. Supported: float16, float32, int16, int8, int8_float16, int8_float32 | `--compute-type` | `SLOWER_WHISPER_COMPUTE_TYPE` |
| `language` | `str \| None` | `None` | Language code (e.g., "en", "es") or None for auto-detect | `--language` | `SLOWER_WHISPER_LANGUAGE` |
| `task` | `str` | `"transcribe"` | Whisper task: "transcribe" or "translate" | `--task` | `SLOWER_WHISPER_TASK` |
| `skip_existing_json` | `bool` | `True` | Skip files with existing JSON output | `--skip-existing-json` | `SLOWER_WHISPER_SKIP_EXISTING_JSON` |
| `vad_min_silence_ms` | `int` | `500` | VAD silence threshold in milliseconds (100-2000) | `--vad-min-silence-ms` | `SLOWER_WHISPER_VAD_MIN_SILENCE_MS` |
| `beam_size` | `int` | `5` | Beam search size (1-10, higher = more accurate but slower) | `--beam-size` | `SLOWER_WHISPER_BEAM_SIZE` |
| `enable_diarization` | `bool` | `False` | Enable speaker diarization (pyannote.audio) | `--enable-diarization` | `SLOWER_WHISPER_ENABLE_DIARIZATION` |
| `diarization_device` | `str` | `"auto"` | Device for diarization: "cuda", "cpu", or "auto" | `--diarization-device` | `SLOWER_WHISPER_DIARIZATION_DEVICE` |
| `min_speakers` | `int \| None` | `None` | Minimum expected speakers (hint for diarization model) | `--min-speakers` | `SLOWER_WHISPER_MIN_SPEAKERS` |
| `max_speakers` | `int \| None` | `None` | Maximum expected speakers (hint for diarization model) | `--max-speakers` | `SLOWER_WHISPER_MAX_SPEAKERS` |
| `overlap_threshold` | `float` | `0.3` | Minimum overlap ratio to assign a speaker to a segment | `--overlap-threshold` | `SLOWER_WHISPER_OVERLAP_THRESHOLD` |
| _backend mode_ | `str` | `"auto"` | pyannote backend selector: `auto`, `stub` (fake), `missing` (simulate dep error) | _env only_ | `SLOWER_WHISPER_PYANNOTE_MODE` |
| _hf token_ | `str` | _none_ | Hugging Face token for pyannote backend (required unless forcing stub/missing) | _env only_ | `HF_TOKEN` |

**Model Sizes:**

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~75MB | Fastest | Lowest | Quick testing, resource-constrained |
| `base` | ~150MB | Very fast | Basic | Development, low-resource environments |
| `small` | ~500MB | Fast | Good | Balanced speed/quality |
| `medium` | ~1.5GB | Medium | Better | Higher quality needs |
| `large-v2` | ~3GB | Slow | Best | Production quality |
| `large-v3` | ~3GB | Slow | Best | Latest, recommended for production |

**Compute Types:**

| Type | Precision | Speed | Quality | GPU Required |
|------|-----------|-------|---------|--------------|
| `float32` | Full | Slowest | Best | No |
| `float16` | Half | Fast | Excellent | Yes (recommended) |
| `int8` | Quantized | Fastest | Good | No |
| `int8_float16` | Mixed | Very fast | Very good | Yes |
| `int16` | 16-bit | Medium | Good | No |
| `int8_float32` | Mixed | Fast | Good | No |

**Loading Methods:**

```python
# From file
config = TranscriptionConfig.from_file("config.json")

# From environment
config = TranscriptionConfig.from_env()

# Direct construction
config = TranscriptionConfig(model="base", language="en")
```

### EnrichmentConfig

```python
EnrichmentConfig(
    skip_existing=True,               # Skip already enriched
    enable_prosody=True,              # Pitch, energy, rate
    enable_emotion=True,              # Dimensional emotion
    enable_categorical_emotion=False, # Categorical (slower)
    device="cpu",                     # "cpu" or "cuda"
    dimensional_model_name=None,      # Override dimensional model
    categorical_model_name=None,      # Override categorical model
)
```

**Field Details:**

| Field | Type | Default | Description | CLI Flag | Env Var |
|-------|------|---------|-------------|----------|---------|
| `skip_existing` | `bool` | `True` | Skip segments with existing audio_state | `--skip-existing` | `SLOWER_WHISPER_ENRICH_SKIP_EXISTING` |
| `enable_prosody` | `bool` | `True` | Extract pitch, energy, speech rate, pauses | `--enable-prosody` | `SLOWER_WHISPER_ENRICH_ENABLE_PROSODY` |
| `enable_emotion` | `bool` | `True` | Extract dimensional emotion (valence/arousal/dominance) | `--enable-emotion` | `SLOWER_WHISPER_ENRICH_ENABLE_EMOTION` |
| `enable_categorical_emotion` | `bool` | `False` | Extract categorical emotions (angry, happy, etc.) - slower | `--enable-categorical-emotion` | `SLOWER_WHISPER_ENRICH_ENABLE_CATEGORICAL_EMOTION` |
| `device` | `str` | `"cpu"` | Device for emotion models: "cpu" or "cuda" | `--device` | `SLOWER_WHISPER_ENRICH_DEVICE` |
| `dimensional_model_name` | `str \| None` | `None` | HuggingFace model name for dimensional emotion | N/A | `SLOWER_WHISPER_ENRICH_DIMENSIONAL_MODEL_NAME` |
| `categorical_model_name` | `str \| None` | `None` | HuggingFace model name for categorical emotion | N/A | `SLOWER_WHISPER_ENRICH_CATEGORICAL_MODEL_NAME` |

**Feature Comparison:**

| Feature | Dependencies | Model Size | Speed | What It Extracts |
|---------|--------------|------------|-------|------------------|
| Prosody | librosa, parselmouth | ~50MB | Fast | Pitch (Hz), energy (dB), rate (syllables/sec), pauses |
| Dimensional Emotion | torch, transformers | ~1.5GB | Medium | Valence (positive/negative), arousal (calm/excited), dominance |
| Categorical Emotion | torch, transformers | ~1.5GB | Slower | Emotion labels: angry, happy, sad, frustrated, etc. |

**Loading Methods:**

```python
# From file
config = EnrichmentConfig.from_file("enrich_config.json")

# From environment
config = EnrichmentConfig.from_env()

# Direct construction
config = EnrichmentConfig(enable_prosody=True, device="cuda")
```

### Configuration File Examples

**Transcription JSON:**
```json
{
  "model": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "language": "en",
  "task": "transcribe",
  "skip_existing_json": true,
  "vad_min_silence_ms": 500,
  "beam_size": 5
}
```

**Transcription JSON (with diarization):**
```json
{
  "model": "large-v3",
  "enable_diarization": true,
  "diarization_device": "auto",
  "min_speakers": 2,
  "max_speakers": 4,
  "overlap_threshold": 0.3
}
```

**Enrichment JSON:**
```json
{
  "skip_existing": true,
  "enable_prosody": true,
  "enable_emotion": true,
  "enable_categorical_emotion": false,
  "device": "cpu",
  "dimensional_model_name": null,
  "categorical_model_name": null
}
```

See [examples/config_examples/](examples/config_examples/) for more examples.

## Access Results

### Transcript Structure

```python
transcript.file_name    # str
transcript.language     # str
transcript.segments     # list[Segment]
transcript.meta         # dict | None
```

### Segment Structure

```python
segment.id              # int
segment.start           # float (seconds)
segment.end             # float (seconds)
segment.text            # str
segment.speaker         # str | None
segment.audio_state     # dict | None
```

### Audio State (if enriched)

```python
if segment.audio_state:
    # Compact rendering
    print(segment.audio_state["rendering"])
    # "[audio: high pitch, loud volume, fast speech]"

    # Prosody features
    prosody = segment.audio_state["prosody"]
    print(prosody["pitch"]["level"])      # "high", "low", "neutral"
    print(prosody["energy"]["level"])     # "loud", "quiet", "normal"
    print(prosody["rate"]["level"])       # "fast", "slow", "normal"

    # Emotion features
    emotion = segment.audio_state["emotion"]
    print(emotion["valence"]["level"])    # "positive", "negative", "neutral"
    print(emotion["arousal"]["level"])    # "high", "low", "medium"
```

## CLI Quick Reference

Use the unified `slower-whisper` entry point with subcommands.

### Transcribe

```bash
uv run slower-whisper transcribe [OPTIONS]
```

| Option | Default | Notes |
|--------|---------|-------|
| `--root PATH` | `.` | Project root with `raw_audio/` |
| `--config FILE` | `None` | Merge order: CLI > file > env > defaults |
| `--model NAME` | `large-v3` | Whisper model |
| `--device {cuda,cpu}` | `cuda` | Auto-fallbacks to CPU if GPU load fails |
| `--compute-type TYPE` | auto (`float16` on CUDA, `int8` on CPU) | Override faster-whisper precision |
| `--language CODE` | auto | Force language |
| `--task {transcribe,translate}` | `transcribe` | Translate outputs English |
| `--vad-min-silence-ms INT` | `500` | VAD split threshold |
| `--beam-size INT` | `5` | Beam search size |
| `--skip-existing-json / --no-skip-existing-json` | `True` | Reuse existing JSON |
| `--enable-diarization` | `False` | Experimental speaker diarization (pyannote) |
| `--diarization-device {auto,cuda,cpu}` | `auto` | Device for diarization |
| `--min-speakers INT` | `None` | Lower bound hint |
| `--max-speakers INT` | `None` | Upper bound hint |
| `--overlap-threshold FLOAT` | `0.3` | Min overlap to assign a speaker |

Quick starts:

```bash
uv run slower-whisper transcribe --language en
uv run slower-whisper transcribe --enable-diarization --min-speakers 2 --max-speakers 4
```

### Enrich

```bash
uv run slower-whisper enrich [OPTIONS]
```

| Option | Default | Notes |
|--------|---------|-------|
| `--root PATH` | `.` | Project root with `whisper_json/` + `input_audio/` |
| `--enrich-config FILE` | `None` | Merge order: CLI > file > env > defaults |
| `--skip-existing / --no-skip-existing` | `True` | Skip segments with `audio_state` |
| `--enable-prosody / --no-enable-prosody` | `True` | Pitch/energy/rate/pauses |
| `--enable-emotion / --no-enable-emotion` | `True` | Dimensional emotion |
| `--enable-categorical-emotion / --no-enable-categorical-emotion` | `False` | Categorical emotion (slower) |
| `--device {cpu,cuda}` | `cpu` | Device for emotion models |

Quick starts:

```bash
uv run slower-whisper enrich
uv run slower-whisper enrich --enable-categorical-emotion --device cuda
uv run slower-whisper enrich --no-enable-emotion   # prosody only
```

## Common Patterns

### Full Pipeline (Transcribe + Enrich)

```python
# Stage 1: Transcribe
trans_cfg = TranscriptionConfig(model="large-v3", language="en")
transcripts = transcribe_directory("/data/project", trans_cfg)

# Stage 2: Enrich
enrich_cfg = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
enriched = enrich_directory("/data/project", enrich_cfg)
```

### Custom Processing

```python
# Load
transcript = load_transcript("transcript.json")

# Modify
for segment in transcript.segments:
    if segment.audio_state:
        # Custom logic based on audio features
        if segment.audio_state["prosody"]["pitch"]["level"] == "high":
            segment.text = segment.text.upper()

# Save
save_transcript(transcript, "modified.json")
```

### Error Handling

```python
try:
    transcript = transcribe_file("audio.mp3", "/project", config)
except FileNotFoundError as e:
    print(f"Audio file not found: {e}")
except Exception as e:
    print(f"Transcription failed: {e}")
```

## Directory Layout

```
project_root/
├── raw_audio/           # Input: Original audio files
├── input_audio/         # Generated: Normalized 16kHz WAVs
├── whisper_json/        # Generated: JSON transcripts
└── transcripts/         # Generated: TXT and SRT files
```

## Tips

1. **Use defaults:** Most parameters have sensible defaults
2. **Check audio_state:** Always check if `audio_state` is not `None` before accessing
3. **Lazy imports:** Enrichment features loaded on-demand (won't fail if not installed)
4. **Batch processing:** Use `_directory()` functions for multiple files
5. **Progressive enhancement:** Transcribe first, enrich later as needed
