# slower-whisper (ffmpeg + faster-whisper)

This project provides a small, structured codebase for running a fully local
transcription pipeline on a machine with an NVIDIA GPU.

It:

- Normalizes audio in `raw_audio/` to 16 kHz mono WAV in `input_audio/` using `ffmpeg`.
- Transcribes using Whisper via `faster-whisper` on CUDA.
- Writes:
  - `transcripts/<name>.txt` – timestamped text.
  - `transcripts/<name>.srt` – subtitles.
  - `whisper_json/<name>.json` – structured JSON for analysis.

JSON is the canonical output format and is designed to be extended later
with tone, speaker, and other annotations.

## Requirements

- Windows (or any OS with Python and ffmpeg).
- Python 3.9+.
- NVIDIA GPU with a recent CUDA-capable driver (for GPU acceleration).
- `ffmpeg` on PATH.
- Python package: `faster-whisper`.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install ffmpeg on Windows (PowerShell, elevated):

```powershell
choco install ffmpeg -y
```

## Directory layout

By default, the pipeline expects the following structure under a root folder:

```text
root/
  transcribe_pipeline.py
  transcription/
    ...
  raw_audio/
  input_audio/
  transcripts/
  whisper_json/
```

The code will create the directories if they do not exist.

- `raw_audio/` – place your original audio here (`.mp3`, `.m4a`, `.wav`, etc.).
- `input_audio/` – normalized 16 kHz mono WAVs (generated).
- `transcripts/` – `.txt` and `.srt` outputs (generated).
- `whisper_json/` – `.json` structured transcripts (generated).

## Usage

From the root directory:

```powershell
python transcribe_pipeline.py
```

This uses defaults:

- root: current directory
- model: `large-v3`
- device: `cuda`
- compute type: `float16`
- VAD min silence: 500 ms
- language: auto-detect
- task: transcribe

You can override these with CLI options:

```powershell
# Force English and skip files already transcribed
python transcribe_pipeline.py --language en --skip-existing-json

# Use a lighter model and quantized weights
python transcribe_pipeline.py --model medium --compute-type int8_float16
```

To skip files that have already been transcribed (i.e. where a JSON output
already exists), add `--skip-existing-json`:

```powershell
python transcribe_pipeline.py --skip-existing-json
```

The pipeline prints per-file progress, basic timing statistics (per-file and
overall), and a summary at the end.

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
  "schema_version": 2,
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
      "tone": null,
      "audio_state": null
    }
  ]
}
```

This schema is stable and is intended to be the basis for future tooling:

- Tone tagging: populate `tone`.
- Speaker diarization: populate `speaker`.
- Audio enrichment: populate `audio_state` with prosody and emotion features.
- Search and analysis: operate over `segments[]`.
- Run-level analysis and reproducibility: read `meta`.

## Audio Feature Enrichment

The project supports optional **Stage 2** audio enrichment that extracts linguistic and emotional features
directly from audio.

### Two-Stage Pipeline

**Stage 1: Transcription** (required)

- Normalize audio to 16 kHz mono WAV
- Transcribe using faster-whisper on NVIDIA GPU
- Output: JSON transcripts with segments

**Stage 2: Audio Enrichment** (optional)

- Extract prosodic features (pitch, energy, speech rate, pauses)
- Extract emotional features (dimensional: valence/arousal/dominance; categorical: emotions)
- Populate `audio_state` field in transcript segments

### What Features Are Extracted

**Prosody Features:**

- **Pitch:** mean frequency (Hz), standard deviation, contour (rising/falling/flat)
- **Energy:** RMS level (dB), variation coefficient
- **Speech Rate:** syllables per second, words per second
- **Pauses:** count, longest duration, density per second

**Emotion Features:**

- **Dimensional:** valence (negative to positive), arousal (calm to excited), dominance (submissive to dominant)
- **Categorical:** primary emotion classification (angry, happy, sad, frustrated, etc.) with confidence scores

All features are automatically categorized into meaningful levels (e.g., "high pitch", "fast speech", "neutral sentiment").

### Usage Example

After transcription, enrich with audio features:

```bash
# Enrich existing transcript with emotion analysis
python examples/emotion_integration.py enrich whisper_json/meeting1.json input_audio/meeting1.wav

# View enriched transcript
cat whisper_json/meeting1.json  # Now includes audio_state in segments

# Analyze emotions across the file
python examples/emotion_integration.py analyze whisper_json/meeting1.json
```

Output JSON will have segments like:

```json
{
  "id": 0,
  "start": 0.0,
  "end": 4.2,
  "text": "Okay, let's get started with today's agenda.",
  "audio_state": {
    "prosody": {
      "pitch": {
        "level": "high",
        "mean_hz": 245.3,
        "std_hz": 32.1,
        "contour": "rising"
      },
      "energy": {
        "level": "loud",
        "db_rms": -8.2
      },
      "rate": {
        "level": "normal",
        "syllables_per_sec": 5.3
      }
    },
    "emotion": {
      "dimensional": {
        "valence": {"level": "positive", "score": 0.72},
        "arousal": {"level": "high", "score": 0.68}
      },
      "categorical": {
        "primary": "happy",
        "confidence": 0.89
      }
    }
  }
}
```

### Installation & Setup

The base pipeline requires only `faster-whisper` and audio tools. Audio enrichment is optional:

```bash
# Install base dependencies (Stage 1: transcription)
pip install faster-whisper>=1.0.0

# Install audio enrichment dependencies (Stage 2: optional)
pip install -r requirements.txt
```

See [docs/AUDIO_ENRICHMENT.md](docs/AUDIO_ENRICHMENT.md) for detailed setup instructions, including model downloads for emotion recognition.

## Extending

The code is structured into modules:

- `transcription.models` – core dataclasses (`Segment`, `Transcript`).
- `transcription.config` – configuration classes (`Paths`, `AsrConfig`, `AppConfig`).
- `transcription.audio_io` – ffmpeg-based normalization.
- `transcription.asr_engine` – faster-whisper wrapper.
- `transcription.writers` – JSON/TXT/SRT writers.
- `transcription.pipeline` – orchestration.
- `transcription.cli` – CLI entrypoint.
- `transcription.enrich` – placeholders for tone and speaker enrichment.

To add tone tagging, diarization, or other analysis, write separate modules
(or expand `transcription.enrich`) that read and modify the JSON or
`Transcript` objects without changing the core pipeline.

## Tests (optional)

A minimal `tests/` directory can be added to validate the JSON schema and
SRT formatting. For example:

- Test that `write_json` produces the documented structure for a simple
  `Transcript`.
- Test that the SRT timestamp formatter produces valid strings.

These are not required for running the pipeline but are useful if you start
evolving the code further.
