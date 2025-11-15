# Local Transcription Pipeline (ffmpeg + faster-whisper)

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
