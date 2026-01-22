# CLI Reference

This document describes the unified `slower-whisper` CLI and its subcommands.

The CLI now uses a single entry point with subcommands:

- `slower-whisper transcribe` — Stage 1 transcription
- `slower-whisper enrich` — Stage 2 enrichment (prosody/emotion)
- `slower-whisper cache` — Inspect or clear model caches
- `slower-whisper samples` — List/download/copy/generate sample datasets

The legacy `slower-whisper-enrich` script still exists for backward compatibility, but the unified CLI is the recommended interface.

---

## Installation & Entry Points

```toml
[project.scripts]
slower-whisper = "transcription.cli:main"
```

Install with uv (recommended):

```bash
# Transcription-only deps
uv sync

# Full stack (transcribe + enrich)
uv sync --extra full

# Development helpers
uv sync --extra dev
```

Or with pip:

```bash
pip install -e .            # base
pip install -e ".[full]"    # full stack
pip install -e ".[dev]"     # dev tools
```

Verify:

```bash
which slower-whisper
slower-whisper --help
```

---

## Command: `slower-whisper transcribe`

Transcribe audio under a project root (creates/uses `raw_audio/`, `input_audio/`, `whisper_json/`, `transcripts/`).

```bash
slower-whisper transcribe [OPTIONS]
```

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `--root PATH` | `.` | Project root containing `raw_audio/` and output folders |
| `--config FILE` | `None` | Transcription config JSON (merged with env + CLI) |
| `--model NAME` | `large-v3` | Whisper model |
| `--device {cuda,cpu}` | `cuda` | Device for ASR (Whisper) inference |
| `--compute-type TYPE` | auto | faster-whisper compute type (float16, float32, int8, int8_float16) |
| `--language CODE` | auto | Force language (e.g. `en`) |
| `--task {transcribe,translate}` | `transcribe` | Whisper task |
| `--vad-min-silence-ms INT` | `500` | Minimum silence to split segments |
| `--beam-size INT` | `5` | Beam search size |
| `--word-timestamps / --no-word-timestamps` | `False` | Extract word-level timestamps (v1.8+) |
| `--skip-existing-json / --no-skip-existing-json` | `True` | Skip files that already have JSON |
| `--enable-diarization` | `False` | Run experimental speaker diarization (pyannote) |
| `--diarization-device {auto,cuda,cpu}` | `auto` | Device for diarization (auto selects cuda if available) |
| `--min-speakers INT` | `None` | Speaker count hint (min) |
| `--max-speakers INT` | `None` | Speaker count hint (max) |
| `--overlap-threshold FLOAT` | `0.3` | Min overlap ratio to assign a speaker |

**Examples**

```bash
# Fast start (auto language)
slower-whisper transcribe

# CPU-only with safer quantization
slower-whisper transcribe --device cpu --compute-type int8

# Diarization with hints
slower-whisper transcribe --enable-diarization --min-speakers 2 --max-speakers 4

# Config layering: file → env → CLI
slower-whisper transcribe --config config/transcribe.json --language en
```

---

## Command: `slower-whisper enrich`

Enrich existing transcripts with prosody and emotion. Requires Stage 1 outputs (`whisper_json/`, `input_audio/`).

```bash
slower-whisper enrich [OPTIONS]
```

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `--root PATH` | `.` | Project root |
| `--config FILE` | `None` | Enrichment config JSON (merged with env + CLI) |
| `--skip-existing / --no-skip-existing` | `True` | Skip segments that already have `audio_state` |
| `--enable-prosody / --no-enable-prosody` | `True` | Extract prosody features |
| `--enable-emotion / --no-enable-emotion` | `True` | Extract dimensional emotion |
| `--enable-categorical-emotion / --no-enable-categorical-emotion` | `False` | Extract categorical emotions (slower) |
| `--enable-speaker-analytics / --no-enable-speaker-analytics` | `None` | Enable both turn metadata and speaker stats (overrides individual flags) |
| `--device {cpu,cuda}` | `cpu` | Device for emotion model inference |
| `--pause-threshold FLOAT` | `None` | Min pause (seconds) to split turns for same speaker |

**Examples**

```bash
# Default enrichment (prosody + dimensional emotion)
slower-whisper enrich

# Prosody only
slower-whisper enrich --no-enable-emotion

# Categorical emotion on GPU
slower-whisper enrich --enable-categorical-emotion --device cuda

# Respect existing audio_state and use a config file
slower-whisper enrich --config config/enrich.json --skip-existing

# Enable speaker analytics (turn metadata + speaker stats)
slower-whisper enrich --enable-speaker-analytics
```

---

## Command: `slower-whisper cache`

Inspect or clear model caches (Whisper, emotion, diarization, HuggingFace, Torch, samples).

```bash
# Show cache locations and sizes
slower-whisper cache --show

# Clear specific caches
slower-whisper cache --clear whisper
slower-whisper cache --clear all
```

---

## Command: `slower-whisper samples`

Manage bundled sample datasets for testing and demos.

```bash
# List available datasets
slower-whisper samples list

# Download and copy to your project
slower-whisper samples download mini_diarization
slower-whisper samples copy mini_diarization --root /path/to/project

# Generate synthetic audio
slower-whisper samples generate --output ./raw_audio --speakers 2
```

---

## End-to-End Workflow

```bash
# Install full stack
uv sync --extra full

# Prepare audio
mkdir -p raw_audio
cp /path/to/meeting.mp3 raw_audio/

# Stage 1
slower-whisper transcribe --language en

# Stage 2
slower-whisper enrich --enable-categorical-emotion

# Inspect results
cat whisper_json/meeting.json
cat transcripts/meeting.txt
```

---

## Troubleshooting

- `command not found`: ensure your virtualenv is activated and `slower-whisper` is on PATH.
- `ModuleNotFoundError: faster_whisper`: install base deps (`uv sync` or `pip install -e .`).
- Enrichment errors about torch/transformers: install extras (`uv sync --extra full` or `pip install -e ".[full]"`).

See [README.md](README.md), [docs/QUICKSTART.md](docs/QUICKSTART.md), and [docs/AUDIO_ENRICHMENT.md](docs/AUDIO_ENRICHMENT.md) for deeper guidance.
