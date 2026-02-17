# slower-whisper

Audio in, receipts out.

`slower-whisper` is a local-first Python toolkit for conversation ETL. It turns raw audio into schema-versioned transcript data with optional speaker/audio enrichment and reproducible run metadata.

[![CI](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/ci.yml/badge.svg)](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/ci.yml)
[![Verify](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/verify.yml/badge.svg)](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/verify.yml)
[![PyPI](https://img.shields.io/pypi/v/slower-whisper)](https://pypi.org/project/slower-whisper/)
[![Python](https://img.shields.io/badge/python-3.12--3.14-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

## What You Get

- Local-first transcription and enrichment (no hosted runtime required)
- Stable schema-versioned transcript JSON (`schema_version = 2`)
- Optional speaker diarization, prosody, and emotion extraction
- Streaming contracts for WebSocket/SSE pipelines
- Benchmark CLI with baseline/gating support
- A `faster-whisper` compatibility shim for migration

## Python Package Map (Crate Equivalents)

| Surface | Purpose | Typical Entry Points |
|---------|---------|----------------------|
| `transcription` | Core public API (batch, file, bytes, enrichment, models, streaming primitives) | `transcribe_directory`, `transcribe_file`, `transcribe_bytes`, `enrich_directory` |
| `slower_whisper` | `faster-whisper` compatible import surface | `WhisperModel`, `Segment`, `Word`, `TranscriptionInfo` |
| `transcription.service*` | FastAPI service and transport layers | `transcription/service.py` |
| `transcription.streaming*` | Real-time session/event contracts, client/server helpers | `streaming.py`, `streaming_ws.py`, `streaming_client.py` |
| `transcription.benchmark_cli` + `benchmarks/` | Multi-track evaluation and regression gates | `slower-whisper benchmark ...` |
| `transcription.store`, `transcription.outcomes`, `transcription.integrations` | Store/outcomes/integration surfaces for downstream automation | package modules + CLI subcommands |

## Install

Base install (transcription only):

```bash
uv add slower-whisper
# or: pip install slower-whisper
```

Extras by use case:

| Extra | Adds |
|-------|------|
| `enrich-basic` | Base DSP stack (`numpy`, `librosa`, `soundfile`) |
| `enrich-prosody` | Praat-based prosody (`praat-parselmouth`) |
| `emotion` | Emotion models (`torch`, `torchaudio`, `transformers`) |
| `diarization` | Speaker diarization (`pyannote.audio`) |
| `full` / `enrich` | Full enrichment bundle |
| `api` | FastAPI service runtime |
| `integrations` | LangChain + LlamaIndex adapters |
| `dev` | Contributor toolchain |

Examples:

```bash
uv sync
uv sync --extra full
uv sync --extra api
uv sync --extra full --extra dev
```

## 5-Minute CLI Quickstart

```bash
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper

# Recommended dev environment
nix develop
uv sync --extra full --extra dev

# Stage 1 transcription (reads raw_audio/, writes whisper_json/)
uv run slower-whisper transcribe --root .

# Optional stage 2 enrichment
uv run slower-whisper enrich --root .
```

## Python API Quickstart

```python
from pathlib import Path

from transcription import TranscriptionConfig, transcribe_file

cfg = TranscriptionConfig(model="base", device="auto", language="en")
transcript = transcribe_file(
    audio_path=Path("raw_audio/meeting.wav"),
    root=Path("."),
    config=cfg,
)

print(transcript.full_text)
```

## `faster-whisper` Drop-In Migration

```python
# Before
from faster_whisper import WhisperModel

# After
from slower_whisper import WhisperModel

model = WhisperModel("base", device="auto")
segments, info = model.transcribe("audio.wav", word_timestamps=True)

# slower-whisper extension
transcript = model.last_transcript
```

See [docs/FASTER_WHISPER_MIGRATION.md](docs/FASTER_WHISPER_MIGRATION.md) for option mapping and compatibility notes.

## Benchmarking

```bash
# Inspect tracks and staged datasets
uv run slower-whisper benchmark list
uv run slower-whisper benchmark status

# Run an ASR smoke benchmark
uv run slower-whisper benchmark run --track asr --dataset smoke

# Compare against stored baseline (gate mode fails on regression)
uv run slower-whisper benchmark compare --track asr --dataset smoke --gate
```

For benchmark details, see [benchmarks/README.md](benchmarks/README.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Local Gate

```bash
./scripts/ci-local.sh
./scripts/ci-local.sh fast
```

Inside devshell, use `nix-clean` wrapper for raw nix commands.

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/INDEX.md](docs/INDEX.md) | Documentation map |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | First-run walkthrough |
| [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | CLI reference |
| [docs/API_QUICK_REFERENCE.md](docs/API_QUICK_REFERENCE.md) | Python API reference |
| [docs/STREAMING_ARCHITECTURE.md](docs/STREAMING_ARCHITECTURE.md) | Streaming contract and architecture |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Config and precedence |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Benchmark command reference |
| [docs/PROJECT_METADATA.md](docs/PROJECT_METADATA.md) | Metadata/governance surfaces |

## Project Metadata Surfaces

- Package metadata: [pyproject.toml](pyproject.toml)
- Citation metadata: [CITATION.cff](CITATION.cff)
- Release/change history: [CHANGELOG.md](CHANGELOG.md)
- Current plan/status: [ROADMAP.md](ROADMAP.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
