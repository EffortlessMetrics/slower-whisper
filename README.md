# slower-whisper

Audio in, receipts out.

Standard transcription gives you words. slower-whisper gives you words + speakers + tone + emotion + timing — locally, as stable JSON. It's a truth layer for LLM pipelines: acoustic ground truth (timestamps, speaker math, prosody) that models can't hallucinate, pre-computed so they don't have to.

"Receipts" = reproducible JSON with config hashes, run IDs, and provenance metadata. Same input and config always produces the same output.

[![CI](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/ci.yml/badge.svg)](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/ci.yml)
[![Verify](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/verify.yml/badge.svg)](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/verify.yml)
[![PyPI](https://img.shields.io/pypi/v/slower-whisper)](https://pypi.org/project/slower-whisper/)
[![Python](https://img.shields.io/badge/python-3.12--3.14-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue)](#license)

## What You Get

- **Local-first** — all processing runs on your hardware; audio never leaves your machine. No accounts, no rate limits, no per-minute billing.
- **Beyond transcription** — optional speaker diarization, prosody extraction, emotion recognition, and semantic annotation as modular layers.
- **Contract-driven** — stable JSON schema (v2) with typed fields, stability tiers, and backward-compatibility guarantees. Every transcript carries `schema_version`, `receipt`, and `run_id` for reproducibility.
- **Streaming** — WebSocket/SSE contracts with event envelopes, resume protocol, and backpressure handling.
- **Pluggable enrichment** — diarization, prosody, emotion, and semantic adapters are optional layers, not monolithic dependencies. Pay only for what you use.
- **Benchmark infrastructure** — CLI with baseline/gating support for ASR, diarization, emotion, and streaming tracks.
- **`faster-whisper` compatible** — drop-in migration shim; change one import line.

### How it compares

| Dimension | Cloud APIs | OSS Toolkits | slower-whisper |
|-----------|-----------|--------------|----------------|
| **Locality** | Cloud-only | Local-capable | **Local-first** |
| **Openness** | Closed | Open components | **Open + unified** |
| **LLM Integration** | Via API | Not designed for it | **LLM-native JSON** |
| **Contracts** | None | None | **BDD + IaC contracts** |
| **Acoustic Features** | Limited | Rich but scattered | **Structured + versioned** |
| **Cost Model** | Per-minute | Free but manual | **Cheap triage + optional LLM** |

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

## Package Map

| Package | Purpose |
|---------|---------|
| `transcription` | Core public API — batch, file, bytes, enrichment, streaming, models |
| `slower_whisper` | `faster-whisper` compatible import surface (`WhisperModel`, `Segment`, `Word`) |
| `slower-whisper` (CLI) | Unified CLI — `transcribe`, `enrich`, `benchmark`, `export`, `validate` |

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
| [docs/POST_PROCESSING.md](docs/POST_PROCESSING.md) | Post-processing pipeline |
| [docs/PROJECT_METADATA.md](docs/PROJECT_METADATA.md) | Metadata/governance surfaces |
| [VISION.md](VISION.md) | Project vision and positioning |

### Quick links by goal

- **I want to transcribe audio** — [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Migrating from faster-whisper** — [docs/FASTER_WHISPER_MIGRATION.md](docs/FASTER_WHISPER_MIGRATION.md)
- **Building LLM apps** — [docs/API_QUICK_REFERENCE.md](docs/API_QUICK_REFERENCE.md) and [docs/LLM_PROMPT_PATTERNS.md](docs/LLM_PROMPT_PATTERNS.md)
- **Understanding the architecture** — [VISION.md](VISION.md) and [docs/STREAMING_ARCHITECTURE.md](docs/STREAMING_ARCHITECTURE.md)

Component README quick links:

- [benchmarks/README.md](benchmarks/README.md)
- [scripts/README.md](scripts/README.md)
- [k8s/README.md](k8s/README.md)

## Project Metadata Surfaces

- Package metadata: [pyproject.toml](pyproject.toml)
- Citation metadata: [CITATION.cff](CITATION.cff)
- Release/change history: [CHANGELOG.md](CHANGELOG.md)
- Current plan/status: [ROADMAP.md](ROADMAP.md)
- Project vision: [VISION.md](VISION.md)

## Community & Support

- **Issues** — [Bug reports and feature requests](https://github.com/EffortlessMetrics/slower-whisper/issues)
- **Security** — [Report a vulnerability](https://github.com/EffortlessMetrics/slower-whisper/security/advisories/new) (see [SECURITY.md](SECURITY.md))
- **Support** — [Getting help](.github/SUPPORT.md)
- **Contributing** — See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, PR process, and coding standards

## License

Dual-licensed under Apache 2.0 or MIT, at your option. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).
