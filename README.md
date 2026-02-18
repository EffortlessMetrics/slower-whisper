# slower-whisper

Audio in, receipts out.

[![CI](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/ci.yml/badge.svg)](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/ci.yml)
[![Verify](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/verify.yml/badge.svg)](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/verify.yml)
[![PyPI](https://img.shields.io/pypi/v/slower-whisper)](https://pypi.org/project/slower-whisper/)
[![Python](https://img.shields.io/badge/python-3.12--3.14-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue)](#license)

```python
from transcription import TranscriptionConfig, transcribe_file

cfg = TranscriptionConfig(model="base", device="auto", language="en")
transcript = transcribe_file("meeting.wav", root=".", config=cfg)

print(transcript.full_text)
print(transcript.segments[0].speaker)       # "spk_0"
print(transcript.segments[0].audio_state)   # prosody, emotion, timing
```

Standard transcription gives you words. slower-whisper gives you words + speakers + tone + emotion + timing — locally, as stable JSON. It's a truth layer for LLM pipelines: acoustic ground truth that models can't hallucinate, pre-computed so they don't have to.

## Install

```bash
uv add slower-whisper
# or: pip install slower-whisper
```

Add what you need:

| Extra | What it adds |
|-------|-------------|
| `enrich-basic` | DSP stack (`numpy`, `librosa`, `soundfile`) |
| `enrich-prosody` | Praat-based prosody (`praat-parselmouth`) |
| `emotion` | Emotion models (`torch`, `torchaudio`, `transformers`) |
| `diarization` | Speaker diarization (`pyannote.audio`) |
| `full` | Everything above |
| `api` | FastAPI service runtime |
| `integrations` | LangChain + LlamaIndex adapters |

```bash
uv sync                              # transcription only
uv sync --extra full                 # all enrichment
uv sync --extra full --extra dev     # contributor toolchain
```

## Package Map

| Package | What it is |
|---------|-----------|
| `transcription` | Core Python API — batch, file, streaming, enrichment, post-processing |
| `slower_whisper` | `faster-whisper` drop-in replacement (`WhisperModel`, `Segment`, `Word`) |
| `slower-whisper` | CLI — `transcribe`, `enrich`, `benchmark`, `export`, `validate` |

## `faster-whisper` Migration

Change one import:

```python
# Before
from faster_whisper import WhisperModel

# After
from slower_whisper import WhisperModel

model = WhisperModel("base", device="auto")
segments, info = model.transcribe("audio.wav", word_timestamps=True)

# slower-whisper extension — full transcript with enrichment
transcript = model.last_transcript
```

Full option mapping: [docs/FASTER_WHISPER_MIGRATION.md](docs/FASTER_WHISPER_MIGRATION.md)

## CLI Quickstart

```bash
# Transcribe (reads raw_audio/, writes whisper_json/)
uv run slower-whisper transcribe --root .

# Enrich with prosody + emotion
uv run slower-whisper enrich --root .

# Benchmark against baselines
uv run slower-whisper benchmark run --track asr --dataset smoke
uv run slower-whisper benchmark compare --track asr --dataset smoke --gate
```

## What You Get

- **Local-first** — all processing on your hardware. No accounts, no rate limits, no per-minute billing.
- **Beyond words** — speaker diarization, prosody, emotion, semantic annotation as opt-in layers.
- **Stable contracts** — JSON schema v2 with `schema_version`, `receipt`, and `run_id`. Same input + config = same output.
- **Streaming** — WebSocket/SSE with event envelopes, resume protocol, and backpressure.
- **Post-processing** — topic segmentation (TF-IDF), turn-taking policies, domain presets for call centers and meetings.
- **Benchmarks** — ASR WER, diarization DER, emotion, streaming latency — with baseline gating in CI.
- **`faster-whisper` compatible** — change one import line.

## Documentation

| Start here | Then |
|-----------|------|
| [Quickstart](docs/QUICKSTART.md) | [CLI Reference](docs/CLI_REFERENCE.md) |
| [Python API](docs/API_QUICK_REFERENCE.md) | [Configuration](docs/CONFIGURATION.md) |
| [faster-whisper Migration](docs/FASTER_WHISPER_MIGRATION.md) | [Streaming Architecture](docs/STREAMING_ARCHITECTURE.md) |
| [Post-Processing](docs/POST_PROCESSING.md) | [Benchmarks](docs/BENCHMARKS.md) |

Full docs index: [docs/INDEX.md](docs/INDEX.md) | Vision and strategy: [VISION.md](VISION.md) | Changelog: [CHANGELOG.md](CHANGELOG.md) | Roadmap: [ROADMAP.md](ROADMAP.md)

## Community & Support

- **Issues** — [Bug reports and feature requests](https://github.com/EffortlessMetrics/slower-whisper/issues)
- **Security** — [Report a vulnerability](https://github.com/EffortlessMetrics/slower-whisper/security/advisories/new) (see [SECURITY.md](SECURITY.md))
- **Contributing** — See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, PR process, and coding standards
- **Support** — [Getting help](.github/SUPPORT.md)

## License

Dual-licensed under Apache 2.0 or MIT, at your option. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).
