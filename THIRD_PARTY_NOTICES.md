# Third-Party Notices

**slower-whisper** is dual-licensed under **Apache-2.0 OR MIT**. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).

This file documents the licenses of third-party dependencies used by slower-whisper. Not all dependencies are required — many are optional extras activated by feature flags.

---

## Core Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| faster-whisper | MIT | CTranslate2-based Whisper inference |
| ctranslate2 | MIT | Efficient transformer inference |
| numpy | BSD-3-Clause | Numerical computing |
| tokenizers | Apache-2.0 | Fast text tokenization |
| huggingface-hub | Apache-2.0 | Model downloads and caching |
| onnxruntime | MIT | ONNX model inference runtime |
| av | BSD-3-Clause | Audio/video demuxing (FFmpeg bindings) |

## Optional: Audio Enrichment

| Package | License | Purpose |
|---------|---------|---------|
| librosa | ISC | Audio analysis and feature extraction |
| soundfile | BSD-3-Clause | Audio file I/O |
| torch | BSD-3-Clause | Deep learning framework |
| torchaudio | BSD-2-Clause | Audio processing for PyTorch |

## Optional: Speaker Diarization

| Package | License | Purpose |
|---------|---------|---------|
| pyannote.audio | MIT | Speaker diarization models |
| pyannote.core | MIT | Core annotation data structures |

## Optional: Emotion Recognition

| Package | License | Purpose |
|---------|---------|---------|
| transformers | Apache-2.0 | HuggingFace model inference |

## Optional: Prosody (Acoustic Features)

| Package | License | Purpose |
|---------|---------|---------|
| praat-parselmouth | **GPL-3.0** | Praat phonetics via Python bindings |

> **GPL-3.0 notice:** `praat-parselmouth` is licensed under GPL-3.0. It is an **optional** dependency (installed via `--extra prosody` or `--extra full`). If GPL-3.0 is incompatible with your use case, omit this extra — all other features remain available under permissive licenses. See [SECURITY.md](SECURITY.md) for additional context.

## Optional: API Service

| Package | License | Purpose |
|---------|---------|---------|
| fastapi | MIT | Web framework for REST API |
| uvicorn | BSD-3-Clause | ASGI server |
| websockets | BSD-3-Clause | WebSocket protocol support |

## Optional: Cloud Semantic Adapters

| Package | License | Purpose |
|---------|---------|---------|
| openai | Apache-2.0 | OpenAI API client |
| anthropic | MIT | Anthropic API client |

## Development Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| pytest | MIT | Test framework |
| ruff | MIT | Linter and formatter |
| mypy | MIT | Static type checker |
| pre-commit | MIT | Git hook management |

---

## Regenerating This File

To regenerate a complete dependency license inventory:

```bash
# Install pip-licenses
uv pip install pip-licenses

# Generate license table for installed packages
uv run pip-licenses --format=markdown --with-urls --order=license
```

Review the output and update this file as needed when dependencies change.

---

## Model Licenses

slower-whisper downloads and uses pre-trained models. Model licenses are separate from code licenses:

| Model | License | Source |
|-------|---------|--------|
| Whisper (OpenAI) | MIT | [openai/whisper](https://github.com/openai/whisper) |
| pyannote speaker diarization | MIT | [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| Wav2Vec2 emotion models | Various (check model card) | HuggingFace Hub |

Always check the model card on HuggingFace Hub for the specific license terms of any model you download.

---

**Last updated:** 2026-02-18
