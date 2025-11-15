# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial packaging and distribution infrastructure
- Comprehensive installation and packaging documentation
- Docker support with CPU and GPU variants
- GitHub Actions CI/CD workflows

## [1.0.0] - 2025-11-15

### Added
- Initial release of slower-whisper
- Core transcription pipeline using faster-whisper
- Audio normalization with ffmpeg
- JSON/TXT/SRT output formats
- Two-stage pipeline architecture (transcription + enrichment)
- Audio enrichment system with:
  - Prosody feature extraction (pitch, energy, speech rate, pauses)
  - Emotion recognition (dimensional and categorical)
  - Audio state annotation in JSON schema
- CLI interfaces:
  - `slower-whisper` / `transcribe` - Main transcription CLI
  - `audio-enrich` - Audio enrichment CLI
- Structured JSON schema (v2) with metadata and segments
- GPU acceleration support via CUDA
- VAD filtering for better segmentation
- Multi-language support with auto-detection
- Configurable model sizes (tiny, base, small, medium, large-v3)
- Comprehensive test suite
- Documentation:
  - README with quickstart guide
  - Audio enrichment guide
  - Prosody reference documentation
  - Contributing guidelines
  - Code quality and type checking setup

### Features
- **Transcription**:
  - Multiple Whisper model sizes supported
  - GPU and CPU inference modes
  - Quantized compute types (float16, int8_float16, int8)
  - Configurable beam size and VAD parameters
  - Language hint and task selection
  - Skip already-transcribed files

- **Audio Enrichment**:
  - Research-grade pitch extraction using Praat
  - Energy level analysis
  - Speech rate calculation (syllables/sec)
  - Pause detection and analysis
  - Dimensional emotion (valence, arousal, dominance)
  - Categorical emotion classification
  - Automatic level categorization

- **Output Formats**:
  - JSON with structured metadata and segments
  - Plain text transcripts with timestamps
  - SRT subtitle files
  - Extensible schema for future features

- **Developer Experience**:
  - Modular codebase with clear separation of concerns
  - Type hints throughout
  - Ruff for linting and formatting
  - Mypy for type checking
  - Pre-commit hooks
  - pytest test suite
  - GitHub Actions CI/CD

### Dependencies
- Python 3.10+ required
- Core: faster-whisper>=1.0.0
- Optional (audio-enrich): soundfile, numpy, librosa, praat-parselmouth, transformers, torch
- System: ffmpeg, libsndfile1 (Linux)

### License
- Apache License 2.0

---

## Version History

### [1.0.0] - 2025-11-15
Initial public release

---

## Versioning Guidelines

This project uses [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0 → 2.0.0): Incompatible API changes
- **MINOR** version (1.0.0 → 1.1.0): New features, backward-compatible
- **PATCH** version (1.0.0 → 1.0.1): Bug fixes, backward-compatible

### Pre-release Versions

- **Alpha** (1.0.0a1): Early testing, unstable API
- **Beta** (1.0.0b1): Feature-complete, API mostly stable
- **Release Candidate** (1.0.0rc1): Final testing before release

---

## Release Notes Format

For each release, include:

### Added
- New features and capabilities

### Changed
- Changes to existing functionality

### Deprecated
- Features marked for removal in future versions

### Removed
- Features removed in this version

### Fixed
- Bug fixes and error corrections

### Security
- Security fixes and vulnerability patches

---

## Migration Guides

### Upgrading from 0.x to 1.0

Not applicable (1.0.0 is the initial release).

---

## Links

- [Homepage](https://github.com/yourusername/slower-whisper)
- [Issue Tracker](https://github.com/yourusername/slower-whisper/issues)
- [Releases](https://github.com/yourusername/slower-whisper/releases)
- [PyPI](https://pypi.org/project/slower-whisper/)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Submitting pull requests
- Updating this changelog
