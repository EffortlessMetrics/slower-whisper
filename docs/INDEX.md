# Documentation Index

Welcome to the slower-whisper documentation. This page provides a complete map of all available documentation.

---

## Quick Navigation

**New to slower-whisper?**
1. Start with the [README](../README.md) for project overview
2. Follow the [Quickstart Guide](QUICKSTART.md) for your first transcription
3. Read [Installation](INSTALLATION.md) for detailed setup

**Need help?**
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions
- [Examples](EXAMPLES.md) - Working code examples
- [API Reference](API_REFERENCE.md) - Function and class documentation

**Want to contribute?**
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Developer Documentation](dev/README.md) - Development guides

---

## User Documentation

### Getting Started

| Document | Description | Audience |
|----------|-------------|----------|
| [README](../README.md) | Project overview, features, quick start | Everyone |
| [Quickstart](QUICKSTART.md) | 5-minute tutorial for first transcription | New users |
| [Installation](INSTALLATION.md) | Detailed installation instructions | All users |

### Core Features

| Document | Description | Topics |
|----------|-------------|--------|
| [Audio Enrichment](AUDIO_ENRICHMENT.md) | Stage 2 audio feature extraction | Prosody, emotion, workflow |
| [Prosody Guide](PROSODY.md) | Prosodic feature extraction details | Pitch, energy, rate, pauses |
| [Prosody Reference](PROSODY_REFERENCE.md) | Quick reference for prosody features | API, thresholds, examples |

### Reference & Help

| Document | Description | Use When |
|----------|-------------|----------|
| [API Reference](API_REFERENCE.md) | Complete API documentation | Looking up functions/classes |
| [Architecture](ARCHITECTURE.md) | System design and structure | Understanding internals |
| [Troubleshooting](TROUBLESHOOTING.md) | Common problems and solutions | Encountering errors |
| [Examples](EXAMPLES.md) | Working code examples | Learning by example |

---

## Developer Documentation

### Contributing

| Document | Description | Topics |
|----------|-------------|--------|
| [Contributing Guide](../CONTRIBUTING.md) | How to contribute to the project | Setup, workflow, PR process |
| [Developer Guide](dev/README.md) | Developer documentation index | Testing, tools, release |

### Development Guides

| Document | Description | Topics |
|----------|-------------|--------|
| [Testing Guide](dev/TESTING.md) | Test suite documentation | Running tests, writing tests |
| [Development Tools](dev/TOOLS.md) | Tool setup and configuration | mypy, ruff, pre-commit |
| [Release Process](dev/RELEASE.md) | How to release new versions | Versioning, packaging, deployment |
| [JSON Schema](dev/JSON_SCHEMA.md) | Transcript schema documentation | Schema versions, validation |
| [UV Reference](dev/UV_REFERENCE.md) | UV package manager guide | Fast installs, commands |
| [Dependencies](dev/DEPENDENCIES.md) | Dependency analysis | Package overview, optimization |

### Specialized Developer Topics

| Document | Description | Topics |
|----------|-------------|--------|
| [Packaging Guide](dev/packaging/GUIDE.md) | Packaging and distribution | PyPI, wheels, dependencies |
| [Distribution](dev/packaging/DISTRIBUTION.md) | Distribution details | Upload, versioning, metadata |
| [Security Audit](dev/security/AUDIT.md) | Security audit findings | Vulnerabilities, mitigations |
| [Security Improvements](dev/security/IMPROVEMENTS.md) | Implemented security enhancements | Fixes, best practices |

---

## Examples Documentation

| Document | Description |
|----------|-------------|
| [Examples Overview](../examples/README.md) | Overview of all example scripts |
| [Examples Index](../examples/INDEX.md) | Detailed examples catalog |
| [Quick Start Examples](../examples/QUICK_START.md) | Quick example snippets |
| [LLM Integration](../examples/llm_integration/README.md) | Using enriched transcripts with LLMs |
| [Workflows](../examples/workflows/README.md) | Complete workflow examples |

---

## Benchmarks Documentation

| Document | Description |
|----------|-------------|
| [Benchmarks Overview](../benchmarks/README.md) | Benchmarking guide |
| [Infrastructure](../benchmarks/INFRASTRUCTURE.md) | Benchmark infrastructure setup |
| [Interpreting Results](../benchmarks/INTERPRETING_RESULTS.md) | How to read benchmark data |
| [Baseline Results](../benchmarks/BASELINE_RESULTS.md) | Performance baselines |

---

## Project Documentation

### Governance & Process

| Document | Description | Location |
|----------|-------------|----------|
| [Security Policy](../SECURITY.md) | Security reporting and policy | Root |
| [Changelog](../CHANGELOG.md) | Version history | Root |
| [License](../LICENSE) | Project license | Root |

---

## Documentation by Use Case

### "I want to transcribe audio files"
1. [Installation](INSTALLATION.md)
2. [Quickstart](QUICKSTART.md)
3. [Troubleshooting](TROUBLESHOOTING.md) (if issues)

### "I want to extract audio features (prosody, emotion)"
1. [Audio Enrichment Guide](AUDIO_ENRICHMENT.md)
2. [Prosody Guide](PROSODY.md)
3. [Examples](EXAMPLES.md)

### "I want to analyze enriched transcripts"
1. [Examples](EXAMPLES.md)
2. [API Reference](API_REFERENCE.md)
3. [Examples scripts](../examples/)

### "I want to integrate with LLMs"
1. [Audio Enrichment Guide](AUDIO_ENRICHMENT.md) (see rendering section)
2. [LLM Integration Examples](../examples/llm_integration/README.md)
3. [Architecture](ARCHITECTURE.md) (see extension points)

### "I want to contribute code"
1. [Contributing Guide](../CONTRIBUTING.md)
2. [Developer Guide](dev/README.md)
3. [Testing Guide](dev/TESTING.md)

### "I want to understand the system design"
1. [Architecture](ARCHITECTURE.md)
2. [API Reference](API_REFERENCE.md)
3. [JSON Schema](dev/JSON_SCHEMA.md)

### "I encountered an error"
1. [Troubleshooting](TROUBLESHOOTING.md)
2. [Installation](INSTALLATION.md) (reinstall/verify)
3. [GitHub Issues](https://github.com/yourusername/slower-whisper/issues)

---

## Documentation Status

### Complete ✅
- README.md
- QUICKSTART.md
- AUDIO_ENRICHMENT.md
- PROSODY.md
- CONTRIBUTING.md
- SECURITY.md
- Examples documentation
- Benchmarks documentation

### In Progress 🚧
- INDEX.md (this file)
- INSTALLATION.md (consolidation needed)
- API_REFERENCE.md (needs creation)
- ARCHITECTURE.md (needs creation from IMPLEMENTATION_SUMMARY)
- TROUBLESHOOTING.md (needs consolidation)
- EXAMPLES.md (needs consolidation)
- Developer guides (needs organization)

### Planned 📋
- Tutorial videos
- Interactive examples
- Performance optimization guide
- Advanced integration patterns

---

## External Resources

### Community
- [GitHub Repository](https://github.com/yourusername/slower-whisper)
- [Issue Tracker](https://github.com/yourusername/slower-whisper/issues)
- [Discussions](https://github.com/yourusername/slower-whisper/discussions)

### Related Projects
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper model
- [Parselmouth](https://parselmouth.readthedocs.io/) - Praat phonetics library
- [Transformers](https://huggingface.co/docs/transformers/) - HuggingFace models

### Academic References
- Whisper paper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- MSP-Podcast emotion dataset
- Wav2vec 2.0 models for emotion recognition

---

## Contributing to Documentation

Found an error or want to improve the docs? See [CONTRIBUTING.md](../CONTRIBUTING.md) section on documentation contributions.

### Documentation Guidelines
- Keep explanations clear and concise
- Include working code examples
- Show expected output
- Link to related documentation
- Test all code snippets
- Update this index when adding new docs

---

## Search Tips

Use your browser's search (Ctrl+F / Cmd+F) or GitHub's search to find specific topics:

- **Installation issues**: Search "troubleshooting" or check TROUBLESHOOTING.md
- **Function usage**: Search function name in API_REFERENCE.md
- **Feature explanations**: Check AUDIO_ENRICHMENT.md or PROSODY.md
- **Examples**: Browse examples/ directory or EXAMPLES.md
- **Error messages**: Search error text in TROUBLESHOOTING.md

---

**Last Updated:** 2025-11-15

**Feedback:** Open an issue or discussion on GitHub if you have suggestions for improving the documentation.
