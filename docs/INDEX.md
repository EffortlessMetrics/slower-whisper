# Documentation Index

Welcome to the **slower-whisper** documentation — your guide to local-first conversation intelligence infrastructure.

---

## What is slower-whisper?

slower-whisper is a **local conversation signal engine** that turns audio into **LLM-ready structured data**, capturing not just *what* was said, but ***how*** it was said (prosody, emotion, speaker dynamics, interaction patterns).

**Key differentiators:**

- 🔒 **Local-first** — runs entirely on your machine, no cloud dependency
- 📋 **Stable JSON schema** — versioned contracts for production use
- 🧩 **Modular architecture** — use only the features you need
- 🧪 **Contract-driven** — BDD scenarios enforce behavioral stability
- 🤖 **LLM-native** — designed for RAG, summarization, analysis

See [VISION.md](../VISION.md) for strategic positioning and [ROADMAP.md](../ROADMAP.md) for development timeline.

---

## Quick Navigation

**New to slower-whisper?**

1. Start with [README](../README.md) — project overview, installation, architecture
2. Read [VISION.md](../VISION.md) — understand the "why" and positioning
3. Follow [Quickstart Guide](QUICKSTART.md) — your first transcription
4. Check [API Quick Reference](API_QUICK_REFERENCE.md) — function usage

**Need help?**

- [Troubleshooting Guide](TROUBLESHOOTING.md) — common issues and solutions
- [ARCHITECTURE](ARCHITECTURE.md) — layered design (L0-L4)
- [Examples](../examples/) — working code examples

**Want to contribute?**

- [Contributing Guide](../CONTRIBUTING.md) — how to contribute
- [ROADMAP.md](../ROADMAP.md) — planned features and priorities
- [CLAUDE.md](../CLAUDE.md) — AI assistant instructions for this codebase

---

## Core Documentation

### Repo root essentials

| Document | Description | Audience |
|----------|-------------|----------|
| [README.md](../README.md) | Project overview, features, installation, and quick start | Everyone |
| [VISION.md](../VISION.md) | Strategic vision, positioning, and long-term goals | Everyone interested in the "why" |
| [ROADMAP.md](../ROADMAP.md) | Detailed development timeline and planned features (v1.1-v3.0) | Contributors, users, stakeholders |
| [CHANGELOG.md](../CHANGELOG.md) | Version history and release notes | All users |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | How to contribute to the project | Contributors |
| [SECURITY.md](../SECURITY.md) | Security policy and vulnerability reporting | Security-conscious users |
| [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) | Community standards and expectations | Community members |
| [CLAUDE.md](../CLAUDE.md) | Project guidance for AI coding assistants | Developers using AI tools |

### API & CLI References (docs/)

| Document | Description | Topics |
|----------|-------------|--------|
| [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md) | Quick reference for Python API | Functions, configs, examples |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Command-line interface documentation | Commands, options, workflows |

### Infrastructure & Deployment (docs/)

| Document | Description | Topics |
|----------|-------------|--------|
| [DOCKER.md](DOCKER.md) | Docker containerization guide | Images, builds, usage |
| [DOCKER_DEPLOYMENT_GUIDE.md](DOCKER_DEPLOYMENT_GUIDE.md) | Production deployment with Docker | Compose, scaling, best practices |
| [API_SERVICE.md](API_SERVICE.md) | RESTful API service setup | Web API, endpoints, integration |

### Release & Packaging (docs/releases/)

| Document | Description |
|----------|-------------|
| [releases/RELEASE.md](releases/RELEASE.md) | Release playbook and checklist pointers |
| [releases/RELEASE_CHECKLIST.md](releases/RELEASE_CHECKLIST.md) | Full release checklist |
| [releases/PRE_RELEASE_TEST_PLAN.md](releases/PRE_RELEASE_TEST_PLAN.md) | Pre-release validation steps |
| [releases/FINAL_VERIFICATION_CHECKLIST.md](releases/FINAL_VERIFICATION_CHECKLIST.md) | Final verification before tagging |
| [releases/PACKAGING.md](releases/PACKAGING.md) | Packaging and distribution guide |
| [releases/GITHUB_RELEASE_v1.1.0.md](releases/GITHUB_RELEASE_v1.1.0.md) | GitHub release template for v1.1.0 |

---

## Feature Documentation (docs/)

### Getting Started

| Document | Description | Use When |
|----------|-------------|----------|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute tutorial for first transcription | Starting from scratch |
| [QUICKSTART_AUDIO_ENRICHMENT.md](QUICKSTART_AUDIO_ENRICHMENT.md) | Quick guide to audio enrichment | Adding audio features |

### Core Features

| Document | Description | Topics |
|----------|-------------|--------|
| [AUDIO_ENRICHMENT.md](AUDIO_ENRICHMENT.md) | Stage 2 audio feature extraction | Prosody, emotion, workflow |
| [PROSODY.md](PROSODY.md) | Prosodic feature extraction details | Pitch, energy, rate, pauses |
| [PROSODY_QUICK_REFERENCE.md](PROSODY_QUICK_REFERENCE.md) | Quick reference for prosody features | API, thresholds, examples |
| [SPEAKER_DIARIZATION.md](SPEAKER_DIARIZATION.md) | Speaker diarization design and implementation | Who spoke when, turn structure |
| [LLM_PROMPT_PATTERNS.md](LLM_PROMPT_PATTERNS.md) | Reference prompts for LLM conversation analysis | Prompts, rendering, use cases |
| [MODEL_CACHE.md](MODEL_CACHE.md) | Model cache management | Cache location, cleanup, troubleshooting |

### System Documentation

| Document | Description | Use When |
|----------|-------------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, architecture, and internals | Understanding the codebase |
| [LOGGING.md](LOGGING.md) | Logging system documentation | Configuring logs, debugging, monitoring |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common problems and solutions | Encountering errors |

### Testing & Quality Assurance

| Document | Description | Topics |
|----------|-------------|--------|
| [TESTING_STRATEGY.md](TESTING_STRATEGY.md) | Quality thresholds and testing strategy | Test categories, metrics, "good enough" thresholds, datasets |
| [BDD_IAC_LOCKDOWN.md](BDD_IAC_LOCKDOWN.md) | BDD and IaC as first-class contracts | Behavioral contracts, IaC validation, verification |
| [BDD_IAC_IMPLEMENTATION_SUMMARY.md](BDD_IAC_IMPLEMENTATION_SUMMARY.md) | Implementation summary and usage guide | Status, scripts, next steps |
| [BDD_IAC_PYTHON_CLI.md](BDD_IAC_PYTHON_CLI.md) | Python CLI verification tooling | Cross-platform verification, testing infrastructure |
| [API_BDD_CONTRACT.md](API_BDD_CONTRACT.md) | FastAPI service behavioral contract | REST API BDD scenarios, black-box testing, smoke and functional tests |

---

## Examples Documentation (examples/)

### Example Guides

| Document | Description |
|----------|-------------|
| [examples/README_EXAMPLES.md](../examples/README_EXAMPLES.md) | Overview of all example scripts |
| [examples/INDEX.md](../examples/INDEX.md) | Detailed examples catalog |
| [examples/QUICK_START.md](../examples/QUICK_START.md) | Quick example snippets |
| [examples/QUICK_REFERENCE.md](../examples/QUICK_REFERENCE.md) | Quick reference for common tasks |
| [examples/API_EXAMPLES_README.md](../examples/API_EXAMPLES_README.md) | API usage examples |

### Specialized Examples

| Document | Description |
|----------|-------------|
| [examples/llm_integration/README.md](../examples/llm_integration/README.md) | Using enriched transcripts with LLMs |
| [examples/llm_integration/INDEX.md](../examples/llm_integration/INDEX.md) | LLM integration examples catalog |
| [examples/workflows/README.md](../examples/workflows/README.md) | Complete workflow examples |
| [examples/workflows/QUICKSTART.md](../examples/workflows/QUICKSTART.md) | Quick start for workflow examples |

---

## Benchmarks Documentation (benchmarks/)

| Document | Description |
|----------|-------------|
| [benchmarks/README.md](../benchmarks/README.md) | Benchmarking guide and overview |
| [benchmarks/INTERPRETING_RESULTS.md](../benchmarks/INTERPRETING_RESULTS.md) | How to read benchmark data |
| [benchmarks/BASELINE_RESULTS.md](../benchmarks/BASELINE_RESULTS.md) | Performance baselines |

---

## Archived Documentation (docs/archive/)

These documents are historical artifacts from the development process and transformation of the project. They are kept for reference but are not part of the active documentation.

### Transformation & Verification Reports

| Document | Description | Date |
|----------|-------------|------|
| [TRANSFORMATION_SUMMARY.md](archive/TRANSFORMATION_SUMMARY.md) | Library transformation to production-ready API | 2025-11-15 |
| [VERIFICATION_SUMMARY.md](archive/VERIFICATION_SUMMARY.md) | Code examples verification summary | 2025-11-15 |
| [detailed_verification_report.md](archive/detailed_verification_report.md) | Detailed code example validation | 2025-11-15 |
| [CODE_EXAMPLES_FIXES.md](archive/CODE_EXAMPLES_FIXES.md) | Fixes for deprecated code examples | 2025-11-15 |
| [FINAL_CODE_VERIFICATION_REPORT.md](archive/FINAL_CODE_VERIFICATION_REPORT.md) | Final verification report | 2025-11-15 |

### Testing & Metrics

| Document | Description | Date |
|----------|-------------|------|
| [CLI_WORKFLOW_TEST_PLAN.md](archive/CLI_WORKFLOW_TEST_PLAN.md) | Comprehensive CLI test plan | 2025-11-15 |
| [PROJECT_STATISTICS.md](archive/PROJECT_STATISTICS.md) | Project statistics and metrics | 2025-11-15 |
| [STATISTICS_SUMMARY.md](archive/STATISTICS_SUMMARY.md) | Statistics summary | 2025-11-15 |
| [STATISTICS_INDEX.md](archive/STATISTICS_INDEX.md) | Statistics index | 2025-11-15 |
| [STATS_BADGES.md](archive/STATS_BADGES.md) | Statistics badges | 2025-11-15 |
| [DEVELOPER_METRICS.md](archive/DEVELOPER_METRICS.md) | Developer productivity metrics | 2025-11-15 |

---

## Documentation by Use Case

### "I want to transcribe audio files"
1. [README](../README.md) - Overview and installation
2. [Quickstart](QUICKSTART.md) - First transcription
3. [CLI Reference](CLI_REFERENCE.md) - Command details
4. [Troubleshooting](TROUBLESHOOTING.md) - If issues arise

### "I want to extract audio features (prosody, emotion)"
1. [Audio Enrichment Guide](AUDIO_ENRICHMENT.md) - Overview
2. [Quickstart Audio Enrichment](QUICKSTART_AUDIO_ENRICHMENT.md) - Quick start
3. [Prosody Guide](PROSODY.md) - Detailed prosody documentation
4. [Examples](../examples/) - Working examples

### "I want to use the Python API"
1. [API Quick Reference](API_QUICK_REFERENCE.md) - Function reference
2. [API Examples](../examples/API_EXAMPLES_README.md) - Usage examples
3. [Architecture](ARCHITECTURE.md) - Understanding the design

### "I want to analyze enriched transcripts with LLMs"
1. [LLM Prompt Patterns](LLM_PROMPT_PATTERNS.md) - Reference prompts and rendering strategies
2. [LLM Integration Examples](../examples/llm_integration/README.md) - Working scripts (summarization, QA scoring, coaching)
3. [README Python API](../README.md#llm-integration-analyze-conversations) - Quick API example with speaker labels
4. [Audio Enrichment Guide](AUDIO_ENRICHMENT.md) - Understand the prosody/emotion features

### "I want to deploy this in production"
1. [Docker Guide](DOCKER.md) - Containerization
2. [Docker Deployment Guide](DOCKER_DEPLOYMENT_GUIDE.md) - Production deployment
3. [API Service](API_SERVICE.md) - Web API setup

### "I want to contribute code"
1. [Contributing Guide](../CONTRIBUTING.md) - Process and standards
2. [CLAUDE.md](../CLAUDE.md) - Codebase guidance
3. [Architecture](ARCHITECTURE.md) - System design
4. [BDD/IaC Contracts](BDD_IAC_LOCKDOWN.md) - Testing and deployment standards

### "I want to understand the system design"
1. [Architecture](ARCHITECTURE.md) - Complete architecture overview
2. [API Quick Reference](API_QUICK_REFERENCE.md) - API structure
3. [Prosody Guide](PROSODY.md) - Feature extraction details

### "I encountered an error"
1. [Troubleshooting](TROUBLESHOOTING.md) - Common issues
2. [README](../README.md) - Verify installation
3. [Security Policy](../SECURITY.md) - Report security issues

---

## External Resources

### Community
- GitHub Repository - View source code and report issues
- Issue Tracker - Report bugs and request features
- Discussions - Ask questions and share knowledge

### Related Projects
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper model
- [Parselmouth](https://parselmouth.readthedocs.io/) - Praat phonetics library
- [Transformers](https://huggingface.co/docs/transformers/) - HuggingFace models

### Academic References
- [Whisper paper](https://arxiv.org/abs/2212.04356) - Robust Speech Recognition via Large-Scale Weak Supervision
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

## Documentation Status

### Complete ✅
- Core documentation (README, CHANGELOG, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT)
- API and CLI references
- Quickstart guides
- Architecture documentation
- Audio enrichment documentation
- Prosody documentation
- Examples documentation
- Benchmarks documentation
- Docker and deployment guides

### Archived 📦
- Transformation and verification reports (docs/archive/)
- Statistics and metrics reports (docs/archive/)
- Historical test plans (docs/archive/)

---

## Search Tips

Use your browser's search (Ctrl+F / Cmd+F) or GitHub's search to find specific topics:

- **Installation issues**: Search "troubleshooting" or check TROUBLESHOOTING.md
- **Function usage**: Search function name in API_QUICK_REFERENCE.md
- **Feature explanations**: Check AUDIO_ENRICHMENT.md or PROSODY.md
- **Examples**: Browse examples/ directory or example README files
- **Error messages**: Search error text in TROUBLESHOOTING.md
- **Deployment**: Check DOCKER.md or DOCKER_DEPLOYMENT_GUIDE.md

---

**Last Updated:** 2025-11-17

**Feedback:** Open an issue or discussion on GitHub if you have suggestions for improving the documentation.
