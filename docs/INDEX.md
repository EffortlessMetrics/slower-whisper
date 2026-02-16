# Documentation Index

Welcome to the **slower-whisper** documentation -- ETL for conversations, audio in, receipts out.

---

## What is slower-whisper?

**slower-whisper is ETL for conversations.** It transforms raw audio into schema-versioned structured data that captures not just *what* was said, but *who* said it, *when*, and *how*.

**Why it exists:**

- **FinOps for LLMs** -- cheap deterministic triage (rules, DSP, speaker math) before expensive model inference
- **Truth layer** -- acoustic ground truth (timestamps, speakers, prosody) that LLMs can't hallucinate
- **Local-first** -- all processing runs on your hardware, data never leaves

**Key properties:**

- **Schema-versioned JSON** (v2) with stability tiers and backward compatibility
- **5 semantic adapters** -- local keywords, local LLM, OpenAI, Anthropic, or bring your own
- **Streaming built-in** -- WebSocket + SSE for real-time pipelines
- **Receipt provenance** -- config hash, run IDs, git commit for reproducibility

See [VISION.md](../VISION.md) for strategic positioning and [ROADMAP.md](../ROADMAP.md) for development timeline.

---

## Quick Navigation

**New to slower-whisper?**

1. Start with [README](../README.md) -- project overview, installation, architecture
2. Read [VISION.md](../VISION.md) -- understand the "why" and positioning
3. Follow [Quickstart Guide](QUICKSTART.md) -- your first transcription
4. Check [API Quick Reference](API_QUICK_REFERENCE.md) -- function usage

**Need help?**

- [Troubleshooting Guide](TROUBLESHOOTING.md) -- common issues and solutions
- [ARCHITECTURE](ARCHITECTURE.md) -- layered design (L0-L4)
- [Examples](../examples/) -- working code examples

**Want to contribute?**

- [Contributing Guide](../CONTRIBUTING.md) -- how to contribute
- [ROADMAP.md](../ROADMAP.md) -- planned features and priorities
- [CLAUDE.md](../CLAUDE.md) -- AI assistant instructions for this codebase

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
| [CITATION.cff](../CITATION.cff) | Citation metadata for software use in papers and reports | Researchers and downstream users |
| [.github/SUPPORT.md](../.github/SUPPORT.md) | Support routes and issue filing expectations | Users and contributors |
| [.github/settings.yml](../.github/settings.yml) | Repository settings as code (topics, labels, branch protection) | Maintainers |

### API & CLI References (docs/)

| Document | Description | Topics |
|----------|-------------|--------|
| [SCHEMA.md](SCHEMA.md) | Transcript JSON schema contract | Normative keys, stability tiers, validation |
| [CONFIGURATION.md](CONFIGURATION.md) | Configuration guide for all layers | Defaults, env vars, files, CLI flags, precedence |
| [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md) | Quick reference for Python API | Functions, configs, examples |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Command-line interface documentation | Commands, options, workflows |
| [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) | Environment variable reference | All supported env vars |

### Infrastructure & Deployment (docs/)

| Document | Description | Topics |
|----------|-------------|--------|
| [DOCKER.md](DOCKER.md) | Docker containerization guide | Images, builds, usage |
| [DOCKER_DEPLOYMENT_GUIDE.md](DOCKER_DEPLOYMENT_GUIDE.md) | Production deployment with Docker | Compose, scaling, best practices |
| [API_SERVICE.md](API_SERVICE.md) | RESTful API service setup | Web API, endpoints, integration |
| [GPU_SETUP.md](GPU_SETUP.md) | GPU configuration and troubleshooting | CUDA, device selection, compute types |
| [DEV_ENV_NIX.md](DEV_ENV_NIX.md) | Nix development environment | Reproducible builds, flake setup |
| [INSTALL.md](INSTALL.md) | Installation guide | Setup, dependencies, verification |
| [CI_CHECKS.md](CI_CHECKS.md) | CI pipeline documentation | GitHub Actions, checks, gates |
| [PERFORMANCE.md](PERFORMANCE.md) | Performance tuning guide | Optimization, profiling, benchmarks |

### Release & Packaging (docs/releases/)

| Document | Description |
|----------|-------------|
| [releases/RELEASE.md](releases/RELEASE.md) | Release playbook and checklist pointers |
| [releases/RELEASE_CHECKLIST_v2.0.0.md](releases/RELEASE_CHECKLIST_v2.0.0.md) | **v2.0.0 release checklist (current)** |
| [releases/RELEASE_CHECKLIST.md](releases/RELEASE_CHECKLIST.md) | v1.1.0 release checklist (reference) |
| [releases/PRE_RELEASE_TEST_PLAN.md](releases/PRE_RELEASE_TEST_PLAN.md) | Pre-release validation steps |
| [releases/FINAL_VERIFICATION_CHECKLIST.md](releases/FINAL_VERIFICATION_CHECKLIST.md) | Final verification before tagging |
| [releases/PACKAGING.md](releases/PACKAGING.md) | Packaging and distribution guide |
| [releases/GITHUB_RELEASE_v1.1.0.md](releases/GITHUB_RELEASE_v1.1.0.md) | GitHub release template for v1.1.0 |
| [releases/v1.0.0-release-notes.md](releases/v1.0.0-release-notes.md) | v1.0.0 release notes |
| [releases/RELEASE_v1.0.0_SUMMARY.md](releases/RELEASE_v1.0.0_SUMMARY.md) | v1.0.0 release summary |
| [releases/RELEASE_EXECUTIVE_SUMMARY.md](releases/RELEASE_EXECUTIVE_SUMMARY.md) | Executive summary for releases |
| [RELEASE_CHECKLIST_NIX.md](RELEASE_CHECKLIST_NIX.md) | Nix-specific release checklist |

---

## Feature Documentation (docs/)

### Getting Started

| Document | Description | Use When |
|----------|-------------|----------|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute tutorial for first transcription | Starting from scratch |
| [FASTER_WHISPER_MIGRATION.md](FASTER_WHISPER_MIGRATION.md) | **Drop-in replacement for faster-whisper** | Migrating from faster-whisper |
| [ADVANTAGE_LAYER_QUICKSTART.md](ADVANTAGE_LAYER_QUICKSTART.md) | **End-to-end workflow: store, outcomes, privacy, webhooks, RAG** | Using the full advantage layer |
| [CONFIGURATION.md](CONFIGURATION.md) | Complete configuration guide | Setting up options, using config files, understanding precedence |
| [QUICKSTART_AUDIO_ENRICHMENT.md](QUICKSTART_AUDIO_ENRICHMENT.md) | Quick guide to audio enrichment | Adding audio features |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | Migration guide for version upgrades | Upgrading between versions |
| [MIGRATION_V2.md](MIGRATION_V2.md) | v2.0 migration guide | Preparing for v2.0 changes |

### Core Features

| Document | Description | Topics |
|----------|-------------|--------|
| [AUDIO_ENRICHMENT.md](AUDIO_ENRICHMENT.md) | Stage 2 audio feature extraction | Prosody, emotion, workflow |
| [PROSODY.md](PROSODY.md) | Prosodic feature extraction details | Pitch, energy, rate, pauses |
| [PROSODY_QUICK_REFERENCE.md](PROSODY_QUICK_REFERENCE.md) | Quick reference for prosody features | API, thresholds, examples |
| [SPEAKER_DIARIZATION.md](SPEAKER_DIARIZATION.md) | Speaker diarization design and implementation | Who spoke when, turn structure |
| [MODEL_CACHE.md](MODEL_CACHE.md) | Model cache management | Cache location, cleanup, troubleshooting |
| [REDACTION.md](REDACTION.md) | PII redaction capabilities | Privacy, data handling |

### Post-Processing (v2.0.1+)

| Document | Description | Topics |
|----------|-------------|--------|
| [POST_PROCESSING.md](POST_PROCESSING.md) | **Post-processing guide** | `PostProcessor`, topic segmentation, turn-taking policies, presets |

| Module | Description | Topics |
|--------|-------------|--------|
| `transcription.post_process` | **Post-processing orchestration** | `PostProcessor`, `PostProcessConfig`, preset configs for call center/meetings |
| `transcription.topic_segmentation` | **Topic segmentation** | TF-IDF similarity detection, `TopicSegmenter`, `StreamingTopicSegmenter` |
| `transcription.turn_taking_policy` | **Turn-taking policies** | `TurnTakingPolicy`, `TurnTakingEvaluator`, aggressive/balanced/conservative presets |

### Advantage Layer (v2.0)

| Document | Description | Topics |
|----------|-------------|--------|
| [ADVANTAGE_LAYER_QUICKSTART.md](ADVANTAGE_LAYER_QUICKSTART.md) | **Complete end-to-end workflow** | Store, outcomes, privacy, webhooks, RAG |
| [DATASET_MANIFEST.md](DATASET_MANIFEST.md) | Dataset manifest schema | Benchmark datasets, CI validation |

### Streaming

| Document | Description | Topics |
|----------|-------------|--------|
| [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) | Streaming transcription design | Callbacks, events, v2.0 protocol |
| [STREAMING_API.md](STREAMING_API.md) | Streaming API reference | WebSocket API, REST SSE, Python client |

### LLM Integration & Semantic Analysis

| Document | Description | Topics |
|----------|-------------|--------|
| [LLM_PROMPT_PATTERNS.md](LLM_PROMPT_PATTERNS.md) | Reference prompts for LLM conversation analysis | Prompts, rendering, use cases |
| [LLM_SEMANTIC_ANNOTATOR.md](LLM_SEMANTIC_ANNOTATOR.md) | LLM-based semantic annotation system | Cloud LLM integration, annotation protocols |
| [SEMANTIC_BENCHMARK.md](SEMANTIC_BENCHMARK.md) | Semantic quality evaluation | LLM-based quality metrics, evaluation |

### System Documentation

| Document | Description | Use When |
|----------|-------------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, architecture, and internals | Understanding the codebase |
| [LOGGING.md](LOGGING.md) | Logging system documentation | Configuring logs, debugging, monitoring |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common problems and solutions | Encountering errors |
| [TYPING_POLICY.md](TYPING_POLICY.md) | Type annotation policy | Type hints, mypy, conventions |
| [DATA_FLOW_ANALYSIS.md](DATA_FLOW_ANALYSIS.md) | Data flow through the pipeline | Understanding data transformations |

### Testing & Quality Assurance

| Document | Description | Topics |
|----------|-------------|--------|
| [TESTING_STRATEGY.md](TESTING_STRATEGY.md) | Quality thresholds and testing strategy | Test categories, metrics, "good enough" thresholds, datasets |
| [BDD_IAC_LOCKDOWN.md](BDD_IAC_LOCKDOWN.md) | BDD and IaC as first-class contracts | Behavioral contracts, IaC validation, verification |
| [BDD_IAC_IMPLEMENTATION_SUMMARY.md](BDD_IAC_IMPLEMENTATION_SUMMARY.md) | Implementation summary and usage guide | Status, scripts, next steps |
| [BDD_IAC_PYTHON_CLI.md](BDD_IAC_PYTHON_CLI.md) | Python CLI verification tooling | Cross-platform verification, testing infrastructure |
| [API_BDD_CONTRACT.md](API_BDD_CONTRACT.md) | FastAPI service behavioral contract | REST API BDD scenarios, black-box testing, smoke and functional tests |
| [CONTRACT_ENFORCEMENT_SUMMARY.md](CONTRACT_ENFORCEMENT_SUMMARY.md) | Contract enforcement overview | How contracts are enforced |
| [TEST_COVERAGE_GAPS.md](TEST_COVERAGE_GAPS.md) | Known test coverage gaps | Areas needing more tests |

### Diarization Deep Dives

| Document | Description |
|----------|-------------|
| [DIARIZATION_DATA_FLOW_SUMMARY.md](DIARIZATION_DATA_FLOW_SUMMARY.md) | Diarization data flow overview |
| [DIARIZATION_VERIFICATION_INDEX.md](DIARIZATION_VERIFICATION_INDEX.md) | Diarization verification index |
| [DIARIZATION_TRACE_LOG.md](DIARIZATION_TRACE_LOG.md) | Diarization trace logging |

### Audit & Process (docs/audit/)

| Document | Description |
|----------|-------------|
| [audit/README.md](audit/README.md) | Audit infrastructure overview + trust loop diagram |
| [audit/AUDIT_PATH.md](audit/AUDIT_PATH.md) | 15-minute cold-reader validation checklist |
| [audit/EXHIBITS.md](audit/EXHIBITS.md) | Annotated PRs demonstrating audit workflow |
| [audit/FAILURE_MODES.md](audit/FAILURE_MODES.md) | Taxonomy of failure modes and prevention patterns |
| [audit/PR_DOSSIER_SCHEMA.md](audit/PR_DOSSIER_SCHEMA.md) | Schema for structured PR analysis |
| [audit/PR_ANALYSIS_WORKFLOW.md](audit/PR_ANALYSIS_WORKFLOW.md) | PR analysis workflow guide |
| [audit/PR_LEDGER_TEMPLATE.md](audit/PR_LEDGER_TEMPLATE.md) | Template for PR ledger entries |

---

## Benchmarks Documentation

### Benchmark Framework (docs/)

| Document | Description |
|----------|-------------|
| [BENCHMARKS.md](BENCHMARKS.md) | Benchmark CLI reference and usage |
| [SEMANTIC_BENCHMARK.md](SEMANTIC_BENCHMARK.md) | LLM-based semantic quality evaluation |
| [BENCHMARK_EVALUATION_QUICKSTART.md](BENCHMARK_EVALUATION_QUICKSTART.md) | Getting started with benchmark evaluation |
| [GETTING_STARTED_EVALUATION.md](GETTING_STARTED_EVALUATION.md) | Evaluation framework introduction |
| [LOCAL_EVAL_WORKFLOW.md](LOCAL_EVAL_WORKFLOW.md) | Local evaluation workflow |
| [EVALUATION_LOOP_QUICKREF.md](EVALUATION_LOOP_QUICKREF.md) | Quick reference for evaluation loop |
| [METRICS_EXAMPLES.md](METRICS_EXAMPLES.md) | Metrics calculation examples |

### Benchmark Results & Baselines (benchmarks/)

| Document | Description |
|----------|-------------|
| [benchmarks/README.md](../benchmarks/README.md) | Benchmarking guide and overview |
| [benchmarks/INTERPRETING_RESULTS.md](../benchmarks/INTERPRETING_RESULTS.md) | How to read benchmark data |
| [benchmarks/BASELINE_RESULTS.md](../benchmarks/BASELINE_RESULTS.md) | Performance baselines |

### Dataset Setup Guides

| Document | Description |
|----------|-------------|
| [DATASET_MANIFEST.md](DATASET_MANIFEST.md) | Dataset manifest infrastructure and schema |
| [LIBRISPEECH_SETUP.md](LIBRISPEECH_SETUP.md) | **LibriSpeech ASR corpus setup (recommended)** |
| [LIBRISPEECH_QUICKSTART.md](LIBRISPEECH_QUICKSTART.md) | LibriSpeech ASR evaluation quick start |
| [LIBRISPEECH_EVAL_RESULTS.md](LIBRISPEECH_EVAL_RESULTS.md) | LibriSpeech evaluation results |
| [AMI_SETUP.md](AMI_SETUP.md) | AMI Meeting Corpus setup for diarization evaluation |
| [AMI_DOWNLOAD_GUIDE.md](AMI_DOWNLOAD_GUIDE.md) | AMI dataset download instructions |
| [AMI_DIRECTORY_LAYOUT.md](AMI_DIRECTORY_LAYOUT.md) | AMI dataset directory structure |
| [AMI_ANNOTATION_SCHEMA.md](AMI_ANNOTATION_SCHEMA.md) | AMI annotation schema reference |
| [AMI_INTEGRATION_SUMMARY.md](AMI_INTEGRATION_SUMMARY.md) | AMI integration summary |
| [AMI_EVAL_SMOKE_TEST_RESULTS.md](AMI_EVAL_SMOKE_TEST_RESULTS.md) | AMI evaluation smoke test results |
| [CALLHOME_SETUP.md](CALLHOME_SETUP.md) | **CALLHOME telephone diarization setup** |
| [IEMOCAP_SETUP.md](IEMOCAP_SETUP.md) | IEMOCAP emotion dataset setup |
| [IEMOCAP_QUICKREF.md](IEMOCAP_QUICKREF.md) | IEMOCAP quick reference |
| [IEMOCAP_LABEL_MAPPING.md](IEMOCAP_LABEL_MAPPING.md) | IEMOCAP emotion label mapping |
| [IEMOCAP_INTEGRATION_SUMMARY.md](IEMOCAP_INTEGRATION_SUMMARY.md) | IEMOCAP integration summary |
| [LIBRICSS_SETUP.md](LIBRICSS_SETUP.md) | LibriCSS overlapping speech / diarization setup |

### Benchmark Cache & Strategy

| Document | Description |
|----------|-------------|
| [BENCHMARK_CACHE_RECOMMENDATIONS.md](BENCHMARK_CACHE_RECOMMENDATIONS.md) | Cache strategy recommendations |
| [BENCHMARK_CACHE_STRATEGY_ANALYSIS.md](BENCHMARK_CACHE_STRATEGY_ANALYSIS.md) | Detailed cache strategy analysis |

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

## Development & Internal Documentation

### Dogfooding & Internal Testing

| Document | Description |
|----------|-------------|
| [DOGFOOD_SETUP.md](DOGFOOD_SETUP.md) | Dogfooding setup guide |
| [DOGFOOD_QUICKSTART.md](DOGFOOD_QUICKSTART.md) | Dogfooding quick start |
| [DOGFOOD_NOTES.md](DOGFOOD_NOTES.md) | Dogfooding notes and observations |

### Planning & Status

| Document | Description |
|----------|-------------|
| [RELEASE_READINESS.md](RELEASE_READINESS.md) | **v2.0 release readiness board** — milestones, exit criteria, validation commands |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | Current development status |
| [EVAL_STATUS.md](EVAL_STATUS.md) | Evaluation status |
| [EVALUATION_LOG.md](EVALUATION_LOG.md) | Evaluation log entries |
| [TIER2_IMPLEMENTATION_PLAN.md](TIER2_IMPLEMENTATION_PLAN.md) | Tier 2 feature implementation plan |
| [V1.1_SKELETON_SUMMARY.md](V1.1_SKELETON_SUMMARY.md) | v1.1 skeleton summary |
| [V1.1_GITHUB_ISSUES.md](V1.1_GITHUB_ISSUES.md) | v1.1 GitHub issues tracking |
| [PROJECT_METADATA.md](PROJECT_METADATA.md) | Canonical metadata and governance map |
| [GITHUB_UI_TASKS.md](GITHUB_UI_TASKS.md) | GitHub UI tasks |

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

### Additional Archives

| Document | Description | Date |
|----------|-------------|------|
| [API_SUMMARY.md](archive/API_SUMMARY.md) | Historical API summary | 2025-11-15 |
| [INDEX.md](archive/INDEX.md) | Historical documentation index | 2025-11-15 |
| [PROJECT_SUMMARY.md](archive/PROJECT_SUMMARY.md) | Historical project summary | 2025-11-15 |
| [DEPLOYMENT_CHECKLIST.md](archive/DEPLOYMENT_CHECKLIST.md) | Historical deployment checklist | 2025-11-15 |
| [CONFIGURATION_DOCUMENTATION_SUMMARY.md](archive/CONFIGURATION_DOCUMENTATION_SUMMARY.md) | Configuration documentation summary | 2025-11-15 |

---

## Documentation by Use Case

### "I want to transcribe audio files"
1. [README](../README.md) - Overview and installation
2. [Quickstart](QUICKSTART.md) - First transcription
3. [Configuration Guide](CONFIGURATION.md) - Setup and customization
4. [CLI Reference](CLI_REFERENCE.md) - Command details
5. [Troubleshooting](TROUBLESHOOTING.md) - If issues arise

### "I want to extract audio features (prosody, emotion)"
1. [Audio Enrichment Guide](AUDIO_ENRICHMENT.md) - Overview
2. [Quickstart Audio Enrichment](QUICKSTART_AUDIO_ENRICHMENT.md) - Quick start
3. [Prosody Guide](PROSODY.md) - Detailed prosody documentation
4. [Examples](../examples/) - Working examples

### "I want to use the Python API"
1. [Migrating from faster-whisper](FASTER_WHISPER_MIGRATION.md) - Drop-in replacement guide
2. [API Quick Reference](API_QUICK_REFERENCE.md) - Function reference
3. [API Examples](../examples/API_EXAMPLES_README.md) - Usage examples
4. [Architecture](ARCHITECTURE.md) - Understanding the design

### "I want to analyze enriched transcripts with LLMs"
1. [LLM Prompt Patterns](LLM_PROMPT_PATTERNS.md) - Reference prompts and rendering strategies
2. [LLM Semantic Annotator](LLM_SEMANTIC_ANNOTATOR.md) - Cloud LLM integration for semantic annotation
3. [LLM Integration Examples](../examples/llm_integration/README.md) - Working scripts (summarization, QA scoring, coaching)
4. [README Python API](../README.md#llm-integration-analyze-conversations) - Quick API example with speaker labels
5. [Audio Enrichment Guide](AUDIO_ENRICHMENT.md) - Understand the prosody/emotion features

### "I want to stream audio in real-time"
1. [Streaming API Reference](STREAMING_API.md) - WebSocket and SSE endpoints
2. [Streaming Architecture](STREAMING_ARCHITECTURE.md) - Internal design
3. [API Service](API_SERVICE.md) - REST API setup

### "I want to use the advantage layer (store, outcomes, webhooks, RAG)"
1. [Advantage Layer Quickstart](ADVANTAGE_LAYER_QUICKSTART.md) - Complete end-to-end workflow
2. CLI: `slower-whisper store` - Conversation store commands
3. CLI: `slower-whisper outcomes` - Outcomes extraction
4. CLI: `slower-whisper privacy` - Privacy/redaction tools
5. CLI: `slower-whisper webhook` - Webhook delivery
6. CLI: `slower-whisper rag` - RAG bundle export
7. CLI: `slower-whisper speakers` - Speaker identity management
8. CLI: `slower-whisper doctor` - System diagnostics

### "I want to run benchmarks and evaluate quality"
1. [Benchmarks](BENCHMARKS.md) - Benchmark CLI reference
2. [Semantic Benchmark](SEMANTIC_BENCHMARK.md) - LLM-based quality evaluation
3. [Getting Started Evaluation](GETTING_STARTED_EVALUATION.md) - Evaluation framework intro
4. [AMI Setup](AMI_SETUP.md) - Diarization benchmark dataset
5. [IEMOCAP Setup](IEMOCAP_SETUP.md) - Emotion benchmark dataset
6. [LibriSpeech Quickstart](LIBRISPEECH_QUICKSTART.md) - ASR benchmark dataset

### "I want to deploy this in production"
1. [Docker Guide](DOCKER.md) - Containerization
2. [Docker Deployment Guide](DOCKER_DEPLOYMENT_GUIDE.md) - Production deployment
3. [API Service](API_SERVICE.md) - Web API setup
4. [Performance](PERFORMANCE.md) - Performance tuning

### "I want to contribute code"
1. [Contributing Guide](../CONTRIBUTING.md) - Process and standards
2. [CLAUDE.md](../CLAUDE.md) - Codebase guidance
3. [Architecture](ARCHITECTURE.md) - System design
4. [Typing Policy](TYPING_POLICY.md) - Type annotation conventions
5. [BDD/IaC Contracts](BDD_IAC_LOCKDOWN.md) - Testing and deployment standards
6. [Audit Infrastructure](audit/README.md) - DevLT, receipts, failure modes

### "I want to understand the system design"
1. [Architecture](ARCHITECTURE.md) - Complete architecture overview
2. [API Quick Reference](API_QUICK_REFERENCE.md) - API structure
3. [Data Flow Analysis](DATA_FLOW_ANALYSIS.md) - Data transformations
4. [Prosody Guide](PROSODY.md) - Feature extraction details

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
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper implementation (slower-whisper provides a [drop-in replacement](FASTER_WHISPER_MIGRATION.md))
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

### Complete
- Core documentation (README, CHANGELOG, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT)
- Configuration guide (defaults, environment variables, config files, CLI flags, precedence)
- API and CLI references
- Quickstart guides
- Architecture documentation
- Audio enrichment documentation
- Prosody documentation
- Examples documentation
- Benchmarks documentation (including semantic benchmarks)
- Docker and deployment guides
- LLM integration documentation (including semantic annotator)
- Streaming documentation

### Archived
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
- **Benchmarks**: Check BENCHMARKS.md or SEMANTIC_BENCHMARK.md
- **LLM integration**: Check LLM_PROMPT_PATTERNS.md or LLM_SEMANTIC_ANNOTATOR.md

---

**Last Updated:** 2026-02-16

**Feedback:** Open an issue or discussion on GitHub if you have suggestions for improving the documentation.
