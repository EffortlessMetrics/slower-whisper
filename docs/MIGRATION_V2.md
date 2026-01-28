# Migration Guide: v1.x to v2.0.0

**Version:** v2.0.0 (Target: Q3-Q4 2026)
**Last Updated:** 2025-12-31

This guide provides comprehensive migration instructions for upgrading from slower-whisper v1.x to v2.0.0.

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Migration Checklist](#pre-migration-checklist)
3. [Breaking Changes Summary](#breaking-changes-summary)
4. [Step-by-Step Migration](#step-by-step-migration)
5. [CLI Changes](#cli-changes)
6. [API Changes](#api-changes)
7. [Schema Changes](#schema-changes)
8. [New Features in v2.0.0](#new-features-in-v200)
9. [Testing Your Migration](#testing-your-migration)
10. [Rollback Guidance](#rollback-guidance)
11. [FAQ](#faq)

---

## Overview

v2.0.0 represents a significant evolution of slower-whisper, introducing:

- **Real-time streaming architecture** with WebSocket support
- **LLM-backed semantic annotation** (local and cloud providers)
- **Expanded benchmark coverage** with CI performance gates
- **Cleanup of deprecated v1.x APIs** for a cleaner codebase

### Version Philosophy

slower-whisper follows a **layered evolution** philosophy:

- **v1.x** - Stabilization, enrichment, and modularization (current)
- **v2.x** - Real-time streaming and architectural extensibility
- **v3.x** - Semantic understanding and domain specialization (future)

**Key Principle**: v1.x JSON transcripts remain **forward-compatible** with v2.x readers. Your existing transcripts will continue to work.

### Long-Term Support

- v1.x receives **security updates for 18 months** after v2.0.0 release
- **Critical bug fixes** for 12 months after the LTS period
- We recommend upgrading to v2.0.0 to access streaming and LLM features

---

## Pre-Migration Checklist

Before upgrading, complete these steps:

- [ ] **Backup transcripts**: Copy your `whisper_json/` directory
- [ ] **Check deprecation warnings**: Run your v1.x workflow and note any deprecation warnings
- [ ] **Review your CLI usage**: Check if you use any deprecated flags or scripts
- [ ] **Test in staging**: Upgrade in a non-production environment first
- [ ] **Update dependencies**: Ensure you have Python 3.12+ (required for v2.0.0)
- [ ] **Review config files**: Check for deprecated configuration keys

### Minimum Requirements for v2.0.0

| Requirement | v1.x | v2.0.0 |
|-------------|------|--------|
| Python | 3.9+ | 3.12+ |
| faster-whisper | 0.9+ | 1.0+ |
| torch (optional) | 2.0+ | 2.1+ |
| ffmpeg | any | any |

---

## Breaking Changes Summary

The following table summarizes all breaking changes from v1.x to v2.0.0:

| Change | v1.x Behavior | v2.0.0 Behavior | Migration Action |
|--------|---------------|-----------------|------------------|
| `--enrich-config` flag | Deprecated alias for `--config` | **Removed** | Use `--config` instead |
| `transcribe_pipeline.py` script | Functional wrapper | **Removed** | Use `slower-whisper transcribe` CLI |
| `audio_enrich.py` script | Functional wrapper | **Removed** | Use `slower-whisper enrich` CLI |
| `slower-whisper-enrich` entry point | Legacy entry point | **Removed** | Use `slower-whisper enrich` |
| `annotations.semantic.version` | `"1.0.0"` | `"2.0.0"` | Auto-upgraded on load; review semantic field structure |

### Deprecation Timeline

The deprecated items followed our standard deprecation policy:

```text
v1.3.0: Legacy scripts deprecated (announcement + warning)
v1.8.0: --enrich-config deprecated (announcement + warning)
v1.9.0: Deprecation warnings logged during usage
v2.0.0: Deprecated items removed
```

---

## Step-by-Step Migration

### Step 1: Update Your Installation

```bash
# Option 1: Using uv (recommended)
uv pip install --upgrade slower-whisper

# Option 2: Using pip
pip install --upgrade slower-whisper

# Verify version
slower-whisper --version
# Should output: slower-whisper 2.0.0
```

### Step 2: Update CLI Commands

Replace deprecated CLI patterns:

**Before (v1.x):**

```bash
# Legacy scripts (REMOVED in v2.0.0)
python scripts/transcribe_pipeline.py --language en
python scripts/audio_enrich.py --skip-existing

# Deprecated flag (REMOVED in v2.0.0)
slower-whisper enrich --enrich-config config.json
```

**After (v2.0.0):**

```bash
# Modern unified CLI
slower-whisper transcribe --language en
slower-whisper enrich --skip-existing

# Use --config for both transcribe and enrich
slower-whisper enrich --config config.json
```

### Step 3: Update Shell Scripts and Automation

Search your codebase for deprecated patterns:

```bash
# Find usage of deprecated scripts
grep -r "transcribe_pipeline.py" .
grep -r "audio_enrich.py" .

# Find usage of deprecated flags
grep -r "\-\-enrich-config" .

# Find usage of legacy entry point
grep -r "slower-whisper-enrich" .
```

Update each occurrence as described in [CLI Changes](#cli-changes).

### Step 4: Update Python Code

If you import or call slower-whisper from Python, update your code:

**Before (v1.x):**

```python
# Legacy import patterns still work, but deprecated items removed
from transcription import (
    transcribe_directory,
    enrich_directory,
    TranscriptionConfig,
    EnrichmentConfig,
)
```

**After (v2.0.0):**

```python
# Same imports - core API remains stable
from transcription import (
    transcribe_directory,
    enrich_directory,
    TranscriptionConfig,
    EnrichmentConfig,
    # New v2.0.0 features
    StreamingTranscriber,      # Real-time streaming
    SemanticLLMConfig,         # LLM-backed annotation
)
```

### Step 5: Verify Your Transcripts Load Correctly

Existing v1.x transcripts are forward-compatible:

```python
from transcription import load_transcript

# v1.x transcripts load without issues
transcript = load_transcript("whisper_json/old_transcript.json")

# Check schema version (will be 2)
print(f"Schema version: {transcript.meta.get('schema_version', 'unknown')}")
```

---

## CLI Changes

### Removed Commands and Flags

| Removed | Replacement | Notes |
|---------|-------------|-------|
| `python scripts/transcribe_pipeline.py` | `slower-whisper transcribe` | Full replacement |
| `python scripts/audio_enrich.py` | `slower-whisper enrich` | Full replacement |
| `slower-whisper-enrich` | `slower-whisper enrich` | Same functionality via subcommand |
| `--enrich-config FILE` | `--config FILE` | Unified flag name for consistency |

### Updated Command Examples

**Transcription:**

```bash
# v1.x (deprecated)
python scripts/transcribe_pipeline.py --language en --model large-v3

# v2.0.0 (current)
slower-whisper transcribe --language en --model large-v3
```

**Enrichment:**

```bash
# v1.x (deprecated)
python scripts/audio_enrich.py --skip-existing --enrich-config config.json
slower-whisper-enrich --enable-prosody

# v2.0.0 (current)
slower-whisper enrich --skip-existing --config config.json
slower-whisper enrich --enable-prosody
```

**Full Workflow:**

```bash
# v2.0.0 recommended workflow
slower-whisper transcribe --language en --enable-diarization
slower-whisper enrich --enable-speaker-analytics
slower-whisper export whisper_json/meeting.json --format csv
slower-whisper validate whisper_json/meeting.json
```

### New CLI Commands in v2.0.0

| Command | Description |
|---------|-------------|
| `slower-whisper benchmark` | Run performance benchmarks |
| `slower-whisper stream` | Start WebSocket streaming server |

```bash
# Run ASR benchmark
slower-whisper benchmark --track asr --dataset librispeech

# Run diarization benchmark
slower-whisper benchmark --track diarization --dataset ami

# Start streaming server
slower-whisper stream --port 8000 --enable-enrichment
```

---

## API Changes

### Stable APIs (No Changes Required)

The following core APIs remain stable and **require no changes**:

```python
# Core transcription
transcribe_directory(root, config)
transcribe_file(audio_path, root, config)

# Core enrichment
enrich_directory(root, config)
enrich_transcript(transcript, audio_path, config)

# I/O
load_transcript(path)
save_transcript(transcript, path)

# LLM rendering
render_conversation_for_llm(transcript, ...)
render_conversation_compact(transcript, ...)
render_segment(segment, ...)
to_turn_view(transcript, ...)
to_speaker_summary(transcript, ...)
```

### New APIs in v2.0.0

**Real-Time Streaming:**

```python
from transcription import StreamingTranscriber

# Initialize real-time transcriber
transcriber = StreamingTranscriber(
    model="large-v3",
    device="cuda",
    callbacks=MyCallbacks()  # Optional event callbacks
)

# Process audio stream
async for event in transcriber.transcribe_stream(audio_generator):
    if event.type == "partial":
        update_live_display(event.segment.text)
    elif event.type == "finalized":
        save_segment(event.segment)
```

**LLM-Backed Semantic Annotation:**

```python
from transcription import SemanticLLMConfig, LLMSemanticAnnotator

# Configure LLM backend
config = SemanticLLMConfig(
    backend="local",           # "local", "openai", or "anthropic"
    model="qwen2.5-7b",        # Local model name
    enable_topics=True,
    enable_risks=True,
    enable_actions=True,
)

# Annotate transcript
annotator = LLMSemanticAnnotator(config)
annotated = annotator.annotate(transcript)

# Access enhanced semantic data
semantic = annotated.annotations.get("semantic", {})
print(f"Topics: {semantic.get('topics', [])}")
print(f"Risks: {semantic.get('risks', [])}")
print(f"Actions: {semantic.get('actions', [])}")
```

### Callback Interface (v1.9.0+)

If you're upgrading from v1.7.0-v1.8.0 to v2.0.0, you can use the callback interface introduced in v1.9.0:

```python
from transcription.streaming import StreamCallbacks

class MyCallbacks(StreamCallbacks):
    def on_segment_finalized(self, segment):
        print(f"Finalized: {segment.text}")

    def on_semantic_update(self, payload):
        if "escalation" in payload.risk_tags:
            self.alert_manager(payload.turn)

    def on_error(self, error, recoverable):
        if not recoverable:
            self.abort_session()

# Use with streaming sessions
session = StreamingEnrichmentSession(
    config=config,
    callbacks=MyCallbacks()
)
```

---

## Schema Changes

### JSON Schema Version

The core JSON schema version remains **v2** with backward compatibility:

```json
{
  "schema_version": 2,
  "file": "audio.wav",
  "language": "en",
  "meta": { ... },
  "segments": [ ... ]
}
```

### Semantic Annotation Schema (v2.0.0)

The `annotations.semantic` structure is enhanced in v2.0.0:

**v1.x (Keyword-based):**

```json
{
  "annotations": {
    "semantic": {
      "version": "1.0.0",
      "keywords": ["proposal", "deadline"],
      "risk_tags": ["escalation"],
      "actions": [{"text": "Send proposal", "speaker_id": "spk_0", "segment_ids": [0]}]
    }
  }
}
```

**v2.0.0 (LLM-backed):**

```json
{
  "annotations": {
    "semantic": {
      "version": "2.0.0",
      "backend": "local",
      "model": "qwen2.5-7b",
      "topics": [
        {"label": "pricing", "confidence": 0.92, "span": [0, 5]}
      ],
      "risks": [
        {"type": "escalation", "severity": "high", "evidence": "I need to speak to a manager"}
      ],
      "actions": [
        {"description": "Send proposal", "assignee": null, "due": null}
      ]
    }
  }
}
```

### Auto-Upgrade Behavior

When loading v1.x transcripts with semantic annotations:

1. The `annotations.semantic` structure is preserved as-is
2. Re-running semantic annotation with v2.0.0 upgrades the version field
3. New fields (`topics`, `risks` with confidence) are populated
4. Old fields remain for backward compatibility

### Migration Script for Semantic Annotations

If you want to re-annotate transcripts with the new LLM backend:

```python
from pathlib import Path
from transcription import load_transcript, save_transcript
from transcription.semantic import LLMSemanticAnnotator, SemanticLLMConfig

def migrate_semantic_annotations(json_dir: str):
    """Re-annotate all transcripts with LLM-backed semantics."""
    config = SemanticLLMConfig(backend="local", model="qwen2.5-7b")
    annotator = LLMSemanticAnnotator(config)

    for json_path in Path(json_dir).glob("*.json"):
        transcript = load_transcript(json_path)

        # Skip if already v2.0.0 semantic
        semantic = (transcript.annotations or {}).get("semantic", {})
        if semantic.get("version") == "2.0.0":
            print(f"Skipping (already v2.0.0): {json_path.name}")
            continue

        # Re-annotate
        annotated = annotator.annotate(transcript)
        save_transcript(annotated, json_path)
        print(f"Migrated: {json_path.name}")

# Usage
migrate_semantic_annotations("whisper_json/")
```

---

## New Features in v2.0.0

### 1. Real-Time Streaming Architecture

WebSocket-based streaming with partial transcripts:

```bash
# Start streaming server
slower-whisper stream --port 8000

# Connect with WebSocket client
ws://localhost:8000/stream
```

REST companion endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream/sessions` | POST | Create new session |
| `/stream/sessions/{id}/status` | GET | Get session status |
| `/stream/sessions/{id}` | DELETE | Force-close session |

### 2. LLM-Backed Semantic Annotation

Configure semantic annotation with local or cloud LLMs:

```python
# Local model (default)
config = SemanticLLMConfig(
    backend="local",
    model="qwen2.5-7b",
)

# OpenAI
config = SemanticLLMConfig(
    backend="openai",
    model="gpt-4o-mini",
    rate_limit_rpm=60,
)

# Anthropic
config = SemanticLLMConfig(
    backend="anthropic",
    model="claude-3-haiku",
)
```

### 3. Expanded Benchmarks

Run benchmarks from CLI with performance gates:

```bash
# ASR benchmark (LibriSpeech)
slower-whisper benchmark --track asr --dataset librispeech

# Diarization benchmark (AMI)
slower-whisper benchmark --track diarization --dataset ami

# Streaming latency benchmark
slower-whisper benchmark --track streaming --duration 1h
```

Performance targets:

| Track | Metric | Target |
|-------|--------|--------|
| ASR | WER on LibriSpeech | < 5% |
| Diarization | DER on AMI | < 15% |
| Streaming | P95 Latency | < 500ms |
| Semantic | Topic F1 | > 0.8 |

### 4. Event Callback Interface

Standardized callbacks for streaming events:

```python
from transcription.streaming import StreamCallbacks

class ProductionCallbacks(StreamCallbacks):
    def on_segment_finalized(self, segment):
        # Persist to database
        self.db.insert(segment.to_dict())

    def on_semantic_update(self, payload):
        # Real-time alerting
        if payload.risk_tags:
            self.alert_system.notify(payload)

    def on_error(self, error, recoverable):
        self.metrics.record_error(error)
```

---

## Testing Your Migration

### Step 1: Run Verification Script

```bash
# Verify installation and basic functionality
slower-whisper-verify --quick

# Full verification (slower, requires audio files)
slower-whisper-verify
```

### Step 2: Test Core Workflows

```bash
# Test transcription
slower-whisper transcribe --root /path/to/test/project --language en

# Test enrichment
slower-whisper enrich --root /path/to/test/project

# Test export
slower-whisper export whisper_json/test.json --format csv

# Test validation
slower-whisper validate whisper_json/test.json
```

### Step 3: Test Python API

```python
from transcription import (
    transcribe_file,
    enrich_transcript,
    load_transcript,
    save_transcript,
    TranscriptionConfig,
    EnrichmentConfig,
)

# Test transcription
config = TranscriptionConfig(model="base", language="en")
transcript = transcribe_file("test.wav", "/project", config)
assert transcript is not None

# Test enrichment
enrich_config = EnrichmentConfig(enable_prosody=True)
enriched = enrich_transcript(transcript, "test.wav", enrich_config)
assert enriched.segments[0].audio_state is not None

# Test I/O
save_transcript(enriched, "output.json")
loaded = load_transcript("output.json")
assert loaded.file_name == enriched.file_name
```

### Step 4: Compare Outputs

Compare v1.x and v2.0.0 outputs for the same audio:

```python
import json

def compare_transcripts(v1_path: str, v2_path: str) -> bool:
    """Compare key fields between v1.x and v2.0.0 outputs."""
    with open(v1_path) as f:
        v1 = json.load(f)
    with open(v2_path) as f:
        v2 = json.load(f)

    # Compare segment count
    if len(v1["segments"]) != len(v2["segments"]):
        print(f"Segment count mismatch: {len(v1['segments'])} vs {len(v2['segments'])}")
        return False

    # Compare segment text
    for i, (s1, s2) in enumerate(zip(v1["segments"], v2["segments"])):
        if s1["text"] != s2["text"]:
            print(f"Text mismatch at segment {i}")
            return False

    print("Outputs match!")
    return True

compare_transcripts("v1_output.json", "v2_output.json")
```

---

## Rollback Guidance

If you encounter issues after upgrading, you can rollback:

### Option 1: Pin Version with pip/uv

```bash
# Rollback to last v1.x release
uv pip install slower-whisper==1.8.0

# Or with pip
pip install slower-whisper==1.8.0
```

### Option 2: Git Checkout (if installed from source)

```bash
cd slower-whisper
git checkout v1.8.0
uv sync --extra full
```

### Option 3: Use Docker Image Tag

```bash
# Use specific version tag
docker pull effortlessmetrics/slower-whisper:1.8.0
docker run effortlessmetrics/slower-whisper:1.8.0 transcribe
```

### Preserving Transcripts During Rollback

Your transcript files are forward-compatible but **not backward-compatible** if you've used v2.0.0-only features:

- **Safe to rollback**: Transcripts generated with core features (transcription, basic enrichment)
- **May need backup**: Transcripts with v2.0.0 semantic annotations (new schema fields)

```bash
# Before upgrading to v2.0.0, backup transcripts
cp -r whisper_json/ whisper_json_backup_v1/

# After rollback, restore if needed
cp -r whisper_json_backup_v1/ whisper_json/
```

---

## FAQ

### Q: Will my existing transcripts work with v2.0.0?

**A**: Yes. v2.0.0 is fully forward-compatible with v1.x transcripts. Your existing JSON files will load and work correctly. The only change is that re-running semantic annotation will upgrade the `annotations.semantic.version` field from `"1.0.0"` to `"2.0.0"`.

### Q: Do I need to re-transcribe my audio files?

**A**: No. Your existing transcriptions remain valid. Only re-transcribe if you want to use new features like word-level timestamps (v1.8+) or streaming transcription (v2.0.0).

### Q: What happens if I use the removed `--enrich-config` flag?

**A**: v2.0.0 will display an error message:

```
Error: No such option: --enrich-config
Hint: Use --config instead (--enrich-config was removed in v2.0.0)
```

### Q: Can I use both v1.x and v2.0.0 in the same project?

**A**: We recommend using a single version per project. If you need both, use separate virtual environments:

```bash
# v1.x environment
python -m venv venv_v1
source venv_v1/bin/activate
pip install slower-whisper==1.8.0

# v2.0.0 environment
python -m venv venv_v2
source venv_v2/bin/activate
pip install slower-whisper==2.0.0
```

### Q: How long will v1.x be supported?

**A**: v1.x receives security updates for 18 months after v2.0.0 release, and critical bug fixes for 12 months after the LTS period ends.

### Q: Do I need a GPU for v2.0.0 streaming features?

**A**: GPU is recommended for real-time streaming performance, but CPU mode is supported with higher latency. Performance targets:

| Mode | GPU | CPU |
|------|-----|-----|
| Streaming latency (P95) | < 300ms | < 1000ms |
| Concurrent sessions | > 10 | > 3 |

### Q: How do I configure the LLM backend for semantic annotation?

**A**: See the [New Features](#new-features-in-v200) section. You can use local models (default), OpenAI, or Anthropic:

```python
from transcription import SemanticLLMConfig

# Local (no API key needed)
config = SemanticLLMConfig(backend="local", model="qwen2.5-7b")

# OpenAI (requires OPENAI_API_KEY env var)
config = SemanticLLMConfig(backend="openai", model="gpt-4o-mini")

# Anthropic (requires ANTHROPIC_API_KEY env var)
config = SemanticLLMConfig(backend="anthropic", model="claude-3-haiku")
```

### Q: Where can I get help if I have migration issues?

**A**:

1. Check the [Troubleshooting](TROUBLESHOOTING.md) guide
2. Search [GitHub Issues](https://github.com/EffortlessMetrics/slower-whisper/issues)
3. Open a new issue with the `migration` label
4. Join community discussions (Discord/Slack - coming soon)

---

## Related Documentation

- [ROADMAP.md](/ROADMAP.md) - Development timeline and versioning philosophy
- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Complete CLI documentation
- [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md) - Python API reference
- [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) - Real-time streaming details
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration management
- [CHANGELOG.md](/CHANGELOG.md) - Version history

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-31 | Draft | Initial migration guide created |
