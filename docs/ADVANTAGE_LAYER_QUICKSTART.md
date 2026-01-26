# Advantage Layer Quickstart

**Status**: v2.0.0 smoke scenario validation

This document walks through a complete end-to-end workflow using the slower-whisper advantage layer: transcription, storage, outcomes extraction, privacy-safe export, webhook delivery, and RAG bundle generation.

If any step in this workflow requires "oh just import this internal module directly", that's a bug.

---

## Prerequisites

```bash
# Install with all features
pip install slower-whisper[full]

# Or via uv
uv sync --extra full --extra dev

# Verify installation
slower-whisper doctor
```

---

## The Smoke Scenario

The advantage layer provides seven operations that work from the CLI:

1. **Transcribe** an audio file
2. **Ingest** into the store
3. **Extract outcomes** with citations
4. **Produce a safe export** (redacted)
5. **Emit a webhook event**
6. **Export a RAG bundle**
7. **View telemetry** and doctor report

---

## Step 1: Transcribe Audio

Set up a project directory and transcribe audio files:

```bash
# Create project structure
mkdir -p my-project/raw_audio
cd my-project

# Add audio file (or use sample)
slower-whisper samples copy mini_diarization --root .

# Transcribe with diarization
slower-whisper transcribe --enable-diarization

# Output appears in whisper_json/
ls whisper_json/
```

Expected output structure:
```
my-project/
├── raw_audio/          # Original audio
├── input_audio/        # Normalized WAV
├── whisper_json/       # Transcripts ← use these
└── transcripts/        # (legacy)
```

---

## Step 2: Enrich Transcripts

Add prosody, emotion, and speaker analytics:

```bash
slower-whisper enrich

# Check enriched transcript
cat whisper_json/meeting.json | jq '.segments[0].audio_state'
```

---

## Step 3: Ingest into Store

Load transcripts into the queryable conversation store:

```bash
# Ingest a single transcript
slower-whisper store ingest whisper_json/meeting.json --tags team-meeting,2025-Q1

# Or ingest all transcripts in a directory
for f in whisper_json/*.json; do
  slower-whisper store ingest "$f" --on-duplicate skip
done

# Verify ingestion
slower-whisper store stats
```

Example output:
```
Conversation Store Statistics
========================================

Total segments:       142
Total transcripts:    3
Total speakers:       4
Total duration:       15.2m

Action items:
  Total:              5
  Open:               5
  Completed:          0

Storage size:         128.5 KB
```

---

## Step 4: Extract Outcomes with Citations

Extract decisions, action items, and risks:

```bash
# Extract outcomes from a transcript
slower-whisper outcomes extract whisper_json/meeting.json

# Output as JSON
slower-whisper outcomes extract whisper_json/meeting.json --json -o outcomes.json

# Use LLM backend for better extraction (requires API key)
slower-whisper outcomes extract whisper_json/meeting.json \
  --backend llm \
  --provider openai \
  --output outcomes.json
```

Example output:
```
=== Outcomes ===

DECISION (1):
  - We will proceed with the new vendor
    Citation: segments[23:25] (3:45-4:12)
    Confidence: 0.85

ACTION_ITEM (2):
  - Follow up with legal on contract terms
    Citation: segments[45] (8:22-8:35)
    Assignee: John

  - Send updated timeline to stakeholders
    Citation: segments[67:68] (12:01-12:18)
```

---

## Step 5: Produce Safe Export (Privacy)

Detect and redact PII before sharing:

```bash
# Detect PII in a transcript
slower-whisper privacy detect whisper_json/meeting.json

# Redact PII (creates meeting_redacted.json)
slower-whisper privacy redact whisper_json/meeting.json \
  --mode mask \
  --output safe_meeting.json

# Export with privacy mode
slower-whisper privacy export whisper_json/meeting.json \
  --mode redacted \
  -o share/meeting_safe.json
```

Redaction modes:
- `mask`: Replace with `[REDACTED]`
- `hash`: Replace with deterministic hash
- `placeholder`: Replace with type placeholder (`[EMAIL]`, `[PHONE]`)

---

## Step 6: Emit Webhook Event

Send transcripts to external systems:

```bash
# Test webhook endpoint
slower-whisper webhook test https://api.example.com/hook \
  --bearer-token $WEBHOOK_TOKEN

# Send transcript to webhook
slower-whisper webhook send whisper_json/meeting.json \
  --url https://api.example.com/hook \
  --bearer-token $WEBHOOK_TOKEN

# With HMAC signature verification
slower-whisper webhook send whisper_json/meeting.json \
  --url https://api.example.com/hook \
  --hmac-secret $HMAC_SECRET

# Retry failed deliveries
slower-whisper webhook retry \
  --url https://api.example.com/hook \
  --dead-letter-path dead_letters.json
```

---

## Step 7: Export RAG Bundle

Generate vector-database-ready chunks:

```bash
# Export with speaker turn chunking (default)
slower-whisper rag whisper_json/meeting.json -o meeting_rag.json

# Chunk by time windows
slower-whisper rag whisper_json/meeting.json \
  --strategy by_time \
  --time-window 30 \
  -o meeting_rag.json

# Include embeddings (requires sentence-transformers)
slower-whisper rag whisper_json/meeting.json \
  --embed \
  --embedding-model all-MiniLM-L6-v2 \
  -o meeting_rag.json
```

Chunking strategies:
- `by_segment`: One chunk per ASR segment
- `by_speaker_turn`: One chunk per speaker turn (default)
- `by_time`: Fixed time windows
- `by_topic`: Topic-based boundaries (experimental)

---

## Step 8: View Telemetry and Doctor Report

Check system health:

```bash
# Run diagnostics
slower-whisper doctor

# JSON output for monitoring
slower-whisper doctor --json
```

Example output:
```
=== slower-whisper Doctor Report ===

System:
  [PASS] Python 3.12.0
  [PASS] Platform: linux x86_64
  [PASS] Memory: 16.0 GB available

Core Dependencies:
  [PASS] faster-whisper 1.0.0
  [PASS] torch 2.1.0 (CUDA available)
  [PASS] numpy 1.26.0

Optional Features:
  [PASS] Diarization: pyannote.audio installed
  [PASS] Emotion: speechbrain installed
  [PASS] Speaker Identity: resemblyzer installed
  [WARN] Embeddings: sentence-transformers not installed

Storage:
  [PASS] Store: 142 segments, 128.5 KB
  [PASS] Speaker Registry: 4 speakers

Overall: PASS (1 warning)
```

---

## Query the Store

Once data is ingested, query it:

```bash
# Text search
slower-whisper store query "budget discussion"

# Filter by speaker
slower-whisper store query "deadline" --speaker SPEAKER_01

# Filter by date range
slower-whisper store query --during 2025-01-01-2025-01-31

# List recent entries
slower-whisper store list --limit 20

# Export to CSV
slower-whisper store export --format csv --output conversations.csv

# Manage action items
slower-whisper store actions list --status open
slower-whisper store actions complete <action-id>
```

---

## Speaker Identity (Cross-Session)

Register known speakers for cross-session identification:

```bash
# Register a speaker from an audio clip
slower-whisper speakers register "Alice" \
  --from-clip samples/alice_intro.wav \
  --start 0.0 --end 5.0

# List registered speakers
slower-whisper speakers list

# Match unknown audio against registry
slower-whisper speakers match unknown_segment.wav --threshold 0.7

# Apply identity mapping to transcript
slower-whisper speakers apply whisper_json/meeting.json \
  --audio input_audio/meeting.wav \
  --threshold 0.7

# Delete a speaker (privacy/GDPR)
slower-whisper speakers delete <speaker-id> --force
```

---

## Complete Workflow Script

Save as `smoke_test.sh`:

```bash
#!/bin/bash
set -e

echo "=== Advantage Layer Smoke Test ==="

# Setup
mkdir -p test-project && cd test-project
slower-whisper samples copy mini_diarization --root .

# 1. Transcribe
echo "[1/7] Transcribing..."
slower-whisper transcribe --enable-diarization

# 2. Enrich
echo "[2/7] Enriching..."
slower-whisper enrich

# 3. Ingest
echo "[3/7] Ingesting to store..."
TRANSCRIPT=$(ls whisper_json/*.json | head -1)
slower-whisper store ingest "$TRANSCRIPT" --tags smoke-test

# 4. Extract outcomes
echo "[4/7] Extracting outcomes..."
slower-whisper outcomes extract "$TRANSCRIPT" --json -o outcomes.json

# 5. Privacy export
echo "[5/7] Creating safe export..."
slower-whisper privacy export "$TRANSCRIPT" --mode redacted -o safe_transcript.json

# 6. RAG export
echo "[6/7] Exporting RAG bundle..."
slower-whisper rag "$TRANSCRIPT" -o rag_bundle.json

# 7. Doctor check
echo "[7/7] Running doctor..."
slower-whisper doctor

# Summary
echo ""
echo "=== Smoke Test Complete ==="
slower-whisper store stats
ls -la *.json

cd .. && rm -rf test-project
echo "[PASS] All steps completed"
```

---

## What's NOT in the Advantage Layer

These are **not** advantage layer features (they're upstream capabilities):

- ASR engine choice (faster-whisper vs others)
- VAD configuration
- Beam search parameters
- Language detection
- Audio normalization

The advantage layer adds **value on top of transcription**:
- Persistent storage and query
- Evidence-grade citations
- Privacy-by-default exports
- Integration sinks (webhooks, RAG)
- Cross-session speaker identity
- Operational telemetry

---

## Troubleshooting

### "No speaker embedding backend available"
```bash
pip install speechbrain  # or: pip install resemblyzer
```

### "Parquet export requires pyarrow"
```bash
pip install pyarrow
```

### "Missing HF_TOKEN for diarization"
```bash
export HF_TOKEN=hf_xxxxx  # from huggingface.co/settings/tokens
```

### Store is empty after ingest
Check the transcript has segments:
```bash
cat whisper_json/meeting.json | jq '.segments | length'
```

---

## Next Steps

- [BENCHMARKS.md](BENCHMARKS.md) - Measure accuracy and performance
- [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) - Real-time processing
- [LLM_SEMANTIC_ANNOTATOR.md](LLM_SEMANTIC_ANNOTATOR.md) - Cloud LLM integration
- [CONFIGURATION.md](CONFIGURATION.md) - Config file reference
