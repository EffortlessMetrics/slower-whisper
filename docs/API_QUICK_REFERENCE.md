# API Quick Reference

## Import

```python
from transcription import (
    # Core Functions
    transcribe_directory,
    transcribe_file,
    enrich_directory,
    enrich_transcript,
    load_transcript,
    save_transcript,
    # LLM Rendering Functions
    render_conversation_for_llm,
    render_conversation_compact,
    render_segment,
    to_turn_view,
    to_speaker_summary,
    # Streaming (post-ASR)
    StreamingSession,
    StreamConfig,
    StreamChunk,
    StreamEvent,
    # Config
    TranscriptionConfig,
    EnrichmentConfig,
    ChunkingConfig,
    # Models
    Transcript,
    Segment,
    Chunk,
    # Chunking
    build_chunks,
)
```

## Transcription

### Transcribe Directory (Batch)

```python
config = TranscriptionConfig(
    model="large-v3",
    language="en",
    device="cuda"
)

transcripts = transcribe_directory("/path/to/project", config)
```

### Single File

```python
config = TranscriptionConfig(model="base", language="en")

transcript = transcribe_file(
    audio_path="interview.mp3",
    root="/path/to/project",
    config=config
)
```

## Enrichment

### Enrich Directory (Batch)

```python
config = EnrichmentConfig(
    enable_prosody=True,
    enable_emotion=True,
    device="cpu"
)

enriched = enrich_directory("/path/to/project", config)
```

### Single Transcript

```python
config = EnrichmentConfig(enable_prosody=True)

enriched = enrich_transcript(
    transcript=transcript,
    audio_path="audio.wav",
    config=config
)
```

## I/O

```python
# Load
transcript = load_transcript("output.json")

# Save
save_transcript(transcript, "output.json")
```

## Streaming (post-ASR v0.1)

### Basic Streaming

```python
from transcription import StreamingSession, StreamConfig

session = StreamingSession(StreamConfig(max_gap_sec=1.0))
for chunk in asr_chunks:  # iterable of post-ASR chunks with start/end/text/speaker_id
    events = session.ingest_chunk(chunk)
    for event in events:
        handle_stream_event(event)

# Flush any remaining partial at EOS
for event in session.end_of_stream():
    handle_stream_event(event)
```

`StreamingSession` stitches ordered ASR chunks into `PartialSegment` and `FinalSegment` events. Current v0.1 API operates on post-ASR text (not raw audio) so you can stream final segments while Whisper runs elsewhere.

### Streaming + Semantic Annotation

Combine streaming with semantic tagging to detect risks and intent in real-time. Build a transcript incrementally and run `KeywordSemanticAnnotator` on finalized segments:

```python
from transcription import (
    StreamingSession,
    StreamConfig,
    Segment,
    Transcript,
    save_transcript,
)
from transcription.streaming import StreamEventType
from transcription.semantic import KeywordSemanticAnnotator

# Initialize streaming and annotation
session = StreamingSession(StreamConfig(max_gap_sec=1.0))
annotator = KeywordSemanticAnnotator()
final_segments: list[Segment] = []

# Process incoming ASR chunks
for chunk in asr_chunks:
    for event in session.ingest_chunk(chunk):
        if event.type != StreamEventType.FINAL_SEGMENT:
            continue

        # Convert StreamSegment → Segment
        stream_seg = event.segment
        segment = Segment(
            id=len(final_segments),
            start=stream_seg.start,
            end=stream_seg.end,
            text=stream_seg.text,
            speaker={"id": stream_seg.speaker_id} if stream_seg.speaker_id else None,
        )
        final_segments.append(segment)

        # Build growing transcript
        transcript = Transcript(
            file_name="live.wav",
            language="en",
            segments=list(final_segments)
        )

        # Run annotation on finalized segments
        annotated = annotator.annotate(transcript)
        semantic = (annotated.annotations or {}).get("semantic", {})

        # Check for risk tags
        risk_tags = semantic.get("risk_tags", [])
        if "escalation" in risk_tags:
            print("⚠️  Escalation detected - route to supervisor")
        if "churn_risk" in risk_tags:
            print("⚠️  Churn risk - offer retention options")

        # Extract actions and keywords
        for action in semantic.get("actions", []):
            print(f"Action: {action['text']} (speaker={action['speaker_id']})")

# Flush remaining partial at end of stream
for event in session.end_of_stream():
    if event.type == StreamEventType.FINAL_SEGMENT:
        # Same conversion and annotation logic
        stream_seg = event.segment
        segment = Segment(
            id=len(final_segments),
            start=stream_seg.start,
            end=stream_seg.end,
            text=stream_seg.text,
            speaker={"id": stream_seg.speaker_id} if stream_seg.speaker_id else None,
        )
        final_segments.append(segment)
        transcript = Transcript(
            file_name="live.wav",
            language="en",
            segments=list(final_segments)
        )
        annotated = annotator.annotate(transcript)
        # Process semantic annotations...

# Save final transcript
save_transcript(annotated, "output.json")
```

**Key points:**
- Only process `final_segment` events for stable semantic tags (skip `partial_segment`)
- Rebuild `Transcript` incrementally as segments finalize
- `risk_tags` include: `escalation`, `churn_risk`, `pricing`, etc.
- See [Pattern 7: Streaming Semantics](./LLM_PROMPT_PATTERNS.md#pattern-7-streaming-semantics-for-real-time-analysis) in LLM_PROMPT_PATTERNS.md for routing prompts by semantic tags

### CLI exports & validation

After transcribing, export/share and schema-check the JSON:

```bash
uv run slower-whisper export transcript.json --format csv --unit turns  # csv/html/vtt/textgrid
uv run slower-whisper validate transcript.json                          # uses built-in v2 schema
```

More examples: metrics/KPIs (`docs/METRICS_EXAMPLES.md`) and PII masking (`docs/REDACTION.md`).

## Configuration

### Configuration Precedence

Settings are loaded in order of priority:

```text
1. CLI flags (highest priority)
   ↓
2. Config file (--config or --enrich-config)
   ↓
3. Environment variables (SLOWER_WHISPER_*)
   ↓
4. Defaults (lowest priority)
```

### TranscriptionConfig

```python
TranscriptionConfig(
    model="large-v3",          # Whisper model name
    device="cuda",             # "cuda" or "cpu"
    compute_type=None,         # Precision (None = auto: float16 on CUDA, int8 on CPU)
    language=None,             # None = auto-detect
    task="transcribe",         # or "translate"
    skip_existing_json=True,   # Skip already transcribed
    vad_min_silence_ms=500,    # VAD silence threshold
    beam_size=5,               # Beam search size
)
```

**Field Details:**

| Field | Type | Default | Description | CLI Flag | Env Var |
|-------|------|---------|-------------|----------|---------|
| `model` | `str` | `"large-v3"` | Whisper model: tiny, base, small, medium, large, large-v2, large-v3 | `--model` | `SLOWER_WHISPER_MODEL` |
| `device` | `str` | `"cuda"` | Computation device: "cuda" or "cpu" | `--device` | `SLOWER_WHISPER_DEVICE` |
| `compute_type` | `str \| None` | `None` (auto) | Precision: auto picks float16 on CUDA, int8 on CPU. Supported: float16, float32, int16, int8, int8_float16, int8_float32 | `--compute-type` | `SLOWER_WHISPER_COMPUTE_TYPE` |
| `language` | `str \| None` | `None` | Language code (e.g., "en", "es") or None for auto-detect | `--language` | `SLOWER_WHISPER_LANGUAGE` |
| `task` | `str` | `"transcribe"` | Whisper task: "transcribe" or "translate" | `--task` | `SLOWER_WHISPER_TASK` |
| `skip_existing_json` | `bool` | `True` | Skip files with existing JSON output | `--skip-existing-json` | `SLOWER_WHISPER_SKIP_EXISTING_JSON` |
| `vad_min_silence_ms` | `int` | `500` | VAD silence threshold in milliseconds (100-2000) | `--vad-min-silence-ms` | `SLOWER_WHISPER_VAD_MIN_SILENCE_MS` |
| `beam_size` | `int` | `5` | Beam search size (1-10, higher = more accurate but slower) | `--beam-size` | `SLOWER_WHISPER_BEAM_SIZE` |
| `enable_diarization` | `bool` | `False` | Enable speaker diarization (pyannote.audio) | `--enable-diarization` | `SLOWER_WHISPER_ENABLE_DIARIZATION` |
| `diarization_device` | `str` | `"auto"` | Device for diarization: "cuda", "cpu", or "auto" | `--diarization-device` | `SLOWER_WHISPER_DIARIZATION_DEVICE` |
| `min_speakers` | `int \| None` | `None` | Minimum expected speakers (hint for diarization model) | `--min-speakers` | `SLOWER_WHISPER_MIN_SPEAKERS` |
| `max_speakers` | `int \| None` | `None` | Maximum expected speakers (hint for diarization model) | `--max-speakers` | `SLOWER_WHISPER_MAX_SPEAKERS` |
| `overlap_threshold` | `float` | `0.3` | Minimum overlap ratio to assign a speaker to a segment | `--overlap-threshold` | `SLOWER_WHISPER_OVERLAP_THRESHOLD` |
| _backend mode_ | `str` | `"auto"` | pyannote backend selector: `auto`, `stub` (fake), `missing` (simulate dep error) | _env only_ | `SLOWER_WHISPER_PYANNOTE_MODE` |
| _hf token_ | `str` | _none_ | Hugging Face token for pyannote backend (required unless forcing stub/missing) | _env only_ | `HF_TOKEN` |

**Model Sizes:**

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~75MB | Fastest | Lowest | Quick testing, resource-constrained |
| `base` | ~150MB | Very fast | Basic | Development, low-resource environments |
| `small` | ~500MB | Fast | Good | Balanced speed/quality |
| `medium` | ~1.5GB | Medium | Better | Higher quality needs |
| `large-v2` | ~3GB | Slow | Best | Production quality |
| `large-v3` | ~3GB | Slow | Best | Latest, recommended for production |

**Compute Types:**

| Type | Precision | Speed | Quality | GPU Required |
|------|-----------|-------|---------|--------------|
| `float32` | Full | Slowest | Best | No |
| `float16` | Half | Fast | Excellent | Yes (recommended) |
| `int8` | Quantized | Fastest | Good | No |
| `int8_float16` | Mixed | Very fast | Very good | Yes |
| `int16` | 16-bit | Medium | Good | No |
| `int8_float32` | Mixed | Fast | Good | No |

**Loading Methods:**

```python
# From file
config = TranscriptionConfig.from_file("config.json")

# From environment
config = TranscriptionConfig.from_env()

# Direct construction
config = TranscriptionConfig(model="base", language="en")
```

### EnrichmentConfig

```python
EnrichmentConfig(
    skip_existing=True,               # Skip already enriched
    enable_prosody=True,              # Pitch, energy, rate
    enable_emotion=True,              # Dimensional emotion
    enable_categorical_emotion=False, # Categorical (slower)
    enable_turn_metadata=True,        # Turn-level analytics (questions, pauses, disfluency)
    enable_speaker_stats=True,        # Per-speaker aggregates
    device="cpu",                     # "cpu" or "cuda"
    dimensional_model_name=None,      # Override dimensional model
    categorical_model_name=None,      # Override categorical model
)
```

**Field Details:**

| Field | Type | Default | Description | CLI Flag | Env Var |
|-------|------|---------|-------------|----------|---------|
| `skip_existing` | `bool` | `True` | Skip segments with existing audio_state | `--skip-existing` | `SLOWER_WHISPER_ENRICH_SKIP_EXISTING` |
| `enable_prosody` | `bool` | `True` | Extract pitch, energy, speech rate, pauses | `--enable-prosody` | `SLOWER_WHISPER_ENRICH_ENABLE_PROSODY` |
| `enable_emotion` | `bool` | `True` | Extract dimensional emotion (valence/arousal/dominance) | `--enable-emotion` | `SLOWER_WHISPER_ENRICH_ENABLE_EMOTION` |
| `enable_categorical_emotion` | `bool` | `False` | Extract categorical emotions (angry, happy, etc.) - slower | `--enable-categorical-emotion` | `SLOWER_WHISPER_ENRICH_ENABLE_CATEGORICAL_EMOTION` |
| `enable_turn_metadata` | `bool` | `True` | Populate `turns[].metadata` (questions, interruptions, pauses, disfluency) | `--enable-turn-metadata` | `SLOWER_WHISPER_ENRICH_ENABLE_TURN_METADATA` |
| `enable_speaker_stats` | `bool` | `True` | Compute `speaker_stats[]` aggregates | `--enable-speaker-stats` | `SLOWER_WHISPER_ENRICH_ENABLE_SPEAKER_STATS` |
| `device` | `str` | `"cpu"` | Device for emotion models: "cpu" or "cuda" | `--device` | `SLOWER_WHISPER_ENRICH_DEVICE` |
| `dimensional_model_name` | `str \| None` | `None` | HuggingFace model name for dimensional emotion | N/A | `SLOWER_WHISPER_ENRICH_DIMENSIONAL_MODEL_NAME` |
| `categorical_model_name` | `str \| None` | `None` | HuggingFace model name for categorical emotion | N/A | `SLOWER_WHISPER_ENRICH_CATEGORICAL_MODEL_NAME` |

Convenience flag: `--enable-speaker-analytics/--no-enable-speaker-analytics` toggles both `enable_turn_metadata` and `enable_speaker_stats` together.

**Feature Comparison:**

| Feature | Dependencies | Model Size | Speed | What It Extracts |
|---------|--------------|------------|-------|------------------|
| Prosody | librosa, parselmouth | ~50MB | Fast | Pitch (Hz), energy (dB), rate (syllables/sec), pauses |
| Dimensional Emotion | torch, transformers | ~1.5GB | Medium | Valence (positive/negative), arousal (calm/excited), dominance |
| Categorical Emotion | torch, transformers | ~1.5GB | Slower | Emotion labels: angry, happy, sad, frustrated, etc. |

**Loading Methods:**

```python
# From file
config = EnrichmentConfig.from_file("enrich_config.json")

# From environment
config = EnrichmentConfig.from_env()

# Direct construction
config = EnrichmentConfig(enable_prosody=True, device="cuda")
```

### Configuration File Examples

**Transcription JSON:**

```json
{
  "model": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "language": "en",
  "task": "transcribe",
  "skip_existing_json": true,
  "vad_min_silence_ms": 500,
  "beam_size": 5
}
```

**Transcription JSON (with diarization):**

```json
{
  "model": "large-v3",
  "enable_diarization": true,
  "diarization_device": "auto",
  "min_speakers": 2,
  "max_speakers": 4,
  "overlap_threshold": 0.3
}
```

**Enrichment JSON:**

```json
{
  "skip_existing": true,
  "enable_prosody": true,
  "enable_emotion": true,
  "enable_categorical_emotion": false,
  "device": "cpu",
  "dimensional_model_name": null,
  "categorical_model_name": null
}
```

See [examples/config_examples/](examples/config_examples/) for more examples.

## Access Results

### Transcript Structure

```python
transcript.file_name    # str
transcript.language     # str
transcript.segments     # list[Segment]
transcript.meta         # dict | None
```

### Segment Structure

```python
segment.id              # int
segment.start           # float (seconds)
segment.end             # float (seconds)
segment.text            # str
segment.speaker         # str | None
segment.audio_state     # dict | None
```

### Audio State (if enriched)

```python
if segment.audio_state:
    # Compact rendering
    print(segment.audio_state["rendering"])
    # "[audio: high pitch, loud volume, fast speech]"

    # Prosody features
    prosody = segment.audio_state["prosody"]
    print(prosody["pitch"]["level"])      # "high", "low", "neutral"
    print(prosody["energy"]["level"])     # "loud", "quiet", "normal"
    print(prosody["rate"]["level"])       # "fast", "slow", "normal"

    # Emotion features
    emotion = segment.audio_state["emotion"]
    print(emotion["valence"]["level"])    # "positive", "negative", "neutral"
    print(emotion["arousal"]["level"])    # "high", "low", "medium"
```

## CLI Quick Reference

Use the unified `slower-whisper` entry point with subcommands.

### Transcribe

```bash
uv run slower-whisper transcribe [OPTIONS]
```

| Option | Default | Notes |
|--------|---------|-------|
| `--root PATH` | `.` | Project root with `raw_audio/` |
| `--config FILE` | `None` | Merge order: CLI > file > env > defaults |
| `--model NAME` | `large-v3` | Whisper model |
| `--device {cuda,cpu}` | `cuda` | Auto-fallbacks to CPU if GPU load fails |
| `--compute-type TYPE` | auto (`float16` on CUDA, `int8` on CPU) | Override faster-whisper precision |
| `--language CODE` | auto | Force language |
| `--task {transcribe,translate}` | `transcribe` | Translate outputs English |
| `--vad-min-silence-ms INT` | `500` | VAD split threshold |
| `--beam-size INT` | `5` | Beam search size |
| `--skip-existing-json / --no-skip-existing-json` | `True` | Reuse existing JSON |
| `--enable-diarization` | `False` | Experimental speaker diarization (pyannote) |
| `--diarization-device {auto,cuda,cpu}` | `auto` | Device for diarization |
| `--min-speakers INT` | `None` | Lower bound hint |
| `--max-speakers INT` | `None` | Upper bound hint |
| `--overlap-threshold FLOAT` | `0.3` | Min overlap to assign a speaker |

Quick starts:

```bash
uv run slower-whisper transcribe --language en
uv run slower-whisper transcribe --enable-diarization --min-speakers 2 --max-speakers 4
```

### Enrich

```bash
uv run slower-whisper enrich [OPTIONS]
```

| Option | Default | Notes |
|--------|---------|-------|
| `--root PATH` | `.` | Project root with `whisper_json/` + `input_audio/` |
| `--enrich-config FILE` | `None` | Merge order: CLI > file > env > defaults |
| `--skip-existing / --no-skip-existing` | `True` | Skip segments with `audio_state` |
| `--enable-prosody / --no-enable-prosody` | `True` | Pitch/energy/rate/pauses |
| `--enable-emotion / --no-enable-emotion` | `True` | Dimensional emotion |
| `--enable-categorical-emotion / --no-enable-categorical-emotion` | `False` | Categorical emotion (slower) |
| `--device {cpu,cuda}` | `cpu` | Device for emotion models |

Quick starts:

```bash
uv run slower-whisper enrich
uv run slower-whisper enrich --enable-categorical-emotion --device cuda
uv run slower-whisper enrich --no-enable-emotion   # prosody only
```

## Chunking for RAG

slower-whisper provides intelligent chunking to split transcripts into RAG-friendly segments that respect conversational boundaries.

### Import

```python
from transcription import (
    build_chunks,
    ChunkingConfig,
    Chunk,
    load_transcript,
)
```

### ChunkingConfig

Configuration for chunk generation with soft and hard limits.

```python
ChunkingConfig(
    target_duration_s=30.0,       # Soft target duration before considering a split
    max_duration_s=45.0,          # Hard maximum duration for a chunk
    target_tokens=400,            # Soft target token budget for a chunk
    pause_split_threshold_s=1.5,  # Pause length that triggers split when near target
)
```

**Field Details:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_duration_s` | `float` | `30.0` | Soft target duration in seconds; chunk will split at next pause after reaching this |
| `max_duration_s` | `float` | `45.0` | Hard maximum duration in seconds; forces split even without pause |
| `target_tokens` | `int` | `400` | Soft target token count (estimated at ~1.3 tokens per word) |
| `pause_split_threshold_s` | `float` | `1.5` | Minimum pause duration in seconds that qualifies as a split point |

**Chunking Strategy:**

The chunking algorithm:
1. Prefers to split at turn boundaries when available (speaker changes)
2. Waits for natural pauses (≥ `pause_split_threshold_s`) after reaching soft targets
3. Forces splits at hard limits (`max_duration_s` or exceeding `target_tokens`)
4. Falls back to segment-based chunking if no turns are available

### build_chunks

Build stable, turn-aware chunks for retrieval applications.

**Signature:**

```python
build_chunks(transcript: Transcript, config: ChunkingConfig) -> list[Chunk]
```

**Parameters:**

- `transcript`: Transcript object loaded from JSON
- `config`: ChunkingConfig with target durations and token limits

**Returns:** List of `Chunk` objects with stable IDs and metadata

**Example:**

```python
from transcription import load_transcript, build_chunks, ChunkingConfig

# Load transcript
transcript = load_transcript("whisper_json/meeting.json")

# Default chunking (30s target, 45s max, 400 tokens)
chunks = build_chunks(transcript, ChunkingConfig())

# Custom chunking for shorter segments
config = ChunkingConfig(
    target_duration_s=20.0,
    max_duration_s=30.0,
    target_tokens=250,
    pause_split_threshold_s=1.0,
)
chunks = build_chunks(transcript, config)

# Access chunk data
for chunk in chunks:
    print(f"Chunk {chunk.id}: {chunk.start:.1f}s - {chunk.end:.1f}s")
    print(f"  Speakers: {', '.join(chunk.speaker_ids)}")
    print(f"  Tokens: ~{chunk.token_count_estimate}")
    print(f"  Segments: {chunk.segment_ids}")
    print(f"  Turns: {chunk.turn_ids}")
    print(f"  Text: {chunk.text[:100]}...")
```

**Note:** `build_chunks()` mutates the transcript by adding a `chunks` attribute for convenience. Subsequent calls can reuse `transcript.chunks` if desired.

### Chunk Dataclass

Represents a single chunk of conversation optimized for retrieval.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique chunk identifier (e.g., "chunk_0", "chunk_1") |
| `start` | `float` | Start time in seconds |
| `end` | `float` | End time in seconds |
| `segment_ids` | `list[int]` | List of segment IDs included in this chunk |
| `turn_ids` | `list[str]` | List of turn IDs included in this chunk (if turns available) |
| `speaker_ids` | `list[str]` | Sorted list of unique speaker IDs in this chunk |
| `token_count_estimate` | `int` | Estimated token count (~1.3 tokens per word) |
| `text` | `str` | Concatenated text from all segments in chunk |

**Methods:**

```python
# Serialize to dict
chunk_dict = chunk.to_dict()

# Deserialize from dict
chunk = Chunk.from_dict(chunk_dict)
```

### RAG Workflow Examples

**1. Basic Chunking for Vector Store:**

```python
from transcription import load_transcript, build_chunks, ChunkingConfig

# Load and chunk
transcript = load_transcript("whisper_json/support_call.json")
chunks = build_chunks(transcript, ChunkingConfig())

# Prepare for vector database
documents = []
for chunk in chunks:
    documents.append({
        "id": chunk.id,
        "text": chunk.text,
        "metadata": {
            "source": transcript.file_name,
            "start_time": chunk.start,
            "end_time": chunk.end,
            "speakers": chunk.speaker_ids,
            "duration": chunk.end - chunk.start,
            "token_estimate": chunk.token_count_estimate,
        }
    })

# Now embed and index documents in your vector store
```

**2. Semantic Search with Speaker Context:**

```python
from transcription import load_transcript, build_chunks, ChunkingConfig

transcript = load_transcript("whisper_json/interview.json")
config = ChunkingConfig(target_duration_s=25.0, target_tokens=300)
chunks = build_chunks(transcript, config)

# Build retrieval corpus with speaker labels
speaker_labels = {"spk_0": "Interviewer", "spk_1": "Candidate"}

retrieval_docs = []
for chunk in chunks:
    # Add speaker context to text
    speakers = ", ".join(speaker_labels.get(spk, spk) for spk in chunk.speaker_ids)
    enriched_text = f"[Speakers: {speakers}]\n{chunk.text}"

    retrieval_docs.append({
        "chunk_id": chunk.id,
        "text": enriched_text,
        "time_window": f"{chunk.start:.1f}s - {chunk.end:.1f}s",
        "speakers": chunk.speaker_ids,
    })

# Use for semantic search queries like:
# "What did the candidate say about Python experience?"
```

**3. Multi-Document RAG Pipeline:**

```python
from pathlib import Path
from transcription import load_transcript, build_chunks, ChunkingConfig

def index_transcript_directory(json_dir: str, config: ChunkingConfig):
    """Index all transcripts in a directory for RAG."""
    documents = []

    for json_path in Path(json_dir).glob("*.json"):
        transcript = load_transcript(json_path)
        chunks = build_chunks(transcript, config)

        for chunk in chunks:
            documents.append({
                "doc_id": f"{transcript.file_name}#{chunk.id}",
                "source_file": transcript.file_name,
                "chunk_id": chunk.id,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "speakers": chunk.speaker_ids,
                "tokens": chunk.token_count_estimate,
            })

    return documents

# Index all meeting transcripts
config = ChunkingConfig(target_duration_s=30.0, target_tokens=400)
all_docs = index_transcript_directory("whisper_json/", config)

# Ready for embedding and vector database ingestion
print(f"Indexed {len(all_docs)} chunks across all transcripts")
```

**4. Turn-Preserving Chunks for Dialogue Analysis:**

```python
from transcription import load_transcript, build_chunks, ChunkingConfig

# Load transcript with diarization and turns
transcript = load_transcript("whisper_json/debate.json")

# Configure for longer chunks that preserve full turns
config = ChunkingConfig(
    target_duration_s=45.0,      # Longer to capture complete exchanges
    max_duration_s=60.0,
    target_tokens=500,
    pause_split_threshold_s=2.0,  # Only split at longer pauses
)

chunks = build_chunks(transcript, config)

# Analyze conversational dynamics within chunks
for chunk in chunks:
    if len(chunk.speaker_ids) > 1:
        # Multi-speaker chunk - capture dialogue
        print(f"Exchange in {chunk.id}:")
        print(f"  Participants: {', '.join(chunk.speaker_ids)}")
        print(f"  Turn count: {len(chunk.turn_ids)}")
        print(f"  Duration: {chunk.end - chunk.start:.1f}s")
        print(f"  Preview: {chunk.text[:150]}...")
```

**5. Integration with LangChain/LlamaIndex:**

slower-whisper provides pre-built loaders for popular RAG frameworks:

```python
# LangChain integration
from integrations.langchain_loader import SlowerWhisperLoader

loader = SlowerWhisperLoader(
    path="whisper_json/",
    chunking_config=ChunkingConfig(target_duration_s=30.0),
)
documents = loader.load()  # Returns list of LangChain Document objects

# LlamaIndex integration
from integrations.llamaindex_reader import SlowerWhisperReader

reader = SlowerWhisperReader(
    path="whisper_json/",
    chunking_config=ChunkingConfig(target_tokens=400),
)
documents = reader.load_data()  # Returns list of LlamaIndex Document objects
```

See `/home/steven/code/Python/slower-whisper/examples/llm_integration/` for complete examples.

## Common Patterns

### Full Pipeline (Transcribe + Enrich)

```python
# Stage 1: Transcribe
trans_cfg = TranscriptionConfig(model="large-v3", language="en")
transcripts = transcribe_directory("/data/project", trans_cfg)

# Stage 2: Enrich
enrich_cfg = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
enriched = enrich_directory("/data/project", enrich_cfg)
```

### Custom Processing

```python
# Load
transcript = load_transcript("transcript.json")

# Modify
for segment in transcript.segments:
    if segment.audio_state:
        # Custom logic based on audio features
        if segment.audio_state["prosody"]["pitch"]["level"] == "high":
            segment.text = segment.text.upper()

# Save
save_transcript(transcript, "modified.json")
```

## LLM Rendering Functions

slower-whisper provides specialized functions to render transcripts into LLM-optimized text formats. These follow the patterns documented in `docs/LLM_PROMPT_PATTERNS.md`.

### Import

```python
from transcription import (
    load_transcript,
    render_conversation_for_llm,
    render_conversation_compact,
    render_segment,
    to_turn_view,
    to_speaker_summary,
)
```

### render_conversation_for_llm

Full conversation rendering with optional metadata header and audio cues.

**Signature:**

```python
render_conversation_for_llm(
    transcript: Transcript,
    mode: str = "turns",
    include_audio_cues: bool = True,
    include_timestamps: bool = False,
    include_metadata: bool = True,
    speaker_labels: dict[str, str] | None = None,
) -> str
```

**Parameters:**

- `mode`: Rendering mode - "turns" (recommended) or "segments"
- `include_audio_cues`: Include prosody/emotion cues like "[high pitch, fast rate]"
- `include_timestamps`: Prefix each turn/segment with timestamp
- `include_metadata`: Prepend conversation metadata header
- `speaker_labels`: Map speaker IDs to human-readable names (e.g., `{"spk_0": "Agent"}`)

**Example:**

```python
transcript = load_transcript("whisper_json/support_call.json")

# Full rendering with metadata and audio cues
context = render_conversation_for_llm(transcript)

# Example output:
# Conversation: support_call.wav (en)
# Duration: 00:05:23 | Speakers: 2 | Turns: 12
#
# [spk_0 | calm tone, moderate pitch] Hello, this is Alex from support.
# [spk_1 | frustrated tone, high pitch, fast rate] Hi, I'm having issues...
# [spk_0 | empathetic tone, slow rate] I understand that's frustrating...

# With custom speaker labels
labels = {"spk_0": "Agent", "spk_1": "Customer"}
context = render_conversation_for_llm(transcript, speaker_labels=labels)

# With timestamps
context = render_conversation_for_llm(
    transcript,
    include_timestamps=True,
    speaker_labels=labels
)
```

### render_conversation_compact

Token-constrained rendering for tight LLM contexts. Omits timestamps and audio cues.

**Signature:**

```python
render_conversation_compact(
    transcript: Transcript,
    max_tokens: int | None = None,
    speaker_labels: dict[str, str] | None = None,
) -> str
```

**Parameters:**

- `max_tokens`: Approximate token limit (uses 4 chars ≈ 1 token heuristic)
- `speaker_labels`: Map speaker IDs to human-readable names

**Example:**

```python
transcript = load_transcript("whisper_json/meeting.json")

# Compact rendering for token-limited contexts
labels = {"spk_0": "Manager", "spk_1": "Developer"}
compact = render_conversation_compact(
    transcript,
    max_tokens=500,
    speaker_labels=labels
)

# Example output:
# Manager: Let's review the sprint goals.
# Developer: I completed the authentication feature.
# Manager: Great work. What's next on your list?
# [...truncated]
```

### render_segment

Render a single segment with optional speaker and audio cues.

**Signature:**

```python
render_segment(
    segment: Segment,
    include_audio_cues: bool = True,
    include_timestamps: bool = False,
    speaker_labels: dict[str, str] | None = None,
) -> str
```

**Parameters:**

- `include_audio_cues`: Include audio state rendering if available
- `include_timestamps`: Include [HH:MM:SS] timestamp prefix
- `speaker_labels`: Map speaker IDs to human-readable names

**Example:**

```python
transcript = load_transcript("whisper_json/interview.json")
segment = transcript.segments[5]

# Render with audio cues
rendered = render_segment(segment)
# Output: [spk_0 | high pitch, loud volume, fast speech] I'm really excited about this!

# With custom labels and timestamp
labels = {"spk_0": "Candidate"}
rendered = render_segment(
    segment,
    include_timestamps=True,
    speaker_labels=labels
)
# Output: [00:01:23] [Candidate | high pitch, loud volume] I'm really excited!

# Text only (no audio cues)
rendered = render_segment(segment, include_audio_cues=False)
# Output: [spk_0] I'm really excited about this!
```

### to_turn_view

Turn-level rendering with aggregated audio cues and metadata for advanced LLM analysis.

**Signature:**

```python
to_turn_view(
    transcript: Transcript,
    turns: list[dict[str, Any] | Any] | None = None,
    include_audio_state: bool = True,
    include_timestamps: bool = True,
    speaker_labels: dict[str, str] | None = None,
) -> str
```

**Parameters:**

- `turns`: Optional turn list (defaults to `transcript.turns`)
- `include_audio_state`: Aggregate audio cues from segments
- `include_timestamps`: Include start/end time window for each turn
- `speaker_labels`: Map speaker IDs to human-readable names

**Example:**

```python
transcript = load_transcript("whisper_json/call.json")

# Full turn view with all metadata
labels = {"spk_0": "Agent", "spk_1": "Customer"}
turn_view = to_turn_view(transcript, speaker_labels=labels)

# Example output:
# [00:00:00.0-00:00:03.5 | Agent | question_count=1 | audio=calm tone, moderate pitch] How can I help you today?
# [00:00:03.5-00:00:08.2 | Customer | interruption_started_here=true | audio=frustrated tone, high pitch] I need to cancel my subscription!
# [00:00:08.2-00:00:12.1 | Agent | avg_pause_ms=350 | audio=empathetic tone, slow rate] I understand. Let me help you with that.

# Without audio cues
turn_view = to_turn_view(
    transcript,
    include_audio_state=False,
    speaker_labels=labels
)

# Without timestamps
turn_view = to_turn_view(
    transcript,
    include_timestamps=False,
    speaker_labels=labels
)
```

**Turn Metadata Fields:**

When `enable_turn_metadata=True` in enrichment config, turns include:

- `question_count`: Number of questions in turn
- `interruption_started_here`: Boolean indicating if speaker interrupted
- `avg_pause_ms`: Average pause duration between segments
- `disfluency_ratio`: Ratio of disfluent words (um, uh, etc.)

### to_speaker_summary

Generate per-speaker analytics summary for LLM context.

**Signature:**

```python
to_speaker_summary(
    transcript: Transcript,
    speaker_stats: list[dict[str, Any] | Any] | None = None,
    speaker_labels: dict[str, str] | None = None,
) -> str
```

**Parameters:**

- `speaker_stats`: Optional explicit stats list (defaults to `transcript.speaker_stats`)
- `speaker_labels`: Map speaker IDs to human-readable names

**Example:**

```python
transcript = load_transcript("whisper_json/meeting.json")

# Generate speaker summary
labels = {"spk_0": "Manager", "spk_1": "Dev1", "spk_2": "Dev2"}
summary = to_speaker_summary(transcript, speaker_labels=labels)

# Example output:
# Speaker stats summary:
# - Manager: 245.3s across 8 turns (avg 30.7s); 2 interruptions started, 0 received; 5 question turns; pitch~=185Hz; energy~=-12.3dB; sentiment+=/=0.65/-=0.15
# - Dev1: 178.9s across 12 turns (avg 14.9s); 1 interruptions started, 1 received; 2 question turns; pitch~=210Hz; energy~=-15.1dB; sentiment+=/=0.55/-=0.22
# - Dev2: 132.4s across 10 turns (avg 13.2s); 0 interruptions started, 1 received; 3 question turns; pitch~=195Hz; energy~=-14.7dB; sentiment+=/=0.70/-=0.10
```

**Speaker Stats Fields:**

When `enable_speaker_stats=True` in enrichment config, stats include:

- `total_talk_time`: Total seconds speaking
- `num_turns`: Number of turns taken
- `avg_turn_duration`: Average turn length in seconds
- `interruptions_initiated`: Times speaker interrupted others
- `interruptions_received`: Times speaker was interrupted
- `question_turns`: Number of turns containing questions
- `prosody_summary`: Median pitch/energy values
- `sentiment_summary`: Positive/negative sentiment ratios

## Semantic Annotation

Semantic annotators add optional semantic tags and intent markers to transcripts. This is a lightweight protocol for plugging in semantic enrichment without modifying the core pipeline.

### SemanticAnnotator Protocol

All semantic annotators implement the `SemanticAnnotator` protocol:

**Interface:**

```python
from typing import Protocol
from transcription import Transcript

class SemanticAnnotator(Protocol):
    """Annotate a transcript with semantic tags."""

    def annotate(self, transcript: Transcript) -> Transcript:
        """Return a transcript with semantic annotations attached."""
```

**Purpose:**
- Add semantic metadata to transcripts (tags, intent labels, etc.)
- Extract conversation topics and themes
- Detect business-relevant patterns (pricing, escalation, churn risk)
- Enrich transcripts for downstream LLM analysis

### NoOpSemanticAnnotator

Default implementation that returns transcripts unchanged. Use when semantic annotation is not needed.

**Usage:**

```python
from transcription import Transcript
from transcription.semantic import NoOpSemanticAnnotator

# Create no-op annotator
annotator = NoOpSemanticAnnotator()

# Pass through unchanged
transcript = load_transcript("whisper_json/call.json")
result = annotator.annotate(transcript)  # Returns transcript unchanged
assert result is transcript  # Same object reference
```

### KeywordSemanticAnnotator

Lightweight rule-based annotator that tags escalation/churn language and action items.

**Example Usage:**

```python
from transcription import load_transcript, save_transcript
from transcription.semantic import KeywordSemanticAnnotator

transcript = load_transcript("whisper_json/support_call.json")
annotator = KeywordSemanticAnnotator()
annotated = annotator.annotate(transcript)

semantic = (annotated.annotations or {}).get("semantic", {})
print("Keywords:", semantic.get("keywords", []))
print("Risk tags:", semantic.get("risk_tags", []))
print("Actions:", semantic.get("actions", []))

save_transcript(annotated, "whisper_json/support_call_annotated.json")
```

**Pipeline Integration:**

```python
from transcription import transcribe_file, TranscriptionConfig
from transcription.semantic import KeywordSemanticAnnotator

# Configure transcription
config = TranscriptionConfig(model="large-v3", language="en")

# Transcribe
transcript = transcribe_file(
    audio_path="call.mp3",
    root="/project",
    config=config
)

# Apply semantic annotation
annotator = KeywordSemanticAnnotator()
annotated = annotator.annotate(transcript)

# Use annotated transcript for downstream tasks
semantic = (annotated.annotations or {}).get("semantic", {})
if "escalation" in semantic.get("risk_tags", []):
    print("⚠️  Alert: Escalation language detected")
    # Trigger workflow, send notification, etc.

for action in semantic.get("actions", []):
    print(f"Action: {action['text']} (speaker={action['speaker_id']})")
```

**How It Works:**

1. Scans each segment (case-insensitive) for escalation/churn lexicon hits.
2. Detects action intents with patterns like "I'll …", "we will …", "let me …".
3. Stores results in `transcript.annotations["semantic"]`:
   - `keywords`: matched lexicon terms
   - `risk_tags`: canonical risk buckets (e.g., `escalation`, `churn_risk`)
   - `actions`: action items with `text`, `speaker_id`, `segment_ids`
   - `matches`: optional debug entries for matched keywords

**Output Structure:**

```json
{
  "annotations": {
    "semantic": {
      "keywords": ["manager", "switch", "unacceptable"],
      "risk_tags": ["churn_risk", "escalation"],
      "actions": [
        {"text": "I'll send the invoice after this call.", "speaker_id": "spk_0", "segment_ids": [0]}
      ],
      "matches": [
        {"risk": "escalation", "keyword": "manager", "segment_id": 0},
        {"risk": "churn_risk", "keyword": "switch", "segment_id": 1}
      ]
    }
  }
}
```

**Limitations:**

- Simple string/pattern matching (case-insensitive)
- No context awareness (can produce false positives)
- No negation handling ("not expensive" matches "pricing")
- Designed for lightweight, fast tagging (not deep NLP analysis)

**Customizing Lexicons:**

```python
custom = KeywordSemanticAnnotator(
    escalation_keywords=("escalate", "unhappy", "manager"),
    churn_keywords=("cancel", "downgrade", "switch"),
    action_patterns=(r"\bi can escalate\b", r"\bwe will investigate\b"),
)
annotated = custom.annotate(transcript)
```

**When to Use:**

- Quick topic detection without ML models
- Filtering conversations by theme
- Triggering workflows based on keywords
- Building training data for semantic models

### Common LLM Integration Patterns

**1. RAG Context Preparation:**

```python
from transcription import load_transcript, render_conversation_for_llm

transcript = load_transcript("whisper_json/sales_call.json")
labels = {"spk_0": "Sales Rep", "spk_1": "Prospect"}

# Prepare context for RAG system
context = render_conversation_for_llm(
    transcript,
    mode="turns",
    include_audio_cues=True,
    speaker_labels=labels
)

# Use with your LLM
response = llm.generate(
    prompt=f"Analyze this sales call:\n\n{context}\n\nWhat objections did the prospect raise?"
)
```

**2. Speaker-Aware Summarization:**

```python
from transcription import (
    load_transcript,
    render_conversation_for_llm,
    to_speaker_summary
)

transcript = load_transcript("whisper_json/interview.json")
labels = {"spk_0": "Interviewer", "spk_1": "Candidate"}

conversation = render_conversation_for_llm(transcript, speaker_labels=labels)
stats = to_speaker_summary(transcript, speaker_labels=labels)

prompt = f"""
{conversation}

{stats}

Summarize this interview, highlighting:
1. Key topics discussed
2. Candidate's communication style (based on audio cues)
3. Balance of speaking time
"""
```

**3. Token-Constrained Analysis:**

```python
from transcription import load_transcript, render_conversation_compact

transcript = load_transcript("whisper_json/long_meeting.json")

# Fit into 1000-token context window
compact = render_conversation_compact(
    transcript,
    max_tokens=1000,
    speaker_labels={"spk_0": "PM", "spk_1": "Eng", "spk_2": "Design"}
)

# Extract action items from truncated conversation
response = llm.generate(
    prompt=f"Extract action items from this meeting:\n\n{compact}"
)
```

**4. Turn-Level Analytics:**

```python
from transcription import load_transcript, to_turn_view, to_speaker_summary

transcript = load_transcript("whisper_json/support_call.json")
labels = {"spk_0": "Agent", "spk_1": "Customer"}

turns = to_turn_view(transcript, speaker_labels=labels)
stats = to_speaker_summary(transcript, speaker_labels=labels)

prompt = f"""
Conversation Turns:
{turns}

Speaker Analytics:
{stats}

Evaluate this support interaction:
1. Did the agent address the customer's frustration effectively?
2. Were there interruptions that hindered resolution?
3. Rate empathy based on audio cues (1-10)
"""
```

### Error Handling

```python
try:
    transcript = transcribe_file("audio.mp3", "/project", config)
except FileNotFoundError as e:
    print(f"Audio file not found: {e}")
except Exception as e:
    print(f"Transcription failed: {e}")
```

## Directory Layout

```text
project_root/
├── raw_audio/           # Input: Original audio files
├── input_audio/         # Generated: Normalized 16kHz WAVs
├── whisper_json/        # Generated: JSON transcripts
└── transcripts/         # Generated: TXT and SRT files
```

## Speaker Analytics Models

Speaker analytics models provide structured representations of conversational turns, speaker statistics, and diarization metadata. These are populated when diarization is enabled and turns/speaker stats are computed.

### Turn

Represents a contiguous speaking turn by a single speaker.

**Fields:**
- `id` (str): Unique turn identifier (e.g., "turn_0")
- `speaker_id` (str): Speaker who produced this turn (e.g., "SPEAKER_00")
- `segment_ids` (list[int]): List of segment IDs included in this turn
- `start` (float): Turn start time in seconds
- `end` (float): Turn end time in seconds
- `text` (str): Concatenated text from all segments in the turn
- `metadata` (dict | None): Optional enrichment metadata (see TurnMeta)

**Methods:**
- `to_dict()`: Serialize to JSON-serializable dict
- `from_dict(d)`: Construct from dict (class method)

**Example:**

```python
# Access turns from transcript
if transcript.turns:
    for turn in transcript.turns:
        # Handle both Turn objects and dicts
        if isinstance(turn, dict):
            speaker = turn["speaker_id"]
            text = turn["text"]
            start = turn["start"]
        else:
            speaker = turn.speaker_id
            text = turn.text
            start = turn.start

        print(f"[{start:.1f}s] {speaker}: {text}")

        # Access enriched metadata if available
        if isinstance(turn, dict):
            meta = turn.get("metadata", {})
        else:
            meta = turn.metadata or {}

        if meta:
            print(f"  Questions: {meta.get('question_count', 0)}")
            print(f"  Interrupted: {meta.get('interruption_started_here', False)}")
```

**Turn Structure in JSON:**

```json
{
  "id": "turn_0",
  "speaker_id": "SPEAKER_00",
  "segment_ids": [0, 1, 2],
  "start": 0.0,
  "end": 5.8,
  "text": "Hello everyone, welcome to the meeting.",
  "metadata": {
    "question_count": 0,
    "interruption_started_here": false,
    "avg_pause_ms": 450.2,
    "disfluency_ratio": 0.05
  }
}
```

### TurnMeta

Enriched metadata for conversational turns (populated by `enable_turn_metadata`).

**Fields:**
- `question_count` (int): Number of questions in this turn (default: 0)
- `interruption_started_here` (bool): Whether this turn interrupted another speaker (default: False)
- `avg_pause_ms` (float | None): Average intra-turn pause duration in milliseconds
- `disfluency_ratio` (float | None): Ratio of disfluency markers (um, uh, like) to total words

**Methods:**
- `to_dict()`: Serialize to JSON-serializable dict

**Example:**

```python
from transcription import TurnMeta

# Access from turn metadata
if transcript.turns:
    for turn in transcript.turns:
        meta = turn.get("metadata") if isinstance(turn, dict) else turn.metadata

        if meta:
            # meta is a dict, can convert to TurnMeta if needed
            turn_meta = TurnMeta(
                question_count=meta.get("question_count", 0),
                interruption_started_here=meta.get("interruption_started_here", False),
                avg_pause_ms=meta.get("avg_pause_ms"),
                disfluency_ratio=meta.get("disfluency_ratio")
            )

            if turn_meta.question_count > 0:
                print(f"Turn has {turn_meta.question_count} questions")

            if turn_meta.interruption_started_here:
                print("This turn was an interruption")
```

### SpeakerStats

Per-speaker aggregate statistics (populated by `enable_speaker_stats`).

**Fields:**
- `speaker_id` (str): Speaker identifier (e.g., "SPEAKER_00")
- `total_talk_time` (float): Total speaking time in seconds
- `num_turns` (int): Number of turns taken by this speaker
- `avg_turn_duration` (float): Average duration per turn in seconds
- `interruptions_initiated` (int): Number of times this speaker interrupted others
- `interruptions_received` (int): Number of times this speaker was interrupted
- `question_turns` (int): Number of turns containing questions
- `prosody_summary` (ProsodySummary): Aggregated prosody features
  - `pitch_median_hz` (float | None): Median pitch in Hz
  - `energy_median_db` (float | None): Median energy in dB
- `sentiment_summary` (SentimentSummary): Aggregated sentiment scores
  - `positive` (float): Proportion of positive sentiment (0-1)
  - `neutral` (float): Proportion of neutral sentiment (0-1)
  - `negative` (float): Proportion of negative sentiment (0-1)

**Methods:**
- `to_dict()`: Serialize to JSON-serializable dict (expands nested summaries)

**Example:**

```python
# Access speaker stats from transcript
if transcript.speaker_stats:
    for stats in transcript.speaker_stats:
        # Handle both SpeakerStats objects and dicts
        if isinstance(stats, dict):
            speaker_id = stats["speaker_id"]
            talk_time = stats["total_talk_time"]
            num_turns = stats["num_turns"]
        else:
            speaker_id = stats.speaker_id
            talk_time = stats.total_talk_time
            num_turns = stats.num_turns

        print(f"\n{speaker_id}:")
        print(f"  Total talk time: {talk_time:.1f}s")
        print(f"  Turns: {num_turns}")

        # Access nested prosody summary
        if isinstance(stats, dict):
            prosody = stats.get("prosody_summary", {})
            sentiment = stats.get("sentiment_summary", {})
        else:
            prosody = stats.prosody_summary
            sentiment = stats.sentiment_summary

        if prosody:
            pitch = prosody.get("pitch_median_hz") if isinstance(prosody, dict) else prosody.pitch_median_hz
            if pitch:
                print(f"  Median pitch: {pitch:.1f} Hz")

        if sentiment:
            pos = sentiment.get("positive") if isinstance(sentiment, dict) else sentiment.positive
            print(f"  Positive sentiment: {pos:.1%}")
```

**SpeakerStats Structure in JSON:**

```json
{
  "speaker_id": "SPEAKER_00",
  "total_talk_time": 45.8,
  "num_turns": 12,
  "avg_turn_duration": 3.82,
  "interruptions_initiated": 2,
  "interruptions_received": 1,
  "question_turns": 5,
  "prosody_summary": {
    "pitch_median_hz": 185.3,
    "energy_median_db": -12.4
  },
  "sentiment_summary": {
    "positive": 0.65,
    "neutral": 0.25,
    "negative": 0.10
  }
}
```

### DiarizationMeta

Metadata describing diarization execution status (stored in `transcript.meta["diarization"]`).

**Fields:**
- `requested` (bool): Whether diarization was requested via config
- `status` (Literal): Execution status - one of:
  - `"disabled"`: Diarization not requested
  - `"skipped"`: Requested but skipped (e.g., existing speakers found)
  - `"ok"`: Successfully completed
  - `"error"`: Failed with error
- `backend` (str | None): Backend used (e.g., "pyannote", "stub")
- `num_speakers` (int | None): Number of speakers detected
- `error_type` (str | None): Error category if status is "error"
- `message` (str | None): Human-readable status message
- `error` (str | None): Error message (synonym for message, backward compatibility)

**Methods:**
- `to_dict()`: Serialize to JSON-serializable dict

**Example:**

```python
# Check diarization status from transcript metadata
if transcript.meta and "diarization" in transcript.meta:
    diar_meta = transcript.meta["diarization"]

    print(f"Diarization status: {diar_meta['status']}")

    if diar_meta["status"] == "ok":
        print(f"Backend: {diar_meta['backend']}")
        print(f"Speakers detected: {diar_meta['num_speakers']}")
    elif diar_meta["status"] == "error":
        print(f"Error type: {diar_meta['error_type']}")
        print(f"Message: {diar_meta['message']}")

# Access from config result
from transcription import transcribe_file, TranscriptionConfig

config = TranscriptionConfig(enable_diarization=True)
transcript = transcribe_file("meeting.wav", "/project", config)

# Inspect diarization outcome
diar = transcript.meta.get("diarization") if transcript.meta else None
if diar and diar["status"] == "ok":
    print(f"✓ Diarization succeeded with {diar['num_speakers']} speakers")
else:
    print(f"✗ Diarization failed or disabled")
```

**DiarizationMeta Structure in JSON:**

```json
{
  "meta": {
    "generated_at": "2025-12-01T12:34:56.789Z",
    "model_name": "large-v3",
    "device": "cuda",
    "diarization": {
      "requested": true,
      "status": "ok",
      "backend": "pyannote",
      "num_speakers": 3,
      "error_type": null,
      "message": "Diarization completed successfully"
    }
  }
}
```

**Complete Example: Analyzing a Conversation**

```python
from transcription import load_transcript

# Load transcript with speaker analytics
transcript = load_transcript("meeting.json")

# Check diarization status
if transcript.meta and "diarization" in transcript.meta:
    diar = transcript.meta["diarization"]
    if diar["status"] == "ok":
        print(f"✓ Found {diar['num_speakers']} speakers\n")

# Analyze speaker statistics
if transcript.speaker_stats:
    print("Speaker Summary:")
    for stats in transcript.speaker_stats:
        sid = stats["speaker_id"] if isinstance(stats, dict) else stats.speaker_id
        talk_time = stats["total_talk_time"] if isinstance(stats, dict) else stats.total_talk_time
        num_turns = stats["num_turns"] if isinstance(stats, dict) else stats.num_turns

        print(f"  {sid}: {talk_time:.1f}s ({num_turns} turns)")

# Review conversational turns
if transcript.turns:
    print("\nConversation Flow:")
    for turn in transcript.turns[:5]:  # First 5 turns
        if isinstance(turn, dict):
            speaker = turn["speaker_id"]
            text = turn["text"]
            meta = turn.get("metadata", {})
        else:
            speaker = turn.speaker_id
            text = turn.text
            meta = turn.metadata or {}

        # Truncate long text
        text_preview = text[:60] + "..." if len(text) > 60 else text

        markers = []
        if meta.get("question_count", 0) > 0:
            markers.append("❓")
        if meta.get("interruption_started_here"):
            markers.append("⚡")

        marker_str = " ".join(markers)
        print(f"  {speaker}: {text_preview} {marker_str}")
```

## Tips

1. **Use defaults:** Most parameters have sensible defaults
2. **Check audio_state:** Always check if `audio_state` is not `None` before accessing
3. **Lazy imports:** Enrichment features loaded on-demand (won't fail if not installed)
4. **Batch processing:** Use `_directory()` functions for multiple files
5. **Progressive enhancement:** Transcribe first, enrich later as needed
6. **Handle both types:** Speaker analytics fields may contain dataclass objects or dicts; handle both cases
