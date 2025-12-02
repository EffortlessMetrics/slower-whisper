# Streaming Architecture (v0.1–v0.2)

Goal: ship a small, predictable streaming surface for slower-whisper without changing the offline JSON contract. This document captures the v0.1 data model and state rules for streaming **post-ASR text chunks** (no audio transport yet).

## Data Model

### StreamChunk (input)

```python
class StreamChunk(TypedDict):
    start: float   # seconds
    end: float
    text: str
    speaker_id: str | None
```

Chunks are **post-ASR** for v0.1. Audio/VAD/encoder streaming will layer on later.

### StreamSegment (state)

```python
@dataclass(slots=True)
class StreamSegment:
    start: float
    end: float
    text: str
    speaker_id: str | None = None
```

### StreamEvent (output)

```python
class StreamEventType(Enum):
    PARTIAL_SEGMENT = "partial_segment"  # may change as more chunks arrive
    FINAL_SEGMENT = "final_segment"      # no further changes expected

@dataclass(slots=True)
class StreamEvent:
    type: StreamEventType
    segment: StreamSegment
```

Optional future events: `TURN_BOUNDARY`, `CHUNK_BOUNDARY`.

### Configuration

```python
@dataclass(slots=True)
class StreamConfig:
    max_gap_sec: float = 1.0  # gap threshold to start a new segment
```

## State rules

- Chunks arrive with **monotonic time**. If `chunk.start < current.end`, `ingest_chunk` raises `ValueError`.
- A chunk **extends the current partial segment** when:
  - Same `speaker_id` (including both `None`), and
  - Gap `chunk.start - current.end` is `<= max_gap_sec`.
- Otherwise, the current partial is finalized and a new partial begins.
- `ingest_chunk(chunk)` returns the events produced by that chunk:
  - Zero or one `FINAL_SEGMENT` for the previous segment (if it closed).
  - Exactly one `PARTIAL_SEGMENT` for the current segment state.
- `end_of_stream()` flushes any remaining partial as a `FINAL_SEGMENT` and clears session state.

## Session surface

```python
class StreamingSession:
    def __init__(self, config: StreamConfig | None = None) -> None: ...
    def ingest_chunk(self, chunk: StreamChunk) -> list[StreamEvent]: ...
    def end_of_stream(self) -> list[StreamEvent]: ...
```

Pure Python, no asyncio or sockets. This simulates how downstream consumers will see a live transcription feed.

## Implementation Status

| Component | Status |
|-----------|--------|
| Type definitions | ✅ Implemented in v0.1 |
| StreamingSession state machine | ✅ Implemented in v0.1 |
| WebSocket server | 🚧 Not started (future) |
| Audio buffering/VAD | 🚧 Not started (future) |
| Event replay/apply | 🚧 Not started (future) |

## Usage

```python
from transcription.streaming import StreamChunk, StreamConfig, StreamingSession

session = StreamingSession(StreamConfig(max_gap_sec=1.0))

for chunk in stream_of_chunks:
    events = session.ingest_chunk(chunk)
    for event in events:
        if event.type == StreamEventType.PARTIAL_SEGMENT:
            print("Partial:", event.segment.text)
        elif event.type == StreamEventType.FINAL_SEGMENT:
            print("Final:", event.segment.text)

for event in session.end_of_stream():
    print("Final:", event.segment.text)
```

---

## Integration Patterns (v0.2)

v0.2 adds **semantic annotation** and **LLM integration** to the streaming pipeline. These patterns show how to combine streaming transcription with real-time semantic enrichment and LLM-based routing.

### Pattern 1: Streaming + Semantic Annotation

Annotate streaming segments with keywords and intent tags as they arrive. Uses `KeywordSemanticAnnotator` from `transcription.semantic` for lightweight, deterministic tagging.

```python
from transcription.streaming import StreamChunk, StreamConfig, StreamingSession, StreamEventType
from transcription.semantic import KeywordSemanticAnnotator

# Initialize streaming and semantic components
stream_config = StreamConfig(max_gap_sec=1.0)

session = StreamingSession(stream_config)
# Use KeywordSemanticAnnotator with default keywords
annotator = KeywordSemanticAnnotator()

# Process chunks with semantic annotation
for chunk in stream_of_chunks:
    events = session.ingest_chunk(chunk)

    for event in events:
        segment = event.segment

        # Annotate FINAL segments only (avoids re-annotation of changing text)
        if event.type == StreamEventType.FINAL_SEGMENT:
            # Create minimal transcript for annotation
            from transcription.models import Transcript
            temp_transcript = Transcript(
                file_name="stream", language="en",
                segments=[segment], meta={}
            )
            annotated_transcript = annotator.annotate(temp_transcript)
            semantic_data = annotated_transcript.segments[0].annotations.get("semantic", {})

            # Emit annotated segment downstream
            annotated = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker_id": segment.speaker_id,
                "keywords": keywords,
                "intent": intent,
                "timestamp": segment.end  # Use end time as stable reference
            }
            yield annotated

        elif event.type == StreamEventType.PARTIAL_SEGMENT:
            # Show partial transcription without annotation
            print(f"[PARTIAL] {segment.text}")

# Finalize and annotate remaining segments
for event in session.end_of_stream():
    segment = event.segment
    keywords = annotator.extract_keywords(segment.text)
    intent = annotator.classify_intent(segment.text)

    annotated = {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "speaker_id": segment.speaker_id,
        "keywords": keywords,
        "intent": intent,
        "timestamp": segment.end
    }
    yield annotated
```

**Key design decisions:**
- Only annotate `FINAL_SEGMENT` to avoid wasted computation on changing text
- Use lightweight keyword extraction (no LLM calls) for real-time performance
- Emit annotated segments incrementally for downstream consumers

**See also:** `docs/LLM_PROMPT_PATTERNS.md` for semantic annotation strategies.

### Pattern 2: Streaming + LLM Routing

Route streaming segments to different LLM workflows based on intent classification. Useful for real-time assistants, meeting bots, or call center analytics.

```python
from transcription.streaming import StreamChunk, StreamConfig, StreamingSession, StreamEventType
from transcription.semantic import KeywordSemanticAnnotator, SemanticConfig
from transcription.llm_utils import render_segment_for_llm
from enum import Enum

class WorkflowType(Enum):
    QUESTION = "question"
    ACTION_ITEM = "action_item"
    DECISION = "decision"
    GENERAL = "general"

def route_to_workflow(intent: str) -> WorkflowType:
    """Map intent tags to LLM workflows."""
    if intent in ("question", "clarification"):
        return WorkflowType.QUESTION
    elif intent in ("action_item", "commitment"):
        return WorkflowType.ACTION_ITEM
    elif intent in ("decision", "agreement"):
        return WorkflowType.DECISION
    else:
        return WorkflowType.GENERAL

# Initialize components
session = StreamingSession(StreamConfig(max_gap_sec=1.0))
annotator = KeywordSemanticAnnotator(SemanticConfig(enable_intent=True))

# Workflow queues (in production: use proper queue like Redis/RabbitMQ)
workflow_queues = {
    WorkflowType.QUESTION: [],
    WorkflowType.ACTION_ITEM: [],
    WorkflowType.DECISION: [],
    WorkflowType.GENERAL: []
}

# Process chunks and route to workflows
for chunk in stream_of_chunks:
    events = session.ingest_chunk(chunk)

    for event in events:
        if event.type == StreamEventType.FINAL_SEGMENT:
            segment = event.segment

            # Classify intent and route
            intent = annotator.classify_intent(segment.text)
            workflow = route_to_workflow(intent)

            # Prepare LLM-ready format
            llm_segment = {
                "text": segment.text,
                "speaker": segment.speaker_id or "Unknown",
                "start": segment.start,
                "end": segment.end,
                "intent": intent
            }

            # Route to appropriate workflow queue
            workflow_queues[workflow].append(llm_segment)

            print(f"[{workflow.value.upper()}] {segment.text}")

            # Trigger LLM processing for high-priority workflows
            if workflow == WorkflowType.QUESTION:
                # Example: immediate LLM processing for questions
                prompt = f"Answer this question from the conversation:\nQ: {segment.text}\nA:"
                # response = call_llm(prompt)  # Placeholder
                print(f"  → Triggering immediate LLM response")

# Flush remaining segments
for event in session.end_of_stream():
    segment = event.segment
    intent = annotator.classify_intent(segment.text)
    workflow = route_to_workflow(intent)

    llm_segment = {
        "text": segment.text,
        "speaker": segment.speaker_id or "Unknown",
        "start": segment.start,
        "end": segment.end,
        "intent": intent
    }
    workflow_queues[workflow].append(llm_segment)
```

**Key design decisions:**
- Deterministic routing based on intent tags (no LLM calls in routing logic)
- Separate queues for different workflow types enable parallel processing
- High-priority workflows (e.g., questions) can trigger immediate LLM responses
- Low-priority workflows can batch process after stream ends

**See also:** `docs/LLM_PROMPT_PATTERNS.md` for workflow-specific prompt templates.

### Pattern 3: Progressive Context Building

Build a rolling context window for LLM consumption as segments arrive. Useful for real-time summarization, meeting assistants, or context-aware chatbots.

```python
from transcription.streaming import StreamChunk, StreamConfig, StreamingSession, StreamEventType
from transcription.llm_utils import render_segment_for_llm
from transcription.semantic import KeywordSemanticAnnotator, SemanticConfig
from collections import deque
from dataclasses import dataclass

@dataclass
class ContextWindow:
    """Rolling window of finalized segments for LLM context."""
    max_segments: int = 10
    max_time_sec: float = 120.0
    segments: deque = None

    def __post_init__(self):
        if self.segments is None:
            self.segments = deque(maxlen=self.max_segments)

    def add(self, segment):
        """Add segment and prune old segments by time."""
        self.segments.append(segment)

        # Prune segments older than max_time_sec
        if len(self.segments) > 1:
            latest_end = self.segments[-1]["end"]
            while self.segments and (latest_end - self.segments[0]["end"]) > self.max_time_sec:
                self.segments.popleft()

    def render_for_llm(self) -> str:
        """Render context window as LLM-ready text."""
        if not self.segments:
            return ""

        lines = ["Recent conversation context:"]
        for seg in self.segments:
            speaker = seg.get("speaker_id") or "Unknown"
            text = seg["text"]
            timestamp = f"{seg['start']:.1f}s"
            lines.append(f"[{timestamp}] {speaker}: {text}")

        return "\n".join(lines)

    def get_keywords(self) -> list[str]:
        """Extract unique keywords from context window."""
        all_keywords = []
        for seg in self.segments:
            all_keywords.extend(seg.get("keywords", []))
        return list(set(all_keywords))  # Deduplicate

# Initialize components
session = StreamingSession(StreamConfig(max_gap_sec=1.0))
annotator = KeywordSemanticAnnotator(SemanticConfig(enable_keywords=True, enable_intent=True))
context = ContextWindow(max_segments=15, max_time_sec=180.0)

# Process chunks with progressive context building
for chunk in stream_of_chunks:
    events = session.ingest_chunk(chunk)

    for event in events:
        if event.type == StreamEventType.FINAL_SEGMENT:
            segment = event.segment

            # Annotate segment
            keywords = annotator.extract_keywords(segment.text)
            intent = annotator.classify_intent(segment.text)

            # Add to context window
            context_segment = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker_id": segment.speaker_id,
                "keywords": keywords,
                "intent": intent
            }
            context.add(context_segment)

            print(f"[FINAL] {segment.text}")
            print(f"  Context: {len(context.segments)} segments, keywords={context.get_keywords()}")

            # Example: trigger LLM summary every 5 segments
            if len(context.segments) % 5 == 0:
                llm_context = context.render_for_llm()
                prompt = f"{llm_context}\n\nSummarize the key points discussed:"
                # summary = call_llm(prompt)  # Placeholder
                print(f"  → Triggering incremental summary")

        elif event.type == StreamEventType.PARTIAL_SEGMENT:
            print(f"[PARTIAL] {event.segment.text}")

# Finalize context and generate final summary
for event in session.end_of_stream():
    segment = event.segment
    keywords = annotator.extract_keywords(segment.text)
    intent = annotator.classify_intent(segment.text)

    context_segment = {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "speaker_id": segment.speaker_id,
        "keywords": keywords,
        "intent": intent
    }
    context.add(context_segment)

# Generate final summary with full context
llm_context = context.render_for_llm()
keywords = context.get_keywords()
prompt = f"""{llm_context}

Conversation keywords: {', '.join(keywords)}

Generate a final summary covering:
1. Main topics discussed
2. Key decisions made
3. Action items identified
"""
# final_summary = call_llm(prompt)  # Placeholder
print(f"\n[STREAM END] Final context: {len(context.segments)} segments")
```

**Key design decisions:**
- Rolling window with dual constraints: max segments AND max time
- Incremental summaries every N segments (configurable trigger)
- Context pruning prevents unbounded memory growth
- Final summary uses full context window
- Keywords tracked across window for topic continuity

**See also:**
- `transcription/llm_utils.py` for `render_segment_for_llm()` utilities
- `docs/LLM_PROMPT_PATTERNS.md` for progressive summarization prompts

---

## Performance Considerations

### Real-Time Constraints

**Latency budget per chunk:**
- ASR decoding: ~100-500ms (faster-whisper on GPU)
- Keyword extraction: ~1-5ms (regex-based)
- Intent classification: ~1-5ms (keyword matching)
- LLM routing decision: <1ms (dict lookup)
- **Total: ~100-510ms** (suitable for real-time streaming)

**NOT real-time:**
- Full LLM summarization: 1-10s per call (depends on model, context length)
- Emotion recognition: 50-200ms per segment (wav2vec2 model)
- Speaker diarization: 100-500ms per segment (pyannote.audio)

### Throughput Optimization

**Batching strategies:**
- **Semantic annotation**: Batch FINAL segments for keyword extraction (not implemented in v0.2, future optimization)
- **LLM calls**: Buffer segments and batch every N segments or T seconds
- **Workflow routing**: Use async queues to decouple streaming from LLM processing

**Memory management:**
- Use `ContextWindow.max_segments` to bound memory usage
- Prune old segments by time window (e.g., keep last 2 minutes)
- For long streams (>1 hour), periodically checkpoint context and reset

### Scaling Patterns

**Horizontal scaling:**
- Run multiple `StreamingSession` instances (one per audio stream)
- Use message queue (Redis Streams, Kafka) to distribute chunks
- Route LLM workflows to separate worker pools by priority

**Vertical scaling:**
- GPU-accelerated ASR (faster-whisper with CUDA)
- CPU-only semantic annotation (keyword extraction is fast)
- Offload LLM calls to separate GPU instances

**Backpressure handling:**
- If LLM workflow queues grow too large, drop low-priority segments
- Implement exponential backoff for retries
- Log dropped segments for offline batch processing

### Monitoring Metrics

**Key metrics to track:**
- Chunk ingestion rate (chunks/sec)
- End-to-end latency (chunk arrival → annotated output)
- Context window size (segments, time span)
- LLM call frequency and latency
- Workflow queue depths

**Example monitoring:**
```python
import time
from dataclasses import dataclass, field

@dataclass
class StreamMetrics:
    chunks_ingested: int = 0
    segments_finalized: int = 0
    segments_partial: int = 0
    llm_calls: int = 0
    start_time: float = field(default_factory=time.time)

    def record_chunk(self):
        self.chunks_ingested += 1

    def record_final_segment(self):
        self.segments_finalized += 1

    def record_partial_segment(self):
        self.segments_partial += 1

    def record_llm_call(self):
        self.llm_calls += 1

    def report(self):
        elapsed = time.time() - self.start_time
        return {
            "elapsed_sec": elapsed,
            "chunks_per_sec": self.chunks_ingested / elapsed,
            "segments_finalized": self.segments_finalized,
            "segments_partial": self.segments_partial,
            "llm_calls": self.llm_calls,
            "llm_calls_per_min": (self.llm_calls / elapsed) * 60
        }

# Usage
metrics = StreamMetrics()
for chunk in stream_of_chunks:
    metrics.record_chunk()
    events = session.ingest_chunk(chunk)
    for event in events:
        if event.type == StreamEventType.FINAL_SEGMENT:
            metrics.record_final_segment()
        else:
            metrics.record_partial_segment()

print(metrics.report())
```

**See also:**
- `transcription/service.py` for production service patterns
- `docs/DEPLOYMENT.md` for scaling and monitoring guidance (future)
