# Post-Processing Guide

**Version:** v2.0.1+ | **Last Updated:** 2026-01-31

This guide covers the post-processing orchestration system for transcript enrichment, including topic segmentation and turn-taking policies.

---

## Table of Contents

1. [Overview](#overview)
2. [PostProcessor](#postprocessor)
3. [Topic Segmentation](#topic-segmentation)
4. [Turn-Taking Policies](#turn-taking-policies)
5. [Preset Configurations](#preset-configurations)
6. [Integration with Streaming](#integration-with-streaming)
7. [API Reference](#api-reference)

---

## Overview

The post-processing system runs after ASR transcription to add semantic structure:

- **Safety processing** - PII detection, moderation, smart formatting
- **Role inference** - Identify agent/customer/facilitator speakers
- **Topic segmentation** - Detect when conversation topics change
- **Turn-taking evaluation** - Determine when speakers finish their turns
- **Environment classification** - Classify audio environment (quiet, noisy, etc.)

All processors run in dependency order and attach results without mutating original text.

### Key Principles

1. **Callbacks never crash pipeline** - Exceptions are caught and logged
2. **Dependency order** - Safety → Environment → Prosody → Turn-taking
3. **Turn-level separate** - Role and topic processing runs via `process_turn()`
4. **Preset-friendly** - Pre-built configs for call centers and meetings

---

## PostProcessor

The `PostProcessor` class orchestrates all post-processing features.

### Basic Usage

```python
from transcription.post_process import (
    PostProcessor,
    PostProcessConfig,
    SegmentContext,
)

# Configure what to enable
config = PostProcessConfig(
    enable_safety=True,
    enable_roles=True,
    enable_topics=True,
    enable_turn_taking=True,
)

# Create processor with optional callbacks
processor = PostProcessor(
    config,
    on_safety_alert=handle_safety_alert,
    on_role_assigned=handle_role_assigned,
    on_topic_boundary=handle_topic_boundary,
    on_end_of_turn_hint=handle_turn_hint,
)

# Process segments as they arrive
for segment in transcript.segments:
    ctx = SegmentContext(
        session_id="session_1",
        segment_id=f"seg_{segment.id}",
        speaker_id=segment.speaker,
        start=segment.start,
        end=segment.end,
        text=segment.text,
    )
    result = processor.process_segment(ctx)

    # Result contains safety, environment, prosody_extended, end_of_turn
    if result.safety:
        print(f"Safety: {result.safety.to_safety_state()}")

# Process complete turns for topics/roles
for turn in turns:
    turn_result = processor.process_turn(turn)
    if turn_result.topic_boundary:
        print(f"Topic changed: {turn_result.topic_boundary.new_topic_id}")

# Finalize to close open topics and trigger final role inference
processor.finalize()

# Get results
roles = processor.get_role_assignments()
topics = processor.get_topics()
```

### PostProcessConfig

```python
@dataclass
class PostProcessConfig:
    enabled: bool = True                    # Master switch

    # Feature toggles
    enable_safety: bool = False             # PII + moderation + formatting
    enable_roles: bool = False              # Role inference (agent/customer)
    enable_topics: bool = False             # Topic segmentation
    enable_turn_taking: bool = False        # Turn-taking evaluation
    enable_environment: bool = False        # Environment classification
    enable_prosody_extended: bool = False   # Extended prosody analysis

    # Nested configurations
    safety_config: SafetyConfig | None = None
    role_config: RoleInferenceConfig | None = None
    topic_config: TopicSegmentationConfig | None = None
    turn_taking_policy: str | TurnTakingPolicy = "balanced"
    environment_config: EnvironmentClassifierConfig | None = None

    # Role inference timing
    role_decision_turns: int = 5            # Turns before role decision
    role_decision_seconds: float = 30.0     # Seconds before role decision
```

---

## Topic Segmentation

Topic segmentation detects when conversation topics change using TF-IDF similarity between rolling windows.

### How It Works

1. Text is tokenized and stopwords removed
2. TF-IDF vectors computed for sliding windows
3. Cosine similarity compared between windows
4. When similarity drops below threshold, a boundary is detected

### Configuration

```python
from transcription.topic_segmentation import (
    TopicSegmenter,
    StreamingTopicSegmenter,
    TopicSegmentationConfig,
)

config = TopicSegmentationConfig(
    enabled=True,
    window_size_turns=5,           # Turns per comparison window
    similarity_threshold=0.35,     # Below this = new topic
    min_topic_duration_sec=30.0,   # Minimum topic duration
    max_topic_duration_sec=300.0,  # Force boundary after this
    min_turns_for_topic=3,         # Minimum turns before topic change
)
```

### Batch Segmentation

```python
segmenter = TopicSegmenter(config)

turns = [
    {"id": "t0", "text": "Let's discuss the budget", "start": 0, "end": 5},
    {"id": "t1", "text": "The budget needs review", "start": 5, "end": 10},
    {"id": "t2", "text": "Now about the timeline", "start": 60, "end": 65},
    # ...
]

topics = segmenter.segment(turns)

for topic in topics:
    print(f"{topic.id}: {topic.start:.1f}s - {topic.end:.1f}s")
    print(f"  Keywords: {topic.keywords[:5]}")
    print(f"  Speakers: {topic.speaker_ids}")
```

### Streaming Segmentation

```python
def on_topic_boundary(payload):
    print(f"Topic changed from {payload.previous_topic_id} to {payload.new_topic_id}")
    print(f"Similarity: {payload.similarity_score:.2f}")
    print(f"Previous keywords: {payload.keywords_previous}")

segmenter = StreamingTopicSegmenter(config, on_boundary=on_topic_boundary)

# Process turns as they arrive
for turn in turn_stream:
    boundary = segmenter.add_turn(turn)
    if boundary:
        print(f"Boundary detected at {boundary.boundary_time}s")

# Get all topics at end
topics = segmenter.finalize()
```

### TopicChunk Structure

```python
@dataclass
class TopicChunk:
    id: str                         # "topic_0", "topic_1", etc.
    start: float                    # Start time (seconds)
    end: float                      # End time (seconds)
    turn_ids: list[str]            # IDs of turns in topic
    turn_range: tuple[int, int]    # (first_idx, last_idx)
    summary_text: str              # Concatenated text (max 500 chars)
    keywords: list[str]            # Top 10 keywords by TF-IDF
    speaker_ids: list[str]         # Participating speakers
```

---

## Turn-Taking Policies

Turn-taking policies control when `END_OF_TURN_HINT` events are emitted. This is critical for conversational AI applications that need to know when a speaker has finished.

### Policy Presets

| Policy | Silence Threshold | Confidence | Use Case |
|--------|-------------------|------------|----------|
| **aggressive** | 300ms | 0.6 | Fast response, chatbots |
| **balanced** | 700ms | 0.75 | General use (default) |
| **conservative** | 1200ms | 0.85 | High accuracy, transcription |

### Using Policies

```python
from transcription.turn_taking_policy import (
    TurnTakingEvaluator,
    get_policy,
    AGGRESSIVE_POLICY,
    BALANCED_POLICY,
    CONSERVATIVE_POLICY,
)

# Get preset by name
policy = get_policy("balanced")

# Or use constant directly
evaluator = TurnTakingEvaluator(CONSERVATIVE_POLICY)

# Evaluate turn end
result = evaluator.evaluate(
    text="How can I help you today?",
    silence_duration_ms=800,
    turn_duration_ms=2500,
    prosody_falling=True,      # Optional: from prosody extraction
    prosody_flat=False,
)

if result.is_end_of_turn:
    print(f"Turn ended with confidence {result.confidence:.2f}")
    print(f"Reasons: {result.reason_codes}")
    # ['TERMINAL_PUNCT', 'SILENCE_THRESHOLD', 'QUESTION_DETECTED']
```

### Custom Policies

```python
from transcription.turn_taking_policy import TurnTakingPolicy

custom_policy = TurnTakingPolicy(
    name="custom",
    silence_threshold_ms=500,
    confidence_threshold=0.7,
    enable_prosody=True,
    enable_punctuation=True,
    enable_sentence_completion=True,
    min_turn_duration_ms=400,
    max_silence_for_continuation_ms=2000,
    prosody_weight=0.30,
    punctuation_weight=0.40,
    silence_weight=0.30,
)

evaluator = TurnTakingEvaluator(custom_policy)
```

### Policy Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `silence_threshold_ms` | Minimum silence to consider turn end | 100-2000 |
| `confidence_threshold` | Minimum confidence to emit hint | 0.0-1.0 |
| `enable_prosody` | Use falling intonation detection | bool |
| `enable_punctuation` | Use terminal punctuation (.!?) | bool |
| `enable_sentence_completion` | Use sentence completion heuristics | bool |
| `min_turn_duration_ms` | Minimum turn before considering end | 100-1000 |
| `max_silence_for_continuation_ms` | Force end after this silence | 1000-10000 |
| `prosody_weight` | Weight for prosodic signals | 0.0-1.0 |
| `punctuation_weight` | Weight for punctuation signals | 0.0-1.0 |
| `silence_weight` | Weight for silence signals | 0.0-1.0 |

### Reason Codes

The `reason_codes` in evaluation results indicate why turn-end was detected:

| Code | Description |
|------|-------------|
| `SILENCE_THRESHOLD` | Silence exceeded threshold |
| `TERMINAL_PUNCT` | Terminal punctuation detected (.!?;) |
| `FALLING_INTONATION` | Prosody indicates falling pitch |
| `COMPLETE_SENTENCE` | Text appears to be complete sentence |
| `QUESTION_DETECTED` | Question mark detected |
| `LONG_PAUSE` | Maximum silence exceeded (forced end) |

---

## Preset Configurations

Pre-built configurations for common use cases.

### Call Center

Optimized for customer service calls with PII protection and role detection.

```python
from transcription.post_process import post_process_config_for_call_center

config = post_process_config_for_call_center()
# Enables:
# - Safety processing with PII masking
# - Role inference (agent/customer)
# - Turn-taking with balanced policy

processor = PostProcessor(config)
```

### Meetings

Optimized for meeting recordings with topic chapters.

```python
from transcription.post_process import post_process_config_for_meetings

config = post_process_config_for_meetings()
# Enables:
# - Topic segmentation (60s-600s topics)
# - Role inference (facilitator detection)
# - Environment classification

processor = PostProcessor(config)
```

### Minimal

Just smart formatting, no other processing.

```python
from transcription.post_process import post_process_config_minimal

config = post_process_config_minimal()
# Enables:
# - Safety with formatting only (no PII, no moderation)

processor = PostProcessor(config)
```

---

## Integration with Streaming

Post-processing integrates with the streaming enrichment system via callbacks.

### Streaming Enrichment Session

```python
from transcription.streaming_enrichment import StreamingEnrichmentSession
from transcription.post_process import PostProcessConfig

# Configure post-processing for streaming
post_config = PostProcessConfig(
    enable_safety=True,
    enable_turn_taking=True,
    turn_taking_policy="aggressive",  # Fast response for real-time
)

session = StreamingEnrichmentSession(
    config=enrichment_config,
    post_process_config=post_config,
    on_end_of_turn_hint=lambda p: print(f"Turn ended: {p.confidence}"),
    on_topic_boundary=lambda p: print(f"New topic: {p.new_topic_id}"),
)
```

### Callback Payloads

**EndOfTurnHintPayload:**
```python
@dataclass
class EndOfTurnHintPayload:
    confidence: float           # 0.0-1.0
    silence_duration: float     # seconds
    terminal_punctuation: bool  # punctuation detected
    partial_text: str           # text that triggered hint
    reason_codes: list[str]     # lowercase snake_case codes
    silence_duration_ms: float  # milliseconds
    policy_name: str            # policy used
```

**TopicBoundaryPayload:**
```python
@dataclass
class TopicBoundaryPayload:
    previous_topic_id: str      # e.g., "topic_0"
    new_topic_id: str           # e.g., "topic_1"
    boundary_turn_id: str       # turn where boundary detected
    boundary_time: float        # seconds
    similarity_score: float     # similarity that triggered boundary
    keywords_previous: list[str]
    keywords_new: list[str]
```

**RoleAssignedPayload:**
```python
@dataclass
class RoleAssignedPayload:
    assignments: dict[str, dict]  # speaker_id -> role info
    timestamp: float              # when decision was made
    trigger: str                  # "turn_count", "elapsed_time", or "finalize"
```

---

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `PostProcessor` | `post_process` | Main orchestrator |
| `PostProcessConfig` | `post_process` | Configuration |
| `SegmentContext` | `post_process` | Segment processing context |
| `PostProcessResult` | `post_process` | Segment processing result |
| `TurnProcessResult` | `post_process` | Turn processing result |
| `TopicSegmenter` | `topic_segmentation` | Batch topic segmentation |
| `StreamingTopicSegmenter` | `topic_segmentation` | Streaming topic segmentation |
| `TopicChunk` | `topic_segmentation` | Topic segment data |
| `TurnTakingEvaluator` | `turn_taking_policy` | Turn-end evaluation |
| `TurnTakingPolicy` | `turn_taking_policy` | Policy configuration |

### Preset Functions

| Function | Description |
|----------|-------------|
| `post_process_config_for_call_center()` | Call center config |
| `post_process_config_for_meetings()` | Meeting config |
| `post_process_config_minimal()` | Minimal config |
| `get_policy(name)` | Get turn-taking policy by name |

### Constants

| Constant | Description |
|----------|-------------|
| `AGGRESSIVE_POLICY` | Fast response turn-taking |
| `BALANCED_POLICY` | Default turn-taking |
| `CONSERVATIVE_POLICY` | High accuracy turn-taking |
| `STOPWORDS` | Default stopwords for topic segmentation |

---

## Related Documentation

- [Streaming Architecture](STREAMING_ARCHITECTURE.md) - Real-time pipeline design
- [Safety Configuration](REDACTION.md) - PII and safety processing
- [Audio Enrichment](AUDIO_ENRICHMENT.md) - Prosody and emotion
- [Configuration](CONFIGURATION.md) - General configuration reference

---

**Last Updated:** 2026-01-31
