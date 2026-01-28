# LLM-Backed Semantic Annotator Design Document

**Version:** 2.0.0 (Implemented)
**Status:** Implemented in v2.0.0
**Last Updated:** 2026-01-28

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Local Provider Setup](#local-provider-setup)
4. [Architecture](#architecture)
5. [Schema Design](#schema-design)
6. [Configuration](#configuration)
7. [Backend Implementations](#backend-implementations)
8. [Prompt Engineering](#prompt-engineering)
9. [Safety and Guardrails](#safety-and-guardrails)
10. [Migration Guide](#migration-guide)
11. [Performance Considerations](#performance-considerations)

---

## Quick Start

### Using the Local Provider (Recommended)

```python
from transcription.semantic import create_adapter, ChunkContext

# Create local LLM adapter (uses Qwen2.5-7B by default)
adapter = create_adapter("local-llm")

# Check availability
health = adapter.health_check()
if not health.available:
    print(f"Local LLM not available: {health.error}")
    # Fall back to keyword-based adapter
    adapter = create_adapter("local")

# Annotate a chunk
context = ChunkContext(speaker_id="agent", start=0.0, end=30.0)
annotation = adapter.annotate_chunk(
    "I'll send you the pricing proposal by end of day tomorrow.",
    context,
)

# Access results
print(f"Topics: {annotation.normalized.topics}")
print(f"Risk tags: {annotation.normalized.risk_tags}")
print(f"Action items: {[a.text for a in annotation.normalized.action_items]}")
print(f"Confidence: {annotation.confidence}")
```

### Using the Provider Class (More Control)

```python
from transcription.semantic_providers.local import (
    LocalSemanticProvider,
    LocalSemanticConfig,
    is_available,
)

# Check if dependencies are installed
if not is_available():
    print("Install torch and transformers: pip install 'slower-whisper[emotion]'")
    exit(1)

# Configure the provider
config = LocalSemanticConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",  # Faster model
    device="auto",  # Use GPU if available
    temperature=0.0,  # Deterministic output
    extraction_mode="combined",  # Extract topics, risks, and actions
)

# Create provider
provider = LocalSemanticProvider(config)

# Run health check
health = provider.health_check()
print(f"Available: {health.available}")
print(f"Latency: {health.latency_ms}ms")
```

---

## Local Provider Setup

### Installation

The local LLM provider requires `torch` and `transformers`. Install with:

```bash
# Install with emotion/LLM dependencies
pip install 'slower-whisper[emotion]'

# Or install directly
pip install torch transformers
```

### Resource Requirements

| Model | Parameters | VRAM (GPU) | RAM (CPU) | Quality | Speed |
|-------|------------|------------|-----------|---------|-------|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~4GB | ~14GB | High | Medium |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ~2GB | ~6GB | Medium | Fast |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | ~1GB | ~4GB | Lower | Very Fast |
| `microsoft/phi-3-mini-4k-instruct` | 3.8B | ~2GB | ~8GB | Medium | Fast |

**Recommendations:**
- **GPU with 4GB+ VRAM**: Use `Qwen/Qwen2.5-7B-Instruct` (best quality)
- **GPU with 2GB VRAM**: Use `Qwen/Qwen2.5-3B-Instruct`
- **CPU only**: Use `HuggingFaceTB/SmolLM2-1.7B-Instruct` (faster) or `Qwen2.5-3B` (better quality)

### Environment Configuration

Configure the local provider via environment variables:

```bash
# Model selection
export SLOWER_WHISPER_SEMANTIC_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"

# Device selection (auto, cuda, cpu)
export SLOWER_WHISPER_SEMANTIC_DEVICE="auto"

# Generation settings
export SLOWER_WHISPER_SEMANTIC_TEMPERATURE="0.1"
export SLOWER_WHISPER_SEMANTIC_MAX_TOKENS="1024"

# Extraction mode (combined, topics, risks, actions)
export SLOWER_WHISPER_SEMANTIC_EXTRACTION_MODE="combined"

# Caching
export SLOWER_WHISPER_SEMANTIC_ENABLE_CACHING="true"
```

### Checking Availability

```python
from transcription.semantic_providers.local import is_available, get_availability_status

# Quick check
if is_available():
    print("Local LLM ready")
else:
    print("Dependencies missing")

# Detailed status
status = get_availability_status()
print(f"torch: {status['torch']}")
print(f"transformers: {status['transformers']}")
print(f"available: {status['available']}")
```

### Error Handling

The local provider handles errors gracefully:

```python
from transcription.semantic import create_adapter, ChunkContext

adapter = create_adapter("local-llm")
context = ChunkContext()

# If model not available, returns empty annotation with confidence 0
result = adapter.annotate_chunk("Test text", context)

if result.confidence == 0.0:
    # Check raw_model_output for error details
    error = result.raw_model_output.get("error") if result.raw_model_output else None
    print(f"Annotation failed: {error}")
```

---

## Overview

### Purpose

The LLM-backed semantic annotator extends slower-whisper's L3 (Semantic Enrichment) layer with intelligent, context-aware annotation capabilities. Unlike the v1.x keyword-based `KeywordSemanticAnnotator`, this system leverages language models to:

1. **Extract nuanced topics** - Beyond keyword matching, understand contextual meaning
2. **Detect complex risks** - Identify subtle escalation signals, sentiment shifts, compliance concerns
3. **Generate actionable items** - Extract commitments, deadlines, and assignees with context
4. **Provide confidence scores** - Quantify annotation reliability for downstream filtering

### Goals

| Goal | Description |
|------|-------------|
| **Local-First** | Default to local SLMs (Qwen2.5, SmolLM) with no cloud dependency |
| **Opt-In Cloud** | Optional OpenAI/Anthropic backends for higher quality when permitted |
| **Backward Compatible** | v1.x consumers can ignore new fields; existing workflows unaffected |
| **Chunk-Aware** | Operate on `Chunk` boundaries for efficient context windowing |
| **Guardrailed** | Rate limiting, cost controls, PII detection, output validation |

### Non-Goals

- **Real-time streaming annotation** - This is handled by `LiveSemanticSession` (v1.7+)
- **Training or fine-tuning** - Use pre-trained models only
- **Cloud-first design** - Cloud backends are optional, not primary

### Relationship to Existing Components

```
                  +-----------------------+
                  |    L0: Ingestion      |
                  |  (audio → normalized) |
                  +-----------+-----------+
                              |
                              v
                  +-----------+-----------+
                  |    L1: ASR (Whisper)  |
                  |  (audio → segments)   |
                  +-----------+-----------+
                              |
                              v
+-----------------+-----------+-----------+------------------+
|                 L2: Acoustic Enrichment                    |
|  - Diarization (pyannote)                                  |
|  - Prosody (librosa, parselmouth)                          |
|  - Emotion (wav2vec2)                                      |
|  - Turns + Speaker Stats                                   |
+-----------------+-----------+------------------------------+
                              |
                              v
+-----------------+-----------+-----------+------------------+
|                 L3: Semantic Enrichment                    |
|  +--------------------------------------------------+      |
|  |  v1.x: KeywordSemanticAnnotator (rule-based)     |      |
|  |  - Regex patterns for escalation, churn, pricing |      |
|  |  - Action item detection via pattern matching    |      |
|  +--------------------------------------------------+      |
|                                                            |
|  +--------------------------------------------------+      |
|  |  v2.0: LLMSemanticAnnotator (NEW)                |      |
|  |  - Topic extraction with confidence scores       |      |
|  |  - Risk detection with severity and evidence     |      |
|  |  - Action items with assignees and deadlines     |      |
|  |  - Local SLM or cloud LLM backends               |      |
|  +--------------------------------------------------+      |
+------------------------------------------------------------+
                              |
                              v
+-----------------+-----------+-----------+------------------+
|                 L4: Task Outputs                           |
|  - Meeting notes, QA, coaching, summarization              |
+------------------------------------------------------------+
```

---

## Architecture

### Component Diagram

```
+-------------------+     +------------------------+
|  EnrichmentConfig |---->| SemanticLLMConfig      |
|  (enable_semantic |     | - backend              |
|   _llm_annotator) |     | - model                |
+-------------------+     | - enable_topics        |
                          | - enable_risks         |
                          | - enable_actions       |
                          | - rate_limit_rpm       |
                          | - max_tokens_per_chunk |
                          +------------------------+
                                     |
                                     v
+-------------------+     +------------------------+
|  Transcript       |---->| LLMSemanticAnnotator   |
|  - chunks[]       |     | - annotate(transcript) |
|  - segments[]     |     | - annotate_chunk(chunk)|
|  - turns[]        |     +------------------------+
+-------------------+                |
                                     |
                          +----------+---------+
                          |                    |
                          v                    v
               +------------------+  +------------------+
               | LocalLLMBackend  |  | CloudLLMBackend  |
               | - qwen2.5-7b     |  | - openai         |
               | - smollm         |  | - anthropic      |
               | - llama.cpp      |  +------------------+
               +------------------+
                          |
                          v
               +------------------------+
               | SemanticAnnotation     |
               | - version: "2.0.0"     |
               | - annotator: "llm"     |
               | - model: "qwen2.5-7b"  |
               | - topics[]             |
               | - risks[]              |
               | - actions[]            |
               +------------------------+
```

### Data Flow

1. **Input**: `Transcript` with populated `chunks[]` (from L1/L2 processing)
2. **Chunking**: Each `Chunk` is processed independently for memory efficiency
3. **Prompt Construction**: Build structured prompt with chunk text and context
4. **LLM Inference**: Send to configured backend (local or cloud)
5. **Output Parsing**: Validate and parse JSON response
6. **Aggregation**: Merge chunk-level annotations into transcript-level summary
7. **Output**: Updated `transcript.annotations.semantic` with v2.0.0 schema

### Protocol Pattern

Following slower-whisper's established protocol pattern (see `emotion.py`, `semantic.py`):

```python
from typing import Protocol

class SemanticLLMAnnotatorLike(Protocol):
    """Protocol for LLM-backed semantic annotators."""

    def annotate(self, transcript: Transcript) -> Transcript:
        """Annotate a transcript with LLM-derived semantic tags."""
        ...

    def annotate_chunk(self, chunk: Chunk, context: str | None = None) -> dict:
        """Annotate a single chunk with optional context window."""
        ...
```

---

## Schema Design

### Full JSON Schema for `annotations.semantic` v2.0.0

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SemanticAnnotation",
  "description": "LLM-backed semantic annotation schema v2.0.0",
  "type": "object",
  "required": ["version", "annotator"],
  "properties": {
    "version": {
      "type": "string",
      "const": "2.0.0",
      "description": "Schema version for semantic annotations"
    },
    "annotator": {
      "type": "string",
      "enum": ["keyword", "llm"],
      "description": "Type of annotator used (keyword=v1.x, llm=v2.0)"
    },
    "model": {
      "type": ["string", "null"],
      "description": "Model identifier when annotator=llm (e.g., qwen2.5-7b, gpt-4o-mini)"
    },
    "backend": {
      "type": ["string", "null"],
      "enum": ["local", "openai", "anthropic", null],
      "description": "Backend type when annotator=llm"
    },
    "generated_at": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp of annotation generation"
    },
    "topics": {
      "type": "array",
      "description": "Detected topics with confidence and span information",
      "items": {
        "$ref": "#/definitions/Topic"
      }
    },
    "risks": {
      "type": "array",
      "description": "Detected risk signals with severity and evidence",
      "items": {
        "$ref": "#/definitions/Risk"
      }
    },
    "actions": {
      "type": "array",
      "description": "Extracted action items with ownership and deadlines",
      "items": {
        "$ref": "#/definitions/Action"
      }
    },
    "summary": {
      "type": ["string", "null"],
      "description": "Optional LLM-generated summary of the conversation"
    },
    "keywords": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Backward-compatible keyword list (merged from v1.x + LLM)"
    },
    "risk_tags": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Backward-compatible risk tag list (merged from v1.x + LLM)"
    },
    "extraction_status": {
      "type": "object",
      "description": "Status of each extraction component",
      "properties": {
        "topics": {"type": "string", "enum": ["success", "partial", "error", "skipped"]},
        "risks": {"type": "string", "enum": ["success", "partial", "error", "skipped"]},
        "actions": {"type": "string", "enum": ["success", "partial", "error", "skipped"]}
      }
    }
  },
  "definitions": {
    "Topic": {
      "type": "object",
      "required": ["label", "confidence"],
      "properties": {
        "label": {
          "type": "string",
          "description": "Topic label (e.g., 'pricing', 'technical_support', 'billing')"
        },
        "confidence": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Confidence score from LLM (0.0-1.0)"
        },
        "span": {
          "type": ["array", "null"],
          "items": {"type": "integer"},
          "minItems": 2,
          "maxItems": 2,
          "description": "Segment ID range [start, end] where topic is discussed"
        },
        "chunk_ids": {
          "type": ["array", "null"],
          "items": {"type": "string"},
          "description": "List of chunk IDs where this topic was detected"
        },
        "evidence": {
          "type": ["string", "null"],
          "description": "Brief excerpt or explanation supporting the topic"
        }
      }
    },
    "Risk": {
      "type": "object",
      "required": ["type", "severity"],
      "properties": {
        "type": {
          "type": "string",
          "enum": [
            "escalation",
            "churn_risk",
            "compliance",
            "sentiment_negative",
            "legal",
            "pricing_objection",
            "competitor_mention",
            "customer_frustration",
            "agent_error",
            "other"
          ],
          "description": "Risk category"
        },
        "severity": {
          "type": "string",
          "enum": ["low", "medium", "high", "critical"],
          "description": "Risk severity level"
        },
        "confidence": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Confidence score from LLM"
        },
        "evidence": {
          "type": ["string", "null"],
          "description": "Quoted text or explanation supporting the risk"
        },
        "segment_ids": {
          "type": ["array", "null"],
          "items": {"type": "integer"},
          "description": "Segment IDs where risk was detected"
        },
        "speaker_id": {
          "type": ["string", "null"],
          "description": "Speaker who triggered the risk (if applicable)"
        },
        "recommended_action": {
          "type": ["string", "null"],
          "description": "Suggested response or mitigation"
        }
      }
    },
    "Action": {
      "type": "object",
      "required": ["description"],
      "properties": {
        "description": {
          "type": "string",
          "description": "Description of the action item"
        },
        "assignee": {
          "type": ["string", "null"],
          "description": "Speaker ID or role assigned to the action (null if unassigned)"
        },
        "assignee_role": {
          "type": ["string", "null"],
          "enum": ["agent", "customer", "manager", "team", "unknown", null],
          "description": "Role of the assignee (inferred from context)"
        },
        "due": {
          "type": ["string", "null"],
          "description": "Due date or deadline (ISO 8601 or relative like 'tomorrow', 'end of week')"
        },
        "priority": {
          "type": ["string", "null"],
          "enum": ["low", "medium", "high", null],
          "description": "Priority level (inferred from urgency signals)"
        },
        "status": {
          "type": "string",
          "enum": ["pending", "in_progress", "completed", "unknown"],
          "default": "pending",
          "description": "Action item status"
        },
        "segment_ids": {
          "type": ["array", "null"],
          "items": {"type": "integer"},
          "description": "Segment IDs where action was mentioned"
        },
        "confidence": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Confidence score for action extraction"
        },
        "verbatim": {
          "type": ["string", "null"],
          "description": "Original text where action was mentioned"
        }
      }
    }
  }
}
```

### Example Output

```json
{
  "annotations": {
    "semantic": {
      "version": "2.0.0",
      "annotator": "llm",
      "model": "qwen2.5-7b",
      "backend": "local",
      "generated_at": "2025-12-31T10:30:00Z",
      "topics": [
        {
          "label": "pricing",
          "confidence": 0.92,
          "span": [0, 5],
          "chunk_ids": ["chunk_0"],
          "evidence": "Customer mentioned budget constraints and asked about discount options"
        },
        {
          "label": "technical_support",
          "confidence": 0.78,
          "span": [6, 12],
          "chunk_ids": ["chunk_1"],
          "evidence": "Discussion of login issues and password reset procedures"
        }
      ],
      "risks": [
        {
          "type": "escalation",
          "severity": "high",
          "confidence": 0.88,
          "evidence": "Customer stated 'I want to speak to your manager immediately'",
          "segment_ids": [3],
          "speaker_id": "spk_1",
          "recommended_action": "Transfer to supervisor and document incident"
        },
        {
          "type": "churn_risk",
          "severity": "medium",
          "confidence": 0.75,
          "evidence": "Customer mentioned considering competitor products",
          "segment_ids": [7, 8],
          "speaker_id": "spk_1",
          "recommended_action": "Offer retention discount and schedule follow-up"
        }
      ],
      "actions": [
        {
          "description": "Send pricing proposal with 20% discount offer",
          "assignee": "spk_0",
          "assignee_role": "agent",
          "due": "2025-01-02",
          "priority": "high",
          "status": "pending",
          "segment_ids": [15],
          "confidence": 0.95,
          "verbatim": "I'll send you the proposal by end of day tomorrow"
        },
        {
          "description": "Schedule technical demo with engineering team",
          "assignee": null,
          "assignee_role": "team",
          "due": "next week",
          "priority": "medium",
          "status": "pending",
          "segment_ids": [18, 19],
          "confidence": 0.82,
          "verbatim": "We should set up a demo with your technical team"
        }
      ],
      "summary": "Customer support call regarding pricing concerns and login issues. Customer expressed frustration and requested escalation. Agent de-escalated successfully and committed to sending discount proposal.",
      "keywords": ["pricing", "discount", "escalation", "login", "competitor"],
      "risk_tags": ["escalation", "churn_risk"],
      "extraction_status": {
        "topics": "success",
        "risks": "success",
        "actions": "success"
      }
    }
  }
}
```

---

## Configuration

### `SemanticLLMConfig` Dataclass

```python
from dataclasses import dataclass, field
from typing import Literal, Any

@dataclass(slots=True)
class SemanticLLMConfig:
    """
    Configuration for LLM-backed semantic annotation.

    This config controls the behavior of the LLMSemanticAnnotator, including
    backend selection, model choice, feature flags, and safety controls.

    Attributes:
        backend: LLM backend to use for annotation.
            - "local": Use local models via transformers or llama.cpp
            - "openai": Use OpenAI API (requires OPENAI_API_KEY)
            - "anthropic": Use Anthropic API (requires ANTHROPIC_API_KEY)
            Default: "local" (no cloud dependency)

        model: Model identifier for the selected backend.
            - Local: "qwen2.5-7b", "qwen2.5-3b", "smollm-1.7b", "llama-3.2-3b"
            - OpenAI: "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"
            - Anthropic: "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"
            Default: "qwen2.5-7b" (good balance of quality and speed)

        enable_topics: Whether to extract topic labels from chunks.
            Default: True

        enable_risks: Whether to detect risk signals (escalation, churn, etc.).
            Default: True

        enable_actions: Whether to extract action items with ownership.
            Default: True

        enable_summary: Whether to generate a conversation summary.
            Default: False (adds latency, use for batch processing)

        max_tokens_per_chunk: Maximum tokens to send per chunk.
            Chunks exceeding this limit are split or truncated.
            Default: 500

        max_output_tokens: Maximum tokens for LLM response.
            Default: 1024

        temperature: LLM temperature for generation.
            Lower = more deterministic, higher = more creative.
            Default: 0.1 (favor consistency for structured extraction)

        rate_limit_rpm: Maximum requests per minute (cloud backends only).
            Default: 60 (OpenAI tier-1 limit)

        rate_limit_tpm: Maximum tokens per minute (cloud backends only).
            Default: 90000 (OpenAI tier-1 limit)

        cost_limit_usd: Maximum cost per transcript in USD (cloud backends only).
            Processing stops if limit is exceeded.
            Default: 1.0

        timeout_seconds: Timeout for LLM API calls.
            Default: 30.0

        retry_attempts: Number of retry attempts for failed API calls.
            Default: 3

        retry_delay_seconds: Initial delay between retries (exponential backoff).
            Default: 1.0

        cache_enabled: Whether to cache LLM responses.
            Default: True (avoid redundant API calls)

        cache_ttl_seconds: Cache time-to-live in seconds.
            Default: 86400 (24 hours)

        pii_detection_enabled: Whether to scan for PII before sending to LLM.
            Default: True (blocks PII from cloud backends)

        pii_action: Action to take when PII is detected.
            - "redact": Replace PII with [REDACTED] tokens
            - "skip": Skip the chunk entirely
            - "warn": Log warning but proceed
            Default: "redact"

        output_validation_enabled: Whether to validate LLM JSON output.
            Default: True (ensures schema compliance)

        fallback_to_keyword: Whether to fall back to KeywordSemanticAnnotator on error.
            Default: True (graceful degradation)

        local_model_device: Device for local model inference.
            - "auto": Automatically select based on availability
            - "cuda": Force GPU (requires CUDA)
            - "cpu": Force CPU
            Default: "auto"

        local_model_quantization: Quantization level for local models.
            - None: No quantization (highest quality, most memory)
            - "4bit": 4-bit quantization (good balance)
            - "8bit": 8-bit quantization (better quality than 4bit)
            Default: "4bit" (fits 7B models in ~4GB VRAM)

        topic_taxonomy: Optional custom topic taxonomy.
            If provided, LLM will classify into these categories.
            Default: None (LLM generates free-form topics)

        risk_taxonomy: Optional custom risk taxonomy.
            Default: None (use built-in risk types)

        context_window_chunks: Number of preceding chunks to include as context.
            Default: 2 (helps with topic continuity)

        merge_with_keyword_annotator: Whether to merge results with KeywordSemanticAnnotator.
            Default: True (combines rule-based and LLM results)

    Example:
        >>> config = SemanticLLMConfig(
        ...     backend="local",
        ...     model="qwen2.5-7b",
        ...     enable_topics=True,
        ...     enable_risks=True,
        ...     enable_actions=True,
        ...     max_tokens_per_chunk=500,
        ... )
        >>> annotator = LLMSemanticAnnotator(config)
    """

    # Backend selection
    backend: Literal["local", "openai", "anthropic"] = "local"
    model: str = "qwen2.5-7b"

    # Feature flags
    enable_topics: bool = True
    enable_risks: bool = True
    enable_actions: bool = True
    enable_summary: bool = False

    # Token limits
    max_tokens_per_chunk: int = 500
    max_output_tokens: int = 1024

    # Generation parameters
    temperature: float = 0.1

    # Rate limiting (cloud backends)
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 90000
    cost_limit_usd: float = 1.0

    # Timeouts and retries
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400

    # Safety
    pii_detection_enabled: bool = True
    pii_action: Literal["redact", "skip", "warn"] = "redact"
    output_validation_enabled: bool = True
    fallback_to_keyword: bool = True

    # Local model settings
    local_model_device: Literal["auto", "cuda", "cpu"] = "auto"
    local_model_quantization: Literal[None, "4bit", "8bit"] = "4bit"

    # Taxonomy customization
    topic_taxonomy: list[str] | None = None
    risk_taxonomy: list[str] | None = None

    # Context
    context_window_chunks: int = 2

    # Integration
    merge_with_keyword_annotator: bool = True

    # Internal tracking
    _source_fields: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_tokens_per_chunk <= 0:
            raise ValueError(f"max_tokens_per_chunk must be > 0, got {self.max_tokens_per_chunk}")
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be > 0, got {self.max_output_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.rate_limit_rpm <= 0:
            raise ValueError(f"rate_limit_rpm must be > 0, got {self.rate_limit_rpm}")
        if self.cost_limit_usd < 0:
            raise ValueError(f"cost_limit_usd must be >= 0, got {self.cost_limit_usd}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.context_window_chunks < 0:
            raise ValueError(f"context_window_chunks must be >= 0, got {self.context_window_chunks}")
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SLOWER_WHISPER_SEMANTIC_BACKEND` | LLM backend (local/openai/anthropic) | `local` |
| `SLOWER_WHISPER_SEMANTIC_MODEL` | Model identifier | `qwen2.5-7b` |
| `SLOWER_WHISPER_SEMANTIC_ENABLE_TOPICS` | Enable topic extraction | `true` |
| `SLOWER_WHISPER_SEMANTIC_ENABLE_RISKS` | Enable risk detection | `true` |
| `SLOWER_WHISPER_SEMANTIC_ENABLE_ACTIONS` | Enable action extraction | `true` |
| `SLOWER_WHISPER_SEMANTIC_RATE_LIMIT_RPM` | Rate limit (requests/min) | `60` |
| `SLOWER_WHISPER_SEMANTIC_COST_LIMIT_USD` | Cost limit per transcript | `1.0` |
| `SLOWER_WHISPER_SEMANTIC_PII_ACTION` | PII action (redact/skip/warn) | `redact` |
| `OPENAI_API_KEY` | OpenAI API key (for openai backend) | - |
| `ANTHROPIC_API_KEY` | Anthropic API key (for anthropic backend) | - |

---

## Backend Implementations

### Local Model Backend

The local backend runs models directly on the user's machine using either HuggingFace Transformers or llama.cpp.

#### Supported Models

| Model | Parameters | VRAM (4-bit) | Quality | Speed | Notes |
|-------|------------|--------------|---------|-------|-------|
| `qwen2.5-7b` | 7B | ~4GB | High | Medium | **Recommended default** |
| `qwen2.5-3b` | 3B | ~2GB | Medium | Fast | Good for CPU/low VRAM |
| `smollm-1.7b` | 1.7B | ~1GB | Low-Medium | Very Fast | Minimal resource usage |
| `llama-3.2-3b` | 3B | ~2GB | Medium | Fast | Alternative to Qwen |

#### Implementation Sketch

```python
from typing import Protocol, Any
import logging

logger = logging.getLogger(__name__)

# Check for transformers availability
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore


class LLMBackendLike(Protocol):
    """Protocol for LLM backend implementations."""

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text completion for the given prompt."""
        ...

    def is_available(self) -> bool:
        """Check if the backend is available and ready."""
        ...


class LocalLLMBackend:
    """
    Local LLM backend using HuggingFace Transformers.

    Supports quantization for memory efficiency and automatic device selection.
    """

    MODEL_REGISTRY = {
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
        "smollm-1.7b": "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    }

    def __init__(self, config: SemanticLLMConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None

    def _lazy_load(self) -> None:
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for local LLM backend. "
                "Install with: pip install transformers torch bitsandbytes"
            )

        model_id = self.MODEL_REGISTRY.get(self.config.model, self.config.model)

        # Determine device
        if self.config.local_model_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.local_model_device

        # Configure quantization
        quantization_config = None
        if self.config.local_model_quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.config.local_model_quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(f"Loading local model: {model_id} on {self._device}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=self._device if quantization_config else None,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )

        if quantization_config is None and self._device == "cuda":
            self._model = self._model.to(self._device)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate completion using local model."""
        self._lazy_load()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated portion (after the prompt)
        return response[len(prompt):].strip()

    def is_available(self) -> bool:
        """Check if transformers is available."""
        return TRANSFORMERS_AVAILABLE


class DummyLocalLLMBackend:
    """Fallback backend when transformers is unavailable."""

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return '{"topics": [], "risks": [], "actions": []}'

    def is_available(self) -> bool:
        return False
```

### OpenAI Backend

```python
import os
import time
from typing import Any

# Check for openai availability
OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore


class OpenAILLMBackend:
    """
    OpenAI API backend with rate limiting and cost tracking.

    Requires OPENAI_API_KEY environment variable.
    """

    # Pricing per 1M tokens (as of Dec 2024)
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, config: SemanticLLMConfig) -> None:
        self.config = config
        self._client = None
        self._request_times: list[float] = []
        self._total_cost_usd: float = 0.0

    def _lazy_init(self) -> None:
        """Initialize OpenAI client on first use."""
        if self._client is not None:
            return

        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Install with: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        self._client = openai.OpenAI(api_key=api_key)

    def _check_rate_limit(self) -> None:
        """Block if rate limit would be exceeded."""
        now = time.time()
        # Remove requests older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.config.rate_limit_rpm:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                time.sleep(wait_time)

        self._request_times.append(now)

    def _check_cost_limit(self, estimated_cost: float) -> None:
        """Raise if cost limit would be exceeded."""
        if self._total_cost_usd + estimated_cost > self.config.cost_limit_usd:
            raise RuntimeError(
                f"Cost limit exceeded: ${self._total_cost_usd:.4f} + ${estimated_cost:.4f} "
                f"> ${self.config.cost_limit_usd:.2f}"
            )

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        pricing = self.PRICING.get(self.config.model, {"input": 0.15, "output": 0.60})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate completion using OpenAI API."""
        self._lazy_init()
        self._check_rate_limit()

        # Estimate input tokens (rough approximation)
        input_tokens = len(prompt) // 4
        estimated_cost = self._estimate_cost(input_tokens, max_tokens)
        self._check_cost_limit(estimated_cost)

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are a semantic annotation assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
            timeout=self.config.timeout_seconds,
        )

        # Track actual cost
        usage = response.usage
        actual_cost = self._estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        self._total_cost_usd += actual_cost

        return response.choices[0].message.content

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY") is not None

    def get_total_cost(self) -> float:
        """Return total cost incurred so far."""
        return self._total_cost_usd
```

### Anthropic Backend

```python
import os
import time

# Check for anthropic availability
ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore


class AnthropicLLMBackend:
    """
    Anthropic Claude API backend with rate limiting.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    # Pricing per 1M tokens (as of Dec 2024)
    PRICING = {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    }

    def __init__(self, config: SemanticLLMConfig) -> None:
        self.config = config
        self._client = None
        self._request_times: list[float] = []
        self._total_cost_usd: float = 0.0

    def _lazy_init(self) -> None:
        """Initialize Anthropic client on first use."""
        if self._client is not None:
            return

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")

        self._client = anthropic.Anthropic(api_key=api_key)

    def _check_rate_limit(self) -> None:
        """Block if rate limit would be exceeded."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.config.rate_limit_rpm:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                time.sleep(wait_time)

        self._request_times.append(now)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate completion using Anthropic API."""
        self._lazy_init()
        self._check_rate_limit()

        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system="You are a semantic annotation assistant. Always respond with valid JSON.",
        )

        # Track cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        pricing = self.PRICING.get(self.config.model, {"input": 0.25, "output": 1.25})
        self._total_cost_usd += (input_tokens / 1_000_000) * pricing["input"]
        self._total_cost_usd += (output_tokens / 1_000_000) * pricing["output"]

        return response.content[0].text

    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY") is not None
```

### Backend Factory

```python
def get_llm_backend(config: SemanticLLMConfig) -> LLMBackendLike:
    """
    Factory function to get the appropriate LLM backend.

    Args:
        config: Semantic LLM configuration.

    Returns:
        LLM backend instance matching the configured backend type.

    Raises:
        ValueError: If backend is not supported or unavailable.
    """
    if config.backend == "local":
        if TRANSFORMERS_AVAILABLE:
            return LocalLLMBackend(config)
        logger.warning("transformers unavailable; using dummy backend")
        return DummyLocalLLMBackend()

    elif config.backend == "openai":
        if not OPENAI_AVAILABLE:
            raise ValueError("OpenAI backend requested but openai package not installed")
        return OpenAILLMBackend(config)

    elif config.backend == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ValueError("Anthropic backend requested but anthropic package not installed")
        return AnthropicLLMBackend(config)

    else:
        raise ValueError(f"Unknown backend: {config.backend}")
```

---

## Prompt Engineering

### System Prompt Template

```python
SYSTEM_PROMPT = """You are a semantic annotation assistant for conversation transcripts.
Your task is to extract structured information from conversation chunks.

OUTPUT FORMAT:
Always respond with valid JSON matching this schema:
{
  "topics": [{"label": string, "confidence": float, "evidence": string}],
  "risks": [{"type": string, "severity": string, "confidence": float, "evidence": string}],
  "actions": [{"description": string, "assignee": string|null, "due": string|null, "priority": string|null}]
}

RISK TYPES: escalation, churn_risk, compliance, sentiment_negative, legal, pricing_objection, competitor_mention, customer_frustration, agent_error, other
SEVERITY LEVELS: low, medium, high, critical
PRIORITY LEVELS: low, medium, high

GUIDELINES:
1. Extract only information explicitly present in the text
2. Use confidence scores: 0.9+ for explicit mentions, 0.7-0.9 for strong inference, 0.5-0.7 for weak inference
3. Provide brief, specific evidence quotes (max 50 chars)
4. For actions, only include clear commitments (not suggestions or wishes)
5. If no items found for a category, return empty array []
"""
```

### Topic Extraction Prompt

```python
TOPIC_EXTRACTION_PROMPT = """Extract topics from this conversation chunk.

CONTEXT (previous chunks):
{context}

CURRENT CHUNK:
Speaker: {speaker_id}
Timestamp: {start}s - {end}s
Text: {text}

{taxonomy_hint}

Extract 1-3 main topics discussed. Focus on:
- Business topics (pricing, billing, features, support)
- Technical topics (bugs, integrations, setup)
- Relationship topics (satisfaction, complaints, requests)

Respond with JSON:
{{"topics": [
  {{"label": "topic_name", "confidence": 0.0-1.0, "evidence": "brief quote"}}
]}}
"""

def build_topic_prompt(
    chunk: Chunk,
    context: str,
    taxonomy: list[str] | None = None
) -> str:
    """Build prompt for topic extraction."""
    taxonomy_hint = ""
    if taxonomy:
        taxonomy_hint = f"Classify into these categories: {', '.join(taxonomy)}"

    return TOPIC_EXTRACTION_PROMPT.format(
        context=context or "No prior context",
        speaker_id=chunk.speaker_ids[0] if chunk.speaker_ids else "unknown",
        start=chunk.start,
        end=chunk.end,
        text=chunk.text,
        taxonomy_hint=taxonomy_hint,
    )
```

### Risk Detection Prompt

```python
RISK_DETECTION_PROMPT = """Detect risk signals in this conversation chunk.

CONTEXT (previous chunks):
{context}

CURRENT CHUNK:
Speaker: {speaker_id}
Timestamp: {start}s - {end}s
Text: {text}

Identify any of these risk signals:
- escalation: Customer requests manager/supervisor, uses threatening language
- churn_risk: Customer mentions leaving, canceling, competitor products
- compliance: Legal threats, regulatory mentions, recording concerns
- sentiment_negative: Strong negative emotion, frustration, anger
- pricing_objection: Budget concerns, price complaints, discount demands
- competitor_mention: References to competing products/services
- customer_frustration: Repeated issues, long wait times, unresolved problems
- agent_error: Mistakes, miscommunication, incorrect information

For each risk found, assess severity:
- critical: Immediate action required (legal threat, explicit churn intent)
- high: Urgent attention needed (escalation request, strong frustration)
- medium: Monitor closely (pricing concerns, mild frustration)
- low: Note for context (competitor mention, minor concerns)

Respond with JSON:
{{"risks": [
  {{"type": "risk_type", "severity": "level", "confidence": 0.0-1.0, "evidence": "brief quote", "recommended_action": "suggestion"}}
]}}
"""
```

### Action Item Extraction Prompt

```python
ACTION_EXTRACTION_PROMPT = """Extract action items from this conversation chunk.

CONTEXT (previous chunks):
{context}

CURRENT CHUNK:
Speaker: {speaker_id}
Timestamp: {start}s - {end}s
Text: {text}

Identify explicit commitments and tasks:
- Look for phrases like "I will", "I'll", "Let me", "We'll send", "I can"
- Note who made the commitment (speaker ID or role: agent/customer)
- Extract deadlines if mentioned (today, tomorrow, end of week, specific dates)
- Assess priority based on urgency signals

Only extract clear, actionable commitments. Do NOT include:
- Vague intentions ("maybe", "might", "could")
- Questions ("should I?", "do you want me to?")
- Past actions already completed

Respond with JSON:
{{"actions": [
  {{"description": "clear action description", "assignee": "speaker_id or null", "due": "deadline or null", "priority": "low|medium|high|null", "verbatim": "exact quote"}}
]}}
"""
```

### Combined Extraction Prompt

For efficiency, a single prompt can extract all annotation types:

```python
COMBINED_EXTRACTION_PROMPT = """Analyze this conversation chunk and extract structured annotations.

CONTEXT (previous chunks):
{context}

CURRENT CHUNK:
Speaker: {speaker_id}
Timestamp: {start}s - {end}s
Text: {text}

Extract the following:

1. TOPICS: Main subjects discussed (1-3 topics max)
2. RISKS: Any concerning signals (escalation, churn, compliance, frustration)
3. ACTIONS: Explicit commitments and tasks

Respond with JSON:
{{
  "topics": [{{"label": "topic", "confidence": 0.0-1.0, "evidence": "quote"}}],
  "risks": [{{"type": "type", "severity": "level", "confidence": 0.0-1.0, "evidence": "quote"}}],
  "actions": [{{"description": "task", "assignee": "speaker|null", "due": "deadline|null", "priority": "level|null"}}]
}}

If no items found for a category, return empty array [].
"""
```

---

## Safety and Guardrails

### Input Sanitization

```python
import re
from typing import Tuple

# PII patterns (simplified - production should use dedicated library)
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


def detect_pii(text: str) -> list[dict]:
    """
    Detect potential PII in text.

    Args:
        text: Text to scan for PII.

    Returns:
        List of detected PII with type and position.
    """
    detections = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            detections.append({
                "type": pii_type,
                "start": match.start(),
                "end": match.end(),
                "value": match.group(),
            })
    return detections


def redact_pii(text: str) -> Tuple[str, list[dict]]:
    """
    Redact PII from text.

    Args:
        text: Text to redact.

    Returns:
        Tuple of (redacted_text, list of redactions).
    """
    detections = detect_pii(text)

    if not detections:
        return text, []

    # Sort by position (reverse) to preserve indices during replacement
    detections.sort(key=lambda x: x["start"], reverse=True)

    redacted = text
    for detection in detections:
        placeholder = f"[REDACTED_{detection['type'].upper()}]"
        redacted = redacted[:detection["start"]] + placeholder + redacted[detection["end"]:]

    return redacted, detections


def sanitize_for_llm(
    text: str,
    config: SemanticLLMConfig,
) -> Tuple[str, bool]:
    """
    Sanitize text before sending to LLM.

    Args:
        text: Text to sanitize.
        config: Semantic LLM configuration.

    Returns:
        Tuple of (sanitized_text, should_skip).
        If should_skip is True, the chunk should be skipped entirely.
    """
    if not config.pii_detection_enabled:
        return text, False

    detections = detect_pii(text)

    if not detections:
        return text, False

    if config.pii_action == "skip":
        logger.warning(f"Skipping chunk due to PII: {[d['type'] for d in detections]}")
        return "", True

    elif config.pii_action == "redact":
        redacted, _ = redact_pii(text)
        logger.debug(f"Redacted {len(detections)} PII items from chunk")
        return redacted, False

    else:  # warn
        logger.warning(f"PII detected but proceeding: {[d['type'] for d in detections]}")
        return text, False
```

### Output Validation

```python
import json
from jsonschema import validate, ValidationError

# Minimal schema for validation
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label"],
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                }
            }
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "severity"],
                "properties": {
                    "type": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                }
            }
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description"],
                "properties": {
                    "description": {"type": "string"},
                }
            }
        },
    },
}


def validate_llm_output(response: str) -> Tuple[dict | None, str | None]:
    """
    Validate and parse LLM JSON output.

    Args:
        response: Raw LLM response string.

    Returns:
        Tuple of (parsed_dict, error_message).
        If parsing fails, parsed_dict is None and error_message explains why.
    """
    # Try to extract JSON from response (handle markdown code blocks)
    json_str = response.strip()
    if json_str.startswith("```"):
        # Extract from code block
        lines = json_str.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        json_str = "\n".join(json_lines)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    try:
        validate(instance=parsed, schema=OUTPUT_SCHEMA)
    except ValidationError as e:
        return None, f"Schema validation failed: {e.message}"

    # Additional sanity checks
    for topic in parsed.get("topics", []):
        if topic.get("confidence", 0) > 1.0 or topic.get("confidence", 0) < 0.0:
            topic["confidence"] = max(0.0, min(1.0, topic.get("confidence", 0.5)))

    for risk in parsed.get("risks", []):
        if risk.get("severity") not in ["low", "medium", "high", "critical"]:
            risk["severity"] = "medium"

    return parsed, None
```

### Cost Controls

```python
class CostTracker:
    """Track and limit LLM API costs."""

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.total_cost_usd = 0.0
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add_request(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        """Record a request's cost."""
        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_cost_usd += cost_usd

    def check_limit(self, estimated_cost: float) -> bool:
        """Check if estimated cost would exceed limit."""
        return self.total_cost_usd + estimated_cost <= self.limit_usd

    def get_summary(self) -> dict:
        """Get cost summary."""
        return {
            "total_cost_usd": self.total_cost_usd,
            "limit_usd": self.limit_usd,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "remaining_budget_usd": self.limit_usd - self.total_cost_usd,
        }
```

---

## Migration Guide

### Migrating from v1.x Keyword-Based Annotator

The v2.0.0 LLM-backed annotator is designed as an **opt-in enhancement**, not a replacement. Existing v1.x workflows continue to work unchanged.

#### Comparison

| Feature | v1.x KeywordSemanticAnnotator | v2.0 LLMSemanticAnnotator |
|---------|------------------------------|---------------------------|
| Dependencies | None (stdlib only) | transformers/openai/anthropic |
| Latency | <1ms per segment | 100ms-2s per chunk (varies by backend) |
| Quality | Pattern-matching only | Contextual understanding |
| Cost | Free | Free (local) or API costs (cloud) |
| Topics | Predefined keywords | Dynamic extraction |
| Risks | Keyword triggers | Contextual severity assessment |
| Actions | Pattern-based | Semantic extraction with metadata |
| Confidence | Binary (matched/not) | Continuous (0.0-1.0) |

#### Migration Steps

**Step 1: Enable LLM annotator alongside keyword annotator**

```python
from transcription import EnrichmentConfig, SemanticLLMConfig

# Default: keyword annotator only (v1.x behavior)
config_v1 = EnrichmentConfig(
    enable_semantic_annotator=True,  # Uses KeywordSemanticAnnotator
)

# v2.0: Add LLM annotator with merging
llm_config = SemanticLLMConfig(
    backend="local",
    model="qwen2.5-7b",
    merge_with_keyword_annotator=True,  # Combine both
)

config_v2 = EnrichmentConfig(
    enable_semantic_annotator=True,
    semantic_llm_config=llm_config,
)
```

**Step 2: Update consumers to handle new schema fields**

```python
def process_semantic_annotations(transcript):
    """Process annotations with v1/v2 compatibility."""
    semantic = transcript.annotations.get("semantic", {})

    # v1.x fields (always present)
    keywords = semantic.get("keywords", [])
    risk_tags = semantic.get("risk_tags", [])
    actions = semantic.get("actions", [])  # v1.x: pattern-matched

    # v2.0 fields (present when annotator="llm")
    version = semantic.get("version", "1.0.0")

    if version.startswith("2."):
        # Use richer v2.0 structures
        topics = semantic.get("topics", [])  # With confidence scores
        risks = semantic.get("risks", [])    # With severity and evidence
        actions_v2 = semantic.get("actions", [])  # With assignees and deadlines

        for topic in topics:
            if topic.get("confidence", 0) > 0.7:
                process_high_confidence_topic(topic)

        for risk in risks:
            if risk.get("severity") in ["high", "critical"]:
                alert_risk(risk)

        for action in actions_v2:
            if action.get("due"):
                schedule_followup(action)
    else:
        # Fall back to v1.x processing
        for tag in risk_tags:
            if tag == "escalation":
                alert_escalation(keywords)
```

**Step 3: Gradual rollout with A/B testing**

```python
import random

def get_enrichment_config(user_id: str) -> EnrichmentConfig:
    """A/B test LLM annotator rollout."""
    # 10% of users get LLM annotator
    use_llm = random.random() < 0.10

    if use_llm:
        return EnrichmentConfig(
            enable_semantic_annotator=True,
            semantic_llm_config=SemanticLLMConfig(
                backend="local",
                model="qwen2.5-3b",  # Start with smaller model
                enable_summary=False,  # Minimize latency
            ),
        )
    else:
        return EnrichmentConfig(enable_semantic_annotator=True)
```

#### Backward Compatibility Guarantees

1. **v1.x fields preserved**: `keywords`, `risk_tags`, `actions` (pattern-based) always present
2. **Version field**: Check `semantic.version` to detect v2.0 annotations
3. **Fallback behavior**: If LLM fails, falls back to keyword annotator
4. **Schema additive**: New fields are optional; v1.x consumers ignore them

---

## Performance Considerations

### Latency Breakdown

| Component | Local (7B, 4-bit) | OpenAI (gpt-4o-mini) | Anthropic (Haiku) |
|-----------|-------------------|----------------------|-------------------|
| Model load (first call) | 5-15s | N/A | N/A |
| Per-chunk inference | 0.5-2s | 0.3-0.8s | 0.3-0.8s |
| Network overhead | N/A | 50-200ms | 50-200ms |
| Output parsing | <10ms | <10ms | <10ms |

### Batching Strategy

```python
class BatchingAnnotator:
    """
    Batch multiple chunks for efficient LLM inference.

    Groups chunks to maximize throughput while respecting token limits.
    """

    def __init__(self, config: SemanticLLMConfig, backend: LLMBackendLike) -> None:
        self.config = config
        self.backend = backend
        self.max_batch_tokens = config.max_tokens_per_chunk * 3  # ~3 chunks per batch

    def annotate_batch(self, chunks: list[Chunk]) -> list[dict]:
        """
        Annotate multiple chunks in a single LLM call.

        Args:
            chunks: List of chunks to annotate.

        Returns:
            List of annotation dicts, one per chunk.
        """
        # Build batched prompt
        prompt_parts = ["Annotate each of the following conversation chunks:\n\n"]

        for i, chunk in enumerate(chunks):
            prompt_parts.append(f"--- CHUNK {i+1} ---")
            prompt_parts.append(f"Speaker: {chunk.speaker_ids[0] if chunk.speaker_ids else 'unknown'}")
            prompt_parts.append(f"Time: {chunk.start:.1f}s - {chunk.end:.1f}s")
            prompt_parts.append(f"Text: {chunk.text}\n")

        prompt_parts.append(
            "Respond with a JSON array of annotations, one object per chunk:\n"
            '[{"topics": [...], "risks": [...], "actions": [...]}, ...]'
        )

        prompt = "\n".join(prompt_parts)

        # Single LLM call for batch
        response = self.backend.generate(prompt, max_tokens=self.config.max_output_tokens * len(chunks))

        # Parse batch response
        try:
            results = json.loads(response)
            if isinstance(results, list) and len(results) == len(chunks):
                return results
        except json.JSONDecodeError:
            pass

        # Fallback: annotate individually
        return [self._annotate_single(chunk) for chunk in chunks]

    def _annotate_single(self, chunk: Chunk) -> dict:
        """Annotate a single chunk."""
        prompt = build_combined_prompt(chunk, context=None)
        response = self.backend.generate(prompt, max_tokens=self.config.max_output_tokens)
        parsed, _ = validate_llm_output(response)
        return parsed or {"topics": [], "risks": [], "actions": []}
```

### Caching Strategy

```python
import hashlib
from functools import lru_cache

def chunk_cache_key(chunk: Chunk, config: SemanticLLMConfig) -> str:
    """Generate cache key for a chunk annotation."""
    content = f"{chunk.text}|{config.model}|{config.enable_topics}|{config.enable_risks}|{config.enable_actions}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class CachedAnnotator:
    """Annotator with response caching."""

    def __init__(self, config: SemanticLLMConfig, backend: LLMBackendLike) -> None:
        self.config = config
        self.backend = backend
        self._cache: dict[str, dict] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def annotate_chunk(self, chunk: Chunk, context: str | None = None) -> dict:
        """Annotate chunk with caching."""
        if not self.config.cache_enabled:
            return self._annotate_uncached(chunk, context)

        cache_key = chunk_cache_key(chunk, self.config)

        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1
        result = self._annotate_uncached(chunk, context)
        self._cache[cache_key] = result
        return result

    def _annotate_uncached(self, chunk: Chunk, context: str | None) -> dict:
        """Perform actual annotation without cache."""
        prompt = build_combined_prompt(chunk, context)
        response = self.backend.generate(prompt, max_tokens=self.config.max_output_tokens)
        parsed, error = validate_llm_output(response)

        if error:
            logger.warning(f"LLM output validation failed: {error}")
            return {"topics": [], "risks": [], "actions": []}

        return parsed

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
        }
```

### Memory Management for Local Models

```python
import gc

class LocalModelManager:
    """Manage local model lifecycle for memory efficiency."""

    _instance = None
    _model = None
    _tokenizer = None
    _model_id = None

    @classmethod
    def get_instance(cls) -> "LocalModelManager":
        """Singleton access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_id: str, config: SemanticLLMConfig) -> None:
        """Load model, unloading previous if different."""
        if self._model_id == model_id and self._model is not None:
            return  # Already loaded

        self.unload_model()
        # ... load new model ...
        self._model_id = model_id

    def unload_model(self) -> None:
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._model_id = None

            # Force garbage collection
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
```

### Recommended Configuration by Use Case

| Use Case | Backend | Model | Batch Size | Notes |
|----------|---------|-------|------------|-------|
| Development/Testing | local | smollm-1.7b | 1 | Fast iteration, CPU-friendly |
| Batch Processing (GPU) | local | qwen2.5-7b | 3-5 | Best quality, ~4GB VRAM |
| Batch Processing (CPU) | local | qwen2.5-3b | 1-2 | Slower but functional |
| Real-time (Quality) | openai | gpt-4o-mini | 1 | Low latency, good quality |
| Real-time (Budget) | anthropic | claude-3-haiku | 1 | Lowest cost cloud option |
| High-Stakes | openai | gpt-4o | 1 | Maximum quality |

---

## Testing Cloud Providers

The cloud provider implementations have comprehensive test coverage with mocked API responses.

### Running Tests

```bash
# Run all cloud provider tests (no API keys required - uses mocks)
uv run python -m pytest tests/test_semantic_cloud_providers.py -v

# Run with real API keys for integration tests
OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \
uv run python -m pytest tests/test_semantic_cloud_providers.py -v -m external
```

### Test Coverage

The test suite (`tests/test_semantic_cloud_providers.py`) covers:

| Category | Tests |
|----------|-------|
| **Factory Function** | Provider creation, kwarg passing |
| **OpenAI Adapter** | Success, timeout, rate limit, malformed JSON, health checks |
| **Anthropic Adapter** | Success, timeout, rate limit, health checks |
| **Retry Logic** | Error classification, backoff calculation, retry decisions |
| **Schema Validation** | Annotation structure, serialization roundtrip |
| **Prompt Building** | Context inclusion, empty context handling |
| **Provider Comparison** | Consistent behavior across providers |

### Mocking Strategy

Tests use `unittest.mock` to mock the `_guarded_provider.complete` method, allowing full
testing without actual API calls:

```python
from unittest.mock import MagicMock

adapter = OpenAISemanticAdapter(api_key="test-key")

mock_response = MagicMock()
mock_response.text = '{"topics": ["test"], "intent": null, ...}'
mock_response.duration_ms = 100

async def mock_complete(system: str, user: str):
    return mock_response

adapter._guarded_provider.complete = mock_complete
```

---

## Related Documentation

- [ROADMAP.md](../ROADMAP.md) - v2.0.0 planning and timeline
- [LLM_PROMPT_PATTERNS.md](LLM_PROMPT_PATTERNS.md) - Prompt patterns for downstream LLM use
- [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) - Real-time annotation via `LiveSemanticSession`
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration management guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-28 | Local LLM provider implemented (#89) with LocalSemanticConfig, LocalSemanticProvider |
| 2.0.0 | 2026-01-28 | Cloud providers implemented (OpenAI, Anthropic) with tests |
| 2.0.0-draft | 2025-12-31 | Initial design document |

---

## Feedback

We welcome feedback via:

- GitHub Issues: Tag with `semantic-llm` label
- GitHub Discussions: v2.0.0 planning thread
- Pull Requests: Improvements to this design document
