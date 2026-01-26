"""Integration sinks for webhooks and RAG export.

This module provides integration sinks for sending transcription events
to external systems:

- WebhookSink: Send events to HTTP webhook endpoints
- RAGExporter: Export transcripts as RAG-ready bundles
- SinkRegistry: Manage and dispatch to multiple sinks

Key features:
- Async event dispatch (non-blocking)
- Retry with exponential backoff
- Dead letter queue for failed sends
- HMAC signature verification
- Multiple chunking strategies for RAG
"""

from __future__ import annotations

from .events import (
    EventType,
    IntegrationEvent,
    create_error_event,
    create_outcome_event,
    create_segment_event,
    create_session_ended_event,
    create_session_started_event,
    create_transcript_event,
)
from .rag_export import (
    ChunkingStrategy,
    RAGBundle,
    RAGChunk,
    RAGExporter,
    RAGExporterConfig,
    export_transcript_to_rag,
)
from .registry import SinkConfig, SinkRegistry
from .webhooks import (
    AuthConfig,
    DeadLetterEntry,
    RetryPolicy,
    WebhookConfig,
    WebhookSink,
    verify_webhook_signature,
)

__all__ = [
    # Events
    "EventType",
    "IntegrationEvent",
    "create_error_event",
    "create_outcome_event",
    "create_segment_event",
    "create_session_ended_event",
    "create_session_started_event",
    "create_transcript_event",
    # Webhooks
    "AuthConfig",
    "DeadLetterEntry",
    "RetryPolicy",
    "WebhookConfig",
    "WebhookSink",
    "verify_webhook_signature",
    # RAG Export
    "ChunkingStrategy",
    "RAGBundle",
    "RAGChunk",
    "RAGExporter",
    "RAGExporterConfig",
    "export_transcript_to_rag",
    # Registry
    "SinkConfig",
    "SinkRegistry",
]
