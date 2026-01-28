"""Semantic annotation providers package.

This package provides semantic annotation providers for transcripts:

Providers:
- local: LocalLLMSemanticAdapter - Local LLM inference (Qwen, SmolLM)
- local_keyword: LocalKeywordAdapter - Rule-based keyword matching
- openai: OpenAISemanticAdapter - OpenAI API (GPT-4o)
- anthropic: AnthropicSemanticAdapter - Anthropic API (Claude)
- noop: NoOpSemanticAdapter - Placeholder that returns empty annotations

Example:
    >>> from transcription.semantic_providers import create_adapter, ChunkContext
    >>> adapter = create_adapter("local-llm", model="Qwen/Qwen2.5-3B-Instruct")
    >>> context = ChunkContext(speaker_id="agent", start=0.0, end=30.0)
    >>> annotation = adapter.annotate_chunk("I'll send you the report.", context)

See Also:
    - transcription.semantic_adapter for adapter protocol and implementations
    - transcription.local_llm_provider for underlying LLM provider
    - docs/LLM_SEMANTIC_ANNOTATOR.md for full documentation
"""

from __future__ import annotations

# Re-export from local_llm_provider
from ..local_llm_provider import (
    LocalLLMProvider,
    LocalLLMResponse,
    MockLocalLLMProvider,
    get_availability_status,
    is_available,
)

# Re-export from semantic.py (legacy KeywordSemanticAnnotator)
from ..semantic import KeywordSemanticAnnotator, NoOpSemanticAnnotator, SemanticAnnotator

# Re-export from semantic_adapter.py (adapter protocol and implementations)
from ..semantic_adapter import (
    COMBINED_EXTRACTION_PROMPT,
    SEMANTIC_SCHEMA_VERSION,
    ActionItem,
    AnthropicSemanticAdapter,
    ChunkContext,
    CloudLLMSemanticAdapter,
    LocalKeywordAdapter,
    LocalLLMSemanticAdapter,
    NormalizedAnnotation,
    OpenAISemanticAdapter,
    ProviderHealth,
    SemanticAdapter,
    SemanticAnnotation,
    create_adapter,
)
from ..semantic_adapter import (
    NoOpSemanticAdapter as NoOpAdapter,
)

__all__ = [
    # Schema version
    "SEMANTIC_SCHEMA_VERSION",
    # Data classes
    "ActionItem",
    "NormalizedAnnotation",
    "SemanticAnnotation",
    "ChunkContext",
    "ProviderHealth",
    # Protocols
    "SemanticAdapter",
    "SemanticAnnotator",
    # Adapters
    "LocalKeywordAdapter",
    "LocalLLMSemanticAdapter",
    "CloudLLMSemanticAdapter",
    "OpenAISemanticAdapter",
    "AnthropicSemanticAdapter",
    "NoOpSemanticAdapter",
    "NoOpAdapter",
    # LLM Provider
    "LocalLLMProvider",
    "LocalLLMResponse",
    "MockLocalLLMProvider",
    # Legacy
    "KeywordSemanticAnnotator",
    # Availability
    "is_available",
    "get_availability_status",
    # Prompts
    "COMBINED_EXTRACTION_PROMPT",
    # Factory
    "create_adapter",
]
