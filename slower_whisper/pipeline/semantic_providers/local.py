"""Local LLM semantic provider implementation (#89).

This module provides the local LLM backend for semantic annotations,
supporting offline and low-latency use cases without cloud dependencies.

Supported Models:
- Qwen/Qwen2.5-7B-Instruct (recommended, ~4GB VRAM with 4-bit quantization)
- Qwen/Qwen2.5-3B-Instruct (faster, ~2GB VRAM)
- HuggingFaceTB/SmolLM2-1.7B-Instruct (very fast, ~1GB VRAM)
- microsoft/phi-3-mini-4k-instruct (efficient, good quality)

Features:
- Lazy model loading (only loads on first use)
- Optional dependency handling (torch/transformers not required at import)
- Configurable device (auto, cuda, cpu)
- Configurable batch processing
- Health check with availability status
- Graceful error handling with empty annotation fallback

Example:
    >>> from transcription.semantic_providers.local import (
    ...     LocalSemanticProvider,
    ...     LocalSemanticConfig,
    ...     is_available,
    ... )
    >>> if is_available():
    ...     config = LocalSemanticConfig(
    ...         model_name="Qwen/Qwen2.5-3B-Instruct",
    ...         device="auto",
    ...     )
    ...     provider = LocalSemanticProvider(config)
    ...     health = provider.health_check()
    ...     print(f"Available: {health.available}")

Resource Requirements:
    | Model              | VRAM (4-bit) | RAM (CPU) | Tokens/sec (GPU) |
    |--------------------|--------------|-----------|------------------|
    | Qwen2.5-7B         | ~4GB         | ~8GB      | ~30-50           |
    | Qwen2.5-3B         | ~2GB         | ~4GB      | ~50-80           |
    | SmolLM2-1.7B       | ~1GB         | ~2GB      | ~80-120          |
    | Phi-3-mini (4k)    | ~2GB         | ~4GB      | ~60-100          |

See Also:
    - docs/LLM_SEMANTIC_ANNOTATOR.md for full documentation
    - transcription.local_llm_provider for underlying LLM provider
    - transcription.semantic_adapter for adapter protocol
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

# Re-export from local_llm_provider for convenience
from ..local_llm_provider import (
    LocalLLMProvider,
    LocalLLMResponse,
    MockLocalLLMProvider,
    get_availability_status,
    is_available,
)
from ..semantic_adapter import (
    SEMANTIC_SCHEMA_VERSION,
    ChunkContext,
    LocalLLMSemanticAdapter,
    NormalizedAnnotation,
    ProviderHealth,
    SemanticAnnotation,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalSemanticConfig:
    """Configuration for local LLM semantic provider.

    This configuration controls the behavior of the local semantic provider,
    including model selection, device placement, and extraction settings.

    Attributes:
        model_name: Model identifier from HuggingFace Hub.
            Defaults to "Qwen/Qwen2.5-7B-Instruct".
            Recommended models:
            - "Qwen/Qwen2.5-7B-Instruct" (best quality)
            - "Qwen/Qwen2.5-3B-Instruct" (faster)
            - "HuggingFaceTB/SmolLM2-1.7B-Instruct" (fastest)

        device: Device for inference.
            - "auto": Use CUDA if available, else CPU
            - "cuda": Force GPU (requires CUDA)
            - "cpu": Force CPU

        temperature: Sampling temperature (0.0 = deterministic).
            Lower values produce more consistent outputs.
            Default: 0.1

        max_tokens: Maximum tokens in LLM response.
            Default: 1024

        batch_size: Number of chunks to process in parallel.
            Higher values use more memory but are faster.
            Default: 1

        extraction_mode: What to extract from text.
            - "combined": Topics, risks, and actions (default)
            - "topics": Only topic labels
            - "risks": Only risk signals
            - "actions": Only action items

        context_window: Number of previous chunks to include for context.
            Default: 3

        enable_caching: Whether to cache LLM responses.
            Default: True

    Example:
        >>> config = LocalSemanticConfig(
        ...     model_name="Qwen/Qwen2.5-3B-Instruct",
        ...     device="cuda",
        ...     temperature=0.0,  # Deterministic
        ...     extraction_mode="combined",
        ... )
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    temperature: float = 0.1
    max_tokens: int = 1024
    batch_size: int = 1
    extraction_mode: Literal["combined", "topics", "risks", "actions"] = "combined"
    context_window: int = 3
    enable_caching: bool = True

    # Internal tracking
    _source_fields: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.context_window < 0:
            raise ValueError(f"context_window must be >= 0, got {self.context_window}")

    @classmethod
    def from_env(cls, prefix: str = "SLOWER_WHISPER_SEMANTIC_") -> LocalSemanticConfig:
        """Load configuration from environment variables.

        Environment variable mapping:
        - {prefix}MODEL_NAME -> model_name
        - {prefix}DEVICE -> device
        - {prefix}TEMPERATURE -> temperature
        - {prefix}MAX_TOKENS -> max_tokens
        - {prefix}BATCH_SIZE -> batch_size
        - {prefix}EXTRACTION_MODE -> extraction_mode

        Args:
            prefix: Environment variable prefix.

        Returns:
            LocalSemanticConfig with values from environment.
        """
        import os

        config_dict: dict[str, Any] = {}

        # String fields
        if model := os.getenv(f"{prefix}MODEL_NAME"):
            config_dict["model_name"] = model

        if device := os.getenv(f"{prefix}DEVICE"):
            if device not in ("auto", "cuda", "cpu"):
                raise ValueError(f"Invalid {prefix}DEVICE: {device}")
            config_dict["device"] = device

        if mode := os.getenv(f"{prefix}EXTRACTION_MODE"):
            if mode not in ("combined", "topics", "risks", "actions"):
                raise ValueError(f"Invalid {prefix}EXTRACTION_MODE: {mode}")
            config_dict["extraction_mode"] = mode

        # Numeric fields
        if temp := os.getenv(f"{prefix}TEMPERATURE"):
            config_dict["temperature"] = float(temp)

        if tokens := os.getenv(f"{prefix}MAX_TOKENS"):
            config_dict["max_tokens"] = int(tokens)

        if batch := os.getenv(f"{prefix}BATCH_SIZE"):
            config_dict["batch_size"] = int(batch)

        if context := os.getenv(f"{prefix}CONTEXT_WINDOW"):
            config_dict["context_window"] = int(context)

        # Boolean fields
        if caching := os.getenv(f"{prefix}ENABLE_CACHING"):
            config_dict["enable_caching"] = caching.lower() in ("true", "1", "yes")

        config = cls(**config_dict)
        config._source_fields = set(config_dict.keys())
        return config


class LocalSemanticProvider:
    """Local LLM semantic provider implementing SemanticAdapter protocol.

    This provider uses local models (via transformers) for semantic annotation,
    enabling offline operation and low-latency inference without cloud APIs.

    The provider wraps LocalLLMSemanticAdapter with additional configuration
    and batch processing support.

    Attributes:
        config: LocalSemanticConfig instance.
        adapter: Underlying LocalLLMSemanticAdapter.

    Example:
        >>> provider = LocalSemanticProvider()
        >>> health = provider.health_check()
        >>> if health.available:
        ...     context = ChunkContext(speaker_id="agent", start=0.0, end=30.0)
        ...     result = provider.annotate_chunk(
        ...         "I'll send you the report tomorrow.",
        ...         context,
        ...     )
        ...     print(result.normalized.action_items)
    """

    def __init__(
        self,
        config: LocalSemanticConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the local semantic provider.

        Args:
            config: LocalSemanticConfig instance. If None, uses defaults.
            **kwargs: Override config fields (model_name, device, etc.).
        """
        if config is None:
            config = LocalSemanticConfig(**kwargs)
        elif kwargs:
            # Merge kwargs into config
            config_dict: dict[str, Any] = {
                "model_name": config.model_name,
                "device": config.device,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "batch_size": config.batch_size,
                "extraction_mode": config.extraction_mode,
                "context_window": config.context_window,
                "enable_caching": config.enable_caching,
            }
            config_dict.update(kwargs)
            config = LocalSemanticConfig(**config_dict)

        self.config = config
        self._adapter: LocalLLMSemanticAdapter | None = None
        self._cache: dict[str, SemanticAnnotation] = {}

    def _get_adapter(self) -> LocalLLMSemanticAdapter:
        """Lazy-create the underlying adapter."""
        if self._adapter is None:
            self._adapter = LocalLLMSemanticAdapter(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                extraction_mode=self.config.extraction_mode,
            )
        return self._adapter

    def _cache_key(self, text: str, context: ChunkContext) -> str:
        """Generate cache key for annotation."""
        import hashlib

        content = f"{text}|{context.speaker_id}|{context.start}|{context.end}|{self.config.extraction_mode}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def annotate_chunk(self, text: str, context: ChunkContext) -> SemanticAnnotation:
        """Annotate a single chunk of text.

        Args:
            text: Text to annotate (typically 60-120 seconds of conversation).
            context: Chunk context with speaker, timing, and history.

        Returns:
            SemanticAnnotation with topics, risks, and action items.
        """
        # Check cache
        if self.config.enable_caching:
            key = self._cache_key(text, context)
            if key in self._cache:
                return self._cache[key]

        # Get annotation
        adapter = self._get_adapter()
        result = adapter.annotate_chunk(text, context)

        # Cache result
        if self.config.enable_caching:
            self._cache[key] = result

        return result

    def annotate_batch(
        self,
        chunks: list[tuple[str, ChunkContext]],
    ) -> list[SemanticAnnotation]:
        """Annotate multiple chunks.

        Currently processes sequentially. Future versions may support
        true batch inference for improved throughput.

        Args:
            chunks: List of (text, context) tuples.

        Returns:
            List of SemanticAnnotation results.
        """
        return [self.annotate_chunk(text, ctx) for text, ctx in chunks]

    def health_check(self) -> ProviderHealth:
        """Check provider availability.

        Returns:
            ProviderHealth with availability status and any errors.
        """
        adapter = self._get_adapter()
        return adapter.health_check()

    def clear_cache(self) -> int:
        """Clear the annotation cache.

        Returns:
            Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache size and status.
        """
        return {
            "size": len(self._cache),
            "enabled": self.config.enable_caching,
        }

    @property
    def model_name(self) -> str:
        """Get configured model name."""
        return self.config.model_name

    @property
    def device(self) -> str:
        """Get configured device."""
        return self.config.device


# Factory function for convenience
def create_local_provider(
    model_name: str | None = None,
    device: str = "auto",
    **kwargs: Any,
) -> LocalSemanticProvider:
    """Create a local semantic provider with the given configuration.

    Args:
        model_name: Model identifier (default: Qwen/Qwen2.5-7B-Instruct).
        device: Device for inference ("auto", "cuda", "cpu").
        **kwargs: Additional config options.

    Returns:
        LocalSemanticProvider instance.

    Raises:
        ImportError: If torch/transformers not installed (on first use).

    Example:
        >>> provider = create_local_provider(
        ...     model_name="Qwen/Qwen2.5-3B-Instruct",
        ...     device="cuda",
        ... )
    """
    config_kwargs: dict[str, Any] = {"device": device}
    if model_name:
        config_kwargs["model_name"] = model_name
    config_kwargs.update(kwargs)
    return LocalSemanticProvider(LocalSemanticConfig(**config_kwargs))


__all__ = [
    # Config
    "LocalSemanticConfig",
    # Provider
    "LocalSemanticProvider",
    "create_local_provider",
    # Re-exports
    "LocalLLMProvider",
    "LocalLLMResponse",
    "MockLocalLLMProvider",
    "LocalLLMSemanticAdapter",
    # Availability
    "is_available",
    "get_availability_status",
    # Data classes
    "ChunkContext",
    "SemanticAnnotation",
    "NormalizedAnnotation",
    "ProviderHealth",
    "SEMANTIC_SCHEMA_VERSION",
]
