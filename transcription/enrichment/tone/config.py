"""Configuration for tone analysis."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ToneConfig:
    """
    Configuration for tone analysis.

    Attributes:
        api_provider: Which API to use ("anthropic", "openai", "local", "mock")
        model_name: Model identifier (e.g. "claude-sonnet-4.5")
        api_key: API key (if None, will check environment variable)
        batch_size: Number of segments to analyze per API call
        max_retries: Maximum retry attempts for failed API calls
        confidence_threshold: Minimum confidence to assign tone (else "neutral")
        include_context: Whether to include surrounding segments for context
        context_window: Number of segments before/after to include as context
    """
    api_provider: str = "anthropic"
    model_name: str = "claude-sonnet-4-5-20250929"
    api_key: Optional[str] = None
    batch_size: int = 10
    max_retries: int = 3
    confidence_threshold: float = 0.6
    include_context: bool = True
    context_window: int = 1


# Supported tone labels
TONE_LABELS = [
    "neutral",      # Calm, matter-of-fact delivery
    "positive",     # Enthusiastic, happy, excited, encouraging
    "negative",     # Frustrated, angry, disappointed, critical
    "questioning",  # Curious, probing, seeking clarification
    "uncertain",    # Hesitant, doubtful, tentative
    "emphatic",     # Strong emphasis, passionate, assertive
]
