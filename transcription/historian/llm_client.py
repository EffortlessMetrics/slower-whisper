"""
LLM provider abstraction for subagent analyzers.

This module provides a unified interface for LLM completions,
with the initial implementation using Claude Code via the Agent SDK.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str  # "claude-code", "anthropic", "openai", "local"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class LLMResponse:
    """Response from LLM completion."""

    text: str
    tokens_used: int | None = None
    duration_ms: int = 0
    raw_response: Any = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def complete(self, system: str, user: str) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            system: System prompt
            user: User prompt

        Returns:
            LLMResponse with the completion text
        """
        ...

    def complete_sync(self, system: str, user: str) -> LLMResponse:
        """Synchronous wrapper for complete()."""
        return asyncio.run(self.complete(system, user))


class ClaudeCodeProvider(LLMProvider):
    """
    LLM provider using Claude Code via the Agent SDK.

    This runs Claude as a subprocess and streams the response.
    """

    async def complete(self, system: str, user: str) -> LLMResponse:
        """Send completion via Claude Agent SDK."""
        start_time = time.time()

        try:
            # Import here to avoid dependency at module load
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                TextBlock,
                query,
            )
        except ImportError as e:
            raise ImportError(
                "claude-agent-sdk not installed. Install with: pip install claude-agent-sdk"
            ) from e

        options = ClaudeAgentOptions(
            system_prompt=system,
            allowed_tools=[],  # No tools - pure analysis
            permission_mode="bypassPermissions",
        )

        if self.config.model:
            options.model = self.config.model

        response_text = ""
        tokens_used = None
        raw_response = None

        async for message in query(prompt=user, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
            elif isinstance(message, ResultMessage):
                raw_response = message
                if hasattr(message, "usage") and message.usage:
                    tokens_used = message.usage.get("total_tokens")

        duration_ms = int((time.time() - start_time) * 1000)

        return LLMResponse(
            text=response_text,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            raw_response=raw_response,
        )


class AnthropicProvider(LLMProvider):
    """
    LLM provider using the official Anthropic Python client.
    """

    async def complete(self, system: str, user: str) -> LLMResponse:
        """Send completion via Anthropic API."""
        start_time = time.time()

        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            ) from e

        # Resolve API key: config > env > error
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY env var or pass in config."
            )

        client = anthropic.AsyncAnthropic(api_key=api_key)
        model = self.config.model or "claude-3-5-sonnet-20241022"

        try:
            message = await client.messages.create(
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system,
                messages=[
                    {"role": "user", "content": user}
                ],
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}") from e

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract text content
        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        return LLMResponse(
            text=response_text,
            tokens_used=message.usage.input_tokens + message.usage.output_tokens if message.usage else None,
            duration_ms=duration_ms,
            raw_response=message,
        )


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(self, config: LLMConfig, responses: dict[str, str] | None = None):
        super().__init__(config)
        self.responses = responses or {}
        self.call_count = 0
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> LLMResponse:
        """Return mock response."""
        self.call_count += 1
        self.calls.append((system, user))

        # Look for matching response
        for key, response in self.responses.items():
            if key in system or key in user:
                return LLMResponse(text=response, duration_ms=100)

        # Default response
        return LLMResponse(
            text=json.dumps({"error": "No mock response configured"}),
            duration_ms=100,
        )


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        config: LLM configuration

    Returns:
        Configured LLM provider instance
    """
    if config.provider == "claude-code":
        return ClaudeCodeProvider(config)
    elif config.provider == "anthropic":
        return AnthropicProvider(config)
    elif config.provider == "mock":
        return MockProvider(config)
    else:
        # Fallback to MockProvider with error if unknown, or raise?
        # Raising is better to fail fast
        raise ValueError(f"Unknown LLM provider: {config.provider}")


# Convenience function for one-off completions
async def llm_complete(
    system: str,
    user: str,
    provider: str = "claude-code",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Convenience function for one-off LLM completions.

    Args:
        system: System prompt
        user: User prompt
        provider: Provider name
        model: Optional model override
        api_key: Optional API key override

    Returns:
        Response text
    """
    config = LLMConfig(provider=provider, model=model, api_key=api_key)
    llm = create_llm_provider(config)
    response = await llm.complete(system, user)
    return response.text
