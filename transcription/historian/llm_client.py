"""Repository-only LLM provider facade for historian tooling.

Generic provider primitives live in :mod:`transcription.llm_provider` so the
installed semantic surface never depends on the excluded historian package.
This module keeps the Claude Agent SDK provider and preserves historian imports.
"""

from __future__ import annotations

import time

from transcription.llm_provider import (
    AnthropicProvider,
    LLMConfig,
    LLMProvider,
    LLMProviderConfigError,
    LLMProviderDependencyError,
    LLMResponse,
    LocalLLMProvider,
    MockProvider,
    OpenAIProvider,
)
from transcription.llm_provider import create_llm_provider as create_public_llm_provider


class ClaudeCodeProvider(LLMProvider):
    """Repository-only provider using Claude Code via the Agent SDK."""

    async def complete(self, system: str, user: str) -> LLMResponse:
        """Send completion via Claude Agent SDK."""
        start_time = time.time()

        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                TextBlock,
                query,
            )
        except ImportError as exc:
            raise ImportError(
                "claude-agent-sdk not installed. Install with: pip install claude-agent-sdk"
            ) from exc

        options = ClaudeAgentOptions(
            system_prompt=system,
            allowed_tools=[],
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


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Create a historian-capable provider while delegating public providers."""
    if config.provider == "claude-code":
        return ClaudeCodeProvider(config)
    return create_public_llm_provider(config)


async def llm_complete(
    system: str,
    user: str,
    provider: str = "claude-code",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Convenience wrapper for historian one-off completions."""
    config = LLMConfig(provider=provider, model=model, api_key=api_key)
    llm = create_llm_provider(config)
    response = await llm.complete(system, user)
    return response.text


__all__ = [
    "AnthropicProvider",
    "ClaudeCodeProvider",
    "LLMConfig",
    "LLMProvider",
    "LLMProviderConfigError",
    "LLMProviderDependencyError",
    "LLMResponse",
    "LocalLLMProvider",
    "MockProvider",
    "OpenAIProvider",
    "create_llm_provider",
    "llm_complete",
]
