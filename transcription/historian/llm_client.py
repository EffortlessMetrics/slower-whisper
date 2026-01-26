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

    def __repr__(self) -> str:
        """Secure repr that masks API key."""
        key_repr = "None"
        if self.api_key:
            if len(self.api_key) > 6:
                key_repr = f"'{self.api_key[:3]}...'"
            else:
                key_repr = "'***'"

        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"api_key={key_repr}, base_url={self.base_url!r}, "
            f"temperature={self.temperature!r}, max_tokens={self.max_tokens!r})"
        )


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

    Requires the anthropic package (pip install anthropic) and either:
    - ANTHROPIC_API_KEY environment variable, or
    - api_key in LLMConfig
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
        model = self.config.model or "claude-sonnet-4-20250514"

        try:
            message = await client.messages.create(
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}") from e

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract text content
        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        tokens_used = None
        if message.usage:
            tokens_used = message.usage.input_tokens + message.usage.output_tokens

        return LLMResponse(
            text=response_text,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            raw_response=message,
        )


class OpenAIProvider(LLMProvider):
    """
    LLM provider using the official OpenAI Python client.

    Requires the openai package (pip install openai) and either:
    - OPENAI_API_KEY environment variable, or
    - api_key in LLMConfig
    """

    async def complete(self, system: str, user: str) -> LLMResponse:
        """Send completion via OpenAI API."""
        start_time = time.time()

        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            ) from e

        # Resolve API key: config > env > error
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or pass in config."
            )

        # Support custom base URL (for Azure, proxies, etc.)
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        client = openai.AsyncOpenAI(**client_kwargs)
        model = self.config.model or "gpt-4o"

        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract text content
        response_text = ""
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.message and choice.message.content:
                response_text = choice.message.content

        tokens_used = None
        if response.usage:
            tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens

        return LLMResponse(
            text=response_text,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            raw_response=response,
        )

    async def complete_streaming(self, system: str, user: str) -> tuple[str, int | None, int]:
        """
        Send completion via OpenAI API with streaming.

        Returns:
            Tuple of (full_text, tokens_used, duration_ms)
        """
        start_time = time.time()

        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            ) from e

        # Resolve API key: config > env > error
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or pass in config."
            )

        # Support custom base URL
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        client = openai.AsyncOpenAI(**client_kwargs)
        model = self.config.model or "gpt-4o"

        try:
            stream = await client.chat.completions.create(
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API streaming call failed: {e}") from e

        response_text = ""
        tokens_used = None

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    response_text += delta.content
            # Usage is included in the final chunk when stream_options.include_usage is True
            if chunk.usage:
                tokens_used = chunk.usage.prompt_tokens + chunk.usage.completion_tokens

        duration_ms = int((time.time() - start_time) * 1000)
        return response_text, tokens_used, duration_ms


class LocalLLMProvider(LLMProvider):
    """
    LLM provider using local models via transformers/llama.cpp.

    Supports models like:
    - qwen2.5-7b-instruct (good balance of quality/speed)
    - smollm-1.7b (very fast, lower quality)
    - phi-3 (Microsoft's efficient model)

    Requires:
    - transformers package
    - torch (for GPU acceleration)
    - Optional: llama-cpp-python for GGUF models

    Model loading is lazy - model is loaded on first completion request.

    Example usage:
        config = LLMConfig(
            provider="local",
            model="Qwen/Qwen2.5-7B-Instruct",
            base_url=None,  # Not used for local
        )
        provider = LocalLLMProvider(config)
        response = await provider.complete("System prompt", "User message")
    """

    # Default model if none specified
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = asyncio.Lock()
        self._device: str | None = None

    async def _load_model(self) -> None:
        """Lazy load model on first use."""
        if self._model is not None:
            return

        async with self._lock:
            # Double-check pattern: model may have been loaded while waiting for lock
            if self._model is not None:
                return  # type: ignore[unreachable]

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "transformers package not installed. "
                    "Install with: pip install transformers torch"
                ) from e

            model_name = self.config.model or self.DEFAULT_MODEL

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self._tokenizer, self._model, self._device = await loop.run_in_executor(
                None,
                self._load_model_sync,
                model_name,
            )

    def _load_model_sync(self, model_name: str) -> tuple[Any, Any, str]:
        """Synchronous model loading."""
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch package not installed. Install with: pip install torch") from e

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu":
            model = model.to(device)

        return tokenizer, model, device

    async def complete(self, system: str, user: str) -> LLMResponse:
        """Generate completion using local model."""
        start_time = time.time()

        await self._load_model()

        # Format prompt for chat model
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Use chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{system}\n\nUser: {user}\n\nAssistant:"

        # Generate in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response_text, tokens_used = await loop.run_in_executor(
            None,
            self._generate_sync,
            prompt,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        return LLMResponse(
            text=response_text,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
        )

    def _generate_sync(self, prompt: str) -> tuple[str, int]:
        """Synchronous generation."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only new tokens
        input_length = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_length:]
        response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response_text, len(response_tokens)


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
    elif config.provider == "openai":
        return OpenAIProvider(config)
    elif config.provider == "local":
        return LocalLLMProvider(config)
    elif config.provider == "mock":
        return MockProvider(config)
    else:
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
        provider: Provider name ("claude-code", "anthropic", "mock")
        model: Optional model override
        api_key: Optional API key (for anthropic provider)

    Returns:
        Response text
    """
    config = LLMConfig(provider=provider, model=model, api_key=api_key)
    llm = create_llm_provider(config)
    response = await llm.complete(system, user)
    return response.text
