"""Public provider and integration package boundaries (#643)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from transcription.historian import llm_client as historian_llm
from transcription.llm_provider import (
    LLMConfig,
    LLMProviderDependencyError,
    OpenAIProvider,
)
from transcription.semantic_adapter import AnthropicSemanticAdapter, OpenAISemanticAdapter


def test_historian_reexports_public_provider_identity() -> None:
    from transcription.llm_provider import AnthropicProvider, OpenAIProvider

    assert historian_llm.OpenAIProvider is OpenAIProvider
    assert historian_llm.AnthropicProvider is AnthropicProvider


def test_cloud_semantic_adapters_use_installed_provider_layer() -> None:
    openai = OpenAISemanticAdapter(api_key="test-key")
    anthropic = AnthropicSemanticAdapter(api_key="test-key")

    assert type(openai._provider).__module__ == "transcription.llm_provider"
    assert type(anthropic._provider).__module__ == "transcription.llm_provider"


def test_missing_openai_sdk_raises_typed_dependency_error() -> None:
    provider = OpenAIProvider(LLMConfig(provider="openai", api_key="test-key"))

    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(LLMProviderDependencyError, match="semantic-openai"):
            asyncio.run(provider.complete("system", "user"))


def test_canonical_integration_modules_are_installed_namespace() -> None:
    from transcription.integrations.langchain_loader import SlowerWhisperLoader
    from transcription.integrations.llamaindex_reader import SlowerWhisperReader

    assert SlowerWhisperLoader.__module__.startswith("transcription.integrations.")
    assert SlowerWhisperReader.__module__.startswith("transcription.integrations.")
