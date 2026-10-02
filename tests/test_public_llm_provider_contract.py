"""Direct behavior coverage for the installed public LLM provider layer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from transcription.llm_provider import (
    AnthropicProvider,
    LLMConfig,
    LLMProviderConfigError,
    LLMProviderDependencyError,
    LocalLLMProvider,
    OpenAIProvider,
)


class TestAnthropicProviderContract:
    @pytest.mark.asyncio
    async def test_missing_sdk_raises_typed_dependency_error(self) -> None:
        provider = AnthropicProvider(LLMConfig(provider="anthropic", api_key="test-key"))

        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(LLMProviderDependencyError, match="semantic-anthropic"):
                await provider.complete("system", "user")

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_typed_config_error(self) -> None:
        provider = AnthropicProvider(LLMConfig(provider="anthropic"))
        sdk = MagicMock()

        with (
            patch.dict("sys.modules", {"anthropic": sdk}),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(LLMProviderConfigError, match="Anthropic API key not found"),
        ):
            await provider.complete("system", "user")

    @pytest.mark.asyncio
    async def test_complete_normalizes_text_and_usage(self) -> None:
        message = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="hello"),
                SimpleNamespace(type="tool_use", text="ignored"),
                SimpleNamespace(type="text", text=" world"),
            ],
            usage=SimpleNamespace(input_tokens=4, output_tokens=6),
        )
        create = AsyncMock(return_value=message)
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        sdk = MagicMock()
        sdk.AsyncAnthropic.return_value = client
        provider = AnthropicProvider(
            LLMConfig(
                provider="anthropic",
                api_key="test-key",
                temperature=0.2,
                max_tokens=17,
            )
        )

        with patch.dict("sys.modules", {"anthropic": sdk}):
            result = await provider.complete("system", "user")

        sdk.AsyncAnthropic.assert_called_once_with(api_key="test-key")
        create.assert_awaited_once_with(
            model="claude-sonnet-4-20250514",
            max_tokens=17,
            temperature=0.2,
            system="system",
            messages=[{"role": "user", "content": "user"}],
        )
        assert result.text == "hello world"
        assert result.tokens_used == 10
        assert result.raw_response is message

    @pytest.mark.asyncio
    async def test_api_failure_preserves_cause(self) -> None:
        upstream = OSError("provider unavailable")
        create = AsyncMock(side_effect=upstream)
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        sdk = MagicMock()
        sdk.AsyncAnthropic.return_value = client
        provider = AnthropicProvider(LLMConfig(provider="anthropic", api_key="test-key"))

        with patch.dict("sys.modules", {"anthropic": sdk}):
            with pytest.raises(RuntimeError, match="Anthropic API call failed") as caught:
                await provider.complete("system", "user")

        assert caught.value.__cause__ is upstream


class TestOpenAIStreamingProviderContract:
    @pytest.mark.asyncio
    async def test_missing_sdk_raises_typed_dependency_error(self) -> None:
        provider = OpenAIProvider(LLMConfig(provider="openai", api_key="test-key"))

        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(LLMProviderDependencyError, match="semantic-openai"):
                await provider.complete_streaming("system", "user")

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_typed_config_error(self) -> None:
        provider = OpenAIProvider(LLMConfig(provider="openai"))
        sdk = MagicMock()

        with (
            patch.dict("sys.modules", {"openai": sdk}),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(LLMProviderConfigError, match="OpenAI API key not found"),
        ):
            await provider.complete_streaming("system", "user")

    @pytest.mark.asyncio
    async def test_streaming_complete_collects_text_and_usage(self) -> None:
        async def chunks():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=" world"))],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
            )

        create = AsyncMock(return_value=chunks())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        sdk = MagicMock()
        sdk.AsyncOpenAI.return_value = client
        provider = OpenAIProvider(
            LLMConfig(
                provider="openai",
                api_key="test-key",
                base_url="https://example.invalid/v1",
                model="test-model",
                temperature=0.4,
                max_tokens=19,
            )
        )

        with patch.dict("sys.modules", {"openai": sdk}):
            text, tokens, duration_ms = await provider.complete_streaming("system", "user")

        sdk.AsyncOpenAI.assert_called_once_with(
            api_key="test-key", base_url="https://example.invalid/v1"
        )
        create.assert_awaited_once_with(
            model="test-model",
            max_tokens=19,
            temperature=0.4,
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert text == "hello world"
        assert tokens == 8
        assert duration_ms >= 0

    @pytest.mark.asyncio
    async def test_streaming_api_failure_preserves_cause(self) -> None:
        upstream = OSError("rate limited")
        create = AsyncMock(side_effect=upstream)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        sdk = MagicMock()
        sdk.AsyncOpenAI.return_value = client
        provider = OpenAIProvider(LLMConfig(provider="openai", api_key="test-key"))

        with patch.dict("sys.modules", {"openai": sdk}):
            with pytest.raises(RuntimeError, match="OpenAI API streaming call failed") as caught:
                await provider.complete_streaming("system", "user")

        assert caught.value.__cause__ is upstream


class TestLocalProviderLoadingContract:
    def test_missing_torch_raises_typed_dependency_error(self) -> None:
        provider = LocalLLMProvider(LLMConfig(provider="local"))

        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(LLMProviderDependencyError, match="semantic-local"):
                provider._load_model_sync("test-model")

    @pytest.mark.parametrize(
        ("cuda_available", "expected_device", "expected_dtype", "expected_device_map"),
        [
            (False, "cpu", "float32", None),
            (True, "cuda", "float16", "auto"),
        ],
    )
    def test_load_model_sync_selects_expected_device(
        self,
        cuda_available: bool,
        expected_device: str,
        expected_dtype: str,
        expected_device_map: str | None,
    ) -> None:
        dtype16 = object()
        dtype32 = object()
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda_available),
            float16=dtype16,
            float32=dtype32,
        )
        tokenizer = object()
        model = MagicMock()
        model.to.return_value = model
        transformers = MagicMock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        transformers.AutoModelForCausalLM.from_pretrained.return_value = model
        provider = LocalLLMProvider(LLMConfig(provider="local"))

        with patch.dict(
            "sys.modules",
            {"torch": torch, "transformers": transformers},
        ):
            loaded_tokenizer, loaded_model, device = provider._load_model_sync("test-model")

        assert loaded_tokenizer is tokenizer
        assert loaded_model is model
        assert device == expected_device
        expected_dtype_value = dtype16 if expected_dtype == "float16" else dtype32
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "test-model",
            torch_dtype=expected_dtype_value,
            device_map=expected_device_map,
        )
        if cuda_available:
            model.to.assert_not_called()
        else:
            model.to.assert_called_once_with("cpu")

    @pytest.mark.asyncio
    async def test_lazy_load_populates_provider_state(self) -> None:
        tokenizer = object()
        model = object()
        transformers = MagicMock()
        provider = LocalLLMProvider(LLMConfig(provider="local"))
        provider._load_model_sync = MagicMock(return_value=(tokenizer, model, "cpu"))

        with patch.dict("sys.modules", {"transformers": transformers}):
            await provider._load_model()

        provider._load_model_sync.assert_called_once_with(provider.DEFAULT_MODEL)
        assert provider._tokenizer is tokenizer
        assert provider._model is model
        assert provider._device == "cpu"

    @pytest.mark.asyncio
    async def test_complete_uses_fallback_prompt_without_chat_template(self) -> None:
        provider = LocalLLMProvider(LLMConfig(provider="local"))
        provider._model = object()
        provider._tokenizer = object()
        provider._generate_sync = MagicMock(return_value=("response", 2))

        result = await provider.complete("system", "user")

        provider._generate_sync.assert_called_once_with("system\n\nUser: user\n\nAssistant:")
        assert result.text == "response"
        assert result.tokens_used == 2
