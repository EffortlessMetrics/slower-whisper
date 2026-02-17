"""Tests for local LLM semantic provider (#89).

This module tests the local semantic provider implementation including:
- LocalSemanticConfig validation and loading
- LocalSemanticProvider annotation functionality
- Health check and error handling
- Caching behavior
- Integration with MockLocalLLMProvider for deterministic tests
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from transcription.local_llm_provider import (
    MockLocalLLMProvider,
    get_availability_status,
    is_available,
)
from transcription.semantic_adapter import (
    SEMANTIC_SCHEMA_VERSION,
    ActionItem,
    ChunkContext,
    LocalLLMSemanticAdapter,
    NormalizedAnnotation,
    ProviderHealth,
    SemanticAnnotation,
    create_adapter,
)
from transcription.semantic_providers.local import (
    LocalSemanticConfig,
    LocalSemanticProvider,
    create_local_provider,
)


class TestLocalSemanticConfig:
    """Tests for LocalSemanticConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LocalSemanticConfig()

        assert config.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.device == "auto"
        assert config.temperature == 0.1
        assert config.max_tokens == 1024
        assert config.batch_size == 1
        assert config.extraction_mode == "combined"
        assert config.context_window == 3
        assert config.enable_caching is True

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = LocalSemanticConfig(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            device="cuda",
            temperature=0.0,
            max_tokens=512,
            batch_size=2,
            extraction_mode="topics",
            context_window=5,
            enable_caching=False,
        )

        assert config.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert config.device == "cuda"
        assert config.temperature == 0.0
        assert config.max_tokens == 512
        assert config.batch_size == 2
        assert config.extraction_mode == "topics"
        assert config.context_window == 5
        assert config.enable_caching is False

    def test_invalid_temperature_low(self) -> None:
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be 0.0-2.0"):
            LocalSemanticConfig(temperature=-0.1)

    def test_invalid_temperature_high(self) -> None:
        """Test that temperature > 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be 0.0-2.0"):
            LocalSemanticConfig(temperature=2.5)

    def test_invalid_max_tokens(self) -> None:
        """Test that non-positive max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            LocalSemanticConfig(max_tokens=0)

    def test_invalid_batch_size(self) -> None:
        """Test that non-positive batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            LocalSemanticConfig(batch_size=0)

    def test_invalid_context_window(self) -> None:
        """Test that negative context_window raises ValueError."""
        with pytest.raises(ValueError, match="context_window must be >= 0"):
            LocalSemanticConfig(context_window=-1)

    def test_from_env_model_name(self) -> None:
        """Test loading model_name from environment."""
        with patch.dict(
            os.environ,
            {"SLOWER_WHISPER_SEMANTIC_MODEL_NAME": "Qwen/Qwen2.5-3B-Instruct"},
        ):
            config = LocalSemanticConfig.from_env()
            assert config.model_name == "Qwen/Qwen2.5-3B-Instruct"

    def test_from_env_device(self) -> None:
        """Test loading device from environment."""
        with patch.dict(os.environ, {"SLOWER_WHISPER_SEMANTIC_DEVICE": "cpu"}):
            config = LocalSemanticConfig.from_env()
            assert config.device == "cpu"

    def test_from_env_invalid_device(self) -> None:
        """Test that invalid device from env raises ValueError."""
        with patch.dict(os.environ, {"SLOWER_WHISPER_SEMANTIC_DEVICE": "mps"}):
            with pytest.raises(ValueError, match="Invalid"):
                LocalSemanticConfig.from_env()

    def test_from_env_extraction_mode(self) -> None:
        """Test loading extraction_mode from environment."""
        with patch.dict(os.environ, {"SLOWER_WHISPER_SEMANTIC_EXTRACTION_MODE": "risks"}):
            config = LocalSemanticConfig.from_env()
            assert config.extraction_mode == "risks"

    def test_from_env_invalid_extraction_mode(self) -> None:
        """Test that invalid extraction_mode from env raises ValueError."""
        with patch.dict(os.environ, {"SLOWER_WHISPER_SEMANTIC_EXTRACTION_MODE": "invalid"}):
            with pytest.raises(ValueError, match="Invalid"):
                LocalSemanticConfig.from_env()

    def test_from_env_numeric_fields(self) -> None:
        """Test loading numeric fields from environment."""
        env_vars = {
            "SLOWER_WHISPER_SEMANTIC_TEMPERATURE": "0.5",
            "SLOWER_WHISPER_SEMANTIC_MAX_TOKENS": "2048",
            "SLOWER_WHISPER_SEMANTIC_BATCH_SIZE": "4",
            "SLOWER_WHISPER_SEMANTIC_CONTEXT_WINDOW": "2",
        }
        with patch.dict(os.environ, env_vars):
            config = LocalSemanticConfig.from_env()
            assert config.temperature == 0.5
            assert config.max_tokens == 2048
            assert config.batch_size == 4
            assert config.context_window == 2

    def test_from_env_caching(self) -> None:
        """Test loading enable_caching from environment."""
        with patch.dict(os.environ, {"SLOWER_WHISPER_SEMANTIC_ENABLE_CACHING": "false"}):
            config = LocalSemanticConfig.from_env()
            assert config.enable_caching is False

    def test_from_env_custom_prefix(self) -> None:
        """Test loading from environment with custom prefix."""
        with patch.dict(os.environ, {"MY_APP_MODEL_NAME": "custom-model"}):
            config = LocalSemanticConfig.from_env(prefix="MY_APP_")
            assert config.model_name == "custom-model"


class TestLocalSemanticProvider:
    """Tests for LocalSemanticProvider."""

    def test_default_construction(self) -> None:
        """Test default provider construction."""
        provider = LocalSemanticProvider()

        assert provider.config.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert provider.config.device == "auto"

    def test_construction_with_config(self) -> None:
        """Test provider construction with config."""
        config = LocalSemanticConfig(model_name="test-model", device="cpu")
        provider = LocalSemanticProvider(config)

        assert provider.config.model_name == "test-model"
        assert provider.config.device == "cpu"

    def test_construction_with_kwargs(self) -> None:
        """Test provider construction with kwargs."""
        provider = LocalSemanticProvider(model_name="test-model", temperature=0.5)

        assert provider.config.model_name == "test-model"
        assert provider.config.temperature == 0.5

    def test_construction_config_plus_kwargs(self) -> None:
        """Test that kwargs override config values."""
        config = LocalSemanticConfig(model_name="config-model", temperature=0.1)
        provider = LocalSemanticProvider(config, temperature=0.5)

        assert provider.config.model_name == "config-model"
        assert provider.config.temperature == 0.5

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        provider = LocalSemanticProvider(model_name="test-model")
        assert provider.model_name == "test-model"

    def test_device_property(self) -> None:
        """Test device property."""
        provider = LocalSemanticProvider(device="cpu")
        assert provider.device == "cpu"


class TestLocalSemanticProviderWithMock:
    """Tests using MockLocalLLMProvider for deterministic results."""

    @pytest.fixture
    def mock_response(self) -> str:
        """JSON response for mock provider."""
        return json.dumps(
            {
                "topics": [{"label": "pricing", "confidence": 0.9, "evidence": "budget"}],
                "risks": [
                    {
                        "type": "pricing_objection",
                        "severity": "medium",
                        "confidence": 0.8,
                        "evidence": "budget concerns",
                    }
                ],
                "actions": [
                    {
                        "description": "Send report",
                        "assignee": "agent",
                        "due": "tomorrow",
                        "priority": "high",
                    }
                ],
            }
        )

    @pytest.fixture
    def mock_provider(self, mock_response: str) -> MockLocalLLMProvider:
        """Create mock provider with predefined response."""
        return MockLocalLLMProvider(default_response=mock_response)

    def test_health_check_available(self) -> None:
        """Test health check reports availability status."""
        provider = LocalSemanticProvider()
        health = provider.health_check()

        assert isinstance(health, ProviderHealth)
        # Available depends on torch/transformers being installed
        # In test environment, it may or may not be available
        assert isinstance(health.available, bool)
        assert health.latency_ms >= 0

    def test_health_check_unavailable_message(self) -> None:
        """Test health check includes error message when unavailable."""
        provider = LocalSemanticProvider()
        health = provider.health_check()

        if not health.available:
            assert health.error is not None
            assert "not available" in health.error.lower() or "transformers" in health.error.lower()

    def test_cache_operations(self) -> None:
        """Test cache clear and stats."""
        provider = LocalSemanticProvider(enable_caching=True)

        stats = provider.get_cache_stats()
        assert stats["size"] == 0
        assert stats["enabled"] is True

        # Clear empty cache
        cleared = provider.clear_cache()
        assert cleared == 0

    def test_cache_disabled(self) -> None:
        """Test that cache can be disabled."""
        provider = LocalSemanticProvider(enable_caching=False)

        stats = provider.get_cache_stats()
        assert stats["enabled"] is False


class TestLocalLLMSemanticAdapter:
    """Tests for LocalLLMSemanticAdapter (underlying adapter)."""

    def test_adapter_creation(self) -> None:
        """Test adapter construction."""
        adapter = LocalLLMSemanticAdapter()

        assert adapter.model == "Qwen/Qwen2.5-7B-Instruct"
        assert adapter.temperature == 0.1
        assert adapter.max_tokens == 1024
        assert adapter.extraction_mode == "combined"

    def test_adapter_custom_model(self) -> None:
        """Test adapter with custom model."""
        adapter = LocalLLMSemanticAdapter(model="custom-model")
        assert adapter.model == "custom-model"

    def test_adapter_extraction_modes(self) -> None:
        """Test different extraction modes."""
        for mode in ("combined", "topics", "risks", "actions"):
            adapter = LocalLLMSemanticAdapter(extraction_mode=mode)  # type: ignore
            assert adapter.extraction_mode == mode

    def test_adapter_health_check(self) -> None:
        """Test adapter health check."""
        adapter = LocalLLMSemanticAdapter()
        health = adapter.health_check()

        assert isinstance(health, ProviderHealth)
        assert isinstance(health.available, bool)

    def test_adapter_unavailable_returns_empty_annotation(self) -> None:
        """Test that unavailable adapter returns low-confidence annotation."""
        # Force unavailable state to avoid loading/inferencing real local models in fast tests.
        with patch("transcription.local_llm_provider.is_available", return_value=False):
            adapter = LocalLLMSemanticAdapter()
            context = ChunkContext(speaker_id="test", start=0.0, end=10.0)

            result = adapter.annotate_chunk("Test text", context)

        assert isinstance(result, SemanticAnnotation)
        assert result.provider == "local-llm"
        assert result.confidence == 0.0


class TestCreateAdapter:
    """Tests for create_adapter factory with local-llm."""

    def test_create_local_llm_adapter(self) -> None:
        """Test creating local-llm adapter via factory."""
        adapter = create_adapter("local-llm")

        assert isinstance(adapter, LocalLLMSemanticAdapter)
        assert adapter.model == "Qwen/Qwen2.5-7B-Instruct"

    def test_create_local_llm_with_model(self) -> None:
        """Test creating local-llm adapter with custom model."""
        adapter = create_adapter("local-llm", model="Qwen/Qwen2.5-3B-Instruct")

        assert isinstance(adapter, LocalLLMSemanticAdapter)
        assert adapter.model == "Qwen/Qwen2.5-3B-Instruct"

    def test_create_local_llm_with_options(self) -> None:
        """Test creating local-llm adapter with options."""
        adapter = create_adapter(
            "local-llm",
            model="test-model",
            temperature=0.5,
            max_tokens=512,
            extraction_mode="risks",
        )

        assert isinstance(adapter, LocalLLMSemanticAdapter)
        assert adapter.model == "test-model"
        assert adapter.temperature == 0.5
        assert adapter.max_tokens == 512
        assert adapter.extraction_mode == "risks"


class TestCreateLocalProvider:
    """Tests for create_local_provider factory."""

    def test_create_default(self) -> None:
        """Test creating provider with defaults."""
        provider = create_local_provider()

        assert isinstance(provider, LocalSemanticProvider)
        assert provider.model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_create_with_model(self) -> None:
        """Test creating provider with model name."""
        provider = create_local_provider(model_name="test-model")

        assert provider.model_name == "test-model"

    def test_create_with_device(self) -> None:
        """Test creating provider with device."""
        provider = create_local_provider(device="cpu")

        assert provider.device == "cpu"

    def test_create_with_kwargs(self) -> None:
        """Test creating provider with additional kwargs."""
        provider = create_local_provider(
            model_name="test-model",
            temperature=0.5,
            enable_caching=False,
        )

        assert provider.model_name == "test-model"
        assert provider.config.temperature == 0.5
        assert provider.config.enable_caching is False


class TestAvailabilityFunctions:
    """Tests for availability checking functions."""

    def test_is_available_returns_bool(self) -> None:
        """Test that is_available returns boolean."""
        result = is_available()
        assert isinstance(result, bool)

    def test_get_availability_status_structure(self) -> None:
        """Test get_availability_status returns expected structure."""
        status = get_availability_status()

        assert isinstance(status, dict)
        assert "torch" in status
        assert "transformers" in status
        assert "available" in status
        assert isinstance(status["torch"], bool)
        assert isinstance(status["transformers"], bool)
        assert isinstance(status["available"], bool)

    def test_availability_consistency(self) -> None:
        """Test that is_available matches get_availability_status."""
        available = is_available()
        status = get_availability_status()

        assert available == status["available"]
        # available should be True only if both dependencies are present
        assert status["available"] == (status["torch"] and status["transformers"])


class TestChunkContextIntegration:
    """Tests for ChunkContext with local provider."""

    def test_context_creation(self) -> None:
        """Test ChunkContext creation for local provider."""
        context = ChunkContext(
            speaker_id="agent",
            segment_ids=[0, 1, 2],
            start=0.0,
            end=30.0,
            previous_chunks=["Previous text"],
            language="en",
        )

        assert context.speaker_id == "agent"
        assert context.segment_ids == [0, 1, 2]
        assert context.start == 0.0
        assert context.end == 30.0
        assert context.previous_chunks == ["Previous text"]
        assert context.language == "en"

    def test_context_defaults(self) -> None:
        """Test ChunkContext default values."""
        context = ChunkContext()

        assert context.speaker_id is None
        assert context.segment_ids == []
        assert context.start == 0.0
        assert context.end == 0.0
        assert context.previous_chunks == []
        assert context.language == "en"


class TestSemanticAnnotationOutput:
    """Tests for SemanticAnnotation output structure."""

    def test_annotation_schema_version(self) -> None:
        """Test annotation includes correct schema version."""
        annotation = SemanticAnnotation(provider="local-llm", model="test")

        assert annotation.schema_version == SEMANTIC_SCHEMA_VERSION

    def test_annotation_serialization(self) -> None:
        """Test annotation serialization to dict."""
        annotation = SemanticAnnotation(
            provider="local-llm",
            model="test-model",
            normalized=NormalizedAnnotation(
                topics=["pricing"],
                risk_tags=["pricing_objection"],
                action_items=[ActionItem(text="Send report", confidence=0.9)],
            ),
            confidence=0.85,
            latency_ms=150,
        )

        d = annotation.to_dict()

        assert d["schema_version"] == SEMANTIC_SCHEMA_VERSION
        assert d["provider"] == "local-llm"
        assert d["model"] == "test-model"
        assert d["normalized"]["topics"] == ["pricing"]
        assert d["normalized"]["risk_tags"] == ["pricing_objection"]
        assert len(d["normalized"]["action_items"]) == 1
        assert d["confidence"] == 0.85
        assert d["latency_ms"] == 150

    def test_annotation_deserialization(self) -> None:
        """Test annotation deserialization from dict."""
        d = {
            "schema_version": SEMANTIC_SCHEMA_VERSION,
            "provider": "local-llm",
            "model": "test-model",
            "normalized": {
                "topics": ["support"],
                "intent": "question",
                "sentiment": "negative",
                "action_items": [{"text": "Follow up", "confidence": 0.8}],
                "risk_tags": ["escalation"],
            },
            "confidence": 0.75,
            "latency_ms": 200,
        }

        annotation = SemanticAnnotation.from_dict(d)

        assert annotation.schema_version == SEMANTIC_SCHEMA_VERSION
        assert annotation.provider == "local-llm"
        assert annotation.model == "test-model"
        assert annotation.normalized.topics == ["support"]
        assert annotation.normalized.intent == "question"
        assert annotation.normalized.sentiment == "negative"
        assert len(annotation.normalized.action_items) == 1
        assert annotation.normalized.risk_tags == ["escalation"]


class TestProviderHealthOutput:
    """Tests for ProviderHealth output structure."""

    def test_health_available(self) -> None:
        """Test health response when available."""
        health = ProviderHealth(available=True, latency_ms=5)

        assert health.available is True
        assert health.quota_remaining is None  # No quota for local
        assert health.error is None
        assert health.latency_ms == 5

    def test_health_unavailable(self) -> None:
        """Test health response when unavailable."""
        health = ProviderHealth(
            available=False,
            error="torch not installed",
            latency_ms=1,
        )

        assert health.available is False
        assert health.error == "torch not installed"

    def test_health_serialization(self) -> None:
        """Test health serialization to dict."""
        health = ProviderHealth(available=True, quota_remaining=None, latency_ms=10)
        d = health.to_dict()

        assert d["available"] is True
        assert d["latency_ms"] == 10
        assert "quota_remaining" not in d  # None values omitted
        assert "error" not in d  # None values omitted


class TestProtocolCompliance:
    """Tests for SemanticAdapter protocol compliance."""

    def test_local_llm_adapter_satisfies_protocol(self) -> None:
        """Verify LocalLLMSemanticAdapter satisfies SemanticAdapter protocol."""
        from transcription.semantic_adapter import SemanticAdapter

        with patch("transcription.local_llm_provider.is_available", return_value=False):
            adapter: SemanticAdapter = LocalLLMSemanticAdapter()
            context = ChunkContext()
            # Should have annotate_chunk method
            result = adapter.annotate_chunk("Test", context)
        assert isinstance(result, SemanticAnnotation)

        # Should have health_check method
        health = adapter.health_check()
        assert isinstance(health, ProviderHealth)

    def test_local_provider_matches_adapter_interface(self) -> None:
        """Verify LocalSemanticProvider has same interface as adapter."""
        with patch("transcription.local_llm_provider.is_available", return_value=False):
            provider = LocalSemanticProvider()
            context = ChunkContext()
            # Should have annotate_chunk method
            result = provider.annotate_chunk("Test", context)
        assert isinstance(result, SemanticAnnotation)

        # Should have health_check method
        health = provider.health_check()
        assert isinstance(health, ProviderHealth)


class TestModuleImports:
    """Tests for module import structure."""

    def test_import_from_semantic_providers_package(self) -> None:
        """Test imports from transcription.semantic_providers package."""
        from transcription.semantic_providers import (
            ChunkContext,
            LocalLLMSemanticAdapter,
            create_adapter,
        )

        assert LocalLLMSemanticAdapter is not None
        assert ChunkContext is not None
        assert create_adapter is not None

    def test_import_from_semantic_adapter(self) -> None:
        """Test imports from transcription.semantic_adapter module."""
        from transcription.local_llm_provider import LocalLLMProvider, is_available
        from transcription.semantic_adapter import LocalLLMSemanticAdapter

        assert LocalLLMSemanticAdapter is not None
        assert LocalLLMProvider is not None
        assert is_available is not None

    def test_import_from_local_module(self) -> None:
        """Test imports from transcription.semantic_providers.local module."""
        from transcription.semantic_providers.local import (
            LocalSemanticConfig,
            LocalSemanticProvider,
            create_local_provider,
            is_available,
        )

        assert LocalSemanticConfig is not None
        assert LocalSemanticProvider is not None
        assert create_local_provider is not None
        assert is_available is not None
