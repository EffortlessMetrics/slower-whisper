"""Tests for cloud LLM semantic providers (Issue #90).

This module tests the OpenAI and Anthropic semantic adapters with:
- Mocked API responses (no real API calls required)
- Schema validation for normalized output
- Retry logic with exponential backoff
- Error classification and handling
- Timeout handling
- Rate limit error detection

Run with:
    uv run python -m pytest tests/test_semantic_cloud_providers.py -v

For integration tests with real API keys:
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \
    uv run python -m pytest tests/test_semantic_cloud_providers.py -v -m external
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

# Import directly from semantic_adapter module to avoid circular import issues
# with the transcription.semantic package
if TYPE_CHECKING:
    from transcription.semantic_adapter import (
        AnthropicSemanticAdapter,
        CloudLLMSemanticAdapter,
        OpenAISemanticAdapter,
    )

# Runtime imports
from transcription.semantic_adapter import (
    SEMANTIC_SCHEMA_VERSION,
    ActionItem,
    AnthropicSemanticAdapter,
    ChunkContext,
    CloudLLMSemanticAdapter,
    NormalizedAnnotation,
    OpenAISemanticAdapter,
    ProviderHealth,
    SemanticAnnotation,
    create_adapter,
)

# -----------------------------------------------------------------------------
# Skip markers for real API tests
# -----------------------------------------------------------------------------

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

requires_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_valid_response() -> str:
    """Return a valid JSON response matching semantic schema."""
    return json.dumps(
        {
            "topics": ["pricing", "contract_terms"],
            "intent": "objection",
            "sentiment": "negative",
            "action_items": [
                {"text": "Send revised proposal", "speaker_id": "agent", "confidence": 0.9}
            ],
            "risk_tags": ["pricing_objection", "churn_risk"],
        }
    )


@pytest.fixture
def sample_minimal_response() -> str:
    """Return a minimal valid JSON response."""
    return json.dumps(
        {
            "topics": [],
            "intent": None,
            "sentiment": None,
            "action_items": [],
            "risk_tags": [],
        }
    )


@pytest.fixture
def sample_chunk_context() -> ChunkContext:
    """Return a sample chunk context for testing."""
    return ChunkContext(
        speaker_id="customer",
        segment_ids=[1, 2, 3],
        start=10.0,
        end=25.0,
        previous_chunks=["Hello, how can I help you today?"],
        language="en",
    )


# -----------------------------------------------------------------------------
# Factory Function Tests
# -----------------------------------------------------------------------------


class TestCreateAdapterFactory:
    """Test the create_adapter factory function for cloud providers."""

    def test_create_openai_adapter(self) -> None:
        """Test creating OpenAI adapter via factory."""
        adapter = create_adapter("openai", api_key="test-key")
        assert isinstance(adapter, OpenAISemanticAdapter)

    def test_create_anthropic_adapter(self) -> None:
        """Test creating Anthropic adapter via factory."""
        adapter = create_adapter("anthropic", api_key="test-key")
        assert isinstance(adapter, AnthropicSemanticAdapter)

    def test_factory_passes_model_kwarg(self) -> None:
        """Test that factory passes model kwarg to adapter."""
        adapter = create_adapter("openai", api_key="test-key", model="gpt-4-turbo")
        assert isinstance(adapter, OpenAISemanticAdapter)
        assert adapter._model == "gpt-4-turbo"

    def test_factory_passes_timeout_kwarg(self) -> None:
        """Test that factory passes timeout kwarg to adapter."""
        adapter = create_adapter("openai", api_key="test-key", timeout_ms=60000)
        assert isinstance(adapter, OpenAISemanticAdapter)
        assert adapter._timeout_ms == 60000

    def test_factory_passes_retry_kwargs(self) -> None:
        """Test that factory passes retry kwargs to adapter."""
        adapter = create_adapter(
            "anthropic",
            api_key="test-key",
            max_retries=5,
            initial_backoff_ms=2000,
        )
        assert isinstance(adapter, AnthropicSemanticAdapter)
        assert adapter._max_retries == 5
        assert adapter._initial_backoff_ms == 2000


# -----------------------------------------------------------------------------
# OpenAI Adapter Tests
# -----------------------------------------------------------------------------


class TestOpenAISemanticAdapter:
    """Test OpenAI semantic adapter with mocked responses."""

    def test_adapter_inherits_from_cloud_base(self) -> None:
        """Test that OpenAISemanticAdapter inherits from CloudLLMSemanticAdapter."""
        adapter = OpenAISemanticAdapter(api_key="test-key")
        assert isinstance(adapter, CloudLLMSemanticAdapter)

    def test_adapter_provider_name(self) -> None:
        """Test that provider name is 'openai'."""
        adapter = OpenAISemanticAdapter(api_key="test-key")
        assert adapter._get_provider_name() == "openai"

    def test_adapter_model_name_default(self) -> None:
        """Test default model name."""
        adapter = OpenAISemanticAdapter(api_key="test-key")
        assert adapter._get_model_name() == "gpt-4o"

    def test_adapter_model_name_custom(self) -> None:
        """Test custom model name."""
        adapter = OpenAISemanticAdapter(api_key="test-key", model="gpt-4o-mini")
        assert adapter._get_model_name() == "gpt-4o-mini"

    def test_annotate_chunk_success(
        self, sample_valid_response: str, sample_chunk_context: ChunkContext
    ) -> None:
        """Test successful annotation with mocked OpenAI response."""
        adapter = OpenAISemanticAdapter(api_key="test-key-123")

        # Mock the guarded provider's complete method
        mock_response = MagicMock()
        mock_response.text = sample_valid_response
        mock_response.duration_ms = 150

        async def mock_complete(system: str, user: str):
            return mock_response

        adapter._guarded_provider.complete = mock_complete

        # Test annotation
        result = adapter.annotate_chunk("The price is too high", sample_chunk_context)

        # Verify result structure
        assert isinstance(result, SemanticAnnotation)
        assert result.schema_version == SEMANTIC_SCHEMA_VERSION
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.confidence == 0.85

        # Verify normalized annotation
        assert "pricing" in result.normalized.topics
        assert result.normalized.intent == "objection"
        assert result.normalized.sentiment == "negative"
        assert "pricing_objection" in result.normalized.risk_tags

        # Verify action items
        assert len(result.normalized.action_items) == 1
        assert result.normalized.action_items[0].text == "Send revised proposal"
        assert result.normalized.action_items[0].speaker_id == "agent"

    def test_annotate_chunk_empty_response(
        self, sample_minimal_response: str, sample_chunk_context: ChunkContext
    ) -> None:
        """Test annotation with minimal/empty response."""
        adapter = OpenAISemanticAdapter(api_key="test-key-123")

        mock_response = MagicMock()
        mock_response.text = sample_minimal_response
        mock_response.duration_ms = 100

        async def mock_complete(system: str, user: str):
            return mock_response

        adapter._guarded_provider.complete = mock_complete

        result = adapter.annotate_chunk("Hello", sample_chunk_context)

        assert isinstance(result, SemanticAnnotation)
        assert result.normalized.topics == []
        assert result.normalized.action_items == []
        assert result.normalized.risk_tags == []

    def test_annotate_chunk_timeout_error(self, sample_chunk_context: ChunkContext) -> None:
        """Test that timeout errors are handled gracefully."""
        adapter = OpenAISemanticAdapter(
            api_key="test-key-123",
            timeout_ms=100,
            max_retries=0,
        )

        async def mock_complete_timeout(system: str, user: str):
            raise TimeoutError("Request timed out")

        adapter._guarded_provider.complete = mock_complete_timeout

        result = adapter.annotate_chunk("Test text", sample_chunk_context)

        # Verify graceful degradation
        assert isinstance(result, SemanticAnnotation)
        assert result.provider == "openai"
        assert result.confidence == 0.0
        assert result.normalized.topics == []
        assert "error" in result.raw_model_output
        assert result.raw_model_output.get("error_type") == "timeout"

    def test_annotate_chunk_rate_limit_error(self, sample_chunk_context: ChunkContext) -> None:
        """Test that rate limit errors are handled gracefully."""
        adapter = OpenAISemanticAdapter(
            api_key="test-key-123",
            max_retries=0,
        )

        async def mock_complete_rate_limit(system: str, user: str):
            raise RuntimeError("Rate limit exceeded: 429 Too Many Requests")

        adapter._guarded_provider.complete = mock_complete_rate_limit

        result = adapter.annotate_chunk("Test text", sample_chunk_context)

        assert isinstance(result, SemanticAnnotation)
        assert result.confidence == 0.0
        assert "error" in result.raw_model_output
        assert result.raw_model_output.get("error_type") == "rate_limit"

    def test_annotate_chunk_malformed_json(self, sample_chunk_context: ChunkContext) -> None:
        """Test that malformed JSON responses are handled gracefully."""
        adapter = OpenAISemanticAdapter(
            api_key="test-key-123",
            max_retries=0,
        )

        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON { broken"
        mock_response.duration_ms = 100

        async def mock_complete_malformed(system: str, user: str):
            return mock_response

        adapter._guarded_provider.complete = mock_complete_malformed

        result = adapter.annotate_chunk("Test text", sample_chunk_context)

        assert isinstance(result, SemanticAnnotation)
        assert result.confidence == 0.0
        assert "error" in result.raw_model_output

    def test_annotate_chunk_with_markdown_code_fence(
        self, sample_chunk_context: ChunkContext
    ) -> None:
        """Test that markdown code fences are stripped from responses."""
        adapter = OpenAISemanticAdapter(api_key="test-key-123")

        # Response wrapped in markdown code fence
        response_with_fence = """```json
{
    "topics": ["support"],
    "intent": "question",
    "sentiment": "neutral",
    "action_items": [],
    "risk_tags": []
}
```"""

        mock_response = MagicMock()
        mock_response.text = response_with_fence
        mock_response.duration_ms = 100

        async def mock_complete(system: str, user: str):
            return mock_response

        adapter._guarded_provider.complete = mock_complete

        result = adapter.annotate_chunk("How do I reset?", sample_chunk_context)

        assert isinstance(result, SemanticAnnotation)
        assert "support" in result.normalized.topics
        assert result.normalized.intent == "question"

    def test_health_check_missing_api_key(self) -> None:
        """Test health check returns unavailable when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
            with patch.dict(os.environ, env_without_key, clear=True):
                adapter = OpenAISemanticAdapter(api_key=None)

                health = adapter.health_check()

                assert isinstance(health, ProviderHealth)
                assert health.available is False
                assert "API key not configured" in (health.error or "")

    def test_health_check_with_api_key(self) -> None:
        """Test health check returns available when API key is set."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai package not installed")

        adapter = OpenAISemanticAdapter(api_key="test-key-123")

        # Mock the openai client and models list
        mock_client = MagicMock()
        mock_client.models.list.return_value = iter([])

        with patch.object(openai, "OpenAI", return_value=mock_client):
            health = adapter.health_check()

        assert isinstance(health, ProviderHealth)
        assert health.available is True
        assert health.latency_ms >= 0


# -----------------------------------------------------------------------------
# Anthropic Adapter Tests
# -----------------------------------------------------------------------------


class TestAnthropicSemanticAdapter:
    """Test Anthropic semantic adapter with mocked responses."""

    def test_adapter_inherits_from_cloud_base(self) -> None:
        """Test that AnthropicSemanticAdapter inherits from CloudLLMSemanticAdapter."""
        adapter = AnthropicSemanticAdapter(api_key="test-key")
        assert isinstance(adapter, CloudLLMSemanticAdapter)

    def test_adapter_provider_name(self) -> None:
        """Test that provider name is 'anthropic'."""
        adapter = AnthropicSemanticAdapter(api_key="test-key")
        assert adapter._get_provider_name() == "anthropic"

    def test_adapter_model_name_default(self) -> None:
        """Test default model name."""
        adapter = AnthropicSemanticAdapter(api_key="test-key")
        assert adapter._get_model_name() == "claude-sonnet-4-20250514"

    def test_adapter_model_name_custom(self) -> None:
        """Test custom model name."""
        adapter = AnthropicSemanticAdapter(api_key="test-key", model="claude-3-5-haiku-20241022")
        assert adapter._get_model_name() == "claude-3-5-haiku-20241022"

    def test_annotate_chunk_success(
        self, sample_valid_response: str, sample_chunk_context: ChunkContext
    ) -> None:
        """Test successful annotation with mocked Anthropic response."""
        adapter = AnthropicSemanticAdapter(api_key="test-key-123")

        # Mock the guarded provider's complete method
        mock_response = MagicMock()
        mock_response.text = sample_valid_response
        mock_response.duration_ms = 200

        async def mock_complete(system: str, user: str):
            return mock_response

        adapter._guarded_provider.complete = mock_complete

        # Test annotation
        result = adapter.annotate_chunk("The price is too high", sample_chunk_context)

        # Verify result structure
        assert isinstance(result, SemanticAnnotation)
        assert result.schema_version == SEMANTIC_SCHEMA_VERSION
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.confidence == 0.85

        # Verify normalized annotation
        assert "pricing" in result.normalized.topics
        assert result.normalized.intent == "objection"
        assert result.normalized.sentiment == "negative"

    def test_annotate_chunk_empty_response(
        self, sample_minimal_response: str, sample_chunk_context: ChunkContext
    ) -> None:
        """Test annotation with minimal/empty response."""
        adapter = AnthropicSemanticAdapter(api_key="test-key-123")

        mock_response = MagicMock()
        mock_response.text = sample_minimal_response
        mock_response.duration_ms = 150

        async def mock_complete(system: str, user: str):
            return mock_response

        adapter._guarded_provider.complete = mock_complete

        result = adapter.annotate_chunk("Hello", sample_chunk_context)

        assert isinstance(result, SemanticAnnotation)
        assert result.normalized.topics == []
        assert result.normalized.action_items == []

    def test_annotate_chunk_timeout_error(self, sample_chunk_context: ChunkContext) -> None:
        """Test that timeout errors are handled gracefully."""
        adapter = AnthropicSemanticAdapter(
            api_key="test-key-123",
            timeout_ms=100,
            max_retries=0,
        )

        async def mock_complete_timeout(system: str, user: str):
            raise TimeoutError("Request timed out")

        adapter._guarded_provider.complete = mock_complete_timeout

        result = adapter.annotate_chunk("Test text", sample_chunk_context)

        assert isinstance(result, SemanticAnnotation)
        assert result.provider == "anthropic"
        assert result.confidence == 0.0
        assert "error" in result.raw_model_output
        assert result.raw_model_output.get("error_type") == "timeout"

    def test_annotate_chunk_rate_limit_error(self, sample_chunk_context: ChunkContext) -> None:
        """Test that rate limit errors are handled gracefully."""
        adapter = AnthropicSemanticAdapter(
            api_key="test-key-123",
            max_retries=0,
        )

        async def mock_complete_rate_limit(system: str, user: str):
            raise RuntimeError("Rate limit exceeded")

        adapter._guarded_provider.complete = mock_complete_rate_limit

        result = adapter.annotate_chunk("Test text", sample_chunk_context)

        assert isinstance(result, SemanticAnnotation)
        assert result.confidence == 0.0
        assert result.raw_model_output.get("error_type") == "rate_limit"

    def test_health_check_missing_api_key(self) -> None:
        """Test health check returns unavailable when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            with patch.dict(os.environ, env_without_key, clear=True):
                adapter = AnthropicSemanticAdapter(api_key=None)

                health = adapter.health_check()

                assert isinstance(health, ProviderHealth)
                assert health.available is False
                assert "API key not configured" in (health.error or "")

    def test_health_check_with_api_key(self) -> None:
        """Test health check returns available when API key is set."""
        try:
            import anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        adapter = AnthropicSemanticAdapter(api_key="test-key-123")

        # Mock the anthropic client
        mock_client = MagicMock()
        mock_client.messages.count_tokens.return_value = MagicMock(input_tokens=5)

        with patch.object(anthropic, "Anthropic", return_value=mock_client):
            health = adapter.health_check()

        assert isinstance(health, ProviderHealth)
        assert health.available is True


# -----------------------------------------------------------------------------
# Retry Logic Tests
# -----------------------------------------------------------------------------


class TestRetryLogic:
    """Test retry logic for cloud adapters."""

    def test_error_classification_rate_limit(self) -> None:
        """Test rate limit error classification."""
        adapter = OpenAISemanticAdapter(api_key="test-key")

        assert adapter._classify_error(RuntimeError("Rate limit exceeded")) == "rate_limit"
        assert adapter._classify_error(RuntimeError("429 Too Many Requests")) == "rate_limit"
        assert adapter._classify_error(RuntimeError("quota exceeded")) == "rate_limit"
        assert adapter._classify_error(RuntimeError("ratelimit error")) == "rate_limit"

    def test_error_classification_timeout(self) -> None:
        """Test timeout error classification."""
        adapter = OpenAISemanticAdapter(api_key="test-key")

        assert adapter._classify_error(RuntimeError("Request timed out")) == "timeout"
        assert adapter._classify_error(RuntimeError("timeout error")) == "timeout"
        assert adapter._classify_error(RuntimeError("deadline exceeded")) == "timeout"

    def test_error_classification_transient(self) -> None:
        """Test transient error classification."""
        adapter = OpenAISemanticAdapter(api_key="test-key")

        assert adapter._classify_error(RuntimeError("Connection refused")) == "transient"
        assert adapter._classify_error(RuntimeError("503 Service Unavailable")) == "transient"
        assert adapter._classify_error(RuntimeError("Network error")) == "transient"
        assert adapter._classify_error(RuntimeError("500 Internal Server Error")) == "transient"
        assert adapter._classify_error(RuntimeError("502 Bad Gateway")) == "transient"
        # Note: "504 Gateway Timeout" is classified as "timeout" not "transient"
        # because "timeout" patterns take precedence

    def test_error_classification_malformed(self) -> None:
        """Test malformed response error classification."""
        adapter = OpenAISemanticAdapter(api_key="test-key")

        try:
            json.loads("invalid json {")
        except json.JSONDecodeError as e:
            assert adapter._classify_error(e) == "malformed"

    def test_error_classification_fatal(self) -> None:
        """Test fatal error classification."""
        adapter = OpenAISemanticAdapter(api_key="test-key")

        assert adapter._classify_error(ValueError("Unknown error")) == "fatal"
        assert adapter._classify_error(TypeError("Type mismatch")) == "fatal"

    def test_should_retry_retryable_errors(self) -> None:
        """Test that retryable errors trigger retry."""
        adapter = OpenAISemanticAdapter(api_key="test-key", max_retries=3)

        assert adapter._should_retry("rate_limit", 0) is True
        assert adapter._should_retry("timeout", 0) is True
        assert adapter._should_retry("transient", 0) is True

    def test_should_retry_non_retryable_errors(self) -> None:
        """Test that non-retryable errors don't trigger retry."""
        adapter = OpenAISemanticAdapter(api_key="test-key", max_retries=3)

        assert adapter._should_retry("fatal", 0) is False
        assert adapter._should_retry("malformed", 0) is False

    def test_should_retry_max_retries_exceeded(self) -> None:
        """Test that max retries limit is respected."""
        adapter = OpenAISemanticAdapter(api_key="test-key", max_retries=3)

        assert adapter._should_retry("rate_limit", 3) is False
        assert adapter._should_retry("transient", 4) is False

    def test_backoff_calculation_exponential(self) -> None:
        """Test exponential backoff calculation."""
        adapter = OpenAISemanticAdapter(
            api_key="test-key",
            initial_backoff_ms=1000,
            max_backoff_ms=32000,
        )

        # Base delays (without jitter)
        delay_0 = adapter._calculate_backoff(0, "transient")
        assert 1.0 <= delay_0 <= 1.25  # 1000ms + up to 25% jitter

        delay_1 = adapter._calculate_backoff(1, "transient")
        assert 2.0 <= delay_1 <= 2.5  # 2000ms + up to 25% jitter

        delay_2 = adapter._calculate_backoff(2, "transient")
        assert 4.0 <= delay_2 <= 5.0  # 4000ms + up to 25% jitter

    def test_backoff_calculation_rate_limit_longer(self) -> None:
        """Test that rate limit errors get longer initial delay."""
        adapter = OpenAISemanticAdapter(
            api_key="test-key",
            initial_backoff_ms=1000,
            max_backoff_ms=32000,
        )

        delay_rate_limit = adapter._calculate_backoff(0, "rate_limit")
        assert delay_rate_limit >= 5.0  # At least 5 seconds for rate limit

    def test_backoff_calculation_max_cap(self) -> None:
        """Test that backoff is capped at max value."""
        adapter = OpenAISemanticAdapter(
            api_key="test-key",
            initial_backoff_ms=1000,
            max_backoff_ms=32000,
        )

        delay_max = adapter._calculate_backoff(10, "transient")
        assert delay_max <= 40.0  # 32000ms + 25% jitter max

    def test_retry_with_eventual_success(self, sample_chunk_context: ChunkContext) -> None:
        """Test that transient errors are retried and eventually succeed."""

        adapter = OpenAISemanticAdapter(
            api_key="test-key-123",
            max_retries=2,
            initial_backoff_ms=10,  # Very short for test
            max_backoff_ms=50,
        )

        call_count = 0

        # Mock that fails twice then succeeds
        mock_success_response = MagicMock()
        mock_success_response.text = json.dumps(
            {
                "topics": ["test"],
                "intent": None,
                "sentiment": None,
                "action_items": [],
                "risk_tags": [],
            }
        )
        mock_success_response.duration_ms = 100

        async def mock_complete_flaky(system: str, user: str):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Connection error: temporary failure")
            return mock_success_response

        adapter._guarded_provider.complete = mock_complete_flaky

        result = adapter.annotate_chunk("Test text", sample_chunk_context)

        # Verify retries happened and eventually succeeded
        assert call_count == 3  # 2 failures + 1 success
        assert isinstance(result, SemanticAnnotation)
        assert result.confidence == 0.85  # Success
        assert "test" in result.normalized.topics

        # Verify retry history is recorded
        assert "retry_history" in result.raw_model_output
        assert len(result.raw_model_output["retry_history"]) == 2


# -----------------------------------------------------------------------------
# Schema Validation Tests
# -----------------------------------------------------------------------------


class TestSchemaValidation:
    """Test that output conforms to semantic schema."""

    def test_normalized_annotation_structure(self) -> None:
        """Test NormalizedAnnotation has correct structure."""
        annotation = NormalizedAnnotation(
            topics=["pricing", "support"],
            intent="question",
            sentiment="neutral",
            action_items=[ActionItem(text="Call back", speaker_id="agent", confidence=0.9)],
            risk_tags=["churn_risk"],
        )

        assert isinstance(annotation.topics, list)
        assert isinstance(annotation.risk_tags, list)
        assert isinstance(annotation.action_items, list)
        assert annotation.intent in ["question", "objection", "statement", "request", None]
        assert annotation.sentiment in ["positive", "negative", "neutral", None]

    def test_semantic_annotation_to_dict(
        self, sample_valid_response: str, sample_chunk_context: ChunkContext
    ) -> None:
        """Test that SemanticAnnotation serializes correctly."""
        adapter = OpenAISemanticAdapter(api_key="test-key-123")

        mock_response = MagicMock()
        mock_response.text = sample_valid_response
        mock_response.duration_ms = 150

        async def mock_complete(system: str, user: str):
            return mock_response

        adapter._guarded_provider.complete = mock_complete

        result = adapter.annotate_chunk("Test", sample_chunk_context)
        result_dict = result.to_dict()

        # Verify required fields
        assert "schema_version" in result_dict
        assert "provider" in result_dict
        assert "model" in result_dict
        assert "normalized" in result_dict
        assert "confidence" in result_dict
        assert "latency_ms" in result_dict

        # Verify normalized structure
        normalized = result_dict["normalized"]
        assert "topics" in normalized
        assert "intent" in normalized
        assert "sentiment" in normalized
        assert "action_items" in normalized
        assert "risk_tags" in normalized

    def test_action_item_serialization(self) -> None:
        """Test ActionItem serialization."""
        action = ActionItem(
            text="Send report",
            speaker_id="agent",
            segment_ids=[1, 2],
            confidence=0.95,
        )

        d = action.to_dict()

        assert d["text"] == "Send report"
        assert d["speaker_id"] == "agent"
        assert d["segment_ids"] == [1, 2]
        assert d["confidence"] == 0.95

    def test_action_item_roundtrip(self) -> None:
        """Test ActionItem serialization roundtrip."""
        original = ActionItem(
            text="Follow up",
            speaker_id="rep",
            segment_ids=[5],
            pattern=None,
            confidence=0.88,
        )

        restored = ActionItem.from_dict(original.to_dict())

        assert restored.text == original.text
        assert restored.speaker_id == original.speaker_id
        assert restored.segment_ids == original.segment_ids
        assert restored.confidence == original.confidence


# -----------------------------------------------------------------------------
# Prompt Building Tests
# -----------------------------------------------------------------------------


class TestPromptBuilding:
    """Test prompt construction for cloud adapters."""

    def test_prompt_includes_context(self) -> None:
        """Test that prompt includes previous context."""
        adapter = OpenAISemanticAdapter(api_key="test-key", context_window=3)

        context = ChunkContext(
            speaker_id="customer",
            previous_chunks=["Hello", "How can I help?", "I have a question"],
            start=10.0,
            end=20.0,
        )

        system, user = adapter._build_prompt("Current text", context)

        # System prompt should contain extraction instructions
        assert "JSON" in system
        assert "topics" in system

        # User prompt should include context and current text
        assert "Current text" in user
        assert "customer" in user

    def test_prompt_handles_empty_context(self) -> None:
        """Test that prompt handles empty previous context."""
        adapter = OpenAISemanticAdapter(api_key="test-key")

        context = ChunkContext(
            speaker_id="unknown",
            previous_chunks=[],
            start=0.0,
            end=5.0,
        )

        system, user = adapter._build_prompt("Test text", context)

        assert "Test text" in user
        assert "(start of conversation)" in user or "Context" in user


# -----------------------------------------------------------------------------
# Integration Tests (Require API Keys)
# -----------------------------------------------------------------------------


@pytest.mark.external
class TestOpenAIIntegration:
    """Integration tests for OpenAI adapter with real API."""

    @requires_openai_key
    def test_real_openai_annotation(self) -> None:
        """Test real OpenAI API call."""
        adapter = create_adapter("openai")
        context = ChunkContext(speaker_id="customer", language="en")

        result = adapter.annotate_chunk(
            "I think the pricing is too high. Can you offer a discount?",
            context,
        )

        assert isinstance(result, SemanticAnnotation)
        assert result.provider == "openai"
        assert result.confidence > 0
        assert result.latency_ms > 0

        # Should detect pricing-related content
        topics_and_risks = result.normalized.topics + result.normalized.risk_tags
        assert any("pric" in t.lower() for t in topics_and_risks)

    @requires_openai_key
    def test_real_openai_health_check(self) -> None:
        """Test real OpenAI health check."""
        adapter = create_adapter("openai")

        health = adapter.health_check()

        assert isinstance(health, ProviderHealth)
        assert health.available is True
        assert health.latency_ms > 0


@pytest.mark.external
class TestAnthropicIntegration:
    """Integration tests for Anthropic adapter with real API."""

    @requires_anthropic_key
    def test_real_anthropic_annotation(self) -> None:
        """Test real Anthropic API call."""
        adapter = create_adapter("anthropic")
        context = ChunkContext(speaker_id="customer", language="en")

        result = adapter.annotate_chunk(
            "This is unacceptable! I want to speak to your manager.",
            context,
        )

        assert isinstance(result, SemanticAnnotation)
        assert result.provider == "anthropic"
        assert result.confidence > 0
        assert result.latency_ms > 0

        # Should detect escalation risk
        risks = result.normalized.risk_tags
        assert any("escalat" in r.lower() for r in risks) or len(risks) > 0

    @requires_anthropic_key
    def test_real_anthropic_health_check(self) -> None:
        """Test real Anthropic health check."""
        adapter = create_adapter("anthropic")

        health = adapter.health_check()

        assert isinstance(health, ProviderHealth)
        assert health.available is True
        assert health.latency_ms > 0


# -----------------------------------------------------------------------------
# Provider Comparison Tests
# -----------------------------------------------------------------------------


class TestProviderComparison:
    """Test that both providers behave consistently."""

    @pytest.mark.parametrize(
        "provider_class,provider_name",
        [
            (OpenAISemanticAdapter, "openai"),
            (AnthropicSemanticAdapter, "anthropic"),
        ],
    )
    def test_both_providers_return_semantic_annotation(
        self,
        provider_class: type,
        provider_name: str,
        sample_valid_response: str,
        sample_chunk_context: ChunkContext,
    ) -> None:
        """Test that both providers return SemanticAnnotation."""
        adapter = provider_class(api_key="test-key")

        mock_response = MagicMock()
        mock_response.text = sample_valid_response
        mock_response.duration_ms = 100

        async def mock_complete(system: str, user: str):
            return mock_response

        adapter._guarded_provider.complete = mock_complete

        result = adapter.annotate_chunk("Test", sample_chunk_context)

        assert isinstance(result, SemanticAnnotation)
        assert result.provider == provider_name

    @pytest.mark.parametrize(
        "provider_class",
        [OpenAISemanticAdapter, AnthropicSemanticAdapter],
    )
    def test_both_providers_handle_errors_gracefully(
        self,
        provider_class: type,
        sample_chunk_context: ChunkContext,
    ) -> None:
        """Test that both providers handle errors gracefully."""
        adapter = provider_class(api_key="test-key", max_retries=0)

        async def mock_complete_error(system: str, user: str):
            raise RuntimeError("API Error")

        adapter._guarded_provider.complete = mock_complete_error

        result = adapter.annotate_chunk("Test", sample_chunk_context)

        assert isinstance(result, SemanticAnnotation)
        assert result.confidence == 0.0
        assert "error" in result.raw_model_output

    @pytest.mark.parametrize(
        "provider_class",
        [OpenAISemanticAdapter, AnthropicSemanticAdapter],
    )
    def test_both_providers_have_health_check(
        self,
        provider_class: type,
    ) -> None:
        """Test that both providers implement health_check."""
        adapter = provider_class(api_key="test-key")

        health = adapter.health_check()

        assert isinstance(health, ProviderHealth)
        assert isinstance(health.available, bool)
        assert health.latency_ms >= 0
