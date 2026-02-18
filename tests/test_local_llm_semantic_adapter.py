"""Tests for LocalLLMSemanticAdapter (#89).

Tests cover:
1. Golden tests that assert stable output fields
2. Graceful degradation when deps missing
3. Tests with mocked model outputs
4. Import-time behavior validation
5. Output normalization and validation
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from slower_whisper.pipeline.semantic_adapter import (
    SEMANTIC_SCHEMA_VERSION,
    ActionItem,
    ChunkContext,
    LocalLLMSemanticAdapter,
    NormalizedAnnotation,
    ProviderHealth,
    SemanticAnnotation,
    create_adapter,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_llm_response_combined() -> str:
    """Mock LLM response with combined extraction (topics, risks, actions)."""
    return json.dumps(
        {
            "topics": [
                {"label": "pricing", "confidence": 0.9, "evidence": "asked about pricing"},
                {"label": "contract_terms", "confidence": 0.85, "evidence": "discussed terms"},
            ],
            "risks": [
                {
                    "type": "churn_risk",
                    "severity": "medium",
                    "confidence": 0.75,
                    "evidence": "mentioned competitor",
                },
            ],
            "actions": [
                {
                    "description": "Send updated quote",
                    "assignee": "agent",
                    "due": "tomorrow",
                    "priority": "high",
                    "verbatim": "I'll send you an updated quote",
                    "confidence": 0.95,
                },
            ],
        }
    )


@pytest.fixture
def mock_llm_response_topics_only() -> str:
    """Mock LLM response with topics only."""
    return json.dumps(
        {
            "topics": [
                {"label": "billing", "confidence": 0.92},
            ]
        }
    )


@pytest.fixture
def mock_llm_response_empty() -> str:
    """Mock LLM response with empty arrays."""
    return json.dumps({"topics": [], "risks": [], "actions": []})


@pytest.fixture
def mock_llm_response_markdown() -> str:
    """Mock LLM response wrapped in markdown code block."""
    return """```json
{
    "topics": [{"label": "support", "confidence": 0.8}],
    "risks": [],
    "actions": []
}
```"""


@pytest.fixture
def mock_llm_response_invalid_json() -> str:
    """Mock LLM response with invalid JSON."""
    return "This is not JSON, just some random text."


@pytest.fixture
def mock_llm_response_partial_json() -> str:
    """Mock LLM response with incomplete JSON."""
    return '{"topics": [{"label": "test"'  # Missing closing braces


@pytest.fixture
def sample_context() -> ChunkContext:
    """Sample ChunkContext for testing."""
    return ChunkContext(
        speaker_id="spk_agent",
        segment_ids=[0, 1, 2],
        start=0.0,
        end=30.0,
        previous_chunks=["Hello, how can I help?", "I'd like to know about pricing."],
        turn_id="turn_1",
        language="en",
    )


# -----------------------------------------------------------------------------
# Import-time behavior tests
# -----------------------------------------------------------------------------


class TestImportBehavior:
    """Test that imports work correctly without optional deps."""

    def test_module_imports_without_torch(self) -> None:
        """Test that semantic_adapter imports without torch installed."""
        # This test passes if we reach here - import already succeeded at module level
        from slower_whisper.pipeline import semantic_adapter

        assert hasattr(semantic_adapter, "LocalLLMSemanticAdapter")

    def test_module_imports_without_transformers(self) -> None:
        """Test that semantic_adapter imports without transformers installed."""
        from slower_whisper.pipeline.semantic_adapter import LocalLLMSemanticAdapter

        # Should be importable even if transformers not installed
        assert LocalLLMSemanticAdapter is not None

    def test_adapter_creation_without_deps(self) -> None:
        """Test that adapter can be created without optional deps."""
        # Creating the adapter should NOT trigger model loading
        adapter = LocalLLMSemanticAdapter(model="test-model")
        assert adapter is not None
        assert adapter.model == "test-model"

    def test_local_llm_provider_module_imports(self) -> None:
        """Test that local_llm_provider module imports cleanly."""
        from slower_whisper.pipeline import local_llm_provider

        # Should have the key exports
        assert hasattr(local_llm_provider, "LocalLLMProvider")
        assert hasattr(local_llm_provider, "is_available")
        assert hasattr(local_llm_provider, "get_availability_status")
        assert hasattr(local_llm_provider, "MockLocalLLMProvider")


# -----------------------------------------------------------------------------
# Graceful degradation tests
# -----------------------------------------------------------------------------


class TestGracefulDegradation:
    """Test graceful degradation when optional deps are missing."""

    def test_health_check_when_deps_missing(self) -> None:
        """Test health_check returns unavailable when deps missing."""
        adapter = LocalLLMSemanticAdapter()

        # Force provider to be unavailable
        adapter._provider_available = False

        health = adapter.health_check()

        # Should indicate unavailable gracefully
        assert isinstance(health, ProviderHealth)
        assert health.available is False
        assert health.error is not None
        assert "transformers" in health.error.lower() or "torch" in health.error.lower()

    def test_annotate_chunk_returns_empty_when_deps_missing(self) -> None:
        """Test annotate_chunk returns low-confidence annotation when deps missing."""
        adapter = LocalLLMSemanticAdapter()
        context = ChunkContext(speaker_id="test")

        # Force provider to be unavailable
        adapter._provider_available = False

        result = adapter.annotate_chunk("Test text", context)

        assert isinstance(result, SemanticAnnotation)
        assert result.provider == "local-llm"
        assert result.confidence == 0.0
        assert result.normalized.topics == []
        assert result.normalized.risk_tags == []
        assert result.normalized.action_items == []
        assert "error" in result.raw_model_output

    def test_factory_creates_adapter_without_crash(self) -> None:
        """Test create_adapter('local-llm') doesn't crash without deps."""
        adapter = create_adapter("local-llm")
        assert isinstance(adapter, LocalLLMSemanticAdapter)


# -----------------------------------------------------------------------------
# Golden output tests (with mocked responses)
# -----------------------------------------------------------------------------


class TestGoldenOutputs:
    """Test that adapter produces stable, predictable output structure."""

    def test_combined_extraction_output_structure(
        self,
        mock_llm_response_combined: str,
        sample_context: ChunkContext,
    ) -> None:
        """Test that combined extraction produces correct output structure."""
        adapter = LocalLLMSemanticAdapter()

        # Mock the provider
        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=mock_llm_response_combined)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk(
            "What's the pricing? We're also looking at competitors.",
            sample_context,
        )

        # Verify structure
        assert isinstance(result, SemanticAnnotation)
        assert result.schema_version == SEMANTIC_SCHEMA_VERSION
        assert result.provider == "local-llm"
        assert result.model == adapter.model

        # Verify normalized annotation
        assert isinstance(result.normalized, NormalizedAnnotation)
        assert "pricing" in result.normalized.topics
        assert "contract_terms" in result.normalized.topics
        assert "churn_risk" in result.normalized.risk_tags

        # Verify action items
        assert len(result.normalized.action_items) == 1
        action = result.normalized.action_items[0]
        assert isinstance(action, ActionItem)
        assert action.text == "Send updated quote"
        assert action.speaker_id == "agent"
        assert action.confidence == 0.95

        # Verify metadata
        assert result.latency_ms >= 0
        assert 0.0 <= result.confidence <= 1.0

    def test_output_is_serializable(
        self,
        mock_llm_response_combined: str,
        sample_context: ChunkContext,
    ) -> None:
        """Test that output can be serialized to JSON."""
        adapter = LocalLLMSemanticAdapter()

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=mock_llm_response_combined)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        # Should serialize without error
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

        # Should round-trip correctly
        restored = SemanticAnnotation.from_dict(json.loads(json_str))
        assert restored.provider == result.provider
        assert restored.normalized.topics == result.normalized.topics

    def test_topics_are_normalized(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that topic labels are normalized (lowercase, underscores)."""
        adapter = LocalLLMSemanticAdapter()

        # Response with mixed-case topics
        response = json.dumps(
            {
                "topics": [
                    {"label": "BILLING ISSUE", "confidence": 0.9},
                    {"label": "Technical Support", "confidence": 0.8},
                ],
                "risks": [],
                "actions": [],
            }
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=response)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        # Should be normalized to lowercase with underscores
        assert "billing_issue" in result.normalized.topics
        assert "technical_support" in result.normalized.topics

    def test_risks_are_validated(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that risk types are validated and normalized."""
        adapter = LocalLLMSemanticAdapter()

        # Response with invalid risk type
        response = json.dumps(
            {
                "topics": [],
                "risks": [
                    {"type": "UNKNOWN_RISK_TYPE", "severity": "extreme", "confidence": 0.9},
                    {"type": "escalation", "severity": "high", "confidence": 0.8},
                ],
                "actions": [],
            }
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=response)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        # Unknown type should be normalized to "other"
        assert "other" in result.normalized.risk_tags
        # Valid type should pass through
        assert "escalation" in result.normalized.risk_tags

    def test_confidence_is_clamped(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that confidence values are clamped to [0, 1]."""
        adapter = LocalLLMSemanticAdapter()

        # Response with out-of-range confidence
        response = json.dumps(
            {
                "topics": [
                    {"label": "test", "confidence": 1.5},  # Over 1.0
                    {"label": "test2", "confidence": -0.5},  # Negative
                ],
                "risks": [],
                "actions": [],
            }
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=response)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        # Overall confidence should be valid
        assert 0.0 <= result.confidence <= 1.0


# -----------------------------------------------------------------------------
# JSON parsing tests
# -----------------------------------------------------------------------------


class TestJsonParsing:
    """Test JSON extraction and parsing from LLM responses."""

    def test_parses_plain_json(
        self,
        mock_llm_response_combined: str,
        sample_context: ChunkContext,
    ) -> None:
        """Test parsing plain JSON response."""
        adapter = LocalLLMSemanticAdapter()

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=mock_llm_response_combined)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert len(result.normalized.topics) > 0

    def test_parses_markdown_json(
        self,
        mock_llm_response_markdown: str,
        sample_context: ChunkContext,
    ) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        adapter = LocalLLMSemanticAdapter()

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=mock_llm_response_markdown)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert "support" in result.normalized.topics

    def test_handles_invalid_json_gracefully(
        self,
        mock_llm_response_invalid_json: str,
        sample_context: ChunkContext,
    ) -> None:
        """Test that invalid JSON returns empty annotation, not crash."""
        adapter = LocalLLMSemanticAdapter()

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=mock_llm_response_invalid_json)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        # Should return empty results, not crash
        assert isinstance(result, SemanticAnnotation)
        assert result.normalized.topics == []
        assert result.normalized.risk_tags == []
        assert result.normalized.action_items == []

    def test_handles_partial_json_gracefully(
        self,
        mock_llm_response_partial_json: str,
        sample_context: ChunkContext,
    ) -> None:
        """Test that partial/truncated JSON is handled gracefully."""
        adapter = LocalLLMSemanticAdapter()

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=mock_llm_response_partial_json)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        # Should return empty results, not crash
        assert isinstance(result, SemanticAnnotation)

    def test_handles_empty_response(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that empty response is handled gracefully."""
        adapter = LocalLLMSemanticAdapter()

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text="")
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert isinstance(result, SemanticAnnotation)
        assert result.normalized.topics == []


# -----------------------------------------------------------------------------
# Extraction mode tests
# -----------------------------------------------------------------------------


class TestExtractionModes:
    """Test different extraction modes (topics, risks, actions, combined)."""

    def test_combined_mode_uses_combined_prompt(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that combined mode uses the combined extraction prompt."""
        adapter = LocalLLMSemanticAdapter(extraction_mode="combined")

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(
            text='{"topics": [], "risks": [], "actions": []}'
        )
        adapter._provider = mock_provider
        adapter._provider_available = True

        adapter.annotate_chunk("Test text", sample_context)

        # Verify generate was called
        mock_provider.generate.assert_called_once()
        call_args = mock_provider.generate.call_args
        prompt = call_args[0][0]  # First positional arg is the prompt

        # Combined prompt should mention all three
        assert "TOPICS" in prompt or "topics" in prompt.lower()

    def test_topics_mode(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test topics-only extraction mode."""
        adapter = LocalLLMSemanticAdapter(extraction_mode="topics")

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(
            text='{"topics": [{"label": "billing", "confidence": 0.9}]}'
        )
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert "billing" in result.normalized.topics

    def test_risks_mode(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test risks-only extraction mode."""
        adapter = LocalLLMSemanticAdapter(extraction_mode="risks")

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(
            text='{"risks": [{"type": "escalation", "severity": "high", "confidence": 0.8}]}'
        )
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert "escalation" in result.normalized.risk_tags

    def test_actions_mode(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test actions-only extraction mode."""
        adapter = LocalLLMSemanticAdapter(extraction_mode="actions")

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(
            text='{"actions": [{"description": "Call back customer", "confidence": 0.9}]}'
        )
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert len(result.normalized.action_items) == 1
        assert result.normalized.action_items[0].text == "Call back customer"


# -----------------------------------------------------------------------------
# Intent and sentiment detection tests
# -----------------------------------------------------------------------------


class TestIntentSentimentDetection:
    """Test intent and sentiment detection from parsed responses."""

    def test_escalation_sets_request_intent(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that escalation risk sets intent to 'request'."""
        adapter = LocalLLMSemanticAdapter()

        response = json.dumps(
            {
                "topics": [],
                "risks": [{"type": "escalation", "severity": "high", "confidence": 0.9}],
                "actions": [],
            }
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=response)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert result.normalized.intent == "request"

    def test_pricing_objection_sets_objection_intent(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that pricing_objection risk sets intent to 'objection'."""
        adapter = LocalLLMSemanticAdapter()

        response = json.dumps(
            {
                "topics": [],
                "risks": [{"type": "pricing_objection", "severity": "medium", "confidence": 0.8}],
                "actions": [],
            }
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=response)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert result.normalized.intent == "objection"

    def test_actions_set_statement_intent(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that actions set intent to 'statement'."""
        adapter = LocalLLMSemanticAdapter()

        response = json.dumps(
            {
                "topics": [],
                "risks": [],
                "actions": [{"description": "Send report", "confidence": 0.9}],
            }
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=response)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert result.normalized.intent == "statement"

    def test_high_severity_negative_risks_set_negative_sentiment(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that high severity negative risks set sentiment to negative."""
        adapter = LocalLLMSemanticAdapter()

        response = json.dumps(
            {
                "topics": [],
                "risks": [{"type": "customer_frustration", "severity": "high", "confidence": 0.9}],
                "actions": [],
            }
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(text=response)
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", sample_context)

        assert result.normalized.sentiment == "negative"


# -----------------------------------------------------------------------------
# Protocol compliance tests
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Test that LocalLLMSemanticAdapter satisfies SemanticAdapter protocol."""

    def test_has_annotate_chunk_method(self) -> None:
        """Test that adapter has annotate_chunk method."""
        adapter = LocalLLMSemanticAdapter()
        assert hasattr(adapter, "annotate_chunk")
        assert callable(adapter.annotate_chunk)

    def test_has_health_check_method(self) -> None:
        """Test that adapter has health_check method."""
        adapter = LocalLLMSemanticAdapter()
        assert hasattr(adapter, "health_check")
        assert callable(adapter.health_check)

    def test_annotate_chunk_returns_semantic_annotation(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that annotate_chunk returns SemanticAnnotation."""
        adapter = LocalLLMSemanticAdapter()
        # Force unavailable to get predictable behavior
        adapter._provider_available = False

        result = adapter.annotate_chunk("Test", sample_context)

        assert isinstance(result, SemanticAnnotation)

    def test_health_check_returns_provider_health(self) -> None:
        """Test that health_check returns ProviderHealth."""
        adapter = LocalLLMSemanticAdapter()
        # Force unavailable to get predictable behavior
        adapter._provider_available = False

        result = adapter.health_check()

        assert isinstance(result, ProviderHealth)

    def test_factory_creates_local_llm_adapter(self) -> None:
        """Test that create_adapter('local-llm') returns LocalLLMSemanticAdapter."""
        adapter = create_adapter("local-llm")
        assert isinstance(adapter, LocalLLMSemanticAdapter)

    def test_factory_passes_kwargs(self) -> None:
        """Test that factory passes kwargs to adapter."""
        adapter = create_adapter(
            "local-llm",
            model="test-model",
            temperature=0.5,
            max_tokens=512,
            extraction_mode="topics",
        )

        assert adapter.model == "test-model"
        assert adapter.temperature == 0.5
        assert adapter.max_tokens == 512
        assert adapter.extraction_mode == "topics"


# -----------------------------------------------------------------------------
# Context handling tests
# -----------------------------------------------------------------------------


class TestContextHandling:
    """Test handling of ChunkContext in prompts."""

    def test_speaker_id_included_in_prompt(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that speaker_id is included in prompt."""
        adapter = LocalLLMSemanticAdapter()

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(
            text='{"topics": [], "risks": [], "actions": []}'
        )
        adapter._provider = mock_provider
        adapter._provider_available = True

        adapter.annotate_chunk("Test", sample_context)

        call_args = mock_provider.generate.call_args
        prompt = call_args[0][0]

        assert "spk_agent" in prompt

    def test_previous_chunks_included_in_prompt(
        self,
        sample_context: ChunkContext,
    ) -> None:
        """Test that previous chunks are included in prompt."""
        adapter = LocalLLMSemanticAdapter()

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(
            text='{"topics": [], "risks": [], "actions": []}'
        )
        adapter._provider = mock_provider
        adapter._provider_available = True

        adapter.annotate_chunk("Test", sample_context)

        call_args = mock_provider.generate.call_args
        prompt = call_args[0][0]

        # Previous chunks should be in prompt
        assert "pricing" in prompt.lower()

    def test_handles_missing_speaker_id(self) -> None:
        """Test that missing speaker_id is handled gracefully."""
        adapter = LocalLLMSemanticAdapter()
        context = ChunkContext()  # No speaker_id

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(
            text='{"topics": [], "risks": [], "actions": []}'
        )
        adapter._provider = mock_provider
        adapter._provider_available = True

        # Should not crash
        result = adapter.annotate_chunk("Test", context)
        assert isinstance(result, SemanticAnnotation)

    def test_handles_empty_previous_chunks(self) -> None:
        """Test that empty previous_chunks is handled gracefully."""
        adapter = LocalLLMSemanticAdapter()
        context = ChunkContext(speaker_id="test", previous_chunks=[])

        mock_provider = MagicMock()
        mock_provider.generate.return_value = MagicMock(
            text='{"topics": [], "risks": [], "actions": []}'
        )
        adapter._provider = mock_provider
        adapter._provider_available = True

        result = adapter.annotate_chunk("Test", context)
        assert isinstance(result, SemanticAnnotation)


# -----------------------------------------------------------------------------
# Mock provider tests
# -----------------------------------------------------------------------------


class TestMockLocalLLMProvider:
    """Test MockLocalLLMProvider for testing scenarios."""

    def test_mock_provider_returns_default_response(self) -> None:
        """Test that mock provider returns default response."""
        from slower_whisper.pipeline.local_llm_provider import MockLocalLLMProvider

        provider = MockLocalLLMProvider()
        response = provider.generate("Test prompt")

        assert response.text is not None
        assert response.model == "mock-local-llm"
        assert response.duration_ms >= 0

    def test_mock_provider_matches_keywords(self) -> None:
        """Test that mock provider matches keywords in prompt."""
        from slower_whisper.pipeline.local_llm_provider import MockLocalLLMProvider

        provider = MockLocalLLMProvider(
            responses={
                "pricing": '{"topics": [{"label": "pricing", "confidence": 1.0}]}',
            }
        )

        response = provider.generate("Tell me about pricing")

        assert "pricing" in response.text

    def test_mock_provider_tracks_calls(self) -> None:
        """Test that mock provider tracks call count and history."""
        from slower_whisper.pipeline.local_llm_provider import MockLocalLLMProvider

        provider = MockLocalLLMProvider()

        provider.generate("First call", "System 1")
        provider.generate("Second call", "System 2")

        assert provider.call_count == 2
        assert len(provider.calls) == 2
        assert provider.calls[0] == ("System 1", "First call")

    def test_mock_provider_is_always_loaded(self) -> None:
        """Test that mock provider reports as loaded."""
        from slower_whisper.pipeline.local_llm_provider import MockLocalLLMProvider

        provider = MockLocalLLMProvider()

        assert provider.is_loaded() is True
        assert provider.get_device() == "cpu"


# -----------------------------------------------------------------------------
# Local LLM provider availability tests
# -----------------------------------------------------------------------------


class TestLocalLLMProviderAvailability:
    """Test availability checking for local LLM provider."""

    def test_get_availability_status_returns_dict(self) -> None:
        """Test that get_availability_status returns correct structure."""
        from slower_whisper.pipeline.local_llm_provider import get_availability_status

        status = get_availability_status()

        assert isinstance(status, dict)
        assert "torch" in status
        assert "transformers" in status
        assert "available" in status
        assert isinstance(status["torch"], bool)
        assert isinstance(status["transformers"], bool)
        assert isinstance(status["available"], bool)

    def test_is_available_returns_bool(self) -> None:
        """Test that is_available returns a boolean."""
        from slower_whisper.pipeline.local_llm_provider import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_availability_is_consistent(self) -> None:
        """Test that availability checks are consistent."""
        from slower_whisper.pipeline.local_llm_provider import get_availability_status, is_available

        status = get_availability_status()
        available = is_available()

        # is_available should match the computed availability
        expected = status["torch"] and status["transformers"]
        assert available == expected
