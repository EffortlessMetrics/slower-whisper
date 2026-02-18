"""Contract tests for semantic adapters.

These tests verify that all semantic adapters (local, cloud, etc.)
conform to the SemanticAdapter protocol and produce consistent output.

Tests use golden files from tests/fixtures/semantic_golden/ to verify:
1. Local adapter produces expected output for known inputs
2. Missing providers gracefully skip (not crash)
3. Deterministic fields match golden output

See docs/SEMANTIC_BENCHMARK.md for evaluation methodology.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slower_whisper.pipeline.semantic_adapter import (
    SEMANTIC_SCHEMA_VERSION,
    ActionItem,
    ChunkContext,
    LocalKeywordAdapter,
    NoOpSemanticAdapter,
    NormalizedAnnotation,
    ProviderHealth,
    SemanticAdapter,
    SemanticAnnotation,
    create_adapter,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def golden_dir() -> Path:
    """Return path to golden files directory."""
    return Path(__file__).parent / "fixtures" / "semantic_golden"


@pytest.fixture
def load_golden_file(golden_dir: Path):
    """Factory fixture to load golden files by name."""

    def _load(filename: str) -> dict[str, Any]:
        filepath = golden_dir / filename
        if not filepath.exists():
            pytest.skip(f"Golden file not found: {filepath}")
        with open(filepath) as f:
            return json.load(f)

    return _load


# -----------------------------------------------------------------------------
# Protocol Compliance Tests
# -----------------------------------------------------------------------------


class TestSemanticAdapterProtocol:
    """Test that all adapters implement the SemanticAdapter protocol correctly."""

    def test_local_adapter_implements_protocol(self) -> None:
        """Test LocalKeywordAdapter satisfies SemanticAdapter protocol."""
        adapter = create_adapter("local")

        # Protocol requires these methods
        assert hasattr(adapter, "annotate_chunk")
        assert hasattr(adapter, "health_check")
        assert callable(adapter.annotate_chunk)
        assert callable(adapter.health_check)

    def test_noop_adapter_implements_protocol(self) -> None:
        """Test NoOpSemanticAdapter satisfies SemanticAdapter protocol."""
        adapter = create_adapter("noop")

        assert hasattr(adapter, "annotate_chunk")
        assert hasattr(adapter, "health_check")
        assert callable(adapter.annotate_chunk)
        assert callable(adapter.health_check)

    def test_local_adapter_type_annotation(self) -> None:
        """Verify LocalKeywordAdapter can be assigned to SemanticAdapter type."""
        adapter: SemanticAdapter = LocalKeywordAdapter()
        context = ChunkContext()

        result = adapter.annotate_chunk("Test text", context)
        assert isinstance(result, SemanticAnnotation)

        health = adapter.health_check()
        assert isinstance(health, ProviderHealth)

    def test_noop_adapter_type_annotation(self) -> None:
        """Verify NoOpSemanticAdapter can be assigned to SemanticAdapter type."""
        adapter: SemanticAdapter = NoOpSemanticAdapter()
        context = ChunkContext()

        result = adapter.annotate_chunk("Test text", context)
        assert isinstance(result, SemanticAnnotation)

        health = adapter.health_check()
        assert isinstance(health, ProviderHealth)

    def test_adapter_returns_correct_types(self) -> None:
        """Test that adapter methods return expected types."""
        for provider in ["local", "noop"]:
            adapter = create_adapter(provider)
            context = ChunkContext(speaker_id="test", language="en")

            result = adapter.annotate_chunk("Some text", context)

            assert isinstance(result.schema_version, str)
            assert isinstance(result.provider, str)
            assert isinstance(result.model, str)
            assert isinstance(result.normalized, NormalizedAnnotation)
            assert isinstance(result.confidence, float)
            assert isinstance(result.latency_ms, int)
            assert result.latency_ms >= 0
            assert 0.0 <= result.confidence <= 1.0


# -----------------------------------------------------------------------------
# Golden File Tests
# -----------------------------------------------------------------------------


class TestGoldenFiles:
    """Test adapters against golden files for deterministic output verification."""

    def test_golden_files_exist(self, golden_dir: Path) -> None:
        """Verify golden files directory exists and has files."""
        assert golden_dir.exists(), f"Golden files directory not found: {golden_dir}"

        golden_files = list(golden_dir.glob("*.json"))
        assert len(golden_files) >= 1, "No golden files found"

    def test_sales_call_pricing(self, load_golden_file) -> None:
        """Test local adapter against sales call pricing golden file."""
        golden = load_golden_file("sales_call_pricing.json")
        adapter = LocalKeywordAdapter()

        # Build context from golden file
        ctx_data = golden["input"]["context"]
        context = ChunkContext(
            speaker_id=ctx_data.get("speaker_id"),
            language=ctx_data.get("language", "en"),
            start=ctx_data.get("start", 0.0),
            end=ctx_data.get("end", 0.0),
        )

        result = adapter.annotate_chunk(golden["input"]["text"], context)
        expected = golden["expected_normalized"]
        allow_partial = golden.get("allow_partial_match", {})

        # Verify risk tags
        if allow_partial.get("risk_tags", False):
            # At least expected tags should be present
            for tag in expected["risk_tags"]:
                assert tag in result.normalized.risk_tags, (
                    f"Expected risk tag '{tag}' not found in {result.normalized.risk_tags}"
                )
        else:
            assert set(result.normalized.risk_tags) == set(expected["risk_tags"])

        # Verify action items
        if expected["action_items"]:
            if allow_partial.get("action_items", False):
                assert len(result.normalized.action_items) >= 1, "Expected at least one action item"
                # Check speaker_id matches if specified
                for expected_action in expected["action_items"]:
                    if expected_action.get("speaker_id"):
                        found = any(
                            a.speaker_id == expected_action["speaker_id"]
                            for a in result.normalized.action_items
                        )
                        assert found, (
                            f"No action item with speaker_id={expected_action['speaker_id']}"
                        )
            else:
                assert len(result.normalized.action_items) == len(expected["action_items"])

    def test_escalation_request(self, load_golden_file) -> None:
        """Test local adapter against escalation request golden file."""
        golden = load_golden_file("escalation_request.json")
        adapter = LocalKeywordAdapter()

        ctx_data = golden["input"]["context"]
        context = ChunkContext(
            speaker_id=ctx_data.get("speaker_id"),
            language=ctx_data.get("language", "en"),
            start=ctx_data.get("start", 0.0),
            end=ctx_data.get("end", 0.0),
        )

        result = adapter.annotate_chunk(golden["input"]["text"], context)
        expected = golden["expected_normalized"]

        # Verify both escalation and churn_risk are detected
        assert "escalation" in result.normalized.risk_tags
        assert "churn_risk" in result.normalized.risk_tags

        # Verify all expected risk tags are present
        for tag in expected["risk_tags"]:
            assert tag in result.normalized.risk_tags

    def test_action_commitment(self, load_golden_file) -> None:
        """Test local adapter against action commitment golden file."""
        golden = load_golden_file("action_commitment.json")
        adapter = LocalKeywordAdapter()

        ctx_data = golden["input"]["context"]
        context = ChunkContext(
            speaker_id=ctx_data.get("speaker_id"),
            language=ctx_data.get("language", "en"),
            start=ctx_data.get("start", 0.0),
            end=ctx_data.get("end", 0.0),
        )

        result = adapter.annotate_chunk(golden["input"]["text"], context)
        expected = golden["expected_normalized"]

        # Should detect action items
        assert len(result.normalized.action_items) >= 1, "Expected at least one action item"

        # Verify speaker_id is carried through
        for action in result.normalized.action_items:
            assert action.speaker_id == ctx_data.get("speaker_id")

        # No risk tags expected
        if not expected["risk_tags"]:
            assert len(result.normalized.risk_tags) == 0, (
                f"Expected no risk tags, got: {result.normalized.risk_tags}"
            )

    @pytest.mark.parametrize(
        "golden_filename",
        [
            "sales_call_pricing.json",
            "escalation_request.json",
            "action_commitment.json",
        ],
    )
    def test_all_golden_files_schema_valid(self, load_golden_file, golden_filename: str) -> None:
        """Verify all golden files produce valid schema-compliant output."""
        golden = load_golden_file(golden_filename)
        adapter = LocalKeywordAdapter()

        ctx_data = golden["input"]["context"]
        context = ChunkContext(
            speaker_id=ctx_data.get("speaker_id"),
            language=ctx_data.get("language", "en"),
            start=ctx_data.get("start", 0.0),
            end=ctx_data.get("end", 0.0),
        )

        result = adapter.annotate_chunk(golden["input"]["text"], context)

        # Verify schema version
        assert result.schema_version == SEMANTIC_SCHEMA_VERSION

        # Verify serialization roundtrip
        serialized = result.to_dict()
        restored = SemanticAnnotation.from_dict(serialized)

        assert restored.schema_version == result.schema_version
        assert restored.provider == result.provider
        assert restored.model == result.model
        assert restored.confidence == result.confidence
        assert restored.normalized.topics == result.normalized.topics
        assert restored.normalized.risk_tags == result.normalized.risk_tags


# -----------------------------------------------------------------------------
# Graceful Degradation Tests
# -----------------------------------------------------------------------------


class TestGracefulDegradation:
    """Test graceful degradation when dependencies are missing or providers unavailable."""

    def test_unknown_provider_raises_value_error(self) -> None:
        """Test that unknown provider raises ValueError with clear message."""
        with pytest.raises(ValueError, match="Unknown semantic provider"):
            create_adapter("nonexistent-provider")

        with pytest.raises(ValueError, match="Unknown semantic provider: xyz"):
            create_adapter("xyz")

    def test_noop_adapter_returns_empty_annotations(self) -> None:
        """Test that NoOpSemanticAdapter returns empty but valid annotations."""
        adapter = create_adapter("noop")
        context = ChunkContext(speaker_id="test", language="en")

        result = adapter.annotate_chunk("Test text with pricing and escalation keywords", context)

        assert isinstance(result, SemanticAnnotation)
        assert result.provider == "noop"
        assert result.model == "none"
        assert result.confidence == 0.0
        assert result.normalized.topics == []
        assert result.normalized.risk_tags == []
        assert result.normalized.action_items == []
        assert result.normalized.intent is None
        assert result.normalized.sentiment is None

    def test_local_adapter_handles_empty_text(self) -> None:
        """Test local adapter handles empty text gracefully."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext()

        result = adapter.annotate_chunk("", context)

        assert isinstance(result, SemanticAnnotation)
        assert result.normalized.topics == []
        assert result.normalized.risk_tags == []
        assert result.normalized.action_items == []

    def test_local_adapter_handles_whitespace_only(self) -> None:
        """Test local adapter handles whitespace-only text gracefully."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext()

        result = adapter.annotate_chunk("   \n\t   ", context)

        assert isinstance(result, SemanticAnnotation)
        assert result.normalized.risk_tags == []
        assert result.normalized.action_items == []

    def test_local_adapter_handles_none_speaker_id(self) -> None:
        """Test local adapter handles None speaker_id in context."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext(speaker_id=None)

        result = adapter.annotate_chunk("I'll send the report", context)

        assert len(result.normalized.action_items) >= 1
        # Speaker ID should be None, not crash
        assert result.normalized.action_items[0].speaker_id is None

    def test_adapter_health_check_always_succeeds_for_local(self) -> None:
        """Test that local adapter health check always returns available."""
        adapter = LocalKeywordAdapter()

        health = adapter.health_check()

        assert health.available is True
        assert health.error is None
        assert health.latency_ms == 0

    def test_adapter_health_check_noop(self) -> None:
        """Test that noop adapter health check returns available."""
        adapter = NoOpSemanticAdapter()

        health = adapter.health_check()

        assert health.available is True


# -----------------------------------------------------------------------------
# Data Serialization Contract Tests
# -----------------------------------------------------------------------------


class TestSerializationContracts:
    """Test that data classes serialize/deserialize correctly."""

    def test_action_item_roundtrip(self) -> None:
        """Test ActionItem serialization roundtrip preserves all fields."""
        original = ActionItem(
            text="Send the report",
            speaker_id="spk_0",
            segment_ids=[1, 2, 3],
            pattern=r"\bi'll send\b",
            confidence=0.95,
        )

        serialized = original.to_dict()
        restored = ActionItem.from_dict(serialized)

        assert restored.text == original.text
        assert restored.speaker_id == original.speaker_id
        assert restored.segment_ids == original.segment_ids
        assert restored.pattern == original.pattern
        assert restored.confidence == original.confidence

    def test_normalized_annotation_roundtrip(self) -> None:
        """Test NormalizedAnnotation serialization roundtrip."""
        original = NormalizedAnnotation(
            topics=["pricing", "contract"],
            intent="objection",
            sentiment="negative",
            action_items=[ActionItem(text="Follow up", speaker_id="agent")],
            risk_tags=["escalation", "churn_risk"],
        )

        serialized = original.to_dict()
        restored = NormalizedAnnotation.from_dict(serialized)

        assert restored.topics == original.topics
        assert restored.intent == original.intent
        assert restored.sentiment == original.sentiment
        assert len(restored.action_items) == len(original.action_items)
        assert restored.action_items[0].text == original.action_items[0].text
        assert restored.risk_tags == original.risk_tags

    def test_semantic_annotation_roundtrip(self) -> None:
        """Test SemanticAnnotation serialization roundtrip."""
        original = SemanticAnnotation(
            schema_version=SEMANTIC_SCHEMA_VERSION,
            provider="local",
            model="keyword-v1",
            normalized=NormalizedAnnotation(
                topics=["pricing"],
                risk_tags=["pricing"],
                action_items=[ActionItem(text="Send quote")],
            ),
            confidence=1.0,
            latency_ms=5,
            raw_model_output={"keywords": ["pricing"]},
        )

        serialized = original.to_dict()
        restored = SemanticAnnotation.from_dict(serialized)

        assert restored.schema_version == original.schema_version
        assert restored.provider == original.provider
        assert restored.model == original.model
        assert restored.confidence == original.confidence
        assert restored.latency_ms == original.latency_ms
        assert restored.raw_model_output == original.raw_model_output
        assert restored.normalized.topics == original.normalized.topics

    def test_chunk_context_roundtrip(self) -> None:
        """Test ChunkContext serialization roundtrip."""
        original = ChunkContext(
            speaker_id="spk_0",
            segment_ids=[0, 1, 2],
            start=10.5,
            end=25.0,
            previous_chunks=["Hello", "How are you?"],
            turn_id="turn_5",
            language="en",
        )

        serialized = original.to_dict()
        restored = ChunkContext.from_dict(serialized)

        assert restored.speaker_id == original.speaker_id
        assert restored.segment_ids == original.segment_ids
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.previous_chunks == original.previous_chunks
        assert restored.turn_id == original.turn_id
        assert restored.language == original.language


# -----------------------------------------------------------------------------
# Provider-Agnostic Behavior Tests
# -----------------------------------------------------------------------------


class TestProviderAgnosticBehavior:
    """Test that adapters follow consistent behavioral contracts."""

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_never_raises_on_valid_input(self, provider: str) -> None:
        """Test that adapters don't raise exceptions on valid input."""
        adapter = create_adapter(provider)
        context = ChunkContext(speaker_id="test", language="en")

        # These should all complete without raising
        adapter.annotate_chunk("Normal text", context)
        adapter.annotate_chunk("", context)
        adapter.annotate_chunk("A" * 10000, context)  # Long text
        adapter.annotate_chunk("Unicode: \u4e2d\u6587 \U0001f600", context)

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_records_latency(self, provider: str) -> None:
        """Test that adapters record latency in results."""
        adapter = create_adapter(provider)
        context = ChunkContext()

        result = adapter.annotate_chunk("Test text", context)

        # Latency should be non-negative integer
        assert isinstance(result.latency_ms, int)
        assert result.latency_ms >= 0

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_includes_provenance(self, provider: str) -> None:
        """Test that adapters include provenance information."""
        adapter = create_adapter(provider)
        context = ChunkContext()

        result = adapter.annotate_chunk("Test text", context)

        assert result.provider == provider
        assert result.model is not None
        assert len(result.model) > 0
        assert result.schema_version == SEMANTIC_SCHEMA_VERSION
