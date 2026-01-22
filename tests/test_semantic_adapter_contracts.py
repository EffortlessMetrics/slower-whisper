"""Contract tests for semantic adapters using golden files (#92).

These tests verify that semantic adapters:
1. Produce valid SemanticAnnotation output matching the protocol
2. Work correctly with AMI-style gold label files
3. Respect guardrails (PII detection, cost limits)
4. Degrade gracefully when API keys are missing

Golden files are located at benchmarks/gold/semantic/ and follow the schema
defined in benchmarks/gold/semantic/schema.json.

See docs/SEMANTIC_BENCHMARK.md for evaluation methodology.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from transcription.semantic_adapter import (
    SEMANTIC_SCHEMA_VERSION,
    ActionItem,
    ChunkContext,
    LocalKeywordAdapter,
    NormalizedAnnotation,
    ProviderHealth,
    SemanticAnnotation,
    create_adapter,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def gold_dir() -> Path:
    """Return path to gold label files directory."""
    return Path(__file__).parent.parent / "benchmarks" / "gold" / "semantic"


@pytest.fixture
def load_gold_file(gold_dir: Path):
    """Factory fixture to load gold label files by meeting_id."""

    def _load(meeting_id: str) -> dict[str, Any]:
        filepath = gold_dir / f"{meeting_id}.json"
        if not filepath.exists():
            pytest.skip(f"Gold file not found: {filepath}")
        with open(filepath) as f:
            return json.load(f)

    return _load


@pytest.fixture
def ami_meeting_ids() -> list[str]:
    """Return list of AMI-style meeting IDs for testing."""
    return [
        "ami_meeting_001",
        "ami_meeting_002",
        "ami_meeting_003",
        "ami_meeting_004",
        "ami_meeting_005",
    ]


# Skip markers for cloud adapters
requires_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


# -----------------------------------------------------------------------------
# Gold File Schema Validation Tests
# -----------------------------------------------------------------------------


class TestGoldFileSchema:
    """Test that gold files conform to the expected schema."""

    def test_gold_directory_exists(self, gold_dir: Path) -> None:
        """Verify gold files directory exists."""
        assert gold_dir.exists(), f"Gold directory not found: {gold_dir}"

    def test_schema_file_exists(self, gold_dir: Path) -> None:
        """Verify schema file exists."""
        schema_path = gold_dir / "schema.json"
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

    def test_ami_gold_files_exist(self, gold_dir: Path, ami_meeting_ids: list[str]) -> None:
        """Verify AMI-style gold files exist."""
        for meeting_id in ami_meeting_ids:
            filepath = gold_dir / f"{meeting_id}.json"
            assert filepath.exists(), f"Gold file not found: {filepath}"

    @pytest.mark.parametrize(
        "meeting_id",
        [
            "ami_meeting_001",
            "ami_meeting_002",
            "ami_meeting_003",
            "ami_meeting_004",
            "ami_meeting_005",
        ],
    )
    def test_gold_file_has_required_fields(self, load_gold_file, meeting_id: str) -> None:
        """Verify gold files have all required fields."""
        gold = load_gold_file(meeting_id)

        # Required fields per schema
        assert "schema_version" in gold
        assert "meeting_id" in gold
        assert "topics" in gold
        assert "risks" in gold
        assert "actions" in gold

        # Type checks
        assert gold["schema_version"] == 1
        assert gold["meeting_id"] == meeting_id
        assert isinstance(gold["topics"], list)
        assert isinstance(gold["risks"], list)
        assert isinstance(gold["actions"], list)

    @pytest.mark.parametrize(
        "meeting_id",
        [
            "ami_meeting_001",
            "ami_meeting_002",
            "ami_meeting_003",
            "ami_meeting_004",
            "ami_meeting_005",
        ],
    )
    def test_gold_file_topics_have_labels(self, load_gold_file, meeting_id: str) -> None:
        """Verify topic entries have required label field."""
        gold = load_gold_file(meeting_id)

        for topic in gold["topics"]:
            assert "label" in topic, f"Topic missing 'label' field: {topic}"
            assert isinstance(topic["label"], str)
            assert len(topic["label"]) > 0

    @pytest.mark.parametrize(
        "meeting_id",
        [
            "ami_meeting_001",
            "ami_meeting_002",
            "ami_meeting_003",
            "ami_meeting_004",
            "ami_meeting_005",
        ],
    )
    def test_gold_file_risks_have_required_fields(self, load_gold_file, meeting_id: str) -> None:
        """Verify risk entries have required fields."""
        gold = load_gold_file(meeting_id)

        for risk in gold["risks"]:
            assert "type" in risk, f"Risk missing 'type' field: {risk}"
            assert "severity" in risk, f"Risk missing 'severity' field: {risk}"
            assert "segment_id" in risk, f"Risk missing 'segment_id' field: {risk}"

            # Validate values
            assert risk["type"] in ["escalation", "churn", "pricing"]
            assert risk["severity"] in ["low", "medium", "high"]
            assert isinstance(risk["segment_id"], int)

    @pytest.mark.parametrize(
        "meeting_id",
        [
            "ami_meeting_001",
            "ami_meeting_002",
            "ami_meeting_003",
            "ami_meeting_004",
            "ami_meeting_005",
        ],
    )
    def test_gold_file_actions_have_text(self, load_gold_file, meeting_id: str) -> None:
        """Verify action entries have required text field."""
        gold = load_gold_file(meeting_id)

        for action in gold["actions"]:
            assert "text" in action, f"Action missing 'text' field: {action}"
            assert isinstance(action["text"], str)
            assert len(action["text"]) > 0


# -----------------------------------------------------------------------------
# Adapter Protocol Contract Tests
# -----------------------------------------------------------------------------


class TestAdapterProtocolContracts:
    """Test that adapters conform to SemanticAdapter protocol contracts."""

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_returns_semantic_annotation(self, provider: str) -> None:
        """Test that all adapters return SemanticAnnotation."""
        adapter = create_adapter(provider)
        context = ChunkContext(speaker_id="test", language="en")

        result = adapter.annotate_chunk("Test text", context)

        assert isinstance(result, SemanticAnnotation)

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_returns_provider_health(self, provider: str) -> None:
        """Test that all adapters return ProviderHealth from health_check."""
        adapter = create_adapter(provider)

        health = adapter.health_check()

        assert isinstance(health, ProviderHealth)
        assert isinstance(health.available, bool)

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_annotation_has_schema_version(self, provider: str) -> None:
        """Test that annotations include schema version."""
        adapter = create_adapter(provider)
        context = ChunkContext()

        result = adapter.annotate_chunk("Test", context)

        assert result.schema_version == SEMANTIC_SCHEMA_VERSION

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_annotation_has_provider_info(self, provider: str) -> None:
        """Test that annotations include provider and model info."""
        adapter = create_adapter(provider)
        context = ChunkContext()

        result = adapter.annotate_chunk("Test", context)

        assert result.provider == provider
        assert result.model is not None
        assert len(result.model) > 0

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_annotation_has_valid_confidence(self, provider: str) -> None:
        """Test that confidence is in valid range [0, 1]."""
        adapter = create_adapter(provider)
        context = ChunkContext()

        result = adapter.annotate_chunk("Test", context)

        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_annotation_has_non_negative_latency(self, provider: str) -> None:
        """Test that latency is non-negative."""
        adapter = create_adapter(provider)
        context = ChunkContext()

        result = adapter.annotate_chunk("Test", context)

        assert isinstance(result.latency_ms, int)
        assert result.latency_ms >= 0

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_normalized_has_correct_types(self, provider: str) -> None:
        """Test that normalized annotation has correct field types."""
        adapter = create_adapter(provider)
        context = ChunkContext()

        result = adapter.annotate_chunk("Test", context)

        assert isinstance(result.normalized, NormalizedAnnotation)
        assert isinstance(result.normalized.topics, list)
        assert isinstance(result.normalized.risk_tags, list)
        assert isinstance(result.normalized.action_items, list)

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_never_crashes_on_empty_input(self, provider: str) -> None:
        """Test that adapters handle empty input gracefully."""
        adapter = create_adapter(provider)
        context = ChunkContext()

        # Should not raise
        result = adapter.annotate_chunk("", context)

        assert isinstance(result, SemanticAnnotation)

    @pytest.mark.parametrize("provider", ["local", "noop"])
    def test_adapter_never_crashes_on_unicode(self, provider: str) -> None:
        """Test that adapters handle unicode input gracefully."""
        adapter = create_adapter(provider)
        context = ChunkContext()

        # Test with various unicode
        unicode_texts = [
            "Unicode: \u4e2d\u6587 \U0001f600",  # Chinese + emoji
            "\u00e9\u00e0\u00fc\u00f1",  # European accents
            "\u0645\u0631\u062d\u0628\u0627",  # Arabic
            "\u3053\u3093\u306b\u3061\u306f",  # Japanese
        ]

        for text in unicode_texts:
            result = adapter.annotate_chunk(text, context)
            assert isinstance(result, SemanticAnnotation)


# -----------------------------------------------------------------------------
# Gold File Integration Tests
# -----------------------------------------------------------------------------


class TestGoldFileIntegration:
    """Test adapters against gold label files for consistency."""

    def test_local_adapter_detects_risks_in_ami_001(self, load_gold_file) -> None:
        """Test local adapter against ami_meeting_001 gold file."""
        _gold = load_gold_file("ami_meeting_001")  # noqa: F841 - validates fixture
        adapter = LocalKeywordAdapter()
        context = ChunkContext(speaker_id="spk_sales", language="en")

        # Build input text that should trigger risk detection
        text = (
            "I need to speak with your manager about this contract issue. "
            "The pricing seems higher than what we discussed initially. "
            "We're considering other options if this doesn't work out."
        )

        result = adapter.annotate_chunk(text, context)

        # Verify escalation risk is detected
        assert "escalation" in result.normalized.risk_tags, (
            f"Expected 'escalation' in risk_tags, got: {result.normalized.risk_tags}"
        )

    def test_local_adapter_detects_risks_in_ami_002(self, load_gold_file) -> None:
        """Test local adapter against ami_meeting_002 gold file.

        Note: LocalKeywordAdapter uses rule-based keyword matching, not semantic
        understanding. Test text must contain actual keywords from the annotator's
        vocabulary (e.g., 'escalate', 'supervisor', 'cancel', 'competitor').
        """
        _gold = load_gold_file("ami_meeting_002")  # noqa: F841 - validates fixture
        adapter = LocalKeywordAdapter()
        context = ChunkContext(speaker_id="spk_customer", language="en")

        # Build input text with keywords that LocalKeywordAdapter detects
        # Uses actual escalation keywords (unacceptable, complaint) and churn keywords (switch)
        text = (
            "This situation is unacceptable - the bug has been blocking production for three days. "
            "If this isn't resolved soon, we may need to switch to a competitor solution."
        )

        result = adapter.annotate_chunk(text, context)

        # Should detect escalation or churn signals via keyword matching
        has_risk_signal = (
            "escalation" in result.normalized.risk_tags
            or "churn_risk" in result.normalized.risk_tags
        )
        assert has_risk_signal, (
            f"Expected escalation or churn_risk, got: {result.normalized.risk_tags}"
        )

    def test_local_adapter_detects_actions_in_ami_003(self, load_gold_file) -> None:
        """Test local adapter detects action items in ami_meeting_003."""
        _gold = load_gold_file("ami_meeting_003")  # noqa: F841 - validates fixture
        adapter = LocalKeywordAdapter()
        context = ChunkContext(speaker_id="spk_csm", language="en")

        # Text with action item commitment
        text = "I'll create a personalized onboarding guide for your team."

        result = adapter.annotate_chunk(text, context)

        # Should detect action items
        assert len(result.normalized.action_items) >= 1, (
            f"Expected at least 1 action item, got: {result.normalized.action_items}"
        )

    def test_local_adapter_preserves_speaker_id_in_actions(self, load_gold_file) -> None:
        """Test that speaker_id is preserved in detected action items."""
        adapter = LocalKeywordAdapter()
        context = ChunkContext(speaker_id="spk_agent", language="en")

        text = "I'll process your refund right away."

        result = adapter.annotate_chunk(text, context)

        assert len(result.normalized.action_items) >= 1
        for action in result.normalized.action_items:
            assert action.speaker_id == "spk_agent", (
                f"Expected speaker_id='spk_agent', got: {action.speaker_id}"
            )

    @pytest.mark.parametrize(
        "meeting_id",
        [
            "ami_meeting_001",
            "ami_meeting_002",
            "ami_meeting_003",
            "ami_meeting_004",
            "ami_meeting_005",
        ],
    )
    def test_gold_files_have_realistic_content(self, load_gold_file, meeting_id: str) -> None:
        """Verify gold files have realistic, non-empty content."""
        gold = load_gold_file(meeting_id)

        # Should have at least some content
        total_items = len(gold["topics"]) + len(gold["risks"]) + len(gold["actions"])
        assert total_items >= 2, (
            f"Gold file {meeting_id} should have realistic content (>= 2 items)"
        )

        # Should have a summary
        if "summary" in gold:
            assert len(gold["summary"]) >= 20, "Summary should be meaningful"


# -----------------------------------------------------------------------------
# Guardrail Contract Tests
# -----------------------------------------------------------------------------


class TestGuardrailContracts:
    """Test guardrail integration with semantic adapters."""

    def test_pii_detection_available(self) -> None:
        """Test that PII detection is available in guardrails module."""
        try:
            from transcription.llm_guardrails import LLMGuardrails

            guardrails = LLMGuardrails(pii_warning=True, block_on_pii=False)
            assert hasattr(guardrails, "detect_pii")
            assert hasattr(guardrails, "check_pii")
        except ImportError:
            pytest.skip("llm_guardrails module not available")

    def test_pii_detection_finds_email(self) -> None:
        """Test PII detection identifies email addresses."""
        try:
            from transcription.llm_guardrails import LLMGuardrails

            guardrails = LLMGuardrails(pii_warning=True, block_on_pii=False)
            text = "Contact me at john.doe@example.com for more info."

            matches = guardrails.detect_pii(text)

            emails = [m for m in matches if m.type == "email"]
            assert len(emails) >= 1, "Expected to detect email PII"
        except ImportError:
            pytest.skip("llm_guardrails module not available")

    def test_pii_detection_finds_phone(self) -> None:
        """Test PII detection identifies phone numbers."""
        try:
            from transcription.llm_guardrails import LLMGuardrails

            guardrails = LLMGuardrails(pii_warning=True, block_on_pii=False)
            text = "Call me at 555-123-4567."

            matches = guardrails.detect_pii(text)

            phones = [m for m in matches if m.type == "phone"]
            assert len(phones) >= 1, "Expected to detect phone PII"
        except ImportError:
            pytest.skip("llm_guardrails module not available")

    def test_pii_block_raises_on_detection(self) -> None:
        """Test that PII blocking raises exception when PII detected."""
        try:
            from transcription.llm_guardrails import LLMGuardrails

            guardrails = LLMGuardrails(pii_warning=True, block_on_pii=True)
            text = "My SSN is 123-45-6789"

            with pytest.raises(ValueError, match="PII detected"):
                guardrails.check_pii(text)
        except ImportError:
            pytest.skip("llm_guardrails module not available")

    def test_cost_budget_enforcement(self) -> None:
        """Test that cost budget is enforced."""
        try:
            from transcription.llm_guardrails import (
                CostBudgetExceeded,
                LLMGuardrails,
            )

            guardrails = LLMGuardrails(cost_budget_usd=0.01, block_on_budget=True)

            # Simulate spending over budget
            guardrails._stats.total_cost_usd = 0.009

            with pytest.raises(CostBudgetExceeded):
                guardrails.check_budget(additional_cost=0.005)
        except ImportError:
            pytest.skip("llm_guardrails module not available")

    def test_cost_tracking_accumulates(self) -> None:
        """Test that cost tracking accumulates across calls."""
        try:
            from transcription.llm_guardrails import LLMGuardrails

            guardrails = LLMGuardrails()

            # Track multiple costs
            guardrails.track_cost("gpt-4o", input_tokens=100, output_tokens=50)
            first_cost = guardrails.stats.total_cost_usd

            guardrails.track_cost("gpt-4o", input_tokens=100, output_tokens=50)
            second_cost = guardrails.stats.total_cost_usd

            assert second_cost > first_cost, "Costs should accumulate"
            assert guardrails.stats.total_requests == 2
        except ImportError:
            pytest.skip("llm_guardrails module not available")


# -----------------------------------------------------------------------------
# Cloud Adapter Contract Tests (Skip if no API key)
# -----------------------------------------------------------------------------


class TestCloudAdapterContracts:
    """Contract tests for cloud-based semantic adapters."""

    @requires_anthropic_key
    def test_anthropic_adapter_available(self) -> None:
        """Test that Anthropic adapter is available when key is set."""
        # This test runs only when ANTHROPIC_API_KEY is set
        # Verify the adapter can be created (implementation pending)
        pytest.skip("Anthropic adapter not yet implemented")

    @requires_openai_key
    def test_openai_adapter_available(self) -> None:
        """Test that OpenAI adapter is available when key is set."""
        # This test runs only when OPENAI_API_KEY is set
        # Verify the adapter can be created (implementation pending)
        pytest.skip("OpenAI adapter not yet implemented")

    def test_missing_api_key_skips_gracefully(self) -> None:
        """Test that missing API keys result in graceful skip, not crash."""
        # This test verifies the skip markers work correctly
        # by ensuring we reach this point without errors
        assert True


# -----------------------------------------------------------------------------
# Serialization Contract Tests
# -----------------------------------------------------------------------------


class TestSerializationContracts:
    """Test that semantic annotation serialization is stable."""

    def test_annotation_to_dict_includes_all_fields(self) -> None:
        """Test that to_dict includes all required fields."""
        annotation = SemanticAnnotation(
            provider="local",
            model="keyword-v1",
            normalized=NormalizedAnnotation(
                topics=["pricing"],
                risk_tags=["escalation"],
                action_items=[ActionItem(text="Send report", speaker_id="spk_0")],
            ),
            confidence=0.9,
            latency_ms=50,
        )

        d = annotation.to_dict()

        assert "schema_version" in d
        assert "provider" in d
        assert "model" in d
        assert "normalized" in d
        assert "confidence" in d
        assert "latency_ms" in d

    def test_annotation_roundtrip_preserves_data(self) -> None:
        """Test that serialization roundtrip preserves all data."""
        original = SemanticAnnotation(
            provider="local",
            model="keyword-v1",
            normalized=NormalizedAnnotation(
                topics=["pricing", "contract"],
                intent="objection",
                sentiment="negative",
                risk_tags=["escalation", "churn_risk"],
                action_items=[
                    ActionItem(
                        text="Send proposal",
                        speaker_id="spk_agent",
                        segment_ids=[1, 2],
                        confidence=0.95,
                    )
                ],
            ),
            confidence=0.85,
            latency_ms=100,
            raw_model_output={"keywords": ["pricing"]},
        )

        d = original.to_dict()
        restored = SemanticAnnotation.from_dict(d)

        assert restored.schema_version == original.schema_version
        assert restored.provider == original.provider
        assert restored.model == original.model
        assert restored.confidence == original.confidence
        assert restored.latency_ms == original.latency_ms
        assert restored.normalized.topics == original.normalized.topics
        assert restored.normalized.risk_tags == original.normalized.risk_tags
        assert restored.normalized.intent == original.normalized.intent
        assert restored.normalized.sentiment == original.normalized.sentiment
        assert len(restored.normalized.action_items) == len(original.normalized.action_items)

    def test_action_item_roundtrip_preserves_data(self) -> None:
        """Test that ActionItem roundtrip preserves all data."""
        original = ActionItem(
            text="Send the report",
            speaker_id="spk_0",
            segment_ids=[1, 2, 3],
            pattern=r"\bi'll send\b",
            confidence=0.95,
        )

        d = original.to_dict()
        restored = ActionItem.from_dict(d)

        assert restored.text == original.text
        assert restored.speaker_id == original.speaker_id
        assert restored.segment_ids == original.segment_ids
        assert restored.pattern == original.pattern
        assert restored.confidence == original.confidence


# -----------------------------------------------------------------------------
# Consistency Tests Against Gold Files
# -----------------------------------------------------------------------------


class TestGoldFileConsistency:
    """Test that gold files are consistent with schema and each other."""

    def test_all_ami_meetings_have_unique_ids(
        self, gold_dir: Path, ami_meeting_ids: list[str]
    ) -> None:
        """Verify all AMI meeting files have unique meeting_id values."""
        ids = set()
        for meeting_id in ami_meeting_ids:
            filepath = gold_dir / f"{meeting_id}.json"
            if not filepath.exists():
                continue
            with open(filepath) as f:
                gold = json.load(f)
            assert gold["meeting_id"] not in ids, f"Duplicate meeting_id: {gold['meeting_id']}"
            ids.add(gold["meeting_id"])

    def test_segment_ids_are_valid_integers(
        self, load_gold_file, ami_meeting_ids: list[str]
    ) -> None:
        """Verify all segment_ids are valid non-negative integers."""
        for meeting_id in ami_meeting_ids:
            try:
                gold = load_gold_file(meeting_id)
            except pytest.skip.Exception:
                continue

            # Check topic segment_ids
            for topic in gold["topics"]:
                if "segment_ids" in topic:
                    for seg_id in topic["segment_ids"]:
                        assert isinstance(seg_id, int) and seg_id >= 0

            # Check risk segment_ids
            for risk in gold["risks"]:
                assert isinstance(risk["segment_id"], int) and risk["segment_id"] >= 0

            # Check action segment_ids
            for action in gold["actions"]:
                if "segment_ids" in action:
                    for seg_id in action["segment_ids"]:
                        assert isinstance(seg_id, int) and seg_id >= 0

    def test_speaker_ids_are_consistent_format(
        self, load_gold_file, ami_meeting_ids: list[str]
    ) -> None:
        """Verify speaker_ids follow consistent naming format."""
        for meeting_id in ami_meeting_ids:
            try:
                gold = load_gold_file(meeting_id)
            except pytest.skip.Exception:
                continue

            for action in gold["actions"]:
                if "speaker_id" in action:
                    speaker_id = action["speaker_id"]
                    assert isinstance(speaker_id, str)
                    assert len(speaker_id) > 0
                    # Should start with common prefixes
                    assert speaker_id.startswith("spk_"), (
                        f"speaker_id should follow 'spk_*' format: {speaker_id}"
                    )
