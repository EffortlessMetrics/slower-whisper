"""
Tests for LLM guardrails.

Test coverage:
- PII detection (email, phone, SSN, credit card)
- Rate limiting with token bucket algorithm
- Cost tracking and budget enforcement
- GuardedLLMProvider wrapper
- Timeout handling
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from transcription.llm_guardrails import (
    CostBudgetExceeded,
    GuardedLLMProvider,
    LLMGuardrails,
    RateLimitExceeded,
    RequestTimeout,
    create_guarded_provider,
)


class TestPIIDetection:
    """Tests for PII detection functionality."""

    def test_detect_email(self) -> None:
        """Test email detection."""
        guardrails = LLMGuardrails()
        text = "Contact me at john.doe@example.com for more info."

        matches = guardrails.detect_pii(text)

        assert len(matches) == 1
        assert matches[0].type == "email"
        assert matches[0].masked == "j***@example.com"

    def test_detect_multiple_emails(self) -> None:
        """Test multiple email detection."""
        guardrails = LLMGuardrails()
        text = "Email alice@test.org or bob@company.io"

        matches = guardrails.detect_pii(text)

        emails = [m for m in matches if m.type == "email"]
        assert len(emails) == 2

    def test_detect_phone_us_format(self) -> None:
        """Test US phone number detection in various formats."""
        guardrails = LLMGuardrails()

        # Test different formats
        formats = [
            "Call 555-123-4567",
            "Phone: (555) 123-4567",
            "Tel: 555.123.4567",
            "Number: 5551234567",
            "US: +1-555-123-4567",
        ]

        for text in formats:
            matches = guardrails.detect_pii(text)
            phones = [m for m in matches if m.type == "phone"]
            assert len(phones) == 1, f"Failed for format: {text}"
            assert phones[0].masked.endswith("4567")

    def test_detect_ssn(self) -> None:
        """Test SSN detection."""
        guardrails = LLMGuardrails()
        text = "SSN: 123-45-6789"

        matches = guardrails.detect_pii(text)

        ssns = [m for m in matches if m.type == "ssn"]
        assert len(ssns) == 1
        assert ssns[0].masked == "***-**-6789"

    def test_detect_credit_card(self) -> None:
        """Test credit card detection."""
        guardrails = LLMGuardrails()

        # Test different formats
        formats = [
            ("Card: 4111-1111-1111-1111", "1111"),  # Visa format
            ("CC: 4111 1111 1111 1111", "1111"),  # Spaced
            ("Number: 4111111111111111", "1111"),  # No separators
        ]

        for text, expected_last4 in formats:
            matches = guardrails.detect_pii(text)
            cards = [m for m in matches if m.type == "credit_card"]
            assert len(cards) == 1, f"Failed for format: {text}"
            assert cards[0].masked.endswith(expected_last4)

    def test_no_pii_in_clean_text(self) -> None:
        """Test that clean text has no PII matches."""
        guardrails = LLMGuardrails()
        text = "This is a regular conversation about weather and sports."

        matches = guardrails.detect_pii(text)

        assert len(matches) == 0

    def test_check_pii_warns(self, caplog) -> None:
        """Test that check_pii logs warnings."""
        guardrails = LLMGuardrails(pii_warning=True, block_on_pii=False)
        text = "Email me at test@example.com"

        import logging

        with caplog.at_level(logging.WARNING):
            matches = guardrails.check_pii(text)

        assert len(matches) == 1
        assert guardrails.stats.pii_warnings_count == 1
        assert "PII detected" in caplog.text

    def test_check_pii_blocks_when_configured(self) -> None:
        """Test that check_pii raises when block_on_pii is True."""
        guardrails = LLMGuardrails(pii_warning=True, block_on_pii=True)
        text = "My SSN is 123-45-6789"

        with pytest.raises(ValueError, match="PII detected"):
            guardrails.check_pii(text)

    def test_pii_disabled(self) -> None:
        """Test that PII detection is skipped when disabled."""
        guardrails = LLMGuardrails(pii_warning=False)
        text = "Email: test@example.com, Phone: 555-123-4567"

        matches = guardrails.check_pii(text)

        assert len(matches) == 0


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_burst(self) -> None:
        """Test that rate limiter allows burst up to limit."""
        guardrails = LLMGuardrails(rate_limit_rpm=10)

        # Should allow 10 requests immediately
        for _ in range(10):
            wait = await guardrails.check_rate_limit(block=False)
            assert wait is None

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess(self) -> None:
        """Test that rate limiter blocks when exceeded."""
        guardrails = LLMGuardrails(rate_limit_rpm=5)

        # Exhaust the bucket
        for _ in range(5):
            await guardrails.check_rate_limit(block=False)

        # Next request should be blocked
        with pytest.raises(RateLimitExceeded) as exc_info:
            await guardrails.check_rate_limit(block=False)

        assert exc_info.value.wait_seconds > 0
        assert guardrails.stats.rate_limited_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_refills(self) -> None:
        """Test that token bucket refills over time."""
        guardrails = LLMGuardrails(rate_limit_rpm=60)  # 1 per second

        # Exhaust bucket
        for _ in range(60):
            await guardrails.check_rate_limit(block=False)

        # Wait for refill (simulate time passing)
        # 60 RPM = 1 per second, so waiting 1.1s should refill ~1 token
        await asyncio.sleep(1.1)

        # Should be able to make at least one more request
        wait = await guardrails.check_rate_limit(block=False)
        assert wait is None  # Should succeed after refill


class TestCostTracking:
    """Tests for cost tracking functionality."""

    def test_track_cost_openai_model(self) -> None:
        """Test cost tracking for OpenAI models."""
        guardrails = LLMGuardrails()

        cost = guardrails.track_cost("gpt-4o", input_tokens=1000, output_tokens=500)

        # gpt-4o: $2.50/1M input, $10.00/1M output
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(cost - expected) < 0.0001
        assert guardrails.stats.total_cost_usd == cost
        assert guardrails.stats.total_tokens_used == 1500
        assert guardrails.stats.total_requests == 1

    def test_track_cost_anthropic_model(self) -> None:
        """Test cost tracking for Anthropic models."""
        guardrails = LLMGuardrails()

        cost = guardrails.track_cost(
            "claude-sonnet-4-20250514", input_tokens=2000, output_tokens=1000
        )

        # claude-sonnet: $3.00/1M input, $15.00/1M output
        expected = (2000 / 1_000_000) * 3.00 + (1000 / 1_000_000) * 15.00
        assert abs(cost - expected) < 0.0001

    def test_track_cost_unknown_model_uses_default(self) -> None:
        """Test that unknown models use default pricing."""
        guardrails = LLMGuardrails()

        cost = guardrails.track_cost("unknown-model-xyz", input_tokens=1000, output_tokens=500)

        # Default: $5.00/1M input, $15.00/1M output
        expected = (1000 / 1_000_000) * 5.00 + (500 / 1_000_000) * 15.00
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_no_tracking(self) -> None:
        """Test that estimate_cost doesn't update stats."""
        guardrails = LLMGuardrails()

        cost = guardrails.estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)

        assert cost > 0
        assert guardrails.stats.total_cost_usd == 0.0
        assert guardrails.stats.total_tokens_used == 0
        assert guardrails.stats.total_requests == 0

    def test_budget_check_allows_within_budget(self) -> None:
        """Test that check_budget allows requests within budget."""
        guardrails = LLMGuardrails(cost_budget_usd=1.0)

        result = guardrails.check_budget(additional_cost=0.5)

        assert result is True

    def test_budget_check_blocks_over_budget(self) -> None:
        """Test that check_budget raises when over budget."""
        guardrails = LLMGuardrails(cost_budget_usd=1.0, block_on_budget=True)

        # Simulate spending
        guardrails._stats.total_cost_usd = 0.9

        with pytest.raises(CostBudgetExceeded) as exc_info:
            guardrails.check_budget(additional_cost=0.2)

        assert exc_info.value.current_cost == 1.1
        assert exc_info.value.budget == 1.0

    def test_budget_check_warns_when_not_blocking(self, caplog) -> None:
        """Test that check_budget warns instead of blocking when configured."""
        guardrails = LLMGuardrails(cost_budget_usd=1.0, block_on_budget=False)
        guardrails._stats.total_cost_usd = 0.9

        import logging

        with caplog.at_level(logging.WARNING):
            result = guardrails.check_budget(additional_cost=0.2)

        assert result is False
        assert "budget exceeded" in caplog.text.lower()

    def test_reset_budget(self) -> None:
        """Test resetting budget tracking."""
        guardrails = LLMGuardrails()
        guardrails._stats.total_cost_usd = 5.0
        guardrails._stats.budget_exceeded_count = 3

        guardrails.reset_budget()

        assert guardrails.stats.total_cost_usd == 0.0
        assert guardrails.stats.budget_exceeded_count == 0

    def test_reset_stats(self) -> None:
        """Test resetting all stats."""
        guardrails = LLMGuardrails()
        guardrails._stats.total_requests = 10
        guardrails._stats.total_cost_usd = 5.0
        guardrails._stats.pii_warnings_count = 2

        guardrails.reset_stats()

        assert guardrails.stats.total_requests == 0
        assert guardrails.stats.total_cost_usd == 0.0
        assert guardrails.stats.pii_warnings_count == 0


class TestGuardedLLMProvider:
    """Tests for GuardedLLMProvider wrapper."""

    def _create_mock_provider(self, tokens_used: int = 100):
        """Create a mock LLM provider."""
        from transcription.historian.llm_client import LLMConfig, LLMResponse

        mock_provider = MagicMock()
        mock_provider.config = LLMConfig(provider="mock", model="gpt-4o")

        async def mock_complete(system, user):
            return LLMResponse(text="Test response", tokens_used=tokens_used, duration_ms=100)

        mock_provider.complete = AsyncMock(side_effect=mock_complete)
        return mock_provider

    @pytest.mark.asyncio
    async def test_guarded_complete_success(self) -> None:
        """Test successful guarded completion."""
        mock_provider = self._create_mock_provider()
        guardrails = LLMGuardrails(rate_limit_rpm=60, cost_budget_usd=10.0)
        guarded = GuardedLLMProvider(mock_provider, guardrails)

        response = await guarded.complete(system="You are helpful.", user="Hello!")

        assert response.text == "Test response"
        mock_provider.complete.assert_called_once_with("You are helpful.", "Hello!")

    @pytest.mark.asyncio
    async def test_guarded_blocks_pii(self) -> None:
        """Test that guarded provider blocks PII when configured."""
        mock_provider = self._create_mock_provider()
        guardrails = LLMGuardrails(pii_warning=True, block_on_pii=True)
        guarded = GuardedLLMProvider(mock_provider, guardrails)

        with pytest.raises(ValueError, match="PII detected"):
            await guarded.complete(system="Analyze this.", user="Email: test@example.com")

        # Provider should not be called
        mock_provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_guarded_blocks_over_budget(self) -> None:
        """Test that guarded provider blocks when over budget."""
        mock_provider = self._create_mock_provider()
        guardrails = LLMGuardrails(cost_budget_usd=0.0001, block_on_budget=True)
        guarded = GuardedLLMProvider(mock_provider, guardrails)

        with pytest.raises(CostBudgetExceeded):
            await guarded.complete(system="System", user="User")

    @pytest.mark.asyncio
    async def test_guarded_timeout(self) -> None:
        """Test that guarded provider handles timeouts."""
        from transcription.historian.llm_client import LLMConfig

        mock_provider = MagicMock()
        mock_provider.config = LLMConfig(provider="mock", model="gpt-4o")

        async def slow_complete(system, user):
            await asyncio.sleep(10)  # Very slow

        mock_provider.complete = AsyncMock(side_effect=slow_complete)

        guardrails = LLMGuardrails(timeout_ms=100)  # 100ms timeout
        guarded = GuardedLLMProvider(mock_provider, guardrails)

        with pytest.raises(RequestTimeout) as exc_info:
            await guarded.complete(system="System", user="User")

        assert exc_info.value.timeout_ms == 100
        assert guardrails.stats.timeout_count == 1

    @pytest.mark.asyncio
    async def test_guarded_tracks_cost(self) -> None:
        """Test that guarded provider tracks costs."""
        mock_provider = self._create_mock_provider(tokens_used=500)
        guardrails = LLMGuardrails(cost_budget_usd=10.0)
        guarded = GuardedLLMProvider(mock_provider, guardrails)

        await guarded.complete(system="System", user="User")

        assert guardrails.stats.total_tokens_used > 0
        assert guardrails.stats.total_cost_usd > 0

    def test_guarded_complete_sync(self) -> None:
        """Test synchronous wrapper."""
        mock_provider = self._create_mock_provider()
        guardrails = LLMGuardrails()
        guarded = GuardedLLMProvider(mock_provider, guardrails)

        response = guarded.complete_sync(system="System", user="User")

        assert response.text == "Test response"

    def test_guarded_stats_access(self) -> None:
        """Test accessing stats through guarded provider."""
        mock_provider = self._create_mock_provider()
        guardrails = LLMGuardrails()
        guarded = GuardedLLMProvider(mock_provider, guardrails)

        stats = guarded.stats

        assert stats.total_requests == 0
        assert stats.total_cost_usd == 0.0


class TestCreateGuardedProvider:
    """Tests for create_guarded_provider convenience function."""

    def test_create_guarded_provider_default_options(self) -> None:
        """Test creating guarded provider with defaults."""
        from transcription.historian.llm_client import LLMConfig

        mock_provider = MagicMock()
        mock_provider.config = LLMConfig(provider="mock")

        guarded = create_guarded_provider(mock_provider)

        assert isinstance(guarded, GuardedLLMProvider)
        assert guarded.guardrails.rate_limit_rpm == 60
        assert guarded.guardrails.cost_budget_usd == 1.0
        assert guarded.guardrails.pii_warning is True
        assert guarded.guardrails.timeout_ms == 30000

    def test_create_guarded_provider_custom_options(self) -> None:
        """Test creating guarded provider with custom options."""
        from transcription.historian.llm_client import LLMConfig

        mock_provider = MagicMock()
        mock_provider.config = LLMConfig(provider="mock")

        guarded = create_guarded_provider(
            mock_provider,
            rate_limit_rpm=30,
            cost_budget_usd=5.0,
            pii_warning=False,
            timeout_ms=60000,
            block_on_pii=True,
            block_on_budget=False,
        )

        assert guarded.guardrails.rate_limit_rpm == 30
        assert guarded.guardrails.cost_budget_usd == 5.0
        assert guarded.guardrails.pii_warning is False
        assert guarded.guardrails.timeout_ms == 60000
        assert guarded.guardrails.block_on_pii is True
        assert guarded.guardrails.block_on_budget is False
