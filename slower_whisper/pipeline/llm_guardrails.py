"""
LLM Guardrails for rate limiting, cost tracking, and PII detection.

This module provides safety wrappers around LLM providers to enforce:
- Rate limits using token bucket algorithm
- Cost budgets per session
- PII detection warnings
- Request timeouts

Usage:
    from transcription.llm_guardrails import LLMGuardrails, GuardedLLMProvider
    from transcription.historian.llm_client import create_llm_provider, LLMConfig

    config = LLMConfig(provider="openai", model="gpt-4o")
    provider = create_llm_provider(config)

    guardrails = LLMGuardrails(
        rate_limit_rpm=60,
        cost_budget_usd=1.0,
        pii_warning=True,
        timeout_ms=30000,
    )

    guarded = GuardedLLMProvider(provider, guardrails)
    response = await guarded.complete(system="...", user="...")
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .historian.llm_client import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# Model pricing per 1M tokens (input/output) as of Jan 2026
# Prices in USD
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    # Anthropic models
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    # Default fallback for unknown models
    "_default": {"input": 5.00, "output": 15.00},
}


@dataclass
class PIIMatch:
    """Represents a detected PII pattern match."""

    type: str  # email, phone, ssn, credit_card
    start: int
    end: int
    masked: str  # The masked version of the match


@dataclass
class GuardrailStats:
    """Statistics tracked by guardrails."""

    total_requests: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    rate_limited_count: int = 0
    pii_warnings_count: int = 0
    timeout_count: int = 0
    budget_exceeded_count: int = 0


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, wait_seconds: float):
        self.wait_seconds = wait_seconds
        super().__init__(f"Rate limit exceeded. Wait {wait_seconds:.2f} seconds.")


class CostBudgetExceeded(Exception):
    """Raised when cost budget is exceeded."""

    def __init__(self, current_cost: float, budget: float):
        self.current_cost = current_cost
        self.budget = budget
        super().__init__(f"Cost budget exceeded: ${current_cost:.4f} >= ${budget:.2f}")


class RequestTimeout(Exception):
    """Raised when request times out."""

    def __init__(self, timeout_ms: int):
        self.timeout_ms = timeout_ms
        super().__init__(f"Request timed out after {timeout_ms}ms")


@dataclass
class LLMGuardrails:
    """
    Configurable guardrails for LLM requests.

    Attributes:
        rate_limit_rpm: Maximum requests per minute (default: 60)
        cost_budget_usd: Maximum spend per session in USD (default: 1.0)
        pii_warning: Whether to warn if PII is detected (default: True)
        timeout_ms: Per-request timeout in milliseconds (default: 30000)
        block_on_pii: If True, block requests with PII; if False, just warn (default: False)
        block_on_budget: If True, raise exception on budget exceeded; if False, just warn
    """

    rate_limit_rpm: int = 60
    cost_budget_usd: float = 1.0
    pii_warning: bool = True
    timeout_ms: int = 30000
    block_on_pii: bool = False
    block_on_budget: bool = True

    # Internal state
    _token_bucket: float = field(default=0.0, init=False, repr=False)
    _last_token_refill: float = field(default=0.0, init=False, repr=False)
    _stats: GuardrailStats = field(default_factory=GuardrailStats, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize token bucket."""
        self._token_bucket = float(self.rate_limit_rpm)
        self._last_token_refill = time.monotonic()

    @property
    def stats(self) -> GuardrailStats:
        """Return current guardrail statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._stats = GuardrailStats()

    def reset_budget(self) -> None:
        """Reset the cost budget tracking."""
        self._stats.total_cost_usd = 0.0
        self._stats.budget_exceeded_count = 0

    async def check_rate_limit(self, block: bool = True) -> float | None:
        """
        Check and consume rate limit token.

        Args:
            block: If True, wait until token is available. If False, raise if unavailable.

        Returns:
            Time waited in seconds, or None if no wait needed.

        Raises:
            RateLimitExceeded: If block=False and no tokens available.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_token_refill

            # Refill tokens based on elapsed time
            # Rate is tokens per second = rpm / 60
            refill_rate = self.rate_limit_rpm / 60.0
            self._token_bucket = min(
                float(self.rate_limit_rpm), self._token_bucket + elapsed * refill_rate
            )
            self._last_token_refill = now

            if self._token_bucket >= 1.0:
                # Consume token
                self._token_bucket -= 1.0
                return None

            # Calculate wait time
            wait_seconds = (1.0 - self._token_bucket) / refill_rate

            if not block:
                self._stats.rate_limited_count += 1
                raise RateLimitExceeded(wait_seconds)

        # Wait outside the lock
        self._stats.rate_limited_count += 1
        await asyncio.sleep(wait_seconds)

        # Try again after waiting
        async with self._lock:
            self._token_bucket = max(0.0, self._token_bucket - 1.0)

        return wait_seconds

    def check_budget(self, additional_cost: float = 0.0) -> bool:
        """
        Check if budget would be exceeded.

        Args:
            additional_cost: Additional cost to add before checking.

        Returns:
            True if within budget, False if exceeded.

        Raises:
            CostBudgetExceeded: If block_on_budget is True and budget exceeded.
        """
        projected_cost = self._stats.total_cost_usd + additional_cost

        if projected_cost >= self.cost_budget_usd:
            self._stats.budget_exceeded_count += 1
            if self.block_on_budget:
                raise CostBudgetExceeded(projected_cost, self.cost_budget_usd)
            logger.warning(
                f"Cost budget exceeded: ${projected_cost:.4f} >= ${self.cost_budget_usd:.2f}"
            )
            return False

        return True

    def track_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Track cost for a completed request.

        Args:
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD for this request.
        """
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["_default"])
        cost = (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]
        self._stats.total_cost_usd += cost
        self._stats.total_tokens_used += input_tokens + output_tokens
        self._stats.total_requests += 1
        return cost

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a request without tracking.

        Args:
            model: Model name
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens

        Returns:
            Estimated cost in USD.
        """
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["_default"])
        return (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]

    def detect_pii(self, text: str) -> list[PIIMatch]:
        """
        Detect PII patterns in text.

        Detects:
        - Email addresses
        - Phone numbers (US format)
        - Social Security Numbers
        - Credit card numbers

        Args:
            text: Text to scan for PII

        Returns:
            List of PIIMatch objects for each detection.
        """
        matches: list[PIIMatch] = []

        # Email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        for match in re.finditer(email_pattern, text):
            local, domain = match.group().split("@", 1)
            masked = f"{local[0]}***@{domain}"
            matches.append(
                PIIMatch(
                    type="email",
                    start=match.start(),
                    end=match.end(),
                    masked=masked,
                )
            )

        # US Phone pattern (various formats)
        phone_pattern = r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        for match in re.finditer(phone_pattern, text):
            # Mask middle digits
            phone = match.group()
            masked = re.sub(r"\d", "*", phone[:-4]) + phone[-4:]
            matches.append(
                PIIMatch(
                    type="phone",
                    start=match.start(),
                    end=match.end(),
                    masked=masked,
                )
            )

        # SSN pattern (XXX-XX-XXXX)
        ssn_pattern = r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
        # Filter out matches that look like phone numbers (already captured)
        for match in re.finditer(ssn_pattern, text):
            # Skip if this overlaps with a phone match
            is_phone = any(m.type == "phone" and m.start <= match.start() < m.end for m in matches)
            if not is_phone:
                matches.append(
                    PIIMatch(
                        type="ssn",
                        start=match.start(),
                        end=match.end(),
                        masked="***-**-" + match.group()[-4:],
                    )
                )

        # Credit card pattern (basic - 13-19 digits with optional separators)
        cc_pattern = r"\b(?:\d{4}[-\s]?){3,4}\d{1,4}\b"
        for match in re.finditer(cc_pattern, text):
            digits = re.sub(r"[-\s]", "", match.group())
            if 13 <= len(digits) <= 19:
                masked = "*" * (len(digits) - 4) + digits[-4:]
                matches.append(
                    PIIMatch(
                        type="credit_card",
                        start=match.start(),
                        end=match.end(),
                        masked=masked,
                    )
                )

        return matches

    def check_pii(self, text: str) -> list[PIIMatch]:
        """
        Check text for PII and optionally warn/block.

        Args:
            text: Text to check

        Returns:
            List of PII matches found.

        Raises:
            ValueError: If block_on_pii is True and PII is detected.
        """
        if not self.pii_warning:
            return []

        matches = self.detect_pii(text)

        if matches:
            self._stats.pii_warnings_count += 1
            pii_types = {m.type for m in matches}
            logger.warning(
                f"PII detected in request: {', '.join(pii_types)} ({len(matches)} instances)"
            )

            if self.block_on_pii:
                raise ValueError(
                    f"PII detected in request: {', '.join(pii_types)}. "
                    "Set block_on_pii=False to allow with warning."
                )

        return matches


class GuardedLLMProvider:
    """
    Wrapper around LLMProvider that enforces guardrails.

    Usage:
        provider = create_llm_provider(config)
        guardrails = LLMGuardrails(rate_limit_rpm=60, cost_budget_usd=1.0)
        guarded = GuardedLLMProvider(provider, guardrails)

        response = await guarded.complete(system="...", user="...")
    """

    def __init__(
        self,
        provider: LLMProvider,
        guardrails: LLMGuardrails,
    ):
        """
        Initialize guarded provider.

        Args:
            provider: The underlying LLM provider
            guardrails: Guardrail configuration
        """
        self.provider = provider
        self.guardrails = guardrails

    @property
    def config(self):
        """Access underlying provider config."""
        return self.provider.config

    @property
    def stats(self) -> GuardrailStats:
        """Get guardrail statistics."""
        return self.guardrails.stats

    async def complete(self, system: str, user: str) -> LLMResponse:
        """
        Send a guarded completion request.

        Enforces:
        1. Rate limiting (waits if necessary)
        2. PII detection
        3. Budget checking
        4. Request timeout

        Args:
            system: System prompt
            user: User prompt

        Returns:
            LLMResponse from the underlying provider

        Raises:
            RateLimitExceeded: If rate limit exceeded (non-blocking mode)
            CostBudgetExceeded: If cost budget exceeded
            ValueError: If PII detected and block_on_pii is True
            RequestTimeout: If request times out
        """
        # Check for PII in both prompts
        combined_text = system + " " + user
        self.guardrails.check_pii(combined_text)

        # Check budget before request
        # Estimate cost based on token count (rough estimate: 4 chars per token)
        estimated_input = len(combined_text) // 4
        estimated_output = self.provider.config.max_tokens // 2  # Assume half max
        estimated_cost = self.guardrails.estimate_cost(
            self.provider.config.model or "gpt-4o",
            estimated_input,
            estimated_output,
        )
        self.guardrails.check_budget(estimated_cost)

        # Wait for rate limit
        await self.guardrails.check_rate_limit(block=True)

        # Execute with timeout
        timeout_seconds = self.guardrails.timeout_ms / 1000.0
        try:
            response = await asyncio.wait_for(
                self.provider.complete(system, user),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            self.guardrails._stats.timeout_count += 1
            raise RequestTimeout(self.guardrails.timeout_ms) from None

        # Track actual cost if we have token counts
        if response.tokens_used is not None:
            # Estimate input/output split (rough approximation)
            # We don't have exact split, so estimate based on response length
            estimated_input = len(combined_text) // 4
            estimated_output = response.tokens_used - estimated_input
            if estimated_output < 0:
                estimated_output = response.tokens_used // 2
                estimated_input = response.tokens_used - estimated_output

            actual_cost = self.guardrails.track_cost(
                self.provider.config.model or "gpt-4o",
                estimated_input,
                max(0, estimated_output),
            )
            logger.debug(f"Request cost: ${actual_cost:.6f}")

        return response

    def complete_sync(self, system: str, user: str) -> LLMResponse:
        """Synchronous wrapper for complete()."""
        return asyncio.run(self.complete(system, user))


def create_guarded_provider(
    provider: LLMProvider,
    rate_limit_rpm: int = 60,
    cost_budget_usd: float = 1.0,
    pii_warning: bool = True,
    timeout_ms: int = 30000,
    block_on_pii: bool = False,
    block_on_budget: bool = True,
) -> GuardedLLMProvider:
    """
    Convenience function to create a guarded LLM provider.

    Args:
        provider: The underlying LLM provider
        rate_limit_rpm: Maximum requests per minute
        cost_budget_usd: Maximum spend per session in USD
        pii_warning: Whether to warn if PII is detected
        timeout_ms: Per-request timeout in milliseconds
        block_on_pii: If True, block requests with PII
        block_on_budget: If True, raise exception on budget exceeded

    Returns:
        GuardedLLMProvider wrapping the given provider
    """
    guardrails = LLMGuardrails(
        rate_limit_rpm=rate_limit_rpm,
        cost_budget_usd=cost_budget_usd,
        pii_warning=pii_warning,
        timeout_ms=timeout_ms,
        block_on_pii=block_on_pii,
        block_on_budget=block_on_budget,
    )
    return GuardedLLMProvider(provider, guardrails)
