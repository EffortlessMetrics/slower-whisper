"""Webhook sink for sending events to HTTP endpoints.

Provides async webhook delivery with:
- Configurable authentication (bearer token, basic auth)
- Retry with exponential backoff
- Dead letter queue for failed deliveries
- HMAC signature verification
- Batch sending support

Example usage:
    >>> config = WebhookConfig(
    ...     url="https://api.example.com/webhook",
    ...     auth=AuthConfig(type="bearer", token="secret"),
    ... )
    >>> sink = WebhookSink(config)
    >>> await sink.send_event(event)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .events import IntegrationEvent

logger = logging.getLogger(__name__)


class AuthType(str, Enum):
    """Authentication types for webhook requests."""

    NONE = "none"
    BEARER = "bearer"
    BASIC = "basic"


@dataclass(slots=True)
class AuthConfig:
    """
    Authentication configuration for webhooks.

    Supports bearer token and basic authentication.

    Attributes:
        type: Authentication type (none, bearer, basic).
        token: Bearer token (for type="bearer").
        username: Username (for type="basic").
        password: Password (for type="basic").
    """

    type: AuthType | str = AuthType.NONE
    token: str | None = None
    username: str | None = None
    password: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = AuthType(self.type)

    def get_header(self) -> dict[str, str]:
        """Get Authorization header for this auth config."""
        if self.type == AuthType.BEARER and self.token:
            return {"Authorization": f"Bearer {self.token}"}
        elif self.type == AuthType.BASIC and self.username and self.password:
            credentials = f"{self.username}:{self.password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            return {"Authorization": f"Basic {encoded}"}
        return {}


@dataclass(slots=True)
class RetryPolicy:
    """
    Retry policy for webhook delivery.

    Uses exponential backoff: delay = base_delay * (2 ** attempt).

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        retry_on_status: HTTP status codes to retry on.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given retry attempt."""
        delay = self.base_delay * (2**attempt)
        return float(min(delay, self.max_delay))


@dataclass
class DeadLetterEntry:
    """
    Entry in the dead letter queue for failed webhook deliveries.

    Attributes:
        event: The event that failed to send.
        error: Error message describing the failure.
        attempts: Number of delivery attempts made.
        last_attempt: Unix timestamp of last delivery attempt.
        url: Target webhook URL.
    """

    event: dict[str, Any]
    error: str
    attempts: int
    last_attempt: float
    url: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "event": self.event,
            "error": self.error,
            "attempts": self.attempts,
            "last_attempt": self.last_attempt,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeadLetterEntry:
        """Create from dictionary."""
        return cls(
            event=data["event"],
            error=data["error"],
            attempts=data["attempts"],
            last_attempt=data["last_attempt"],
            url=data["url"],
        )


@dataclass
class WebhookConfig:
    """
    Configuration for webhook sink.

    Attributes:
        url: Target webhook URL.
        headers: Additional HTTP headers to include.
        auth: Authentication configuration.
        retry: Retry policy for failed deliveries.
        timeout: Request timeout in seconds.
        hmac_secret: Secret for HMAC signature (optional).
        hmac_header: Header name for HMAC signature.
        dead_letter_path: Path to store failed deliveries (optional).
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    auth: AuthConfig = field(default_factory=AuthConfig)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float = 30.0
    hmac_secret: str | None = None
    hmac_header: str = "X-Webhook-Signature"
    dead_letter_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebhookConfig:
        """Create config from dictionary."""
        auth_data = data.get("auth", {})
        retry_data = data.get("retry", {})

        return cls(
            url=data["url"],
            headers=data.get("headers", {}),
            auth=AuthConfig(
                type=auth_data.get("type", "none"),
                token=auth_data.get("token"),
                username=auth_data.get("username"),
                password=auth_data.get("password"),
            ),
            retry=RetryPolicy(
                max_retries=retry_data.get("max_retries", 3),
                base_delay=retry_data.get("base_delay", 1.0),
                max_delay=retry_data.get("max_delay", 30.0),
                retry_on_status=tuple(retry_data.get("retry_on_status", [429, 500, 502, 503, 504])),
            ),
            timeout=data.get("timeout", 30.0),
            hmac_secret=data.get("hmac_secret"),
            hmac_header=data.get("hmac_header", "X-Webhook-Signature"),
            dead_letter_path=Path(data["dead_letter_path"])
            if data.get("dead_letter_path")
            else None,
        )


class HttpClientProtocol(Protocol):
    """Protocol for HTTP client (for dependency injection in tests)."""

    async def post(
        self,
        url: str,
        *,
        json: Any,
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse:
        """Send POST request."""
        ...


@dataclass
class HttpResponse:
    """Simple HTTP response wrapper."""

    status_code: int
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)


class WebhookSink:
    """
    Webhook sink for sending events to HTTP endpoints.

    Supports async sending, retry with exponential backoff, and
    dead letter queue for failed deliveries.

    Example:
        >>> config = WebhookConfig(url="https://api.example.com/webhook")
        >>> sink = WebhookSink(config)
        >>> await sink.send_event(event)
        >>> await sink.close()
    """

    def __init__(
        self,
        config: WebhookConfig,
        http_client: HttpClientProtocol | None = None,
    ) -> None:
        """
        Initialize webhook sink.

        Args:
            config: Webhook configuration.
            http_client: Optional HTTP client (for testing).
        """
        self.config = config
        self._http_client = http_client
        self._dead_letter_queue: list[DeadLetterEntry] = []
        self._pending_tasks: list[asyncio.Task[Any]] = []
        self._closed = False

        # Load existing dead letter entries if path exists
        if config.dead_letter_path and config.dead_letter_path.exists():
            self._load_dead_letter_queue()

    def _get_http_client(self) -> HttpClientProtocol:
        """Get or create HTTP client."""
        if self._http_client is not None:
            return self._http_client

        # Lazy import to avoid requiring httpx for non-webhook usage
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for webhook sink. Install with: pip install httpx"
            ) from e

        # Create an adapter that wraps httpx.AsyncClient
        class HttpxAdapter:
            def __init__(self) -> None:
                self._client: httpx.AsyncClient | None = None

            async def _ensure_client(self) -> httpx.AsyncClient:
                if self._client is None:
                    self._client = httpx.AsyncClient()
                return self._client

            async def post(
                self,
                url: str,
                *,
                json: Any,
                headers: dict[str, str],
                timeout: float,
            ) -> HttpResponse:
                client = await self._ensure_client()
                response = await client.post(
                    url,
                    json=json,
                    headers=headers,
                    timeout=timeout,
                )
                return HttpResponse(
                    status_code=response.status_code,
                    text=response.text,
                    headers=dict(response.headers),
                )

            async def close(self) -> None:
                if self._client:
                    await self._client.aclose()
                    self._client = None

        adapter: HttpClientProtocol = HttpxAdapter()  # HttpxAdapter implements the protocol
        self._http_client = adapter
        return self._http_client

    def _compute_signature(self, payload: str) -> str:
        """Compute HMAC-SHA256 signature for payload."""
        if not self.config.hmac_secret:
            return ""
        signature = hmac.new(
            self.config.hmac_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        )
        return f"sha256={signature.hexdigest()}"

    def _build_headers(self, payload: str) -> dict[str, str]:
        """Build request headers including auth and signature."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "slower-whisper-webhook/1.0",
        }
        headers.update(self.config.headers)
        headers.update(self.config.auth.get_header())

        if self.config.hmac_secret:
            headers[self.config.hmac_header] = self._compute_signature(payload)

        return headers

    async def send_event(
        self,
        event: IntegrationEvent,
        *,
        blocking: bool = False,
    ) -> bool:
        """
        Send a single event to the webhook.

        Args:
            event: Event to send.
            blocking: If True, wait for delivery. If False, return immediately.

        Returns:
            True if event was sent successfully (only meaningful when blocking=True).
        """
        if self._closed:
            raise RuntimeError("WebhookSink is closed")

        if blocking:
            return await self._send_with_retry(event)
        else:
            task = asyncio.create_task(self._send_with_retry(event))
            self._pending_tasks.append(task)
            # Clean up completed tasks
            self._pending_tasks = [t for t in self._pending_tasks if not t.done()]
            return True

    async def send_batch(
        self,
        events: list[IntegrationEvent],
        *,
        blocking: bool = False,
    ) -> list[bool]:
        """
        Send multiple events to the webhook.

        Args:
            events: List of events to send.
            blocking: If True, wait for all deliveries.

        Returns:
            List of success status for each event (only meaningful when blocking=True).
        """
        if blocking:
            results = await asyncio.gather(
                *[self._send_with_retry(e) for e in events],
                return_exceptions=True,
            )
            return [r is True for r in results]
        else:
            for event in events:
                await self.send_event(event, blocking=False)
            return [True] * len(events)

    async def send_transcript(
        self,
        transcript: Any,  # Transcript type
        *,
        blocking: bool = False,
    ) -> bool:
        """
        Send a completed transcript event.

        Args:
            transcript: The completed Transcript object.
            blocking: If True, wait for delivery.

        Returns:
            True if sent successfully.
        """
        from .events import create_transcript_event

        event = create_transcript_event(transcript)
        return await self.send_event(event, blocking=blocking)

    async def send_outcomes(
        self,
        outcomes: list[dict[str, Any]],
        source: str,
        *,
        blocking: bool = False,
    ) -> bool:
        """
        Send detected outcomes event.

        Args:
            outcomes: List of detected outcomes.
            source: Source identifier.
            blocking: If True, wait for delivery.

        Returns:
            True if sent successfully.
        """
        from .events import create_outcome_event

        event = create_outcome_event(outcomes, source)
        return await self.send_event(event, blocking=blocking)

    async def _send_with_retry(self, event: IntegrationEvent) -> bool:
        """Send event with retry logic."""
        payload = json.dumps(event.to_dict())
        headers = self._build_headers(payload)
        client = self._get_http_client()

        last_error = ""
        for attempt in range(self.config.retry.max_retries + 1):
            try:
                response = await client.post(
                    self.config.url,
                    json=event.to_dict(),
                    headers=headers,
                    timeout=self.config.timeout,
                )

                if response.status_code < 300:
                    logger.debug(
                        "Webhook delivered: event_id=%s, status=%d",
                        event.event_id,
                        response.status_code,
                    )
                    return True

                if response.status_code in self.config.retry.retry_on_status:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if attempt < self.config.retry.max_retries:
                        delay = self.config.retry.get_delay(attempt)
                        logger.warning(
                            "Webhook delivery failed (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1,
                            self.config.retry.max_retries + 1,
                            delay,
                            last_error,
                        )
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Non-retryable error
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    break

            except Exception as e:
                last_error = str(e)
                if attempt < self.config.retry.max_retries:
                    delay = self.config.retry.get_delay(attempt)
                    logger.warning(
                        "Webhook delivery error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.config.retry.max_retries + 1,
                        delay,
                        last_error,
                    )
                    await asyncio.sleep(delay)
                    continue
                break

        # All retries exhausted, add to dead letter queue
        logger.error(
            "Webhook delivery failed after %d attempts: event_id=%s, error=%s",
            self.config.retry.max_retries + 1,
            event.event_id,
            last_error,
        )
        self._add_to_dead_letter_queue(event, last_error)
        return False

    def _add_to_dead_letter_queue(self, event: IntegrationEvent, error: str) -> None:
        """Add failed event to dead letter queue."""
        entry = DeadLetterEntry(
            event=event.to_dict(),
            error=error,
            attempts=self.config.retry.max_retries + 1,
            last_attempt=time.time(),
            url=self.config.url,
        )
        self._dead_letter_queue.append(entry)
        self._save_dead_letter_queue()

    def _save_dead_letter_queue(self) -> None:
        """Save dead letter queue to disk."""
        if not self.config.dead_letter_path:
            return

        try:
            self.config.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
            data = [e.to_dict() for e in self._dead_letter_queue]
            self.config.dead_letter_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save dead letter queue: %s", e)

    def _load_dead_letter_queue(self) -> None:
        """Load dead letter queue from disk."""
        if not self.config.dead_letter_path or not self.config.dead_letter_path.exists():
            return

        try:
            data = json.loads(self.config.dead_letter_path.read_text())
            self._dead_letter_queue = [DeadLetterEntry.from_dict(e) for e in data]
            logger.info("Loaded %d entries from dead letter queue", len(self._dead_letter_queue))
        except Exception as e:
            logger.warning("Failed to load dead letter queue: %s", e)

    def get_dead_letter_queue(self) -> list[DeadLetterEntry]:
        """Get current dead letter queue entries."""
        return list(self._dead_letter_queue)

    def clear_dead_letter_queue(self) -> int:
        """Clear dead letter queue and return count of cleared entries."""
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        self._save_dead_letter_queue()
        return count

    async def retry_dead_letters(self) -> tuple[int, int]:
        """
        Retry all entries in the dead letter queue.

        Returns:
            Tuple of (successful_count, failed_count).
        """
        from .events import IntegrationEvent

        entries = list(self._dead_letter_queue)
        self._dead_letter_queue.clear()

        successful = 0
        failed = 0

        for entry in entries:
            event = IntegrationEvent.from_dict(entry.event)
            if await self._send_with_retry(event):
                successful += 1
            else:
                failed += 1

        self._save_dead_letter_queue()
        return successful, failed

    async def close(self) -> None:
        """Close the sink and wait for pending deliveries."""
        self._closed = True

        # Wait for pending tasks
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

        # Close HTTP client if we created it
        if self._http_client is not None and hasattr(self._http_client, "close"):
            await self._http_client.close()

        # Save any remaining dead letter entries
        self._save_dead_letter_queue()


def verify_webhook_signature(
    payload: str | bytes,
    signature: str,
    secret: str,
) -> bool:
    """
    Verify HMAC-SHA256 signature of webhook payload.

    Use this to verify incoming webhook requests from slower-whisper.

    Args:
        payload: Raw request body.
        signature: Value of X-Webhook-Signature header.
        secret: Shared HMAC secret.

    Returns:
        True if signature is valid.

    Example:
        >>> if verify_webhook_signature(request.body, request.headers["X-Webhook-Signature"], secret):
        ...     process_event(json.loads(request.body))
    """
    if isinstance(payload, str):
        payload = payload.encode()

    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    )
    expected_sig = f"sha256={expected.hexdigest()}"

    return hmac.compare_digest(signature, expected_sig)
