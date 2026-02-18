"""Sink registry for managing multiple integration sinks.

Provides a central registry for dispatching events to multiple sinks
simultaneously. Supports configuration via YAML/JSON.

Example usage:
    >>> registry = SinkRegistry.from_yaml("sinks.yaml")
    >>> await registry.dispatch(event)
    >>> await registry.close()

YAML configuration example:
    sinks:
      - name: production_webhook
        type: webhook
        config:
          url: https://api.example.com/webhook
          auth:
            type: bearer
            token: ${WEBHOOK_TOKEN}
          retry:
            max_retries: 5
      - name: analytics_webhook
        type: webhook
        config:
          url: https://analytics.example.com/events
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .events import IntegrationEvent

logger = logging.getLogger(__name__)


class SinkProtocol(Protocol):
    """Protocol for integration sinks."""

    async def send_event(
        self,
        event: IntegrationEvent,
        *,
        blocking: bool = False,
    ) -> bool:
        """Send a single event."""
        ...

    async def send_batch(
        self,
        events: list[IntegrationEvent],
        *,
        blocking: bool = False,
    ) -> list[bool]:
        """Send multiple events."""
        ...

    async def close(self) -> None:
        """Close the sink."""
        ...


@dataclass
class SinkConfig:
    """
    Configuration for a single sink.

    Attributes:
        name: Unique name for the sink.
        type: Sink type (webhook, rag, etc.).
        config: Type-specific configuration.
        enabled: Whether the sink is enabled.
        event_filter: Optional list of event types to accept.
    """

    name: str
    type: str
    config: dict[str, Any]
    enabled: bool = True
    event_filter: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SinkConfig:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type=data["type"],
            config=data.get("config", {}),
            enabled=data.get("enabled", True),
            event_filter=data.get("event_filter"),
        )


@dataclass
class SinkRegistry:
    """
    Registry for managing multiple integration sinks.

    Supports dispatching events to all registered sinks and
    filtering events by type per sink.

    Example:
        >>> registry = SinkRegistry()
        >>> registry.register("webhook", sink)
        >>> await registry.dispatch(event)
    """

    sinks: dict[str, SinkProtocol] = field(default_factory=dict)
    configs: dict[str, SinkConfig] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False)

    def register(
        self,
        name: str,
        sink: SinkProtocol,
        config: SinkConfig | None = None,
    ) -> None:
        """
        Register a sink.

        Args:
            name: Unique name for the sink.
            sink: Sink instance.
            config: Optional configuration (for filtering).
        """
        if self._closed:
            raise RuntimeError("Registry is closed")

        if name in self.sinks:
            logger.warning("Overwriting existing sink: %s", name)

        self.sinks[name] = sink
        if config:
            self.configs[name] = config

        logger.info("Registered sink: %s (type=%s)", name, type(sink).__name__)

    def unregister(self, name: str) -> bool:
        """
        Unregister a sink.

        Args:
            name: Name of the sink to remove.

        Returns:
            True if sink was removed, False if not found.
        """
        if name in self.sinks:
            del self.sinks[name]
            self.configs.pop(name, None)
            logger.info("Unregistered sink: %s", name)
            return True
        return False

    def get_sink(self, name: str) -> SinkProtocol | None:
        """Get a sink by name."""
        return self.sinks.get(name)

    def list_sinks(self) -> list[str]:
        """List registered sink names."""
        return list(self.sinks.keys())

    def _should_dispatch(self, sink_name: str, event: IntegrationEvent) -> bool:
        """Check if event should be dispatched to sink."""
        config = self.configs.get(sink_name)
        if not config:
            return True

        if not config.enabled:
            return False

        if config.event_filter:
            return event.event_type.value in config.event_filter

        return True

    async def dispatch(
        self,
        event: IntegrationEvent,
        *,
        blocking: bool = False,
    ) -> dict[str, bool]:
        """
        Dispatch event to all registered sinks.

        Args:
            event: Event to dispatch.
            blocking: If True, wait for all deliveries.

        Returns:
            Dict mapping sink name to success status.
        """
        if self._closed:
            raise RuntimeError("Registry is closed")

        results = {}
        tasks = []

        for name, sink in self.sinks.items():
            if not self._should_dispatch(name, event):
                logger.debug("Skipping sink %s for event %s", name, event.event_type.value)
                continue

            if blocking:
                tasks.append((name, sink.send_event(event, blocking=True)))
            else:
                # Fire and forget
                try:
                    await sink.send_event(event, blocking=False)
                    results[name] = True
                except Exception as e:
                    logger.error("Failed to dispatch to %s: %s", name, e)
                    results[name] = False

        if blocking and tasks:
            gathered = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True,
            )
            for (name, _), result in zip(tasks, gathered, strict=True):
                results[name] = result is True

        return results

    async def dispatch_batch(
        self,
        events: list[IntegrationEvent],
        *,
        blocking: bool = False,
    ) -> dict[str, list[bool]]:
        """
        Dispatch multiple events to all registered sinks.

        Args:
            events: Events to dispatch.
            blocking: If True, wait for all deliveries.

        Returns:
            Dict mapping sink name to list of success statuses.
        """
        if self._closed:
            raise RuntimeError("Registry is closed")

        results: dict[str, list[bool]] = {}

        for name, sink in self.sinks.items():
            # Filter events for this sink
            filtered = [e for e in events if self._should_dispatch(name, e)]
            if not filtered:
                continue

            try:
                sink_results = await sink.send_batch(filtered, blocking=blocking)
                results[name] = sink_results
            except Exception as e:
                logger.error("Failed to dispatch batch to %s: %s", name, e)
                results[name] = [False] * len(filtered)

        return results

    async def close(self) -> None:
        """Close all registered sinks."""
        self._closed = True

        close_tasks = []
        for name, sink in self.sinks.items():
            logger.debug("Closing sink: %s", name)
            close_tasks.append(sink.close())

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self.sinks.clear()
        self.configs.clear()
        logger.info("Closed all sinks")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SinkRegistry:
        """
        Create registry from configuration dictionary.

        Args:
            config: Configuration with "sinks" key.

        Returns:
            Configured SinkRegistry.
        """
        registry = cls()

        for sink_data in config.get("sinks", []):
            sink_config = SinkConfig.from_dict(sink_data)
            if not sink_config.enabled:
                logger.info("Skipping disabled sink: %s", sink_config.name)
                continue

            sink = _create_sink(sink_config)
            registry.register(sink_config.name, sink, sink_config)

        return registry

    @classmethod
    def from_yaml(cls, path: Path | str) -> SinkRegistry:
        """
        Create registry from YAML configuration file.

        Supports environment variable substitution with ${VAR} syntax.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Configured SinkRegistry.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML configuration. Install with: pip install pyyaml"
            ) from e

        path = Path(path)
        content = path.read_text()

        # Substitute environment variables
        content = _substitute_env_vars(content)

        config = yaml.safe_load(content)
        return cls.from_config(config)

    @classmethod
    def from_json(cls, path: Path | str) -> SinkRegistry:
        """
        Create registry from JSON configuration file.

        Supports environment variable substitution with ${VAR} syntax.

        Args:
            path: Path to JSON configuration file.

        Returns:
            Configured SinkRegistry.
        """
        path = Path(path)
        content = path.read_text()

        # Substitute environment variables
        content = _substitute_env_vars(content)

        config = json.loads(content)
        return cls.from_config(config)


def _substitute_env_vars(content: str) -> str:
    """
    Substitute ${VAR} patterns with environment variable values.

    Args:
        content: String with ${VAR} patterns.

    Returns:
        String with substituted values.
    """
    pattern = r"\$\{([^}]+)\}"

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name, "")
        if not value:
            logger.warning("Environment variable not set: %s", var_name)
        return value

    return re.sub(pattern, replacer, content)


def _create_sink(config: SinkConfig) -> SinkProtocol:
    """
    Create a sink from configuration.

    Args:
        config: Sink configuration.

    Returns:
        Sink instance.

    Raises:
        ValueError: If sink type is unknown.
    """
    if config.type == "webhook":
        from .webhooks import WebhookConfig, WebhookSink

        webhook_config = WebhookConfig.from_dict(config.config)
        return WebhookSink(webhook_config)

    else:
        raise ValueError(f"Unknown sink type: {config.type}")
