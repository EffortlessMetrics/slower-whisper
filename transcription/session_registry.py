"""Session registry for managing streaming sessions across REST and WebSocket interfaces.

This module provides a thread-safe registry for tracking streaming sessions,
enabling REST endpoints to manage WebSocket sessions.

Features:
- Session lifecycle management with valid state transitions
- TTL-based idle session reaping
- Disconnection handling with reconnection support
- Force close semantics with proper cleanup

Issue #85: REST streaming endpoints for session management.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# State Transition Rules
# =============================================================================

# Valid state transitions: (from_state, to_state) -> is_valid
# See docs/STREAMING_ARCHITECTURE.md for state machine specification
VALID_TRANSITIONS: dict[tuple[str, str], bool] = {
    # Normal lifecycle
    ("created", "active"): True,
    ("active", "ending"): True,
    ("ending", "ended"): True,
    # Error from any active state
    ("created", "error"): True,
    ("active", "error"): True,
    ("ending", "error"): True,
    # Disconnection handling
    ("active", "disconnected"): True,
    ("disconnected", "active"): True,  # Resume
    ("disconnected", "ended"): True,  # TTL expiry while disconnected
    ("disconnected", "error"): True,
}


class SessionStatus(str, Enum):
    """Session lifecycle status for REST API."""

    CREATED = "created"
    ACTIVE = "active"
    ENDING = "ending"
    ENDED = "ended"
    ERROR = "error"
    DISCONNECTED = "disconnected"

    @classmethod
    def is_valid_transition(cls, from_status: SessionStatus, to_status: SessionStatus) -> bool:
        """Check if a state transition is valid.

        Args:
            from_status: Current session status
            to_status: Target session status

        Returns:
            True if transition is valid, False otherwise
        """
        return VALID_TRANSITIONS.get((from_status.value, to_status.value), False)

    @classmethod
    def is_terminal(cls, status: SessionStatus) -> bool:
        """Check if a status is terminal (no further transitions allowed).

        Args:
            status: Session status to check

        Returns:
            True if status is terminal (ended or error)
        """
        return status in (cls.ENDED, cls.ERROR)


@dataclass
class SessionInfo:
    """
    Information about a streaming session for REST API.

    This is a snapshot of session state that can be serialized to JSON
    for the REST API without exposing internal session objects.
    """

    session_id: str
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    config: dict[str, Any]
    stats: dict[str, Any]
    last_event_id: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "config": self.config,
            "stats": self.stats,
            "last_event_id": self.last_event_id,
            "error": self.error,
        }


class InvalidStateTransitionError(Exception):
    """Raised when an invalid session state transition is attempted."""

    def __init__(
        self,
        session_id: str,
        from_status: SessionStatus,
        to_status: SessionStatus,
    ) -> None:
        self.session_id = session_id
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Invalid state transition for session {session_id}: "
            f"{from_status.value} -> {to_status.value}"
        )


@dataclass
class RegisteredSession:
    """
    Internal session tracking record.

    Holds the session object and metadata for registry management.

    Attributes:
        session_id: Unique session identifier
        session: WebSocketStreamingSession instance
        created_at: Unix timestamp when session was created
        last_activity: Unix timestamp of last activity
        websocket: WebSocket connection if connected
        status: Current session status (may differ from session.state for disconnected)
        pending_tasks: Set of pending async tasks for cancellation on force close
        disconnected_at: Unix timestamp when session was disconnected (for TTL)
    """

    session_id: str
    session: Any  # WebSocketStreamingSession
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    websocket: Any = None  # WebSocket connection if connected
    status: SessionStatus = SessionStatus.CREATED
    pending_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    disconnected_at: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def transition_to(self, new_status: SessionStatus) -> bool:
        """
        Attempt to transition to a new status.

        Args:
            new_status: Target session status

        Returns:
            True if transition was successful

        Raises:
            InvalidStateTransitionError: If transition is invalid
        """
        if self.status == new_status:
            return True  # No-op, same status

        if not SessionStatus.is_valid_transition(self.status, new_status):
            raise InvalidStateTransitionError(self.session_id, self.status, new_status)

        old_status = self.status
        self.status = new_status
        self.update_activity()

        # Track disconnection time for TTL
        if new_status == SessionStatus.DISCONNECTED:
            self.disconnected_at = time.time()
        elif old_status == SessionStatus.DISCONNECTED:
            self.disconnected_at = None  # Clear on reconnect

        logger.debug(
            "Session %s transitioned: %s -> %s",
            self.session_id,
            old_status.value,
            new_status.value,
        )
        return True

    def add_task(self, task: asyncio.Task[Any]) -> None:
        """Add a task to track for cleanup."""
        self.pending_tasks.add(task)
        task.add_done_callback(self.pending_tasks.discard)

    def cancel_pending_tasks(self) -> int:
        """Cancel all pending tasks.

        Returns:
            Number of tasks cancelled
        """
        count = 0
        for task in list(self.pending_tasks):
            if not task.done():
                task.cancel()
                count += 1
        self.pending_tasks.clear()
        return count


class SessionRegistry:
    """
    Thread-safe registry for streaming sessions.

    Provides session management across REST and WebSocket interfaces:
    - Track active sessions by ID
    - Get session status and stats
    - Force close sessions with proper cleanup
    - TTL-based idle session cleanup
    - Disconnection handling with reconnection support

    This is a singleton - use get_registry() to obtain the instance.

    Configuration:
        idle_timeout_sec: Seconds before idle sessions are reaped (default: 300)
        cleanup_interval_sec: Interval between cleanup runs (default: 60)
        disconnected_ttl_sec: TTL for disconnected sessions before ended (default: 120)
    """

    _instance: SessionRegistry | None = None
    _lock: Lock = Lock()

    # Instance attributes - declared for type checking
    _sessions: dict[str, RegisteredSession]
    _registry_lock: Lock
    _cleanup_task: asyncio.Task[None] | None
    _shutdown_event: asyncio.Event | None
    _idle_timeout_sec: float
    _cleanup_interval_sec: float
    _disconnected_ttl_sec: float

    def __new__(cls) -> SessionRegistry:
        """Singleton pattern - ensure only one registry exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._sessions = {}
                    instance._registry_lock = Lock()
                    instance._cleanup_task = None
                    instance._shutdown_event = None
                    # Configuration
                    instance._idle_timeout_sec = 300.0  # 5 minutes for idle sessions
                    instance._cleanup_interval_sec = 60.0  # Run cleanup every minute
                    instance._disconnected_ttl_sec = 120.0  # 2 minutes for disconnected
                    cls._instance = instance
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing only)."""
        with cls._lock:
            if cls._instance is not None:
                # Stop cleanup task if running
                if cls._instance._cleanup_task is not None:
                    cls._instance._cleanup_task.cancel()
                    cls._instance._cleanup_task = None
                if cls._instance._shutdown_event is not None:
                    cls._instance._shutdown_event.set()
                    cls._instance._shutdown_event = None
                cls._instance._sessions.clear()
                cls._instance = None

    def configure(
        self,
        idle_timeout_sec: float | None = None,
        cleanup_interval_sec: float | None = None,
        disconnected_ttl_sec: float | None = None,
    ) -> None:
        """
        Configure registry settings.

        Args:
            idle_timeout_sec: Seconds before idle sessions are reaped
            cleanup_interval_sec: Interval between cleanup runs
            disconnected_ttl_sec: TTL for disconnected sessions
        """
        if idle_timeout_sec is not None:
            self._idle_timeout_sec = idle_timeout_sec
        if cleanup_interval_sec is not None:
            self._cleanup_interval_sec = cleanup_interval_sec
        if disconnected_ttl_sec is not None:
            self._disconnected_ttl_sec = disconnected_ttl_sec

        logger.info(
            "Registry configured: idle_timeout=%s, cleanup_interval=%s, disconnected_ttl=%s",
            self._idle_timeout_sec,
            self._cleanup_interval_sec,
            self._disconnected_ttl_sec,
        )

    def register(
        self,
        session: Any,
        websocket: Any = None,
    ) -> str:
        """
        Register a new session.

        Args:
            session: WebSocketStreamingSession instance
            websocket: Optional WebSocket connection

        Returns:
            Session ID (same as session.stream_id)
        """
        session_id = session.stream_id
        # Derive initial status from session state
        initial_status = SessionStatus(session.state.value)

        with self._registry_lock:
            self._sessions[session_id] = RegisteredSession(
                session_id=session_id,
                session=session,
                websocket=websocket,
                status=initial_status,
            )
        logger.info("Registered session: %s (status=%s)", session_id, initial_status.value)
        return str(session_id)

    def unregister(self, session_id: str) -> bool:
        """
        Unregister a session.

        Args:
            session_id: Session ID to unregister

        Returns:
            True if session was found and removed, False otherwise
        """
        with self._registry_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Unregistered session: %s", session_id)
                return True
        return False

    def get(self, session_id: str) -> RegisteredSession | None:
        """
        Get a registered session by ID.

        Args:
            session_id: Session ID to look up

        Returns:
            RegisteredSession if found, None otherwise
        """
        with self._registry_lock:
            return self._sessions.get(session_id)

    def get_info(self, session_id: str) -> SessionInfo | None:
        """
        Get session info for REST API.

        Args:
            session_id: Session ID to look up

        Returns:
            SessionInfo snapshot if found, None otherwise
        """
        registered = self.get(session_id)
        if registered is None:
            return None

        session = registered.session

        # Use registry's tracked status (handles disconnection correctly)
        status = registered.status

        # Sync status with underlying session state if needed
        # (e.g., if session transitioned internally)
        session_status = SessionStatus(session.state.value)
        if session_status != status and session_status in (
            SessionStatus.ENDED,
            SessionStatus.ERROR,
        ):
            # Session ended/errored internally, update registry status
            try:
                registered.transition_to(session_status)
                status = session_status
            except InvalidStateTransitionError:
                # Keep current status if transition invalid
                pass

        return SessionInfo(
            session_id=session_id,
            status=status,
            created_at=datetime.fromtimestamp(registered.created_at, tz=UTC),
            last_activity=datetime.fromtimestamp(registered.last_activity, tz=UTC),
            config={
                "max_gap_sec": session.config.max_gap_sec,
                "enable_prosody": session.config.enable_prosody,
                "enable_emotion": session.config.enable_emotion,
                "enable_diarization": session.config.enable_diarization,
                "sample_rate": session.config.sample_rate,
            },
            stats=session.stats.to_dict(),
            last_event_id=session._event_counter.current,
        )

    def list_sessions(self) -> list[SessionInfo]:
        """
        List all registered sessions.

        Returns:
            List of SessionInfo for all sessions
        """
        result = []
        with self._registry_lock:
            session_ids = list(self._sessions.keys())

        for session_id in session_ids:
            info = self.get_info(session_id)
            if info is not None:
                result.append(info)

        return result

    async def close_session(
        self,
        session_id: str,
        *,
        emit_session_ended: bool = True,
    ) -> bool:
        """
        Force close a session with proper cleanup.

        This method:
        1. Cancels all pending async operations
        2. Emits SESSION_ENDED event if possible
        3. Closes WebSocket connection
        4. Clears replay buffer
        5. Removes session from registry

        Args:
            session_id: Session ID to close
            emit_session_ended: Whether to emit SESSION_ENDED event (default: True)

        Returns:
            True if session was found and closed, False otherwise
        """
        registered = self.get(session_id)
        if registered is None:
            return False

        session = registered.session
        events_to_send: list[Any] = []

        try:
            # 1. Cancel pending tasks
            cancelled_count = registered.cancel_pending_tasks()
            if cancelled_count > 0:
                logger.info(
                    "Cancelled %d pending tasks for session %s",
                    cancelled_count,
                    session_id,
                )

            # 2. End the session and collect final events
            if session.state.value == "active":
                try:
                    final_events = await session.end()
                    if emit_session_ended:
                        events_to_send.extend(final_events)
                except Exception as e:
                    logger.warning(
                        "Error ending session %s during force close: %s",
                        session_id,
                        e,
                    )
                    # Create error event if session end failed
                    if emit_session_ended:
                        try:
                            error_event = session.create_error_event(
                                code="FORCE_CLOSE",
                                message=f"Session force closed: {e}",
                                recoverable=False,
                            )
                            events_to_send.append(error_event)
                        except Exception:
                            pass

            # 3. Send final events via WebSocket if connected
            if registered.websocket is not None and events_to_send:
                try:
                    for event in events_to_send:
                        await registered.websocket.send_json(event.to_dict())
                except Exception as e:
                    logger.warning(
                        "Error sending final events for %s: %s",
                        session_id,
                        e,
                    )

            # 4. Close WebSocket connection
            if registered.websocket is not None:
                try:
                    await registered.websocket.close(code=1000, reason="Session closed")
                except Exception as e:
                    logger.warning("Error closing WebSocket for %s: %s", session_id, e)

            # 5. Clear replay buffer
            try:
                session._replay_buffer.clear()
            except Exception:
                pass

            # 6. Transition to ended and unregister
            try:
                registered.transition_to(SessionStatus.ENDED)
            except InvalidStateTransitionError:
                pass  # Already in terminal state

            self.unregister(session_id)
            logger.info("Force closed session: %s", session_id)
            return True

        except Exception as e:
            logger.error("Error closing session %s: %s", session_id, e)
            # Still try to unregister
            self.unregister(session_id)
            return True

    def update_websocket(self, session_id: str, websocket: Any) -> bool:
        """
        Update WebSocket connection for a session.

        When websocket is set to None (disconnection), the session status
        transitions to DISCONNECTED (not ENDED). The session can be resumed
        within the disconnected TTL window.

        Args:
            session_id: Session ID
            websocket: New WebSocket connection (or None for disconnection)

        Returns:
            True if session found and updated, False otherwise
        """
        registered = self.get(session_id)
        if registered is None:
            return False

        old_websocket = registered.websocket
        registered.websocket = websocket
        registered.update_activity()

        # Handle disconnection: transition to DISCONNECTED (not ENDED)
        if websocket is None and old_websocket is not None:
            if registered.status == SessionStatus.ACTIVE:
                try:
                    registered.transition_to(SessionStatus.DISCONNECTED)
                    logger.info(
                        "Session %s disconnected, can reconnect within %s seconds",
                        session_id,
                        self._disconnected_ttl_sec,
                    )
                except InvalidStateTransitionError as e:
                    logger.warning("Failed to transition to disconnected: %s", e)

        # Handle reconnection: transition back to ACTIVE
        elif websocket is not None and old_websocket is None:
            if registered.status == SessionStatus.DISCONNECTED:
                try:
                    registered.transition_to(SessionStatus.ACTIVE)
                    registered.session.stats.resume_attempts += 1
                    logger.info("Session %s reconnected", session_id)
                except InvalidStateTransitionError as e:
                    logger.warning("Failed to transition to active on reconnect: %s", e)

        return True

    def mark_disconnected(self, session_id: str) -> bool:
        """
        Mark a session as disconnected.

        This is called when a WebSocket connection is lost. The session
        remains available for reconnection within the TTL window.

        Args:
            session_id: Session ID

        Returns:
            True if session was marked disconnected, False otherwise
        """
        registered = self.get(session_id)
        if registered is None:
            return False

        if registered.status == SessionStatus.ACTIVE:
            try:
                registered.websocket = None
                registered.transition_to(SessionStatus.DISCONNECTED)
                return True
            except InvalidStateTransitionError as e:
                logger.warning("Failed to mark session disconnected: %s", e)
                return False

        return False

    def mark_active(self, session_id: str) -> bool:
        """
        Mark a session as active.

        This syncs the registry status when a session starts.

        Args:
            session_id: Session ID

        Returns:
            True if session was marked active, False otherwise
        """
        registered = self.get(session_id)
        if registered is None:
            return False

        if registered.status == SessionStatus.CREATED:
            try:
                registered.transition_to(SessionStatus.ACTIVE)
                return True
            except InvalidStateTransitionError as e:
                logger.warning("Failed to mark session active: %s", e)
                return False

        return registered.status == SessionStatus.ACTIVE

    def touch(self, session_id: str) -> bool:
        """
        Update last activity timestamp for a session.

        Args:
            session_id: Session ID

        Returns:
            True if session found, False otherwise
        """
        registered = self.get(session_id)
        if registered is None:
            return False

        registered.update_activity()
        return True

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions based on TTL rules.

        Cleanup rules:
        1. Sessions in terminal states (ended, error) are removed immediately
        2. Disconnected sessions are removed after disconnected_ttl_sec
        3. Idle sessions (no activity) are removed after idle_timeout_sec
           only if they have no active WebSocket connection

        Returns:
            Number of sessions cleaned up
        """
        now = time.time()
        to_cleanup: list[tuple[str, str]] = []  # (session_id, reason)

        with self._registry_lock:
            for session_id, registered in self._sessions.items():
                # Rule 1: Terminal states - clean up immediately
                if SessionStatus.is_terminal(registered.status):
                    to_cleanup.append((session_id, "terminal_state"))
                    continue

                # Rule 2: Disconnected sessions - check disconnected TTL
                if registered.status == SessionStatus.DISCONNECTED:
                    if registered.disconnected_at is not None:
                        disconnected_duration = now - registered.disconnected_at
                        if disconnected_duration > self._disconnected_ttl_sec:
                            to_cleanup.append((session_id, "disconnected_ttl"))
                            continue
                    # Fallback: check last_activity if disconnected_at not set
                    elif now - registered.last_activity > self._disconnected_ttl_sec:
                        to_cleanup.append((session_id, "disconnected_idle"))
                        continue

                # Rule 3: Idle sessions without WebSocket
                if registered.websocket is None:
                    if now - registered.last_activity > self._idle_timeout_sec:
                        to_cleanup.append((session_id, "idle_timeout"))
                        continue

        # Perform cleanup
        count = 0
        for session_id, reason in to_cleanup:
            # For disconnected sessions exceeding TTL, transition to ended first
            cleanup_registered = self.get(session_id)
            if cleanup_registered and cleanup_registered.status == SessionStatus.DISCONNECTED:
                try:
                    cleanup_registered.transition_to(SessionStatus.ENDED)
                except InvalidStateTransitionError:
                    pass

            if await self.close_session(session_id, emit_session_ended=False):
                count += 1
                logger.info("Cleaned up session %s (reason: %s)", session_id, reason)

        if count > 0:
            logger.info("Cleanup completed: %d sessions removed", count)

        return count

    # Keep old method name as alias for backwards compatibility
    async def cleanup_idle_sessions(self) -> int:
        """Alias for cleanup_expired_sessions (backwards compatibility)."""
        return await self.cleanup_expired_sessions()

    async def start_cleanup_task(self) -> asyncio.Task[None]:
        """
        Start the background cleanup task.

        The cleanup task runs periodically to remove expired sessions.
        It can be stopped by calling stop_cleanup_task().

        Returns:
            The cleanup task (can be awaited or cancelled)
        """
        if self._cleanup_task is not None and not self._cleanup_task.done():
            logger.warning("Cleanup task already running")
            return self._cleanup_task

        self._shutdown_event = asyncio.Event()

        shutdown_event = self._shutdown_event

        async def cleanup_loop() -> None:
            """Background loop that periodically cleans up expired sessions."""
            logger.info(
                "Starting cleanup task (interval=%ss)",
                self._cleanup_interval_sec,
            )
            assert shutdown_event is not None  # Guaranteed by caller
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=self._cleanup_interval_sec,
                    )
                    # shutdown_event was set, exit loop
                    break
                except TimeoutError:
                    # Normal timeout, run cleanup
                    try:
                        await self.cleanup_expired_sessions()
                    except Exception as e:
                        logger.error("Error in cleanup task: %s", e)

            logger.info("Cleanup task stopped")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        return self._cleanup_task

    async def stop_cleanup_task(self) -> None:
        """
        Stop the background cleanup task.

        This signals the cleanup task to stop and waits for it to complete.
        """
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        if self._cleanup_task is not None:
            try:
                # Wait briefly for graceful shutdown
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except TimeoutError:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            self._cleanup_task = None

        logger.info("Cleanup task stopped")

    def get_stats(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry stats including:
            - total_sessions: Total number of registered sessions
            - by_status: Breakdown by session status
            - connected_count: Number of sessions with active WebSocket
            - idle_timeout_sec: Configured idle timeout
            - disconnected_ttl_sec: Configured disconnected TTL
            - cleanup_task_running: Whether cleanup task is active
        """
        with self._registry_lock:
            total = len(self._sessions)
            by_status: dict[str, int] = {}
            connected_count = 0
            for registered in self._sessions.values():
                status = registered.status.value
                by_status[status] = by_status.get(status, 0) + 1
                if registered.websocket is not None:
                    connected_count += 1

        return {
            "total_sessions": total,
            "by_status": by_status,
            "connected_count": connected_count,
            "idle_timeout_sec": self._idle_timeout_sec,
            "disconnected_ttl_sec": self._disconnected_ttl_sec,
            "cleanup_interval_sec": self._cleanup_interval_sec,
            "cleanup_task_running": self._cleanup_task is not None
            and not self._cleanup_task.done(),
        }


def get_registry() -> SessionRegistry:
    """
    Get the global session registry instance.

    Returns:
        SessionRegistry singleton instance
    """
    return SessionRegistry()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Registry
    "SessionRegistry",
    "get_registry",
    # Data classes
    "SessionInfo",
    "SessionStatus",
    "RegisteredSession",
    # Exceptions
    "InvalidStateTransitionError",
    # Constants
    "VALID_TRANSITIONS",
]
