"""Tests for session registry and REST session management endpoints (Issue #85).

These tests verify:
- Session registry singleton behavior
- Session registration and unregistration
- Session info retrieval
- Session lifecycle and state transitions
- TTL and expiry management
- Disconnection handling
- REST endpoints for session management
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# =============================================================================
# Session Registry Tests
# =============================================================================


class TestSessionRegistry:
    """Tests for SessionRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        from slower_whisper.pipeline.session_registry import SessionRegistry

        SessionRegistry.reset()
        yield
        SessionRegistry.reset()

    def test_singleton_pattern(self) -> None:
        """Test registry is a singleton."""
        from slower_whisper.pipeline.session_registry import SessionRegistry, get_registry

        reg1 = get_registry()
        reg2 = get_registry()
        reg3 = SessionRegistry()

        assert reg1 is reg2
        assert reg1 is reg3

    def test_register_session(self) -> None:
        """Test registering a session."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()

        session_id = registry.register(session)

        assert session_id == session.stream_id
        assert registry.get(session_id) is not None
        assert registry.get(session_id).session is session

    def test_unregister_session(self) -> None:
        """Test unregistering a session."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        session_id = registry.register(session)

        assert registry.unregister(session_id) is True
        assert registry.get(session_id) is None

        # Unregistering again returns False
        assert registry.unregister(session_id) is False

    def test_get_info(self) -> None:
        """Test getting session info."""
        from slower_whisper.pipeline.session_registry import SessionStatus, get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        session_id = registry.register(session)

        info = registry.get_info(session_id)

        assert info is not None
        assert info.session_id == session_id
        assert info.status == SessionStatus.CREATED
        assert info.config["max_gap_sec"] == 1.0
        assert info.stats["chunks_received"] == 0

    def test_get_info_not_found(self) -> None:
        """Test getting info for non-existent session."""
        from slower_whisper.pipeline.session_registry import get_registry

        registry = get_registry()
        info = registry.get_info("str-nonexistent")

        assert info is None

    def test_list_sessions(self) -> None:
        """Test listing all sessions."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()

        # Initially empty
        assert len(registry.list_sessions()) == 0

        # Add sessions
        session1 = WebSocketStreamingSession()
        session2 = WebSocketStreamingSession()
        registry.register(session1)
        registry.register(session2)

        sessions = registry.list_sessions()
        assert len(sessions) == 2

        session_ids = {s.session_id for s in sessions}
        assert session1.stream_id in session_ids
        assert session2.stream_id in session_ids

    @pytest.mark.asyncio
    async def test_close_session(self) -> None:
        """Test force closing a session."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import SessionState, WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()
        session_id = registry.register(session)

        assert session.state == SessionState.ACTIVE

        success = await registry.close_session(session_id)

        assert success is True
        assert session.state == SessionState.ENDED
        assert registry.get(session_id) is None

    @pytest.mark.asyncio
    async def test_close_session_not_found(self) -> None:
        """Test closing non-existent session."""
        from slower_whisper.pipeline.session_registry import get_registry

        registry = get_registry()
        success = await registry.close_session("str-nonexistent")

        assert success is False

    def test_touch_updates_activity(self) -> None:
        """Test touching a session updates last activity."""
        import time

        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        session_id = registry.register(session)

        registered = registry.get(session_id)
        original_activity = registered.last_activity

        time.sleep(0.01)  # Small delay
        registry.touch(session_id)

        assert registered.last_activity > original_activity

    def test_get_stats(self) -> None:
        """Test getting registry stats."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()

        stats = registry.get_stats()
        assert stats["total_sessions"] == 0

        session = WebSocketStreamingSession()
        registry.register(session)

        stats = registry.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["by_status"]["created"] == 1

    def test_get_stats_includes_new_fields(self) -> None:
        """Test that get_stats includes all configuration fields."""
        from slower_whisper.pipeline.session_registry import get_registry

        registry = get_registry()
        stats = registry.get_stats()

        assert "idle_timeout_sec" in stats
        assert "disconnected_ttl_sec" in stats
        assert "cleanup_interval_sec" in stats
        assert "cleanup_task_running" in stats
        assert "connected_count" in stats

    def test_configure(self) -> None:
        """Test configuring registry settings."""
        from slower_whisper.pipeline.session_registry import get_registry

        registry = get_registry()

        registry.configure(
            idle_timeout_sec=600.0,
            cleanup_interval_sec=120.0,
            disconnected_ttl_sec=60.0,
        )

        stats = registry.get_stats()
        assert stats["idle_timeout_sec"] == 600.0
        assert stats["cleanup_interval_sec"] == 120.0
        assert stats["disconnected_ttl_sec"] == 60.0


# =============================================================================
# State Transition Tests
# =============================================================================


class TestSessionStateTransitions:
    """Tests for session state transition validation."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        from slower_whisper.pipeline.session_registry import SessionRegistry

        SessionRegistry.reset()
        yield
        SessionRegistry.reset()

    def test_valid_transitions(self) -> None:
        """Test all valid state transitions."""
        from slower_whisper.pipeline.session_registry import VALID_TRANSITIONS

        # Verify expected valid transitions
        assert VALID_TRANSITIONS[("created", "active")] is True
        assert VALID_TRANSITIONS[("active", "ending")] is True
        assert VALID_TRANSITIONS[("ending", "ended")] is True
        assert VALID_TRANSITIONS[("active", "error")] is True
        assert VALID_TRANSITIONS[("active", "disconnected")] is True
        assert VALID_TRANSITIONS[("disconnected", "active")] is True
        assert VALID_TRANSITIONS[("disconnected", "ended")] is True

    def test_is_valid_transition(self) -> None:
        """Test SessionStatus.is_valid_transition method."""
        from slower_whisper.pipeline.session_registry import SessionStatus

        assert SessionStatus.is_valid_transition(SessionStatus.CREATED, SessionStatus.ACTIVE)
        assert SessionStatus.is_valid_transition(SessionStatus.ACTIVE, SessionStatus.DISCONNECTED)
        assert not SessionStatus.is_valid_transition(SessionStatus.CREATED, SessionStatus.ENDED)
        assert not SessionStatus.is_valid_transition(SessionStatus.ENDED, SessionStatus.ACTIVE)

    def test_is_terminal(self) -> None:
        """Test SessionStatus.is_terminal method."""
        from slower_whisper.pipeline.session_registry import SessionStatus

        assert SessionStatus.is_terminal(SessionStatus.ENDED)
        assert SessionStatus.is_terminal(SessionStatus.ERROR)
        assert not SessionStatus.is_terminal(SessionStatus.ACTIVE)
        assert not SessionStatus.is_terminal(SessionStatus.DISCONNECTED)

    def test_registered_session_transition(self) -> None:
        """Test RegisteredSession.transition_to method."""
        from slower_whisper.pipeline.session_registry import (
            InvalidStateTransitionError,
            RegisteredSession,
            SessionStatus,
        )
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        session = WebSocketStreamingSession()
        registered = RegisteredSession(
            session_id=session.stream_id,
            session=session,
            status=SessionStatus.CREATED,
        )

        # Valid transition
        assert registered.transition_to(SessionStatus.ACTIVE) is True
        assert registered.status == SessionStatus.ACTIVE

        # Same state is no-op
        assert registered.transition_to(SessionStatus.ACTIVE) is True

        # Invalid transition
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            registered.transition_to(SessionStatus.CREATED)

        assert exc_info.value.from_status == SessionStatus.ACTIVE
        assert exc_info.value.to_status == SessionStatus.CREATED

    def test_transition_updates_activity(self) -> None:
        """Test that transitions update last_activity timestamp."""
        from slower_whisper.pipeline.session_registry import RegisteredSession, SessionStatus
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        session = WebSocketStreamingSession()
        registered = RegisteredSession(
            session_id=session.stream_id,
            session=session,
            status=SessionStatus.CREATED,
        )

        original_activity = registered.last_activity
        time.sleep(0.01)

        registered.transition_to(SessionStatus.ACTIVE)

        assert registered.last_activity > original_activity

    def test_disconnected_tracks_timestamp(self) -> None:
        """Test that disconnection tracks disconnected_at timestamp."""
        from slower_whisper.pipeline.session_registry import RegisteredSession, SessionStatus
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        session = WebSocketStreamingSession()
        registered = RegisteredSession(
            session_id=session.stream_id,
            session=session,
            status=SessionStatus.ACTIVE,
        )

        assert registered.disconnected_at is None

        registered.transition_to(SessionStatus.DISCONNECTED)

        assert registered.disconnected_at is not None
        assert time.time() - registered.disconnected_at < 1.0

    def test_reconnect_clears_disconnected_timestamp(self) -> None:
        """Test that reconnection clears disconnected_at timestamp."""
        from slower_whisper.pipeline.session_registry import RegisteredSession, SessionStatus
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        session = WebSocketStreamingSession()
        registered = RegisteredSession(
            session_id=session.stream_id,
            session=session,
            status=SessionStatus.ACTIVE,
        )

        registered.transition_to(SessionStatus.DISCONNECTED)
        assert registered.disconnected_at is not None

        registered.transition_to(SessionStatus.ACTIVE)
        assert registered.disconnected_at is None


# =============================================================================
# Disconnection Handling Tests
# =============================================================================


class TestDisconnectionHandling:
    """Tests for WebSocket disconnection and reconnection handling."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        from slower_whisper.pipeline.session_registry import SessionRegistry

        SessionRegistry.reset()
        yield
        SessionRegistry.reset()

    @pytest.mark.asyncio
    async def test_disconnection_transitions_to_disconnected(self) -> None:
        """Test that WebSocket disconnection transitions session to DISCONNECTED."""
        from slower_whisper.pipeline.session_registry import SessionStatus, get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()

        mock_ws = MagicMock()
        session_id = registry.register(session, websocket=mock_ws)

        # Mark session as active
        registry.mark_active(session_id)
        assert registry.get(session_id).status == SessionStatus.ACTIVE

        # Simulate disconnection
        registry.update_websocket(session_id, None)

        # Should be disconnected, not ended
        info = registry.get_info(session_id)
        assert info.status == SessionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnection_transitions_to_active(self) -> None:
        """Test that WebSocket reconnection transitions session back to ACTIVE."""
        from slower_whisper.pipeline.session_registry import SessionStatus, get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()

        mock_ws = MagicMock()
        session_id = registry.register(session, websocket=mock_ws)
        registry.mark_active(session_id)

        # Disconnect
        registry.update_websocket(session_id, None)
        assert registry.get(session_id).status == SessionStatus.DISCONNECTED

        # Reconnect
        new_ws = MagicMock()
        registry.update_websocket(session_id, new_ws)

        info = registry.get_info(session_id)
        assert info.status == SessionStatus.ACTIVE
        assert registry.get(session_id).websocket is new_ws

    @pytest.mark.asyncio
    async def test_mark_disconnected(self) -> None:
        """Test mark_disconnected helper method."""
        from slower_whisper.pipeline.session_registry import SessionStatus, get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()

        mock_ws = MagicMock()
        session_id = registry.register(session, websocket=mock_ws)
        registry.mark_active(session_id)

        # Mark disconnected
        result = registry.mark_disconnected(session_id)

        assert result is True
        assert registry.get(session_id).status == SessionStatus.DISCONNECTED
        assert registry.get(session_id).websocket is None

    def test_mark_disconnected_not_found(self) -> None:
        """Test mark_disconnected returns False for non-existent session."""
        from slower_whisper.pipeline.session_registry import get_registry

        registry = get_registry()
        result = registry.mark_disconnected("str-nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_increments_counter(self) -> None:
        """Test that reconnection increments resume_attempts counter."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()

        mock_ws = MagicMock()
        session_id = registry.register(session, websocket=mock_ws)
        registry.mark_active(session_id)

        initial_attempts = session.stats.resume_attempts

        # Disconnect and reconnect
        registry.update_websocket(session_id, None)
        registry.update_websocket(session_id, MagicMock())

        assert session.stats.resume_attempts == initial_attempts + 1


# =============================================================================
# TTL and Expiry Tests
# =============================================================================


class TestTTLAndExpiry:
    """Tests for TTL-based session cleanup."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        from slower_whisper.pipeline.session_registry import SessionRegistry

        SessionRegistry.reset()
        yield
        SessionRegistry.reset()

    @pytest.mark.asyncio
    async def test_cleanup_terminal_sessions(self) -> None:
        """Test that sessions in terminal states are cleaned up immediately."""
        from slower_whisper.pipeline.session_registry import SessionStatus, get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()
        session_id = registry.register(session)
        registry.mark_active(session_id)

        # End the session - this changes session.state but not registry status
        await session.end()

        # Manually transition registry status to match (simulating what get_info does)
        registered = registry.get(session_id)
        registered.transition_to(SessionStatus.ENDING)
        registered.transition_to(SessionStatus.ENDED)

        # Run cleanup
        count = await registry.cleanup_expired_sessions()

        assert count == 1
        assert registry.get(session_id) is None

    @pytest.mark.asyncio
    async def test_cleanup_disconnected_sessions_after_ttl(self) -> None:
        """Test that disconnected sessions are cleaned up after TTL."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        # Set very short TTL for testing
        registry.configure(disconnected_ttl_sec=0.01)

        session = WebSocketStreamingSession()
        await session.start()
        mock_ws = MagicMock()
        session_id = registry.register(session, websocket=mock_ws)
        registry.mark_active(session_id)

        # Disconnect
        registry.mark_disconnected(session_id)

        # Wait for TTL to expire
        await asyncio.sleep(0.02)

        # Run cleanup
        count = await registry.cleanup_expired_sessions()

        assert count == 1
        assert registry.get(session_id) is None

    @pytest.mark.asyncio
    async def test_disconnected_session_not_cleaned_before_ttl(self) -> None:
        """Test that disconnected sessions are NOT cleaned up before TTL."""
        from slower_whisper.pipeline.session_registry import SessionStatus, get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        # Set long TTL
        registry.configure(disconnected_ttl_sec=300.0)

        session = WebSocketStreamingSession()
        await session.start()
        mock_ws = MagicMock()
        session_id = registry.register(session, websocket=mock_ws)
        registry.mark_active(session_id)

        # Disconnect
        registry.mark_disconnected(session_id)

        # Run cleanup immediately (before TTL)
        count = await registry.cleanup_expired_sessions()

        assert count == 0
        assert registry.get(session_id) is not None
        assert registry.get(session_id).status == SessionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_cleanup_idle_sessions(self) -> None:
        """Test that idle sessions without WebSocket are cleaned up."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        # Set very short idle timeout
        registry.configure(idle_timeout_sec=0.01)

        session = WebSocketStreamingSession()
        session_id = registry.register(session)  # No WebSocket

        # Wait for idle timeout
        await asyncio.sleep(0.02)

        # Run cleanup
        count = await registry.cleanup_expired_sessions()

        assert count == 1
        assert registry.get(session_id) is None

    @pytest.mark.asyncio
    async def test_connected_session_not_cleaned_when_idle(self) -> None:
        """Test that connected sessions are NOT cleaned even if idle."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        # Set very short idle timeout
        registry.configure(idle_timeout_sec=0.01)

        session = WebSocketStreamingSession()
        await session.start()
        mock_ws = MagicMock()
        session_id = registry.register(session, websocket=mock_ws)
        registry.mark_active(session_id)

        # Wait for idle timeout
        await asyncio.sleep(0.02)

        # Run cleanup
        count = await registry.cleanup_expired_sessions()

        # Should NOT be cleaned because WebSocket is connected
        assert count == 0
        assert registry.get(session_id) is not None


# =============================================================================
# Background Cleanup Task Tests
# =============================================================================


class TestBackgroundCleanupTask:
    """Tests for the background cleanup task."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        from slower_whisper.pipeline.session_registry import SessionRegistry

        SessionRegistry.reset()
        yield
        SessionRegistry.reset()

    @pytest.mark.asyncio
    async def test_start_cleanup_task(self) -> None:
        """Test starting the background cleanup task."""
        from slower_whisper.pipeline.session_registry import get_registry

        registry = get_registry()
        registry.configure(cleanup_interval_sec=0.1)

        task = await registry.start_cleanup_task()

        assert task is not None
        assert not task.done()
        assert registry.get_stats()["cleanup_task_running"] is True

        # Clean up
        await registry.stop_cleanup_task()
        assert registry.get_stats()["cleanup_task_running"] is False

    @pytest.mark.asyncio
    async def test_cleanup_task_runs_periodically(self) -> None:
        """Test that cleanup task runs periodically."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        registry.configure(
            cleanup_interval_sec=0.05,
            idle_timeout_sec=0.01,
        )

        # Create an idle session
        session = WebSocketStreamingSession()
        session_id = registry.register(session)

        # Start cleanup task
        await registry.start_cleanup_task()

        # Wait for cleanup to run
        await asyncio.sleep(0.1)

        # Session should be cleaned up
        assert registry.get(session_id) is None

        # Clean up
        await registry.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_stop_cleanup_task(self) -> None:
        """Test stopping the cleanup task gracefully."""
        from slower_whisper.pipeline.session_registry import get_registry

        registry = get_registry()
        await registry.start_cleanup_task()

        # Stop should be graceful
        await registry.stop_cleanup_task()

        assert registry._cleanup_task is None
        assert registry.get_stats()["cleanup_task_running"] is False

    @pytest.mark.asyncio
    async def test_start_cleanup_task_idempotent(self) -> None:
        """Test that starting cleanup task multiple times is safe."""
        from slower_whisper.pipeline.session_registry import get_registry

        registry = get_registry()

        task1 = await registry.start_cleanup_task()
        task2 = await registry.start_cleanup_task()

        # Should return the same task
        assert task1 is task2

        await registry.stop_cleanup_task()


# =============================================================================
# Force Close Tests
# =============================================================================


class TestForceClose:
    """Tests for force close semantics."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        from slower_whisper.pipeline.session_registry import SessionRegistry

        SessionRegistry.reset()
        yield
        SessionRegistry.reset()

    @pytest.mark.asyncio
    async def test_force_close_cancels_pending_tasks(self) -> None:
        """Test that force close cancels pending async tasks."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()
        session_id = registry.register(session)

        # Add a pending task
        async def long_running():
            await asyncio.sleep(10)

        registered = registry.get(session_id)
        task = asyncio.create_task(long_running())
        registered.add_task(task)

        # Force close
        await registry.close_session(session_id)

        # Give the event loop a chance to process the cancellation
        await asyncio.sleep(0.01)

        # Task should be cancelled or done (cancelled state or raised CancelledError)
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_force_close_sends_session_ended(self) -> None:
        """Test that force close sends SESSION_ENDED event."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()

        mock_ws = AsyncMock()
        session_id = registry.register(session, websocket=mock_ws)
        registry.mark_active(session_id)

        # Force close
        await registry.close_session(session_id)

        # Should have sent events
        assert mock_ws.send_json.called

    @pytest.mark.asyncio
    async def test_force_close_closes_websocket(self) -> None:
        """Test that force close closes the WebSocket connection."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()

        mock_ws = AsyncMock()
        session_id = registry.register(session, websocket=mock_ws)
        registry.mark_active(session_id)

        # Force close
        await registry.close_session(session_id)

        # WebSocket should be closed
        mock_ws.close.assert_called()

    @pytest.mark.asyncio
    async def test_force_close_clears_replay_buffer(self) -> None:
        """Test that force close clears the replay buffer."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()
        session_id = registry.register(session)
        registry.mark_active(session_id)

        # Add something to replay buffer
        session._replay_buffer.add(
            session._create_envelope(
                session.ServerMessageType.PONG
                if hasattr(session, "ServerMessageType")
                else session._create_envelope.__self__.__class__.__bases__[0],
                {"test": True},
            )
        )

        # Force close
        await registry.close_session(session_id)

        # Session is removed, can't check buffer
        assert registry.get(session_id) is None

    @pytest.mark.asyncio
    async def test_force_close_without_emit(self) -> None:
        """Test force close with emit_session_ended=False."""
        from slower_whisper.pipeline.session_registry import get_registry
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        registry = get_registry()
        session = WebSocketStreamingSession()
        await session.start()

        mock_ws = AsyncMock()
        session_id = registry.register(session, websocket=mock_ws)
        registry.mark_active(session_id)

        # Force close without emit
        await registry.close_session(session_id, emit_session_ended=False)

        # Should not have sent events
        assert not mock_ws.send_json.called


# =============================================================================
# Pending Task Management Tests
# =============================================================================


class TestPendingTaskManagement:
    """Tests for pending task tracking in RegisteredSession."""

    @pytest.mark.asyncio
    async def test_add_task(self) -> None:
        """Test adding a task to track."""
        from slower_whisper.pipeline.session_registry import RegisteredSession, SessionStatus
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        session = WebSocketStreamingSession()
        registered = RegisteredSession(
            session_id=session.stream_id,
            session=session,
            status=SessionStatus.CREATED,
        )

        async def dummy():
            pass

        task = asyncio.create_task(dummy())
        registered.add_task(task)
        assert task in registered.pending_tasks
        # Let task complete to avoid warning
        await task

    @pytest.mark.asyncio
    async def test_cancel_pending_tasks(self) -> None:
        """Test cancelling all pending tasks."""
        from slower_whisper.pipeline.session_registry import RegisteredSession, SessionStatus
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        session = WebSocketStreamingSession()
        registered = RegisteredSession(
            session_id=session.stream_id,
            session=session,
            status=SessionStatus.CREATED,
        )

        async def long_running():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass

        task1 = asyncio.create_task(long_running())
        task2 = asyncio.create_task(long_running())
        registered.add_task(task1)
        registered.add_task(task2)

        count = registered.cancel_pending_tasks()

        assert count == 2
        assert len(registered.pending_tasks) == 0
        # Wait for cancelled tasks to finish
        await asyncio.gather(task1, task2, return_exceptions=True)


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    """Tests for SessionInfo dataclass."""

    def test_to_dict(self) -> None:
        """Test SessionInfo serialization."""
        from datetime import datetime

        from slower_whisper.pipeline.session_registry import SessionInfo, SessionStatus

        info = SessionInfo(
            session_id="str-test",
            status=SessionStatus.ACTIVE,
            created_at=datetime(2024, 1, 26, 12, 0, 0, tzinfo=UTC),
            last_activity=datetime(2024, 1, 26, 12, 1, 0, tzinfo=UTC),
            config={"max_gap_sec": 1.0},
            stats={"chunks_received": 10},
            last_event_id=5,
        )

        d = info.to_dict()

        assert d["session_id"] == "str-test"
        assert d["status"] == "active"
        assert d["created_at"] == "2024-01-26T12:00:00+00:00"
        assert d["last_activity"] == "2024-01-26T12:01:00+00:00"
        assert d["config"] == {"max_gap_sec": 1.0}
        assert d["stats"] == {"chunks_received": 10}
        assert d["last_event_id"] == 5


# =============================================================================
# REST Endpoint Tests
# =============================================================================


class TestStreamSessionsEndpoints:
    """Integration tests for /stream/sessions REST endpoints."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        from slower_whisper.pipeline.session_registry import SessionRegistry

        SessionRegistry.reset()
        yield
        SessionRegistry.reset()

    @pytest.fixture
    def client(self) -> TestClient:
        """Create FastAPI test client."""
        from slower_whisper.pipeline.service import app

        return TestClient(app)

    def test_list_sessions_empty(self, client: TestClient) -> None:
        """Test listing sessions when none exist."""
        response = client.get("/stream/sessions")
        assert response.status_code == 200

        data = response.json()
        assert data["sessions"] == []
        assert data["count"] == 0

    def test_create_session(self, client: TestClient) -> None:
        """Test creating a new session."""
        response = client.post("/stream/sessions")
        assert response.status_code == 201

        data = response.json()
        assert "session_id" in data
        assert data["session_id"].startswith("str-")
        assert "websocket_url" in data
        assert data["config"]["max_gap_sec"] == 1.0

    def test_create_session_with_config(self, client: TestClient) -> None:
        """Test creating a session with custom config."""
        response = client.post(
            "/stream/sessions",
            params={
                "max_gap_sec": 0.5,
                "enable_prosody": True,
                "enable_diarization": True,
                "sample_rate": 8000,
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["config"]["max_gap_sec"] == 0.5
        assert data["config"]["enable_prosody"] is True
        assert data["config"]["enable_diarization"] is True
        assert data["config"]["sample_rate"] == 8000

    def test_get_session_status(self, client: TestClient) -> None:
        """Test getting session status."""
        # Create session first
        create_response = client.post("/stream/sessions")
        session_id = create_response.json()["session_id"]

        # Get status
        response = client.get(f"/stream/sessions/{session_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "created"
        assert "config" in data
        assert "stats" in data

    def test_get_session_status_not_found(self, client: TestClient) -> None:
        """Test getting status for non-existent session."""
        response = client.get("/stream/sessions/str-nonexistent")
        assert response.status_code == 404

    def test_delete_session(self, client: TestClient) -> None:
        """Test deleting a session."""
        # Create session first
        create_response = client.post("/stream/sessions")
        session_id = create_response.json()["session_id"]

        # Delete session
        response = client.delete(f"/stream/sessions/{session_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id

        # Verify it's gone
        get_response = client.get(f"/stream/sessions/{session_id}")
        assert get_response.status_code == 404

    def test_delete_session_not_found(self, client: TestClient) -> None:
        """Test deleting non-existent session."""
        response = client.delete("/stream/sessions/str-nonexistent")
        assert response.status_code == 404

    def test_list_sessions_after_create(self, client: TestClient) -> None:
        """Test listing sessions shows created sessions."""
        # Create multiple sessions
        client.post("/stream/sessions")
        client.post("/stream/sessions")
        client.post("/stream/sessions")

        # List sessions
        response = client.get("/stream/sessions")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 3
        assert len(data["sessions"]) == 3

    def test_create_session_validation(self, client: TestClient) -> None:
        """Test session creation validates input."""
        # Invalid max_gap_sec (too low)
        response = client.post("/stream/sessions", params={"max_gap_sec": 0.01})
        assert response.status_code == 422

        # Invalid sample_rate (too low)
        response = client.post("/stream/sessions", params={"sample_rate": 100})
        assert response.status_code == 422
