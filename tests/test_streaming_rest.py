"""Integration tests for REST streaming endpoints (Issue #85).

These tests verify the REST endpoints for streaming session management:
- POST /stream/sessions (create session)
- GET /stream/sessions (list sessions)
- GET /stream/sessions/{session_id} (get status)
- DELETE /stream/sessions/{session_id} (close session)
- GET /stream/config (default config)

The REST endpoints provide session management for WebSocket streaming,
enabling clients to create sessions, monitor status, and force-close sessions.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestStreamConfigEndpoint:
    """Tests for GET /stream/config endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create FastAPI test client."""
        from slower_whisper.pipeline.service import app

        return TestClient(app)

    def test_get_default_config(self, client: TestClient) -> None:
        """Test retrieving default streaming configuration."""
        response = client.get("/stream/config")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert "default_config" in data
        assert "supported_audio_formats" in data
        assert "supported_sample_rates" in data
        assert "message_types" in data

        # Verify default config values
        config = data["default_config"]
        assert config["max_gap_sec"] == 1.0
        assert config["enable_prosody"] is False
        assert config["enable_emotion"] is False
        assert config["sample_rate"] == 16000
        assert config["audio_format"] == "pcm_s16le"

        # Verify supported formats
        assert "pcm_s16le" in data["supported_audio_formats"]
        assert 16000 in data["supported_sample_rates"]

        # Verify message types
        assert "client" in data["message_types"]
        assert "server" in data["message_types"]
        assert "START_SESSION" in data["message_types"]["client"]
        assert "AUDIO_CHUNK" in data["message_types"]["client"]
        assert "END_SESSION" in data["message_types"]["client"]
        assert "SESSION_STARTED" in data["message_types"]["server"]
        assert "FINALIZED" in data["message_types"]["server"]
        assert "SESSION_ENDED" in data["message_types"]["server"]


class TestStreamSessionsEndpoints:
    """Tests for REST session management endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create FastAPI test client."""
        from slower_whisper.pipeline.service import app

        return TestClient(app)

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> None:
        """Reset session registry before each test."""
        from slower_whisper.pipeline.session_registry import SessionRegistry

        SessionRegistry.reset()
        yield
        SessionRegistry.reset()

    def test_create_session_default_config(self, client: TestClient) -> None:
        """Test creating a session with default configuration."""
        response = client.post("/stream/sessions")

        assert response.status_code == 201
        data = response.json()

        # Verify response structure
        assert "session_id" in data
        assert "websocket_url" in data
        assert "config" in data

        # Verify session_id format (str-{uuid})
        assert data["session_id"].startswith("str-")

        # Verify websocket URL contains session_id
        assert data["session_id"] in data["websocket_url"]

        # Verify config has default values
        config = data["config"]
        assert config["max_gap_sec"] == 1.0
        assert config["enable_prosody"] is False
        assert config["enable_emotion"] is False
        assert config["enable_diarization"] is False
        assert config["sample_rate"] == 16000

    def test_create_session_custom_config(self, client: TestClient) -> None:
        """Test creating a session with custom configuration."""
        response = client.post(
            "/stream/sessions",
            params={
                "max_gap_sec": 0.5,
                "enable_prosody": True,
                "enable_emotion": True,
                "enable_diarization": True,
                "sample_rate": 8000,
            },
        )

        assert response.status_code == 201
        data = response.json()

        # Verify custom config values
        config = data["config"]
        assert config["max_gap_sec"] == 0.5
        assert config["enable_prosody"] is True
        assert config["enable_emotion"] is True
        assert config["enable_diarization"] is True
        assert config["sample_rate"] == 8000

    def test_create_session_validates_max_gap_sec(self, client: TestClient) -> None:
        """Test that max_gap_sec is validated (must be 0.1-10.0)."""
        # Too small
        response = client.post("/stream/sessions", params={"max_gap_sec": 0.01})
        assert response.status_code == 422

        # Too large
        response = client.post("/stream/sessions", params={"max_gap_sec": 100.0})
        assert response.status_code == 422

    def test_create_session_validates_sample_rate(self, client: TestClient) -> None:
        """Test that sample_rate is validated (must be 8000-48000)."""
        # Too small
        response = client.post("/stream/sessions", params={"sample_rate": 1000})
        assert response.status_code == 422

        # Too large
        response = client.post("/stream/sessions", params={"sample_rate": 96000})
        assert response.status_code == 422

    def test_list_sessions_empty(self, client: TestClient) -> None:
        """Test listing sessions when none exist."""
        response = client.get("/stream/sessions")

        assert response.status_code == 200
        data = response.json()

        assert "sessions" in data
        assert "count" in data
        assert "registry_stats" in data
        assert data["count"] == 0
        assert len(data["sessions"]) == 0

    def test_list_sessions_with_sessions(self, client: TestClient) -> None:
        """Test listing sessions after creating some."""
        # Create two sessions
        client.post("/stream/sessions")
        client.post("/stream/sessions", params={"enable_prosody": True})

        response = client.get("/stream/sessions")

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 2
        assert len(data["sessions"]) == 2

        # Verify session info structure
        for session in data["sessions"]:
            assert "session_id" in session
            assert "status" in session
            assert "created_at" in session
            assert "last_activity" in session
            assert "config" in session
            assert "stats" in session
            assert session["session_id"].startswith("str-")

    def test_get_session_status(self, client: TestClient) -> None:
        """Test getting status of a specific session."""
        # Create a session
        create_response = client.post(
            "/stream/sessions",
            params={"max_gap_sec": 2.0, "enable_prosody": True},
        )
        session_id = create_response.json()["session_id"]

        # Get status
        response = client.get(f"/stream/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["session_id"] == session_id
        assert data["status"] == "created"  # Not yet started via WebSocket
        assert "created_at" in data
        assert "last_activity" in data
        assert "config" in data
        assert "stats" in data
        assert "last_event_id" in data

        # Verify config matches what was requested
        assert data["config"]["max_gap_sec"] == 2.0
        assert data["config"]["enable_prosody"] is True

        # Verify stats structure
        stats = data["stats"]
        assert "chunks_received" in stats
        assert "bytes_received" in stats
        assert "segments_finalized" in stats

    def test_get_session_status_not_found(self, client: TestClient) -> None:
        """Test getting status of a non-existent session returns 404."""
        response = client.get("/stream/sessions/str-nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_close_session(self, client: TestClient) -> None:
        """Test closing (deleting) a session."""
        # Create a session
        create_response = client.post("/stream/sessions")
        session_id = create_response.json()["session_id"]

        # Verify it exists
        status_response = client.get(f"/stream/sessions/{session_id}")
        assert status_response.status_code == 200

        # Close the session
        close_response = client.delete(f"/stream/sessions/{session_id}")

        assert close_response.status_code == 200
        data = close_response.json()
        assert data["session_id"] == session_id
        assert "closed" in data["message"].lower()

        # Verify session no longer exists
        status_response = client.get(f"/stream/sessions/{session_id}")
        assert status_response.status_code == 404

    def test_close_session_not_found(self, client: TestClient) -> None:
        """Test closing a non-existent session returns 404."""
        response = client.delete("/stream/sessions/str-nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_session_lifecycle(self, client: TestClient) -> None:
        """Test complete session lifecycle via REST endpoints."""
        # 1. Create session
        create_response = client.post(
            "/stream/sessions",
            params={
                "max_gap_sec": 1.5,
                "enable_prosody": True,
                "enable_emotion": False,
            },
        )
        assert create_response.status_code == 201
        session_id = create_response.json()["session_id"]

        # 2. List sessions - should include our session
        list_response = client.get("/stream/sessions")
        assert list_response.status_code == 200
        sessions = list_response.json()["sessions"]
        assert any(s["session_id"] == session_id for s in sessions)

        # 3. Get session status
        status_response = client.get(f"/stream/sessions/{session_id}")
        assert status_response.status_code == 200
        assert status_response.json()["session_id"] == session_id
        assert status_response.json()["status"] == "created"

        # 4. Close session
        close_response = client.delete(f"/stream/sessions/{session_id}")
        assert close_response.status_code == 200

        # 5. Verify session is gone
        list_response = client.get("/stream/sessions")
        sessions = list_response.json()["sessions"]
        assert not any(s["session_id"] == session_id for s in sessions)

    def test_registry_stats_in_list(self, client: TestClient) -> None:
        """Test that registry stats are included in session list response."""
        # Create a session
        client.post("/stream/sessions")

        response = client.get("/stream/sessions")
        data = response.json()

        # Verify registry stats structure
        stats = data["registry_stats"]
        assert "total_sessions" in stats
        assert "by_status" in stats
        assert "connected_count" in stats
        assert "idle_timeout_sec" in stats
        assert "disconnected_ttl_sec" in stats
        assert "cleanup_interval_sec" in stats
        assert "cleanup_task_running" in stats

        # Verify stats values
        assert stats["total_sessions"] == 1
        assert stats["connected_count"] == 0  # No WebSocket connected via REST


class TestStreamSessionsOpenAPISchemas:
    """Tests to verify OpenAPI documentation is properly generated."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create FastAPI test client."""
        from slower_whisper.pipeline.service import app

        return TestClient(app)

    def test_openapi_schema_includes_streaming_endpoints(self, client: TestClient) -> None:
        """Test that OpenAPI schema includes all streaming endpoints."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        paths = schema["paths"]

        # Verify all streaming endpoints are documented
        assert "/stream/config" in paths
        assert "/stream/sessions" in paths
        assert "/stream/sessions/{session_id}" in paths

        # Verify HTTP methods for each endpoint
        assert "get" in paths["/stream/config"]
        assert "get" in paths["/stream/sessions"]
        assert "post" in paths["/stream/sessions"]
        assert "get" in paths["/stream/sessions/{session_id}"]
        assert "delete" in paths["/stream/sessions/{session_id}"]

    def test_openapi_schema_has_streaming_tag(self, client: TestClient) -> None:
        """Test that streaming endpoints have the 'Streaming' tag."""
        response = client.get("/openapi.json")
        schema = response.json()

        paths = schema["paths"]

        # Check that all streaming endpoints use the Streaming tag
        streaming_endpoints = [
            "/stream/config",
            "/stream/sessions",
            "/stream/sessions/{session_id}",
        ]

        for endpoint in streaming_endpoints:
            for method_info in paths[endpoint].values():
                if isinstance(method_info, dict) and "tags" in method_info:
                    assert "Streaming" in method_info["tags"], (
                        f"Endpoint {endpoint} should have 'Streaming' tag"
                    )

    def test_openapi_schema_has_operation_summaries(self, client: TestClient) -> None:
        """Test that streaming endpoints have summaries and descriptions."""
        response = client.get("/openapi.json")
        schema = response.json()

        paths = schema["paths"]

        # Verify POST /stream/sessions has summary
        create_session = paths["/stream/sessions"]["post"]
        assert "summary" in create_session
        assert "create" in create_session["summary"].lower()

        # Verify GET /stream/sessions/{session_id} has summary
        get_session = paths["/stream/sessions/{session_id}"]["get"]
        assert "summary" in get_session
        assert "status" in get_session["summary"].lower()

        # Verify DELETE /stream/sessions/{session_id} has summary
        delete_session = paths["/stream/sessions/{session_id}"]["delete"]
        assert "summary" in delete_session
        assert "close" in delete_session["summary"].lower()


class TestConcurrentSessions:
    """Tests for multiple concurrent sessions."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create FastAPI test client."""
        from slower_whisper.pipeline.service import app

        return TestClient(app)

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> None:
        """Reset session registry before each test."""
        from slower_whisper.pipeline.session_registry import SessionRegistry

        SessionRegistry.reset()
        yield
        SessionRegistry.reset()

    def test_create_multiple_sessions(self, client: TestClient) -> None:
        """Test creating multiple concurrent sessions."""
        session_ids = []

        # Create 5 sessions
        for i in range(5):
            response = client.post(
                "/stream/sessions",
                params={"max_gap_sec": 1.0 + i * 0.1},
            )
            assert response.status_code == 201
            session_ids.append(response.json()["session_id"])

        # All session IDs should be unique
        assert len(set(session_ids)) == 5

        # List should show all 5 sessions
        list_response = client.get("/stream/sessions")
        assert list_response.json()["count"] == 5

    def test_close_multiple_sessions(self, client: TestClient) -> None:
        """Test closing multiple sessions in sequence."""
        # Create 3 sessions
        session_ids = []
        for _ in range(3):
            response = client.post("/stream/sessions")
            session_ids.append(response.json()["session_id"])

        # Close them one by one
        for i, session_id in enumerate(session_ids):
            # Close
            close_response = client.delete(f"/stream/sessions/{session_id}")
            assert close_response.status_code == 200

            # Verify remaining count
            list_response = client.get("/stream/sessions")
            expected_count = len(session_ids) - i - 1
            assert list_response.json()["count"] == expected_count
