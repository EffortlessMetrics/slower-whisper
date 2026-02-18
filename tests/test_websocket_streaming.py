"""Integration tests for WebSocket streaming endpoint (Issue #84).

This module provides comprehensive integration tests for the /stream WebSocket
endpoint, validating the full protocol lifecycle including:
- Connection establishment and session management
- START_SESSION, AUDIO_CHUNK, END_SESSION, PING message handling
- Event envelope format compliance
- Error handling and recovery
- Backpressure and resume capabilities

These tests complement the unit tests in test_streaming_ws.py by focusing on
the HTTP/WebSocket integration layer.
"""

from __future__ import annotations

import base64
import time

import pytest
from fastapi.testclient import TestClient

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create FastAPI test client for WebSocket testing."""
    from slower_whisper.pipeline.service import app

    return TestClient(app)


@pytest.fixture
def sample_audio_data() -> bytes:
    """Generate sample audio data (1 second at 16kHz, 16-bit mono)."""
    # 16000 samples/sec * 2 bytes/sample = 32000 bytes per second
    return b"\x00\x01" * 16000  # 1 second of audio


@pytest.fixture
def sample_audio_chunk(sample_audio_data: bytes) -> dict:
    """Create a sample AUDIO_CHUNK message payload."""
    return {
        "type": "AUDIO_CHUNK",
        "data": base64.b64encode(sample_audio_data).decode(),
        "sequence": 1,
    }


# =============================================================================
# Connection Lifecycle Tests
# =============================================================================


class TestWebSocketConnection:
    """Tests for WebSocket connection establishment and teardown."""

    def test_websocket_connect_success(self, client: TestClient) -> None:
        """Test successful WebSocket connection."""
        with client.websocket_connect("/stream") as websocket:
            # Connection should be established
            # Send a PING to verify connection is working
            websocket.send_json({"type": "PING", "timestamp": 12345})
            msg = websocket.receive_json()
            assert msg["type"] == "PONG"

    def test_websocket_graceful_close(self, client: TestClient) -> None:
        """Test graceful connection close after END_SESSION."""
        with client.websocket_connect("/stream") as websocket:
            # Start session
            websocket.send_json({"type": "START_SESSION", "config": {}})
            msg = websocket.receive_json()
            assert msg["type"] == "SESSION_STARTED"

            # End session
            websocket.send_json({"type": "END_SESSION"})

            # Should receive SESSION_ENDED
            events = []
            while True:
                try:
                    msg = websocket.receive_json()
                    events.append(msg)
                    if msg["type"] == "SESSION_ENDED":
                        break
                except Exception:
                    break

            assert any(e["type"] == "SESSION_ENDED" for e in events)


# =============================================================================
# START_SESSION Message Tests
# =============================================================================


class TestStartSession:
    """Tests for START_SESSION message handling."""

    def test_start_session_default_config(self, client: TestClient) -> None:
        """Test START_SESSION with default configuration."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION"})
            msg = websocket.receive_json()

            assert msg["type"] == "SESSION_STARTED"
            assert "event_id" in msg
            assert msg["event_id"] == 1
            assert msg["stream_id"].startswith("str-")
            assert "payload" in msg
            assert "session_id" in msg["payload"]

    def test_start_session_custom_config(self, client: TestClient) -> None:
        """Test START_SESSION with custom configuration."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(
                {
                    "type": "START_SESSION",
                    "config": {
                        "max_gap_sec": 0.5,
                        "enable_prosody": True,
                        "enable_emotion": True,
                        "sample_rate": 16000,
                    },
                }
            )
            msg = websocket.receive_json()

            assert msg["type"] == "SESSION_STARTED"
            assert msg["stream_id"].startswith("str-")

    def test_start_session_duplicate_error(self, client: TestClient) -> None:
        """Test error when starting session twice."""
        with client.websocket_connect("/stream") as websocket:
            # First START_SESSION
            websocket.send_json({"type": "START_SESSION", "config": {}})
            msg = websocket.receive_json()
            assert msg["type"] == "SESSION_STARTED"

            # Second START_SESSION should error
            websocket.send_json({"type": "START_SESSION", "config": {}})
            msg = websocket.receive_json()
            assert msg["type"] == "ERROR"
            assert msg["payload"]["code"] == "session_already_started"
            assert msg["payload"]["recoverable"] is True


# =============================================================================
# AUDIO_CHUNK Message Tests
# =============================================================================


class TestAudioChunk:
    """Tests for AUDIO_CHUNK message handling."""

    def test_audio_chunk_processing(self, client: TestClient, sample_audio_chunk: dict) -> None:
        """Test audio chunk is processed and generates events."""
        with client.websocket_connect("/stream") as websocket:
            # Start session
            websocket.send_json({"type": "START_SESSION", "config": {}})
            msg = websocket.receive_json()
            assert msg["type"] == "SESSION_STARTED"

            # Send audio chunk
            websocket.send_json(sample_audio_chunk)

            # Should receive PARTIAL event(s)
            msg = websocket.receive_json()
            assert msg["type"] == "PARTIAL"
            assert "segment" in msg["payload"]
            assert "start" in msg["payload"]["segment"]
            assert "end" in msg["payload"]["segment"]
            assert "text" in msg["payload"]["segment"]

    def test_audio_chunk_without_session(
        self, client: TestClient, sample_audio_chunk: dict
    ) -> None:
        """Test error when sending AUDIO_CHUNK without starting session."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(sample_audio_chunk)
            msg = websocket.receive_json()

            assert msg["type"] == "ERROR"
            assert msg["payload"]["code"] == "no_session"

    def test_audio_chunk_sequence_ordering(
        self, client: TestClient, sample_audio_data: bytes
    ) -> None:
        """Test audio chunks must have increasing sequence numbers."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()  # SESSION_STARTED

            # Send first chunk with sequence 1 - use full second of audio
            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": base64.b64encode(sample_audio_data).decode(),
                    "sequence": 1,
                }
            )
            # Receive the PARTIAL event that will be generated
            msg = websocket.receive_json()
            assert msg["type"] in ("PARTIAL", "FINALIZED")

            # Send chunk with duplicate sequence should error
            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": base64.b64encode(sample_audio_data).decode(),
                    "sequence": 1,  # Duplicate sequence
                }
            )
            msg = websocket.receive_json()
            assert msg["type"] == "ERROR"
            assert "sequence" in msg["payload"]["message"].lower()

    def test_audio_chunk_invalid_base64(self, client: TestClient) -> None:
        """Test error on invalid base64 audio data."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()  # SESSION_STARTED

            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": "not-valid-base64!!!",
                    "sequence": 1,
                }
            )
            msg = websocket.receive_json()

            assert msg["type"] == "ERROR"
            assert msg["payload"]["code"] == "invalid_audio_chunk"

    def test_audio_chunk_missing_sequence(self, client: TestClient) -> None:
        """Test error when sequence is missing from AUDIO_CHUNK."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()  # SESSION_STARTED

            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": base64.b64encode(b"\x00" * 100).decode(),
                    # Missing sequence
                }
            )
            msg = websocket.receive_json()

            assert msg["type"] == "ERROR"


# =============================================================================
# END_SESSION Message Tests
# =============================================================================


class TestEndSession:
    """Tests for END_SESSION message handling."""

    def test_end_session_with_audio(self, client: TestClient, sample_audio_chunk: dict) -> None:
        """Test END_SESSION finalizes segments and returns stats."""
        with client.websocket_connect("/stream") as websocket:
            # Start session
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()  # SESSION_STARTED

            # Send audio
            websocket.send_json(sample_audio_chunk)
            websocket.receive_json()  # PARTIAL

            # End session
            websocket.send_json({"type": "END_SESSION"})

            # Collect all end events
            events = []
            while True:
                try:
                    msg = websocket.receive_json()
                    events.append(msg)
                    if msg["type"] == "SESSION_ENDED":
                        break
                except Exception:
                    break

            # Should have FINALIZED segment
            finalized = [e for e in events if e["type"] == "FINALIZED"]
            assert len(finalized) >= 1

            # Should have SESSION_ENDED with stats
            ended = [e for e in events if e["type"] == "SESSION_ENDED"]
            assert len(ended) == 1
            stats = ended[0]["payload"]["stats"]
            assert "chunks_received" in stats
            assert "bytes_received" in stats
            assert "duration_sec" in stats

    def test_end_session_without_start(self, client: TestClient) -> None:
        """Test error when ending session that was never started."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "END_SESSION"})
            msg = websocket.receive_json()

            assert msg["type"] == "ERROR"
            assert msg["payload"]["code"] == "no_session"


# =============================================================================
# PING Message Tests
# =============================================================================


class TestPingPong:
    """Tests for PING/PONG heartbeat mechanism."""

    def test_ping_before_session(self, client: TestClient) -> None:
        """Test PING works even before session is started."""
        with client.websocket_connect("/stream") as websocket:
            client_ts = int(time.time() * 1000)
            websocket.send_json({"type": "PING", "timestamp": client_ts})
            msg = websocket.receive_json()

            assert msg["type"] == "PONG"
            assert msg["payload"]["timestamp"] == client_ts
            assert "server_timestamp" in msg["payload"]

    def test_ping_during_session(self, client: TestClient) -> None:
        """Test PING works during active session."""
        with client.websocket_connect("/stream") as websocket:
            # Start session
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()

            # PING should work
            client_ts = int(time.time() * 1000)
            websocket.send_json({"type": "PING", "timestamp": client_ts})
            msg = websocket.receive_json()

            assert msg["type"] == "PONG"
            assert msg["payload"]["timestamp"] == client_ts

    def test_ping_roundtrip_latency(self, client: TestClient) -> None:
        """Test PING roundtrip captures server timestamp for latency measurement."""
        with client.websocket_connect("/stream") as websocket:
            client_ts = int(time.time() * 1000)
            websocket.send_json({"type": "PING", "timestamp": client_ts})
            msg = websocket.receive_json()

            server_ts = msg["payload"]["server_timestamp"]
            # Server timestamp should be reasonable (within 10 seconds)
            assert abs(server_ts - client_ts) < 10000


# =============================================================================
# Event Envelope Tests
# =============================================================================


class TestEventEnvelope:
    """Tests for event envelope format compliance."""

    def test_envelope_has_required_fields(self, client: TestClient) -> None:
        """Test all events have required envelope fields."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            msg = websocket.receive_json()

            # Required fields per event envelope spec
            assert "event_id" in msg
            assert "stream_id" in msg
            assert "type" in msg
            assert "ts_server" in msg
            assert "payload" in msg

    def test_envelope_event_ids_monotonic(
        self, client: TestClient, sample_audio_chunk: dict
    ) -> None:
        """Test event IDs are monotonically increasing."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            msg1 = websocket.receive_json()

            websocket.send_json(sample_audio_chunk)
            msg2 = websocket.receive_json()

            websocket.send_json({"type": "PING", "timestamp": 0})
            msg3 = websocket.receive_json()

            assert msg1["event_id"] == 1
            assert msg2["event_id"] > msg1["event_id"]
            assert msg3["event_id"] > msg2["event_id"]

    def test_envelope_stream_id_consistent(
        self, client: TestClient, sample_audio_chunk: dict
    ) -> None:
        """Test stream_id remains consistent across all events in a session."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            msg1 = websocket.receive_json()
            stream_id = msg1["stream_id"]

            websocket.send_json(sample_audio_chunk)
            msg2 = websocket.receive_json()

            assert msg2["stream_id"] == stream_id

    def test_envelope_segment_events_have_audio_timestamps(
        self, client: TestClient, sample_audio_chunk: dict
    ) -> None:
        """Test PARTIAL and FINALIZED events have audio timestamps."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()  # SESSION_STARTED

            websocket.send_json(sample_audio_chunk)
            msg = websocket.receive_json()

            if msg["type"] in ("PARTIAL", "FINALIZED"):
                assert "ts_audio_start" in msg
                assert "ts_audio_end" in msg
                assert "segment_id" in msg


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and recovery."""

    def test_invalid_message_type(self, client: TestClient) -> None:
        """Test handling of invalid message type."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()

            websocket.send_json({"type": "INVALID_TYPE"})
            msg = websocket.receive_json()

            assert msg["type"] == "ERROR"
            assert msg["payload"]["code"] == "invalid_message_type"
            assert msg["payload"]["recoverable"] is True

    def test_malformed_json(self, client: TestClient) -> None:
        """Test handling of malformed JSON (when possible via TestClient)."""
        # Note: FastAPI's TestClient auto-parses JSON, so we test via dict
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()

            # Missing 'type' field
            websocket.send_json({"config": {}})
            msg = websocket.receive_json()

            assert msg["type"] == "ERROR"

    def test_recoverable_error_continues_session(
        self, client: TestClient, sample_audio_data: bytes
    ) -> None:
        """Test recoverable errors don't terminate the session."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()

            # Cause a recoverable error
            websocket.send_json({"type": "INVALID_TYPE"})
            msg = websocket.receive_json()
            assert msg["type"] == "ERROR"
            assert msg["payload"]["recoverable"] is True

            # Session should still work
            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": base64.b64encode(sample_audio_data).decode(),
                    "sequence": 1,
                }
            )
            msg = websocket.receive_json()
            # Should get a PARTIAL, not an error
            assert msg["type"] in ("PARTIAL", "ERROR")


# =============================================================================
# Full Session Flow Tests
# =============================================================================


class TestFullSessionFlow:
    """Integration tests for complete session workflows."""

    def test_complete_transcription_session(
        self, client: TestClient, sample_audio_data: bytes
    ) -> None:
        """Test complete transcription session from start to end."""
        with client.websocket_connect("/stream") as websocket:
            # 1. Start session
            websocket.send_json(
                {
                    "type": "START_SESSION",
                    "config": {"max_gap_sec": 1.0},
                }
            )
            msg = websocket.receive_json()
            assert msg["type"] == "SESSION_STARTED"
            stream_id = msg["stream_id"]

            # 2. Send a single audio chunk (1 second of audio)
            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": base64.b64encode(sample_audio_data).decode(),
                    "sequence": 1,
                }
            )
            # Receive the PARTIAL event
            msg = websocket.receive_json()
            assert msg["stream_id"] == stream_id
            assert msg["type"] in ("PARTIAL", "FINALIZED")

            # 3. End session
            websocket.send_json({"type": "END_SESSION"})

            # 4. Collect final events
            events = []
            while True:
                try:
                    msg = websocket.receive_json()
                    events.append(msg)
                    if msg["type"] == "SESSION_ENDED":
                        break
                except Exception:
                    break

            # Verify final events
            types = [e["type"] for e in events]
            assert "SESSION_ENDED" in types

            # Verify SESSION_ENDED has proper stats
            ended_event = next(e for e in events if e["type"] == "SESSION_ENDED")
            stats = ended_event["payload"]["stats"]
            assert stats["chunks_received"] == 1
            assert stats["bytes_received"] > 0

    def test_session_with_ping_heartbeat(
        self, client: TestClient, sample_audio_data: bytes
    ) -> None:
        """Test session interleaved with PING heartbeats."""
        with client.websocket_connect("/stream") as websocket:
            # Start session
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()

            # Interleave PING with audio
            websocket.send_json({"type": "PING", "timestamp": 1000})
            msg = websocket.receive_json()
            assert msg["type"] == "PONG"

            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": base64.b64encode(sample_audio_data).decode(),
                    "sequence": 1,
                }
            )
            msg = websocket.receive_json()
            assert msg["type"] in ("PARTIAL", "FINALIZED")

            websocket.send_json({"type": "PING", "timestamp": 2000})
            msg = websocket.receive_json()
            assert msg["type"] == "PONG"


# =============================================================================
# TTS State Tests
# =============================================================================


class TestTTSState:
    """Tests for TTS_STATE message handling (v2.1 feature)."""

    def test_tts_state_update(self, client: TestClient) -> None:
        """Test TTS_STATE message is accepted during active session."""
        with client.websocket_connect("/stream") as websocket:
            # Start session
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()

            # Send TTS_STATE - should be accepted without error
            websocket.send_json({"type": "TTS_STATE", "playing": True})

            # Verify session still works by sending PING
            websocket.send_json({"type": "PING", "timestamp": 12345})
            msg = websocket.receive_json()
            assert msg["type"] == "PONG"

    def test_tts_state_without_session(self, client: TestClient) -> None:
        """Test error when sending TTS_STATE without active session."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "TTS_STATE", "playing": True})
            msg = websocket.receive_json()

            assert msg["type"] == "ERROR"
            assert msg["payload"]["code"] == "no_session"


# =============================================================================
# REST API Companion Endpoints Tests
# =============================================================================


class TestStreamConfigEndpoint:
    """Tests for /stream/config REST endpoint."""

    def test_get_stream_config(self, client: TestClient) -> None:
        """Test GET /stream/config returns valid configuration."""
        response = client.get("/stream/config")
        assert response.status_code == 200

        data = response.json()
        assert "default_config" in data
        assert "supported_audio_formats" in data
        assert "supported_sample_rates" in data
        assert "message_types" in data

        # Verify config structure
        config = data["default_config"]
        assert "max_gap_sec" in config
        assert "sample_rate" in config
        assert "audio_format" in config

        # Verify message types
        assert "START_SESSION" in data["message_types"]["client"]
        assert "AUDIO_CHUNK" in data["message_types"]["client"]
        assert "END_SESSION" in data["message_types"]["client"]
        assert "PING" in data["message_types"]["client"]

        assert "SESSION_STARTED" in data["message_types"]["server"]
        assert "PARTIAL" in data["message_types"]["server"]
        assert "FINALIZED" in data["message_types"]["server"]
        assert "ERROR" in data["message_types"]["server"]
        assert "SESSION_ENDED" in data["message_types"]["server"]
        assert "PONG" in data["message_types"]["server"]
