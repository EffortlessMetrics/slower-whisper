"""Tests for integration sinks (webhooks, RAG export).

Tests cover:
- Webhook retry logic with mock server
- RAG chunking strategies
- Embedding optional behavior
- Event creation and serialization
- Sink registry dispatch
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from slower_whisper.pipeline.integrations.events import (
    EventType,
    IntegrationEvent,
    create_error_event,
    create_outcome_event,
    create_segment_event,
    create_session_ended_event,
    create_session_started_event,
    create_transcript_event,
)
from slower_whisper.pipeline.integrations.rag_export import (
    ChunkingStrategy,
    RAGBundle,
    RAGChunk,
    RAGExporter,
    RAGExporterConfig,
)
from slower_whisper.pipeline.integrations.registry import (
    SinkConfig,
    SinkRegistry,
    _substitute_env_vars,
)
from slower_whisper.pipeline.integrations.webhooks import (
    AuthConfig,
    HttpResponse,
    RetryPolicy,
    WebhookConfig,
    WebhookSink,
    verify_webhook_signature,
)
from slower_whisper.pipeline.models import Segment, Transcript

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_transcript() -> Transcript:
    """Create a sample transcript for testing."""
    segments = [
        Segment(id=0, start=0.0, end=5.0, text="Hello, how are you?", speaker={"id": "spk_0"}),
        Segment(
            id=1, start=5.1, end=10.0, text="I'm doing great, thanks.", speaker={"id": "spk_1"}
        ),
        Segment(
            id=2, start=10.5, end=15.0, text="That's wonderful to hear.", speaker={"id": "spk_0"}
        ),
        Segment(
            id=3, start=15.5, end=20.0, text="Let's discuss the project.", speaker={"id": "spk_0"}
        ),
        Segment(
            id=4, start=20.5, end=25.0, text="Sure, I have some updates.", speaker={"id": "spk_1"}
        ),
    ]
    return Transcript(
        file_name="test_meeting.wav",
        language="en",
        segments=segments,
    )


@pytest.fixture
def sample_segment() -> Segment:
    """Create a sample segment for testing."""
    return Segment(
        id=0,
        start=0.0,
        end=5.0,
        text="Hello, this is a test.",
        speaker={"id": "spk_0"},
        audio_state={
            "prosody": {"pitch_median_hz": 150.0, "energy_median_db": -20.0},
            "emotion": {"valence": 0.7, "arousal": 0.5},
        },
    )


# =============================================================================
# Event Tests
# =============================================================================


class TestIntegrationEvent:
    """Tests for IntegrationEvent dataclass."""

    def test_event_creation(self) -> None:
        """Test basic event creation."""
        event = IntegrationEvent(
            event_id="test-123",
            event_type=EventType.SESSION_STARTED,
            timestamp=1234567890.0,
            source="test-session",
            payload={"session_id": "test-session"},
        )
        assert event.event_id == "test-123"
        assert event.event_type == EventType.SESSION_STARTED
        assert event.source == "test-session"

    def test_event_to_dict(self) -> None:
        """Test event serialization."""
        event = IntegrationEvent(
            event_id="test-123",
            event_type=EventType.TRANSCRIPT_COMPLETED,
            timestamp=1234567890.0,
            source="test.wav",
            payload={"segments": 10},
            metadata={"version": "1.0"},
        )
        d = event.to_dict()
        assert d["event_id"] == "test-123"
        assert d["event_type"] == "transcript.completed"
        assert d["payload"]["segments"] == 10
        assert d["metadata"]["version"] == "1.0"

    def test_event_from_dict(self) -> None:
        """Test event deserialization."""
        data = {
            "event_id": "test-456",
            "event_type": "segment.finalized",
            "timestamp": 1234567890.0,
            "source": "stream-1",
            "payload": {"text": "hello"},
        }
        event = IntegrationEvent.from_dict(data)
        assert event.event_id == "test-456"
        assert event.event_type == EventType.SEGMENT_FINALIZED


class TestEventCreators:
    """Tests for event creator functions."""

    def test_create_transcript_event(self, sample_transcript: Transcript) -> None:
        """Test transcript event creation."""
        event = create_transcript_event(sample_transcript)
        assert event.event_type == EventType.TRANSCRIPT_COMPLETED
        assert event.source == "test_meeting.wav"
        assert event.payload["segment_count"] == 5
        assert "spk_0" in event.payload["speaker_ids"]

    def test_create_segment_event(self, sample_segment: Segment) -> None:
        """Test segment event creation."""
        event = create_segment_event(sample_segment, source="stream-1")
        assert event.event_type == EventType.SEGMENT_FINALIZED
        assert event.payload["text"] == "Hello, this is a test."
        assert "audio_state_summary" in event.payload

    def test_create_outcome_event(self) -> None:
        """Test outcome event creation."""
        outcomes = [
            {"type": "action_item", "text": "Complete the report"},
            {"type": "decision", "text": "Use the new framework"},
        ]
        event = create_outcome_event(outcomes, source="meeting-1", segment_id=5)
        assert event.event_type == EventType.OUTCOME_DETECTED
        assert event.payload["outcome_count"] == 2
        assert event.payload["segment_id"] == 5

    def test_create_session_started_event(self) -> None:
        """Test session started event creation."""
        event = create_session_started_event(
            session_id="sess-123",
            config={"sample_rate": 16000},
        )
        assert event.event_type == EventType.SESSION_STARTED
        assert event.payload["session_id"] == "sess-123"
        assert event.payload["config"]["sample_rate"] == 16000

    def test_create_session_ended_event(self) -> None:
        """Test session ended event creation."""
        event = create_session_ended_event(
            session_id="sess-123",
            stats={"chunks_received": 100},
        )
        assert event.event_type == EventType.SESSION_ENDED
        assert event.payload["stats"]["chunks_received"] == 100

    def test_create_error_event(self) -> None:
        """Test error event creation."""
        event = create_error_event(
            error_code="ASR_FAILED",
            error_message="Model loading failed",
            source="transcription",
            recoverable=False,
        )
        assert event.event_type == EventType.ERROR_OCCURRED
        assert event.payload["code"] == "ASR_FAILED"
        assert event.payload["recoverable"] is False


# =============================================================================
# Webhook Tests
# =============================================================================


class TestAuthConfig:
    """Tests for AuthConfig."""

    def test_bearer_auth_header(self) -> None:
        """Test bearer token header generation."""
        auth = AuthConfig(type="bearer", token="secret-token")
        header = auth.get_header()
        assert header["Authorization"] == "Bearer secret-token"

    def test_basic_auth_header(self) -> None:
        """Test basic auth header generation."""
        auth = AuthConfig(type="basic", username="user", password="pass")
        header = auth.get_header()
        assert "Basic" in header["Authorization"]

    def test_no_auth_header(self) -> None:
        """Test no auth returns empty dict."""
        auth = AuthConfig(type="none")
        header = auth.get_header()
        assert header == {}


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_exponential_backoff(self) -> None:
        """Test exponential backoff calculation."""
        policy = RetryPolicy(base_delay=1.0, max_delay=30.0)
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
        assert policy.get_delay(3) == 8.0

    def test_max_delay_cap(self) -> None:
        """Test delay is capped at max_delay."""
        policy = RetryPolicy(base_delay=1.0, max_delay=10.0)
        assert policy.get_delay(10) == 10.0


class TestWebhookSink:
    """Tests for WebhookSink."""

    @pytest.mark.asyncio
    async def test_successful_send(self) -> None:
        """Test successful event delivery."""
        # Mock HTTP client
        mock_client = AsyncMock()
        mock_client.post.return_value = HttpResponse(status_code=200)

        config = WebhookConfig(url="https://api.example.com/webhook")
        sink = WebhookSink(config, http_client=mock_client)

        event = create_session_started_event("test-session")
        success = await sink.send_event(event, blocking=True)

        assert success is True
        mock_client.post.assert_called_once()
        await sink.close()

    @pytest.mark.asyncio
    async def test_retry_on_500(self) -> None:
        """Test retry on server error."""
        mock_client = AsyncMock()
        # First call fails, second succeeds
        mock_client.post.side_effect = [
            HttpResponse(status_code=500, text="Internal Server Error"),
            HttpResponse(status_code=200),
        ]

        config = WebhookConfig(
            url="https://api.example.com/webhook",
            retry=RetryPolicy(max_retries=2, base_delay=0.01),
        )
        sink = WebhookSink(config, http_client=mock_client)

        event = create_session_started_event("test-session")
        success = await sink.send_event(event, blocking=True)

        assert success is True
        assert mock_client.post.call_count == 2
        await sink.close()

    @pytest.mark.asyncio
    async def test_dead_letter_queue_on_failure(self) -> None:
        """Test events are added to DLQ on all retries exhausted."""
        mock_client = AsyncMock()
        mock_client.post.return_value = HttpResponse(status_code=500, text="Error")

        config = WebhookConfig(
            url="https://api.example.com/webhook",
            retry=RetryPolicy(max_retries=1, base_delay=0.01),
        )
        sink = WebhookSink(config, http_client=mock_client)

        event = create_session_started_event("test-session")
        success = await sink.send_event(event, blocking=True)

        assert success is False
        dlq = sink.get_dead_letter_queue()
        assert len(dlq) == 1
        assert dlq[0].url == "https://api.example.com/webhook"
        await sink.close()

    @pytest.mark.asyncio
    async def test_batch_send(self) -> None:
        """Test batch event sending."""
        mock_client = AsyncMock()
        mock_client.post.return_value = HttpResponse(status_code=200)

        config = WebhookConfig(url="https://api.example.com/webhook")
        sink = WebhookSink(config, http_client=mock_client)

        events = [
            create_session_started_event("sess-1"),
            create_session_started_event("sess-2"),
        ]
        results = await sink.send_batch(events, blocking=True)

        assert all(results)
        assert mock_client.post.call_count == 2
        await sink.close()

    @pytest.mark.asyncio
    async def test_send_transcript(self, sample_transcript: Transcript) -> None:
        """Test send_transcript convenience method."""
        mock_client = AsyncMock()
        mock_client.post.return_value = HttpResponse(status_code=200)

        config = WebhookConfig(url="https://api.example.com/webhook")
        sink = WebhookSink(config, http_client=mock_client)

        success = await sink.send_transcript(sample_transcript, blocking=True)
        assert success is True
        await sink.close()

    @pytest.mark.asyncio
    async def test_hmac_signature(self) -> None:
        """Test HMAC signature is added to headers."""
        mock_client = AsyncMock()
        mock_client.post.return_value = HttpResponse(status_code=200)

        config = WebhookConfig(
            url="https://api.example.com/webhook",
            hmac_secret="test-secret",
        )
        sink = WebhookSink(config, http_client=mock_client)

        event = create_session_started_event("test-session")
        await sink.send_event(event, blocking=True)

        # Check that headers include signature
        call_args = mock_client.post.call_args
        headers = call_args.kwargs["headers"]
        assert "X-Webhook-Signature" in headers
        assert headers["X-Webhook-Signature"].startswith("sha256=")
        await sink.close()


class TestWebhookSignatureVerification:
    """Tests for webhook signature verification."""

    def test_valid_signature(self) -> None:
        """Test valid signature verification."""
        secret = "test-secret"
        payload = '{"event": "test"}'

        # Generate signature
        import hashlib
        import hmac as hmac_mod

        sig = hmac_mod.new(secret.encode(), payload.encode(), hashlib.sha256)
        signature = f"sha256={sig.hexdigest()}"

        assert verify_webhook_signature(payload, signature, secret) is True

    def test_invalid_signature(self) -> None:
        """Test invalid signature is rejected."""
        secret = "test-secret"
        payload = '{"event": "test"}'
        signature = "sha256=invalid"

        assert verify_webhook_signature(payload, signature, secret) is False


# =============================================================================
# RAG Export Tests
# =============================================================================


class TestRAGChunk:
    """Tests for RAGChunk dataclass."""

    def test_chunk_serialization(self) -> None:
        """Test chunk to_dict and from_dict."""
        chunk = RAGChunk(
            id="chunk_001",
            text="Hello world",
            metadata={"source": "test.wav"},
            embedding=[0.1, 0.2, 0.3],
        )
        d = chunk.to_dict()
        assert d["id"] == "chunk_001"
        assert d["embedding"] == [0.1, 0.2, 0.3]

        restored = RAGChunk.from_dict(d)
        assert restored.id == chunk.id
        assert restored.embedding == chunk.embedding


class TestRAGExporter:
    """Tests for RAGExporter."""

    def test_chunk_by_segment(self, sample_transcript: Transcript) -> None:
        """Test by_segment chunking strategy."""
        config = RAGExporterConfig(chunking_strategy=ChunkingStrategy.BY_SEGMENT)
        exporter = RAGExporter(config)
        bundle = exporter.export(sample_transcript)

        assert len(bundle.chunks) == 5  # One per segment
        assert bundle.chunks[0].text == "Hello, how are you?"

    def test_chunk_by_speaker_turn(self, sample_transcript: Transcript) -> None:
        """Test by_speaker_turn chunking strategy."""
        config = RAGExporterConfig(chunking_strategy=ChunkingStrategy.BY_SPEAKER_TURN)
        exporter = RAGExporter(config)
        bundle = exporter.export(sample_transcript)

        # Should group consecutive same-speaker segments
        # spk_0: seg 0, spk_1: seg 1, spk_0: seg 2-3, spk_1: seg 4
        assert len(bundle.chunks) == 4
        assert "spk_0" in bundle.chunks[0].metadata.get("speaker", "")

    def test_chunk_by_time(self, sample_transcript: Transcript) -> None:
        """Test by_time chunking strategy."""
        config = RAGExporterConfig(
            chunking_strategy=ChunkingStrategy.BY_TIME,
            time_window_seconds=10.0,
        )
        exporter = RAGExporter(config)
        bundle = exporter.export(sample_transcript)

        # 25 seconds of audio with 10s windows = ~3 chunks
        assert 2 <= len(bundle.chunks) <= 4

    def test_bundle_metadata(self, sample_transcript: Transcript) -> None:
        """Test bundle metadata is populated."""
        exporter = RAGExporter()
        bundle = exporter.export(sample_transcript)

        assert bundle.metadata["total_chunks"] == len(bundle.chunks)
        assert bundle.metadata["source_file"] == "test_meeting.wav"
        assert bundle.metadata["language"] == "en"
        assert "spk_0" in bundle.metadata["speakers"]

    def test_embedding_disabled_by_default(self, sample_transcript: Transcript) -> None:
        """Test embeddings are not generated by default."""
        exporter = RAGExporter()
        bundle = exporter.export(sample_transcript)

        for chunk in bundle.chunks:
            assert chunk.embedding is None

    def test_bundle_save_and_load(self, sample_transcript: Transcript, tmp_path: Path) -> None:
        """Test bundle save and load."""
        exporter = RAGExporter()
        bundle = exporter.export(sample_transcript)

        output_path = tmp_path / "bundle.json"
        bundle.save(output_path)

        loaded = RAGBundle.load(output_path)
        assert len(loaded.chunks) == len(bundle.chunks)
        assert loaded.metadata["source_file"] == bundle.metadata["source_file"]


class TestRAGExporterWithEmbeddings:
    """Tests for RAGExporter with embeddings (requires sentence-transformers)."""

    @pytest.mark.slow
    def test_embedding_generation(self, sample_transcript: Transcript) -> None:
        """Test embedding generation when enabled."""
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        config = RAGExporterConfig(
            chunking_strategy=ChunkingStrategy.BY_SEGMENT,
            include_embeddings=True,
            embedding_model="all-MiniLM-L6-v2",
        )
        exporter = RAGExporter(config)
        bundle = exporter.export(sample_transcript)

        for chunk in bundle.chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) > 0  # Model produces 384-dim vectors


# =============================================================================
# Registry Tests
# =============================================================================


class TestSinkRegistry:
    """Tests for SinkRegistry."""

    @pytest.mark.asyncio
    async def test_register_and_dispatch(self) -> None:
        """Test sink registration and event dispatch."""
        registry = SinkRegistry()

        mock_sink = AsyncMock()
        mock_sink.send_event.return_value = True

        registry.register("test", mock_sink)

        event = create_session_started_event("sess-1")
        results = await registry.dispatch(event, blocking=True)

        assert results["test"] is True
        mock_sink.send_event.assert_called_once()
        await registry.close()

    @pytest.mark.asyncio
    async def test_event_filter(self) -> None:
        """Test event filtering by sink config."""
        registry = SinkRegistry()

        mock_sink = AsyncMock()
        mock_sink.send_event.return_value = True

        config = SinkConfig(
            name="filtered",
            type="webhook",
            config={},
            event_filter=["transcript.completed"],  # Only this event type
        )
        registry.register("filtered", mock_sink, config)

        # This event should be filtered out
        event = create_session_started_event("sess-1")
        results = await registry.dispatch(event, blocking=True)

        assert "filtered" not in results
        mock_sink.send_event.assert_not_called()
        await registry.close()

    @pytest.mark.asyncio
    async def test_dispatch_batch(self) -> None:
        """Test batch dispatch."""
        registry = SinkRegistry()

        mock_sink = AsyncMock()
        mock_sink.send_batch.return_value = [True, True]

        registry.register("test", mock_sink)

        events = [
            create_session_started_event("sess-1"),
            create_session_ended_event("sess-1"),
        ]
        results = await registry.dispatch_batch(events, blocking=True)

        assert results["test"] == [True, True]
        await registry.close()


class TestEnvVarSubstitution:
    """Tests for environment variable substitution."""

    def test_substitute_single_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test single variable substitution."""
        monkeypatch.setenv("TEST_TOKEN", "secret123")
        content = "token: ${TEST_TOKEN}"
        result = _substitute_env_vars(content)
        assert result == "token: secret123"

    def test_substitute_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test multiple variable substitution."""
        monkeypatch.setenv("HOST", "example.com")
        monkeypatch.setenv("PORT", "8080")
        content = "url: https://${HOST}:${PORT}/api"
        result = _substitute_env_vars(content)
        assert result == "url: https://example.com:8080/api"

    def test_missing_var_becomes_empty(self) -> None:
        """Test missing variable becomes empty string."""
        content = "token: ${NONEXISTENT_VAR}"
        result = _substitute_env_vars(content)
        assert result == "token: "


class TestRegistryFromConfig:
    """Tests for registry configuration loading."""

    @pytest.mark.asyncio
    async def test_from_config_dict(self) -> None:
        """Test registry creation from config dict."""
        config = {
            "sinks": [
                {
                    "name": "test_webhook",
                    "type": "webhook",
                    "config": {
                        "url": "https://api.example.com/webhook",
                        "timeout": 10.0,
                    },
                }
            ]
        }

        # Create registry without mocking - just verify structure
        registry = SinkRegistry.from_config(config)

        assert "test_webhook" in registry.list_sinks()
        # Close gracefully (sink may not have an actual HTTP client)
        await registry.close()
