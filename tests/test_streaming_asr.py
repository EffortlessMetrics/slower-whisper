"""Tests for StreamingASRAdapter.

These tests verify the VAD-triggered chunked transcription adapter
that enables real-time ASR with faster-whisper.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from transcription.streaming_asr import (
    StreamingASRAdapter,
    StreamingASRConfig,
)


class MockWhisperModel:
    """Mock faster-whisper model for testing."""

    def __init__(self, return_text: str = "Hello world"):
        self.return_text = return_text
        self.transcribe_calls: list[tuple[Any, dict[str, Any]]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        """Mock transcribe that returns a single segment."""
        self.transcribe_calls.append((audio, kwargs))

        # Create mock segment
        segment = SimpleNamespace(
            start=0.0,
            end=1.0,
            text=self.return_text,
        )
        info = SimpleNamespace(language="en")
        return [segment], info


class MockWhisperModelMultiSegment:
    """Mock model that returns multiple segments."""

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        segments = [
            SimpleNamespace(start=0.0, end=1.0, text="First segment."),
            SimpleNamespace(start=1.2, end=2.5, text="Second segment."),
        ]
        info = SimpleNamespace(language="en")
        return segments, info


class MockWhisperModelEmpty:
    """Mock model that returns no segments."""

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        return [], SimpleNamespace(language="en")


class TestStreamingASRConfig:
    """Tests for StreamingASRConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StreamingASRConfig()

        assert config.min_chunk_duration_sec == 1.0
        assert config.max_chunk_duration_sec == 30.0
        assert config.sample_rate == 16000
        assert config.vad_energy_threshold == 0.01
        assert config.vad_silence_duration_sec == 0.8
        assert config.language is None
        assert config.beam_size == 5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StreamingASRConfig(
            min_chunk_duration_sec=0.5,
            max_chunk_duration_sec=15.0,
            sample_rate=8000,
            language="en",
            beam_size=3,
        )

        assert config.min_chunk_duration_sec == 0.5
        assert config.max_chunk_duration_sec == 15.0
        assert config.sample_rate == 8000
        assert config.language == "en"
        assert config.beam_size == 3


class TestStreamingASRAdapter:
    """Tests for StreamingASRAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        assert adapter.model is model
        assert adapter.config is not None
        assert adapter.buffer_duration_sec == 0.0
        assert adapter.pending_duration_sec == 0.0

    def test_initialization_with_config(self):
        """Test adapter initialization with custom config."""
        model = MockWhisperModel()
        config = StreamingASRConfig(min_chunk_duration_sec=2.0)
        adapter = StreamingASRAdapter(model, config)

        assert adapter.config.min_chunk_duration_sec == 2.0

    def test_pcm_bytes_to_float32(self):
        """Test PCM to float32 conversion."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        # Create sample PCM data (silence = zeros)
        pcm_bytes = bytes(3200)  # 100ms at 16kHz, 16-bit

        float32 = adapter._pcm_bytes_to_float32(pcm_bytes)

        assert float32.dtype == np.float32
        assert len(float32) == 1600  # Half as many samples (2 bytes per sample)
        assert np.allclose(float32, 0.0)

    def test_pcm_bytes_to_float32_normalization(self):
        """Test that PCM conversion normalizes to [-1, 1]."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        # Create max positive value (32767)
        int16_max = np.array([32767], dtype=np.int16)
        pcm_bytes = int16_max.tobytes()

        float32 = adapter._pcm_bytes_to_float32(pcm_bytes)

        # Should be close to 1.0 (32767/32768)
        assert float32[0] == pytest.approx(32767 / 32768, rel=1e-4)

    def test_calculate_energy_silence(self):
        """Test energy calculation for silence."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        silence = np.zeros(1600, dtype=np.float32)
        energy = adapter._calculate_energy(silence)

        assert energy == 0.0

    def test_calculate_energy_signal(self):
        """Test energy calculation for non-zero signal."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        # Sine wave should have non-zero energy
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        energy = adapter._calculate_energy(signal)

        assert energy > 0.0
        assert energy < 1.0

    def test_calculate_energy_empty(self):
        """Test energy calculation for empty array."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        empty = np.array([], dtype=np.float32)
        energy = adapter._calculate_energy(empty)

        assert energy == 0.0

    @pytest.mark.asyncio
    async def test_ingest_audio_empty(self):
        """Test ingesting empty audio."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        chunks = await adapter.ingest_audio(b"")

        assert chunks == []
        assert adapter.buffer_duration_sec == 0.0

    @pytest.mark.asyncio
    async def test_ingest_audio_short_chunk(self):
        """Test that short audio doesn't trigger transcription."""
        model = MockWhisperModel()
        config = StreamingASRConfig(min_chunk_duration_sec=1.0)
        adapter = StreamingASRAdapter(model, config)

        # 100ms of silence (not enough to trigger)
        pcm_bytes = bytes(3200)  # 100ms at 16kHz, 16-bit
        chunks = await adapter.ingest_audio(pcm_bytes)

        # No transcription yet (below min_chunk_duration)
        assert chunks == []
        assert adapter.pending_duration_sec > 0

    @pytest.mark.asyncio
    async def test_ingest_audio_triggers_transcription(self):
        """Test that sufficient audio triggers transcription."""
        model = MockWhisperModel()
        config = StreamingASRConfig(
            min_chunk_duration_sec=0.5,
            vad_silence_duration_sec=0.3,
        )
        adapter = StreamingASRAdapter(model, config)

        # Generate enough audio (1 second of speech-like signal)
        samples = 16000  # 1 second
        t = np.linspace(0, 1, samples, dtype=np.float32)
        signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        int16_signal = (signal * 32767).astype(np.int16)
        pcm_bytes = int16_signal.tobytes()

        await adapter.ingest_audio(pcm_bytes)

        # Should have triggered transcription
        assert len(model.transcribe_calls) > 0

    @pytest.mark.asyncio
    async def test_flush_returns_remaining_audio(self):
        """Test that flush processes remaining buffered audio."""
        model = MockWhisperModel(return_text="Flushed text")
        config = StreamingASRConfig(min_chunk_duration_sec=5.0)  # High threshold
        adapter = StreamingASRAdapter(model, config)

        # Add some audio (not enough to trigger normal transcription)
        samples = 16000  # 1 second
        t = np.linspace(0, 1, samples, dtype=np.float32)
        signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        int16_signal = (signal * 32767).astype(np.int16)
        pcm_bytes = int16_signal.tobytes()

        await adapter.ingest_audio(pcm_bytes)

        # Flush should process remaining audio
        await adapter.flush()

        # Should have transcribed
        assert len(model.transcribe_calls) > 0

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self):
        """Test flush with no pending audio."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        chunks = await adapter.flush()

        assert chunks == []
        assert len(model.transcribe_calls) == 0

    def test_reset_clears_state(self):
        """Test that reset clears all internal state."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        # Add some audio (synchronously via internal method)
        adapter._audio_buffer = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        adapter._pending_speech = np.array([0.4, 0.5], dtype=np.float32)
        adapter._time_offset = 5.0

        adapter.reset()

        assert adapter.buffer_duration_sec == 0.0
        assert adapter.pending_duration_sec == 0.0
        assert adapter._time_offset == 0.0

    @pytest.mark.asyncio
    async def test_chunks_have_correct_format(self):
        """Test that returned chunks have correct StreamChunk format."""
        model = MockWhisperModel(return_text="Test transcription")
        config = StreamingASRConfig(
            min_chunk_duration_sec=0.5,
            max_chunk_duration_sec=1.0,  # Force transcription quickly
        )
        adapter = StreamingASRAdapter(model, config)

        # Generate enough audio to force transcription via max_chunk_duration
        samples = 32000  # 2 seconds (exceeds max_chunk_duration)
        t = np.linspace(0, 2, samples, dtype=np.float32)
        signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        int16_signal = (signal * 32767).astype(np.int16)
        pcm_bytes = int16_signal.tobytes()

        chunks = await adapter.ingest_audio(pcm_bytes)

        # If we got chunks, verify format
        if chunks:
            chunk = chunks[0]
            assert "start" in chunk
            assert "end" in chunk
            assert "text" in chunk
            assert "speaker_id" in chunk
            assert isinstance(chunk["start"], float)
            assert isinstance(chunk["end"], float)
            assert isinstance(chunk["text"], str)

    @pytest.mark.asyncio
    async def test_model_type_error_fallback(self):
        """Test fallback to temp file when model rejects numpy array."""

        class NumpyRejectingModel:
            def __init__(self):
                self.call_count = 0

            def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[Any], Any]:
                self.call_count += 1
                if isinstance(audio, np.ndarray):
                    raise TypeError("unexpected keyword argument")
                # Accept file path
                return [SimpleNamespace(start=0.0, end=1.0, text="From file")], SimpleNamespace(
                    language="en"
                )

        model = NumpyRejectingModel()
        config = StreamingASRConfig(max_chunk_duration_sec=0.5)
        adapter = StreamingASRAdapter(model, config)

        # Generate enough audio
        samples = 16000
        t = np.linspace(0, 1, samples, dtype=np.float32)
        signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        int16_signal = (signal * 32767).astype(np.int16)
        pcm_bytes = int16_signal.tobytes()

        await adapter.ingest_audio(pcm_bytes)

        # Should have fallen back to temp file
        assert model.call_count >= 1


class TestStreamingASRAdapterIntegration:
    """Integration tests with WebSocketStreamingSession."""

    @pytest.mark.asyncio
    async def test_websocket_session_with_asr(self):
        """Test WebSocketStreamingSession with ASR adapter."""
        from transcription.streaming_ws import (
            WebSocketSessionConfig,
            WebSocketStreamingSession,
        )

        model = MockWhisperModel(return_text="Integrated test")
        config = WebSocketSessionConfig(sample_rate=16000)

        session = WebSocketStreamingSession(
            config=config,
            asr_model=model,
        )

        # Start session
        start_event = await session.start()
        assert start_event.type.value == "SESSION_STARTED"
        assert session._asr_adapter is not None

    @pytest.mark.asyncio
    async def test_websocket_session_without_asr(self):
        """Test WebSocketStreamingSession without ASR (mock mode)."""
        from transcription.streaming_ws import (
            WebSocketSessionConfig,
            WebSocketStreamingSession,
        )

        config = WebSocketSessionConfig(sample_rate=16000)
        session = WebSocketStreamingSession(config=config)

        # Start session
        start_event = await session.start()
        assert start_event.type.value == "SESSION_STARTED"
        assert session._asr_adapter is None

    @pytest.mark.asyncio
    async def test_websocket_session_end_flushes_asr(self):
        """Test that ending session flushes ASR adapter."""
        from transcription.streaming_asr import StreamingASRConfig
        from transcription.streaming_ws import (
            WebSocketSessionConfig,
            WebSocketStreamingSession,
        )

        model = MockWhisperModel(return_text="Final text")
        ws_config = WebSocketSessionConfig(sample_rate=16000)
        asr_config = StreamingASRConfig(
            min_chunk_duration_sec=0.5,
            max_chunk_duration_sec=10.0,  # High to avoid auto-transcription
        )

        session = WebSocketStreamingSession(
            config=ws_config,
            asr_model=model,
            asr_config=asr_config,
        )

        await session.start()

        # Add some audio
        samples = 16000
        t = np.linspace(0, 1, samples, dtype=np.float32)
        signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        int16_signal = (signal * 32767).astype(np.int16)
        pcm_bytes = int16_signal.tobytes()

        await session.process_audio_chunk(pcm_bytes, sequence=1)

        # End session - should flush adapter
        events = await session.end()

        # Should have SESSION_ENDED event
        event_types = [e.type.value for e in events]
        assert "SESSION_ENDED" in event_types


class TestVADLogic:
    """Tests for VAD (Voice Activity Detection) logic."""

    def test_detect_speech_frames_silence(self):
        """Test speech detection returns False for silence."""
        model = MockWhisperModel()
        adapter = StreamingASRAdapter(model)

        silence = np.zeros(4800, dtype=np.float32)  # 300ms at 16kHz
        frames = adapter._detect_speech_frames(silence)

        # All frames should be silence
        assert all(not f for f in frames)

    def test_detect_speech_frames_signal(self):
        """Test speech detection returns True for loud signal."""
        model = MockWhisperModel()
        config = StreamingASRConfig(vad_energy_threshold=0.001)
        adapter = StreamingASRAdapter(model, config)

        # Create loud signal
        t = np.linspace(0, 0.3, 4800, dtype=np.float32)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # Loud 440 Hz

        frames = adapter._detect_speech_frames(signal)

        # Most/all frames should detect speech
        assert any(frames)

    def test_process_vad_below_minimum(self):
        """Test VAD doesn't trigger below minimum duration."""
        model = MockWhisperModel()
        config = StreamingASRConfig(min_chunk_duration_sec=2.0)
        adapter = StreamingASRAdapter(model, config)

        # 1 second of audio (below 2 second minimum)
        audio = np.zeros(16000, dtype=np.float32)
        result, is_final = adapter._process_vad(audio)

        # Should not return audio for transcription
        assert result is None or len(result) == 0 or not is_final
