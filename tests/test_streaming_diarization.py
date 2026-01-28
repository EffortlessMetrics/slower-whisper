"""
Tests for incremental diarization hooks (#86).

Tests cover:
- MockDiarizationHook behavior
- EnergyVADDiarizer voice activity detection
- PyAnnoteIncrementalDiarizer (with mocked pyannote)
- Factory functions
- Integration with WebSocketStreamingSession
"""

from __future__ import annotations

import struct
import wave
from unittest.mock import MagicMock

import pytest

from transcription.streaming_diarization import (
    EnergyVADConfig,
    EnergyVADDiarizer,
    IncrementalDiarizationConfig,
    MockDiarizationHook,
    PyAnnoteIncrementalDiarizer,
    create_energy_vad_hook,
    create_mock_hook,
    create_pyannote_hook,
)
from transcription.streaming_ws import SpeakerAssignment

# =============================================================================
# Test Fixtures
# =============================================================================


def make_silence_audio(duration_sec: float, sample_rate: int = 16000) -> bytes:
    """Generate silent audio (all zeros)."""
    num_samples = int(duration_sec * sample_rate)
    return b"\x00\x00" * num_samples


def make_tone_audio(
    duration_sec: float,
    frequency_hz: float = 440.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> bytes:
    """Generate a sine wave tone."""
    import math

    num_samples = int(duration_sec * sample_rate)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = amplitude * math.sin(2 * math.pi * frequency_hz * t)
        # Convert to 16-bit signed integer
        int_value = int(value * 32767)
        samples.append(int_value)

    return struct.pack(f"<{num_samples}h", *samples)


def make_alternating_audio(
    durations: list[float],
    tones: list[bool],
    sample_rate: int = 16000,
) -> bytes:
    """Generate audio with alternating tone/silence segments."""
    audio_parts = []
    for duration, is_tone in zip(durations, tones, strict=True):
        if is_tone:
            audio_parts.append(make_tone_audio(duration, sample_rate=sample_rate))
        else:
            audio_parts.append(make_silence_audio(duration, sample_rate=sample_rate))
    return b"".join(audio_parts)


# =============================================================================
# MockDiarizationHook Tests
# =============================================================================


class TestMockDiarizationHook:
    """Tests for MockDiarizationHook."""

    @pytest.mark.asyncio
    async def test_basic_mock_diarization(self) -> None:
        """Test mock hook generates alternating speaker assignments."""
        hook = MockDiarizationHook(segment_duration=3.0, num_speakers=2)

        # 9 seconds of audio
        audio = make_silence_audio(9.0)
        assignments = await hook(audio, sample_rate=16000)

        assert len(assignments) == 3
        assert assignments[0].speaker_id == "spk_0"
        assert assignments[1].speaker_id == "spk_1"
        assert assignments[2].speaker_id == "spk_0"

        # Check time ranges
        assert assignments[0].start == pytest.approx(0.0)
        assert assignments[0].end == pytest.approx(3.0)
        assert assignments[1].start == pytest.approx(3.0)
        assert assignments[1].end == pytest.approx(6.0)
        assert assignments[2].start == pytest.approx(6.0)
        assert assignments[2].end == pytest.approx(9.0)

    @pytest.mark.asyncio
    async def test_mock_with_three_speakers(self) -> None:
        """Test mock hook with three speakers."""
        hook = MockDiarizationHook(segment_duration=2.0, num_speakers=3)

        audio = make_silence_audio(6.0)
        assignments = await hook(audio, sample_rate=16000)

        assert len(assignments) == 3
        assert assignments[0].speaker_id == "spk_0"
        assert assignments[1].speaker_id == "spk_1"
        assert assignments[2].speaker_id == "spk_2"

    @pytest.mark.asyncio
    async def test_mock_short_audio(self) -> None:
        """Test mock hook with audio shorter than segment duration."""
        hook = MockDiarizationHook(segment_duration=5.0)

        audio = make_silence_audio(2.0)
        assignments = await hook(audio, sample_rate=16000)

        assert len(assignments) == 1
        assert assignments[0].speaker_id == "spk_0"
        assert assignments[0].end == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_mock_with_latency(self) -> None:
        """Test mock hook simulated latency."""
        import time

        hook = MockDiarizationHook(latency_ms=50.0)

        audio = make_silence_audio(1.0)
        start = time.time()
        await hook(audio, sample_rate=16000)
        elapsed = time.time() - start

        assert elapsed >= 0.05  # At least 50ms

    @pytest.mark.asyncio
    async def test_mock_confidence(self) -> None:
        """Test mock hook sets confidence scores."""
        hook = MockDiarizationHook()

        audio = make_silence_audio(3.0)
        assignments = await hook(audio, sample_rate=16000)

        assert all(a.confidence == 0.9 for a in assignments)


# =============================================================================
# EnergyVADDiarizer Tests
# =============================================================================


class TestEnergyVADDiarizer:
    """Tests for EnergyVADDiarizer."""

    @pytest.mark.asyncio
    async def test_detects_tone_as_speech(self) -> None:
        """Test VAD detects tone as speech."""
        hook = EnergyVADDiarizer(
            EnergyVADConfig(
                energy_threshold=0.01,
                min_speech_duration_sec=0.3,
            )
        )

        # 2 seconds of loud tone
        audio = make_tone_audio(2.0, amplitude=0.5)
        assignments = await hook(audio, sample_rate=16000)

        assert len(assignments) >= 1
        assert assignments[0].speaker_id == "spk_0"
        # Should cover most of the audio
        assert assignments[0].end > 1.0

    @pytest.mark.asyncio
    async def test_silence_not_detected(self) -> None:
        """Test VAD ignores silence."""
        hook = EnergyVADDiarizer(
            EnergyVADConfig(
                energy_threshold=0.01,
                min_speech_duration_sec=0.3,
            )
        )

        audio = make_silence_audio(2.0)
        assignments = await hook(audio, sample_rate=16000)

        assert len(assignments) == 0

    @pytest.mark.asyncio
    async def test_alternating_speech_silence(self) -> None:
        """Test VAD detects alternating speech/silence pattern."""
        hook = EnergyVADDiarizer(
            EnergyVADConfig(
                energy_threshold=0.005,
                min_speech_duration_sec=0.2,
                min_silence_duration_sec=0.3,
            )
        )

        # 1s tone, 1s silence, 1s tone
        audio = make_alternating_audio(
            durations=[1.0, 1.0, 1.0],
            tones=[True, False, True],
        )
        assignments = await hook(audio, sample_rate=16000)

        # Should detect two speech regions (or merged if silence too short)
        assert len(assignments) >= 1
        assert all(a.speaker_id == "spk_0" for a in assignments)

    @pytest.mark.asyncio
    async def test_vad_confidence(self) -> None:
        """Test VAD sets low confidence (0.5)."""
        hook = EnergyVADDiarizer()

        audio = make_tone_audio(1.0, amplitude=0.5)
        assignments = await hook(audio, sample_rate=16000)

        if assignments:
            assert all(a.confidence == 0.5 for a in assignments)

    @pytest.mark.asyncio
    async def test_short_speech_filtered(self) -> None:
        """Test very short speech regions are filtered out."""
        hook = EnergyVADDiarizer(
            EnergyVADConfig(
                energy_threshold=0.01,
                min_speech_duration_sec=1.0,  # Require 1 second minimum
            )
        )

        # 0.2 second tone (too short)
        audio = make_tone_audio(0.2, amplitude=0.5)
        assignments = await hook(audio, sample_rate=16000)

        assert len(assignments) == 0


# =============================================================================
# PyAnnoteIncrementalDiarizer Tests
# =============================================================================


class TestPyAnnoteIncrementalDiarizer:
    """Tests for PyAnnoteIncrementalDiarizer with mocked pyannote."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = IncrementalDiarizationConfig()
        assert config.device == "auto"
        assert config.min_speakers is None
        assert config.max_speakers is None
        assert config.min_audio_duration_sec == 5.0
        assert config.use_sliding_window is False

    def test_diarizer_creation(self) -> None:
        """Test diarizer creation with config."""
        config = IncrementalDiarizationConfig(
            device="cpu",
            min_speakers=2,
            max_speakers=4,
        )
        diarizer = PyAnnoteIncrementalDiarizer(config)

        assert diarizer.config.device == "cpu"
        assert diarizer.config.min_speakers == 2
        assert diarizer.config.max_speakers == 4
        assert diarizer._diarizer is None  # Lazy initialization

    @pytest.mark.asyncio
    async def test_short_audio_skipped(self) -> None:
        """Test audio shorter than minimum is skipped."""
        config = IncrementalDiarizationConfig(
            min_audio_duration_sec=5.0,
        )
        diarizer = PyAnnoteIncrementalDiarizer(config)

        # 2 seconds of audio (less than 5s minimum)
        audio = make_silence_audio(2.0)
        assignments = await diarizer(audio, sample_rate=16000)

        assert assignments == []
        assert diarizer._diarizer is None  # Should not have initialized

    @pytest.mark.asyncio
    async def test_diarization_with_mock_backend(self) -> None:
        """Test diarization with mocked pyannote backend."""
        from transcription.diarization import SpeakerTurn

        # Create mock diarizer
        mock_diarizer = MagicMock()
        mock_diarizer.run.return_value = [
            SpeakerTurn(start=0.0, end=3.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=3.0, end=6.0, speaker_id="SPEAKER_01"),
        ]

        config = IncrementalDiarizationConfig(min_audio_duration_sec=1.0)
        diarizer = PyAnnoteIncrementalDiarizer(config)
        diarizer._diarizer = mock_diarizer
        diarizer._initialized = True

        # 6 seconds of audio
        audio = make_silence_audio(6.0)
        assignments = await diarizer(audio, sample_rate=16000)

        assert len(assignments) == 2
        assert assignments[0].speaker_id == "spk_0"
        assert assignments[1].speaker_id == "spk_1"
        assert assignments[0].start == 0.0
        assert assignments[0].end == 3.0

    @pytest.mark.asyncio
    async def test_speaker_id_normalization(self) -> None:
        """Test speaker IDs are normalized consistently."""
        from transcription.diarization import SpeakerTurn

        mock_diarizer = MagicMock()
        mock_diarizer.run.return_value = [
            SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_01"),  # Note: 01 first
            SpeakerTurn(start=1.0, end=2.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=2.0, end=3.0, speaker_id="SPEAKER_01"),
        ]

        config = IncrementalDiarizationConfig(min_audio_duration_sec=1.0)
        diarizer = PyAnnoteIncrementalDiarizer(config)
        diarizer._diarizer = mock_diarizer
        diarizer._initialized = True

        audio = make_silence_audio(3.0)
        assignments = await diarizer(audio, sample_rate=16000)

        # SPEAKER_01 appears first, so it should be spk_0
        assert assignments[0].speaker_id == "spk_0"  # SPEAKER_01
        assert assignments[1].speaker_id == "spk_1"  # SPEAKER_00
        assert assignments[2].speaker_id == "spk_0"  # SPEAKER_01 again

    def test_reset_speaker_map(self) -> None:
        """Test speaker map reset."""
        diarizer = PyAnnoteIncrementalDiarizer()
        diarizer._speaker_map = {"SPEAKER_00": 0, "SPEAKER_01": 1}

        diarizer.reset_speaker_map()

        assert diarizer._speaker_map == {}

    @pytest.mark.asyncio
    async def test_sliding_window(self) -> None:
        """Test sliding window extracts recent audio."""
        from transcription.diarization import SpeakerTurn

        mock_diarizer = MagicMock()
        mock_diarizer.run.return_value = [
            SpeakerTurn(start=0.0, end=10.0, speaker_id="SPEAKER_00"),
        ]

        config = IncrementalDiarizationConfig(
            min_audio_duration_sec=1.0,
            use_sliding_window=True,
            window_duration_sec=10.0,
        )
        diarizer = PyAnnoteIncrementalDiarizer(config)
        diarizer._diarizer = mock_diarizer
        diarizer._initialized = True

        # 20 seconds of audio - should only process last 10s
        audio = make_silence_audio(20.0)
        assignments = await diarizer(audio, sample_rate=16000)

        # The assignment timestamps should be offset by 10s
        assert len(assignments) == 1
        assert assignments[0].start == pytest.approx(10.0)
        assert assignments[0].end == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_diarization_failure_graceful(self) -> None:
        """Test diarization failure returns empty list."""
        mock_diarizer = MagicMock()
        mock_diarizer.run.side_effect = RuntimeError("Diarization failed")

        config = IncrementalDiarizationConfig(min_audio_duration_sec=1.0)
        diarizer = PyAnnoteIncrementalDiarizer(config)
        diarizer._diarizer = mock_diarizer
        diarizer._initialized = True

        audio = make_silence_audio(5.0)
        assignments = await diarizer(audio, sample_rate=16000)

        assert assignments == []

    def test_audio_buffer_to_wav(self) -> None:
        """Test audio buffer conversion to WAV file."""
        diarizer = PyAnnoteIncrementalDiarizer()

        audio = make_tone_audio(1.0)
        wav_path = diarizer._audio_buffer_to_wav(audio, sample_rate=16000)

        try:
            assert wav_path.exists()
            with wave.open(str(wav_path), "rb") as wav:
                assert wav.getnchannels() == 1
                assert wav.getsampwidth() == 2
                assert wav.getframerate() == 16000
                assert wav.getnframes() == 16000
        finally:
            wav_path.unlink()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_pyannote_hook(self) -> None:
        """Test create_pyannote_hook factory."""
        hook = create_pyannote_hook(
            device="cpu",
            min_speakers=2,
            max_speakers=4,
            min_audio_duration_sec=10.0,
        )

        assert isinstance(hook, PyAnnoteIncrementalDiarizer)
        assert hook.config.device == "cpu"
        assert hook.config.min_speakers == 2
        assert hook.config.max_speakers == 4
        assert hook.config.min_audio_duration_sec == 10.0

    def test_create_pyannote_hook_sliding_window(self) -> None:
        """Test create_pyannote_hook with sliding window."""
        hook = create_pyannote_hook(
            use_sliding_window=True,
            window_duration_sec=30.0,
        )

        assert hook.config.use_sliding_window is True
        assert hook.config.window_duration_sec == 30.0

    def test_create_energy_vad_hook(self) -> None:
        """Test create_energy_vad_hook factory."""
        hook = create_energy_vad_hook(
            energy_threshold=0.02,
            min_speech_duration_sec=0.5,
        )

        assert isinstance(hook, EnergyVADDiarizer)
        assert hook.config.energy_threshold == 0.02
        assert hook.config.min_speech_duration_sec == 0.5

    def test_create_mock_hook(self) -> None:
        """Test create_mock_hook factory."""
        hook = create_mock_hook(
            segment_duration=5.0,
            num_speakers=3,
            latency_ms=100.0,
        )

        assert isinstance(hook, MockDiarizationHook)
        assert hook.segment_duration == 5.0
        assert hook.num_speakers == 3
        assert hook.latency_ms == 100.0


# =============================================================================
# Integration Tests with WebSocketStreamingSession
# =============================================================================


class TestWebSocketSessionIntegration:
    """Integration tests with WebSocketStreamingSession."""

    @pytest.mark.asyncio
    async def test_session_with_mock_hook(self) -> None:
        """Test WebSocketStreamingSession with mock diarization hook."""
        from transcription.streaming_ws import (
            ServerMessageType,
            WebSocketSessionConfig,
            WebSocketStreamingSession,
        )

        hook = create_mock_hook(segment_duration=2.0)

        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=1.0,  # Trigger every 1 second
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=hook)

        await session.start()

        # Send 3 seconds of audio (should trigger diarization)
        audio = make_silence_audio(3.0)
        events = await session.process_audio_chunk(audio, sequence=1)

        # Should have diarization update
        diar_events = [e for e in events if e.type == ServerMessageType.DIARIZATION_UPDATE]
        assert len(diar_events) == 1

        # Check payload
        payload = diar_events[0].payload
        assert payload["num_speakers"] == 2
        assert "spk_0" in payload["speaker_ids"]
        assert "spk_1" in payload["speaker_ids"]
        assert len(payload["assignments"]) >= 1

    @pytest.mark.asyncio
    async def test_session_final_diarization_on_end(self) -> None:
        """Test final diarization is triggered on session end."""
        from transcription.streaming_ws import (
            ServerMessageType,
            WebSocketSessionConfig,
            WebSocketStreamingSession,
        )

        hook = create_mock_hook(segment_duration=10.0)

        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=60.0,  # High interval - won't trigger during processing
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=hook)

        await session.start()

        # Send audio (won't trigger diarization due to high interval)
        audio = make_silence_audio(5.0)
        events = await session.process_audio_chunk(audio, sequence=1)
        diar_events = [e for e in events if e.type == ServerMessageType.DIARIZATION_UPDATE]
        assert len(diar_events) == 0  # Not triggered yet

        # End session should trigger final diarization
        end_events = await session.end()
        final_diar = [e for e in end_events if e.type == ServerMessageType.DIARIZATION_UPDATE]
        assert len(final_diar) == 1

    @pytest.mark.asyncio
    async def test_session_speaker_assignments_accessible(self) -> None:
        """Test speaker assignments can be retrieved from session."""
        from transcription.streaming_ws import (
            WebSocketSessionConfig,
            WebSocketStreamingSession,
        )

        hook = create_mock_hook(segment_duration=1.0)

        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=0.5,
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=hook)

        await session.start()

        # Initially no assignments
        assert session.get_speaker_assignments() == []

        # Trigger diarization
        audio = make_silence_audio(2.0)
        await session.process_audio_chunk(audio, sequence=1)

        # Now should have assignments
        assignments = session.get_speaker_assignments()
        assert len(assignments) >= 1
        assert assignments[0].speaker_id in ["spk_0", "spk_1"]

    @pytest.mark.asyncio
    async def test_session_diarization_disabled(self) -> None:
        """Test diarization hook not called when disabled."""
        from transcription.streaming_ws import (
            WebSocketSessionConfig,
            WebSocketStreamingSession,
        )

        call_count = 0

        async def counting_hook(audio_buffer: bytes, sample_rate: int):
            nonlocal call_count
            call_count += 1
            return []

        config = WebSocketSessionConfig(
            enable_diarization=False,  # Disabled
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=counting_hook)

        await session.start()
        audio = make_silence_audio(5.0)
        await session.process_audio_chunk(audio, sequence=1)

        assert call_count == 0


# =============================================================================
# SpeakerAssignment Tests
# =============================================================================


class TestSpeakerAssignment:
    """Tests for SpeakerAssignment dataclass."""

    def test_to_dict_with_confidence(self) -> None:
        """Test serialization with confidence."""
        assignment = SpeakerAssignment(
            start=1.5,
            end=3.0,
            speaker_id="spk_0",
            confidence=0.85,
        )

        d = assignment.to_dict()
        assert d == {
            "start": 1.5,
            "end": 3.0,
            "speaker_id": "spk_0",
            "confidence": 0.85,
        }

    def test_to_dict_without_confidence(self) -> None:
        """Test serialization without confidence."""
        assignment = SpeakerAssignment(
            start=0.0,
            end=2.0,
            speaker_id="spk_1",
        )

        d = assignment.to_dict()
        assert d == {
            "start": 0.0,
            "end": 2.0,
            "speaker_id": "spk_1",
        }
        assert "confidence" not in d
