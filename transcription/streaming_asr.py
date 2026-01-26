"""Streaming ASR adapter for WebSocket real-time transcription.

Adapts batch faster-whisper for streaming use via VAD-triggered chunked transcription.
Since faster-whisper has no native streaming mode, this adapter implements a
VAD-triggered approach:

1. Audio chunks arrive as PCM bytes
2. Audio is buffered internally as float32 numpy array
3. Energy-based VAD detects speech boundaries
4. Complete speech segments are transcribed using faster-whisper
5. Results are returned as StreamChunk objects

This provides reasonable streaming latency while leveraging the full
accuracy of faster-whisper's batch transcription.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .streaming import StreamChunk

if TYPE_CHECKING:
    from .asr_engine import WhisperModelProtocol

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamingASRConfig:
    """Configuration for StreamingASRAdapter.

    Attributes:
        min_chunk_duration_sec: Minimum audio duration before attempting transcription.
            Shorter values increase responsiveness but may produce less accurate results.
        max_chunk_duration_sec: Maximum audio duration before forcing transcription.
            Ensures latency doesn't grow unbounded during continuous speech.
        sample_rate: Expected audio sample rate in Hz. Must be 16000 for Whisper.
        vad_energy_threshold: RMS energy threshold for speech detection (0.0-1.0).
            Values below this are considered silence.
        vad_silence_duration_sec: Duration of silence needed to finalize a segment.
        language: Optional language code for transcription.
        beam_size: Beam size for decoding (higher = more accurate but slower).
    """

    min_chunk_duration_sec: float = 1.0
    max_chunk_duration_sec: float = 30.0
    sample_rate: int = 16000
    vad_energy_threshold: float = 0.01
    vad_silence_duration_sec: float = 0.8
    language: str | None = None
    beam_size: int = 5


@dataclass
class VADState:
    """Internal state for voice activity detection.

    Tracks speech/silence transitions and accumulated audio for
    determining when to trigger transcription.
    """

    # Current speech detection state
    is_speech_active: bool = False

    # Audio accumulated during current speech segment
    speech_start_sample: int = 0
    speech_end_sample: int = 0

    # Silence tracking for segment finalization
    silence_start_sample: int | None = None

    # Total samples processed (for absolute timestamps)
    total_samples_processed: int = 0


@dataclass
class TranscriptionSegment:
    """Internal representation of a transcribed segment.

    Used to track partial and finalized transcription results
    before conversion to StreamChunk.
    """

    start: float  # Start time in seconds (absolute)
    end: float  # End time in seconds (absolute)
    text: str
    is_final: bool = False


class StreamingASRAdapter:
    """Adapts batch faster-whisper for streaming use via VAD-triggered chunked transcription.

    This adapter enables real-time transcription by:
    1. Buffering incoming PCM audio as float32 numpy arrays
    2. Using energy-based VAD to detect speech boundaries
    3. Transcribing complete speech segments using faster-whisper
    4. Returning results as StreamChunk objects compatible with StreamingSession

    The adapter maintains internal state across audio chunks, enabling it to
    detect speech segments that span multiple `ingest_audio` calls.

    Example:
        >>> adapter = StreamingASRAdapter(model, config)
        >>> for pcm_bytes in audio_stream:
        ...     chunks = await adapter.ingest_audio(pcm_bytes)
        ...     for chunk in chunks:
        ...         print(f"[{chunk['start']:.2f}-{chunk['end']:.2f}] {chunk['text']}")
        >>> final_chunks = await adapter.flush()  # End-of-stream

    Thread Safety:
        This class is NOT thread-safe. Each WebSocket session should have
        its own adapter instance.
    """

    def __init__(
        self,
        model: WhisperModelProtocol,
        config: StreamingASRConfig | None = None,
    ) -> None:
        """Initialize the streaming ASR adapter.

        Args:
            model: A faster-whisper model instance (WhisperModel or compatible).
                   Must implement the transcribe() method accepting numpy arrays.
            config: Configuration for streaming behavior. Defaults to StreamingASRConfig().
        """
        self.model = model
        self.config = config or StreamingASRConfig()

        # Audio buffer (float32, mono, 16kHz)
        self._audio_buffer: np.ndarray = np.array([], dtype=np.float32)

        # VAD state
        self._vad_state = VADState()

        # Pending speech audio for transcription
        self._pending_speech: np.ndarray = np.array([], dtype=np.float32)

        # Track absolute time offset for accurate timestamps
        self._time_offset: float = 0.0

        # Samples per second (for timestamp calculation)
        self._samples_per_sec = self.config.sample_rate

        logger.debug(
            "StreamingASRAdapter initialized: config=%s",
            self.config,
        )

    def _pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert PCM S16LE bytes to float32 numpy array.

        Args:
            pcm_bytes: Raw PCM audio in signed 16-bit little-endian format.

        Returns:
            Numpy float32 array with values normalized to [-1.0, 1.0].
        """
        # Convert bytes to int16 array
        int16_array = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Normalize to float32 [-1.0, 1.0]
        float32_array = int16_array.astype(np.float32) / 32768.0

        return float32_array

    def _calculate_energy(self, audio: np.ndarray) -> float:
        """Calculate RMS energy of audio segment.

        Args:
            audio: Float32 audio array.

        Returns:
            RMS energy value (0.0-1.0 for normalized audio).
        """
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio**2)))

    def _detect_speech_frames(self, audio: np.ndarray, frame_size_ms: int = 30) -> list[bool]:
        """Detect speech in audio using frame-wise energy analysis.

        Args:
            audio: Float32 audio array.
            frame_size_ms: Frame size in milliseconds for energy calculation.

        Returns:
            List of booleans indicating speech presence for each frame.
        """
        frame_size = int(self.config.sample_rate * frame_size_ms / 1000)
        num_frames = len(audio) // frame_size

        speech_frames = []
        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            energy = self._calculate_energy(frame)
            speech_frames.append(energy > self.config.vad_energy_threshold)

        return speech_frames

    def _process_vad(
        self,
        new_audio: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        """Process audio through VAD to detect speech segments.

        Updates internal VAD state and determines when to trigger transcription.

        Args:
            new_audio: New audio samples to process.

        Returns:
            Tuple of (audio_to_transcribe, is_final):
            - audio_to_transcribe: Speech audio ready for transcription, or None
            - is_final: Whether this is a finalized segment (silence detected)
        """
        if len(new_audio) == 0:
            return None, False

        # Append new audio to pending speech buffer
        self._pending_speech = np.concatenate([self._pending_speech, new_audio])

        # Calculate current duration
        pending_duration = len(self._pending_speech) / self._samples_per_sec

        # Check if we have enough audio
        if pending_duration < self.config.min_chunk_duration_sec:
            return None, False

        # Force transcription if we've hit max duration
        if pending_duration >= self.config.max_chunk_duration_sec:
            audio_to_transcribe = self._pending_speech.copy()
            self._pending_speech = np.array([], dtype=np.float32)
            logger.debug(
                "Force transcription at max duration: %.2f sec",
                pending_duration,
            )
            return audio_to_transcribe, False

        # Analyze recent audio for silence
        # Look at the last vad_silence_duration_sec of audio
        silence_samples = int(self.config.vad_silence_duration_sec * self._samples_per_sec)
        if len(self._pending_speech) >= silence_samples:
            recent_audio = self._pending_speech[-silence_samples:]
            speech_frames = self._detect_speech_frames(recent_audio)

            # If all recent frames are silence, finalize the segment
            if speech_frames and not any(speech_frames):
                # Trim silence from the end
                audio_to_transcribe = self._pending_speech[:-silence_samples]
                if len(audio_to_transcribe) > 0:
                    self._pending_speech = np.array([], dtype=np.float32)
                    logger.debug(
                        "Finalize segment on silence: %.2f sec",
                        len(audio_to_transcribe) / self._samples_per_sec,
                    )
                    return audio_to_transcribe, True
                else:
                    # Only silence, discard it
                    self._pending_speech = np.array([], dtype=np.float32)
                    return None, False

        # Not enough audio or speech ongoing - return partial if we have min duration
        if pending_duration >= self.config.min_chunk_duration_sec:
            # Return a copy for partial transcription without clearing buffer
            # This enables showing partial results while accumulating more audio
            return self._pending_speech.copy(), False

        return None, False

    async def _transcribe_audio(
        self,
        audio: np.ndarray,
    ) -> list[TranscriptionSegment]:
        """Transcribe audio using faster-whisper model.

        This method runs the blocking transcription in a thread pool to avoid
        blocking the asyncio event loop.

        Args:
            audio: Float32 audio array (16kHz mono).

        Returns:
            List of TranscriptionSegment objects.
        """
        if len(audio) == 0:
            return []

        def _sync_transcribe() -> list[TranscriptionSegment]:
            """Synchronous transcription worker."""
            try:
                # Check minimum audio length (faster-whisper requires at least 400ms)
                min_samples = int(0.4 * self._samples_per_sec)
                if len(audio) < min_samples:
                    logger.debug(
                        "Audio too short for transcription: %d samples (need %d)",
                        len(audio),
                        min_samples,
                    )
                    return []

                # Build kwargs for transcribe
                kwargs: dict[str, Any] = {
                    "beam_size": self.config.beam_size,
                }
                if self.config.language:
                    kwargs["language"] = self.config.language

                # Check if model supports numpy array input
                # Some older versions or mocks may not
                # Note: faster-whisper accepts str | BinaryIO | np.ndarray but our
                # WhisperModelProtocol only declares str for compatibility with
                # the batch transcription path.
                try:
                    # Cast to Any to bypass the Protocol's str-only signature
                    # faster-whisper actually accepts numpy arrays
                    audio_input: Any = audio
                    segments_iter, _info = self.model.transcribe(audio_input, **kwargs)
                except TypeError:
                    # Fall back to temp file approach
                    logger.debug("Model doesn't accept numpy array, using temp file")
                    return self._transcribe_via_tempfile(audio, kwargs)

                # Collect segments
                segments: list[TranscriptionSegment] = []
                for seg in segments_iter:
                    text = getattr(seg, "text", "").strip()
                    if text:
                        segments.append(
                            TranscriptionSegment(
                                start=float(getattr(seg, "start", 0.0)),
                                end=float(getattr(seg, "end", 0.0)),
                                text=text,
                            )
                        )

                return segments

            except Exception as e:
                logger.warning("Transcription failed: %s", e)
                return []

        # Run in thread pool to avoid blocking event loop
        return await asyncio.to_thread(_sync_transcribe)

    def _transcribe_via_tempfile(
        self,
        audio: np.ndarray,
        kwargs: dict[str, Any],
    ) -> list[TranscriptionSegment]:
        """Transcribe audio by writing to a temporary WAV file.

        Fallback method when model doesn't support numpy array input.

        Args:
            audio: Float32 audio array.
            kwargs: Additional transcribe kwargs.

        Returns:
            List of TranscriptionSegment objects.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp_path = Path(tmp.name)

            # Convert float32 to int16 for WAV
            int16_audio = (audio * 32767).astype(np.int16)

            # Write WAV file
            with wave.open(str(tmp_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self.config.sample_rate)
                wav.writeframes(int16_audio.tobytes())

            # Transcribe from file
            segments_iter, _info = self.model.transcribe(str(tmp_path), **kwargs)

            segments: list[TranscriptionSegment] = []
            for seg in segments_iter:
                text = getattr(seg, "text", "").strip()
                if text:
                    segments.append(
                        TranscriptionSegment(
                            start=float(getattr(seg, "start", 0.0)),
                            end=float(getattr(seg, "end", 0.0)),
                            text=text,
                        )
                    )

            return segments

    def _segments_to_chunks(
        self,
        segments: list[TranscriptionSegment],
        audio_duration: float,
    ) -> list[StreamChunk]:
        """Convert TranscriptionSegments to StreamChunk format.

        Adjusts timestamps to absolute time in the stream.

        Args:
            segments: Transcription segments with relative timestamps.
            audio_duration: Duration of the transcribed audio segment.

        Returns:
            List of StreamChunk objects with absolute timestamps.
        """
        chunks: list[StreamChunk] = []

        for seg in segments:
            # Adjust timestamps to absolute stream time
            abs_start = self._time_offset + seg.start
            abs_end = self._time_offset + seg.end

            chunk: StreamChunk = {
                "start": abs_start,
                "end": abs_end,
                "text": seg.text,
                "speaker_id": None,  # Streaming adapter doesn't do diarization
            }
            chunks.append(chunk)

        return chunks

    async def ingest_audio(self, pcm_bytes: bytes) -> list[StreamChunk]:
        """Ingest PCM audio bytes and return any resulting transcription chunks.

        This is the main entry point for streaming audio. Audio is buffered
        and processed through VAD to detect speech segments. When a complete
        speech segment is detected, it is transcribed and returned as StreamChunk
        objects.

        Args:
            pcm_bytes: Raw PCM audio in signed 16-bit little-endian format (PCM_S16LE).
                      Expected sample rate is configured in StreamingASRConfig.

        Returns:
            List of StreamChunk objects (may be empty if no transcription ready).

        Example:
            >>> chunks = await adapter.ingest_audio(audio_bytes)
            >>> for chunk in chunks:
            ...     session.ingest_chunk(chunk)  # Feed to StreamingSession
        """
        if not pcm_bytes:
            return []

        # Convert PCM to float32
        new_audio = self._pcm_bytes_to_float32(pcm_bytes)

        # Append to buffer
        self._audio_buffer = np.concatenate([self._audio_buffer, new_audio])

        # Process through VAD
        audio_to_transcribe, is_final = self._process_vad(new_audio)

        if audio_to_transcribe is None:
            return []

        # Transcribe the audio
        segments = await self._transcribe_audio(audio_to_transcribe)

        if not segments:
            return []

        # Calculate audio duration for time offset update
        audio_duration = len(audio_to_transcribe) / self._samples_per_sec

        # Convert to StreamChunk format
        chunks = self._segments_to_chunks(segments, audio_duration)

        # Update time offset if this was a finalized segment
        if is_final:
            self._time_offset += audio_duration

        logger.debug(
            "Transcribed %d chunks from %.2f sec audio (final=%s)",
            len(chunks),
            audio_duration,
            is_final,
        )

        return chunks

    async def flush(self) -> list[StreamChunk]:
        """Flush any remaining audio buffer and return final transcription.

        Call this at end-of-stream to ensure all accumulated audio is
        transcribed, even if it doesn't meet the normal VAD criteria.

        Returns:
            List of final StreamChunk objects.
        """
        if len(self._pending_speech) == 0:
            return []

        # Transcribe whatever we have left
        segments = await self._transcribe_audio(self._pending_speech)

        if not segments:
            self._pending_speech = np.array([], dtype=np.float32)
            return []

        # Calculate audio duration
        audio_duration = len(self._pending_speech) / self._samples_per_sec

        # Convert to StreamChunk format
        chunks = self._segments_to_chunks(segments, audio_duration)

        # Update time offset
        self._time_offset += audio_duration

        # Clear buffer
        self._pending_speech = np.array([], dtype=np.float32)

        logger.debug("Flushed %d final chunks from %.2f sec audio", len(chunks), audio_duration)

        return chunks

    def reset(self) -> None:
        """Reset adapter state for a new stream.

        Clears all internal buffers and resets time tracking.
        Useful when reusing an adapter for a new session.
        """
        self._audio_buffer = np.array([], dtype=np.float32)
        self._pending_speech = np.array([], dtype=np.float32)
        self._vad_state = VADState()
        self._time_offset = 0.0
        logger.debug("StreamingASRAdapter reset")

    @property
    def buffer_duration_sec(self) -> float:
        """Return current buffer duration in seconds.

        Useful for monitoring backpressure or debugging.
        """
        return len(self._audio_buffer) / self._samples_per_sec

    @property
    def pending_duration_sec(self) -> float:
        """Return duration of pending speech audio in seconds.

        Audio accumulated since last transcription.
        """
        return len(self._pending_speech) / self._samples_per_sec


__all__ = [
    "StreamingASRAdapter",
    "StreamingASRConfig",
]
