#!/usr/bin/env python3
"""
WebSocket streaming client demo: real-time transcription via WebSocket (#223).

This example demonstrates how to use the StreamingClient for real-time
audio transcription over WebSocket. It covers:

- Connection management with automatic reconnection
- Session lifecycle (start, send audio, end)
- Event handling via callbacks and async iteration
- Statistics tracking

**Requirements**:
    - websockets package: pip install websockets
    - A running slower-whisper server: uv run python -m transcription.service

**Usage**:

    # Start the server first (in another terminal):
    uv run python -m transcription.service

    # Run the demo:
    uv run python examples/streaming/websocket_client_demo.py

    # With custom server URL:
    uv run python examples/streaming/websocket_client_demo.py --url ws://custom:9000/stream

    # With audio file (simulated streaming):
    uv run python examples/streaming/websocket_client_demo.py --audio path/to/audio.wav

    # Verbose mode:
    uv run python examples/streaming/websocket_client_demo.py -v

**Architecture**:

    Audio Source -> StreamingClient -> WebSocket -> Server -> Events -> Display
                       |                              ^
                       +--- send_audio() chunks ------+
                       |                              |
                       +--- events() async iter <-----+

**Key Features Demonstrated**:

1. **Callback Pattern**: Register callbacks for specific event types
2. **Async Iteration**: Process events via async for loop
3. **Session Management**: Start/end sessions with configuration
4. **Error Handling**: Handle recoverable and non-recoverable errors
5. **Statistics**: Track client-side metrics
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Check for websockets before importing streaming_client
try:
    import websockets  # noqa: F401
except ImportError:
    print("Error: websockets package is required for this demo.")
    print("Install it with: pip install websockets")
    sys.exit(1)

from slower_whisper.pipeline.streaming_client import (
    EventType,
    StreamEvent,
    StreamingClient,
    StreamingConfig,
)

# =============================================================================
# Event Handler Implementation
# =============================================================================


@dataclass
class DemoEventHandler:
    """
    Example event handler that logs events to console.

    This demonstrates the callback pattern for handling streaming events.
    Each callback receives a StreamEvent with the event data.

    Attributes:
        verbose: If True, prints detailed event information.
        segment_count: Running count of finalized segments.
        error_count: Running count of errors encountered.
        segments: List of all finalized segment texts.
    """

    verbose: bool = False
    segment_count: int = field(default=0, init=False)
    error_count: int = field(default=0, init=False)
    segments: list[str] = field(default_factory=list, init=False)

    def on_session_started(self, event: StreamEvent) -> None:
        """Called when session starts successfully."""
        session_id = event.payload.get("session_id", "unknown")
        print(f"\n[SESSION STARTED] stream_id={event.stream_id}")
        print(f"  Session ID: {session_id}")
        print(f"  Server time: {event.ts_server}")

    def on_partial(self, event: StreamEvent) -> None:
        """Called for PARTIAL (in-progress) segments."""
        if not self.verbose:
            return

        segment = event.payload.get("segment", {})
        text = segment.get("text", "")
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)

        print(f"\r[PARTIAL] {start:.2f}s-{end:.2f}s: {text[:50]}...", end="", flush=True)

    def on_finalized(self, event: StreamEvent) -> None:
        """Called when a segment is finalized."""
        self.segment_count += 1

        segment = event.payload.get("segment", {})
        text = segment.get("text", "")
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        speaker = segment.get("speaker_id") or "unknown"

        self.segments.append(text)

        print(f"\n[FINALIZED #{self.segment_count}] {start:.2f}s - {end:.2f}s | speaker={speaker}")
        print(f'  "{text}"')

        if self.verbose:
            audio_state = segment.get("audio_state")
            if audio_state:
                rendering = audio_state.get("rendering", "[no rendering]")
                print(f"  Audio: {rendering}")

    def on_error(self, event: StreamEvent) -> None:
        """Called when an error occurs."""
        self.error_count += 1

        code = event.payload.get("code", "unknown")
        message = event.payload.get("message", "Unknown error")
        recoverable = event.payload.get("recoverable", True)

        severity = "WARNING" if recoverable else "ERROR"
        print(f"\n[{severity}] {code}: {message}")

        if not recoverable:
            print("  This error is not recoverable. Session may be terminated.")

    def on_session_ended(self, event: StreamEvent) -> None:
        """Called when session ends."""
        stats = event.payload.get("stats", {})

        print(f"\n[SESSION ENDED] stream_id={event.stream_id}")
        print("  Server stats:")
        print(f"    Chunks received: {stats.get('chunks_received', 0)}")
        print(f"    Bytes received: {stats.get('bytes_received', 0)}")
        print(f"    Segments finalized: {stats.get('segments_finalized', 0)}")
        print(f"    Duration: {stats.get('duration_sec', 0):.2f}s")

    def print_summary(self) -> None:
        """Print summary at end of demo."""
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"Total segments finalized: {self.segment_count}")
        print(f"Total errors: {self.error_count}")

        if self.segments:
            print("\nFull transcript:")
            print("-" * 40)
            print(" ".join(self.segments))


# =============================================================================
# Audio Generation
# =============================================================================


def generate_silence_chunks(
    duration_sec: float,
    chunk_duration_sec: float = 0.1,
    sample_rate: int = 16000,
) -> list[bytes]:
    """
    Generate silence audio chunks for testing.

    Args:
        duration_sec: Total duration in seconds.
        chunk_duration_sec: Duration per chunk in seconds.
        sample_rate: Audio sample rate in Hz.

    Returns:
        List of audio byte chunks (PCM 16-bit little-endian).
    """
    bytes_per_sample = 2  # 16-bit
    samples_per_chunk = int(sample_rate * chunk_duration_sec)
    bytes_per_chunk = samples_per_chunk * bytes_per_sample

    num_chunks = int(duration_sec / chunk_duration_sec)

    # Generate silence (zeros)
    silence_chunk = b"\x00" * bytes_per_chunk

    return [silence_chunk] * num_chunks


def load_audio_chunks(
    audio_path: Path,
    chunk_duration_sec: float = 0.1,
    sample_rate: int = 16000,
) -> list[bytes]:
    """
    Load audio file and split into chunks.

    Args:
        audio_path: Path to WAV file (must be 16kHz mono).
        chunk_duration_sec: Duration per chunk in seconds.
        sample_rate: Expected sample rate.

    Returns:
        List of audio byte chunks.

    Raises:
        ValueError: If audio format is incompatible.
    """
    import wave

    with wave.open(str(audio_path), "rb") as wav:
        if wav.getframerate() != sample_rate:
            raise ValueError(
                f"Audio must be {sample_rate}Hz, got {wav.getframerate()}Hz. "
                f"Resample with: ffmpeg -i {audio_path} -ar {sample_rate} -ac 1 output.wav"
            )
        if wav.getnchannels() != 1:
            raise ValueError(
                f"Audio must be mono, got {wav.getnchannels()} channels. "
                f"Convert with: ffmpeg -i {audio_path} -ac 1 output.wav"
            )
        if wav.getsampwidth() != 2:
            raise ValueError(f"Audio must be 16-bit, got {wav.getsampwidth() * 8}-bit.")

        frames = wav.readframes(wav.getnframes())

    # Split into chunks
    bytes_per_chunk = int(sample_rate * chunk_duration_sec * 2)  # 16-bit = 2 bytes
    chunks = []

    for i in range(0, len(frames), bytes_per_chunk):
        chunk = frames[i : i + bytes_per_chunk]
        if len(chunk) > 0:
            # Pad last chunk if needed
            if len(chunk) < bytes_per_chunk:
                chunk = chunk + b"\x00" * (bytes_per_chunk - len(chunk))
            chunks.append(chunk)

    return chunks


# =============================================================================
# Main Demo Functions
# =============================================================================


async def run_callback_demo(
    client: StreamingClient,
    audio_chunks: list[bytes],
    chunk_interval_sec: float = 0.1,
) -> None:
    """
    Run demo using callback pattern.

    Args:
        client: Connected StreamingClient.
        audio_chunks: Audio chunks to send.
        chunk_interval_sec: Interval between chunks (simulates real-time).
    """
    print("\n--- Callback Demo ---")
    print(f"Sending {len(audio_chunks)} chunks...")

    # Start session
    await client.start_session(max_gap_sec=1.0)

    # Send audio chunks
    for i, chunk in enumerate(audio_chunks):
        await client.send_audio(chunk)

        # Simulate real-time pacing
        if chunk_interval_sec > 0:
            await asyncio.sleep(chunk_interval_sec)

        # Show progress
        if (i + 1) % 10 == 0:
            print(f"  Sent {i + 1}/{len(audio_chunks)} chunks...", end="\r")

    print(f"  Sent {len(audio_chunks)}/{len(audio_chunks)} chunks")

    # End session and get final events
    final_events = await client.end_session()

    print(f"\nReceived {len(final_events)} final events")


async def run_async_iter_demo(
    client: StreamingClient,
    audio_chunks: list[bytes],
    chunk_interval_sec: float = 0.1,
) -> list[StreamEvent]:
    """
    Run demo using async iteration pattern.

    Args:
        client: Connected StreamingClient.
        audio_chunks: Audio chunks to send.
        chunk_interval_sec: Interval between chunks.

    Returns:
        List of all received events.
    """
    print("\n--- Async Iteration Demo ---")
    print(f"Sending {len(audio_chunks)} chunks...")

    # Start session
    await client.start_session(max_gap_sec=1.0)

    received_events: list[StreamEvent] = []

    # Send audio in background task
    async def send_audio() -> None:
        for i, chunk in enumerate(audio_chunks):
            await client.send_audio(chunk)
            if chunk_interval_sec > 0:
                await asyncio.sleep(chunk_interval_sec)
            if (i + 1) % 10 == 0:
                print(f"  Sent {i + 1}/{len(audio_chunks)} chunks...", end="\r")
        print(f"  Sent {len(audio_chunks)}/{len(audio_chunks)} chunks")
        await client.end_session()

    # Start sending in background
    send_task = asyncio.create_task(send_audio())

    # Receive events via async iteration
    print("\nReceiving events...")
    async for event in client.events():
        received_events.append(event)

        if event.type == EventType.PARTIAL.value:
            print(".", end="", flush=True)
        elif event.type == EventType.FINALIZED.value:
            print("F", end="", flush=True)
        elif event.type == EventType.SESSION_ENDED.value:
            print("\n[SESSION_ENDED]")
            break

    # Wait for send task to complete
    await send_task

    print(f"\nReceived {len(received_events)} events total")
    return received_events


async def run_demo(
    url: str,
    audio_path: Path | None,
    duration_sec: float,
    verbose: bool,
    use_callbacks: bool,
) -> None:
    """
    Main demo runner.

    Args:
        url: WebSocket server URL.
        audio_path: Optional path to audio file.
        duration_sec: Duration of silence to generate if no audio file.
        verbose: Enable verbose output.
        use_callbacks: Use callback pattern instead of async iteration.
    """
    # Create event handler
    handler = DemoEventHandler(verbose=verbose)

    # Create client config
    config = StreamingConfig(
        url=url,
        max_gap_sec=1.0,
        enable_prosody=False,
        enable_emotion=False,
        sample_rate=16000,
        reconnect_attempts=3,
        reconnect_delay=1.0,
        ping_interval=0,  # Disable ping for demo
    )

    # Create client with callbacks
    client = StreamingClient(
        config=config,
        on_partial=handler.on_partial if use_callbacks else None,
        on_finalized=handler.on_finalized if use_callbacks else None,
        on_error=handler.on_error,
        on_session_started=handler.on_session_started if use_callbacks else None,
        on_session_ended=handler.on_session_ended if use_callbacks else None,
    )

    # Load or generate audio
    if audio_path:
        print(f"Loading audio from: {audio_path}")
        audio_chunks = load_audio_chunks(audio_path)
        chunk_interval = 0.1  # Real-time pacing
    else:
        print(f"Generating {duration_sec}s of silence for testing...")
        audio_chunks = generate_silence_chunks(duration_sec)
        chunk_interval = 0.05  # Faster for silence

    print(f"Prepared {len(audio_chunks)} audio chunks")

    # Connect and run demo
    try:
        print(f"\nConnecting to {url}...")
        async with client:
            print(f"Connected! State: {client.state}")

            if use_callbacks:
                await run_callback_demo(client, audio_chunks, chunk_interval)
            else:
                await run_async_iter_demo(client, audio_chunks, chunk_interval)

            # Print client stats
            print("\n--- Client Statistics ---")
            stats = client.stats.to_dict()
            print(f"  Chunks sent: {stats['chunks_sent']}")
            print(f"  Bytes sent: {stats['bytes_sent']}")
            print(f"  Events received: {stats['events_received']}")
            print(f"  Partials: {stats['partials_received']}")
            print(f"  Finalized: {stats['finalized_received']}")
            print(f"  Errors: {stats['errors_received']}")

    except ConnectionRefusedError:
        print(f"\nError: Could not connect to {url}")
        print("Make sure the server is running:")
        print("  uv run python -m transcription.service")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        raise

    # Print handler summary
    if use_callbacks:
        handler.print_summary()


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebSocket streaming client demo (#223)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic demo with generated silence (requires server running)
  %(prog)s

  # With custom server URL
  %(prog)s --url ws://custom:9000/stream

  # With audio file
  %(prog)s --audio path/to/audio.wav

  # Use async iteration instead of callbacks
  %(prog)s --async-iter

  # Verbose output
  %(prog)s -v
        """,
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/stream",
        help="WebSocket server URL (default: ws://localhost:8000/stream)",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Path to audio file (16kHz mono WAV). If not provided, generates silence.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of generated silence in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--async-iter",
        action="store_true",
        help="Use async iteration pattern instead of callbacks",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (show partial segments)",
    )
    args = parser.parse_args()

    # Validate audio file if provided
    if args.audio and not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    # Print header
    print("=" * 60)
    print("WEBSOCKET STREAMING CLIENT DEMO (#223)")
    print("=" * 60)
    print(f"Server URL: {args.url}")
    print(f"Audio: {args.audio or 'generated silence'}")
    print(f"Mode: {'async iteration' if args.async_iter else 'callbacks'}")
    print(f"Verbose: {args.verbose}")

    # Run async demo
    try:
        asyncio.run(
            run_demo(
                url=args.url,
                audio_path=args.audio,
                duration_sec=args.duration,
                verbose=args.verbose,
                use_callbacks=not args.async_iter,
            )
        )
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")


if __name__ == "__main__":
    main()
