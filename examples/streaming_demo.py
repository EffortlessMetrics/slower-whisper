#!/usr/bin/env python3
"""
WebSocket Streaming Demo: v2.0 Streaming API demonstration.

This example demonstrates the complete v2.0 WebSocket streaming API by:

1. Starting a development server (or connecting to an existing one)
2. Streaming a WAV file via WebSocket in chunks
3. Receiving and displaying PARTIAL and FINALIZED events
4. Showing session statistics upon completion
5. (Optional) Demonstrating the resume protocol

**Protocol Flow**:

    Client                          Server
      |                               |
      | CONNECT ws://localhost:8000/stream
      |------------------------------>|
      |                               |
      | START_SESSION {config}        |
      |------------------------------>|
      |                               |
      |      SESSION_STARTED          |
      |<------------------------------|
      |                               |
      | AUDIO_CHUNK {data, sequence}  |
      |------------------------------>|
      |                               |
      |      PARTIAL {segment}        |
      |<------------------------------|
      |                               |
      | ...more audio chunks...       |
      |                               |
      | END_SESSION                   |
      |------------------------------>|
      |                               |
      |      FINALIZED {segment}      |
      |<------------------------------|
      |      SESSION_ENDED {stats}    |
      |<------------------------------|
      |                               |

**Usage**:

    # Stream an audio file to a running server
    python examples/streaming_demo.py path/to/audio.wav

    # Start a dev server and stream (requires uvicorn installed)
    python examples/streaming_demo.py path/to/audio.wav --start-server

    # Custom server URL
    python examples/streaming_demo.py audio.wav --url ws://myserver:8000/stream

    # With optional features enabled
    python examples/streaming_demo.py audio.wav --enable-prosody

    # Demonstrate resume protocol (simulates disconnect)
    python examples/streaming_demo.py audio.wav --demo-resume

**Test Audio**:

    If you don't have an audio file, generate one with:

    # Create 5-second test tone (requires ffmpeg)
    ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 test_audio.wav

    # Or use included test fixtures
    python examples/streaming_demo.py tests/fixtures/synthetic_2speaker.wav

**Requirements**:

    - websockets package: pip install websockets
    - Running server: uvicorn transcription.service:app --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ===========================
# Configuration
# ===========================

DEFAULT_URL = "ws://localhost:8000/stream"
CHUNK_SIZE_BYTES = 4096  # ~0.13 seconds at 16kHz
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit audio


# ===========================
# Event Statistics
# ===========================


@dataclass
class StreamStats:
    """Track streaming session statistics."""

    chunks_sent: int = 0
    bytes_sent: int = 0
    partial_count: int = 0
    finalized_count: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_event_id: int = 0
    segments: list[dict[str, Any]] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print final statistics."""
        duration = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Duration: {duration:.2f}s")
        print(f"Chunks sent: {self.chunks_sent}")
        print(f"Bytes sent: {self.bytes_sent:,}")
        print(f"Partial events: {self.partial_count}")
        print(f"Finalized events: {self.finalized_count}")
        print(f"Errors: {self.errors}")
        print(f"Last event ID: {self.last_event_id}")

        if self.segments:
            print(f"\n--- Finalized Segments ({len(self.segments)}) ---")
            for seg in self.segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "")[:60]
                print(f"  [{start:.2f}s - {end:.2f}s] {text}")


# ===========================
# WebSocket Client
# ===========================


async def stream_audio_file(
    ws_url: str,
    audio_path: Path,
    config: dict[str, Any],
    demo_resume: bool = False,
    verbose: bool = False,
) -> StreamStats:
    """
    Stream an audio file via WebSocket and process events.

    Args:
        ws_url: WebSocket URL to connect to
        audio_path: Path to WAV audio file
        config: Session configuration dict
        demo_resume: If True, simulate disconnect and resume
        verbose: If True, print all events

    Returns:
        StreamStats with session statistics
    """
    try:
        import websockets
    except ImportError:
        print("ERROR: websockets package is required")
        print("Install with: pip install websockets")
        sys.exit(1)

    stats = StreamStats()

    # Read WAV file
    print(f"\nReading audio file: {audio_path}")
    try:
        with wave.open(str(audio_path), "rb") as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_data = wav.readframes(n_frames)

        duration_sec = n_frames / framerate
        print(f"  Channels: {n_channels}, Sample width: {sample_width} bytes")
        print(f"  Sample rate: {framerate} Hz, Duration: {duration_sec:.2f}s")
        print(f"  Total bytes: {len(audio_data):,}")

        if framerate != SAMPLE_RATE:
            print(f"  WARNING: Expected {SAMPLE_RATE}Hz, got {framerate}Hz")
            print("           Audio may not be processed correctly")
    except Exception as e:
        print(f"ERROR: Failed to read audio file: {e}")
        sys.exit(1)

    # Connect to WebSocket
    print(f"\nConnecting to {ws_url}...")

    try:
        async with websockets.connect(ws_url) as ws:
            print("Connected!")

            # Send START_SESSION
            print("\nSending START_SESSION...")
            start_msg = {"type": "START_SESSION", "config": config}
            await ws.send(json.dumps(start_msg))

            # Wait for SESSION_STARTED
            response = await ws.recv()
            event = json.loads(response)
            stats.last_event_id = event.get("event_id", 0)

            if event.get("type") != "SESSION_STARTED":
                print(f"ERROR: Expected SESSION_STARTED, got {event.get('type')}")
                if event.get("type") == "ERROR":
                    print(f"  Code: {event.get('payload', {}).get('code')}")
                    print(f"  Message: {event.get('payload', {}).get('message')}")
                return stats

            session_id = event.get("payload", {}).get("session_id")
            print(f"Session started: {session_id}")

            # Stream audio in chunks
            print(f"\nStreaming audio ({len(audio_data):,} bytes)...")
            offset = 0
            sequence = 0

            # Determine disconnect point for resume demo
            disconnect_point = len(audio_data) // 2 if demo_resume else -1
            disconnected = False

            while offset < len(audio_data):
                # Get next chunk
                chunk = audio_data[offset : offset + CHUNK_SIZE_BYTES]
                if not chunk:
                    break

                sequence += 1
                stats.chunks_sent += 1
                stats.bytes_sent += len(chunk)

                # Encode and send
                chunk_b64 = base64.b64encode(chunk).decode("ascii")
                chunk_msg = {
                    "type": "AUDIO_CHUNK",
                    "data": chunk_b64,
                    "sequence": sequence,
                }
                await ws.send(json.dumps(chunk_msg))

                offset += CHUNK_SIZE_BYTES

                # Demo resume: simulate disconnect at midpoint
                if demo_resume and offset >= disconnect_point and not disconnected:
                    print(f"\n[DEMO] Simulating disconnect at {offset:,} bytes...")
                    print(f"       Last event ID: {stats.last_event_id}")
                    disconnected = True
                    # In a real scenario, we'd reconnect and send RESUME_SESSION
                    # For demo purposes, we'll just continue

                # Show progress every 10 chunks
                if stats.chunks_sent % 10 == 0:
                    pct = (offset / len(audio_data)) * 100
                    print(f"  Progress: {pct:.1f}% ({stats.chunks_sent} chunks)", end="\r")

                # Process any incoming events (non-blocking)
                try:
                    while True:
                        response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                        event = json.loads(response)
                        _handle_event(event, stats, verbose)
                except TimeoutError:
                    pass  # No pending events

            print(f"  Progress: 100% ({stats.chunks_sent} chunks)")

            # Send END_SESSION
            print("\nSending END_SESSION...")
            await ws.send(json.dumps({"type": "END_SESSION"}))

            # Process remaining events until SESSION_ENDED
            print("Processing final events...")
            while True:
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    event = json.loads(response)
                    _handle_event(event, stats, verbose)

                    if event.get("type") == "SESSION_ENDED":
                        # Print server-side stats
                        server_stats = event.get("payload", {}).get("stats", {})
                        print(f"\nServer stats: {server_stats}")
                        break

                except TimeoutError:
                    print("Timeout waiting for SESSION_ENDED")
                    break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")
    except ConnectionRefusedError:
        print(f"ERROR: Could not connect to {ws_url}")
        print("Make sure the server is running:")
        print("  uvicorn transcription.service:app --port 8000")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        stats.errors += 1

    return stats


def _handle_event(event: dict[str, Any], stats: StreamStats, verbose: bool) -> None:
    """Handle an incoming WebSocket event."""
    event_type = event.get("type")
    event_id = event.get("event_id", 0)
    stats.last_event_id = max(stats.last_event_id, event_id)

    if event_type == "PARTIAL":
        stats.partial_count += 1
        if verbose:
            segment = event.get("payload", {}).get("segment", {})
            text = segment.get("text", "")[:50]
            print(f"\n  [PARTIAL #{event_id}] {text}")

    elif event_type == "FINALIZED":
        stats.finalized_count += 1
        segment = event.get("payload", {}).get("segment", {})
        stats.segments.append(segment)
        text = segment.get("text", "")[:60]
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        print(f"\n  [FINALIZED #{event_id}] [{start:.2f}s - {end:.2f}s] {text}")

    elif event_type == "DIARIZATION_UPDATE":
        payload = event.get("payload", {})
        num_speakers = payload.get("num_speakers", 0)
        speaker_ids = payload.get("speaker_ids", [])
        print(f"\n  [DIARIZATION] {num_speakers} speakers: {speaker_ids}")

    elif event_type == "ERROR":
        stats.errors += 1
        payload = event.get("payload", {})
        code = payload.get("code", "unknown")
        message = payload.get("message", "")
        recoverable = payload.get("recoverable", True)
        print(f"\n  [ERROR] {code}: {message} (recoverable={recoverable})")

    elif event_type == "PONG":
        if verbose:
            print(f"\n  [PONG] server_ts={event.get('payload', {}).get('server_timestamp')}")

    elif event_type == "SESSION_STARTED":
        pass  # Already handled

    elif event_type == "SESSION_ENDED":
        pass  # Handled in main loop

    else:
        if verbose:
            print(f"\n  [{event_type}] {event}")


# ===========================
# Server Management
# ===========================


def start_dev_server(port: int = 8000) -> subprocess.Popen:
    """Start a development server in the background."""
    print(f"Starting development server on port {port}...")

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "transcription.service:app",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start with retry
        print("Waiting for server to be ready...")
        import socket

        max_wait = 15  # seconds
        start_wait = time.time()
        while time.time() - start_wait < max_wait:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect(("localhost", port))
                    break
            except (OSError, ConnectionRefusedError):
                time.sleep(0.5)
        else:
            print(f"WARNING: Server may not be ready after {max_wait}s")

        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            print("ERROR: Server failed to start")
            if stderr:
                print(f"  {stderr[:200]}")
            sys.exit(1)

        print(f"Server running on port {port}")
        return proc

    except FileNotFoundError:
        print("ERROR: uvicorn not found")
        print("Install with: pip install uvicorn")
        sys.exit(1)


# ===========================
# Main
# ===========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebSocket Streaming Demo: v2.0 API demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream audio to running server
  %(prog)s path/to/audio.wav

  # Start server automatically
  %(prog)s audio.wav --start-server

  # Custom server URL
  %(prog)s audio.wav --url ws://myserver:8000/stream

  # With prosody enabled
  %(prog)s audio.wav --enable-prosody

  # Demonstrate resume protocol
  %(prog)s audio.wav --demo-resume --verbose

Test Audio:
  If you don't have an audio file, use included fixtures:
    %(prog)s tests/fixtures/synthetic_2speaker.wav

  Or generate a test tone with ffmpeg:
    ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 test.wav
        """,
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to WAV audio file (16kHz mono recommended)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"WebSocket URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start a development server before streaming",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for dev server (default: 8000)",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=1.0,
        help="Gap threshold for segment finalization (default: 1.0)",
    )
    parser.add_argument(
        "--enable-prosody",
        action="store_true",
        help="Enable prosody feature extraction",
    )
    parser.add_argument(
        "--enable-emotion",
        action="store_true",
        help="Enable emotion feature extraction",
    )
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable incremental speaker diarization",
    )
    parser.add_argument(
        "--demo-resume",
        action="store_true",
        help="Demonstrate resume protocol (simulates disconnect)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all events including partials",
    )
    args = parser.parse_args()

    # Validate audio file
    if not args.audio.exists():
        print(f"ERROR: Audio file not found: {args.audio}")
        print("\nYou can use included test fixtures:")
        print("  tests/fixtures/synthetic_2speaker.wav")
        print("  benchmarks/test_audio/test_audio_10s.wav")
        print("\nOr generate a test tone:")
        print('  ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 test.wav')
        sys.exit(1)

    # Build configuration
    config = {
        "max_gap_sec": args.max_gap_sec,
        "enable_prosody": args.enable_prosody,
        "enable_emotion": args.enable_emotion,
        "enable_diarization": args.enable_diarization,
        "sample_rate": SAMPLE_RATE,
        "audio_format": "pcm_s16le",
    }

    # Print header
    print("=" * 60)
    print("WEBSOCKET STREAMING DEMO (v2.0)")
    print("=" * 60)
    print(f"\nAudio: {args.audio}")
    print(f"Server: {args.url}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Start server if requested
    server_proc = None
    if args.start_server:
        server_proc = start_dev_server(args.port)
        # Update URL to match port
        args.url = f"ws://localhost:{args.port}/stream"

    try:
        # Run streaming
        stats = asyncio.run(
            stream_audio_file(
                ws_url=args.url,
                audio_path=args.audio,
                config=config,
                demo_resume=args.demo_resume,
                verbose=args.verbose,
            )
        )

        # Print summary
        stats.print_summary()

    finally:
        # Cleanup server
        if server_proc:
            print("\nStopping development server...")
            server_proc.terminate()
            server_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
