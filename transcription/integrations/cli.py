"""CLI commands for integrations (webhooks, RAG export).

Provides CLI commands for:
- `slower-whisper export rag`: Export transcript to RAG bundle
- `slower-whisper webhook test`: Send test event to webhook
- `slower-whisper webhook send`: Send transcript to webhook

Example usage:
    $ slower-whisper export rag transcript.json --strategy by_turn -o output.json
    $ slower-whisper webhook test https://api.example.com/webhook
    $ slower-whisper webhook send transcript.json --url https://api.example.com/webhook
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_integrations_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Build CLI parsers for integration commands."""
    # Export RAG subcommand (added to existing export command)
    # We'll add this as a separate top-level command since export already exists

    # ============================================================================
    # rag subcommand
    # ============================================================================
    p_rag = subparsers.add_parser(
        "rag",
        help="Export transcript to RAG bundle for vector database ingestion.",
    )
    p_rag.add_argument(
        "transcript",
        type=Path,
        help="Path to transcript JSON file.",
    )
    p_rag.add_argument(
        "--strategy",
        choices=["by_segment", "by_speaker_turn", "by_time", "by_topic"],
        default="by_speaker_turn",
        help="Chunking strategy (default: by_speaker_turn).",
    )
    p_rag.add_argument(
        "--time-window",
        type=float,
        default=30.0,
        help="Time window in seconds for by_time strategy (default: 30).",
    )
    p_rag.add_argument(
        "--embed",
        action="store_true",
        help="Generate embeddings using sentence-transformers.",
    )
    p_rag.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for embeddings (default: all-MiniLM-L6-v2).",
    )
    p_rag.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <transcript>_rag.json).",
    )

    # ============================================================================
    # webhook subcommand
    # ============================================================================
    p_webhook = subparsers.add_parser(
        "webhook",
        help="Webhook integration utilities.",
    )
    webhook_sub = p_webhook.add_subparsers(dest="webhook_action", required=True)

    # webhook test
    p_test = webhook_sub.add_parser(
        "test",
        help="Send a test event to verify webhook endpoint.",
    )
    p_test.add_argument(
        "url",
        help="Webhook URL to test.",
    )
    p_test.add_argument(
        "--bearer-token",
        help="Bearer token for authentication.",
    )
    p_test.add_argument(
        "--hmac-secret",
        help="HMAC secret for signature verification.",
    )
    p_test.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds (default: 10).",
    )

    # webhook send
    p_send = webhook_sub.add_parser(
        "send",
        help="Send transcript to webhook endpoint.",
    )
    p_send.add_argument(
        "transcript",
        type=Path,
        help="Path to transcript JSON file.",
    )
    p_send.add_argument(
        "--url",
        required=True,
        help="Webhook URL to send to.",
    )
    p_send.add_argument(
        "--bearer-token",
        help="Bearer token for authentication.",
    )
    p_send.add_argument(
        "--hmac-secret",
        help="HMAC secret for signature verification.",
    )
    p_send.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30).",
    )

    # webhook retry
    p_retry = webhook_sub.add_parser(
        "retry",
        help="Retry failed deliveries from dead letter queue.",
    )
    p_retry.add_argument(
        "--url",
        required=True,
        help="Webhook URL (must match original URL).",
    )
    p_retry.add_argument(
        "--dead-letter-path",
        type=Path,
        default=Path("dead_letters.json"),
        help="Path to dead letter queue file (default: dead_letters.json).",
    )
    p_retry.add_argument(
        "--bearer-token",
        help="Bearer token for authentication.",
    )


def handle_rag_command(args: argparse.Namespace) -> int:
    """Handle rag export command."""
    from ..writers import load_transcript_from_json
    from .rag_export import ChunkingStrategy, RAGExporter, RAGExporterConfig

    # Load transcript
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}", file=sys.stderr)
        return 1

    transcript = load_transcript_from_json(transcript_path)

    # Configure exporter
    strategy = ChunkingStrategy(args.strategy)
    config = RAGExporterConfig(
        chunking_strategy=strategy,
        time_window_seconds=args.time_window,
        include_embeddings=args.embed,
        embedding_model=args.embedding_model,
    )

    # Export
    exporter = RAGExporter(config)
    bundle = exporter.export(transcript)

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = transcript_path.with_suffix("").with_suffix("_rag.json")

    # Save
    bundle.save(output_path)

    print(f"[done] Exported {len(bundle.chunks)} chunks to {output_path}")
    print(f"  Strategy: {strategy.value}")
    print(f"  Duration: {bundle.metadata['total_duration']:.1f}s")
    print(f"  Speakers: {', '.join(bundle.metadata['speakers']) or 'none'}")
    if args.embed:
        print(f"  Embeddings: {config.embedding_model}")

    return 0


def handle_webhook_command(args: argparse.Namespace) -> int:
    """Handle webhook subcommands."""
    if args.webhook_action == "test":
        return asyncio.run(_webhook_test(args))
    elif args.webhook_action == "send":
        return asyncio.run(_webhook_send(args))
    elif args.webhook_action == "retry":
        return asyncio.run(_webhook_retry(args))
    else:
        print(f"Unknown webhook action: {args.webhook_action}", file=sys.stderr)
        return 1


async def _webhook_test(args: argparse.Namespace) -> int:
    """Send test event to webhook."""
    from .events import create_session_started_event
    from .webhooks import AuthConfig, WebhookConfig, WebhookSink

    # Build config
    auth = AuthConfig()
    if args.bearer_token:
        auth = AuthConfig(type="bearer", token=args.bearer_token)

    config = WebhookConfig(
        url=args.url,
        auth=auth,
        timeout=args.timeout,
        hmac_secret=args.hmac_secret,
    )

    # Create test event
    event = create_session_started_event(
        session_id="test-session-001",
        config={"test": True},
        metadata={"source": "slower-whisper webhook test"},
    )

    # Send
    sink = WebhookSink(config)
    try:
        success = await sink.send_event(event, blocking=True)
        if success:
            print(f"[ok] Test event sent successfully to {args.url}")
            print(f"  Event ID: {event.event_id}")
            print(f"  Event Type: {event.event_type.value}")
            return 0
        else:
            print(f"[error] Failed to send test event to {args.url}", file=sys.stderr)
            dlq = sink.get_dead_letter_queue()
            if dlq:
                print(f"  Error: {dlq[-1].error}", file=sys.stderr)
            return 1
    finally:
        await sink.close()


async def _webhook_send(args: argparse.Namespace) -> int:
    """Send transcript to webhook."""
    from ..writers import load_transcript_from_json
    from .events import create_transcript_event
    from .webhooks import AuthConfig, WebhookConfig, WebhookSink

    # Load transcript
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}", file=sys.stderr)
        return 1

    transcript = load_transcript_from_json(transcript_path)

    # Build config
    auth = AuthConfig()
    if args.bearer_token:
        auth = AuthConfig(type="bearer", token=args.bearer_token)

    config = WebhookConfig(
        url=args.url,
        auth=auth,
        timeout=args.timeout,
        hmac_secret=args.hmac_secret,
    )

    # Create event
    event = create_transcript_event(transcript)

    # Send
    sink = WebhookSink(config)
    try:
        success = await sink.send_event(event, blocking=True)
        if success:
            print(f"[ok] Transcript sent successfully to {args.url}")
            print(f"  File: {transcript.file_name}")
            print(f"  Segments: {len(transcript.segments)}")
            print(f"  Duration: {transcript.duration:.1f}s")
            return 0
        else:
            print(f"[error] Failed to send transcript to {args.url}", file=sys.stderr)
            dlq = sink.get_dead_letter_queue()
            if dlq:
                print(f"  Error: {dlq[-1].error}", file=sys.stderr)
            return 1
    finally:
        await sink.close()


async def _webhook_retry(args: argparse.Namespace) -> int:
    """Retry failed deliveries from dead letter queue."""
    from .webhooks import AuthConfig, WebhookConfig, WebhookSink

    # Check dead letter file exists
    dlq_path = Path(args.dead_letter_path)
    if not dlq_path.exists():
        print(f"Error: Dead letter queue file not found: {dlq_path}", file=sys.stderr)
        return 1

    # Build config
    auth = AuthConfig()
    if args.bearer_token:
        auth = AuthConfig(type="bearer", token=args.bearer_token)

    config = WebhookConfig(
        url=args.url,
        auth=auth,
        dead_letter_path=dlq_path,
    )

    # Create sink (will load dead letter queue)
    sink = WebhookSink(config)
    try:
        dlq = sink.get_dead_letter_queue()
        if not dlq:
            print("No entries in dead letter queue.")
            return 0

        print(f"Retrying {len(dlq)} entries from dead letter queue...")
        successful, failed = await sink.retry_dead_letters()

        print("[done] Retry complete")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        if failed > 0:
            return 1
        return 0
    finally:
        await sink.close()


def handle_integrations_command(args: argparse.Namespace) -> int:
    """Route to appropriate integration command handler."""
    if args.command == "rag":
        return handle_rag_command(args)
    elif args.command == "webhook":
        return handle_webhook_command(args)
    else:
        print(f"Unknown integration command: {args.command}", file=sys.stderr)
        return 1
