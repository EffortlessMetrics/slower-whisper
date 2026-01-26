"""CLI commands for the Conversation Store.

Provides the `slower-whisper store` subcommand group with commands for:
- ingest: Import transcript JSON files into the store
- query: Search the store with text and filters
- list: List recent entries
- export: Export entries to various formats
- actions list: List action items
- actions complete: Mark an action item as completed
- stats: Show store statistics

Example usage:
    slower-whisper store ingest transcript.json --tags meeting,2024
    slower-whisper store query "budget discussion" --speaker SPEAKER_01
    slower-whisper store list --limit 20 --format json
    slower-whisper store export --format csv --output conversations.csv
    slower-whisper store actions list --status open
    slower-whisper store actions complete abc123
    slower-whisper store stats
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from .store import ConversationStore, get_default_store_path
from .types import (
    ActionItem,
    ActionStatus,
    ConversationEntry,
    DuplicateHandling,
    QueryFilter,
    StoreStats,
    TimeRange,
)


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def _format_size(bytes_size: int) -> str:
    """Format bytes as human-readable string."""
    size: float = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _truncate(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_entries_table(entries: list[ConversationEntry]) -> str:
    """Format entries as a simple table."""
    if not entries:
        return "No entries found."

    lines = []
    # Header
    lines.append(f"{'ID':<30} {'Speaker':<12} {'Time':<12} {'Text':<50}")
    lines.append("-" * 104)

    for entry in entries:
        speaker = entry.speaker_id or "-"
        time_str = f"{entry.start_time:.1f}-{entry.end_time:.1f}s"
        text = _truncate(entry.text, 50)
        entry_id = _truncate(entry.id, 30)
        lines.append(f"{entry_id:<30} {speaker:<12} {time_str:<12} {text:<50}")

    return "\n".join(lines)


def _format_entries_json(entries: list[ConversationEntry]) -> str:
    """Format entries as JSON."""
    return json.dumps([e.to_dict() for e in entries], indent=2)


def _format_actions_table(actions: list[ActionItem]) -> str:
    """Format action items as a simple table."""
    if not actions:
        return "No action items found."

    lines = []
    # Header
    lines.append(f"{'ID':<36} {'Status':<10} {'Source':<20} {'Text':<40}")
    lines.append("-" * 106)

    for action in actions:
        status = action.status.value
        source = _truncate(action.source_file, 20)
        text = _truncate(action.text, 40)
        lines.append(f"{action.id:<36} {status:<10} {source:<20} {text:<40}")

    return "\n".join(lines)


def _format_actions_json(actions: list[ActionItem]) -> str:
    """Format action items as JSON."""
    return json.dumps([a.to_dict() for a in actions], indent=2)


def _format_stats_table(stats: StoreStats) -> str:
    """Format store statistics as a human-readable table."""
    # Handle both TypedDict (dict) and dataclass access patterns
    segment_count = (
        stats.get("segment_count", 0)
        if isinstance(stats, dict)
        else getattr(stats, "segment_count", 0)
    )
    transcript_count = (
        stats.get("transcript_count", 0)
        if isinstance(stats, dict)
        else getattr(stats, "transcript_count", 0)
    )
    speaker_count = (
        stats.get("speaker_count", 0)
        if isinstance(stats, dict)
        else getattr(stats, "speaker_count", 0)
    )
    total_duration = (
        stats.get("total_duration_seconds", 0.0)
        if isinstance(stats, dict)
        else getattr(stats, "total_duration_seconds", 0.0)
    )
    action_count = (
        stats.get("action_item_count", 0)
        if isinstance(stats, dict)
        else getattr(stats, "action_item_count", 0)
    )
    open_actions = (
        stats.get("open_action_items", 0)
        if isinstance(stats, dict)
        else getattr(stats, "open_action_items", 0)
    )
    completed_actions = (
        stats.get("completed_action_items", 0)
        if isinstance(stats, dict)
        else getattr(stats, "completed_action_items", 0)
    )
    storage_size = (
        stats.get("database_size_bytes", 0)
        if isinstance(stats, dict)
        else getattr(stats, "database_size_bytes", 0)
    )
    oldest = (
        stats.get("oldest_ingestion")
        if isinstance(stats, dict)
        else getattr(stats, "oldest_ingestion", None)
    )
    newest = (
        stats.get("newest_ingestion")
        if isinstance(stats, dict)
        else getattr(stats, "newest_ingestion", None)
    )
    topics = stats.get("topics", []) if isinstance(stats, dict) else getattr(stats, "topics", [])

    lines = [
        "Conversation Store Statistics",
        "=" * 40,
        "",
        f"Total segments:       {segment_count:,}",
        f"Total transcripts:    {transcript_count:,}",
        f"Total speakers:       {speaker_count:,}",
        f"Total duration:       {_format_duration(total_duration)}",
        "",
        "Action items:",
        f"  Total:              {action_count:,}",
        f"  Open:               {open_actions:,}",
        f"  Completed:          {completed_actions:,}",
        "",
        f"Storage size:         {_format_size(storage_size)}",
    ]

    if oldest and newest:
        lines.extend(
            [
                "",
                "Date range:",
                f"  Oldest:             {oldest[:16] if oldest else 'N/A'}",
                f"  Newest:             {newest[:16] if newest else 'N/A'}",
            ]
        )

    if topics:
        lines.extend(["", "Top topics:"])
        for topic, count in topics[:5]:
            lines.append(f"  {topic}: {count}")

    return "\n".join(lines)


def _format_stats_json(stats: StoreStats) -> str:
    """Format store statistics as JSON."""
    # StoreStats is a TypedDict (dict), so we can serialize it directly
    return json.dumps(stats, indent=2)


def _export_jsonl(entries: list[dict[str, Any]]) -> str:
    """Export entries as JSON Lines format."""
    lines = [json.dumps(e) for e in entries]
    return "\n".join(lines)


def _export_csv(entries: list[dict[str, Any]]) -> str:
    """Export entries as CSV format."""
    if not entries:
        return ""

    output = io.StringIO()
    # Flatten nested fields for CSV
    fieldnames = [
        "id",
        "text",
        "speaker_id",
        "start_time",
        "end_time",
        "source_file",
        "ingested_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for entry in entries:
        # Create a flat row
        row = {k: entry.get(k, "") for k in fieldnames}
        writer.writerow(row)

    return output.getvalue()


def build_store_parser(subparsers: argparse._SubParsersAction) -> None:
    """Build the store subcommand parser.

    Args:
        subparsers: The subparsers action from the main CLI parser.
    """
    p_store = subparsers.add_parser(
        "store",
        help="Manage the conversation store (ingest, query, export).",
    )

    store_subparsers = p_store.add_subparsers(dest="store_action", required=True)

    # Common store path argument helper
    def add_store_arg(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--store",
            type=Path,
            default=None,
            help=f"Path to store database (default: {get_default_store_path()})",
        )

    # =========================================================================
    # store ingest
    # =========================================================================
    p_ingest = store_subparsers.add_parser(
        "ingest",
        help="Ingest a transcript JSON file into the store.",
    )
    p_ingest.add_argument(
        "file",
        type=Path,
        help="Path to transcript JSON file to ingest.",
    )
    add_store_arg(p_ingest)
    p_ingest.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tags to apply to ingested entries.",
    )
    p_ingest.add_argument(
        "--on-duplicate",
        choices=["skip", "replace", "error"],
        default="skip",
        help="How to handle duplicate entries (default: skip).",
    )

    # =========================================================================
    # store query
    # =========================================================================
    p_query = store_subparsers.add_parser(
        "query",
        help="Search the store for matching entries.",
    )
    p_query.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to search for in entries.",
    )
    add_store_arg(p_query)
    p_query.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Filter by speaker ID.",
    )
    p_query.add_argument(
        "--during",
        type=str,
        default=None,
        metavar="START-END",
        help="Filter by date range (format: YYYY-MM-DD-YYYY-MM-DD).",
    )
    p_query.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Filter by topic.",
    )
    p_query.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results (default: 10).",
    )
    p_query.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )

    # =========================================================================
    # store list
    # =========================================================================
    p_list = store_subparsers.add_parser(
        "list",
        help="List recent entries in the store.",
    )
    add_store_arg(p_list)
    p_list.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of entries to list (default: 10).",
    )
    p_list.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )

    # =========================================================================
    # store export
    # =========================================================================
    p_export = store_subparsers.add_parser(
        "export",
        help="Export entries from the store.",
    )
    add_store_arg(p_export)
    p_export.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional text query to filter exported entries.",
    )
    p_export.add_argument(
        "--format",
        choices=["jsonl", "csv", "parquet"],
        default="jsonl",
        help="Export format (default: jsonl). Parquet requires pyarrow.",
    )
    p_export.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout).",
    )

    # =========================================================================
    # store actions
    # =========================================================================
    p_actions = store_subparsers.add_parser(
        "actions",
        help="Manage action items.",
    )
    actions_subparsers = p_actions.add_subparsers(dest="actions_action", required=True)

    # store actions list
    p_actions_list = actions_subparsers.add_parser(
        "list",
        help="List action items.",
    )
    add_store_arg(p_actions_list)
    p_actions_list.add_argument(
        "--status",
        choices=["open", "completed"],
        default=None,
        help="Filter by status (default: all).",
    )
    p_actions_list.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )

    # store actions complete
    p_actions_complete = actions_subparsers.add_parser(
        "complete",
        help="Mark an action item as completed.",
    )
    p_actions_complete.add_argument(
        "id",
        type=str,
        help="ID of the action item to complete.",
    )
    add_store_arg(p_actions_complete)

    # =========================================================================
    # store stats
    # =========================================================================
    p_stats = store_subparsers.add_parser(
        "stats",
        help="Show store statistics.",
    )
    add_store_arg(p_stats)
    p_stats.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )


def handle_store_command(args: argparse.Namespace) -> int:
    """Handle the store subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        store = ConversationStore(args.store)

        if args.store_action == "ingest":
            return _handle_ingest(store, args)
        elif args.store_action == "query":
            return _handle_query(store, args)
        elif args.store_action == "list":
            return _handle_list(store, args)
        elif args.store_action == "export":
            return _handle_export(store, args)
        elif args.store_action == "actions":
            return _handle_actions(store, args)
        elif args.store_action == "stats":
            return _handle_stats(store, args)
        else:
            print(f"Unknown store action: {args.store_action}", file=sys.stderr)
            return 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1
    finally:
        if "store" in locals():
            store.close()


def _handle_ingest(store: ConversationStore, args: argparse.Namespace) -> int:
    """Handle the ingest subcommand."""
    tags = args.tags.split(",") if args.tags else []
    on_duplicate = DuplicateHandling(args.on_duplicate)

    count = store.ingest_file(args.file, tags=tags, on_duplicate=on_duplicate)

    if count > 0:
        print(f"Ingested {count} entries from {args.file.name}")
    else:
        print(f"No new entries ingested from {args.file.name} (already exists or empty)")

    return 0


def _handle_query(store: ConversationStore, args: argparse.Namespace) -> int:
    """Handle the query subcommand."""
    time_range = None
    if args.during:
        try:
            time_range = TimeRange.parse(args.during)
        except ValueError as e:
            print(f"Invalid date range format: {e}", file=sys.stderr)
            return 1

    filter = QueryFilter(
        text=args.text,
        speaker_id=args.speaker,
        time_range=time_range,
        topic=args.topic,
        limit=args.limit,
    )

    result = store.query(filter)

    if args.format == "json":
        print(_format_entries_json(result.entries))
    else:
        print(_format_entries_table(result.entries))
        if result.total_count > len(result.entries):
            print(f"\nShowing {len(result.entries)} of {result.total_count} total matches")
        print(f"\nQuery completed in {result.query_time_ms:.1f}ms")

    return 0


def _handle_list(store: ConversationStore, args: argparse.Namespace) -> int:
    """Handle the list subcommand."""
    result = store.list_entries(limit=args.limit)

    if args.format == "json":
        print(_format_entries_json(result.entries))
    else:
        print(_format_entries_table(result.entries))
        print(f"\nTotal entries in store: {result.total_count}")

    return 0


def _handle_export(store: ConversationStore, args: argparse.Namespace) -> int:
    """Handle the export subcommand."""
    # Handle parquet export separately (requires output file)
    if args.format == "parquet":
        if not args.output:
            print("Error: Parquet export requires --output file path", file=sys.stderr)
            return 1

        try:
            from .types import StoreQuery, TextQuery

            query = None
            if args.query:
                query = StoreQuery(text=TextQuery(args.query), limit=100000)

            result = store.export_parquet(str(args.output), query=query)
            print(f"Exported {result.record_count} entries to {args.output}")
            return 0
        except ImportError:
            print(
                "Error: Parquet export requires pyarrow. Install with: pip install pyarrow",
                file=sys.stderr,
            )
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Handle other formats using legacy API (returns list of dicts)
    query_filter = QueryFilter(text=args.query, limit=10000) if args.query else None
    export_result = store.export(query=query_filter, format=args.format)

    # The legacy export API with QueryFilter returns a list of dicts
    # Type narrowing for the union type
    entries: list[dict[str, Any]]
    if isinstance(export_result, list):
        entries = export_result
    else:
        # This case happens with StoreQuery input (returns ExportResult)
        # But we're using QueryFilter here, so this shouldn't occur
        print(f"Exported {export_result.record_count} entries")
        return 0

    if args.format == "csv":
        output = _export_csv(entries)
    else:  # jsonl
        output = _export_jsonl(entries)

    if args.output:
        args.output.write_text(output)
        print(f"Exported {len(entries)} entries to {args.output}")
    else:
        print(output)

    return 0


def _handle_actions(store: ConversationStore, args: argparse.Namespace) -> int:
    """Handle the actions subcommand."""
    if args.actions_action == "list":
        status = ActionStatus(args.status) if args.status else None
        actions = store.list_actions(status=status)

        if args.format == "json":
            print(_format_actions_json(actions))
        else:
            print(_format_actions_table(actions))

        return 0

    elif args.actions_action == "complete":
        if store.complete_action(args.id):
            print(f"Action item {args.id} marked as completed")
            return 0
        else:
            print(f"Action item not found: {args.id}", file=sys.stderr)
            return 1

    else:
        print(f"Unknown actions subcommand: {args.actions_action}", file=sys.stderr)
        return 1


def _handle_stats(store: ConversationStore, args: argparse.Namespace) -> int:
    """Handle the stats subcommand."""
    stats = store.get_stats()

    if args.format == "json":
        print(_format_stats_json(stats))
    else:
        print(_format_stats_table(stats))

    return 0
