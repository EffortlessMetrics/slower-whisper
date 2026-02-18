"""
Stable ID generators for run, stream, event, and segment identifiers.

This module provides standardized ID generation functions with documented formats
and stability guarantees for use across the transcription pipeline.

ID Formats (per GitHub issue #136):
- run_id:     `run-YYYYMMDD-HHMMSS-XXXXXX` (6 random alphanumeric chars)
- stream_id:  `str-{uuid4}`
- event_id:   Monotonically increasing positive integer per stream
- segment_id: `seg-{seq}` where seq is a zero-indexed sequence number

These IDs are used in:
- Batch transcription receipts (run_id)
- Streaming sessions (stream_id, event_id, segment_id)
- Event envelopes for WebSocket protocol
- Provenance tracking for audit trails

Example:
    >>> from transcription.ids import generate_run_id, generate_stream_id
    >>> run_id = generate_run_id()
    >>> run_id
    'run-20260128-143052-x7k9p2'
    >>> stream_id = generate_stream_id()
    >>> stream_id
    'str-550e8400-e29b-41d4-a716-446655440000'
"""

from __future__ import annotations

import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

# =============================================================================
# ID Format Constants
# =============================================================================

RUN_ID_PREFIX = "run"
STREAM_ID_PREFIX = "str"
SEGMENT_ID_PREFIX = "seg"

# Characters used for random suffix in run_id
RUN_ID_RANDOM_CHARS = string.ascii_lowercase + string.digits
RUN_ID_RANDOM_LENGTH = 6


# =============================================================================
# ID Generation Functions
# =============================================================================


def generate_run_id(timestamp: datetime | None = None) -> str:
    """
    Generate a unique run ID for batch transcription runs.

    Format: `run-YYYYMMDD-HHMMSS-XXXXXX`
    - YYYYMMDD: Date in ISO format
    - HHMMSS: Time in 24-hour format
    - XXXXXX: 6 random alphanumeric characters for uniqueness

    Args:
        timestamp: Optional datetime to use for the timestamp portion.
                   Defaults to current UTC time.

    Returns:
        A unique run ID string.

    Example:
        >>> run_id = generate_run_id()
        >>> run_id
        'run-20260128-143052-x7k9p2'
    """
    ts = timestamp or datetime.now(UTC)
    date_part = ts.strftime("%Y%m%d")
    time_part = ts.strftime("%H%M%S")
    random_suffix = "".join(random.choices(RUN_ID_RANDOM_CHARS, k=RUN_ID_RANDOM_LENGTH))
    return f"{RUN_ID_PREFIX}-{date_part}-{time_part}-{random_suffix}"


def generate_stream_id() -> str:
    """
    Generate a unique stream ID for WebSocket streaming sessions.

    Format: `str-{uuid4}`

    The stream_id uniquely identifies a WebSocket connection and is immutable
    for the session duration. It is globally unique across all streams.

    Returns:
        A unique stream ID string.

    Example:
        >>> stream_id = generate_stream_id()
        >>> stream_id
        'str-550e8400-e29b-41d4-a716-446655440000'
    """
    return f"{STREAM_ID_PREFIX}-{uuid.uuid4()}"


def generate_segment_id(sequence: int) -> str:
    """
    Generate a segment ID from a sequence number.

    Format: `seg-{seq}`

    Segment IDs are scoped to a single stream and link PARTIAL events
    to their final FINALIZED event.

    Args:
        sequence: Zero-indexed sequence number for the segment.

    Returns:
        A segment ID string.

    Example:
        >>> generate_segment_id(0)
        'seg-0'
        >>> generate_segment_id(42)
        'seg-42'
    """
    return f"{SEGMENT_ID_PREFIX}-{sequence}"


# =============================================================================
# ID Validation Functions
# =============================================================================


def is_valid_run_id(run_id: str) -> bool:
    """
    Validate that a string matches the run_id format.

    Args:
        run_id: String to validate.

    Returns:
        True if the string matches `run-YYYYMMDD-HHMMSS-XXXXXX` format.

    Example:
        >>> is_valid_run_id('run-20260128-143052-x7k9p2')
        True
        >>> is_valid_run_id('invalid')
        False
    """
    if not run_id.startswith(f"{RUN_ID_PREFIX}-"):
        return False

    parts = run_id.split("-")
    if len(parts) != 4:
        return False

    prefix, date_part, time_part, random_part = parts

    if prefix != RUN_ID_PREFIX:
        return False

    # Validate date format (YYYYMMDD)
    if len(date_part) != 8 or not date_part.isdigit():
        return False

    # Validate time format (HHMMSS)
    if len(time_part) != 6 or not time_part.isdigit():
        return False

    # Validate random suffix
    if len(random_part) != RUN_ID_RANDOM_LENGTH:
        return False
    if not all(c in RUN_ID_RANDOM_CHARS for c in random_part):
        return False

    return True


def is_valid_stream_id(stream_id: str) -> bool:
    """
    Validate that a string matches the stream_id format.

    Args:
        stream_id: String to validate.

    Returns:
        True if the string matches `str-{uuid4}` format.

    Example:
        >>> is_valid_stream_id('str-550e8400-e29b-41d4-a716-446655440000')
        True
        >>> is_valid_stream_id('invalid')
        False
    """
    if not stream_id.startswith(f"{STREAM_ID_PREFIX}-"):
        return False

    uuid_part = stream_id[len(STREAM_ID_PREFIX) + 1 :]

    try:
        uuid.UUID(uuid_part)
        return True
    except ValueError:
        return False


def is_valid_segment_id(segment_id: str) -> bool:
    """
    Validate that a string matches the segment_id format.

    Args:
        segment_id: String to validate.

    Returns:
        True if the string matches `seg-{seq}` format where seq is a non-negative integer.

    Example:
        >>> is_valid_segment_id('seg-0')
        True
        >>> is_valid_segment_id('seg-42')
        True
        >>> is_valid_segment_id('seg-abc')
        False
    """
    if not segment_id.startswith(f"{SEGMENT_ID_PREFIX}-"):
        return False

    seq_part = segment_id[len(SEGMENT_ID_PREFIX) + 1 :]

    if not seq_part.isdigit():
        return False

    return int(seq_part) >= 0


def parse_segment_sequence(segment_id: str) -> int:
    """
    Extract the sequence number from a segment ID.

    Args:
        segment_id: A valid segment ID string.

    Returns:
        The sequence number as an integer.

    Raises:
        ValueError: If the segment_id is not valid.

    Example:
        >>> parse_segment_sequence('seg-42')
        42
    """
    if not is_valid_segment_id(segment_id):
        raise ValueError(f"Invalid segment_id format: {segment_id}")

    seq_part = segment_id[len(SEGMENT_ID_PREFIX) + 1 :]
    return int(seq_part)


# =============================================================================
# Event ID Counter
# =============================================================================


@dataclass
class EventIdCounter:
    """
    Monotonically increasing event ID counter for a single stream.

    The counter starts at 0 and increments by 1 for each call to next().
    Event IDs are positive integers starting at 1 (first call returns 1).

    This class is designed to be used within a single streaming session
    and should not be shared across streams.

    Attributes:
        _counter: Internal counter value (starts at 0, next() returns 1).

    Example:
        >>> counter = EventIdCounter()
        >>> counter.next()
        1
        >>> counter.next()
        2
        >>> counter.current
        2
    """

    _counter: int = field(default=0, init=False)

    def next(self) -> int:
        """
        Generate the next event ID.

        Returns:
            The next monotonically increasing positive integer.
        """
        self._counter += 1
        return self._counter

    @property
    def current(self) -> int:
        """
        Get the current event ID (last generated value).

        Returns:
            The most recently generated event ID, or 0 if none generated.
        """
        return self._counter

    def reset(self) -> None:
        """
        Reset the counter to 0.

        Note: This should only be used in testing. In production, event IDs
        should never reset within a stream session.
        """
        self._counter = 0


@dataclass
class SegmentIdCounter:
    """
    Sequential segment ID counter for a single stream.

    Generates segment IDs in the format `seg-{seq}` where seq starts at 0.

    Attributes:
        _counter: Internal counter value (starts at 0).

    Example:
        >>> counter = SegmentIdCounter()
        >>> counter.next()
        'seg-0'
        >>> counter.next()
        'seg-1'
        >>> counter.current
        'seg-1'
    """

    _counter: int = field(default=0, init=False)
    _last_id: str | None = field(default=None, init=False)

    def next(self) -> str:
        """
        Generate the next segment ID.

        Returns:
            A segment ID string in the format `seg-{seq}`.
        """
        segment_id = generate_segment_id(self._counter)
        self._last_id = segment_id
        self._counter += 1
        return segment_id

    @property
    def current(self) -> str | None:
        """
        Get the current segment ID (last generated value).

        Returns:
            The most recently generated segment ID, or None if none generated.
        """
        return self._last_id

    @property
    def next_sequence(self) -> int:
        """
        Get the next sequence number that will be used.

        Returns:
            The next sequence number (current counter value).
        """
        return self._counter

    def reset(self) -> None:
        """
        Reset the counter to 0.

        Note: This should only be used in testing. In production, segment IDs
        should not reset within a stream session.
        """
        self._counter = 0
        self._last_id = None


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "RUN_ID_PREFIX",
    "STREAM_ID_PREFIX",
    "SEGMENT_ID_PREFIX",
    # Generation functions
    "generate_run_id",
    "generate_stream_id",
    "generate_segment_id",
    # Validation functions
    "is_valid_run_id",
    "is_valid_stream_id",
    "is_valid_segment_id",
    "parse_segment_sequence",
    # Counter classes
    "EventIdCounter",
    "SegmentIdCounter",
]
