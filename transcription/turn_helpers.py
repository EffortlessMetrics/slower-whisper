"""Helper utilities for working with Turn objects.

This module provides shared conversion helpers used across
turns_enrich.py and speaker_stats.py.
"""

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, cast

logger = logging.getLogger(__name__)


def turn_to_dict(t: Any, *, copy: bool = False) -> dict[str, Any]:
    """Convert a Turn (dataclass or dict) to a plain dict.

    Args:
        t: Turn instance or dict
        copy: If True, return shallow copy of dicts (default False)

    Handles:
    - dict: returned as-is or shallow copy based on `copy` param
    - object with to_dict(): calls to_dict()
    - dataclass: converted via asdict()

    Raises:
        TypeError: for unsupported types

    Examples:
        >>> turn_to_dict({"id": 1, "text": "hello"})
        {'id': 1, 'text': 'hello'}
        >>> turn_to_dict({"id": 1, "text": "hello"}, copy=True)
        {'id': 1, 'text': 'hello'}
        >>> turn_to_dict(Turn(id=1, text="hello"))  # dataclass
        {'id': 1, 'text': 'hello'}
    """
    if isinstance(t, dict):
        return dict(t) if copy else t
    to_dict_fn = getattr(t, "to_dict", None)
    if callable(to_dict_fn):
        result: dict[str, Any] = to_dict_fn()
        return result
    if is_dataclass(t):
        return asdict(cast(Any, t))
    raise TypeError(f"Unsupported turn type: {type(t)}")
