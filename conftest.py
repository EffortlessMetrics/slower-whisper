"""Repository-level pytest ownership boundaries."""

from __future__ import annotations

import pytest
from pytest import Item

_RETIRED_STREAMING_ROUTE_ASSERTIONS = {
    (
        "tests/test_streaming_rest.py::TestStreamConfigEndpoint::"
        "test_get_default_config"
    ): (
        "The earned public config contract is covered by "
        "tests/test_websocket_streaming.py::TestStreamConfigEndpoint::"
        "test_get_stream_config_reports_only_earned_surface."
    ),
    (
        "tests/test_streaming_ws.py::TestWebSocketEndpoint::"
        "test_websocket_full_session"
    ): (
        "The production route requires a process-owned runtime and is covered by "
        "tests/test_websocket_streaming.py::TestAudioChunk::"
        "test_audio_produces_real_replacement_event plus TestEndSession."
    ),
    (
        "tests/test_streaming_ws.py::TestWebSocketEndpoint::"
        "test_websocket_invalid_message_type"
    ): (
        "Invalid-message recovery on the production route is covered by "
        "tests/test_websocket_streaming.py::TestErrorHandling::"
        "test_invalid_message_is_recoverable."
    ),
}


def pytest_collection_modifyitems(items: list[Item]) -> None:
    """Retire exact duplicate assertions from the pre-runtime public route."""
    for item in items:
        reason = _RETIRED_STREAMING_ROUTE_ASSERTIONS.get(item.nodeid)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
