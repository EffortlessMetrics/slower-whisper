"""Source ownership policy for the stable public `/stream` ASR path."""

from __future__ import annotations

import ast
from pathlib import Path

import transcription


def module_path(name: str) -> Path:
    return Path(transcription.__file__).resolve().parent / name


def function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    extracted = ast.get_source_segment(source, matches[0])
    assert extracted is not None
    return extracted


def test_public_route_constructs_revision_controller_not_legacy_adapter() -> None:
    path = module_path("service_streaming.py")
    route = function_source(path, "websocket_stream")

    assert "RevisionStreamingController" in route
    assert "controller.handle_start_session" in route
    assert "controller.handle_end_session" in route
    assert "StreamingASRAdapter" not in route
    assert "[processing...]" not in route
    assert "[final segment]" not in route


def test_public_route_resolves_the_process_owned_runtime() -> None:
    route = function_source(module_path("service_streaming.py"), "websocket_stream")

    assert 'websocket.scope.get("app")' in route
    assert '"asr_runtime"' in route
    assert "runtime.ready" in route
    assert "RuntimeNotReadyError" in route


def test_public_route_does_not_return_raw_exception_text() -> None:
    route = function_source(module_path("service_streaming.py"), "websocket_stream")

    assert '"message": str(' not in route
    assert "Internal server error:" not in route
    assert "Unexpected error:" not in route
    assert '"message": "Unexpected streaming failure"' in route


def test_revision_controller_is_the_public_asr_event_projection_owner() -> None:
    source = module_path("streaming_revision_controller.py").read_text(encoding="utf-8")

    assert "ASRRevision" in source
    assert "segment_id" in source
    assert "revision" in source
    assert "start_sample" in source
    assert "end_sample" in source
    assert "final_reason" in source
    assert 'frozenset({"[processing...]", "[final segment]"})' in source
