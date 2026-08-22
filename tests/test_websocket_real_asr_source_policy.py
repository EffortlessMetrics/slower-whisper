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
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    extracted = ast.get_source_segment(source, matches[0])
    assert extracted is not None
    return extracted


def test_public_route_constructs_controller_and_never_invokes_legacy_asr() -> None:
    route = function_source(module_path("service_runtime_streaming.py"), "websocket_stream")

    assert "RevisionStreamingController" in route
    assert "controller.process_audio_chunk" in route
    assert "controller.end" in route
    assert "StreamingASRAdapter" not in route
    assert "session.process_audio_chunk" not in route
    assert "session.end()" not in route
    assert "[processing...]" not in route
    assert "[final segment]" not in route


def test_public_route_resolves_the_process_owned_runtime() -> None:
    route = function_source(module_path("service_runtime_streaming.py"), "websocket_stream")

    assert "_runtime(websocket)" in route
    assert "runtime.ready" in route
    assert "RuntimeNotReadyError" in route


def test_public_route_uses_stable_generic_error_messages() -> None:
    source = module_path("service_runtime_streaming.py").read_text(encoding="utf-8")

    assert '"message": str(' not in source
    assert "Internal server error:" not in source
    assert "Unexpected error:" not in source
    assert '"Unexpected streaming failure"' in source
    assert '"Streaming ASR failed"' in source


def test_controller_uses_the_session_envelope_authority_without_monkeypatching() -> None:
    source = module_path("streaming_revision_controller.py").read_text(encoding="utf-8")

    assert "ASRRevision" in source
    assert "session._create_envelope" in source
    assert "ServerMessageType.PARTIAL" in source
    assert "ServerMessageType.FINALIZED" in source
    assert "segment_id" in source
    assert "revision" in source
    assert "start_sample" in source
    assert "end_sample" in source
    assert "final_reason" in source
    assert "MethodType" not in source
    assert "session.send_event" not in source
    assert "[processing...]" not in source
    assert "[final segment]" not in source


def test_service_mounts_the_runtime_streaming_router() -> None:
    source = module_path("service.py").read_text(encoding="utf-8")

    assert "from .service_runtime_streaming import router as streaming_router" in source
    assert "from .service_streaming import router as streaming_router" not in source
