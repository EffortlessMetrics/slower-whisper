"""Stable public WebSocket route backed by the process-owned ASR runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from . import service_streaming as _legacy_streaming
from .exceptions import (
    ASRInferenceError,
    RuntimeNotReadyError,
    StreamingNegotiationError,
    TranscriptionError,
)
from .service_runtime import ASRRuntime
from .streaming_revision_controller import (
    RevisionStreamingController,
    SpeechClassifier,
)
from .streaming_ws import (
    ClientMessageType,
    EventEnvelope,
    ServerMessageType,
    SessionState,
    WebSocketSessionConfig,
    WebSocketStreamingSession,
    decode_audio_chunk,
    parse_client_message,
)

logger = logging.getLogger(__name__)
router = APIRouter()

SpeechClassifierFactory = Callable[[], SpeechClassifier]


async def _send(websocket: WebSocket, event: EventEnvelope) -> None:
    await websocket.send_json(event.to_dict())


def _application_state(websocket: WebSocket) -> Any:
    application = websocket.scope.get("app")
    return getattr(application, "state", None)


def _runtime(websocket: WebSocket) -> ASRRuntime | None:
    runtime = getattr(_application_state(websocket), "asr_runtime", None)
    return runtime if isinstance(runtime, ASRRuntime) else None


def _classifier(websocket: WebSocket) -> SpeechClassifier | None:
    factory = getattr(
        _application_state(websocket),
        "streaming_speech_classifier_factory",
        None,
    )
    if factory is None:
        return None
    if not callable(factory):
        raise TypeError("streaming_speech_classifier_factory must be callable")
    return cast(SpeechClassifierFactory, factory)()


def _terminal_event(
    session: WebSocketStreamingSession,
    error: TranscriptionError,
) -> EventEnvelope:
    session.state = SessionState.ERROR
    session.stats.errors += 1
    session.stats.events_sent += 1
    if isinstance(error, StreamingNegotiationError):
        message = "Unsupported streaming audio configuration"
    elif isinstance(error, RuntimeNotReadyError):
        message = "Streaming ASR runtime is not ready"
    else:
        message = "Streaming ASR failed"
    return session._create_envelope(
        ServerMessageType.ERROR,
        {
            "code": error.reason_code,
            "message": message,
            "recoverable": False,
            "context": dict(error.context),
        },
    )


def _recoverable_event(
    session: WebSocketStreamingSession,
    *,
    code: str,
    message: str,
) -> EventEnvelope:
    return session.create_error_event(
        code=code,
        message=message,
        recoverable=True,
    )


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """Run stable revision-aware ASR on one accepted WebSocket connection."""
    await websocket.accept()
    session: WebSocketStreamingSession | None = None
    controller: RevisionStreamingController | None = None

    try:
        while True:
            try:
                message = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception as error:  # noqa: BLE001 - return bounded protocol detail
                logger.warning("Failed to decode WebSocket message", exc_info=error)
                target = session or WebSocketStreamingSession()
                await _send(
                    websocket,
                    _recoverable_event(
                        target,
                        code="invalid_message",
                        message="WebSocket message is invalid",
                    ),
                )
                continue

            try:
                message_type, payload = parse_client_message(message)
            except (TypeError, ValueError) as error:
                logger.warning("Invalid WebSocket message type", exc_info=error)
                target = session or WebSocketStreamingSession()
                await _send(
                    websocket,
                    _recoverable_event(
                        target,
                        code="invalid_message_type",
                        message="WebSocket message type is invalid",
                    ),
                )
                continue

            if message_type is ClientMessageType.START_SESSION:
                if session is not None:
                    await _send(
                        websocket,
                        _recoverable_event(
                            session,
                            code="session_already_started",
                            message="Session already started",
                        ),
                    )
                    continue

                config_data = payload.get("config", {})
                if not isinstance(config_data, dict):
                    config_data = {"config": config_data}
                try:
                    config = WebSocketSessionConfig.from_dict(config_data)
                except (TypeError, ValueError) as error:
                    logger.warning("Invalid streaming configuration", exc_info=error)
                    session = WebSocketStreamingSession()
                    negotiation_error = StreamingNegotiationError(
                        "Streaming audio configuration is unsupported",
                        context={"mismatches": {"config": "invalid"}},
                    )
                    await _send(websocket, _terminal_event(session, negotiation_error))
                    await websocket.close(code=1003)
                    return

                session = WebSocketStreamingSession(config=config)
                runtime = _runtime(websocket)
                if runtime is None or not runtime.ready:
                    state = runtime.state.value if runtime is not None else "missing"
                    readiness_error = RuntimeNotReadyError(
                        "The process-owned ASR runtime is not ready",
                        context={"state": state},
                    )
                    await _send(websocket, _terminal_event(session, readiness_error))
                    await websocket.close(code=1013)
                    return

                try:
                    controller = RevisionStreamingController(
                        session,
                        runtime,
                        classifier=_classifier(websocket),
                    )
                    await _send(websocket, await controller.start(config_data))
                except TranscriptionError as error:
                    await _send(
                        websocket,
                        (
                            controller.terminal_error_event(error)
                            if controller is not None
                            else _terminal_event(session, error)
                        ),
                    )
                    await websocket.close(code=1003)
                    return
                continue

            if session is None or controller is None:
                temporary = WebSocketStreamingSession()
                await _send(
                    websocket,
                    _recoverable_event(
                        temporary,
                        code="no_session",
                        message="Send START_SESSION before this message",
                    ),
                )
                continue

            if message_type is ClientMessageType.AUDIO_CHUNK:
                try:
                    audio_bytes, sequence = decode_audio_chunk(payload)
                    events = await controller.process_audio_chunk(
                        audio_bytes,
                        sequence,
                    )
                except (TypeError, ValueError) as error:
                    logger.warning("Invalid streaming audio chunk", exc_info=error)
                    await _send(
                        websocket,
                        _recoverable_event(
                            session,
                            code="invalid_audio_chunk",
                            message="Audio chunk is invalid",
                        ),
                    )
                    continue
                except TranscriptionError as error:
                    logger.error(
                        "Streaming ASR failed [%s]",
                        error.reason_code,
                        exc_info=error,
                    )
                    await _send(websocket, controller.terminal_error_event(error))
                    await websocket.close(code=1011)
                    return

                for event in events:
                    await _send(websocket, event)
                continue

            if message_type is ClientMessageType.END_SESSION:
                try:
                    events = await controller.end()
                except TranscriptionError as error:
                    logger.error(
                        "Streaming ASR finalization failed [%s]",
                        error.reason_code,
                        exc_info=error,
                    )
                    await _send(websocket, controller.terminal_error_event(error))
                    await websocket.close(code=1011)
                    return
                for event in events:
                    await _send(websocket, event)
                await websocket.close(code=1000)
                return

            if message_type is ClientMessageType.RESUME_SESSION:
                requested_session_id = payload.get("session_id")
                last_event_id = payload.get("last_event_id", 0)
                if requested_session_id != session.stream_id:
                    await _send(
                        websocket,
                        _recoverable_event(
                            session,
                            code="session_mismatch",
                            message="Requested session does not match this connection",
                        ),
                    )
                    continue
                try:
                    cursor = int(last_event_id)
                except (TypeError, ValueError):
                    await _send(
                        websocket,
                        _recoverable_event(
                            session,
                            code="invalid_resume_cursor",
                            message="Resume cursor is invalid",
                        ),
                    )
                    continue
                replay, gap_detected = session.get_events_for_resume(cursor)
                if gap_detected:
                    await _send(websocket, session.create_resume_gap_error(cursor))
                    await websocket.close(code=1008)
                    return
                for event in replay:
                    await _send(websocket, event)
                continue

            if message_type is ClientMessageType.PING:
                timestamp = payload.get("timestamp", 0)
                try:
                    client_timestamp = int(timestamp)
                except (TypeError, ValueError):
                    client_timestamp = 0
                await _send(
                    websocket,
                    session.create_pong_event(client_timestamp),
                )
                continue

            if message_type is ClientMessageType.TTS_STATE:
                session.set_tts_state(payload.get("playing") is True)
                continue

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected: stream_id=%s",
            session.stream_id if session is not None else "no_session",
        )
        if controller is not None:
            controller.abort()
    except Exception as error:  # noqa: BLE001 - log locally, sanitize remotely
        logger.exception("Unexpected error in stable WebSocket handler")
        if session is not None:
            typed = ASRInferenceError(
                "Unexpected streaming failure",
                context={
                    "phase": "route",
                    "violation": "unexpected_streaming_error",
                },
            )
            typed.__cause__ = error
            try:
                event = (
                    controller.terminal_error_event(typed)
                    if controller is not None
                    else _terminal_event(session, typed)
                )
                await _send(websocket, event)
            except Exception:
                logger.debug("Failed to send terminal WebSocket error", exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            logger.debug("Failed to close failed WebSocket", exc_info=True)
    finally:
        logger.info("Stable WebSocket connection closed")


@router.get(
    "/stream/config",
    summary="Get stable streaming configuration",
    tags=["Streaming"],
)
async def get_stream_config() -> JSONResponse:
    """Return the presently earned public streaming contract."""
    return JSONResponse(
        status_code=200,
        content={
            "default_config": {
                "max_gap_sec": 1.0,
                "sample_rate": 16_000,
                "channels": 1,
                "audio_format": "pcm_s16le",
            },
            "supported_audio_formats": ["pcm_s16le"],
            "supported_sample_rates": [16_000],
            "supported_channels": [1],
            "optional_live_enrichment": False,
            "message_types": {
                "client": [
                    "START_SESSION",
                    "AUDIO_CHUNK",
                    "END_SESSION",
                    "PING",
                    "TTS_STATE",
                ],
                "server": [
                    "SESSION_STARTED",
                    "PARTIAL",
                    "FINALIZED",
                    "ERROR",
                    "SESSION_ENDED",
                    "PONG",
                ],
            },
        },
    )


for _route in _legacy_streaming.router.routes:
    if getattr(_route, "path", None) not in {"/stream", "/stream/config"}:
        router.routes.append(_route)
