"""Stable public WebSocket route backed by the process-owned ASR runtime."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from . import service_streaming as _legacy_streaming
from .exceptions import (
    ASRInferenceError,
    RuntimeNotReadyError,
    StreamingInboundLimitError,
    StreamingNegotiationError,
    StreamingSessionLimitError,
    StreamingTransportCapacityError,
    StreamingTransportDeliveryError,
    TranscriptionError,
)
from .service_runtime import ASRRuntime
from .streaming_revision_controller import (
    RevisionStreamingController,
    SpeechClassifier,
)
from .streaming_transport import (
    DEFAULT_STREAMING_TRANSPORT_LIMITS,
    OutboundEventController,
    StreamingAdmissionController,
    StreamingTransportLimits,
    decode_audio_chunk_bounded,
    disable_legacy_replay,
    session_deadline,
)
from .streaming_ws import (
    ClientMessageType,
    EventEnvelope,
    ServerMessageType,
    SessionState,
    WebSocketSessionConfig,
    WebSocketStreamingSession,
    parse_client_message,
)

logger = logging.getLogger(__name__)
router = APIRouter()

SpeechClassifierFactory = Callable[[], SpeechClassifier]


async def _send(
    transport: OutboundEventController,
    event: EventEnvelope,
    *,
    replay_eligible: bool = True,
) -> None:
    await transport.publish(event, replay_eligible=replay_eligible)


async def _send_terminal(
    transport: OutboundEventController,
    event: EventEnvelope,
) -> None:
    await transport.publish_terminal(event)
    await transport.drain()


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


def _limits(websocket: WebSocket) -> StreamingTransportLimits:
    configured = getattr(
        _application_state(websocket),
        "streaming_transport_limits",
        None,
    )
    if configured is None:
        return DEFAULT_STREAMING_TRANSPORT_LIMITS
    if not isinstance(configured, StreamingTransportLimits):
        raise TypeError("streaming_transport_limits must be StreamingTransportLimits")
    return configured


def _admission(
    websocket: WebSocket,
    limits: StreamingTransportLimits,
) -> StreamingAdmissionController:
    state = _application_state(websocket)
    configured = getattr(state, "streaming_admission_controller", None)
    if configured is None:
        configured = StreamingAdmissionController(limits.max_active_sessions)
        setattr(state, "streaming_admission_controller", configured)
    if not isinstance(configured, StreamingAdmissionController):
        raise TypeError(
            "streaming_admission_controller must be StreamingAdmissionController"
        )
    return configured


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
    elif isinstance(error, StreamingInboundLimitError):
        message = "Streaming audio message exceeds the server limit"
    elif isinstance(error, StreamingSessionLimitError):
        message = "Streaming session limit reached"
    elif isinstance(error, StreamingTransportCapacityError):
        message = "Streaming client is not consuming events fast enough"
    elif isinstance(error, StreamingTransportDeliveryError):
        message = "Streaming delivery failed"
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


def _resume_gap_event(
    session: WebSocketStreamingSession,
    *,
    last_event_id: int,
) -> EventEnvelope:
    session.state = SessionState.ERROR
    session.stats.errors += 1
    return session._create_envelope(
        ServerMessageType.ERROR,
        {
            "code": "RESUME_GAP",
            "message": "Delivered replay history cannot satisfy the requested cursor",
            "recoverable": False,
            "context": {"requested_event_id": last_event_id},
        },
    )


def _client_timestamp(payload: dict[str, Any]) -> int:
    timestamp = payload.get("timestamp", 0)
    try:
        return int(timestamp)
    except (TypeError, ValueError):
        return 0


def _client_owned_limit_mismatches(config_data: dict[str, Any]) -> dict[str, str]:
    return {
        name: "server_owned"
        for name in ("replay_buffer_size", "backpressure_threshold")
        if name in config_data
    }


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """Run stable revision-aware ASR on one bounded WebSocket connection."""

    await websocket.accept()
    limits = _limits(websocket)
    transport = OutboundEventController(websocket.send_json, limits=limits)
    await transport.start()
    admission = _admission(websocket, limits)

    connection_controls = WebSocketStreamingSession()
    disable_legacy_replay(connection_controls)
    session: WebSocketStreamingSession | None = None
    controller: RevisionStreamingController | None = None
    admitted_stream_id: str | None = None
    started_at: float | None = None

    try:
        while True:
            try:
                await transport.wait_below_pressure()
                if started_at is None:
                    message = await websocket.receive_json()
                else:
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=session_deadline(started_at, limits=limits),
                    )
            except WebSocketDisconnect:
                raise
            except (TimeoutError, StreamingSessionLimitError):
                if session is None:
                    raise
                duration_error = StreamingSessionLimitError(
                    "Streaming session duration limit reached",
                    context={
                        "max_session_duration_sec": limits.max_session_duration_sec,
                    },
                )
                await _send_terminal(
                    transport,
                    (
                        controller.terminal_error_event(duration_error)
                        if controller is not None
                        else _terminal_event(session, duration_error)
                    ),
                )
                await websocket.close(code=1008)
                return
            except StreamingTransportDeliveryError:
                raise
            except Exception as error:  # noqa: BLE001 - return bounded protocol detail
                logger.warning("Failed to decode WebSocket message", exc_info=error)
                await _send(
                    transport,
                    _recoverable_event(
                        session or connection_controls,
                        code="invalid_message",
                        message="WebSocket message is invalid",
                    ),
                    replay_eligible=session is not None,
                )
                continue

            try:
                message_type, payload = parse_client_message(message)
            except (TypeError, ValueError) as error:
                logger.warning("Invalid WebSocket message type", exc_info=error)
                await _send(
                    transport,
                    _recoverable_event(
                        session or connection_controls,
                        code="invalid_message_type",
                        message="WebSocket message type is invalid",
                    ),
                    replay_eligible=session is not None,
                )
                continue

            if message_type is ClientMessageType.PING:
                await _send(
                    transport,
                    (session or connection_controls).create_pong_event(
                        _client_timestamp(payload)
                    ),
                    replay_eligible=False,
                )
                continue

            if message_type is ClientMessageType.START_SESSION:
                if session is not None:
                    await _send(
                        transport,
                        _recoverable_event(
                            session,
                            code="session_already_started",
                            message="Session already started",
                        ),
                    )
                    continue

                config_data = payload.get("config", {})
                if not isinstance(config_data, dict):
                    session = WebSocketStreamingSession()
                    disable_legacy_replay(session)
                    negotiation_error = StreamingNegotiationError(
                        "Streaming audio configuration is unsupported",
                        context={"mismatches": {"config": "must_be_object"}},
                    )
                    await _send_terminal(
                        transport,
                        _terminal_event(session, negotiation_error),
                    )
                    await websocket.close(code=1003)
                    return

                mismatches = _client_owned_limit_mismatches(config_data)
                if mismatches:
                    session = WebSocketStreamingSession()
                    disable_legacy_replay(session)
                    negotiation_error = StreamingNegotiationError(
                        "Streaming transport limits are server owned",
                        context={"mismatches": mismatches},
                    )
                    await _send_terminal(
                        transport,
                        _terminal_event(session, negotiation_error),
                    )
                    await websocket.close(code=1003)
                    return

                try:
                    config = WebSocketSessionConfig.from_dict(config_data)
                except (TypeError, ValueError) as error:
                    logger.warning("Invalid streaming configuration", exc_info=error)
                    session = WebSocketStreamingSession()
                    disable_legacy_replay(session)
                    negotiation_error = StreamingNegotiationError(
                        "Streaming audio configuration is unsupported",
                        context={"mismatches": {"config": "invalid"}},
                    )
                    await _send_terminal(
                        transport,
                        _terminal_event(session, negotiation_error),
                    )
                    await websocket.close(code=1003)
                    return

                session = WebSocketStreamingSession(config=config)
                disable_legacy_replay(session)
                transport.bind_stream(session.stream_id)

                runtime = _runtime(websocket)
                if runtime is None or not runtime.ready:
                    state = runtime.state.value if runtime is not None else "missing"
                    readiness_error = RuntimeNotReadyError(
                        "The process-owned ASR runtime is not ready",
                        context={"state": state},
                    )
                    await _send_terminal(
                        transport,
                        _terminal_event(session, readiness_error),
                    )
                    await websocket.close(code=1013)
                    return

                try:
                    await admission.acquire(session.stream_id)
                    admitted_stream_id = session.stream_id
                    controller = RevisionStreamingController(
                        session,
                        runtime,
                        classifier=_classifier(websocket),
                    )
                    started_at = time.monotonic()
                    await _send(transport, await controller.start(config_data))
                except TranscriptionError as error:
                    await _send_terminal(
                        transport,
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
                await _send(
                    transport,
                    _recoverable_event(
                        connection_controls,
                        code="no_session",
                        message="Send START_SESSION before this message",
                    ),
                    replay_eligible=False,
                )
                continue

            if message_type is ClientMessageType.AUDIO_CHUNK:
                try:
                    audio_bytes, sequence = decode_audio_chunk_bounded(
                        payload,
                        max_decoded_bytes=limits.max_decoded_audio_message_bytes,
                    )
                    events = await controller.process_audio_chunk(
                        audio_bytes,
                        sequence,
                    )
                except StreamingInboundLimitError as error:
                    logger.warning(
                        "Streaming audio chunk exceeded the server limit",
                        exc_info=error,
                    )
                    await _send_terminal(
                        transport,
                        controller.terminal_error_event(error),
                    )
                    await websocket.close(code=1009)
                    return
                except (TypeError, ValueError) as error:
                    logger.warning("Invalid streaming audio chunk", exc_info=error)
                    await _send(
                        transport,
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
                    await _send_terminal(
                        transport,
                        controller.terminal_error_event(error),
                    )
                    await websocket.close(code=1011)
                    return

                await transport.publish_many(events)
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
                    await _send_terminal(
                        transport,
                        controller.terminal_error_event(error),
                    )
                    await websocket.close(code=1011)
                    return
                await transport.publish_many(events)
                await transport.drain()
                await transport.close(drain=False)
                await websocket.close(code=1000)
                return

            if message_type is ClientMessageType.RESUME_SESSION:
                requested_session_id = payload.get("session_id")
                last_event_id = payload.get("last_event_id", 0)
                if requested_session_id != session.stream_id:
                    await _send(
                        transport,
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
                        transport,
                        _recoverable_event(
                            session,
                            code="invalid_resume_cursor",
                            message="Resume cursor is invalid",
                        ),
                    )
                    continue
                session.stats.resume_attempts += 1
                replay, gap_detected = transport.replay_since(cursor)
                if gap_detected:
                    await _send_terminal(
                        transport,
                        _resume_gap_event(
                            session,
                            last_event_id=cursor,
                        ),
                    )
                    await websocket.close(code=1008)
                    return
                await transport.publish_replay(replay)
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
    except StreamingTransportDeliveryError:
        logger.exception("Stable WebSocket delivery authority failed")
        if controller is not None:
            controller.abort()
        try:
            await websocket.close(code=1011)
        except Exception:
            logger.debug("Failed to close failed WebSocket", exc_info=True)
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
                await _send_terminal(transport, event)
            except Exception:
                logger.debug("Failed to send terminal WebSocket error", exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            logger.debug("Failed to close failed WebSocket", exc_info=True)
    finally:
        if admitted_stream_id is not None:
            await admission.release(admitted_stream_id)
        await transport.abort()
        logger.info(
            "Stable WebSocket connection closed: transport=%s",
            transport.metrics_receipt(),
        )


@router.get(
    "/stream/config",
    summary="Get stable streaming configuration",
    tags=["Streaming"],
)
async def get_stream_config() -> JSONResponse:
    """Return the presently earned public streaming contract."""

    limits = DEFAULT_STREAMING_TRANSPORT_LIMITS
    return JSONResponse(
        status_code=200,
        content={
            "default_config": {
                "max_gap_sec": 1.0,
                "sample_rate": 16_000,
                "channels": 1,
                "audio_format": "pcm_s16le",
            },
            "server_limits": {
                "max_decoded_audio_message_bytes": (
                    limits.max_decoded_audio_message_bytes
                ),
                "max_session_duration_sec": limits.max_session_duration_sec,
                "max_active_sessions": limits.max_active_sessions,
                "max_outbound_queue_events": limits.max_queue_events,
                "max_outbound_queue_bytes": limits.max_queue_bytes,
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
                    "RESUME_SESSION",
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
