"""WebSocket and REST streaming session endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# WebSocket Streaming Endpoint (v2.0.0)
# =============================================================================


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time audio streaming and transcription.

    This endpoint enables bidirectional communication for streaming audio
    transcription. Clients send audio chunks and receive transcription
    events in real-time.

    Protocol:
        1. Client connects to /stream
        2. Client sends START_SESSION with configuration
        3. Server responds with SESSION_STARTED
        4. Client sends AUDIO_CHUNK messages with base64-encoded audio
        5. Server sends PARTIAL and FINALIZED segment events
        6. Client sends END_SESSION to finalize
        7. Server sends SESSION_ENDED with statistics
        8. Connection closes

    Client Message Types:
        - START_SESSION: {type, config: {max_gap_sec, enable_prosody, ...}}
        - AUDIO_CHUNK: {type, data: base64_string, sequence: int}
        - END_SESSION: {type}
        - PING: {type, timestamp: int}

    Server Message Types:
        - SESSION_STARTED: {event_id, stream_id, type, ts_server, payload: {session_id}}
        - PARTIAL: {event_id, stream_id, segment_id, type, ts_server, ts_audio_*, payload: {segment}}
        - FINALIZED: {event_id, stream_id, segment_id, type, ts_server, ts_audio_*, payload: {segment}}
        - ERROR: {event_id, stream_id, type, ts_server, payload: {code, message, recoverable}}
        - SESSION_ENDED: {event_id, stream_id, type, ts_server, payload: {stats}}
        - PONG: {event_id, stream_id, type, ts_server, payload: {timestamp, server_timestamp}}

    Example (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/stream');
        ws.onopen = () => {
            ws.send(JSON.stringify({
                type: 'START_SESSION',
                config: {max_gap_sec: 1.0, enable_prosody: true}
            }));
        };
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'FINALIZED') {
                console.log('Segment:', msg.payload.segment.text);
            }
        };
    """
    from .streaming_ws import (
        ClientMessageType,
        WebSocketSessionConfig,
        WebSocketStreamingSession,
        decode_audio_chunk,
        parse_client_message,
    )

    await websocket.accept()
    logger.info("WebSocket connection accepted")

    session: WebSocketStreamingSession | None = None

    try:
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                # Re-raise to be handled by outer exception handler
                raise
            except Exception as e:
                logger.warning("Failed to receive/parse WebSocket message: %s", e)
                if session:
                    error_event = session.create_error_event(
                        code="invalid_message",
                        message=f"Failed to parse message: {e}",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                continue

            # Parse message type and payload
            try:
                msg_type, payload = parse_client_message(data)
            except ValueError as e:
                logger.warning("Invalid client message: %s", e)
                if session:
                    error_event = session.create_error_event(
                        code="invalid_message_type",
                        message=str(e),
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                continue

            # Handle message based on type
            if msg_type == ClientMessageType.START_SESSION:
                if session is not None:
                    error_event = session.create_error_event(
                        code="session_already_started",
                        message="Session already started",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                    continue

                # Parse configuration from payload
                config_data = payload.get("config", {})
                config = WebSocketSessionConfig.from_dict(config_data)

                # Create and start session
                session = WebSocketStreamingSession(config=config)
                try:
                    start_event = await session.start()
                    await websocket.send_json(start_event.to_dict())
                    logger.info(
                        "WebSocket session started: stream_id=%s",
                        session.stream_id,
                    )
                except Exception as e:
                    logger.error("Failed to start session: %s", e, exc_info=True)
                    error_event = session.create_error_event(
                        code="session_start_failed",
                        message=f"Failed to start session: {e}",
                        recoverable=False,
                    )
                    await websocket.send_json(error_event.to_dict())
                    session = None

            elif msg_type == ClientMessageType.AUDIO_CHUNK:
                if session is None:
                    # Create temporary session just to send error
                    temp_session = WebSocketStreamingSession()
                    error_event = temp_session.create_error_event(
                        code="no_session",
                        message="No active session. Send START_SESSION first.",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                    continue

                try:
                    audio_bytes, sequence = decode_audio_chunk(payload)

                    # Check backpressure before processing
                    if session.check_backpressure():
                        # Drop partial events when under backpressure
                        dropped = session.drop_partial_events()
                        if dropped > 0:
                            # Emit buffer overflow error to inform client
                            overflow_event = session.create_buffer_overflow_error()
                            await websocket.send_json(overflow_event.to_dict())

                    events = await session.process_audio_chunk(audio_bytes, sequence)
                    for event in events:
                        await websocket.send_json(event.to_dict())
                except ValueError as e:
                    logger.warning("Invalid audio chunk: %s", e)
                    error_event = session.create_error_event(
                        code="invalid_audio_chunk",
                        message=str(e),
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                except Exception as e:
                    logger.error("Error processing audio chunk: %s", e, exc_info=True)
                    error_event = session.create_error_event(
                        code="processing_error",
                        message=f"Error processing audio: {e}",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())

            elif msg_type == ClientMessageType.END_SESSION:
                if session is None:
                    temp_session = WebSocketStreamingSession()
                    error_event = temp_session.create_error_event(
                        code="no_session",
                        message="No active session to end.",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                    continue

                try:
                    end_events = await session.end()
                    for event in end_events:
                        await websocket.send_json(event.to_dict())
                    logger.info(
                        "WebSocket session ended: stream_id=%s",
                        session.stream_id,
                    )
                except Exception as e:
                    logger.error("Error ending session: %s", e, exc_info=True)
                    error_event = session.create_error_event(
                        code="end_session_error",
                        message=f"Error ending session: {e}",
                        recoverable=False,
                    )
                    await websocket.send_json(error_event.to_dict())

                # Close connection after END_SESSION
                break

            elif msg_type == ClientMessageType.RESUME_SESSION:
                # Handle session resume after reconnection
                requested_session_id = payload.get("session_id")
                last_event_id = payload.get("last_event_id", 0)

                if session is None:
                    temp_session = WebSocketStreamingSession()
                    error_event = temp_session.create_error_event(
                        code="no_session",
                        message="No active session to resume. Send START_SESSION first.",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                    continue

                # Validate session ID matches current session
                if requested_session_id != session.stream_id:
                    error_event = session.create_error_event(
                        code="session_mismatch",
                        message=(
                            f"Session ID mismatch: requested '{requested_session_id}' "
                            f"but current session is '{session.stream_id}'"
                        ),
                        recoverable=False,
                    )
                    await websocket.send_json(error_event.to_dict())
                    await websocket.close()
                    break

                # Get events for resume
                events_to_replay, gap_detected = session.get_events_for_resume(last_event_id)

                if gap_detected:
                    # Cannot resume - gap in event buffer
                    gap_error = session.create_resume_gap_error(last_event_id)
                    await websocket.send_json(gap_error.to_dict())
                    await websocket.close()
                    break

                # Replay missed events
                logger.info(
                    "Replaying %d events for resume: stream_id=%s, from_event_id=%d",
                    len(events_to_replay),
                    session.stream_id,
                    last_event_id,
                )
                for event in events_to_replay:
                    await websocket.send_json(event.to_dict())

                # Continue normal operation after replay

            elif msg_type == ClientMessageType.PING:
                if session is None:
                    temp_session = WebSocketStreamingSession()
                    pong_event = temp_session.create_pong_event(payload.get("timestamp", 0))
                    await websocket.send_json(pong_event.to_dict())
                else:
                    pong_event = session.create_pong_event(payload.get("timestamp", 0))
                    await websocket.send_json(pong_event.to_dict())

            elif msg_type == ClientMessageType.TTS_STATE:
                if session is None:
                    temp_session = WebSocketStreamingSession()
                    error_event = temp_session.create_error_event(
                        code="no_session",
                        message="No active session. Send START_SESSION first.",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                    continue

                # Update TTS playback state (used for barge-in detection)
                playing = payload.get("playing", False)
                session.set_tts_state(playing)
                logger.debug(
                    "TTS state updated: stream_id=%s, playing=%s",
                    session.stream_id,
                    playing,
                )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected: stream_id=%s",
            session.stream_id if session else "no_session",
        )
        # Clean up session if it was active
        if session and session.state.value == "active":
            try:
                await session.end()
            except Exception as e:
                logger.warning("Error cleaning up session on disconnect: %s", e)

    except Exception:
        logger.exception("Unexpected error in WebSocket handler")
        if session:
            try:
                error_event = session.create_error_event(
                    code="internal_error",
                    message="An unexpected error occurred",
                    recoverable=False,
                )
                await websocket.send_json(error_event.to_dict())
            except Exception:
                pass  # Connection may already be closed

    finally:
        logger.info("WebSocket connection closed")


# =============================================================================
# WebSocket Session Management Endpoints (v2.0.0)
# =============================================================================


@router.get(
    "/stream/config",
    summary="Get default streaming configuration",
    description="Returns the default configuration for WebSocket streaming sessions.",
    tags=["Streaming"],
)
async def get_stream_config() -> JSONResponse:
    """
    Get default streaming configuration.

    Returns the default values for WebSocket streaming session configuration.
    Clients can use this to understand available options before connecting.

    Returns:
        JSONResponse with default configuration options.
    """
    from .streaming_ws import WebSocketSessionConfig

    default_config = WebSocketSessionConfig()
    return JSONResponse(
        status_code=200,
        content={
            "default_config": {
                "max_gap_sec": default_config.max_gap_sec,
                "enable_prosody": default_config.enable_prosody,
                "enable_emotion": default_config.enable_emotion,
                "enable_categorical_emotion": default_config.enable_categorical_emotion,
                "sample_rate": default_config.sample_rate,
                "audio_format": default_config.audio_format,
            },
            "supported_audio_formats": ["pcm_s16le"],
            "supported_sample_rates": [16000],
            "message_types": {
                "client": ["START_SESSION", "AUDIO_CHUNK", "END_SESSION", "PING"],
                "server": [
                    "SESSION_STARTED",
                    "PARTIAL",
                    "FINALIZED",
                    "SPEAKER_TURN",
                    "SEMANTIC_UPDATE",
                    "ERROR",
                    "SESSION_ENDED",
                    "PONG",
                ],
            },
        },
    )


# =============================================================================
# REST Session Management Endpoints (Issue #85)
# =============================================================================


@router.get(
    "/stream/sessions",
    summary="List active streaming sessions",
    description="Returns a list of all registered streaming sessions and their status.",
    tags=["Streaming"],
)
async def list_sessions() -> JSONResponse:
    """
    List all registered streaming sessions.

    Returns session info including status, config, and stats for each session.
    This endpoint provides visibility into active WebSocket sessions via REST.

    Returns:
        JSONResponse with list of sessions.
    """
    from .session_registry import get_registry

    registry = get_registry()
    sessions = registry.list_sessions()

    return JSONResponse(
        status_code=200,
        content={
            "sessions": [s.to_dict() for s in sessions],
            "count": len(sessions),
            "registry_stats": registry.get_stats(),
        },
    )


@router.post(
    "/stream/sessions",
    summary="Create a new streaming session",
    description="Creates a new streaming session for subsequent WebSocket connection.",
    tags=["Streaming"],
)
async def create_session(
    max_gap_sec: float = Query(default=1.0, ge=0.1, le=10.0),
    enable_prosody: bool = Query(default=False),
    enable_emotion: bool = Query(default=False),
    enable_diarization: bool = Query(default=False),
    sample_rate: int = Query(default=16000, ge=8000, le=48000),
) -> JSONResponse:
    """
    Create a new streaming session.

    Creates a session that can be connected to via WebSocket at /stream.
    The session_id returned should be passed in the WebSocket connection.

    Args:
        max_gap_sec: Gap threshold to finalize segment (0.1-10.0 seconds)
        enable_prosody: Extract prosodic features from audio
        enable_emotion: Extract dimensional emotion features
        enable_diarization: Enable incremental speaker diarization
        sample_rate: Expected audio sample rate (8000-48000 Hz)

    Returns:
        JSONResponse with session_id and WebSocket URL.
    """
    from .session_registry import get_registry
    from .streaming_ws import WebSocketSessionConfig, WebSocketStreamingSession

    config = WebSocketSessionConfig(
        max_gap_sec=max_gap_sec,
        enable_prosody=enable_prosody,
        enable_emotion=enable_emotion,
        enable_diarization=enable_diarization,
        sample_rate=sample_rate,
    )

    session = WebSocketStreamingSession(config=config)
    registry = get_registry()
    session_id = registry.register(session)

    logger.info("Created session via REST: %s", session_id)

    return JSONResponse(
        status_code=201,
        content={
            "session_id": session_id,
            "websocket_url": f"/stream?session_id={session_id}",
            "config": {
                "max_gap_sec": config.max_gap_sec,
                "enable_prosody": config.enable_prosody,
                "enable_emotion": config.enable_emotion,
                "enable_diarization": config.enable_diarization,
                "sample_rate": config.sample_rate,
                "audio_format": config.audio_format,
            },
        },
    )


@router.get(
    "/stream/sessions/{session_id}",
    summary="Get streaming session status",
    description="Returns detailed status and statistics for a specific session.",
    tags=["Streaming"],
)
async def get_session_status(session_id: str) -> JSONResponse:
    """
    Get status of a specific streaming session.

    Returns detailed information about the session including:
    - Current state (created, active, ending, ended, error, disconnected)
    - Configuration used
    - Runtime statistics
    - Last event ID (for resume)

    Args:
        session_id: Session ID to look up

    Returns:
        JSONResponse with session info.

    Raises:
        HTTPException: 404 if session not found.
    """
    from .session_registry import get_registry

    registry = get_registry()
    info = registry.get_info(session_id)

    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    return JSONResponse(
        status_code=200,
        content=info.to_dict(),
    )


@router.delete(
    "/stream/sessions/{session_id}",
    summary="Close streaming session",
    description="Force closes a streaming session and cleans up resources.",
    tags=["Streaming"],
)
async def close_session(session_id: str) -> JSONResponse:
    """
    Force close a streaming session.

    Ends the session if active, closes the WebSocket connection if connected,
    and removes the session from the registry.

    Args:
        session_id: Session ID to close

    Returns:
        JSONResponse confirming closure.

    Raises:
        HTTPException: 404 if session not found.
    """
    from .session_registry import get_registry

    registry = get_registry()

    # Check if session exists first
    info = registry.get_info(session_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    success = await registry.close_session(session_id)

    if success:
        logger.info("Closed session via REST: %s", session_id)
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Session {session_id} closed",
                "session_id": session_id,
            },
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close session: {session_id}",
        )
