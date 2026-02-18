"""Error response helpers and exception handler registration."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .service_settings import HTTP_422_UNPROCESSABLE

logger = logging.getLogger(__name__)


def create_error_response(
    status_code: int,
    error_type: str,
    message: str,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """
    Create a standardized error response.

    Args:
        status_code: HTTP status code
        error_type: Error type identifier (e.g., "validation_error", "transcription_error")
        message: Human-readable error message
        request_id: Optional request ID for tracing
        details: Optional additional error details

    Returns:
        JSONResponse with structured error format
    """
    error_data: dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code,
        },
        # Backward compatibility: include "detail" field for legacy clients
        "detail": message,
    }

    if request_id:
        error_data["error"]["request_id"] = request_id

    if details:
        error_data["error"]["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=error_data,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors (422).

    FastAPI raises RequestValidationError when request data fails Pydantic validation
    (e.g., invalid query parameters, missing required fields, type mismatches).
    """
    request_id = getattr(request.state, "request_id", None)

    # Extract validation error details
    errors = exc.errors()
    logger.warning(
        "Validation error: %s %s [request_id=%s] - %d validation errors",
        request.method,
        request.url.path,
        request_id,
        len(errors),
        extra={
            "request_id": request_id,
            "validation_errors": errors,
        },
    )

    # Format validation errors for response
    formatted_errors = [
        {
            "loc": list(err.get("loc", [])),
            "msg": err.get("msg", "Validation error"),
            "type": err.get("type", "unknown"),
        }
        for err in errors
    ]

    return create_error_response(
        status_code=HTTP_422_UNPROCESSABLE,
        error_type="validation_error",
        message="Request validation failed",
        request_id=request_id,
        details={"validation_errors": formatted_errors},
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTPException (4xx/5xx errors raised by endpoint logic).

    This handler provides structured error responses for all HTTPException instances,
    including those raised explicitly in endpoints (400 Bad Request, 500 Internal Server Error).
    """
    request_id = getattr(request.state, "request_id", None)

    logger.warning(
        "HTTP exception: %s %s -> %d [request_id=%s] - %s",
        request.method,
        request.url.path,
        exc.status_code,
        request_id,
        exc.detail,
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )

    # Determine error type from status code
    error_type_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "file_too_large",
        500: "internal_error",
        503: "service_unavailable",
    }
    error_type = error_type_map.get(exc.status_code, "http_error")

    return create_error_response(
        status_code=exc.status_code,
        error_type=error_type,
        message=str(exc.detail),
        request_id=request_id,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all unhandled exceptions (500 Internal Server Error).

    This is the global catch-all handler for any exception not caught by
    endpoint logic or other handlers. Logs full traceback and returns
    generic error to avoid leaking internal details.
    """
    request_id = getattr(request.state, "request_id", None)

    logger.exception(
        "Unhandled exception: %s %s [request_id=%s]",
        request.method,
        request.url.path,
        request_id,
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
        },
        exc_info=exc,
    )

    # Security fix: Remove exception type information from client response
    # This prevents potential information disclosure about internal implementation
    # Exception details are still logged server-side for debugging
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="internal_error",
        message="An unexpected internal error occurred",
        request_id=request_id,
        details={
            "hint": "Check server logs for details",
        },
    )


def register_exception_handlers(app) -> None:
    """Register API exception handlers on the FastAPI app."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
