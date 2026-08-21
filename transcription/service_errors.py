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
    """Create the stable structured API error response."""
    error_data: dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code,
        },
        "detail": message,
    }
    if request_id:
        error_data["error"]["request_id"] = request_id
    if details:
        error_data["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=error_data)


def _validation_errors_for_log(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce validation failures to bounded fields safe for structured logs."""
    reduced: list[dict[str, Any]] = []
    for error in errors:
        location = []
        for item in error.get("loc", []):
            if isinstance(item, int):
                location.append(item)
                continue
            normalized = str(item).replace("\r", "\\r").replace("\n", "\\n")
            location.append(normalized[:128])
        reduced.append(
            {
                "loc": location,
                "type": str(error.get("type", "unknown"))[:128],
            }
        )
    return reduced


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic/FastAPI request validation errors."""
    request_id = getattr(request.state, "request_id", None)
    errors = exc.errors()
    formatted_errors = [
        {
            "loc": list(error.get("loc", [])),
            "msg": error.get("msg", "Validation error"),
            "type": error.get("type", "unknown"),
        }
        for error in errors
    ]
    logger.warning(
        "Validation error: %s %s [request_id=%s] - %d validation errors",
        request.method,
        request.url.path,
        request_id,
        len(errors),
        extra={
            "request_id": request_id,
            "validation_errors": _validation_errors_for_log(errors),
        },
    )
    return create_error_response(
        status_code=HTTP_422_UNPROCESSABLE,
        error_type="validation_error",
        message="Request validation failed",
        request_id=request_id,
        details={"validation_errors": formatted_errors},
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize endpoint HTTPException responses."""
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
    error_type_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "file_too_large",
        500: "internal_error",
        503: "service_unavailable",
    }
    return create_error_response(
        status_code=exc.status_code,
        error_type=error_type_map.get(exc.status_code, "http_error"),
        message=str(exc.detail),
        request_id=request_id,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log full unexpected failures while returning a sanitized response."""
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
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="internal_error",
        message="An unexpected internal error occurred",
        request_id=request_id,
        details={"hint": "Check server logs for details"},
    )


def register_exception_handlers(app) -> None:
    """Register API exception handlers on the FastAPI app."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
