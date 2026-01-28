"""Middleware used by the API service."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request

logger = logging.getLogger(__name__)


async def log_requests(request: Request, call_next):
    """
    Log all requests with timing and request ID.

    This middleware:
    - Generates a unique request_id for tracing
    - Logs request method, path, and query params
    - Measures request duration
    - Logs response status and timing
    - Attaches request_id to request.state for use in exception handlers
    - Records metrics for Prometheus endpoint
    """
    from .telemetry import get_metrics

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Log incoming request
    logger.info(
        "Request started: %s %s [request_id=%s]",
        request.method,
        request.url.path,
        request_id,
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
        },
    )

    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    # Log response
    logger.info(
        "Request completed: %s %s -> %d (%.2f ms) [request_id=%s]",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    # Record metrics for Prometheus
    # Skip metrics endpoint to avoid recursive counting
    if request.url.path != "/metrics":
        metrics = get_metrics()
        metrics.record_request(request.url.path, response.status_code, duration_ms)

    # Add request_id to response headers
    response.headers["X-Request-ID"] = request_id

    return response


async def add_security_headers(request: Request, call_next):
    """
    Add security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - Content-Security-Policy: default-src 'self'
    """
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"

    # CSP: Allow Swagger UI/Redoc assets (CDN) + 'self'
    # script-src/style-src need 'unsafe-inline' for Swagger UI
    csp_directives = [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net",
        "img-src 'self' data: fastapi.tiangolo.com",
    ]
    response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

    return response
