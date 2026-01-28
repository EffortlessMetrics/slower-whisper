"""Metrics endpoint for the API service."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Export metrics in Prometheus text format for monitoring and alerting.",
    tags=["System"],
    response_class=JSONResponse,
)
async def prometheus_metrics() -> JSONResponse:
    """
    Export metrics in Prometheus text format.

    This endpoint provides operational metrics compatible with Prometheus scraping:
    - Request counts by endpoint and status code
    - Request latency histograms
    - Active streaming session count
    - Error counts by type

    Returns:
        Plain text response in Prometheus exposition format
    """
    from .telemetry import get_metrics

    metrics = get_metrics()
    prometheus_text = metrics.to_prometheus_format()

    return JSONResponse(
        content={"metrics": prometheus_text},
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )
