"""Shared configuration constants for the API service."""

import os

from fastapi import status

# Maximum allowed file size in megabytes (configurable)
MAX_AUDIO_SIZE_MB = 500
MAX_TRANSCRIPT_SIZE_MB = 10  # JSON transcripts are typically small

# Streaming upload chunk size (1MB)
STREAMING_CHUNK_SIZE = 1024 * 1024

# Starlette renamed a couple status constants. Keep runtime compatibility
# without breaking mypy on older/lagging stubs.
HTTP_413_TOO_LARGE: int = getattr(
    status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
)
HTTP_422_UNPROCESSABLE: int = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY
)

# CORS configuration
_allowed_origins_str = os.environ.get("SLOWER_WHISPER_ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins_str.split(",") if origin.strip()]
