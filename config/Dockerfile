# syntax=docker/dockerfile:1.4

# =============================================================================
# Production-ready Dockerfile for slower-whisper (CPU version)
# Multi-stage build with pinned versions and uv dependency management
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with Python and system dependencies
# -----------------------------------------------------------------------------
# Using pinned Python version for reproducibility
FROM python:3.12.8-slim-bookworm AS base

# Metadata
LABEL maintainer="slower-whisper contributors"
LABEL description="Local audio transcription pipeline with faster-whisper and audio enrichment"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/steven/slower-whisper"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Increase thread pool for faster-whisper
    OMP_NUM_THREADS=4

# Install system dependencies
# ffmpeg: Audio normalization and conversion
# libsndfile1: WAV file I/O for audio enrichment
# libgomp1: OpenMP runtime for parallel processing
# ca-certificates: SSL/TLS certificates for model downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
# Explicitly create a same-named group (-U) so named ownership (appuser:appuser) is always resolvable.
RUN useradd -m -u 1000 -U -s /bin/bash appuser

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Builder - Install uv and Python dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

# Install build dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv - pinned version for reproducibility
# uv is Astral's fast Python package manager
ENV UV_VERSION=0.5.11
RUN pip install --no-cache-dir uv==${UV_VERSION}

# Copy dependency files and application code
# Using layer caching optimization: copy dependency files first
COPY pyproject.toml uv.lock README.md ./
COPY transcription/ ./transcription/
COPY scripts/ ./scripts/
COPY integrations/ ./integrations/

# Install Python dependencies using uv
# ARG INSTALL_MODE controls which dependencies to install
# - base: faster-whisper only (minimal, ~2.5GB)
# - full: all enrichment features (prosody + emotion, ~4GB additional)
ARG INSTALL_MODE=full
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
ARG PYPI_INDEX=https://pypi.org/simple
ARG TORCH_VERSION=2.8.0+cpu
ARG TORCHAUDIO_VERSION=2.8.0+cpu

RUN cat <<EOF > /tmp/constraints-cpu.txt
--index-url ${PYPI_INDEX}
--extra-index-url ${TORCH_CPU_INDEX}
torch==${TORCH_VERSION}
torchaudio==${TORCHAUDIO_VERSION}
EOF

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_MODE" = "base" ]; then \
        uv pip install --system \
            --index-url "${PYPI_INDEX}" \
            --extra-index-url "${TORCH_CPU_INDEX}" \
            --index-strategy unsafe-best-match \
            -c /tmp/constraints-cpu.txt \
            -e .; \
    elif [ "$INSTALL_MODE" = "full" ]; then \
        uv pip install --system \
            --index-url "${TORCH_CPU_INDEX}" \
            --extra-index-url "${PYPI_INDEX}" \
            "torch==${TORCH_VERSION}" \
            "torchaudio==${TORCHAUDIO_VERSION}"; \
        uv pip install --system \
            --index-url "${PYPI_INDEX}" \
            --extra-index-url "${TORCH_CPU_INDEX}" \
            --index-strategy unsafe-best-match \
            -c /tmp/constraints-cpu.txt \
            -e ".[full]"; \
    else \
        echo "Invalid INSTALL_MODE: $INSTALL_MODE (must be 'base' or 'full')" && exit 1; \
    fi

RUN rm -f /tmp/constraints-cpu.txt

# -----------------------------------------------------------------------------
# Stage 3: Runtime image
# -----------------------------------------------------------------------------
FROM base AS runtime

# Copy Python packages from builder
# This keeps the final image smaller by excluding build tools
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser transcription/ /app/transcription/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser integrations/ /app/integrations/
COPY --chown=appuser:appuser pyproject.toml /app/

# Copy legacy entry points for backward compatibility
COPY --chown=appuser:appuser scripts/transcribe_pipeline.py scripts/audio_enrich.py /app/

# Create data directories
# These will typically be mounted as volumes
RUN mkdir -p /app/raw_audio /app/input_audio /app/transcripts /app/whisper_json && \
    chown -R appuser:appuser /app

# Create cache directory for model downloads
RUN mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser/.cache

# Switch to non-root user
USER appuser

# Expose volume mount points
VOLUME ["/app/raw_audio", "/app/input_audio", "/app/transcripts", "/app/whisper_json"]

# Health check (optional - verifies CLI is working)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD slower-whisper --help || exit 1

# Set default entrypoint to slower-whisper CLI
ENTRYPOINT ["slower-whisper"]
CMD ["--help"]

# =============================================================================
# Build Instructions:
# =============================================================================
#
# Minimal build (transcription only, ~2.5GB):
#   docker build --build-arg INSTALL_MODE=base -t slower-whisper:cpu-base .
#
# Full build (with audio enrichment, ~6.5GB):
#   docker build --build-arg INSTALL_MODE=full -t slower-whisper:cpu .
#   docker build -t slower-whisper:cpu .  # (full is default)
#
# =============================================================================
# Usage Examples:
# =============================================================================
#
# Transcribe audio files in a directory:
#   docker run --rm \
#     -v $(pwd)/raw_audio:/app/raw_audio \
#     -v $(pwd)/transcripts:/app/transcripts \
#     -v $(pwd)/whisper_json:/app/whisper_json \
#     slower-whisper:cpu \
#     transcribe --model large-v3 --language en --device cpu
#
# Enrich existing transcripts with audio features:
#   docker run --rm \
#     -v $(pwd)/whisper_json:/app/whisper_json \
#     -v $(pwd)/input_audio:/app/input_audio \
#     slower-whisper:cpu \
#     enrich --model dimensional
#
# Interactive shell for debugging:
#   docker run --rm -it \
#     -v $(pwd)/raw_audio:/app/raw_audio \
#     slower-whisper:cpu \
#     /bin/bash
#
# Using legacy entry point (backward compatibility):
#   docker run --rm \
#     -v $(pwd)/raw_audio:/app/raw_audio \
#     slower-whisper:cpu \
#     python transcribe_pipeline.py --help
#
# =============================================================================
# Development Build:
# =============================================================================
#
# For development with live code changes:
#   docker run --rm -it \
#     -v $(pwd):/app \
#     -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
#     slower-whisper:cpu \
#     /bin/bash
#
