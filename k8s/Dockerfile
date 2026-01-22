# Multi-stage Dockerfile for slower-whisper
# Optimized for production Kubernetes deployments

# Stage 1: Base image with CUDA runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip3 install --no-cache-dir uv

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
# Use --extra full for all features (transcription + enrichment)
RUN uv sync --extra full --no-dev

# Stage 3: Final production image
FROM base AS production

# Copy installed dependencies from dependencies stage
COPY --from=dependencies /app/.venv /app/.venv

# Copy application code
COPY transcription/ ./transcription/
COPY README.md CHANGELOG.md ./

# Copy pyproject.toml for package metadata
COPY pyproject.toml ./

# Set up cache directories
RUN mkdir -p /cache/huggingface && \
    mkdir -p /data/raw_audio /data/input_audio /data/transcripts /data/whisper_json && \
    chown -R appuser:appuser /app /cache /data

# Set environment variables
ENV PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD nvidia-smi || exit 1

# Default command
ENTRYPOINT ["uv", "run"]
CMD ["slower-whisper", "--help"]

# Labels for metadata
LABEL maintainer="your-email@example.com" \
      description="slower-whisper: Local audio transcription with enrichment" \
      version="1.0.0"
