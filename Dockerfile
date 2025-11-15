# syntax=docker/dockerfile:1

# =============================================================================
# Multi-stage Dockerfile for slower-whisper
# Provides both CPU and GPU variants with optional audio enrichment
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with Python and system dependencies
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Metadata
LABEL maintainer="slower-whisper contributors"
LABEL description="Local transcription pipeline with faster-whisper and audio enrichment"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Build stage - Install Python dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency resolution
RUN pip install uv

# Copy dependency files
COPY requirements-base.txt requirements-enrich.txt pyproject.toml ./

# Install Python dependencies
# ARG INSTALL_MODE can be: base, enrich, all
ARG INSTALL_MODE=enrich

RUN if [ "$INSTALL_MODE" = "base" ]; then \
        uv pip install --system -r requirements-base.txt; \
    elif [ "$INSTALL_MODE" = "enrich" ]; then \
        uv pip install --system -r requirements-enrich.txt; \
    elif [ "$INSTALL_MODE" = "all" ]; then \
        uv pip install --system -r requirements-enrich.txt; \
    fi

# -----------------------------------------------------------------------------
# Stage 3: Runtime image (CPU version)
# -----------------------------------------------------------------------------
FROM base AS runtime-cpu

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser transcription/ /app/transcription/
COPY --chown=appuser:appuser transcribe_pipeline.py audio_enrich.py /app/
COPY --chown=appuser:appuser examples/ /app/examples/

# Create data directories
RUN mkdir -p /app/raw_audio /app/input_audio /app/transcripts /app/whisper_json && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set default command
ENTRYPOINT ["python", "transcribe_pipeline.py"]
CMD ["--help"]

# -----------------------------------------------------------------------------
# Stage 4: Runtime image (GPU version)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser transcription/ /app/transcription/
COPY --chown=appuser:appuser transcribe_pipeline.py audio_enrich.py /app/
COPY --chown=appuser:appuser examples/ /app/examples/

# Create data directories
RUN mkdir -p /app/raw_audio /app/input_audio /app/transcripts /app/whisper_json && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set default command
ENTRYPOINT ["python", "transcribe_pipeline.py"]
CMD ["--help"]

# =============================================================================
# Build Instructions:
# =============================================================================
#
# CPU version (minimal, transcription only):
#   docker build --target runtime-cpu --build-arg INSTALL_MODE=base -t slower-whisper:cpu-base .
#
# CPU version (with audio enrichment):
#   docker build --target runtime-cpu --build-arg INSTALL_MODE=enrich -t slower-whisper:cpu .
#
# GPU version (minimal, transcription only):
#   docker build --target runtime-gpu --build-arg INSTALL_MODE=base -t slower-whisper:gpu-base .
#
# GPU version (with audio enrichment):
#   docker build --target runtime-gpu --build-arg INSTALL_MODE=enrich -t slower-whisper:gpu .
#
# =============================================================================
# Usage Examples:
# =============================================================================
#
# Run transcription (GPU):
#   docker run --gpus all \
#     -v $(pwd)/raw_audio:/app/raw_audio \
#     -v $(pwd)/output:/app/output \
#     slower-whisper:gpu \
#     --model large-v3 --language en
#
# Run transcription (CPU):
#   docker run \
#     -v $(pwd)/raw_audio:/app/raw_audio \
#     -v $(pwd)/output:/app/output \
#     slower-whisper:cpu \
#     --device cpu --model medium
#
# Interactive shell:
#   docker run -it --gpus all slower-whisper:gpu /bin/bash
#
# Audio enrichment:
#   docker run --gpus all \
#     -v $(pwd)/whisper_json:/app/whisper_json \
#     -v $(pwd)/input_audio:/app/input_audio \
#     slower-whisper:gpu \
#     python audio_enrich.py enrich whisper_json/file.json input_audio/file.wav
#
# =============================================================================
# Docker Compose:
# =============================================================================
#
# See docker-compose.yml for orchestration examples
#
