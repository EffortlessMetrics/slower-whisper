# Docker Deployment Guide

This document provides comprehensive guidance for deploying **slower-whisper** using Docker containers.

## Files Created/Updated

### Dockerfiles
- **`Dockerfile`** - Production-ready CPU version (181 lines)
- **`Dockerfile.gpu`** - Production-ready GPU version with CUDA support (235 lines)
- **`.dockerignore`** - Build context exclusions (212 lines)
- **`docker-compose.yml`** - Orchestration configuration (existing, comprehensive)

## Key Features

### Production-Ready Features

#### Pinned Versions
- **Python**: `3.12.8-slim-bookworm` (CPU) / `3.12` via deadsnakes PPA (GPU)
- **CUDA**: `12.1.0-runtime-ubuntu22.04` (GPU only)
- **uv**: `0.5.11` (Python package manager)
- **PyTorch**: `2.5.1+cu121` (GPU only)

#### Multi-Stage Builds
Both Dockerfiles use multi-stage builds to minimize final image size:
1. **Base stage**: System dependencies and Python
2. **Builder stage**: uv installation and Python dependencies
3. **Runtime stage**: Final minimal image with only runtime requirements

#### Security
- Non-root user (`appuser`, UID 1000)
- Minimal base images (slim/runtime only, no unnecessary tools)
- No credentials or secrets in images
- Read-only volume mounts where appropriate

#### Optimization
- Layer caching for faster rebuilds
- Build cache mounts for uv (`--mount=type=cache`)
- Proper `.dockerignore` to minimize build context
- Separated dependency installation from code copying

## Build Instructions

### CPU Version

**Minimal build (transcription only, ~2.5GB)**:
```bash
docker build --build-arg INSTALL_MODE=base -t slower-whisper:cpu-base .
```

**Full build (with audio enrichment, ~6.5GB)**:
```bash
docker build --build-arg INSTALL_MODE=full -t slower-whisper:cpu .
# OR (full is default)
docker build -t slower-whisper:cpu .
```

### GPU Version

**Prerequisites**:
- NVIDIA GPU with compute capability >= 7.0 (Volta or newer: RTX 20/30/40 series, Tesla V100/A100)
- NVIDIA driver version >= 525.60.13
- NVIDIA Docker runtime installed ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

**Verify GPU setup**:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Minimal build (transcription only, ~2.5GB)**:
```bash
docker build -f Dockerfile.gpu --build-arg INSTALL_MODE=base -t slower-whisper:gpu-base .
```

**Full build (with audio enrichment and CUDA, ~8.5GB)**:
```bash
docker build -f Dockerfile.gpu --build-arg INSTALL_MODE=full -t slower-whisper:gpu .
# OR (full is default)
docker build -f Dockerfile.gpu -t slower-whisper:gpu .
```

## Usage Examples

### CPU Version

**Transcribe audio files**:
```bash
docker run --rm \
  -v $(pwd)/raw_audio:/app/raw_audio \
  -v $(pwd)/transcripts:/app/transcripts \
  -v $(pwd)/whisper_json:/app/whisper_json \
  slower-whisper:cpu \
  transcribe --model large-v3 --language en --device cpu
```

**Enrich existing transcripts**:
```bash
docker run --rm \
  -v $(pwd)/whisper_json:/app/whisper_json \
  -v $(pwd)/input_audio:/app/input_audio \
  slower-whisper:cpu \
  enrich --model dimensional
```

**Interactive shell for debugging**:
```bash
docker run --rm -it \
  -v $(pwd)/raw_audio:/app/raw_audio \
  slower-whisper:cpu \
  /bin/bash
```

### GPU Version

**Transcribe audio files using GPU**:
```bash
docker run --rm --gpus all \
  -v $(pwd)/raw_audio:/app/raw_audio \
  -v $(pwd)/transcripts:/app/transcripts \
  -v $(pwd)/whisper_json:/app/whisper_json \
  slower-whisper:gpu \
  transcribe --model large-v3 --language en --device cuda
```

**Transcribe with specific GPU**:
```bash
docker run --rm --gpus '"device=0"' \
  -v $(pwd)/raw_audio:/app/raw_audio \
  -v $(pwd)/transcripts:/app/transcripts \
  slower-whisper:gpu \
  transcribe --model large-v3 --device cuda
```

**Enrich with GPU acceleration**:
```bash
docker run --rm --gpus all \
  -v $(pwd)/whisper_json:/app/whisper_json \
  -v $(pwd)/input_audio:/app/input_audio \
  slower-whisper:gpu \
  enrich --model dimensional --device cuda
```

**Verify GPU accessibility**:
```bash
docker run --rm --gpus all slower-whisper:gpu \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

### Using Legacy Entry Points

For backward compatibility, legacy scripts are still available:

```bash
docker run --rm \
  -v $(pwd)/raw_audio:/app/raw_audio \
  slower-whisper:cpu \
  python transcribe_pipeline.py --help
```

## Docker Compose Usage

The project includes a comprehensive `docker-compose.yml` file for easy orchestration.

### Services Available

- **`transcribe-cpu`** - CPU-based transcription
- **`transcribe-gpu`** - GPU-accelerated transcription
- **`enrich`** - Audio enrichment (prosody + emotion)
- **`batch-processor`** - Continuous batch processing (daemon mode)
- **`dev`** - Interactive development shell

### Quick Start

**Build services**:
```bash
docker compose build transcribe-cpu
docker compose build transcribe-gpu
```

**Run transcription (CPU)**:
```bash
docker compose run --rm transcribe-cpu
```

**Run transcription (GPU)**:
```bash
docker compose run --rm transcribe-gpu
```

**Run with custom command**:
```bash
docker compose run --rm transcribe-cpu transcribe --model medium --language fr
docker compose run --rm transcribe-gpu transcribe --model large-v3 --device cuda
```

**Run audio enrichment**:
```bash
docker compose run --rm enrich
```

**Interactive development shell**:
```bash
docker compose run --rm dev
```

**Start batch processor (daemon mode)**:
```bash
docker compose up -d batch-processor
docker compose logs -f batch-processor
```

**Stop services**:
```bash
docker compose down
```

**Clean up volumes**:
```bash
docker compose down -v
```

## Directory Structure

Ensure these directories exist in your project root before running:

```bash
mkdir -p raw_audio input_audio transcripts whisper_json
```

**Directory purposes**:
- **`raw_audio/`** - Original audio files (input)
- **`input_audio/`** - Normalized 16kHz mono WAV files (auto-generated)
- **`transcripts/`** - TXT and SRT outputs (auto-generated)
- **`whisper_json/`** - Structured JSON transcripts (auto-generated)

## Volume Mounts

### Model Caches

To avoid re-downloading models on every container restart, mount cache directories:

```bash
docker run --rm \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  slower-whisper:cpu \
  transcribe
```

This is automatically configured in `docker-compose.yml`.

### Development Mode

For development with live code changes:

```bash
docker run --rm -it \
  -v $(pwd):/app \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  slower-whisper:cpu \
  /bin/bash
```

## Build Arguments

### `INSTALL_MODE`

Controls which dependencies to install:

- **`base`** - Faster-whisper only (minimal, ~2.5GB)
- **`full`** - All enrichment features: prosody + emotion (~6.5GB CPU, ~8.5GB GPU)

Default: `full`

Example:
```bash
docker build --build-arg INSTALL_MODE=base -t slower-whisper:cpu-minimal .
```

## Environment Variables

### Transcription Configuration

Set via environment variables or `.env` file:

```bash
SLOWER_WHISPER_MODEL=large-v3        # Whisper model size
SLOWER_WHISPER_DEVICE=cuda           # cuda or cpu
SLOWER_WHISPER_LANGUAGE=auto         # Language code or 'auto'
SLOWER_WHISPER_COMPUTE_TYPE=float16  # Compute precision
```

### Performance Tuning

```bash
OMP_NUM_THREADS=4                    # OpenMP threads for CPU
CUDA_VISIBLE_DEVICES=0               # Specific GPU to use
```

### GPU Configuration

```bash
NVIDIA_VISIBLE_DEVICES=all           # Which GPUs to expose
NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_MODULE_LOADING=LAZY             # Lazy CUDA module loading
```

## Health Checks

Both Dockerfiles include health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD slower-whisper --help || exit 1
```

This verifies the CLI is working and allows orchestration tools (Kubernetes, Docker Swarm) to detect unhealthy containers.

## Resource Limits

### CPU Version

Recommended limits:
- **CPUs**: 2-8 cores
- **Memory**: 4-8GB (depends on model size)

Example:
```bash
docker run --rm \
  --cpus=4 \
  --memory=8g \
  slower-whisper:cpu \
  transcribe
```

### GPU Version

Recommended limits:
- **Memory**: 8-16GB (depends on model size and batch size)
- **GPU**: 1 GPU (or specific GPU via `--gpus '"device=0"'`)

Example:
```bash
docker run --rm \
  --gpus all \
  --memory=16g \
  slower-whisper:gpu \
  transcribe
```

## Troubleshooting

### GPU Not Detected

**Error**: `RuntimeError: No CUDA GPUs are available`

**Solutions**:
1. Verify NVIDIA Docker runtime is installed:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

2. Check driver version:
   ```bash
   nvidia-smi
   ```
   Should show driver version >= 525.60.13

3. Ensure `--gpus all` flag is passed:
   ```bash
   docker run --rm --gpus all slower-whisper:gpu transcribe
   ```

### Out of Memory (OOM)

**Error**: Container exits with OOM error

**Solutions**:
1. Use smaller model: `--model medium` or `--model base`
2. Increase memory limit: `--memory=16g`
3. Reduce compute type: `--compute-type int8` (CPU) or `--compute-type int8_float16` (GPU)
4. Process files one at a time instead of batch

### Build Fails: uv pip install

**Error**: `uv pip install` fails with dependency conflicts

**Solutions**:
1. Update `uv.lock`:
   ```bash
   uv lock --upgrade
   ```

2. Clear build cache:
   ```bash
   docker builder prune
   ```

3. Rebuild without cache:
   ```bash
   docker build --no-cache -t slower-whisper:cpu .
   ```

### Permission Denied

**Error**: `Permission denied` when writing to volume mounts

**Solutions**:
1. Ensure directories are writable:
   ```bash
   chmod -R 755 raw_audio input_audio transcripts whisper_json
   ```

2. Match host UID with container UID (1000):
   ```bash
   sudo chown -R 1000:1000 raw_audio input_audio transcripts whisper_json
   ```

### Slow Build Times

**Solutions**:
1. Use build cache mounts (requires Docker BuildKit):
   ```bash
   DOCKER_BUILDKIT=1 docker build -t slower-whisper:cpu .
   ```

2. Minimize build context via `.dockerignore` (already configured)

3. Use multi-stage builds (already implemented)

## Best Practices

### Production Deployment

1. **Use specific tags**: Never use `:latest` in production
   ```bash
   docker tag slower-whisper:cpu slower-whisper:cpu-1.0.0
   ```

2. **Pin all versions**: Already done in Dockerfiles (Python 3.12.8, uv 0.5.11, CUDA 12.1.0)

3. **Resource limits**: Always set CPU and memory limits in production

4. **Health checks**: Enabled by default, configure as needed

5. **Logging**: Use centralized logging (Docker logs, ELK stack, etc.)

6. **Secrets**: Never bake secrets into images - use environment variables or Docker secrets

### Development

1. **Mount source code**: For live reloading during development
   ```bash
   docker run -v $(pwd):/app slower-whisper:cpu /bin/bash
   ```

2. **Use dev service**: Configured in `docker-compose.yml`
   ```bash
   docker compose run --rm dev
   ```

3. **Preserve caches**: Mount HuggingFace cache to avoid re-downloading models

### Performance

1. **GPU acceleration**: Always use GPU version for production transcription (10-20x faster)

2. **Batch processing**: Use `batch-processor` service for continuous processing

3. **Model caching**: Mount cache volumes to avoid re-downloading on every run

4. **Compute type**: Use `float16` (GPU) or `int8` (CPU) for best performance

## Security Considerations

### Non-root User

Both Dockerfiles run as non-root user `appuser` (UID 1000) for security.

### Minimal Base Images

- CPU: `python:3.12.8-slim-bookworm` (Debian-based, minimal)
- GPU: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` (runtime only, no dev tools)

### No Secrets

No credentials, API keys, or secrets are baked into images. All configuration is via environment variables or volume mounts.

### Network Isolation

Containers only require outbound network access to download models (first run). No inbound network access required.

## Maintenance

### Updating Python Version

Edit Dockerfiles:
```dockerfile
FROM python:3.12.9-slim-bookworm AS base  # Update version
```

### Updating CUDA Version

Edit `Dockerfile.gpu`:
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base  # Update version
```

Also update PyTorch version to match CUDA:
```bash
uv pip install --system torch==2.5.1+cu122 --index-url https://download.pytorch.org/whl/cu122
```

### Updating uv Version

Edit Dockerfiles:
```dockerfile
ENV UV_VERSION=0.6.0  # Update version
```

### Rebuilding Images

After updating Dockerfiles:
```bash
docker build -t slower-whisper:cpu .
docker build -f Dockerfile.gpu -t slower-whisper:gpu .
docker compose build
```

## Comparison: CPU vs GPU

| Feature | CPU Version | GPU Version |
|---------|-------------|-------------|
| **Base Image** | `python:3.12.8-slim-bookworm` | `nvidia/cuda:12.1.0-runtime-ubuntu22.04` |
| **Size (base)** | ~2.5GB | ~2.5GB |
| **Size (full)** | ~6.5GB | ~8.5GB (includes CUDA libraries) |
| **Speed** | Baseline (1x) | 10-20x faster |
| **Hardware** | Any x86_64 CPU | NVIDIA GPU (compute >= 7.0) |
| **Recommended Model** | `medium` or `small` | `large-v3` |
| **Compute Type** | `int8` | `float16` |
| **Memory** | 4-8GB | 8-16GB |
| **Use Case** | Development, testing, low-volume | Production, batch processing |

## Summary

These production-ready Dockerfiles provide:

- **Pinned versions** for reproducibility (Python 3.12.8, uv 0.5.11, CUDA 12.1.0)
- **Multi-stage builds** for minimal image size
- **Security** via non-root user and minimal base images
- **Flexibility** via build args (`INSTALL_MODE=base|full`)
- **Performance** via GPU acceleration and build cache mounts
- **Documentation** via comprehensive inline comments and usage examples
- **Integration** with existing `docker-compose.yml` for easy orchestration

Both CPU and GPU versions use the modern `slower-whisper` CLI as the default entrypoint, with backward compatibility for legacy scripts.
