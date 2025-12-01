# Docker Setup and Usage Guide

Complete guide for running slower-whisper in Docker containers for local development and testing.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Services Overview](#services-overview)
- [Common Usage Patterns](#common-usage-patterns)
- [Environment Configuration](#environment-configuration)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required

- **Docker Engine** 20.10+ ([Install Docker](https://docs.docker.com/engine/install/))
- **Docker Compose** v2.0+ (included with Docker Desktop)
- **8GB+ RAM** (16GB recommended for large models)
- **10GB+ disk space** for models and dependencies

### Optional (for GPU acceleration)

- **NVIDIA GPU** with CUDA support
- **NVIDIA Docker Runtime** ([Setup Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- **CUDA 12.1+** drivers

### Verify GPU Support

```bash
# Check if NVIDIA Docker runtime is available
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Expected output: GPU information and driver version
```

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your preferred settings (optional)
# Defaults are already configured for GPU with large-v3 model
```

### 2. Prepare Data Directories

```bash
# Create data directories
mkdir -p data/{raw_audio,input_audio,transcripts,whisper_json}

# Copy your audio files to raw_audio/
cp /path/to/audio/*.wav data/raw_audio/
```

### 3. Build Images

```bash
# Build GPU image (recommended if you have NVIDIA GPU)
docker compose build transcribe-gpu

# OR build CPU image (no GPU required)
docker compose build transcribe-cpu
```

### 4. Run Transcription

```bash
# GPU transcription (fast)
docker compose run --rm transcribe-gpu

# CPU transcription (slower but works everywhere)
docker compose run --rm transcribe-cpu
```

Output files will be in:
- `data/transcripts/` - TXT and SRT files
- `data/whisper_json/` - Structured JSON transcripts

## Services Overview

### Production Services

| Service | Purpose | GPU Required | Command |
|---------|---------|--------------|---------|
| `transcribe-gpu` | GPU-accelerated transcription | Yes | `docker compose run --rm transcribe-gpu` |
| `transcribe-cpu` | CPU-only transcription | No | `docker compose run --rm transcribe-cpu` |
| `enrich` | Audio enrichment (Stage 2) | Yes | `docker compose run --rm enrich` |
| `batch-processor` | Continuous batch processing | Yes | `docker compose up -d batch-processor` |

### Development Services

| Service | Purpose | Command |
|---------|---------|---------|
| `dev` | Interactive development shell | `docker compose run --rm dev` |
| `dev-tools` | Testing and linting | `docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev-tools` |
| `jupyter` | Jupyter notebook server | `docker compose -f docker-compose.yml -f docker-compose.dev.yml up jupyter` |
| `docs` | Documentation builder | `docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm docs` |

## Common Usage Patterns

### Basic Transcription

```bash
# Transcribe with default settings (large-v3 model, auto language)
docker compose run --rm transcribe-gpu

# Transcribe with specific model and language
docker compose run --rm transcribe-gpu --model medium --language en

# Transcribe with custom compute type (faster, lower quality)
docker compose run --rm transcribe-gpu --compute-type int8_float16

# CPU transcription (no GPU)
docker compose run --rm transcribe-cpu --model base
```

### Audio Enrichment

```bash
# Enrich all transcripts with prosody and emotion features
docker compose run --rm enrich

# Enrich specific transcript
docker compose run --rm enrich python audio_enrich.py enrich /app/whisper_json/file.json

# Prosody only (skip emotion recognition)
docker compose run --rm enrich --prosody-only
```

### Batch Processing

```bash
# Start batch processor in background
docker compose up -d batch-processor

# Watch logs
docker compose logs -f batch-processor

# Stop batch processor
docker compose down batch-processor
```

### Development

```bash
# Interactive shell with GPU
docker compose run --rm dev

# Inside the container:
uv run slower-whisper --help
uv run pytest
uv run ruff check .

# Run tests with coverage
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev-tools

# Start Jupyter notebook
docker compose -f docker-compose.yml -f docker-compose.dev.yml up jupyter
# Open http://localhost:8888

# Run pre-commit hooks
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm pre-commit
```

### Custom Commands

```bash
# Run specific Python script
docker compose run --rm dev python examples/basic_transcription.py

# Run benchmarks
docker compose run --rm dev python benchmarks/benchmark_audio_enrich.py

# Access Python REPL
docker compose run --rm dev python

# Execute shell commands
docker compose run --rm dev bash -c "ls -la && uv run slower-whisper --help"
```

## Environment Configuration

### Using .env File

Create `.env` from `.env.example` and customize:

```bash
# .env
SLOWER_WHISPER_MODEL=large-v3
SLOWER_WHISPER_LANGUAGE=en
SLOWER_WHISPER_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0
```

Then run:
```bash
docker compose run --rm transcribe-gpu
# Uses settings from .env
```

### Using Command Line Arguments

Override settings directly:

```bash
# Override model
docker compose run --rm transcribe-gpu --model medium

# Override language
docker compose run --rm transcribe-gpu --language ja

# Multiple overrides
docker compose run --rm transcribe-gpu \
  --model base \
  --language es \
  --compute-type int8_float16
```

### Using Environment Variables

```bash
# One-time override
SLOWER_WHISPER_MODEL=medium docker compose run --rm transcribe-gpu

# Specify GPU
CUDA_VISIBLE_DEVICES=1 docker compose run --rm transcribe-gpu
```

## Development Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-repo/slower-whisper.git
cd slower-whisper

# Build development image
docker compose -f docker-compose.yml -f docker-compose.dev.yml build dev

# Start interactive shell
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev
```

### 2. Make Code Changes

Edit files on your host machine. Changes are immediately reflected in the container due to volume mounts.

### 3. Run Tests

```bash
# Inside dev container
uv run pytest -v

# With coverage
uv run pytest --cov=transcription --cov-report=html

# Specific test file
uv run pytest tests/test_prosody.py -v

# Fast tests only (skip slow and GPU tests)
uv run pytest -m "not slow and not requires_gpu"
```

### 4. Code Quality Checks

```bash
# Inside dev container

# Format code
uv run ruff format transcription/ tests/

# Lint code
uv run ruff check transcription/ tests/

# Auto-fix linting issues
uv run ruff check --fix transcription/ tests/

# Type check
uv run mypy transcription/

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### 5. Run Development Tools Service

```bash
# Automated testing and linting (from host)
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev-tools

# This runs:
# - ruff check
# - mypy
# - pytest with coverage
```

### 6. Interactive Notebooks

```bash
# Start Jupyter Lab
docker compose -f docker-compose.yml -f docker-compose.dev.yml up jupyter

# Access at http://localhost:8888
# No password required (development only!)
```

### 7. Profile Performance

```bash
# Run profiling service
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm profiler

# Results saved to ./profiling/
```

### 8. Build Documentation

```bash
# Build and serve docs
docker compose -f docker-compose.yml -f docker-compose.dev.yml up docs

# Access at http://localhost:8000
```

## Volume Management

### Data Volumes

Data directories are mounted from `./data/`:

```
./data/
├── raw_audio/         # Input: Your audio files (read-only in container)
├── input_audio/       # Generated: Normalized 16kHz WAV files
├── transcripts/       # Generated: TXT and SRT outputs
└── whisper_json/      # Generated: JSON structured transcripts
```

### Cache Volumes

Models are cached to avoid re-downloading:

```bash
# View volumes
docker volume ls | grep slower-whisper

# Remove cache volumes (models will re-download)
docker compose down -v

# Inspect volume
docker volume inspect slower-whisper_huggingface-cache
```

Cache volumes reuse your host cache:
- HuggingFace: `~/.cache/huggingface`
- Whisper: `~/.cache/whisper`

## Troubleshooting

### GPU Not Detected

**Problem:** Container doesn't see GPU

**Solution:**
```bash
# 1. Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 2. Check docker-compose.yml uses correct runtime
grep "runtime: nvidia" docker-compose.yml

# 3. Fall back to CPU
docker compose run --rm transcribe-cpu
```

### Out of Memory

**Problem:** Container killed due to OOM

**Solution:**
```bash
# 1. Use smaller model
docker compose run --rm transcribe-gpu --model medium

# 2. Use lower precision
docker compose run --rm transcribe-gpu --compute-type int8_float16

# 3. Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory > 16GB

# 4. Edit docker-compose.yml memory limits
```

### Slow Transcription

**Problem:** Transcription is very slow

**Solution:**
```bash
# 1. Verify GPU is being used
docker compose run --rm dev nvidia-smi

# 2. Check compute type
docker compose run --rm transcribe-gpu --compute-type float16

# 3. Use smaller model for faster (but less accurate) results
docker compose run --rm transcribe-gpu --model base
```

### Permission Errors

**Problem:** Cannot write to output directories

**Solution:**
```bash
# Fix permissions
sudo chown -R $USER:$USER data/

# Or run with current user ID
docker compose run --rm --user $(id -u):$(id -g) transcribe-gpu
```

### Build Failures

**Problem:** Docker build fails

**Solution:**
```bash
# 1. Clean build cache
docker compose build --no-cache transcribe-gpu

# 2. Prune Docker resources
docker system prune -a

# 3. Check disk space
df -h

# 4. Update Docker
docker --version  # Should be 20.10+
```

### Model Download Failures

**Problem:** HuggingFace models fail to download

**Solution:**
```bash
# 1. Check internet connection
docker compose run --rm dev ping -c 3 huggingface.co

# 2. Pre-download models on host
python -c "from transformers import AutoModel; AutoModel.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')"

# 3. Mount pre-downloaded cache
# (volumes already configured in docker-compose.yml)
```

### Container Exits Immediately

**Problem:** Container starts and immediately exits

**Solution:**
```bash
# 1. Check logs
docker compose logs transcribe-gpu

# 2. Run interactively to debug
docker compose run --rm transcribe-gpu /bin/bash

# 3. Check command syntax
docker compose run --rm transcribe-gpu --help
```

## Advanced Configuration

### Multi-GPU Setup

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=1 docker compose run --rm transcribe-gpu

# Use multiple GPUs (not supported by faster-whisper, but available)
CUDA_VISIBLE_DEVICES=0,1 docker compose run --rm transcribe-gpu
```

### Custom Data Directory

```bash
# Edit docker-compose.yml volumes section:
volumes:
  - /custom/path/audio:/app/raw_audio:ro
  - /custom/path/output:/app/whisper_json
```

### Networking

```bash
# Expose Jupyter to specific IP
# Edit docker-compose.dev.yml:
ports:
  - "127.0.0.1:8888:8888"  # Localhost only
  - "0.0.0.0:8888:8888"    # All interfaces (use with caution)
```

### Resource Limits

Edit `docker-compose.yml` deploy section:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'       # Max CPU cores
      memory: 16G     # Max RAM
    reservations:
      memory: 4G      # Reserved RAM
```

## Production Deployment

For production use, consider:

1. **Use specific image tags** instead of `latest`
2. **Configure resource limits** based on your hardware
3. **Set up log rotation** for long-running services
4. **Use Docker secrets** for any credentials
5. **Enable health checks** for monitoring
6. **Run as non-root user** (already configured)
7. **Use read-only volumes** where possible (already configured for `raw_audio`)

Example production command:

```bash
# Production transcription with all optimizations
docker compose run --rm \
  --memory=8g \
  --cpus=4 \
  transcribe-gpu \
  --model large-v3 \
  --language en \
  --compute-type float16 \
  --skip-existing-json
```

## Useful Docker Commands

```bash
# List running containers
docker compose ps

# View logs
docker compose logs -f [service-name]

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# Rebuild specific service
docker compose build transcribe-gpu

# Clean up Docker resources
docker system prune -a --volumes

# Check resource usage
docker stats

# Execute command in running container
docker compose exec dev bash

# View container processes
docker compose top dev
```

## Further Reading

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Docker Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html)
- [slower-whisper Architecture](docs/ARCHITECTURE.md)
- [slower-whisper README](README.md)
