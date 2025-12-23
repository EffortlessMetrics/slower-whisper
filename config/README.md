# Configuration Files

This directory contains configuration files for deploying and running slower-whisper.

## Files

- `docker-compose.yml` - Main Docker Compose configuration for local development and testing
- `docker-compose.api.yml` - Docker Compose configuration for API service deployment
- `docker-compose.dev.yml` - Development overrides for Docker Compose
- `Dockerfile` - Standard Docker image for CPU-based deployment
- `Dockerfile.api` - Docker image for FastAPI service
- `Dockerfile.gpu` - Docker image with GPU support
- `.dockerignore` - Files to exclude from Docker build context

## Usage

### Basic Usage
```bash
# Use default compose file
docker compose up

# Use API configuration
docker compose -f docker-compose.api.yml up

# Use development overrides
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Building Images
```bash
# Build standard image
docker build -f Dockerfile -t slower-whisper:cpu

# Build GPU image
docker build -f Dockerfile.gpu -t slower-whisper:gpu

# Build API image
docker build -f Dockerfile.api -t slower-whisper:api
```

## Configuration Notes

- Environment variables can be set in `.env` file
- See individual compose files for specific configuration options
- GPU images require NVIDIA Docker runtime and CUDA-compatible drivers
