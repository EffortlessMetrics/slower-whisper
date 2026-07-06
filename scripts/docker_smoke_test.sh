#!/usr/bin/env bash
# Smoke test Docker images to verify they build and run
#
# Usage: ./scripts/docker_smoke_test.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐋 Docker Image Smoke Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# CPU image
echo "Building CPU image..."
docker build -t slower-whisper:test-cpu -f config/Dockerfile . --quiet

echo "✅ CPU image built"
echo "Testing CLI in CPU image..."
docker run --rm slower-whisper:test-cpu --help | grep -q "transcribe"
echo "✅ CLI works in CPU image"
echo ""

# GPU image (build only, requires NVIDIA runtime to run)
echo "Building GPU image..."
docker build -t slower-whisper:test-gpu -f config/Dockerfile.gpu . --quiet
echo "✅ GPU image built (runtime test skipped - requires NVIDIA Docker)"
echo ""

# API image
echo "Building API image..."
docker build -t slower-whisper:test-api -f config/Dockerfile.api . --quiet
echo "✅ API image built"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All Docker images smoke tested successfully"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Images ready:"
echo "  - slower-whisper:test-cpu"
echo "  - slower-whisper:test-gpu"
echo "  - slower-whisper:test-api"
echo ""
echo "Test with:"
echo "  docker run --rm slower-whisper:test-cpu --version"
