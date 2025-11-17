#!/bin/bash
# Build and push slower-whisper Docker image to registry
# Usage: ./build-and-push.sh [registry] [tag]

set -e

# Configuration
REGISTRY=${1:-"your-registry"}
TAG=${2:-"latest"}
IMAGE_NAME="slower-whisper"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building slower-whisper Docker image...${NC}"
echo "Image: ${FULL_IMAGE}"
echo ""

# Build image
echo -e "${YELLOW}Step 1: Building image...${NC}"
docker build \
  -t "${IMAGE_NAME}:${TAG}" \
  -t "${FULL_IMAGE}" \
  -f Dockerfile \
  .. # Build context is parent directory

if [ $? -ne 0 ]; then
  echo -e "${RED}Build failed!${NC}"
  exit 1
fi

echo -e "${GREEN}Build successful!${NC}"
echo ""

# Test image
echo -e "${YELLOW}Step 2: Testing image...${NC}"
docker run --rm "${IMAGE_NAME}:${TAG}" slower-whisper --help

if [ $? -ne 0 ]; then
  echo -e "${RED}Image test failed!${NC}"
  exit 1
fi

echo -e "${GREEN}Image test passed!${NC}"
echo ""

# Show image size
echo -e "${YELLOW}Image size:${NC}"
docker images "${IMAGE_NAME}:${TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo ""

# Ask for confirmation before pushing
read -p "Push image to ${REGISTRY}? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo -e "${YELLOW}Step 3: Pushing image to registry...${NC}"

  # Login to registry (if needed)
  # docker login "${REGISTRY}"

  docker push "${FULL_IMAGE}"

  if [ $? -ne 0 ]; then
    echo -e "${RED}Push failed!${NC}"
    exit 1
  fi

  echo -e "${GREEN}Push successful!${NC}"
  echo ""
  echo "Image available at: ${FULL_IMAGE}"
  echo ""
  echo "To deploy to Kubernetes:"
  echo "  1. Update k8s/deployment.yaml with image: ${FULL_IMAGE}"
  echo "  2. Run: kubectl apply -f k8s/deployment.yaml"
else
  echo -e "${YELLOW}Skipping push.${NC}"
  echo "To push manually later:"
  echo "  docker push ${FULL_IMAGE}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
