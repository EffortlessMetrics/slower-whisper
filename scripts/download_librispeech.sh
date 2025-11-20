#!/bin/bash
# Download and setup LibriSpeech dev-clean dataset for benchmarking
#
# Usage: ./scripts/download_librispeech.sh [--subset SUBSET]
#
# Options:
#   --subset SUBSET    Which subset to download (default: dev-clean)
#                      Options: dev-clean, test-clean, dev-other, test-other

set -e

SUBSET="${1:-dev-clean}"
LIB_ROOT="${HOME}/.cache/slower-whisper/benchmarks/librispeech"

echo "=========================================="
echo "LibriSpeech Dataset Download"
echo "=========================================="
echo "Subset: ${SUBSET}"
echo "Target: ${LIB_ROOT}"
echo ""

# Create directory
mkdir -p "${LIB_ROOT}"
cd "${LIB_ROOT}"

# Check if already downloaded
if [ -d "LibriSpeech/${SUBSET}" ]; then
    echo "✓ LibriSpeech ${SUBSET} already exists at:"
    echo "  ${LIB_ROOT}/LibriSpeech/${SUBSET}"
    echo ""
    echo "Speaker directories:"
    ls "LibriSpeech/${SUBSET}" | head -5
    echo "  ... ($(ls LibriSpeech/${SUBSET} | wc -l) total speakers)"
    exit 0
fi

# Download if tarball doesn't exist
if [ ! -f "${SUBSET}.tar.gz" ]; then
    echo "Downloading ${SUBSET}.tar.gz (~337 MB)..."
    echo "Source: https://www.openslr.org/resources/12/${SUBSET}.tar.gz"
    echo ""

    wget --show-progress \
         --no-clobber \
         "https://www.openslr.org/resources/12/${SUBSET}.tar.gz"

    echo ""
    echo "✓ Download complete"
else
    echo "✓ Tarball already exists: ${SUBSET}.tar.gz"
fi

# Extract
if [ ! -d "LibriSpeech/${SUBSET}" ]; then
    echo ""
    echo "Extracting ${SUBSET}.tar.gz..."
    tar -xzf "${SUBSET}.tar.gz"
    echo "✓ Extraction complete"
fi

# Verify structure
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
echo "Directory: ${LIB_ROOT}/LibriSpeech/${SUBSET}"
echo ""
echo "Speaker directories (first 10):"
ls "LibriSpeech/${SUBSET}" | head -10

TOTAL_SPEAKERS=$(ls "LibriSpeech/${SUBSET}" | wc -l)
echo ""
echo "Total speakers: ${TOTAL_SPEAKERS}"

# Count audio files
TOTAL_FILES=$(find "LibriSpeech/${SUBSET}" -name "*.flac" | wc -l)
echo "Total .flac files: ${TOTAL_FILES}"

echo ""
echo "=========================================="
echo "✓ LibriSpeech ${SUBSET} ready for use!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run smoke test (5 samples):"
echo "     uv run python benchmarks/eval_asr_diarization.py \\"
echo "       --dataset librispeech \\"
echo "       --n 5 \\"
echo "       --model base \\"
echo "       --device cpu \\"
echo "       --output benchmarks/results/asr_librispeech_dev_clean_5.json"
echo ""
echo "  2. View results:"
echo "     cat benchmarks/results/asr_librispeech_dev_clean_5.json | jq '.aggregate, .samples'"
