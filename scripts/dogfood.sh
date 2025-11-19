#!/bin/bash
#
# Dogfood workflow: Run complete diarization + LLM test on a sample file
#
# Usage:
#   ./scripts/dogfood.sh raw_audio/sample.wav
#   ./scripts/dogfood.sh raw_audio/sample.wav --skip-transcribe  # Use existing JSON
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <audio_file> [--skip-transcribe]"
    echo ""
    echo "Example:"
    echo "  $0 raw_audio/mini_diarization_test.wav"
    echo "  $0 raw_audio/mini_diarization_test.wav --skip-transcribe  # Use existing JSON"
    exit 1
fi

AUDIO_FILE="$1"
SKIP_TRANSCRIBE=false

if [ "$2" == "--skip-transcribe" ]; then
    SKIP_TRANSCRIBE=true
fi

# Extract filename without path and extension
BASENAME=$(basename "$AUDIO_FILE" .wav)
JSON_FILE="whisper_json/${BASENAME}.json"

echo -e "${BLUE}=== Dogfood Workflow: ${BASENAME} ===${NC}\n"

# Check audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}Error: Audio file not found: $AUDIO_FILE${NC}"
    exit 1
fi

# Check environment variables
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set${NC}"
    echo "Set with: export HF_TOKEN=hf_..."
    echo "Get token from: https://huggingface.co/settings/tokens"
    echo ""
fi

# Step 1: Check model cache
echo -e "${BLUE}Step 1: Checking model cache...${NC}"
uv run python scripts/check_model_cache.py | grep -E "(✓|✗)" | head -10
echo ""

# Step 2: Transcribe with diarization (unless skipped)
if [ "$SKIP_TRANSCRIBE" = true ]; then
    echo -e "${BLUE}Step 2: Skipping transcription (using existing JSON)${NC}"
    if [ ! -f "$JSON_FILE" ]; then
        echo -e "${RED}Error: JSON file not found: $JSON_FILE${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}Step 2: Transcribing with diarization...${NC}"
    echo "Command: uv run slower-whisper transcribe --enable-diarization --min-speakers 2 --max-speakers 2"
    echo ""

    uv run slower-whisper transcribe \
        --enable-diarization \
        --min-speakers 2 \
        --max-speakers 2

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Transcription failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Transcription complete${NC}\n"
fi

# Step 3: Generate diarization stats
echo -e "${BLUE}Step 3: Diarization statistics${NC}"
uv run python scripts/diarization_stats.py "$JSON_FILE"

# Step 4: Show JSON preview
echo -e "${BLUE}Step 4: JSON preview (first 3 segments with speakers)${NC}"
jq '.segments[0:3] | map({id, start, end, speaker: .speaker.id, text})' "$JSON_FILE"
echo ""

# Step 5: LLM integration (optional, requires API key)
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "${BLUE}Step 5: Testing LLM integration...${NC}"
    echo "Command: python examples/llm_integration/summarize_with_diarization.py $JSON_FILE"
    echo ""

    python examples/llm_integration/summarize_with_diarization.py "$JSON_FILE"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ LLM integration test complete${NC}\n"
    else
        echo -e "${YELLOW}⚠ LLM integration failed (check API key)${NC}\n"
    fi
else
    echo -e "${BLUE}Step 5: Skipping LLM integration (ANTHROPIC_API_KEY not set)${NC}"
    echo "To test LLM integration:"
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    echo "  python examples/llm_integration/summarize_with_diarization.py $JSON_FILE"
    echo ""
fi

# Summary
echo -e "${GREEN}=== Dogfood Complete ===${NC}\n"
echo "Next steps:"
echo "  1. Review stats output above"
echo "  2. Inspect JSON: jq . $JSON_FILE | less"
echo "  3. Record findings in DOGFOOD_NOTES.md"
echo ""
echo "To re-run LLM test only:"
echo "  python examples/llm_integration/summarize_with_diarization.py $JSON_FILE"
echo ""
echo "To re-run with different settings:"
echo "  $0 $AUDIO_FILE --skip-transcribe  # Use existing JSON, skip re-transcribing"
