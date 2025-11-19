#!/bin/bash
#
# Dogfood workflow: Run complete diarization + LLM test on a sample
#
# Usage:
#   ./scripts/dogfood.sh --sample synthetic
#   ./scripts/dogfood.sh --sample mini-diarization
#   ./scripts/dogfood.sh --file raw_audio/custom.wav
#   ./scripts/dogfood.sh --file raw_audio/custom.wav --skip-transcribe
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
MODE=""
AUDIO_FILE=""
SKIP_TRANSCRIBE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample)
            MODE="sample"
            SAMPLE_NAME="$2"
            shift 2
            ;;
        --file)
            MODE="file"
            AUDIO_FILE="$2"
            shift 2
            ;;
        --skip-transcribe)
            SKIP_TRANSCRIBE=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            echo "Usage:"
            echo "  $0 --sample <name>              # Use built-in sample (synthetic, mini-diarization)"
            echo "  $0 --file <path>                # Use custom audio file"
            echo "  $0 --file <path> --skip-transcribe  # Use existing JSON"
            echo ""
            echo "Examples:"
            echo "  $0 --sample synthetic           # Generate and use synthetic 2-speaker audio"
            echo "  $0 --sample mini-diarization    # Use cached mini-diarization dataset"
            echo "  $0 --file raw_audio/my.wav      # Custom file"
            exit 1
            ;;
    esac
done

# Resolve audio file based on mode
if [ "$MODE" == "sample" ]; then
    case "$SAMPLE_NAME" in
        synthetic)
            echo -e "${BLUE}Preparing synthetic 2-speaker sample...${NC}"
            uv run slower-whisper samples generate
            AUDIO_FILE="raw_audio/synthetic_2speaker.wav"
            ;;
        mini-diarization)
            echo -e "${BLUE}Preparing mini-diarization sample...${NC}"
            uv run slower-whisper samples copy mini_diarization
            # Get the actual filename from samples CLI
            AUDIO_FILE="raw_audio/mini_diarization_test.wav"
            ;;
        *)
            echo -e "${RED}Unknown sample: $SAMPLE_NAME${NC}"
            echo "Available samples: synthetic, mini-diarization"
            echo "Use 'slower-whisper samples list' to see details"
            exit 1
            ;;
    esac
elif [ "$MODE" == "file" ]; then
    # User provided a custom file path
    if [ ! -f "$AUDIO_FILE" ]; then
        echo -e "${RED}Error: Audio file not found: $AUDIO_FILE${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: Must specify --sample or --file${NC}"
    echo "Use --help or run without arguments for usage info"
    exit 1
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
if [ -z "$HF_TOKEN" ] && [ "$MODE" == "sample" ] && [ "$SAMPLE_NAME" != "synthetic" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set${NC}"
    echo "Required for diarization models"
    echo "Set with: export HF_TOKEN=hf_..."
    echo "Get token from: https://huggingface.co/settings/tokens"
    echo ""
    echo -e "${YELLOW}Also ensure you've accepted pyannote model terms:${NC}"
    echo "  https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "  https://huggingface.co/pyannote/segmentation-3.0"
    echo ""
fi

# Step 1: Check model cache
echo -e "${BLUE}Step 1: Checking model cache...${NC}"
uv run python scripts/check_model_cache.py
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
echo ""

# Step 4: Show JSON preview (if jq is available)
if command -v jq &> /dev/null; then
    echo -e "${BLUE}Step 4: JSON preview (first 3 segments with speakers)${NC}"
    jq '.segments[0:3] | map({id, start, end, speaker, text})' "$JSON_FILE" 2>/dev/null || true
    echo ""
fi

# Step 5: LLM integration (optional, requires API key)
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "${BLUE}Step 5: Testing LLM integration...${NC}"
    echo "Command: python examples/llm_integration/summarize_with_diarization.py $JSON_FILE"
    echo ""

    if [ -f "examples/llm_integration/summarize_with_diarization.py" ]; then
        python examples/llm_integration/summarize_with_diarization.py "$JSON_FILE" || true
        echo -e "${GREEN}✓ LLM integration test complete${NC}\n"
    else
        echo -e "${YELLOW}⚠ LLM example script not found (skipping)${NC}\n"
    fi
else
    echo -e "${BLUE}Step 5: Skipping LLM integration (ANTHROPIC_API_KEY not set)${NC}"
    echo "To test LLM integration:"
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    if [ -f "examples/llm_integration/summarize_with_diarization.py" ]; then
        echo "  python examples/llm_integration/summarize_with_diarization.py $JSON_FILE"
    fi
    echo ""
fi

# Summary
echo -e "${GREEN}=== Dogfood Complete ===${NC}\n"
echo "Results:"
echo "  JSON:   $JSON_FILE"
echo "  Audio:  $AUDIO_FILE"
echo ""
echo "Next steps:"
echo "  1. Review stats output above"
echo "  2. Inspect JSON: jq . $JSON_FILE | less"
echo "  3. Record findings in DOGFOOD_NOTES.md"
echo ""
echo "Quick commands:"
if [ -f "examples/llm_integration/summarize_with_diarization.py" ]; then
    echo "  # Re-run LLM test"
    echo "  python examples/llm_integration/summarize_with_diarization.py $JSON_FILE"
    echo ""
fi
echo "  # Re-run with different settings"
echo "  $0 --file $AUDIO_FILE --skip-transcribe  # Use existing JSON"
echo ""
echo "  # Try another sample"
echo "  $0 --sample synthetic  # Quick synthetic test"
