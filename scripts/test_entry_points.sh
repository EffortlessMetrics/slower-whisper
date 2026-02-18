#!/bin/bash
# Test script to verify entry points after installation
# Usage: bash test_entry_points.sh

set -e  # Exit on error

echo "=========================================="
echo "Entry Points Installation Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Helper function for tests
test_command() {
    local test_name="$1"
    local command="$2"

    echo -n "Testing: $test_name ... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((FAILED++))
        return 1
    fi
}

# Test 1: Check if package is installed
echo "📦 Package Installation"
echo "----------------------------------------"
test_command "Package installed" "pip show slower-whisper"
echo ""

# Test 2: Check if entry point commands exist
echo "🔧 Entry Point Commands"
echo "----------------------------------------"

if command -v slower-whisper &> /dev/null; then
    echo -e "${GREEN}✅ PASS${NC} - slower-whisper command found"
    ((PASSED++))
    SLOWER_WHISPER_PATH=$(which slower-whisper)
    echo "  Location: $SLOWER_WHISPER_PATH"
else
    echo -e "${RED}❌ FAIL${NC} - slower-whisper command not found"
    ((FAILED++))
fi

if command -v slower-whisper-enrich &> /dev/null; then
    echo -e "${GREEN}✅ PASS${NC} - slower-whisper-enrich command found"
    ((PASSED++))
    ENRICH_PATH=$(which slower-whisper-enrich)
    echo "  Location: $ENRICH_PATH"
else
    echo -e "${RED}❌ FAIL${NC} - slower-whisper-enrich command not found"
    ((FAILED++))
fi

echo ""

# Test 3: Check help output
echo "📖 Help Output"
echo "----------------------------------------"
test_command "slower-whisper --help" "slower-whisper --help"
echo ""

# Test 4: Verify module imports
echo "🐍 Python Module Imports"
echo "----------------------------------------"
test_command "Import slower_whisper.pipeline.cli" "python3 -c 'from slower_whisper.pipeline.cli import main'"
echo ""

# Test 5: Check dependencies
echo "📚 Dependencies"
echo "----------------------------------------"

# Core dependencies (always required)
test_command "faster-whisper installed" "python3 -c 'import faster_whisper'"

# Optional dependencies (for enrichment)
echo -e "${YELLOW}Note: Optional dependencies (skip if not installed)${NC}"
python3 -c 'import soundfile' 2>/dev/null && echo -e "${GREEN}✅${NC} soundfile (optional)" || echo -e "${YELLOW}⚠️${NC} soundfile (optional - not installed)"
python3 -c 'import librosa' 2>/dev/null && echo -e "${GREEN}✅${NC} librosa (optional)" || echo -e "${YELLOW}⚠️${NC} librosa (optional - not installed)"
python3 -c 'import parselmouth' 2>/dev/null && echo -e "${GREEN}✅${NC} parselmouth (optional)" || echo -e "${YELLOW}⚠️${NC} parselmouth (optional - not installed)"
python3 -c 'import torch' 2>/dev/null && echo -e "${GREEN}✅${NC} torch (optional)" || echo -e "${YELLOW}⚠️${NC} torch (optional - not installed)"
python3 -c 'import transformers' 2>/dev/null && echo -e "${GREEN}✅${NC} transformers (optional)" || echo -e "${YELLOW}⚠️${NC} transformers (optional - not installed)"

echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    echo "You can now use the following commands:"
    echo "  slower-whisper [OPTIONS]"
    echo "  slower-whisper-enrich [OPTIONS]"
    echo ""
    echo "Run with --help for usage information:"
    echo "  slower-whisper --help"
    echo "  slower-whisper-enrich --help"
    exit 0
else
    echo -e "${RED}❌ Some tests failed.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure the package is installed:"
    echo "   pip install -e ."
    echo "   or: uv sync"
    echo ""
    echo "2. Verify you're in the correct virtual environment:"
    echo "   which python3"
    echo "   pip list | grep slower-whisper"
    echo ""
    echo "3. Try reinstalling:"
    echo "   pip install -e . --force-reinstall"
    echo "   or: uv sync --reinstall-package slower-whisper"
    exit 1
fi
