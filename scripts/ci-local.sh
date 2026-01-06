#!/usr/bin/env bash
# ci-local.sh - Canonical local CI gate for slower-whisper
#
# This script runs the same checks that CI would run.
# If this passes locally, you're safe to merge.
#
# Usage:
#   ./scripts/ci-local.sh        # Full suite
#   ./scripts/ci-local.sh fast   # Quick checks only
#
# Prerequisites:
#   - Run from inside a nix develop shell, OR
#   - Have uv and python3.12 available
#
set -euo pipefail

MODE="${1:-full}"

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  slower-whisper local CI ($MODE mode)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Track failures
FAILED_CHECKS=()

# Helper: run a check and track result
run_check() {
    local name="$1"
    shift
    echo -e "${BLUE}▶ $name...${NC}"
    if "$@"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ $name failed${NC}"
        echo ""
        FAILED_CHECKS+=("$name")
        return 1
    fi
}

# Ensure dependencies are installed
echo -e "${BLUE}▶ Checking Python dependencies...${NC}"
if [ ! -d ".venv" ]; then
    echo "No .venv found. Installing dependencies..."
    uv sync --frozen --extra full --extra diarization --extra dev
else
    echo "✓ .venv exists"
fi
echo ""

# Check 1: Pre-commit hooks (includes ruff lint + format)
run_check "Pre-commit hooks" \
    uv run pre-commit run --all-files

# Check 2: Type-check (typed surface)
run_check "Type-check (mypy)" \
    uv run mypy \
        transcription/ \
        tests/test_llm_utils.py \
        tests/test_writers.py \
        tests/test_turn_helpers.py \
        tests/test_audio_state_schema.py

# Check 3: Fast tests
run_check "Fast tests (pytest -m 'not slow and not heavy')" \
    uv run pytest -q -m "not slow and not heavy"

# Check 4: Quick verification
run_check "Verification suite" \
    uv run slower-whisper-verify --quick

# Stop here if mode=fast
if [ "$MODE" = "fast" ]; then
    echo -e "${YELLOW}⚡ Fast mode: skipping nix checks${NC}"
    echo ""
else
    # Check 5: Nix flake check (lint + format via nixpkgs ruff)
    # Use nix-clean if available (inside devshell), otherwise plain nix
    if type nix-clean &>/dev/null; then
        NIX_CMD="nix-clean"
    else
        NIX_CMD="nix"
    fi

    run_check "Nix flake check" \
        $NIX_CMD flake check

    run_check "Nix verify app" \
        $NIX_CMD run .#verify -- --quick
fi

# Summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All local CI checks passed!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ ${#FAILED_CHECKS[@]} check(s) failed:${NC}"
    for check in "${FAILED_CHECKS[@]}"; do
        echo -e "  ${RED}✗ $check${NC}"
    done
    echo ""
    exit 1
fi
