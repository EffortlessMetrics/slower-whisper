#!/usr/bin/env bash
# Setup script for slower-whisper development environment
# Detects Nix and guides users to the appropriate setup method

set -euo pipefail

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}🚀 slower-whisper Development Environment Setup${NC}"
echo ""

# Check if Nix is installed
if command -v nix &> /dev/null; then
    echo -e "${GREEN}✅ Nix detected:${NC} $(nix --version | head -1)"
    echo ""
    echo -e "${BOLD}Recommended: Use Nix for reproducible development${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Enter Nix dev shell:"
    echo -e "     ${BLUE}nix develop${NC}"
    echo ""
    echo "  2. Install Python dependencies:"
    echo -e "     ${BLUE}uv sync --extra full --extra diarization --extra dev${NC}"
    echo ""
    echo "  3. Run local CI checks (same as GitHub Actions):"
    echo -e "     ${BLUE}nix flake check${NC}"
    echo ""
    echo -e "${BOLD}Optional: Enable direnv for automatic shell activation${NC}"
    echo "  If you have direnv installed:"
    echo -e "     ${BLUE}direnv allow${NC}"
    echo "  This will auto-activate the Nix shell when you cd into this directory."
    echo ""
    echo "See docs/DEV_ENV_NIX.md for full documentation."

else
    echo -e "${YELLOW}⚠️  Nix not detected${NC}"
    echo ""
    echo -e "${BOLD}You're using the fallback (traditional) setup method.${NC}"
    echo ""
    echo -e "${YELLOW}Warning:${NC} Traditional setup works but lacks reproducibility guarantees."
    echo "  - You may encounter environment-specific issues"
    echo "  - Your environment won't match CI or other developers' machines"
    echo "  - System dependency conflicts are possible"
    echo ""
    echo -e "${BOLD}Strongly recommended: Install Nix for a better experience${NC}"
    echo ""
    echo "To install Nix (one-time setup):"
    echo -e "  ${BLUE}sh <(curl -L https://nixos.org/nix/install) --daemon${NC}"
    echo ""
    echo "  Then enable flakes:"
    echo -e "  ${BLUE}mkdir -p ~/.config/nix${NC}"
    echo -e "  ${BLUE}echo 'experimental-features = nix-command flakes' >> ~/.config/nix/nix.conf${NC}"
    echo ""
    echo "  After installing Nix, run this script again."
    echo ""
    echo "---"
    echo ""
    echo -e "${BOLD}Continuing with traditional setup...${NC}"
    echo ""

    # Check for required system dependencies
    missing_deps=()

    if ! command -v ffmpeg &> /dev/null; then
        missing_deps+=("ffmpeg")
    fi

    if ! command -v uv &> /dev/null; then
        missing_deps+=("uv")
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${RED}❌ Missing system dependencies:${NC} ${missing_deps[*]}"
        echo ""
        echo "Install instructions:"
        echo ""

        if [[ " ${missing_deps[*]} " =~ " ffmpeg " ]]; then
            echo -e "${BOLD}ffmpeg:${NC}"
            echo "  - Ubuntu/Debian: sudo apt-get install -y ffmpeg libsndfile1"
            echo "  - macOS: brew install ffmpeg"
            echo "  - Windows: choco install ffmpeg -y"
            echo ""
        fi

        if [[ " ${missing_deps[*]} " =~ " uv " ]]; then
            echo -e "${BOLD}uv (Python package manager):${NC}"
            echo "  - Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "  - Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
            echo ""
        fi

        echo "After installing missing dependencies, run this script again."
        exit 1
    fi

    echo -e "${GREEN}✅ System dependencies detected:${NC}"
    echo "  - ffmpeg: $(ffmpeg -version | head -1 | cut -d' ' -f3)"
    echo "  - uv: $(uv --version)"
    echo ""
    echo "Next steps:"
    echo "  1. Install Python dependencies:"
    echo -e "     ${BLUE}uv sync --extra full --extra diarization --extra dev${NC}"
    echo ""
    echo "  2. Run tests:"
    echo -e "     ${BLUE}uv run pytest -m 'not slow and not heavy'${NC}"
    echo ""
    echo -e "${YELLOW}Note:${NC} You won't be able to run local CI checks without Nix."
    echo "  To run CI checks locally, install Nix and use: nix flake check"
    echo ""
fi

echo -e "${BOLD}📖 Documentation:${NC}"
echo "  - README.md - User-facing setup instructions"
echo "  - docs/DEV_ENV_NIX.md - Nix environment guide"
echo "  - docs/ARCHITECTURE.md - Technical architecture"
echo "  - CLAUDE.md - Development guide for contributors"
echo ""
echo -e "${GREEN}Setup guidance complete!${NC}"
