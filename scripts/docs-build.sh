#!/usr/bin/env bash
# Build Sphinx documentation
#
# Usage:
#   ./scripts/docs-build.sh          # Build with warnings (non-fatal)
#   ./scripts/docs-build.sh --strict # Build with -W (fail on warnings)
#   ./scripts/docs-build.sh --clean  # Clean build directory first
set -euo pipefail

cd "$(dirname "$0")/.."

# Fix locale for Nix environments
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"

STRICT_FLAG=""
CLEAN=false

for arg in "$@"; do
    case "$arg" in
        --strict) STRICT_FLAG="-W" ;;
        --clean) CLEAN=true ;;
    esac
done

if [ "$CLEAN" = true ]; then
    echo "=== Cleaning build directory ==="
    rm -rf docs/_build
fi

echo "=== Building documentation ==="
uv run sphinx-build -b html docs docs/_build/html $STRICT_FLAG --keep-going

echo ""
echo "Built: docs/_build/html/index.html"
echo "Open with: open docs/_build/html/index.html"
