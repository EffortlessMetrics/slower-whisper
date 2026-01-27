#!/usr/bin/env bash
# Build Sphinx documentation
set -euo pipefail

cd "$(dirname "$0")/.."

# Fix locale for Nix environments
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"

echo "=== Building documentation ==="
uv run sphinx-build -b html docs docs/_build/html --keep-going

echo ""
echo "Built: docs/_build/html/index.html"
echo "Open with: open docs/_build/html/index.html"
