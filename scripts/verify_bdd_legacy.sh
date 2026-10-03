#!/usr/bin/env bash
# Legacy BDD verification script
#
# This script is kept for backward compatibility.
# It now delegates to the repository verifier for cross-platform support.
#
# Usage: ./scripts/verify_bdd_legacy.sh

set -e

echo "⚠️  This shell script is deprecated. Use: uv run python scripts/verify_all.py --quick"
echo ""

# Delegate to the repository verifier's BDD phase
exec uv run python -c "from scripts.verify_all import verify_bdd; verify_bdd()"
