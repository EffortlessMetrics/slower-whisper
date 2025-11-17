#!/usr/bin/env bash
# Legacy BDD verification script
#
# This script is kept for backward compatibility.
# It now delegates to the Python CLI for cross-platform support.
#
# Usage: ./scripts/verify_bdd_legacy.sh

set -e

echo "⚠️  This shell script is deprecated. Use: uv run slower-whisper-verify"
echo ""

# Delegate to Python CLI
exec uv run python -c "from scripts.verify_all import verify_bdd; verify_bdd()"
