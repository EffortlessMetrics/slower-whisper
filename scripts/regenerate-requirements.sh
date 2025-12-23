#!/usr/bin/env bash
# Regenerate requirements*.txt from uv.lock
#
# These files are derived artifacts - don't edit them by hand.
# Always run this script after modifying pyproject.toml dependencies.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Locking dependencies..."
uv lock

echo "Exporting requirements-base.txt (no extras)..."
uv export --frozen --no-dev -o requirements-base.txt

echo "Exporting requirements.txt (full extras)..."
uv export --frozen --no-dev --extra full -o requirements.txt

echo "Exporting requirements-dev.txt (dev + full extras)..."
uv export --frozen --extra dev --extra full -o requirements-dev.txt

echo "Done. Requirements files regenerated from uv.lock."
