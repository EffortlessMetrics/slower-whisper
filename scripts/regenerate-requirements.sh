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

echo "Exporting requirements-enrich.txt (basic enrichment)..."
uv export --frozen --no-dev --extra enrich-basic -o requirements-enrich.txt

echo "Exporting requirements.txt (full extras)..."
uv export --frozen --no-dev --extra full -o requirements.txt

echo "Exporting requirements-dev.txt (dev extras, includes full)..."
uv export --frozen --extra dev -o requirements-dev.txt

echo "Done. Requirements files regenerated from uv.lock."
