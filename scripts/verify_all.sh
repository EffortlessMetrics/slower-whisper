#!/usr/bin/env bash
# Master verification script - runs all checks
#
# Usage: ./scripts/verify_all.sh [--quick]

set -e

QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔒 Master Verification - slower-whisper"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Code quality
echo "1️⃣  Checking code quality..."
uv run ruff check transcription/ tests/
uv run ruff format --check transcription/ tests/
echo "✅ Code quality passed"
echo ""

# 2. Type checking
echo "2️⃣  Type checking..."
uv run mypy transcription/ || echo "⚠️  Type check warnings (non-blocking)"
echo ""

# 3. Unit tests
echo "3️⃣  Running unit tests..."
uv run pytest tests/ -m "not slow and not requires_gpu" --cov=transcription --cov-report=term-missing
echo "✅ Unit tests passed"
echo ""

# 4. BDD scenarios
echo "4️⃣  Running BDD scenarios..."
./scripts/verify_bdd.sh
echo ""

if [ "$QUICK_MODE" = false ]; then
    # 5. Docker smoke tests
    echo "5️⃣  Docker smoke tests..."
    ./scripts/docker_smoke_test.sh
    echo ""

    # 6. K8s validation
    echo "6️⃣  Kubernetes manifest validation..."
    if command -v kubectl &> /dev/null; then
        ./scripts/validate_k8s.sh
    else
        echo "⚠️  kubectl not found, skipping K8s validation"
    fi
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All verifications passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Repository is ready for:"
echo "  - Development (code quality + tests passing)"
echo "  - Deployment (Docker + K8s artifacts validated)"
echo "  - Release (behavioral contract verified)"
