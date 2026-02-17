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

# Ensure binary wheels can load their native libs when .venv uses Nix Python
# outside a nix develop shell.
ensure_native_wheel_runtime() {
    local numpy_err
    local post_fix_err

    # Fast path: runtime already healthy.
    if uv run python -c "import numpy" >/dev/null 2>&1; then
        return 0
    fi

    numpy_err="$(uv run python -c "import numpy" 2>&1 >/dev/null || true)"

    # Fresh envs may not have NumPy yet; don't treat that as a native lib failure.
    if grep -q "No module named 'numpy'" <<<"$numpy_err"; then
        return 0
    fi

    # Only attempt linker path repair for known native-library signatures.
    if ! grep -Eq "(libstdc\\+\\+\\.so\\.6|libz\\.so\\.1|GLIBCXX_|CXXABI_|cannot open shared object file)" <<<"$numpy_err"; then
        return 0
    fi

    local python_lib
    local gcc_lib
    local python_store
    local zlib_store

    python_lib="$(ldd .venv/bin/python 2>/dev/null | awk '/libpython3\.[0-9]+\.so/ {print $3; exit}')"
    gcc_lib="$(dirname "$(ldd .venv/bin/python 2>/dev/null | awk '/libgcc_s.so.1/ {print $3; exit}')")"

    if [[ -z "${python_lib:-}" || -z "${gcc_lib:-}" ]]; then
        return 0
    fi

    # Interpreter derivation root in /nix/store.
    python_store="$(echo "$python_lib" | sed -E 's#/lib/libpython3\.[0-9]+\.so(\.[0-9]+)?$##')"
    if [[ ! "$python_store" =~ ^/nix/store/ ]]; then
        return 0
    fi

    if ! command -v nix-store >/dev/null 2>&1; then
        return 0
    fi

    zlib_store="$(nix-store -q --references "$python_store" 2>/dev/null | awk '/-zlib-[0-9]/ {print; exit}')"
    if [[ -z "${zlib_store:-}" || ! -d "$zlib_store/lib" ]]; then
        return 0
    fi

    export LD_LIBRARY_PATH="$gcc_lib:$zlib_store/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "✓ Auto-configured LD_LIBRARY_PATH for Nix Python wheel runtime"

    if uv run python -c "import numpy" >/dev/null 2>&1; then
        return 0
    fi

    post_fix_err="$(uv run python -c "import numpy" 2>&1 >/dev/null || true)"
    if grep -q "No module named 'numpy'" <<<"$post_fix_err"; then
        return 0
    fi

    if ! grep -Eq "(libstdc\\+\\+\\.so\\.6|libz\\.so\\.1|GLIBCXX_|CXXABI_|cannot open shared object file)" <<<"$post_fix_err"; then
        return 0
    fi

    if [[ -n "$post_fix_err" ]]; then
        echo -e "${RED}NumPy import still failing after runtime path auto-config.${NC}"
        echo "$post_fix_err"
        echo "Run inside nix develop (or use nix-clean wrappers) and retry."
        return 1
    fi

    return 0
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

if ! ensure_native_wheel_runtime; then
    FAILED_CHECKS+=("Python runtime libs")
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}✗ Local CI bootstrap failed${NC}"
    echo ""
    exit 1
fi
echo ""

# Check 0: Docs sanity (fast, runs early)
docs_sanity_check() {
    local errors=0

    # 1. Check for broken internal links in critical control surface docs only
    # These are the docs that users hit first and must stay accurate
    echo "  Checking internal markdown links in critical docs..."
    local key_docs=(
        "README.md"
        "CLAUDE.md"
        "docs/INDEX.md"
        "docs/SCHEMA.md"
        "docs/audit/README.md"
        "docs/audit/AUDIT_PATH.md"
    )

    for file in "${key_docs[@]}"; do
        [[ ! -f "$file" ]] && continue
        while IFS= read -r link; do
            # Skip external links, anchors-only, and empty
            [[ -z "$link" ]] && continue
            [[ "$link" =~ ^https?:// ]] && continue
            [[ "$link" =~ ^# ]] && continue
            [[ "$link" =~ ^mailto: ]] && continue

            # Strip anchor from link
            link_path="${link%%#*}"
            [[ -z "$link_path" ]] && continue

            # Resolve relative to file's directory
            file_dir=$(dirname "$file")
            resolved="$file_dir/$link_path"

            if [[ ! -e "$resolved" ]]; then
                echo -e "    ${RED}Broken link in $file: $link${NC}"
                ((errors++))
            fi
        done < <(grep -oP '\]\(\K[^)]+' "$file" 2>/dev/null || true)
    done

    # 2. Check for forbidden JSON keys in snippets (file_name should be file in JSON output)
    # Only check _snippets dir which should have correct schema keys
    echo "  Checking for forbidden keys in snippets..."
    if grep -rn '"file_name"' docs/_snippets/ 2>/dev/null; then
        echo -e "    ${RED}Found '\"file_name\"' in snippets (should be '\"file\"')${NC}"
        ((errors++))
    fi

    # 3. Verify generated snippets exist
    echo "  Checking generated snippets..."
    local expected_snippets=(
        "docs/_snippets/schema_example.json"
        "docs/_snippets/schema_full_example.json"
    )
    for snippet in "${expected_snippets[@]}"; do
        if [[ ! -f "$snippet" ]]; then
            echo -e "    ${RED}Missing snippet: $snippet${NC}"
            ((errors++))
        fi
    done

    if [[ $errors -gt 0 ]]; then
        echo -e "  ${RED}Found $errors doc issue(s)${NC}"
        return 1
    fi
    echo "  All docs checks passed"
    return 0
}

run_check "Docs sanity" docs_sanity_check

# Check 1: Formatting (fail early with a clear local signal)
run_check "Formatting (ruff)" \
    uv run ruff format . --check

# Check 2: Pre-commit hooks (includes ruff lint + format)
run_check "Pre-commit hooks" \
    uv run pre-commit run --all-files

# Check 3: Type-check (typed surface)
run_check "Type-check (mypy)" \
    uv run mypy \
        transcription/ \
        tests/test_llm_utils.py \
        tests/test_writers.py \
        tests/test_turn_helpers.py \
        tests/test_audio_state_schema.py

# Check 4: Fast tests
run_check "Fast tests (pytest -m 'not slow and not heavy')" \
    uv run pytest -q -m "not slow and not heavy"

# Check 5: Quick verification
run_check "Verification suite" \
    uv run slower-whisper-verify --quick

# Stop here if mode=fast
if [ "$MODE" = "fast" ]; then
    echo -e "${YELLOW}⚡ Fast mode: skipping nix checks${NC}"
    echo ""
else
    # Check 6: Nix flake check (lint + format via nixpkgs ruff)
    # Use nix-clean if available (inside devshell), otherwise plain nix
    if type nix-clean &>/dev/null; then
        NIX_CMD="nix-clean"
    else
        NIX_CMD="nix"
    fi

    # Keep Python wheel runtime fixes from contaminating Nix's own dynamic linker env.
    run_nix_clean_env() {
        env -u LD_LIBRARY_PATH -u LD_PRELOAD "$@"
    }

    run_check "Nix flake check" \
        run_nix_clean_env "$NIX_CMD" flake check

    run_check "Nix verify app" \
        run_nix_clean_env "$NIX_CMD" run .#verify -- --quick
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
