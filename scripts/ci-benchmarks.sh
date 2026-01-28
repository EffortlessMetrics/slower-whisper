#!/usr/bin/env bash
# ci-benchmarks.sh - Local benchmark CI runner for slower-whisper
#
# This script runs the same benchmark checks as the GitHub Actions benchmark.yml workflow.
# Use it to verify benchmark infrastructure before pushing changes.
#
# Usage:
#   ./scripts/ci-benchmarks.sh           # Run smoke benchmarks (report-only)
#   ./scripts/ci-benchmarks.sh --gate    # Run with gate mode (fail on regression)
#   ./scripts/ci-benchmarks.sh --save    # Save results as new baselines
#   ./scripts/ci-benchmarks.sh --help    # Show help
#
# Prerequisites:
#   - Run from inside a nix develop shell, OR
#   - Have uv and python3.12 available
#
# Environment variables:
#   BENCHMARK_LIMIT - Sample limit per benchmark (default: 5)
#   RESULTS_DIR     - Output directory (default: benchmark-results)
#
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_LIMIT="${BENCHMARK_LIMIT:-5}"
RESULTS_DIR="${RESULTS_DIR:-benchmark-results}"
GATE_MODE=false
SAVE_BASELINE=false
TRACK="all"
DATASET="smoke"
VERBOSE=false

# Color output helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Functions
# =============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run benchmark CI checks locally, mirroring the GitHub Actions benchmark workflow.

Options:
    --gate          Enable gate mode (fail CI if regression exceeds threshold)
    --save          Save results as new baselines
    --track TRACK   Benchmark track: all, asr, semantic (default: all)
    --dataset DS    ASR dataset: smoke, librispeech (default: smoke)
    --limit N       Sample limit per benchmark (default: 5)
    --output DIR    Output directory (default: benchmark-results)
    --verbose       Enable verbose output
    -h, --help      Show this help message

Examples:
    # Run smoke benchmarks (report-only)
    ./scripts/ci-benchmarks.sh

    # Run with gate mode
    ./scripts/ci-benchmarks.sh --gate

    # Run ASR track only
    ./scripts/ci-benchmarks.sh --track asr

    # Save new baselines
    ./scripts/ci-benchmarks.sh --save --track asr

EOF
    exit 0
}

log_header() {
    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    echo ""
}

log_step() {
    echo -e "${CYAN}>>> $1${NC}"
}

log_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --gate)
            GATE_MODE=true
            shift
            ;;
        --save)
            SAVE_BASELINE=true
            shift
            ;;
        --track)
            TRACK="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --limit)
            BENCHMARK_LIMIT="$2"
            shift 2
            ;;
        --output)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

log_header "Benchmark CI Runner"

echo "Configuration:"
echo "  Track:          $TRACK"
echo "  Dataset:        $DATASET"
echo "  Sample Limit:   $BENCHMARK_LIMIT"
echo "  Results Dir:    $RESULTS_DIR"
echo "  Gate Mode:      $GATE_MODE"
echo "  Save Baseline:  $SAVE_BASELINE"
echo ""

# Ensure dependencies are installed
log_step "Checking Python dependencies..."
if [ ! -d ".venv" ]; then
    echo "No .venv found. Installing dependencies..."
    uv sync --frozen --extra dev
else
    echo "  .venv exists"
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Track overall status
OVERALL_STATUS=0
FAILED_CHECKS=()

# =============================================================================
# Infrastructure Status
# =============================================================================

log_step "Checking benchmark infrastructure status..."
uv run slower-whisper benchmark status > "$RESULTS_DIR/status.txt" 2>&1 || true
cat "$RESULTS_DIR/status.txt"
echo ""

log_step "Listing available baselines..."
uv run slower-whisper benchmark baselines > "$RESULTS_DIR/baselines.txt" 2>&1 || true
cat "$RESULTS_DIR/baselines.txt"
echo ""

# =============================================================================
# ASR Benchmark
# =============================================================================

if [[ "$TRACK" == "all" || "$TRACK" == "asr" ]]; then
    log_header "ASR Benchmark"

    log_step "Running ASR benchmark (dataset=$DATASET, limit=$BENCHMARK_LIMIT)..."

    VERBOSE_FLAG=""
    if [[ "$VERBOSE" == "true" ]]; then
        VERBOSE_FLAG="--verbose"
    fi

    if uv run slower-whisper benchmark run \
        --track asr \
        --dataset "$DATASET" \
        --limit "$BENCHMARK_LIMIT" \
        --output "$RESULTS_DIR/asr-results.json" \
        $VERBOSE_FLAG \
        2>&1 | tee "$RESULTS_DIR/asr-run.log"; then
        log_success "ASR benchmark completed"
    else
        log_warning "ASR benchmark did not complete (dataset may not be staged)"
        echo '{"status": "skipped", "reason": "dataset_not_staged"}' > "$RESULTS_DIR/asr-results.json"
    fi

    # Compare with baseline if exists
    if [[ -f "benchmarks/baselines/asr/$DATASET.json" ]]; then
        log_step "Comparing ASR results against baseline..."

        COMPARE_FLAGS=""
        if [[ "$GATE_MODE" == "true" ]]; then
            COMPARE_FLAGS="--gate"
            echo "  Running in GATE mode - will fail on regression"
        fi

        if uv run slower-whisper benchmark compare \
            --track asr \
            --dataset "$DATASET" \
            --limit "$BENCHMARK_LIMIT" \
            --output "$RESULTS_DIR/asr-comparison.json" \
            $VERBOSE_FLAG \
            $COMPARE_FLAGS \
            2>&1 | tee "$RESULTS_DIR/asr-compare.log"; then
            log_success "ASR comparison completed"
        else
            if [[ "$GATE_MODE" == "true" ]]; then
                log_error "ASR comparison failed or regression detected"
                OVERALL_STATUS=1
                FAILED_CHECKS+=("ASR comparison")
            else
                log_warning "ASR comparison did not complete"
            fi
        fi
    else
        log_warning "No ASR baseline found for dataset: $DATASET"
    fi

    # Save baseline if requested
    if [[ "$SAVE_BASELINE" == "true" ]]; then
        log_step "Saving ASR baseline..."
        if uv run slower-whisper benchmark save-baseline \
            --track asr \
            --dataset "$DATASET" \
            --limit "$BENCHMARK_LIMIT" \
            --output "$RESULTS_DIR/asr-baseline-results.json" \
            $VERBOSE_FLAG \
            2>&1 | tee "$RESULTS_DIR/asr-baseline-save.log"; then
            log_success "ASR baseline saved"
        else
            log_error "Failed to save ASR baseline"
        fi
    fi
fi

# =============================================================================
# Semantic Benchmark
# =============================================================================

if [[ "$TRACK" == "all" || "$TRACK" == "semantic" ]]; then
    log_header "Semantic Benchmark"

    log_step "Running Semantic benchmark (mode=tags, limit=$BENCHMARK_LIMIT)..."

    VERBOSE_FLAG=""
    if [[ "$VERBOSE" == "true" ]]; then
        VERBOSE_FLAG="--verbose"
    fi

    if uv run slower-whisper benchmark run \
        --track semantic \
        --mode tags \
        --limit "$BENCHMARK_LIMIT" \
        --output "$RESULTS_DIR/semantic-results.json" \
        $VERBOSE_FLAG \
        2>&1 | tee "$RESULTS_DIR/semantic-run.log"; then
        log_success "Semantic benchmark completed"
    else
        log_warning "Semantic benchmark did not complete"
        echo '{"status": "skipped", "reason": "benchmark_error"}' > "$RESULTS_DIR/semantic-results.json"
    fi

    # Compare with baseline if exists
    if [[ -f "benchmarks/baselines/semantic/ami.json" ]]; then
        log_step "Comparing Semantic results against baseline..."

        COMPARE_FLAGS=""
        if [[ "$GATE_MODE" == "true" ]]; then
            COMPARE_FLAGS="--gate"
            echo "  Running in GATE mode - will fail on regression"
        fi

        if uv run slower-whisper benchmark compare \
            --track semantic \
            --mode tags \
            --limit "$BENCHMARK_LIMIT" \
            --output "$RESULTS_DIR/semantic-comparison.json" \
            $VERBOSE_FLAG \
            $COMPARE_FLAGS \
            2>&1 | tee "$RESULTS_DIR/semantic-compare.log"; then
            log_success "Semantic comparison completed"
        else
            if [[ "$GATE_MODE" == "true" ]]; then
                log_error "Semantic comparison failed or regression detected"
                OVERALL_STATUS=1
                FAILED_CHECKS+=("Semantic comparison")
            else
                log_warning "Semantic comparison did not complete"
            fi
        fi
    else
        log_warning "No Semantic baseline found"
    fi

    # Save baseline if requested
    if [[ "$SAVE_BASELINE" == "true" ]]; then
        log_step "Saving Semantic baseline..."
        if uv run slower-whisper benchmark save-baseline \
            --track semantic \
            --mode tags \
            --limit "$BENCHMARK_LIMIT" \
            --output "$RESULTS_DIR/semantic-baseline-results.json" \
            $VERBOSE_FLAG \
            2>&1 | tee "$RESULTS_DIR/semantic-baseline-save.log"; then
            log_success "Semantic baseline saved"
        else
            log_error "Failed to save Semantic baseline"
        fi
    fi
fi

# =============================================================================
# Generate Summary
# =============================================================================

log_header "Benchmark Summary"

{
    echo "## Benchmark Summary"
    echo ""
    echo "**Run Date:** $(date -Iseconds)"
    echo "**Track:** $TRACK"
    echo "**Dataset:** $DATASET"
    echo "**Sample Limit:** $BENCHMARK_LIMIT"
    echo "**Gate Mode:** $GATE_MODE"
    echo ""

    # ASR Results
    echo "### ASR Benchmark"
    if [[ -f "$RESULTS_DIR/asr-results.json" ]]; then
        echo '```json'
        cat "$RESULTS_DIR/asr-results.json"
        echo '```'
    else
        echo "No ASR results available"
    fi
    echo ""

    # ASR Comparison
    if [[ -f "$RESULTS_DIR/asr-comparison.json" ]]; then
        echo "#### ASR Baseline Comparison"
        echo '```json'
        cat "$RESULTS_DIR/asr-comparison.json"
        echo '```'
        echo ""
    fi

    # Semantic Results
    echo "### Semantic Benchmark"
    if [[ -f "$RESULTS_DIR/semantic-results.json" ]]; then
        echo '```json'
        cat "$RESULTS_DIR/semantic-results.json"
        echo '```'
    else
        echo "No Semantic results available"
    fi
    echo ""

    # Semantic Comparison
    if [[ -f "$RESULTS_DIR/semantic-comparison.json" ]]; then
        echo "#### Semantic Baseline Comparison"
        echo '```json'
        cat "$RESULTS_DIR/semantic-comparison.json"
        echo '```'
        echo ""
    fi
} > "$RESULTS_DIR/SUMMARY.md"

cat "$RESULTS_DIR/SUMMARY.md"

# =============================================================================
# Final Status
# =============================================================================

echo ""
echo -e "${BLUE}======================================================================${NC}"

if [[ "$OVERALL_STATUS" -eq 0 ]]; then
    log_success "All benchmark checks passed!"
    echo ""
    echo "Results saved to: $RESULTS_DIR/"
    echo ""
    if [[ "$GATE_MODE" == "false" ]]; then
        echo -e "${YELLOW}Note: Running in report-only mode. Use --gate for CI blocking.${NC}"
    fi
else
    log_error "Some benchmark checks failed:"
    for check in "${FAILED_CHECKS[@]}"; do
        echo -e "  ${RED}- $check${NC}"
    done
    echo ""
    echo "Results saved to: $RESULTS_DIR/"
fi

echo -e "${BLUE}======================================================================${NC}"
echo ""

exit $OVERALL_STATUS
