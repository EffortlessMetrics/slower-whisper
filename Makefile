.PHONY: help verify verify-quick test lint format clean install

# Default target: show help
help:
	@echo "slower-whisper development commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install all dependencies (dev mode)"
	@echo ""
	@echo "Verification (required before pushing):"
	@echo "  make verify-quick   Run quick verification (code + tests + BDD + API)"
	@echo "  make verify         Run full verification (includes Docker + K8s)"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run tests"
	@echo "  make lint           Check code quality (ruff)"
	@echo "  make format         Auto-format code (ruff)"
	@echo "  make clean          Clean build artifacts and caches"

# Installation
install:
	uv sync --extra dev

# Quick verification (REQUIRED before pushing)
verify-quick:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Running quick verification..."
	@echo "This verifies: code quality + tests + BDD + API BDD"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	uv run slower-whisper-verify --quick

# Full verification (recommended before PRs)
verify:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Running full verification..."
	@echo "This includes: Docker builds + K8s validation"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	uv run slower-whisper-verify

# Development commands
test:
	uv run pytest -v

lint:
	uv run ruff check transcription/ tests/

format:
	uv run ruff format transcription/ tests/

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build slower_whisper.egg-info .benchmarks
	find . -type d -name __pycache__ -not -path "./.venv/*" -not -path "./.direnv/*" -not -path "./.cache/*" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./.venv/*" -not -path "./.direnv/*" -not -path "./.cache/*" -not -path "./.git/*" -delete
