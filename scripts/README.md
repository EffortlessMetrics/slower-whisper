# Development Scripts

This directory contains development and maintenance scripts used during development and testing of slower-whisper.

## Scripts

### test_entry_points.py / test_entry_points.sh

Verify that CLI entry points are correctly installed and functional after package installation.

**Usage:**
```bash
# Python version
uv run python scripts/test_entry_points.py

# Shell version
./scripts/test_entry_points.sh
```

Tests:
- Package importability
- CLI command availability (`slower-whisper`)
- Version string consistency
- Help output correctness

### verify_code_examples.py

Extract and validate Python code examples from documentation files (README.md, docs/ARCHITECTURE.md, docs/API_QUICK_REFERENCE.md).

**Usage:**
```bash
uv run python scripts/verify_code_examples.py
```

Validates:
- Python syntax correctness
- Import statement accuracy
- Function signature consistency
- Variable naming conventions

**Purpose:** Ensures documentation code examples stay synchronized with actual API changes.

## For Contributors

These scripts are part of the development workflow but are not required for end users. They help maintain code quality and documentation accuracy during development.

For general development guidelines, see [../CONTRIBUTING.md](../CONTRIBUTING.md).
