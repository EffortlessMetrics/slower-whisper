# BDD & IaC Verification: Python CLI Migration

**Date:** 2025-11-17
**Status:** ✅ Complete
**Purpose:** Cross-platform, versioned verification tooling

---

## Overview

The BDD and IaC verification tooling has been migrated from shell scripts to a **Python CLI** (`slower-whisper-verify`) for:

- ✅ **Cross-platform support** (Linux, macOS, Windows)
- ✅ **Version control** alongside code
- ✅ **Dependency management** via uv/pip
- ✅ **Testable infrastructure** (tests for the test tooling)
- ✅ **Unified interface** with other project CLIs

---

## Quick Start

### Installation

The verification CLI is installed automatically with the project:

```bash
# Install project with dev dependencies
uv sync --extra dev

# Verify installation
uv run slower-whisper-verify --help
```

### Usage

```bash
# Quick verification (code + tests + BDD only)
uv run slower-whisper-verify --quick

# Full verification (includes Docker and K8s)
uv run slower-whisper-verify

# Alternative invocations
python scripts/verify_all.py --quick
uv run python -m scripts.verify_all --quick
```

---

## What It Does

The verification CLI runs a comprehensive suite of checks:

### 1. Code Quality (ruff)
- Lints `transcription/` and `tests/` directories
- Catches style violations and potential bugs

### 2. Unit/Fast Tests (pytest)
- Runs tests marked `not slow and not requires_gpu`
- Includes coverage reporting
- Fast feedback for local development

### 3. BDD Acceptance Scenarios (pytest-bdd)
- Runs all Gherkin scenarios from `tests/features/`
- Detects ffmpeg availability and handles xfail gracefully
- Validates behavioral contract

### 4. Docker Smoke Tests (optional, requires `--full` or no `--quick`)
- Builds CPU image (`Dockerfile`)
- Builds API image (`Dockerfile.api`)
- Runs CLI help in CPU image to verify entry points
- Validates IaC deployment contract

### 5. Kubernetes Validation (optional, requires `--full` or no `--quick`)
- Validates all `k8s/*.yaml` manifests with `kubectl --dry-run=client`
- Ensures manifests are syntactically correct
- Validates IaC deployment contract

---

## Architecture

### File Structure

```
scripts/
  __init__.py              # Makes scripts a Python package
  verify_all.py            # Main verification CLI (200 lines)
  verify_bdd.sh            # Legacy shell script (deprecated)
  docker_smoke_test.sh     # Legacy shell script (deprecated)
  validate_k8s.sh          # Legacy shell script (deprecated)
  verify_all.sh            # Legacy shell script (deprecated)
  verify_bdd_legacy.sh     # Thin wrapper to Python CLI

tests/
  test_verify_all.py       # Tests for the verification CLI
```

### Entry Points

The CLI is registered in `pyproject.toml`:

```toml
[project.scripts]
slower-whisper-verify = "scripts.verify_all:main"
```

This creates a console command when the package is installed.

### Individual Functions

The `scripts/verify_all.py` module exposes individual verification functions that can be imported and used programmatically:

```python
from scripts.verify_all import (
    check_ruff,        # Run ruff linter
    run_tests_fast,    # Run fast pytest suite
    verify_bdd,        # Run BDD scenarios
    docker_smoke,      # Build and test Docker images
    validate_k8s,      # Validate K8s manifests
)

# Use individually
check_ruff()
verify_bdd()
```

---

## Testing the Verification CLI

The verification CLI itself is tested:

```bash
# Run verification CLI tests
uv run pytest tests/test_verify_all.py -v
```

**Tests include:**
- Help flag works (`--help`)
- Module can be imported
- Quick mode dry-run (with mocked subprocess)
- All components exist and are callable
- Can be run as module (`python -m scripts.verify_all`)
- Can be run as direct script

This meta-testing ensures the verification tooling is reliable.

---

## Migration from Shell Scripts

### Before (shell scripts)

```bash
./scripts/verify_all.sh --quick
./scripts/verify_bdd.sh
./scripts/docker_smoke_test.sh
./scripts/validate_k8s.sh
```

**Limitations:**
- ❌ Not cross-platform (Windows compatibility issues)
- ❌ Not versioned (harder to track changes)
- ❌ Harder to test (no unit tests for shell logic)
- ❌ Duplicate logic across scripts

### After (Python CLI)

```bash
uv run slower-whisper-verify --quick
# or
python scripts/verify_all.py --quick
```

**Advantages:**
- ✅ Cross-platform (Windows, Linux, macOS)
- ✅ Versioned alongside code
- ✅ Testable (6 test cases in `test_verify_all.py`)
- ✅ Single source of truth
- ✅ Integrated with uv/pip dependency management
- ✅ Console script entry point

### Backward Compatibility

Legacy shell scripts remain in place but are **deprecated**:
- Original shell scripts kept for reference
- `verify_bdd_legacy.sh` wraps Python CLI for compatibility
- Documentation updated to recommend Python CLI

---

## Configuration

### Quick Mode (`--quick`)

Runs only:
- Code quality (ruff)
- Fast tests (pytest)
- BDD scenarios

**Use case:** Pre-commit, local development

### Full Mode (default)

Runs everything including:
- Docker image builds
- Kubernetes validation

**Use case:** Pre-release, CI/CD

### Environment Detection

The CLI automatically detects:
- `ffmpeg` availability (for BDD tests)
- `docker` availability (for Docker smoke tests)
- `kubectl` availability (for K8s validation)

Missing tools result in graceful warnings, not failures.

---

## Integration Points

### Pre-commit Hook (recommended)

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: verify-quick
      name: Run quick verification suite
      entry: uv run slower-whisper-verify --quick
      language: system
      pass_filenames: false
      always_run: true
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Run verification suite
  run: uv run slower-whisper-verify --quick
```

### Manual Testing

```bash
# Before committing
uv run slower-whisper-verify --quick

# Before releasing
uv run slower-whisper-verify
```

---

## Extending the CLI

To add a new verification step:

1. Add function to `scripts/verify_all.py`:
   ```python
   def verify_new_thing() -> None:
       print("━" * 60)
       print("6️⃣  New verification step")
       print("━" * 60)
       # implementation
   ```

2. Call in `main()`:
   ```python
   def main(argv: list[str] | None = None) -> int:
       # ... existing steps
       verify_new_thing()
       # ...
   ```

3. Add tests in `tests/test_verify_all.py`:
   ```python
   def test_verify_new_thing():
       from scripts.verify_all import verify_new_thing
       assert callable(verify_new_thing)
   ```

---

## Troubleshooting

### "No module named 'scripts'"

**Cause:** Scripts package not installed

**Fix:**
```bash
uv sync --extra dev
# or
pip install -e ".[dev]"
```

### "Command not found: slower-whisper-verify"

**Cause:** Package not installed or not in PATH

**Fix:**
```bash
uv run slower-whisper-verify  # Use uv run prefix
# or
uv sync --extra dev           # Reinstall
```

### BDD tests failing/xfailing

**Cause:** ffmpeg not available

**Expected:** Tests gracefully xfail when ffmpeg unavailable
**Fix (if you need them to pass):** Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

### Docker tests skipped

**Cause:** Docker daemon not running or not installed

**Expected:** Tests skip gracefully
**Fix:** Install Docker and ensure daemon is running

### K8s validation skipped

**Cause:** kubectl not installed

**Expected:** Validation skips gracefully
**Fix:** Install kubectl (only needed for K8s deployment)

---

## Implementation Metrics

- **Lines of Code:** 200 (verify_all.py)
- **Test Coverage:** 6 test cases
- **Entry Points:** 3 (console script, direct script, module)
- **Dependencies:** Python 3.11+, uv/pip (runtime), pytest (testing)
- **Platforms:** Linux, macOS, Windows
- **Backward Compatibility:** Shell scripts deprecated but functional

---

## Next Steps (Optional)

1. **Add to pre-commit hooks** (`.pre-commit-config.yaml`)
2. **Integrate into CI/CD** when GitHub Actions re-enabled
3. **Add performance benchmarks** for verification suite runtime
4. **Create alias shortcuts** (e.g., `make verify-quick`)
5. **Add JSON output mode** for machine-readable results

---

## Summary

The migration from shell scripts to a Python CLI provides:

- ✅ Cross-platform verification tooling
- ✅ Versioned contract enforcement
- ✅ Testable infrastructure
- ✅ Unified developer experience
- ✅ Production-ready verification

The BDD and IaC contracts are now enforced through **first-class, tested tooling** that runs everywhere the codebase runs. 🚀
