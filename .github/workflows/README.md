# GitHub Actions Workflows

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### 📋 `ci.yml` - Continuous Integration

Runs on every push to `main` and on all pull requests.

**Jobs:**

1. **Lint** (`lint`)
   - Runs `ruff check` to enforce code quality standards
   - Fast-fail job that must pass before other jobs run

2. **Format Check** (`format`)
   - Runs `ruff format --check` to ensure consistent code formatting
   - Fast-fail job

3. **Type Check** (`type-check`)
   - Runs `mypy` for static type checking
   - Checks `transcription` and `tests` directories

4. **Test** (`test`)
   - Runs on Python 3.12 and 3.13 in parallel
   - Executes pytest with coverage reporting
   - Excludes `slow` and `heavy` marked tests
   - Uploads coverage to Codecov (for Python 3.12 only)
   - Installs system dependencies (ffmpeg, libsndfile1)

5. **Heavy Tests** (`test-heavy`)
   - Only runs on `main` branch or manual workflow dispatch
   - Tests that require downloading emotion recognition models
   - Set to `continue-on-error: true` to not block CI

6. **Integration Tests** (`test-integration`)
   - Runs after lint and format checks pass
   - Tests end-to-end workflows in `test_integration.py`

7. **CI Success** (`ci-success`)
   - Summary job that gates on all required checks
   - Fails if any required job fails
   - Use this as a branch protection rule

**Usage:**
```bash
# Trigger workflow manually
gh workflow run ci.yml

# View workflow runs
gh run list --workflow=ci.yml
```

### 🚀 `release.yml` - Release to PyPI

Triggered on version tags (e.g., `v2.0.0`) or manual dispatch.

**Jobs:**

1. **Test** - Runs full CI suite before building
2. **Build** - Creates source and wheel distributions
3. **Publish to PyPI** - Uploads to PyPI using trusted publishing
4. **Publish to TestPyPI** - For testing (manual dispatch only)
5. **GitHub Release** - Creates GitHub release with changelog

**Creating a Release:**

```bash
# 1. Update version in pyproject.toml
# 2. Commit changes
git add pyproject.toml
git commit -m "Bump version to 2.0.1"

# 3. Create and push tag
git tag v2.0.1
git push origin main --tags

# 4. Workflow automatically runs and publishes
```

**Manual Release (TestPyPI):**

```bash
gh workflow run release.yml -f version=2.0.1-rc1
```

**PyPI Trusted Publishing Setup:**

1. Go to PyPI project settings: https://pypi.org/manage/project/slower-whisper/settings/
2. Navigate to "Publishing" section
3. Add a new publisher:
   - PyPI Project Name: `slower-whisper`
   - Owner: `EffortlessMetrics`
   - Repository name: `slower-whisper`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

### 🤖 `dependabot.yml` - Automated Dependency Updates

Dependabot automatically creates PRs for dependency updates.

**Update Schedule:**
- **Weekly** (Monday) for all ecosystems
- GitHub Actions updates
- Python package updates (grouped by category)

**Dependency Groups:**
- `dev-dependencies`: pytest, ruff, mypy, coverage
- `audio-dependencies`: librosa, soundfile, numpy, parselmouth
- `ml-dependencies`: torch, transformers

**Configuration:**
```yaml
# Ignores major version updates for ML packages
# to prevent breaking changes
```

## Running Tests Locally

### Quick Tests (default)
```bash
# Run fast tests only
uv run pytest

# With coverage
uv run pytest --cov=transcription --cov-report=html
```

### All Tests (including heavy)
```bash
# Run all tests
uv run pytest -m ""

# Run only heavy tests
uv run pytest -m "heavy"

# Run integration tests
uv run pytest tests/test_integration.py
```

### Linting and Formatting
```bash
# Check code style
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Check formatting
uv run ruff format --check .

# Format code
uv run ruff format .
```

### Type Checking
```bash
# Run mypy
uv run mypy transcription tests
```

## Coverage Reports

After running tests with coverage:
```bash
# Terminal report
uv run pytest --cov=transcription --cov-report=term-missing

# HTML report (opens in browser)
uv run pytest --cov=transcription --cov-report=html
open htmlcov/index.html
```

## Branch Protection Rules

Source of truth: [`.github/settings.yml`](../settings.yml) manages branch protection as code.

Current `main` requirements:

1. **Require status checks to pass**:
   - `CI Success` (from `ci.yml`)
   - `Verify (quick)` (from `verify.yml`)

2. **Require branches to be up to date** (`strict: true`)

3. **Require one approving review and CODEOWNERS review**

4. **Enforce rules for admins**

## Secrets Required

### For CI (`ci.yml`)
- `CODECOV_TOKEN` (optional) - For coverage reporting to Codecov

### For Release (`release.yml`)
- None! Uses PyPI Trusted Publishing (OIDC)
- Must configure trusted publisher on PyPI

### 🤖 `verify.yml` - Verification CLI (quick + full)

Runs the `slower-whisper-verify` CLI inside the Nix dev shell.

**Triggers:**
- Pull requests and pushes to `main` (quick mode)
- Nightly schedule (`0 6 * * *`) and manual dispatch (full mode)

**Jobs:**
1. **Verify (quick)** — `nix run .#verify -- --quick`
2. **Verify (full)** — `nix develop .# --command uv run slower-whisper-verify` (includes Docker + K8s; uses `HF_TOKEN` secret for real pyannote)

**Caches:**
- `.venv` and `.cache/uv` (Python deps)
- `~/.cache/huggingface` (full mode)

## Troubleshooting

### Tests failing on CI but passing locally?

1. Check Python version compatibility
2. Verify system dependencies are installed
3. Check for environment-specific issues

### Heavy tests timing out?

- Increase timeout in workflow file
- Or disable heavy tests on CI (already set to continue-on-error)

### Codecov upload failing?

- Verify `CODECOV_TOKEN` is set in repository secrets
- Set `fail_ci_if_error: false` to make it non-blocking

### Release workflow not triggering?

- Verify tag format: `v*.*.*` (e.g., `v2.0.0`)
- Check branch protection rules aren't blocking
- Ensure PyPI trusted publishing is configured

## Monitoring

### View workflow status
```bash
# List recent runs
gh run list

# Watch a running workflow
gh run watch

# View workflow logs
gh run view --log
```

### Badges

Add to your README.md:

```markdown
[![CI](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/ci.yml/badge.svg)](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/ci.yml)
[![Verify](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/verify.yml/badge.svg)](https://github.com/EffortlessMetrics/slower-whisper/actions/workflows/verify.yml)
```

See `.github/workflows/badges.md` for more badge options.
