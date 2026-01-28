# CI/CD Setup Guide

Complete guide to setting up the GitHub Actions CI/CD pipeline for slower-whisper.

## 🚀 Quick Start

The workflows are ready to use! Simply push to the repository:

```bash
git add .github/
git commit -m "Add CI/CD workflows"
git push origin main
```

The CI workflow will automatically run on this push.

## 📋 Step-by-Step Setup

### 1. Enable GitHub Actions

GitHub Actions should be enabled by default. Verify:

1. Go to repository **Settings** → **Actions** → **General**
2. Ensure "Allow all actions and reusable workflows" is selected
3. Set workflow permissions to "Read and write permissions"

### 2. Configure Branch Protection (Recommended)

Protect your `main` branch:

1. Go to **Settings** → **Branches**
2. Click "Add rule" or edit existing rule
3. Branch name pattern: `main`
4. Enable:
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - Select: `CI Success` (will appear after first workflow run)
   - ✅ Require linear history (optional)
   - ✅ Do not allow bypassing the above settings

### 3. Set up Codecov (Optional but Recommended)

For coverage reporting and badges:

1. Go to [codecov.io](https://codecov.io/) and sign in with GitHub
2. Add your repository
3. Copy the repository upload token
4. Add to GitHub Secrets:
   - Go to **Settings** → **Secrets and variables** → **Actions**
   - Click "New repository secret"
   - Name: `CODECOV_TOKEN`
   - Value: [paste token]

### 4. Configure PyPI Publishing (For Releases)

#### Option A: Trusted Publishing (Recommended - No tokens!)

1. Create a PyPI account at [pypi.org](https://pypi.org/)
2. Reserve your package name by uploading v0.0.1 manually (one-time):
   ```bash
   uv build
   uv pip install twine
   twine upload dist/*
   ```
3. Configure trusted publisher:
   - Go to [pypi.org/manage/projects/](https://pypi.org/manage/projects/)
   - Select your project
   - Click "Publishing" → "Add a new pending publisher"
   - Fill in:
     - **PyPI Project Name**: `slower-whisper`
     - **Owner**: `YOUR_GITHUB_USERNAME`
     - **Repository name**: `slower-whisper`
     - **Workflow name**: `release.yml`
     - **Environment name**: `pypi` (leave blank if not using environments)
   - Save

4. (Optional) Do the same for TestPyPI at [test.pypi.org](https://test.pypi.org/)

#### Option B: Using API Tokens (Legacy)

1. Generate PyPI API token:
   - Go to [pypi.org/manage/account/](https://pypi.org/manage/account/)
   - Scroll to "API tokens" → "Add API token"
   - Token name: "slower-whisper-github-actions"
   - Scope: "Entire account" or specific to "slower-whisper"
2. Add to GitHub Secrets:
   - Name: `PYPI_API_TOKEN`
   - Value: [paste token starting with `pypi-`]
3. Update `release.yml` to use token instead of OIDC

### 5. Enable Dependabot

Dependabot is automatically enabled! GitHub will start creating PRs for dependency updates.

To configure alerts:
1. Go to **Settings** → **Security** → **Code security and analysis**
2. Enable "Dependabot alerts"
3. Enable "Dependabot security updates"

## 🧪 Testing the Setup

### Test CI Workflow

Create a test branch and PR:

```bash
git checkout -b test-ci
echo "# Test" >> README.md
git add README.md
git commit -m "Test CI workflow"
git push origin test-ci
```

Then create a PR on GitHub. The CI workflow should run automatically.

### Test Release Workflow (Dry Run)

Test the release workflow without publishing:

```bash
# Create a test tag
git tag v0.0.1-test
git push origin v0.0.1-test

# Or trigger manually
gh workflow run release.yml -f version=0.0.1-test
```

**Note**: This will attempt to publish to PyPI. Use TestPyPI for testing:
- Set up TestPyPI trusted publisher
- Modify `release.yml` to publish to TestPyPI by default

## 📊 Monitoring Workflows

### View Workflow Runs

```bash
# List all workflow runs
gh run list

# List runs for specific workflow
gh run list --workflow=ci.yml

# Watch a running workflow
gh run watch

# View logs for a specific run
gh run view RUN_ID --log
```

### GitHub UI

Go to the **Actions** tab in your repository to see:
- All workflow runs
- Status of each job
- Detailed logs
- Artifacts (if any)

## 🔧 Local Development

### Install Pre-commit Hooks (Optional)

Run checks locally before committing:

```bash
# Install pre-commit
uv pip install pre-commit

# Install git hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Configuration is in `.github/pre-commit-config.yaml`.

### Run CI Checks Locally

Simulate what CI will run:

```bash
# 1. Linting
uv run ruff check .

# 2. Formatting
uv run ruff format --check .

# 3. Type checking
uv run mypy transcription tests

# 4. Tests
uv run pytest

# All together
uv run ruff check . && \
uv run ruff format --check . && \
uv run mypy transcription tests && \
uv run pytest
```

### Install Development Dependencies

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Or just dev dependencies
uv sync --extra dev
```

## 📝 Adding Status Badges

Add these to your README.md:

```markdown
# slower-whisper

[![CI](https://github.com/YOUR_USERNAME/slower-whisper/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/slower-whisper/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/slower-whisper/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/slower-whisper)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

See `.github/workflows/badges.md` for more options.

## 🐛 Troubleshooting

### Workflow doesn't trigger

**Problem**: Pushed code but workflow didn't run

**Solutions**:
1. Check if GitHub Actions is enabled (Settings → Actions)
2. Verify `.github/workflows/` directory is committed
3. Check workflow YAML syntax (validate with `yamllint` or online validator)
4. Ensure workflow triggers match your branch/event

### Tests fail on CI but pass locally

**Problem**: Tests pass on your machine but fail on GitHub Actions

**Solutions**:
1. Check Python version compatibility (test locally with multiple versions)
2. Verify all dependencies are in `pyproject.toml`
3. Check for hardcoded paths (use `Path` from `pathlib`)
4. Look for environment-specific issues (e.g., Windows vs Linux)
5. Check GitHub Actions logs for detailed error messages

### Codecov upload fails

**Problem**: Coverage upload step fails

**Solutions**:
1. Verify `CODECOV_TOKEN` secret is set correctly
2. Check Codecov dashboard for error messages
3. Ensure repository is public or Codecov account has access
4. Set `fail_ci_if_error: false` to make it non-blocking

### PyPI publish fails

**Problem**: Release workflow fails at PyPI publish step

**Solutions**:
1. **For Trusted Publishing**:
   - Verify trusted publisher is configured on PyPI
   - Check owner, repository, and workflow names match exactly
   - Ensure workflow is running from `main` branch
2. **For API Token**:
   - Verify `PYPI_API_TOKEN` secret is set
   - Token should start with `pypi-`
   - Check token has correct permissions
3. Check package name isn't already taken on PyPI
4. Verify `pyproject.toml` has correct metadata

### Dependabot PRs failing

**Problem**: Dependabot creates PRs but CI fails

**Solutions**:
1. Check if updated dependencies have breaking changes
2. Review Dependabot PR description for upgrade notes
3. Test locally with updated dependencies
4. May need to update code for new dependency versions

## 🔒 Security Best Practices

1. **Never commit secrets**: Use GitHub Secrets for tokens
2. **Use Trusted Publishing**: Avoid storing PyPI tokens
3. **Minimal permissions**: Only grant necessary workflow permissions
4. **Pin actions versions**: Use specific versions (e.g., `@v4`) not `@main`
5. **Review Dependabot PRs**: Don't blindly auto-merge
6. **Enable branch protection**: Require CI to pass before merging

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.com/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)

## 🆘 Getting Help

If you encounter issues:

1. Check workflow logs in GitHub Actions tab
2. Review this guide and troubleshooting section
3. Search GitHub Actions community forums
4. Open an issue in the repository

## ✅ Verification Checklist

- [ ] GitHub Actions enabled
- [ ] Workflows pushed to repository
- [ ] CI workflow runs successfully
- [ ] Branch protection configured
- [ ] Codecov integration set up (optional)
- [ ] PyPI trusted publishing configured (for releases)
- [ ] Status badges added to README
- [ ] Pre-commit hooks installed (optional)
- [ ] Team members can run tests locally

---

**Last Updated**: 2025-11-15
**Maintained By**: slower-whisper contributors
