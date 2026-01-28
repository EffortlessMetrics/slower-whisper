# PACKAGING.md

How we build, validate, and smoke-test distribution artifacts (wheel + sdist)
for slower-whisper. Uses `uv` for dependency management and `build`/`twine`
for packaging.

## Prerequisites

- Python 3.12+
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Optional: `nix develop` for a fully reproducible toolchain

## Build artifacts

```bash
# Clean old outputs
rm -rf dist/ build/ *.egg-info

# Build wheel + sdist
uv build            # or: uv run python -m build

# Inspect contents
ls dist/
tar -tf dist/slower-whisper-*.tar.gz | head
```

## Validate metadata

```bash
uv run twine check dist/*
```

## Smoke-test installation

```bash
# Fresh venv
python -m venv /tmp/slower-whisper-packaging-test
source /tmp/slower-whisper-packaging-test/bin/activate

# Install from built wheel (adjust filename to match version)
pip install dist/slower_whisper-*-py3-none-any.whl

# Verify entry points and imports
slower-whisper --version
python - <<'PY'
from transcription import __version__
print(__version__)
PY

deactivate
rm -rf /tmp/slower-whisper-packaging-test
```

## Extras and optional features

- Full enrichment: `pip install "slower-whisper[full]"`
- Diarization: `pip install "slower-whisper[diarization]"` and export `HF_TOKEN`
- API server: `pip install "slower-whisper[api]"`

## Publishing (optional)

```bash
# TestPyPI first (recommended)
uv run twine upload --repository testpypi dist/*
# Then verify:
pip install --index-url https://test.pypi.org/simple/ slower-whisper

# Production
uv run twine upload dist/*
```

## Notes

- `MANIFEST.in` already includes docs, examples, tests, and configs needed for
  the sdist.
- Docker images build from source via the same `pyproject.toml`; no extra
  packaging steps required for containers.
