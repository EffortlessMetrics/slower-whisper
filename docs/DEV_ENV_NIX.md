# Nix Development Environment

This guide explains how to use Nix for reproducible development and CI environments with slower-whisper.

## Overview

slower-whisper uses a **hybrid approach**:

- **Nix** manages system dependencies (ffmpeg, libsndfile, Python, etc.)
- **uv** manages Python packages (faster-whisper, torch, etc.)

This gives you:

- ✅ **Reproducible environments** across machines (WSL, NixOS, macOS, CI)
- ✅ **Same environment locally as in CI** - if `nix flake check` passes locally, it should pass in CI
- ✅ **Fast Python package management** via uv
- ✅ **No containerization overhead** for local development

## Prerequisites

### 1. Install Nix

**On WSL2 / Linux / macOS:**

```bash
# Single-user installation (recommended for WSL)
sh <(curl -L https://nixos.org/nix/install) --daemon

# Enable flakes
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

**On NixOS:**

Flakes should already be available. If not, add to `/etc/nixos/configuration.nix`:

```nix
nix.settings.experimental-features = [ "nix-command" "flakes" ];
```

### 2. Install uv (if not using Nix shell)

If you want to use uv outside the Nix shell:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or install via Nix if available in your channel.

## Usage

### Development Shell

**Full development environment (ASR + diarization + enrichment):**

```bash
# Enter Nix dev shell
nix develop

# Inside the shell, sync Python dependencies
uv sync --extra full --extra diarization --extra dev

# Run tests
uv run pytest -m "not slow and not heavy"

# Run transcription
uv run slower-whisper transcribe --help
```

**Lightweight environment (ASR only, no enrichment):**

```bash
# Enter light dev shell
nix develop .#light

# Sync minimal dependencies
uv sync --extra dev

# Run basic tests
uv run pytest tests/test_models.py
```

### Local CI (Run All Checks)

Run the **exact same checks** that run in GitHub Actions:

```bash
# Run all CI checks
nix flake check

# Run specific check
nix build .#checks.x86_64-linux.lint
nix build .#checks.x86_64-linux.test-fast
nix build .#checks.x86_64-linux.bdd-library
```

Available checks:

- `lint` - ruff linting
- `format` - ruff formatting check
- `type-check` - mypy type checking (non-blocking)
- `test-fast` - fast unit tests (excludes slow/heavy/GPU tests)
- `test-integration` - integration tests
- `bdd-library` - library BDD contract tests
- `bdd-api` - API BDD smoke tests
- `ci-all` - runs all checks (combined)

### Optional: direnv Integration

If you use [direnv](https://direnv.net/), you can automatically enter the Nix shell when you `cd` into the project:

**Create `.envrc`:**

```bash
use flake
```

**Allow direnv:**

```bash
direnv allow
```

Now the Nix shell will activate automatically when you enter the directory.

## How This Works

### System Dependencies (Nix)

The `flake.nix` provides these system packages:

- **Python 3.12** - base Python runtime
- **ffmpeg** - audio normalization (required for ASR)
- **libsndfile** - audio I/O for enrichment
- **portaudio** - audio device access
- **pkg-config, gcc** - build tools for native extensions
- **git, jq, curl** - utilities

### Python Dependencies (uv)

All Python packages are managed by uv via `pyproject.toml`:

- **Base dependencies**: faster-whisper (~2.5GB)
- **Enrichment extras**: librosa, praat-parselmouth, soundfile
- **Diarization extras**: pyannote.audio, torch, torchaudio
- **Dev extras**: pytest, ruff, mypy

```bash
# Install minimal dependencies
uv sync

# Install with enrichment features
uv sync --extra full

# Install with diarization
uv sync --extra diarization

# Install everything (dev + all features)
uv sync --all-extras
```

### CI Alignment

The `flake.nix` checks mirror the jobs in `.github/workflows/ci.yml`:

| Nix Check          | GitHub Actions Job | Description                          |
|--------------------|-------------------|--------------------------------------|
| `lint`             | `lint`            | ruff check (code quality)            |
| `format`           | `format`          | ruff format check                    |
| `type-check`       | `type-check`      | mypy type checking                   |
| `test-fast`        | `test`            | pytest (fast unit tests)             |
| `test-integration` | `test-integration`| integration tests                    |
| `bdd-library`      | `bdd-library`     | library BDD contract                 |
| `bdd-api`          | `bdd-api`         | API BDD smoke tests                  |

## Troubleshooting

### "uv not found in PATH"

If you see this warning in the Nix shell:

**Option 1: Install uv globally**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Option 2: Use Nix to install uv (if available)**

If `pkgs.uv` exists in your nixpkgs channel, it's already included in the shell.

### "Cannot access gated repo" (pyannote models)

For diarization features, you need:

1. HuggingFace account with token: https://huggingface.co/settings/tokens
2. Accept licenses for pyannote models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1

3. Set token in environment:

   ```bash
   export HF_TOKEN="hf_..."
   ```

### "nix flake check" fails but GitHub CI passes

This usually means:

1. **Different nixpkgs version** - the flake uses `nixos-24.05`; try updating:

   ```bash
   nix flake update
   ```

2. **Cache differences** - some tests might depend on cached data. Clear cache:

   ```bash
   rm -rf ~/.cache/slower-whisper
   rm -rf ~/.cache/huggingface
   ```

3. **Environment variables** - CI might set extra env vars (like `HF_TOKEN`). Check `.github/workflows/ci.yml`.

### Tests fail with "ffmpeg not found"

Make sure you're inside the Nix shell:

```bash
# Should show Nix store path
which ffmpeg
# Example: /nix/store/.../bin/ffmpeg
```

If not in shell:

```bash
nix develop
# Now try again
which ffmpeg
```

## Migrating from Non-Nix Setup

If you're currently using:

```bash
# Old workflow
sudo apt-get install ffmpeg libsndfile1
uv sync --all-extras
uv run pytest
```

Switch to:

```bash
# New Nix workflow
nix develop
uv sync --all-extras
uv run pytest
```

The only difference: **Nix provides system deps** instead of `apt-get`.

## When to Use Nix vs. Traditional Setup

### Use Nix when:

- ✅ You want **reproducible environments** across machines
- ✅ You're on **NixOS** or already use Nix
- ✅ You want to **mirror CI locally** (`nix flake check`)
- ✅ You're setting up a **new development machine**
- ✅ You need **specific versions** of system dependencies

### Use traditional setup when:

- ✅ You're on a system where Nix is hard to install (corporate Windows, etc.)
- ✅ You already have working system deps and don't want to change
- ✅ You only need to run tests occasionally
- ✅ You're contributing a small fix and don't want to install Nix

**Both approaches are fully supported.** Nix is optional and provides stronger reproducibility guarantees.

## Future: Nix-Native Python (Optional)

Currently, Nix provides system deps and uv handles Python packages. For **fully Nix-native** Python:

- Use `poetry2nix` or `uv2nix` to build Python packages via Nix
- Pros: Stronger reproducibility, offline builds, binary caching
- Cons: Slower iteration, steeper learning curve

This is not required for v1.1 and may be explored later.

## References

- [Nix Manual](https://nixos.org/manual/nix/stable/)
- [Nix Flakes](https://nixos.wiki/wiki/Flakes)
- [uv Documentation](https://docs.astral.sh/uv/)
- [direnv Integration](https://github.com/nix-community/nix-direnv)

## Support

If you encounter issues with the Nix setup:

1. Check this document first
2. Check if the issue exists in non-Nix setup (to isolate Nix-specific problems)
3. Open an issue with:
   - Nix version: `nix --version`
   - System: `uname -a`
   - Error output from `nix flake check` or `nix develop`
