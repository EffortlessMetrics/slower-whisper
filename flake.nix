{
  description = "slower-whisper: local-first conversation signal engine (Nix + uv)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # System dependencies needed by slower-whisper
        systemDeps = with pkgs; [
          # Python
          python312
          uv

          # Audio processing
          ffmpeg
          libsndfile
          portaudio

          # Build tools
          pkg-config
          gcc

          # Utilities
          git
          jq
          curl

          # SSL/compression
          openssl
          zlib
        ];

        # Common shell environment
        commonShellHook = ''
          export SLOWER_WHISPER_CACHE_ROOT="''${HOME}/.cache/slower-whisper"
          export PYTHONPATH="$PWD:''${PYTHONPATH:-}"
        '';

      in {
        # Development shell: nix develop
        devShells.default = pkgs.mkShell {
          buildInputs = systemDeps;

          shellHook = commonShellHook + ''
            echo ""
            echo "🚀 slower-whisper Nix dev shell ready"
            echo ""
            echo "Next steps:"
            echo "  1) uv sync --extra full --extra diarization --extra dev"
            echo "  2) uv run pytest -m 'not slow and not heavy'"
            echo "  3) uv run slower-whisper transcribe --help"
            echo ""
            echo "Run local CI:"
            echo "  nix flake check              # run all CI checks"
            echo "  nix build .#checks.${system}.lint     # individual checks"
            echo ""
            echo "Dogfooding:"
            echo "  nix run .#dogfood -- --sample synthetic --skip-llm"
            echo "  nix run .#verify -- --quick"
            echo ""
          '';
        };

        # Lightweight dev shell (ASR only, no enrichment)
        devShells.light = pkgs.mkShell {
          buildInputs = systemDeps;

          shellHook = commonShellHook + ''
            echo ""
            echo "🚀 slower-whisper Nix dev shell (light - ASR only)"
            echo ""
            echo "Next steps:"
            echo "  1) uv sync --extra dev"
            echo "  2) uv run pytest tests/test_models.py"
            echo ""
          '';
        };

        # CI checks: nix flake check
        checks = {
          # Lint check (ruff check)
          lint = pkgs.runCommand "slower-whisper-lint" {
            buildInputs = systemDeps;
            src = ./.;
          } ''
            set -euo pipefail
            cd $src

            echo "[CI/lint] Installing dependencies..."
            export HOME=$(mktemp -d)
            export UV_CACHE_DIR=$HOME/.cache/uv

            uv sync --no-dev
            uv pip install ruff

            echo "[CI/lint] Running ruff check..."
            uv run ruff check transcription/ tests/ examples/ benchmarks/

            touch $out
          '';

          # Format check (ruff format)
          format = pkgs.runCommand "slower-whisper-format" {
            buildInputs = systemDeps;
            src = ./.;
          } ''
            set -euo pipefail
            cd $src

            echo "[CI/format] Installing dependencies..."
            export HOME=$(mktemp -d)
            export UV_CACHE_DIR=$HOME/.cache/uv

            uv sync --no-dev
            uv pip install ruff

            echo "[CI/format] Checking code formatting..."
            uv run ruff format --check transcription/ tests/ examples/ benchmarks/

            touch $out
          '';

          # Type check (mypy)
          type-check = pkgs.runCommand "slower-whisper-type-check" {
            buildInputs = systemDeps;
            src = ./.;
          } ''
            set -euo pipefail
            cd $src

            echo "[CI/type-check] Installing dependencies..."
            export HOME=$(mktemp -d)
            export UV_CACHE_DIR=$HOME/.cache/uv

            uv sync --extra dev

            echo "[CI/type-check] Running mypy..."
            # Allow type check failures (continue-on-error: true in CI)
            uv run mypy transcription/ tests/ || echo "⚠️  Type check warnings (non-blocking)"

            touch $out
          '';

          # Fast unit tests (no slow/heavy/GPU)
          test-fast = pkgs.runCommand "slower-whisper-test-fast" {
            buildInputs = systemDeps;
            src = ./.;
          } ''
            set -euo pipefail
            cd $src

            echo "[CI/test-fast] Installing dependencies..."
            export HOME=$(mktemp -d)
            export UV_CACHE_DIR=$HOME/.cache/uv

            uv sync --extra dev

            echo "[CI/test-fast] Running fast test suite..."
            uv run pytest \
              --cov=transcription \
              --cov-report=term-missing \
              -v \
              -m "not slow and not heavy" \
              --tb=short

            touch $out
          '';

          # Integration tests
          test-integration = pkgs.runCommand "slower-whisper-test-integration" {
            buildInputs = systemDeps;
            src = ./.;
          } ''
            set -euo pipefail
            cd $src

            echo "[CI/test-integration] Installing dependencies..."
            export HOME=$(mktemp -d)
            export UV_CACHE_DIR=$HOME/.cache/uv

            uv sync --extra dev

            echo "[CI/test-integration] Running integration tests..."
            uv run pytest tests/test_*integration*.py -v --tb=short

            touch $out
          '';

          # BDD library contract
          bdd-library = pkgs.runCommand "slower-whisper-bdd-library" {
            buildInputs = systemDeps;
            src = ./.;
          } ''
            set -euo pipefail
            cd $src

            echo "[CI/bdd-library] Installing dependencies..."
            export HOME=$(mktemp -d)
            export UV_CACHE_DIR=$HOME/.cache/uv

            uv sync --extra dev

            echo "[CI/bdd-library] Running library BDD scenarios..."
            uv run pytest tests/steps/ -v -m "not slow and not requires_gpu" --tb=short

            touch $out
          '';

          # BDD API contract (smoke tests only)
          bdd-api = pkgs.runCommand "slower-whisper-bdd-api" {
            buildInputs = systemDeps;
            src = ./.;
          } ''
            set -euo pipefail
            cd $src

            echo "[CI/bdd-api] Installing dependencies..."
            export HOME=$(mktemp -d)
            export UV_CACHE_DIR=$HOME/.cache/uv

            uv sync --extra dev

            echo "[CI/bdd-api] Running API BDD scenarios (smoke tests)..."
            uv run pytest features/ -v -m "api and smoke" --tb=short

            touch $out
          '';

          # Verification check (slower-whisper-verify --quick)
          verify = pkgs.runCommand "slower-whisper-verify" {
            buildInputs = systemDeps;
            src = ./.;
          } ''
            set -euo pipefail
            cd $src

            echo "[CI/verify] Installing dependencies..."
            export HOME=$(mktemp -d)
            export UV_CACHE_DIR=$HOME/.cache/uv
            export SLOWER_WHISPER_CACHE_ROOT=$HOME/.cache/slower-whisper

            uv sync --extra full --extra diarization --extra dev

            echo "[CI/verify] Running slower-whisper-verify --quick..."
            uv run slower-whisper-verify --quick

            touch $out
          '';

          # Dogfood smoke test (synthetic audio, no network)
          dogfood-smoke = pkgs.runCommand "slower-whisper-dogfood-smoke" {
            buildInputs = systemDeps;
            src = ./.;
          } ''
            set -euo pipefail
            cd $src

            echo "[CI/dogfood-smoke] Installing dependencies..."
            export HOME=$(mktemp -d)
            export UV_CACHE_DIR=$HOME/.cache/uv
            export SLOWER_WHISPER_CACHE_ROOT=$HOME/.cache/slower-whisper

            uv sync --extra full --extra dev

            echo "[CI/dogfood-smoke] Running dogfood smoke test (synthetic audio, no LLM)..."
            uv run slower-whisper-dogfood --sample synthetic --skip-llm

            touch $out
          '';

          # Combined CI check (runs all checks)
          ci-all = pkgs.runCommand "slower-whisper-ci-all" {
            buildInputs = systemDeps;
          } ''
            set -euo pipefail

            # Force all checks to be built / evaluated
            test -f ${self.checks.${system}.lint}
            test -f ${self.checks.${system}.format}
            test -f ${self.checks.${system}.type-check}
            test -f ${self.checks.${system}.test-fast}
            test -f ${self.checks.${system}.test-integration}
            test -f ${self.checks.${system}.bdd-library}
            test -f ${self.checks.${system}.bdd-api}
            test -f ${self.checks.${system}.verify}
            test -f ${self.checks.${system}.dogfood-smoke}

            echo "✅ All CI checks passed!"
            echo ""
            echo "  ✅ Code quality verified (lint + format)"
            echo "  ✅ Type check completed"
            echo "  ✅ Fast tests passed"
            echo "  ✅ Integration tests passed"
            echo "  ✅ Library BDD contract enforced"
            echo "  ✅ API BDD contract enforced"
            echo "  ✅ Verification suite passed"
            echo "  ✅ Dogfood smoke test passed"
            echo ""
            touch $out
          '';
        };

        # Apps: nix run .#<app>
        apps = {
          # Dogfooding workflow (runs slower-whisper-dogfood)
          dogfood = {
            type = "app";
            program = toString (pkgs.writeShellScript "dogfood" ''
              set -euo pipefail
              export SLOWER_WHISPER_CACHE_ROOT="''${SLOWER_WHISPER_CACHE_ROOT:-$HOME/.cache/slower-whisper}"

              # Run dogfood with all args passed through
              exec ${pkgs.uv}/bin/uv run slower-whisper-dogfood "$@"
            '');
          };

          # Verification workflow (runs slower-whisper-verify)
          verify = {
            type = "app";
            program = toString (pkgs.writeShellScript "verify" ''
              set -euo pipefail
              export SLOWER_WHISPER_CACHE_ROOT="''${SLOWER_WHISPER_CACHE_ROOT:-$HOME/.cache/slower-whisper}"

              # Run verify with all args passed through
              exec ${pkgs.uv}/bin/uv run slower-whisper-verify "$@"
            '');
          };
        };

        # Default package (for nix build)
        packages.default = pkgs.python312Packages.buildPythonPackage {
          pname = "slower-whisper";
          version = "1.1.0-dev";
          src = ./.;

          propagatedBuildInputs = systemDeps;

          # Note: This is a minimal package definition
          # For full packaging, consider poetry2nix or uv2nix
          doCheck = false;
        };
      }
    );
}
