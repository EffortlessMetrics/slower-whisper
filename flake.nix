{
  description = "slower-whisper: local-first conversation signal engine (Nix + uv)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, flake-utils, nixpkgs-unstable }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        unstablePkgs = import nixpkgs-unstable { inherit system; };
        ruffPkg =
          let pkg = unstablePkgs.ruff; in
          assert (pkg.version == "0.14.9"); pkg;

        runtimeBinPath = pkgs.lib.makeBinPath [ pkgs.uv pkgs.ffmpeg ];
        runtimeLibPath = pkgs.lib.makeLibraryPath [ pkgs.ffmpeg pkgs.zlib pkgs.stdenv.cc.cc ];

        # System dependencies needed by slower-whisper
        systemDeps = with pkgs; [
          # Python
          python312
          uv  # 0.5.x+ from nixos-24.11

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

          # Code quality (for pure checks)
          ruffPkg
          python312Packages.mypy

          # SSL/compression
          openssl
          zlib
        ];

        # Common shell environment
        commonShellHook = ''
          export PATH="${runtimeBinPath}:''${PATH:-}"
          # Set uv to use Nix's Python and local venv
          export UV_PYTHON="${pkgs.python312}/bin/python"
          export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
          export UV_CACHE_DIR="$PWD/.cache/uv"

          # slower-whisper cache
          export SLOWER_WHISPER_CACHE_ROOT="''${HOME}/.cache/slower-whisper"
          # Prefer the project venv on PYTHONPATH so uv-installed deps override nixpkgs shims
          export PYTHONPATH="$PWD/.venv/lib/python3.12/site-packages:$PWD:''${PYTHONPATH:-}"
          # Make sure Python wheels (numpy/torch/ffmpeg) find runtime libs
          export LD_LIBRARY_PATH="${runtimeLibPath}:''${LD_LIBRARY_PATH:-}"
        '';

      in {
        # Development shell: nix develop
        devShells.default = pkgs.mkShell {
          buildInputs = systemDeps;

          shellHook = commonShellHook + ''
            echo ""
            echo "🚀 slower-whisper Nix dev shell ready"
            echo ""
            echo "Setup (first time):"
            echo "  uv sync --extra full --extra diarization --extra dev"
            echo ""
            echo "Run local CI:"
            echo "  nix run .#ci           # full test suite"
            echo "  nix run .#ci -- fast   # quick checks only"
            echo ""
            echo "Pure checks (offline):"
            echo "  nix flake check        # lint + format only"
            echo ""
            echo "Dogfooding & verification:"
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
            echo "Setup: uv sync --extra dev"
            echo "Tests: uv run pytest tests/test_models.py"
            echo ""
          '';
        };

        # Pure checks (offline-friendly via nixpkgs ruff/mypy)
        checks = {
          # Lint check (ruff pinned via nixpkgs, matches lockfile 0.14.9)
          lint = pkgs.runCommand "slower-whisper-lint" {
            src = ./.;
          } ''
            cd $src
            ${ruffPkg}/bin/ruff check \
              --no-cache \
              transcription/ tests/ examples/ benchmarks/ || {
              echo "❌ Lint failed. Run: nix run .#ci -- fast"
              exit 1
            }
            touch $out
          '';

          # Format check (ruff pinned via nixpkgs, matches lockfile 0.14.9)
          format = pkgs.runCommand "slower-whisper-format" {
            src = ./.;
          } ''
            cd $src
            ${ruffPkg}/bin/ruff format \
              --no-cache \
              --check \
              transcription/ tests/ examples/ benchmarks/ || {
              echo "❌ Format check failed. Run: ruff format ."
              exit 1
            }
            touch $out
          '';
        };

        # Apps: nix run .#<app>
        apps = {
          # Main CI orchestrator
          ci = {
            type = "app";
            program = toString (pkgs.writeShellScript "slower-whisper-ci" ''
              set -euo pipefail

              PATH="${runtimeBinPath}:''${PATH:-}"
              export PATH
              export UV_PYTHON="${pkgs.python312}/bin/python"
              export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
              export UV_CACHE_DIR="$PWD/.cache/uv"
              export LD_LIBRARY_PATH="${runtimeLibPath}:''${LD_LIBRARY_PATH:-}"

              # Parse mode argument (default: full)
              MODE="''${1:-full}"

              # Color output helpers
              RED='\033[0;31m'
              GREEN='\033[0;32m'
              YELLOW='\033[1;33m'
              BLUE='\033[0;34m'
              NC='\033[0m' # No Color

              echo ""
              echo -e "''${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━''${NC}"
              echo -e "''${BLUE}  slower-whisper CI ($MODE mode)''${NC}"
              echo -e "''${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━''${NC}"
              echo ""

              # Track failures
              FAILED_CHECKS=()

              # Helper: run a check and track result
              run_check() {
                local name="$1"
                shift
                echo -e "''${BLUE}▶ $name...''${NC}"
                if "$@"; then
                  echo -e "''${GREEN}✓ $name passed''${NC}"
                  echo ""
                  return 0
                else
                  echo -e "''${RED}✗ $name failed''${NC}"
                  echo ""
                  FAILED_CHECKS+=("$name")
                  return 1
                fi
              }

              # Ensure dependencies are installed
              echo -e "''${BLUE}▶ Checking Python dependencies...''${NC}"
              if [ ! -d ".venv" ]; then
                echo "No .venv found. Installing dependencies..."
                ${pkgs.uv}/bin/uv sync --extra dev --extra full --extra diarization
              else
                echo "✓ .venv exists (run 'uv sync' manually to update)"
              fi
              echo ""

              # Check 1: Lint
              run_check "Lint (ruff check)" \
                ${pkgs.uv}/bin/uv run ruff check \
                  transcription/ tests/ examples/ benchmarks/

              # Check 2: Format
              run_check "Format (ruff format --check)" \
                ${pkgs.uv}/bin/uv run ruff format --check \
                  transcription/ tests/ examples/ benchmarks/

              # Check 3: Type-check (typed surface)
              run_check "Type-check (mypy)" \
                ${pkgs.uv}/bin/uv run mypy \
                  transcription/ \
                  tests/test_llm_utils.py \
                  tests/test_writers.py \
                  tests/test_turn_helpers.py \
                  tests/test_audio_state_schema.py

              # Check 4: Fast tests
              run_check "Fast tests (pytest -m 'not slow and not heavy')" \
                ${pkgs.uv}/bin/uv run pytest \
                  --cov=transcription \
                  --cov-report=term-missing:skip-covered \
                  -v \
                  -m "not slow and not heavy" \
                  --tb=short

              # Stop here if mode=fast
              if [ "$MODE" = "fast" ]; then
                echo -e "''${YELLOW}⚡ Fast mode: skipping slow checks''${NC}"
                echo ""
              else
                # Check 5: Integration tests
                run_check "Integration tests" \
                  ${pkgs.uv}/bin/uv run pytest tests/test_*integration*.py -v --tb=short

                # Check 6: BDD library
                run_check "BDD library contract" \
                  ${pkgs.uv}/bin/uv run pytest tests/steps/ \
                    -v -m "not slow and not requires_gpu" --tb=short

                # Check 7: BDD API (if features/ dir exists)
                if [ -d "features" ]; then
                  run_check "BDD API contract" \
                    ${pkgs.uv}/bin/uv run pytest features/ \
                      -v -m "api and smoke" --tb=short
                fi

                # Check 8: Verify (requires HF_TOKEN for diarization models)
                if [ -n "''${HF_TOKEN:-}" ]; then
                  run_check "Verification suite (slower-whisper-verify --quick)" \
                    ${pkgs.uv}/bin/uv run slower-whisper-verify --quick
                else
                  echo -e "''${YELLOW}⚠ Skipping verify: HF_TOKEN not set''${NC}"
                  echo ""
                fi

                # Check 9: Dogfood smoke
                run_check "Dogfood smoke (synthetic audio)" \
                  ${pkgs.uv}/bin/uv run slower-whisper-dogfood \
                    --sample synthetic --skip-llm
              fi

              # Summary
              echo -e "''${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━''${NC}"
              if [ ''${#FAILED_CHECKS[@]} -eq 0 ]; then
                echo -e "''${GREEN}✓ All CI checks passed!''${NC}"
                echo ""
                exit 0
              else
                echo -e "''${RED}✗ ''${#FAILED_CHECKS[@]} check(s) failed:''${NC}"
                for check in "''${FAILED_CHECKS[@]}"; do
                  echo -e "  ''${RED}✗ $check''${NC}"
                done
                echo ""
                exit 1
              fi
            '');
          };

          # Dogfooding workflow (runs slower-whisper-dogfood)
          dogfood = {
            type = "app";
            program = toString (pkgs.writeShellScript "dogfood" ''
              set -euo pipefail
              PATH="${runtimeBinPath}:''${PATH:-}"
              export PATH
              export UV_PYTHON="${pkgs.python312}/bin/python"
              export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
              export UV_CACHE_DIR="$PWD/.cache/uv"
              export LD_LIBRARY_PATH="${runtimeLibPath}:''${LD_LIBRARY_PATH:-}"
              export SLOWER_WHISPER_CACHE_ROOT="''${SLOWER_WHISPER_CACHE_ROOT:-$HOME/.cache/slower-whisper}"
              exec ${pkgs.uv}/bin/uv run slower-whisper-dogfood "$@"
            '');
          };

          # Verification workflow (runs slower-whisper-verify)
          verify = {
            type = "app";
            program = toString (pkgs.writeShellScript "verify" ''
              set -euo pipefail
              PATH="${runtimeBinPath}:''${PATH:-}"
              export PATH
              export UV_PYTHON="${pkgs.python312}/bin/python"
              export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
              export UV_CACHE_DIR="$PWD/.cache/uv"
              export LD_LIBRARY_PATH="${runtimeLibPath}:''${LD_LIBRARY_PATH:-}"
              export SLOWER_WHISPER_CACHE_ROOT="''${SLOWER_WHISPER_CACHE_ROOT:-$HOME/.cache/slower-whisper}"
              exec ${pkgs.uv}/bin/uv run slower-whisper-verify "$@"
            '');
          };
        };

        # Default package (minimal, for nix build)
        packages.default = pkgs.python312Packages.buildPythonPackage {
          pname = "slower-whisper";
          version = "1.3.0";
          src = ./.;

          propagatedBuildInputs = systemDeps;

          # Note: This is a minimal package definition
          # For full packaging, consider poetry2nix or uv2nix
          doCheck = false;
        };
      }
    );
}
