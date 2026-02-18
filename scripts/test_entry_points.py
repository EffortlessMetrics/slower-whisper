#!/usr/bin/env python3
"""
Test script to verify entry points after installation.

Usage: python test_entry_points.py
"""

import shutil
import subprocess
import sys


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color

    @classmethod
    def disable(cls):
        """Disable colors for Windows or non-TTY environments."""
        cls.GREEN = ""
        cls.RED = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.NC = ""


# Disable colors on Windows if not supported
if sys.platform == "win32" and not sys.stdout.isatty():
    Colors.disable()


def print_header(title):
    """Print a section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_section(title):
    """Print a subsection header."""
    print()
    print(title)
    print("-" * 60)


def test_command(test_name, command, check_output=False):
    """
    Run a test command and report results.

    Args:
        test_name: Human-readable test name
        command: Command to execute (string or list)
        check_output: If True, return stdout; otherwise return success status

    Returns:
        Tuple of (success: bool, output: str)
    """
    print(f"Testing: {test_name} ... ", end="", flush=True)

    try:
        if isinstance(command, str):
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        else:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
            )

        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ PASS{Colors.NC}")
            return True, result.stdout
        else:
            print(f"{Colors.RED}❌ FAIL{Colors.NC}")
            if result.stderr:
                print(f"  Error: {result.stderr.strip()[:100]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}❌ TIMEOUT{Colors.NC}")
        return False, "Command timed out"
    except Exception as e:
        print(f"{Colors.RED}❌ ERROR{Colors.NC}")
        print(f"  Exception: {str(e)[:100]}")
        return False, str(e)


def test_import(module_path):
    """Test if a Python module can be imported."""
    test_name = f"Import {module_path}"
    command = f"python3 -c 'import {module_path}'"
    return test_command(test_name, command)[0]


def main():
    """Run all entry point tests."""
    print_header("Entry Points Installation Test")

    passed = 0
    failed = 0

    # Test 1: Package installation
    print_section("📦 Package Installation")

    # Try pip show
    success, output = test_command(
        "Package installed (pip show)",
        [sys.executable, "-m", "pip", "show", "slower-whisper"],
    )
    if success:
        passed += 1
    else:
        failed += 1
        print(f"{Colors.YELLOW}  Note: Run 'pip install -e .' to install{Colors.NC}")

    # Test 2: Entry point commands exist
    print_section("🔧 Entry Point Commands")

    # Check slower-whisper
    slower_whisper_path = shutil.which("slower-whisper")
    if slower_whisper_path:
        print(f"{Colors.GREEN}✅ PASS{Colors.NC} - slower-whisper command found")
        print(f"  Location: {slower_whisper_path}")
        passed += 1
    else:
        print(f"{Colors.RED}❌ FAIL{Colors.NC} - slower-whisper command not found")
        failed += 1

    # Check slower-whisper-enrich
    enrich_path = shutil.which("slower-whisper-enrich")
    if enrich_path:
        print(f"{Colors.GREEN}✅ PASS{Colors.NC} - slower-whisper-enrich command found")
        print(f"  Location: {enrich_path}")
        passed += 1
    else:
        print(f"{Colors.RED}❌ FAIL{Colors.NC} - slower-whisper-enrich command not found")
        failed += 1

    # Test 3: Help output
    print_section("📖 Help Output")

    success, _ = test_command("slower-whisper --help", ["slower-whisper", "--help"])
    if success:
        passed += 1
    else:
        failed += 1

    success, _ = test_command(
        "slower-whisper-enrich --help",
        ["slower-whisper-enrich", "--help"],
    )
    if success:
        passed += 1
    else:
        failed += 1

    # Test 4: Module imports
    print_section("🐍 Python Module Imports")

    success = test_import("slower_whisper.pipeline.cli")
    if success:
        passed += 1
    else:
        failed += 1

    # Test 5: Dependencies
    print_section("📚 Dependencies")

    # Core dependencies
    success = test_import("faster_whisper")
    if success:
        passed += 1
    else:
        failed += 1
        print(f"{Colors.YELLOW}  Install with: pip install faster-whisper{Colors.NC}")

    # Optional dependencies
    print(f"\n{Colors.YELLOW}Optional dependencies (for audio enrichment):{Colors.NC}")

    optional_deps = [
        "soundfile",
        "librosa",
        "parselmouth",
        "torch",
        "transformers",
    ]

    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"{Colors.GREEN}✅{Colors.NC} {dep} (optional)")
        except ImportError:
            print(f"{Colors.YELLOW}⚠️{Colors.NC}  {dep} (optional - not installed)")

    # Summary
    print_header("Test Summary")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.NC}")
    print(f"Failed: {Colors.RED}{failed}{Colors.NC}")
    print()

    if failed == 0:
        print(f"{Colors.GREEN}✅ All tests passed!{Colors.NC}")
        print()
        print("You can now use the following commands:")
        print("  slower-whisper [OPTIONS]")
        print("  slower-whisper-enrich [OPTIONS]")
        print()
        print("Run with --help for usage information:")
        print("  slower-whisper --help")
        print("  slower-whisper-enrich --help")
        return 0
    else:
        print(f"{Colors.RED}❌ Some tests failed.{Colors.NC}")
        print()
        print("Troubleshooting:")
        print("1. Make sure the package is installed:")
        print("   pip install -e .")
        print("   or: uv sync")
        print()
        print("2. Verify you're in the correct virtual environment:")
        print(f"   which python  # Current: {sys.executable}")
        print("   pip list | grep slower-whisper")
        print()
        print("3. Try reinstalling:")
        print("   pip install -e . --force-reinstall")
        print("   or: uv sync --reinstall-package slower-whisper")
        return 1


if __name__ == "__main__":
    sys.exit(main())
