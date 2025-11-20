#!/usr/bin/env python3
"""Verify IEMOCAP integration is correctly set up (without requiring dataset).

This script checks that all IEMOCAP components are properly integrated:
- iter_iemocap_clips() function exists and is callable
- Emotion models can be imported
- Label mapping is defined
- Evaluation script is executable
- Documentation files exist

Usage:
    python benchmarks/verify_iemocap_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_iterator():
    """Check that iter_iemocap_clips() is available."""
    print("Checking iter_iemocap_clips()...", end=" ")
    try:
        from transcription.benchmarks import iter_iemocap_clips

        # Verify it's callable
        assert callable(iter_iemocap_clips)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def check_emotion_models():
    """Check that emotion extraction functions are available."""
    print("Checking emotion models...", end=" ")
    try:
        from transcription.emotion import (
            EMOTION_AVAILABLE,
            extract_emotion_categorical,
            extract_emotion_dimensional,
        )

        if not EMOTION_AVAILABLE:
            print("⚠ Available but dependencies not installed (run: uv sync --extra emotion)")
            return True  # Not an error, just needs installation

        assert callable(extract_emotion_categorical)
        assert callable(extract_emotion_dimensional)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def check_label_mapping():
    """Check that label mapping constants are defined."""
    print("Checking label mapping...", end=" ")
    try:
        # Import eval_emotion module
        sys.path.insert(0, str(Path(__file__).parent))
        import eval_emotion

        assert hasattr(eval_emotion, "IEMOCAP_EMOTION_MAP")
        assert hasattr(eval_emotion, "IEMOCAP_TO_MODEL_CATEGORICAL")

        # Verify mappings are non-empty
        assert len(eval_emotion.IEMOCAP_EMOTION_MAP) > 0
        assert len(eval_emotion.IEMOCAP_TO_MODEL_CATEGORICAL) > 0

        # Verify key emotions are mapped
        assert "ang" in eval_emotion.IEMOCAP_TO_MODEL_CATEGORICAL
        assert "hap" in eval_emotion.IEMOCAP_TO_MODEL_CATEGORICAL
        assert "neu" in eval_emotion.IEMOCAP_TO_MODEL_CATEGORICAL

        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def check_eval_script():
    """Check that eval_emotion.py exists and is executable."""
    print("Checking eval_emotion.py...", end=" ")
    try:
        eval_script = Path(__file__).parent / "eval_emotion.py"
        assert eval_script.exists(), f"Script not found: {eval_script}"
        assert eval_script.stat().st_mode & 0o111, f"Script not executable: {eval_script}"
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def check_documentation():
    """Check that documentation files exist."""
    print("Checking documentation...", end=" ")
    docs_dir = Path(__file__).parent.parent / "docs"

    required_docs = [
        "IEMOCAP_SETUP.md",
        "IEMOCAP_LABEL_MAPPING.md",
        "IEMOCAP_QUICKREF.md",
    ]

    missing = []
    for doc in required_docs:
        doc_path = docs_dir / doc
        if not doc_path.exists():
            missing.append(doc)

    if missing:
        print(f"✗ Missing: {', '.join(missing)}")
        return False

    print("✓")
    return True


def check_benchmarks_integration():
    """Check that IEMOCAP is registered in benchmarks."""
    print("Checking benchmarks integration...", end=" ")
    try:
        from transcription.benchmarks import list_available_benchmarks

        benchmarks = list_available_benchmarks()
        assert "iemocap" in benchmarks, "IEMOCAP not in benchmarks list"

        iemocap_info = benchmarks["iemocap"]
        assert "setup_doc" in iemocap_info
        assert "description" in iemocap_info
        assert "tasks" in iemocap_info
        assert "emotion" in iemocap_info["tasks"]

        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def check_cache_paths():
    """Check that CachePaths includes benchmarks_root."""
    print("Checking cache paths...", end=" ")
    try:
        from transcription.cache import CachePaths

        paths = CachePaths.from_env()
        assert hasattr(paths, "benchmarks_root")
        assert paths.benchmarks_root is not None

        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("IEMOCAP Integration Verification")
    print("=" * 60)
    print()

    checks = [
        ("Cache Paths", check_cache_paths),
        ("Iterator Function", check_iterator),
        ("Emotion Models", check_emotion_models),
        ("Label Mapping", check_label_mapping),
        ("Evaluation Script", check_eval_script),
        ("Documentation", check_documentation),
        ("Benchmarks Integration", check_benchmarks_integration),
    ]

    results = []
    for _name, check_func in checks:
        results.append(check_func())

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All checks passed ({passed}/{total})")
        print()
        print("Next steps:")
        print("  1. Obtain IEMOCAP dataset from https://sail.usc.edu/iemocap/")
        print("  2. Stage dataset at ~/.cache/slower-whisper/benchmarks/iemocap/")
        print("  3. Run: uv run python benchmarks/eval_emotion.py --limit 10")
        print()
        return 0
    else:
        failed = total - passed
        print(f"✗ {failed} check(s) failed ({passed}/{total} passed)")
        print()
        print("Please fix the failing checks before proceeding.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
