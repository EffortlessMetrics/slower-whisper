# tests/test_versioning.py
"""Guardrail test to prevent version sync issues like v1.9.1."""

from __future__ import annotations

import importlib.metadata as md

import pytest

import slower_whisper.pipeline


def test_runtime_version_matches_package_metadata() -> None:
    """Ensure runtime __version__ matches installed package metadata.

    This test prevents regressions where a hardcoded version constant
    could drift from pyproject.toml. When the package is installed,
    both must report the same version.
    """
    try:
        expected = md.version("slower-whisper")
    except md.PackageNotFoundError:
        pytest.skip("slower-whisper not installed in this environment")
    assert slower_whisper.pipeline.__version__ == expected


def test_version_is_not_dev_when_installed() -> None:
    """Ensure we don't accidentally ship with dev version string."""
    try:
        _ = md.version("slower-whisper")
    except md.PackageNotFoundError:
        pytest.skip("slower-whisper not installed in this environment")
    # If package is installed, version should not be the fallback
    assert slower_whisper.pipeline.__version__ != "0.0.0-dev"
