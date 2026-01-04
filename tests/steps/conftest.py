"""
Pytest configuration for BDD tests.

This module provides fixtures and configuration specific to pytest-bdd tests.
All BDD tests are marked as integration tests by default.
"""

from __future__ import annotations

import pytest
from pytest import Config, Item


def pytest_configure(config: Config) -> None:
    """Register custom markers for BDD tests."""
    config.addinivalue_line("markers", "bdd: mark test as a BDD scenario")


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """
    Automatically mark BDD tests with appropriate markers.

    All tests in the steps/ directory are:
    - Integration tests
    - BDD tests
    """
    for item in items:
        # Mark all BDD tests as integration tests
        if "test_transcription_steps" in item.nodeid or "test_enrichment_steps" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.bdd)

            # Mark enrichment tests as requiring enrichment dependencies
            if "test_enrichment_steps" in item.nodeid:
                item.add_marker(pytest.mark.requires_enrich)
