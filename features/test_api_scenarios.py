"""
Test module for API service BDD scenarios.

This module connects the Gherkin feature file (api_service.feature) to
the step definitions (steps/api_steps.py).

pytest-bdd requires this linkage file to discover and run scenarios.
"""

from __future__ import annotations

import httpx
import pytest
from pytest_bdd import scenarios

# Ensure step definitions are imported
pytest_plugins = ["features.steps.api_steps"]

# Skip the BDD suite if the API service isn't running locally
try:
    httpx.get("http://localhost:8765/health", timeout=1.0)
except Exception:
    pytest.skip("API service not running on http://localhost:8765", allow_module_level=True)

# Load all scenarios from the feature file
scenarios("api_service.feature")
