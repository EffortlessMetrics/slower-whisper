"""
Test module for API service BDD scenarios.

This module connects the Gherkin feature file (api_service.feature) to
the step definitions (steps/api_steps.py).

pytest-bdd requires this linkage file to discover and run scenarios.
"""

from __future__ import annotations

from pytest_bdd import scenarios

# Load all scenarios from the feature file
scenarios("api_service.feature")
