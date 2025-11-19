"""
Pytest configuration for BDD feature tests.

This module provides:
- API service startup/teardown fixtures
- Common test fixtures for BDD scenarios
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

# Check for httpx (required for API tests)
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# =============================================================================
# API Service Fixture
# =============================================================================


@pytest.fixture(scope="session")
def api_service_process():
    """
    Start the FastAPI service for the test session and tear it down afterward.

    The service runs on port 8765 (different from default 8000 to avoid conflicts).

    This fixture:
    1. Starts the uvicorn server in a subprocess
    2. Waits for the service to be ready (health check)
    3. Yields control to tests
    4. Tears down the service after all API tests complete

    Requires:
    - uvicorn to be installed
    - Port 8765 to be available
    """
    if not HTTPX_AVAILABLE:
        pytest.skip("httpx not installed (required for API tests)")

    # Check if uvicorn is available
    try:
        subprocess.run(
            ["uvicorn", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("uvicorn not installed (required for API tests)")

    # Start the service
    port = 8765
    host = "127.0.0.1"

    # Find the project root (features/ is at root level)
    features_dir = Path(__file__).parent
    project_root = features_dir.parent

    process = subprocess.Popen(
        [
            "uvicorn",
            "transcription.service:app",
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "warning",  # Reduce noise in test output
        ],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for the service to be ready
    base_url = f"http://{host}:{port}"
    max_wait = 10  # seconds
    start_time = time.time()
    ready = False

    while time.time() - start_time < max_wait:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    ready = True
                    break
        except (httpx.RequestError, httpx.TimeoutException):
            time.sleep(0.5)

    if not ready:
        process.terminate()
        process.wait(timeout=5)
        pytest.fail(
            f"API service did not become ready within {max_wait}s. "
            f"Check that port {port} is available and the service can start."
        )

    # Yield control to tests
    yield {"process": process, "base_url": base_url, "host": host, "port": port}

    # Teardown: stop the service
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


# =============================================================================
# Pytest BDD Hooks (if using pytest-bdd for feature files)
# =============================================================================

# Note: If you're using pytest-bdd, the scenarios will automatically use
# the api_service_process fixture if any step references it.
# The fixture scope is "session", so it starts once for all API tests.
