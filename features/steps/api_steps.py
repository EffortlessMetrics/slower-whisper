"""
Step definitions for API service BDD scenarios.

These steps test the FastAPI service as a black-box REST API,
verifying that endpoints respond correctly and return valid data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, then, when

# API client (httpx) - lazy import to avoid dependency issues
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def api_base_url() -> str:
    """
    Base URL for the API service.

    The service is expected to be running (started by conftest.py fixture).
    """
    return "http://localhost:8765"


@pytest.fixture(scope="module")
def http_client(api_base_url: str):
    """
    HTTP client for making API requests.

    Uses httpx for async-compatible requests.
    """
    if not HTTPX_AVAILABLE:
        pytest.skip("httpx not installed (required for API tests)")

    with httpx.Client(base_url=api_base_url, timeout=30.0) as client:
        yield client


# =============================================================================
# Background Steps
# =============================================================================


@given("the API service is running", target_fixture="api_context")
def api_service_running(api_base_url: str, http_client: httpx.Client) -> dict:
    """
    Verify the API service is running and accessible.

    This step checks that the service responds to health checks.
    The actual service startup is handled by conftest.py fixtures.
    """
    try:
        response = http_client.get("/health", timeout=5.0)
        if response.status_code != 200:
            pytest.fail(
                f"API service health check failed with status {response.status_code}. "
                f"Ensure the service is running at {api_base_url}."
            )
    except httpx.RequestError as e:
        pytest.fail(
            f"Cannot connect to API service at {api_base_url}. "
            f"Error: {e}. "
            f"Ensure the service is started before running API tests."
        )

    # Return context dict to be used by other steps
    return {
        "base_url": api_base_url,
        "client": http_client,
        "last_response": None,
        "test_files": {},
    }


# =============================================================================
# Given Steps (Preconditions)
# =============================================================================


@given(parsers.parse('I have a sample audio file "{filename}"'), target_fixture="sample_audio")
def sample_audio_file(api_context: dict, filename: str, tmp_path: Path) -> Path:
    """
    Create a minimal valid audio file for testing.

    Uses ffmpeg to generate a 1-second sine wave WAV file.
    """
    audio_path = tmp_path / filename

    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available (required for audio file generation)")

    # Generate a 1-second sine wave at 440Hz (A4 note)
    import subprocess

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to generate test audio: {e.stderr.decode()}")

    api_context["test_files"]["audio"] = audio_path
    return audio_path


@given(
    parsers.parse('I have a sample transcript file "{filename}"'),
    target_fixture="sample_transcript",
)
def sample_transcript_file(api_context: dict, filename: str, tmp_path: Path) -> Path:
    """
    Create a minimal valid transcript JSON for testing enrichment.

    This is a schema v2 transcript with one segment.
    """
    transcript_data = {
        "schema_version": 2,
        "file_name": "test_audio.wav",
        "language": "en",
        "meta": {
            "generated_at": "2025-01-01T00:00:00Z",
            "model_name": "base",
            "device": "cpu",
        },
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "This is a test.",
                "speaker": None,
                "tone": None,
                "audio_state": None,
            }
        ],
    }

    transcript_path = tmp_path / filename
    transcript_path.write_text(json.dumps(transcript_data, indent=2))

    api_context["test_files"]["transcript"] = transcript_path
    return transcript_path


# =============================================================================
# When Steps (Actions)
# =============================================================================


@when(parsers.parse('I send a GET request to "{endpoint}"'))
def get_request(api_context: dict, endpoint: str):
    """
    Send a GET request to the specified endpoint.
    """
    client: httpx.Client = api_context["client"]
    response = client.get(endpoint)
    api_context["last_response"] = response


@when(parsers.parse('I POST the audio to "{endpoint}" with model "{model}" and device "{device}"'))
def post_audio_transcribe(api_context: dict, endpoint: str, model: str, device: str):
    """
    POST an audio file to the transcribe endpoint.
    """
    client: httpx.Client = api_context["client"]
    audio_path = api_context["test_files"].get("audio")

    if not audio_path:
        pytest.fail("No audio file available. Ensure 'Given I have a sample audio file' ran.")

    with open(audio_path, "rb") as audio_file:
        files = {"audio": (audio_path.name, audio_file, "audio/wav")}
        params = {"model": model, "device": device}
        response = client.post(endpoint, files=files, params=params)

    api_context["last_response"] = response


@when(parsers.parse('I POST both files to "{endpoint}" with prosody enabled'))
def post_enrich(api_context: dict, endpoint: str):
    """
    POST transcript and audio files to the enrich endpoint.
    """
    client: httpx.Client = api_context["client"]
    transcript_path = api_context["test_files"].get("transcript")
    audio_path = api_context["test_files"].get("audio")

    if not transcript_path:
        pytest.fail(
            "No transcript file available. Ensure 'Given I have a sample transcript file' ran."
        )
    if not audio_path:
        pytest.fail("No audio file available. Ensure 'Given I have a sample audio file' ran.")

    with (
        open(transcript_path, "rb") as transcript_file,
        open(audio_path, "rb") as audio_file,
    ):
        files = {
            "transcript": (transcript_path.name, transcript_file, "application/json"),
            "audio": (audio_path.name, audio_file, "audio/wav"),
        }
        params = {"enable_prosody": "true", "enable_emotion": "false", "device": "cpu"}
        response = client.post(endpoint, files=files, params=params)

    api_context["last_response"] = response


# =============================================================================
# Then Steps (Assertions)
# =============================================================================


@then(parsers.parse("the response status code should be {status_code:d}"))
def check_status_code(api_context: dict, status_code: int):
    """
    Verify the HTTP response status code.
    """
    response = api_context["last_response"]
    assert response is not None, "No response available. Ensure a request was made."

    actual_status = response.status_code
    assert actual_status == status_code, (
        f"Expected status code {status_code}, got {actual_status}. "
        f"Response body: {response.text[:200]}"
    )


@then(parsers.parse('the response should contain "{key}"'))
def check_response_contains_key(api_context: dict, key: str):
    """
    Verify the response JSON contains a specific key.
    """
    response = api_context["last_response"]
    assert response is not None, "No response available."

    try:
        data = response.json()
    except Exception as e:
        pytest.fail(f"Response is not valid JSON: {e}. Body: {response.text[:200]}")

    assert key in data, (
        f"Response JSON does not contain key '{key}'. Keys present: {list(data.keys())}"
    )


@then(parsers.parse('the response content type should be "{content_type}"'))
def check_content_type(api_context: dict, content_type: str):
    """
    Verify the response Content-Type header.
    """
    response = api_context["last_response"]
    assert response is not None, "No response available."

    actual_content_type = response.headers.get("content-type", "")
    assert content_type in actual_content_type, (
        f"Expected content type to contain '{content_type}', got '{actual_content_type}'"
    )


@then(parsers.parse("the schema version should be {version:d}"))
def check_schema_version(api_context: dict, version: int):
    """
    Verify the transcript schema version in the response.
    """
    response = api_context["last_response"]
    assert response is not None, "No response available."

    data = response.json()
    actual_version = data.get("schema_version")

    assert actual_version == version, (
        f"Expected schema version {version}, got {actual_version}. "
        f"Ensure the API returns schema v{version} transcripts."
    )


@then('at least one segment should have "audio_state"')
def check_audio_state_present(api_context: dict):
    """
    Verify that at least one segment has audio_state populated.
    """
    response = api_context["last_response"]
    assert response is not None, "No response available."

    data = response.json()
    segments = data.get("segments", [])

    assert segments, "Response contains no segments."

    has_audio_state = any(seg.get("audio_state") is not None for seg in segments)

    assert has_audio_state, (
        "None of the segments have audio_state populated. "
        "Enrichment may have failed or been skipped."
    )
