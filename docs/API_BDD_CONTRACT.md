# API Service Behavioral Contract

**Date:** 2025-11-17
**Commit:** `7c20f69`
**Status:** ✅ Complete

---

## Overview

The slower-whisper FastAPI service now has **explicit behavioral contracts** defined via BDD (Behavior-Driven Development) scenarios. These contracts guarantee specific REST API behaviors that cannot break without explicit discussion and version coordination.

This document describes:
- The API service behavioral contract
- How BDD scenarios enforce the contract
- How to run and extend API tests
- Integration with the verification pipeline

---

## Why API BDD Matters

### Before: Implicit Behaviors

Before this work, the FastAPI service had:
- ✅ Code that worked
- ✅ Some manual testing
- ❌ No automated acceptance tests
- ❌ No explicit behavioral guarantees
- ❌ No protection against regressions

### After: Explicit Behavioral Contracts

Now the API service has:
- ✅ **5 Gherkin scenarios** defining guaranteed behaviors
- ✅ **Black-box REST API tests** (service as a whole, not internals)
- ✅ **Automated verification** via `slower-whisper-verify`
- ✅ **Smoke tests** (health, docs) and **functional tests** (transcribe, enrich)
- ✅ **Same rigor as library BDD**: both levels contractually locked-in

---

## The Contract: 5 Guaranteed API Behaviors

All scenarios are defined in `features/api_service.feature` and enforced by `features/steps/api_steps.py`.

### Smoke Tests (Always Run)

#### 1. Health Check Endpoint Responds
```gherkin
@api @smoke
Scenario: Health check endpoint responds
  When I send a GET request to "/health"
  Then the response status code should be 200
  And the response should contain "status"
  And the response should contain "service"
  And the response should contain "schema_version"
```

**Contract:** The `/health` endpoint MUST return HTTP 200 with a JSON object containing `status`, `service`, and `schema_version` keys.

**Rationale:** Load balancers, monitoring tools, and orchestration systems depend on this endpoint for service discovery and health checks.

---

#### 2. OpenAPI Documentation is Available
```gherkin
@api @smoke
Scenario: OpenAPI documentation is available
  When I send a GET request to "/docs"
  Then the response status code should be 200
  And the response content type should be "text/html"
```

**Contract:** The `/docs` endpoint MUST return HTTP 200 with HTML content (Swagger UI).

**Rationale:** API consumers need interactive documentation to explore endpoints, try requests, and understand schemas.

---

#### 3. OpenAPI JSON Schema is Available
```gherkin
@api @smoke
Scenario: OpenAPI JSON schema is available
  When I send a GET request to "/openapi.json"
  Then the response status code should be 200
  And the response content type should be "application/json"
  And the response should contain "openapi"
  And the response should contain "paths"
```

**Contract:** The `/openapi.json` endpoint MUST return HTTP 200 with a valid OpenAPI 3.x JSON schema.

**Rationale:** Code generation tools, API clients, and testing frameworks need machine-readable API specifications.

---

### Functional Tests (Require Dependencies)

#### 4. Transcribe Endpoint Accepts Audio and Returns Transcript
```gherkin
@api @functional @requires_ffmpeg
Scenario: Transcribe endpoint accepts audio and returns transcript
  Given I have a sample audio file "test_audio.wav"
  When I POST the audio to "/transcribe" with model "base" and device "cpu"
  Then the response status code should be 200
  And the response should contain "segments"
  And the response should contain "schema_version"
  And the schema version should be 2
```

**Contract:** The `/transcribe` endpoint MUST:
- Accept multipart/form-data with an audio file
- Return HTTP 200 on success
- Return a JSON transcript with `segments` and `schema_version` keys
- Use schema version 2 (current stable schema)

**Dependencies:** Requires `ffmpeg` for audio normalization.

**Rationale:** This is the core functionality of the service. Breaking this breaks all API consumers doing transcription.

---

#### 5. Enrich Endpoint Adds Audio Features to Transcript
```gherkin
@api @functional @requires_enrich
Scenario: Enrich endpoint adds audio features to transcript
  Given I have a sample transcript file "test_transcript.json"
  And I have a sample audio file "test_audio.wav"
  When I POST both files to "/enrich" with prosody enabled
  Then the response status code should be 200
  And the response should contain "segments"
  And at least one segment should have "audio_state"
```

**Contract:** The `/enrich` endpoint MUST:
- Accept multipart/form-data with transcript JSON and audio WAV
- Return HTTP 200 on success
- Return an enriched transcript where at least one segment has `audio_state` populated

**Dependencies:** Requires enrichment dependencies (librosa, parselmouth, etc.).

**Rationale:** This is the value-add feature that distinguishes slower-whisper. Breaking it breaks the audio enrichment workflow.

---

## How It Works: Black-Box Testing

### Service Lifecycle

The BDD tests treat the FastAPI service as a **black box**:

1. **Startup** (session-scoped fixture in `features/conftest.py`):
   - Starts `uvicorn` on port 8765
   - Waits for `/health` to respond with HTTP 200
   - Fails fast if service doesn't start within 10 seconds

2. **Testing** (scenarios in `features/api_service.feature`):
   - Uses `httpx.Client` to make real HTTP requests
   - Verifies response status codes, headers, and JSON content
   - Generates test audio/transcript files with `ffmpeg`

3. **Teardown** (automatic after all tests):
   - Sends SIGTERM to uvicorn process
   - Waits up to 5 seconds for graceful shutdown
   - Kills process if necessary

### Why Black-Box?

- ✅ **Tests the real API surface** that clients see
- ✅ **Catches integration issues** (routing, serialization, error handling)
- ✅ **Decoupled from internal implementation** (can refactor service internals without breaking tests)
- ✅ **Realistic test environment** (HTTP server, network serialization, etc.)

---

## Running API BDD Tests

### Prerequisites

```bash
# Install development dependencies (includes httpx, uvicorn)
uv sync --extra dev
```

### Run All API Scenarios

```bash
# Run all 5 API scenarios
uv run pytest features/ -v -m api

# Alternative: use verification CLI
uv run slower-whisper-verify --quick  # Includes API tests
```

### Run Specific Scenario Categories

```bash
# Smoke tests only (health, docs) - no dependencies required
uv run pytest features/ -v -m "api and smoke"

# Functional tests only (transcribe, enrich) - requires ffmpeg + enrichment deps
uv run pytest features/ -v -m "api and functional"
```

### Skip API Tests

```bash
# Skip API tests (useful if httpx/uvicorn not installed)
uv run slower-whisper-verify --skip-api
```

---

## Integration with Verification Pipeline

The API BDD tests are step 4 in the verification pipeline:

```bash
uv run slower-whisper-verify
```

**Pipeline steps:**
1. ✅ Code quality (ruff)
2. ✅ Fast unit tests (pytest)
3. ✅ Library BDD scenarios (transcription, enrichment)
4. ✅ **API BDD scenarios** ← NEW
5. ✅ Docker smoke tests (if not --quick)
6. ✅ K8s manifest validation (if not --quick)

**Quick mode** (skips Docker and K8s but includes API):
```bash
uv run slower-whisper-verify --quick
```

---

## Extending the API Contract

### Adding a New Endpoint

If you add a new API endpoint (e.g., `/v2/batch-transcribe`), follow these steps:

#### 1. Design the Gherkin Scenario

Add to `features/api_service.feature`:

```gherkin
@api @functional
Scenario: Batch transcribe endpoint processes multiple files
  Given I have 3 sample audio files
  When I POST the files to "/v2/batch-transcribe" with model "base"
  Then the response status code should be 200
  And the response should contain "results"
  And the results should have 3 transcripts
```

#### 2. Implement the Step Definitions

Add to `features/steps/api_steps.py`:

```python
@given("I have 3 sample audio files", target_fixture="sample_audio_batch")
def sample_audio_batch(api_context: dict, tmp_path: Path) -> list[Path]:
    # Generate 3 test audio files
    ...

@when(parsers.parse('I POST the files to "{endpoint}" with model "{model}"'))
def post_batch_transcribe(api_context: dict, endpoint: str, model: str):
    # Upload multiple files
    ...

@then(parsers.parse("the results should have {count:d} transcripts"))
def check_result_count(api_context: dict, count: int):
    data = api_context["last_response"].json()
    assert len(data["results"]) == count
```

#### 3. Update the Contract Documentation

Add the new scenario to this document's "The Contract" section with:
- Scenario description
- Contract statement (what MUST happen)
- Dependencies (if any)
- Rationale (why this behavior is guaranteed)

#### 4. Verify the Contract

```bash
# Run the new scenario
uv run pytest features/ -v -k "batch_transcribe"

# Run full verification
uv run slower-whisper-verify --quick
```

---

## API BDD vs Library BDD

### Library BDD (`tests/features/`)

**Focus:** Python API behaviors (transcribe_file, enrich_transcript, etc.)

**Test type:** Unit/integration tests with direct function calls

**Fixtures:** In-memory transcripts, mock audio files

**Example:**
```python
transcript = transcribe_file(audio_path, root, config)
assert len(transcript.segments) > 0
```

### API BDD (`features/`)

**Focus:** REST API behaviors (GET /health, POST /transcribe, etc.)

**Test type:** Black-box HTTP tests with real service

**Fixtures:** HTTP client, running uvicorn server, real audio files

**Example:**
```python
response = client.post("/transcribe", files={"audio": audio_file})
assert response.status_code == 200
```

### Why Both?

- **Library BDD:** Ensures the core Python package is correct
- **API BDD:** Ensures the HTTP wrapper around the package is correct
- **Together:** Complete behavioral coverage of both interfaces

---

## Contract Stability Guarantees

### What Changes Require a Version Bump?

**Breaking changes** (require major version bump):
- Removing an endpoint
- Changing required request parameters
- Changing response schema structure
- Changing HTTP status codes for existing behaviors

**Non-breaking changes** (can be done in minor/patch releases):
- Adding new endpoints
- Adding optional request parameters
- Adding new response fields
- Adding new error messages

### How Breaking Changes Are Handled

1. Update the Gherkin scenario to reflect new behavior
2. Verify all scenarios still pass (or update failing scenarios)
3. Update this contract document
4. Bump API version (if using versioned endpoints like `/v2/...`)
5. Document migration path in CHANGELOG

---

## Troubleshooting

### API Service Won't Start

**Error:** `API service did not become ready within 10s`

**Causes:**
- Port 8765 already in use
- uvicorn not installed
- Import errors in transcription.service module

**Solutions:**
```bash
# Check port availability
lsof -i :8765

# Ensure uvicorn installed
uv sync --extra dev

# Test service manually
uv run uvicorn transcription.service:app --port 8765
```

---

### Scenarios Failing with httpx Errors

**Error:** `httpx not installed (required for API tests)`

**Solution:**
```bash
uv sync --extra dev
```

---

### Functional Tests Skipping

**Error:** Scenarios marked `@requires_ffmpeg` or `@requires_enrich` are skipped

**Cause:** Missing dependencies (ffmpeg, librosa, parselmouth, etc.)

**Solutions:**
```bash
# Install ffmpeg (system dependency)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Install enrichment dependencies
uv sync --extra full
```

---

## Summary

✅ **5 API BDD scenarios** define the behavioral contract
✅ **Black-box testing** via real HTTP requests
✅ **Smoke tests** (health, docs) always run
✅ **Functional tests** (transcribe, enrich) require dependencies
✅ **Integrated with verification CLI** (`slower-whisper-verify`)
✅ **Same rigor as library BDD**: both interfaces contractually locked-in

**Bottom line:** The slower-whisper API service now has **explicit, testable, versioned behavioral guarantees**. Breaking these contracts requires intentional discussion and coordination, not accidental regressions.
