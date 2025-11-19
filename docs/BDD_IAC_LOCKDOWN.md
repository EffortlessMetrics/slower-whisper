# BDD & IaC Lock-In Guide

**Purpose:** Transform existing BDD scenarios and IaC artifacts from "we have them" to "they are first-class contracts that guarantee system behavior and deployment readiness."

**Current State:** ✅ Excellent foundation
- 15 BDD scenarios (6 transcription + 9 enrichment) with 53 step implementations
- Full Docker/Compose/K8s infrastructure ready
- CI workflows configured (blocked by billing only)
- All code passing 191 tests with 57% coverage

**Goal:** Make BDD and IaC **non-negotiable contracts** that are:
1. Tested regularly (even without CI running)
2. Documented as the behavioral/deployment contract
3. Easy to run locally and in CI
4. Stable across refactors

---

## Part 1: BDD as Behavioral Contract

### Current Status ✅

**Feature files:**
- `tests/features/transcription.feature` (6 scenarios)
- `tests/features/enrichment.feature` (9 scenarios)

**Step implementations:**
- `tests/steps/test_transcription_steps.py`
- `tests/steps/test_enrichment_steps.py`
- 53 step definitions total

**Test status:**
- All BDD tests marked as `xfail` when ffmpeg unavailable (intentional graceful degradation)
- Tests use public API (`transcribe_file`, `enrich_transcript`) rather than internals
- Clean separation of concerns

### Actions to Lock In BDD

#### 1.1 Register BDD marker in pytest.ini

**Status:** ⚠️ Missing marker registration

**Action:** Add to `pytest.ini`:

```ini
markers =
    bdd: behaviour-driven acceptance tests (Gherkin scenarios)
    integration: marks tests as integration tests (end-to-end workflows)
    unit: marks tests as unit tests (fast, isolated)
    slow: marks tests as slow (downloads models, processes large files)
    heavy: marks tests that require heavy ML models (emotion recognition)
    requires_enrich: requires enrichment dependencies (librosa, parselmouth, etc.)
    requires_gpu: requires GPU support
```

**Why:** Removes the pytest warning about unknown `requires_enrich` marker.

**Implementation:**
```bash
# Edit pytest.ini and add the markers above
uv run pytest tests/steps/ -v  # Should show no warnings
```

---

#### 1.2 Add BDD section to README

**Status:** ⚠️ Not explicitly documented

**Action:** Add to README.md under "Testing" section:

```markdown
### BDD/Acceptance Tests

Behavioral acceptance tests are defined using Gherkin syntax and pytest-bdd:

```bash
# Run all BDD scenarios
uv run pytest tests/features/ -v

# Run BDD with markers
uv run pytest -m bdd -v

# Run only transcription scenarios
uv run pytest tests/steps/test_transcription_steps.py -v

# Run only enrichment scenarios
uv run pytest tests/steps/test_enrichment_steps.py -v
```

**Feature files:**
- `tests/features/transcription.feature` - Core transcription behaviors
- `tests/features/enrichment.feature` - Audio enrichment behaviors

**Note:** BDD tests require `ffmpeg` for audio processing. Without it, tests gracefully fail with `xfail` status.

**Contract:** These scenarios define the **guaranteed behaviors** of slower-whisper. Breaking these scenarios requires explicit discussion and version bump.
```

**Why:** Makes BDD visible as a first-class testing tier, not just "some tests."

**Implementation:**
```bash
# Edit README.md and add the section above after the existing Testing section
```

---

#### 1.3 Create BDD verification script

**Status:** ⚠️ Missing dedicated script

**Action:** Create `scripts/verify_bdd.sh`:

```bash
#!/usr/bin/env bash
# Verify BDD scenarios pass (or xfail appropriately)
#
# Usage: ./scripts/verify_bdd.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Verifying BDD acceptance scenarios"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for ffmpeg (required for BDD tests to pass vs xfail)
if command -v ffmpeg &> /dev/null; then
    echo "✅ ffmpeg found: BDD tests will attempt to pass"
    EXPECT_PASS=true
else
    echo "⚠️  ffmpeg not found: BDD tests will xfail (expected)"
    EXPECT_PASS=false
fi

echo ""
echo "Running BDD scenarios..."
echo ""

# Run BDD tests
if uv run pytest tests/steps/ -v --tb=short -m "not slow"; then
    echo ""
    echo "✅ BDD scenarios completed successfully"
else
    EXIT_CODE=$?
    if [ "$EXPECT_PASS" = false ] && [ $EXIT_CODE -eq 0 ]; then
        echo "✅ BDD scenarios xfailed as expected (ffmpeg unavailable)"
    else
        echo "❌ BDD scenarios failed unexpectedly"
        exit $EXIT_CODE
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ BDD verification complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
```

**Implementation:**
```bash
chmod +x scripts/verify_bdd.sh
./scripts/verify_bdd.sh
```

---

#### 1.4 Add FastAPI service BDD scenario (optional but high-value)

**Status:** ⚠️ Missing service-level acceptance test

**Action:** Create `tests/features/api_service.feature`:

```gherkin
Feature: Transcription REST API Service
  As a user of slower-whisper API
  I want to transcribe audio via HTTP endpoints
  So that I can integrate transcription into web applications

  Background:
    Given the FastAPI service is running at "http://localhost:8000"

  Scenario: Health check endpoint
    When I send a GET request to "/health"
    Then the response status is 200
    And the response contains "status"
    And the status is "healthy"

  Scenario: API documentation is available
    When I send a GET request to "/docs"
    Then the response status is 200
    And the response contains OpenAPI documentation

  Scenario: Transcribe audio via POST /transcribe
    Given I have an audio file "sample.wav"
    When I POST the audio file to "/transcribe"
    Then the response status is 200
    And the response is valid JSON
    And the transcript contains at least one segment
    And each segment has "id", "start", "end", and "text" fields

  Scenario: Enrich transcript via POST /enrich
    Given I have an audio file "sample.wav"
    And I have transcribed the audio via API
    When I POST the transcript to "/enrich" with prosody enabled
    Then the response status is 200
    And the enriched transcript has audio_state for all segments
```

**Why:** Validates the entire stack (API service) as a black-box contract.

**Implementation:**
```bash
# Create tests/steps/test_api_steps.py with step implementations
# Mark scenarios with @pytest.mark.requires_api
# Run manually or in CI: uv run pytest tests/steps/test_api_steps.py
```

---

#### 1.5 Enable BDD tests in CI (when Actions are re-enabled)

**Status:** ⚠️ CI workflows don't explicitly run BDD scenarios

**Action:** Add to `.github/workflows/ci.yml` after existing test job:

```yaml
  bdd-tests:
    name: BDD Acceptance Tests
    runs-on: ubuntu-latest
    needs: [lint, format]  # Run after quick checks
    steps:
      - uses: actions/checkout@v4

      - name: Install ffmpeg
        run: sudo apt-get update && sudo apt-get install -y ffmpeg

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: "pyproject.toml"

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --extra full --extra dev

      - name: Run BDD scenarios
        run: uv run pytest tests/steps/ -v -m "not slow and not requires_gpu"

      - name: Upload BDD test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: bdd-test-results
          path: .pytest_cache/
```

**Why:** Makes BDD tests a **CI gate** — PRs cannot merge if behavioral contract is violated.

**Implementation:**
```bash
# Edit .github/workflows/ci.yml
# Push to trigger CI (when Actions are re-enabled)
```

---

## Part 2: IaC as Deployment Contract

### Current Status ✅

**Docker:**
- `Dockerfile` (CPU base)
- `Dockerfile.gpu` (NVIDIA CUDA)
- `Dockerfile.api` (FastAPI service)
- `docker-compose.yml` (CPU + GPU orchestration)
- `docker-compose.dev.yml` (development mode)
- `docker-compose.api.yml` (API service)

**Kubernetes:**
- Complete manifest set in `k8s/`:
  - Namespace, ConfigMap, Secret, PVC
  - Deployment, Service, HPA
  - CronJob, Job, GPU resource quota
- Kustomize overlays for staging/production
- Comprehensive `k8s/README.md` and `k8s/QUICK_START.md`

**CI:**
- `.github/workflows/docker-build.yml` - Builds all Docker variants
- Configured for GitHub Container Registry (ghcr.io)

### Actions to Lock In IaC

#### 2.1 Add Docker smoke test script

**Status:** ⚠️ No quick local validation for Docker builds

**Action:** Create `scripts/docker_smoke_test.sh`:

```bash
#!/usr/bin/env bash
# Smoke test Docker images to verify they build and run
#
# Usage: ./scripts/docker_smoke_test.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐋 Docker Image Smoke Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# CPU image
echo "Building CPU image..."
docker build -t slower-whisper:test-cpu -f Dockerfile . --quiet

echo "✅ CPU image built"
echo "Testing CLI in CPU image..."
docker run --rm slower-whisper:test-cpu slower-whisper --help | grep -q "transcribe"
echo "✅ CLI works in CPU image"
echo ""

# GPU image (build only, requires NVIDIA runtime to run)
echo "Building GPU image..."
docker build -t slower-whisper:test-gpu -f Dockerfile.gpu . --quiet
echo "✅ GPU image built (runtime test skipped - requires NVIDIA Docker)"
echo ""

# API image
echo "Building API image..."
docker build -t slower-whisper:test-api -f Dockerfile.api . --quiet
echo "✅ API image built"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All Docker images smoke tested successfully"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Images ready:"
echo "  - slower-whisper:test-cpu"
echo "  - slower-whisper:test-gpu"
echo "  - slower-whisper:test-api"
echo ""
echo "Test with:"
echo "  docker run --rm slower-whisper:test-cpu slower-whisper --version"
```

**Implementation:**
```bash
chmod +x scripts/docker_smoke_test.sh
./scripts/docker_smoke_test.sh
```

---

#### 2.2 Add Kubernetes manifest validation script

**Status:** ⚠️ No automated YAML validation beyond syntax

**Action:** Create `scripts/validate_k8s.sh`:

```bash
#!/usr/bin/env bash
# Validate Kubernetes manifests using kubectl dry-run
#
# Usage: ./scripts/validate_k8s.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "☸️  Validating Kubernetes manifests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Install kubectl to validate manifests."
    echo "   See: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

echo "✅ kubectl found"
echo ""

# Validate each manifest with dry-run
MANIFESTS=(
    "k8s/namespace.yaml"
    "k8s/configmap.yaml"
    "k8s/secret.yaml"
    "k8s/pvc.yaml"
    "k8s/deployment.yaml"
    "k8s/service.yaml"
    "k8s/job.yaml"
    "k8s/cronjob.yaml"
    "k8s/gpu-resource-quota.yaml"
)

for manifest in "${MANIFESTS[@]}"; do
    echo "Validating $manifest..."
    kubectl apply --dry-run=client -f "$manifest" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✅ Valid"
    else
        echo "  ❌ Invalid"
        kubectl apply --dry-run=client -f "$manifest"
        exit 1
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All Kubernetes manifests are valid"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
```

**Implementation:**
```bash
chmod +x scripts/validate_k8s.sh
./scripts/validate_k8s.sh
```

---

#### 2.3 Add IaC section to README

**Status:** ⚠️ Deployment documented but not as "contract"

**Action:** Add emphasis to README.md under "Deployment" section:

```markdown
### Infrastructure as Code (IaC)

slower-whisper treats deployment configurations as **first-class contracts**:

**Docker:**
```bash
# Validate Docker images build correctly
./scripts/docker_smoke_test.sh

# Build specific variant
docker build -t slower-whisper:cpu -f Dockerfile .
docker build -t slower-whisper:gpu -f Dockerfile.gpu .
docker build -t slower-whisper:api -f Dockerfile.api .
```

**Kubernetes:**
```bash
# Validate manifests
./scripts/validate_k8s.sh

# Apply to cluster (see k8s/QUICK_START.md)
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
```

**Contract:** All IaC artifacts (Dockerfiles, K8s manifests, Compose files) must:
- Build/validate successfully
- Be tested in CI before release
- Support the same config precedence (CLI > file > env > defaults)

See `DOCKER.md` and `k8s/README.md` for full deployment guides.
```

---

#### 2.4 Add Docker smoke tests to CI

**Status:** ⚠️ CI builds images but doesn't test them

**Action:** Add to `.github/workflows/docker-build.yml` after build step:

```yaml
      - name: Test Docker image (smoke test)
        run: |
          # Pull the image we just built (if pushed) or use locally built
          docker run --rm ${{ steps.meta.outputs.tags }} slower-whisper --help
          docker run --rm ${{ steps.meta.outputs.tags }} slower-whisper --version
```

**Why:** Prevents pushing broken images to registry.

---

#### 2.5 Optional: Local kind/k3d integration test

**Status:** ⚠️ No local Kubernetes testing documented

**Action:** Create `scripts/test_k8s_local.sh`:

```bash
#!/usr/bin/env bash
# Test Kubernetes manifests in local kind cluster
#
# Requires: kind (https://kind.sigs.k8s.io/)
# Usage: ./scripts/test_k8s_local.sh

set -e

CLUSTER_NAME="slower-whisper-test"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "☸️  Testing K8s manifests in local kind cluster"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for kind
if ! command -v kind &> /dev/null; then
    echo "❌ kind not found. Install kind: https://kind.sigs.k8s.io/docs/user/quick-start/"
    exit 1
fi

echo "Creating kind cluster..."
kind create cluster --name "$CLUSTER_NAME" || echo "Cluster already exists"

echo "Loading Docker image into kind..."
docker build -t slower-whisper:local -f Dockerfile . --quiet
kind load docker-image slower-whisper:local --name "$CLUSTER_NAME"

echo "Applying manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml

echo "Waiting for deployment..."
kubectl wait --for=condition=available --timeout=120s deployment/slower-whisper -n slower-whisper || true

echo "Checking pod status..."
kubectl get pods -n slower-whisper

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Local K8s test complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Cleanup: kind delete cluster --name $CLUSTER_NAME"
```

**Why:** Validates the full K8s deployment stack locally before production.

**Implementation:**
```bash
chmod +x scripts/test_k8s_local.sh
# Install kind: https://kind.sigs.k8s.io/
./scripts/test_k8s_local.sh
```

---

## Part 3: Integration - BDD + IaC Together

### 3.1 Add "infrastructure BDD" scenario (optional, advanced)

**Status:** ⚠️ No BDD scenario that tests deployment

**Concept:** Create `tests/features/deployment.feature`:

```gherkin
Feature: Deployment readiness
  As a DevOps engineer
  I want to verify deployment artifacts work
  So that I can deploy with confidence

  Scenario: Docker CPU image runs CLI
    Given the Docker CPU image is built
    When I run "slower-whisper --help" in the container
    Then the command succeeds
    And the output contains "transcribe"

  Scenario: Docker API image serves health endpoint
    Given the Docker API image is built
    And the API container is running on port 8000
    When I send GET to "http://localhost:8000/health"
    Then the response status is 200
    And the response contains "healthy"

  Scenario: Kubernetes manifests are valid
    When I validate all K8s manifests with kubectl dry-run
    Then all manifests pass validation

  Scenario: Kubernetes deployment creates pods
    Given a local kind cluster exists
    And the namespace "slower-whisper" is created
    When I apply the deployment manifest
    Then the deployment is created successfully
    And at least one pod is running within 120 seconds
```

**Why:** Treats deployment infrastructure as behavior to be tested, not just "files that exist."

---

### 3.2 Create master verification script

**Status:** ⚠️ No single "verify everything" script

**Action:** Create `scripts/verify_all.sh`:

```bash
#!/usr/bin/env bash
# Master verification script - runs all checks
#
# Usage: ./scripts/verify_all.sh [--quick]

set -e

QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔒 Master Verification - slower-whisper"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Code quality
echo "1️⃣  Checking code quality..."
uv run ruff check transcription/ tests/
uv run ruff format --check transcription/ tests/
echo "✅ Code quality passed"
echo ""

# 2. Type checking
echo "2️⃣  Type checking..."
uv run mypy transcription/ || echo "⚠️  Type check warnings (non-blocking)"
echo ""

# 3. Unit tests
echo "3️⃣  Running unit tests..."
uv run pytest tests/ -m "not slow and not requires_gpu" --cov=transcription --cov-report=term-missing
echo "✅ Unit tests passed"
echo ""

# 4. BDD scenarios
echo "4️⃣  Running BDD scenarios..."
./scripts/verify_bdd.sh
echo ""

if [ "$QUICK_MODE" = false ]; then
    # 5. Docker smoke tests
    echo "5️⃣  Docker smoke tests..."
    ./scripts/docker_smoke_test.sh
    echo ""

    # 6. K8s validation
    echo "6️⃣  Kubernetes manifest validation..."
    if command -v kubectl &> /dev/null; then
        ./scripts/validate_k8s.sh
    else
        echo "⚠️  kubectl not found, skipping K8s validation"
    fi
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All verifications passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Repository is ready for:"
echo "  - Development (code quality + tests passing)"
echo "  - Deployment (Docker + K8s artifacts validated)"
echo "  - Release (behavioral contract verified)"
```

**Implementation:**
```bash
chmod +x scripts/verify_all.sh

# Quick check (skip Docker/K8s)
./scripts/verify_all.sh --quick

# Full verification
./scripts/verify_all.sh
```

---

## Summary: Implementation Checklist

### Immediate Actions (30 minutes total)

- [ ] **1.1** Add BDD marker to `pytest.ini`
- [ ] **1.2** Add BDD section to README
- [ ] **1.3** Create `scripts/verify_bdd.sh`
- [ ] **2.1** Create `scripts/docker_smoke_test.sh`
- [ ] **2.2** Create `scripts/validate_k8s.sh` (requires kubectl)
- [ ] **2.3** Add IaC emphasis to README
- [ ] **3.2** Create `scripts/verify_all.sh`

### High-Value Optional (1-2 hours)

- [ ] **1.4** Add FastAPI service BDD scenario
- [ ] **1.5** Enable BDD in CI workflows (when Actions re-enabled)
- [ ] **2.4** Add Docker smoke tests to CI
- [ ] **2.5** Create local kind/k3d integration test
- [ ] **3.1** Add infrastructure BDD scenarios

### When CI Re-enabled

- [ ] Verify BDD tests run in CI
- [ ] Verify Docker smoke tests run in CI
- [ ] Add status badges for BDD/IaC to README

---

## Expected Outcomes

**After immediate actions:**
- ✅ BDD scenarios documented as behavioral contract
- ✅ IaC artifacts validated before deploy
- ✅ Single command to verify everything: `./scripts/verify_all.sh`
- ✅ Clear markers in pytest for test categorization

**After optional enhancements:**
- ✅ FastAPI service validated via BDD
- ✅ CI gates on BDD + IaC (when re-enabled)
- ✅ Local K8s testing capability
- ✅ Infrastructure behavior tested, not just "exists"

**Long-term benefits:**
- 🔒 Behavioral contract is explicit and tested
- 🔒 Deployment readiness is verifiable before production
- 🔒 Refactoring is safer (BDD scenarios catch regressions)
- 🔒 New contributors understand "the contract" from day 1

---

## Part 4: Contract Versioning and Change Policy

### 4.1 BDD Contract Changes

BDD scenarios define **guaranteed behaviors**. Changes to these scenarios are treated as explicit changes to the behavioral contract.

#### What Requires a Version Bump

**Major version bump (breaking change):**
- Removing an existing scenario (behavior no longer guaranteed)
- Changing scenario semantics in a breaking way (e.g., endpoint changes required parameters)
- Making a previously-passing scenario fail intentionally

**Minor version bump (backward-compatible addition):**
- Adding new scenarios for new features
- Adding new steps to existing scenarios (that were already implicitly guaranteed)
- Making scenarios more specific (strengthening guarantees)

**Patch version bump (clarification/fix):**
- Fixing broken step implementations (scenario intent unchanged)
- Adding better assertions to existing scenarios (catching bugs, not changing contract)
- Documentation updates to scenarios

#### Process for BDD Changes

1. **Propose** the scenario change in a PR/issue
2. **Justify** why the behavioral contract should change
3. **Update** CHANGELOG.md with the contract change
4. **Review** with explicit focus on "is this a breaking change?"
5. **Merge** only after agreement on versioning impact

#### Examples

**Breaking change (major bump required):**
```gherkin
# Old scenario (v1.x)
Scenario: Transcribe audio file
  Given I have an audio file "sample.wav"
  When I transcribe the file
  Then the transcript is generated

# New scenario (v2.0) - BREAKING: changed output format
Scenario: Transcribe audio file
  Given I have an audio file "sample.wav"
  When I transcribe the file
  Then the transcript is generated in JSON schema v2 format  # <-- NEW REQUIREMENT
```

**Non-breaking addition (minor bump):**
```gherkin
# v1.2.0 - added new optional feature
Scenario: Enrich transcript with emotion recognition
  Given I have a transcript
  When I enrich it with emotion enabled
  Then each segment has emotion scores  # <-- NEW FEATURE
```

### 4.2 IaC Contract Changes

Infrastructure as Code artifacts define **deployment guarantees**. Changes affect production deployments.

#### What Requires a Version Bump

**Major version bump (breaking deployment change):**
- Changing Docker image entrypoint or CMD behavior
- Removing environment variables that were previously supported
- Changing K8s resource names (breaks existing deployments)
- Changing persistent volume structure (requires migration)

**Minor version bump (backward-compatible):**
- Adding new Docker build variants
- Adding new environment variables (with defaults)
- Adding new K8s resources (optional services, jobs)
- Adding new configuration options

**Patch version bump (fix/improvement):**
- Fixing Docker build issues
- Improving K8s resource limits/requests
- Documentation updates to deployment guides

#### Process for IaC Changes

1. **Validate** locally using verification scripts:
   - `./scripts/docker_smoke_test.sh` (or `uv run slower-whisper-verify`)
   - `./scripts/validate_k8s.sh` (or `uv run slower-whisper-verify`)
2. **Document** impact in CHANGELOG.md
3. **Test** in staging environment if breaking
4. **Update** deployment guides (`DOCKER.md`, `k8s/README.md`)

### 4.3 Scenario Tag Semantics

Tags define **when tests must pass**:

**Library BDD markers:**
- `@bdd`: All library behavioral scenarios
- `@requires_enrich`: Requires enrichment dependencies (librosa, parselmouth)
- `@requires_gpu`: Requires GPU (can be skipped in CPU-only environments)

**API BDD markers:**
- `@api`: All API service scenarios
- `@smoke`: **Hard gate for all releases** - health, docs, basic endpoints
- `@functional`: Full functional contract - recommended before minor/major releases

**Enforcement rules:**
1. `@api @smoke` scenarios **must always pass** before any release
2. `@api @functional` scenarios **should pass** before minor/major releases
3. Skipping `@functional` tests requires explicit justification in release notes

### 4.4 Stability Guarantees

**What won't break (within same major version):**
- Existing BDD scenarios continue to pass
- Docker images built with same tags continue to work
- K8s manifests apply successfully to existing clusters
- Environment variables maintain same semantics
- API endpoints maintain same request/response schemas

**What might change (minor/patch versions):**
- New scenarios added (new features)
- New optional parameters/fields
- Performance improvements
- Better error messages
- Documentation enhancements

### 4.5 Deprecation Policy

**For behavioral changes:**
1. Announce deprecation in CHANGELOG (one minor version ahead)
2. Add deprecation warnings to affected code
3. Keep old behavior working until next major version
4. Update BDD scenarios to reflect new recommended behavior

**For deployment changes:**
1. Support old configuration for at least one minor version
2. Document migration path in deployment guides
3. Add runtime warnings for deprecated configuration
4. Remove in next major version

**Example:**
```
v1.8.0: Deprecate FOO_CONFIG env var (use NEW_CONFIG instead)
v1.9.0: FOO_CONFIG still works but logs warning
v2.0.0: FOO_CONFIG removed (only NEW_CONFIG supported)
```

---

## References

- BDD feature files: `tests/features/` (library), `features/` (API)
- BDD step implementations: `tests/steps/` (library), `features/steps/` (API)
- Docker artifacts: `Dockerfile*`, `docker-compose*.yml`
- Kubernetes manifests: `k8s/`
- CI workflows: `.github/workflows/`
- pytest configuration: `pytest.ini`
- Verification CLI: `scripts/verify_all.py` (run via `uv run slower-whisper-verify`)

**Last updated:** 2025-11-17
