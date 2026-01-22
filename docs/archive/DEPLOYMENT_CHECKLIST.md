# Deployment Verification Checklist

This checklist ensures slower-whisper is deployment-ready across all environments: local, Docker, Kubernetes, and CI/CD pipelines.

**Version:** 1.0.0
**Last Updated:** 2025-11-15

---

## Table of Contents

- [Pre-Deployment Verification](#pre-deployment-verification)
- [Docker Deployment Verification](#docker-deployment-verification)
- [Kubernetes Deployment Verification](#kubernetes-deployment-verification)
- [CI/CD Pipeline Verification](#cicd-pipeline-verification)
- [Documentation Verification](#documentation-verification)
- [Release Readiness](#release-readiness)
- [Post-Deployment Verification](#post-deployment-verification)

---

## Pre-Deployment Verification

### Code Quality

- [ ] **All tests passing**
  ```bash
  # Run full test suite
  uv run pytest -v

  # Expected: All tests pass or skip (no failures)
  # Acceptable: 52+ tests passing, 5 skipped (optional deps), 1 known prosody issue
  ```

- [ ] **Linting clean (ruff check)**
  ```bash
  # Check linting
  uv run ruff check transcription/ tests/ examples/ benchmarks/

  # Expected: No errors
  # Auto-fix minor issues:
  uv run ruff check --fix transcription/ tests/ examples/ benchmarks/
  ```

- [ ] **Formatting clean (ruff format)**
  ```bash
  # Check formatting
  uv run ruff format --check transcription/ tests/ examples/ benchmarks/

  # Expected: All files formatted correctly
  # Auto-fix:
  uv run ruff format transcription/ tests/ examples/ benchmarks/
  ```

- [ ] **Type checking clean (mypy)**
  ```bash
  # Run type checking
  uv run mypy transcription/ tests/

  # Expected: No critical errors (warnings acceptable)
  # Note: Third-party libraries (faster-whisper, librosa, transformers) are ignored
  ```

- [ ] **Pre-commit hooks passing**
  ```bash
  # Install and run pre-commit
  uv run pre-commit install
  uv run pre-commit run --all-files

  # Expected: All hooks pass
  ```

- [ ] **No uncommitted changes**
  ```bash
  git status

  # Expected: Working tree clean (or only deployment artifacts)
  ```

### Configuration Files

- [ ] **`pyproject.toml` valid**
  ```bash
  # Validate TOML syntax
  python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"

  # Check package metadata
  uv run python -c "from importlib.metadata import metadata; print(metadata('slower-whisper'))"

  # Expected: No syntax errors, metadata loads correctly
  ```

- [ ] **Dependency groups defined correctly**
  - `base`: faster-whisper only (~2.5GB)
  - `enrich-basic`: soundfile, numpy, librosa (~1GB additional)
  - `enrich-prosody`: praat-parselmouth (~36MB additional)
  - `emotion`: torch, transformers (~4GB additional)
  - `full`: enrich-prosody + emotion (~4GB total additional)
  - `dev`: full + testing/linting/docs tools
  - `api`: FastAPI service dependencies

- [ ] **Entry points configured**
  - `slower-whisper` → `transcription.cli:main` (unified CLI)

### Version Numbers

- [ ] **Version in `pyproject.toml` updated**
  ```bash
  grep 'version = ' pyproject.toml

  # Expected: version = "1.0.0" (or appropriate version)
  ```

- [ ] **Version in `transcription/__init__.py` matches**
  ```bash
  grep '__version__' transcription/__init__.py

  # Expected: __version__ = "1.0.0"
  ```

- [ ] **Schema versions documented**
  - `SCHEMA_VERSION = 2` in `transcription/models.py`
  - `AUDIO_STATE_VERSION = "1.0.0"` in `transcription/models.py`

### Documentation Updated

- [ ] **`CHANGELOG.md` includes latest changes**
  - Version number matches `pyproject.toml`
  - All major changes documented
  - Migration guide provided (if breaking changes)

- [ ] **`README.md` accurate**
  - Quick Start instructions work
  - Installation commands correct
  - Examples functional
  - Links not broken

- [ ] **API documentation matches implementation**
  - `API_QUICK_REFERENCE.md` up to date
  - Function signatures match actual code
  - Example code runs without errors

---

## Docker Deployment Verification

### Docker Images Build Successfully

- [ ] **CPU base image builds**
  ```bash
  docker build --target runtime-cpu --build-arg INSTALL_MODE=base -t slower-whisper:cpu-base .

  # Expected: Build succeeds, image size ~2.5GB
  ```

- [ ] **CPU full image builds**
  ```bash
  docker build --target runtime-cpu --build-arg INSTALL_MODE=enrich -t slower-whisper:cpu .

  # Expected: Build succeeds, image size ~3.5GB
  ```

- [ ] **GPU base image builds**
  ```bash
  docker build -f Dockerfile.gpu --target runtime-gpu --build-arg INSTALL_MODE=base -t slower-whisper:gpu-base .

  # Expected: Build succeeds, image size ~4GB (includes CUDA runtime)
  ```

- [ ] **GPU full image builds**
  ```bash
  docker build -f Dockerfile.gpu --target runtime-gpu --build-arg INSTALL_MODE=enrich -t slower-whisper:gpu .

  # Expected: Build succeeds, image size ~7GB (includes CUDA + ML models)
  ```

- [ ] **API service image builds**
  ```bash
  docker build -f Dockerfile.api -t slower-whisper:api .

  # Expected: Build succeeds, includes FastAPI dependencies
  ```

### Containers Run and Execute

- [ ] **CPU container runs help command**
  ```bash
  docker run --rm slower-whisper:cpu slower-whisper --help

  # Expected: Help text displays, no errors
  ```

- [ ] **GPU container runs help command**
  ```bash
  docker run --rm --gpus all slower-whisper:gpu slower-whisper --help

  # Expected: Help text displays, CUDA detected
  ```

- [ ] **Container can access volumes**
  ```bash
  # Create test directory
  mkdir -p test_audio
  touch test_audio/sample.mp3

  # Run container with volume mount
  docker run --rm -v $(pwd)/test_audio:/data slower-whisper:cpu ls /data

  # Expected: Lists sample.mp3
  ```

- [ ] **API service container starts**
  ```bash
  docker run -d -p 8000:8000 --name test-api slower-whisper:api
  curl http://localhost:8000/health
  docker stop test-api && docker rm test-api

  # Expected: Returns {"status": "healthy"}
  ```

### Docker Compose Services

- [ ] **Development compose starts**
  ```bash
  docker-compose -f docker-compose.dev.yml up -d
  docker-compose -f docker-compose.dev.yml ps

  # Expected: All services running
  docker-compose -f docker-compose.dev.yml down
  ```

- [ ] **Production compose starts (CPU)**
  ```bash
  docker-compose up -d slower-whisper-cpu
  docker-compose ps

  # Expected: Service running, healthy
  docker-compose down
  ```

- [ ] **Production compose starts (GPU)**
  ```bash
  # Requires NVIDIA Docker runtime
  docker-compose up -d slower-whisper-gpu
  docker-compose ps

  # Expected: Service running with GPU access
  docker-compose down
  ```

- [ ] **API service compose starts**
  ```bash
  docker-compose -f docker-compose.api.yml up -d
  curl http://localhost:8000/health
  docker-compose -f docker-compose.api.yml down

  # Expected: API accessible and healthy
  ```

### Multi-stage Build Optimization

- [ ] **Build cache works correctly**
  ```bash
  # Build once
  docker build -t slower-whisper:test .

  # Build again (should use cache)
  docker build -t slower-whisper:test .

  # Expected: Second build much faster, cache hits on most layers
  ```

- [ ] **Base stage is minimal**
  ```bash
  docker build --target base -t slower-whisper:base-only .
  docker images slower-whisper:base-only

  # Expected: Minimal Python + ffmpeg only, < 500MB
  ```

---

## Kubernetes Deployment Verification

### Manifest Validation

- [ ] **All YAML manifests are valid**
  ```bash
  # Validate each manifest
  kubectl apply --dry-run=client -f k8s/namespace.yaml
  kubectl apply --dry-run=client -f k8s/configmap.yaml
  kubectl apply --dry-run=client -f k8s/pvc.yaml
  kubectl apply --dry-run=client -f k8s/deployment.yaml
  kubectl apply --dry-run=client -f k8s/job.yaml
  kubectl apply --dry-run=client -f k8s/service.yaml
  kubectl apply --dry-run=client -f k8s/cronjob.yaml
  kubectl apply --dry-run=client -f k8s/secret.yaml
  kubectl apply --dry-run=client -f k8s/gpu-resource-quota.yaml

  # Expected: All validate without errors
  ```

- [ ] **Kustomization builds correctly**
  ```bash
  kubectl kustomize k8s/

  # Expected: Outputs combined manifests without errors
  ```

### Cluster Application (Test Cluster)

- [ ] **Namespace creates successfully**
  ```bash
  kubectl apply -f k8s/namespace.yaml
  kubectl get namespace slower-whisper

  # Expected: Namespace created
  ```

- [ ] **ConfigMap applies correctly**
  ```bash
  kubectl apply -f k8s/configmap.yaml -n slower-whisper
  kubectl get configmap slower-whisper-config -n slower-whisper -o yaml

  # Expected: ConfigMap created with correct data
  ```

- [ ] **PVC creates and binds**
  ```bash
  kubectl apply -f k8s/pvc.yaml -n slower-whisper
  kubectl get pvc -n slower-whisper

  # Expected: PVC created, status Bound (or Pending if StorageClass needs provisioning)
  ```

- [ ] **Deployment creates pods**
  ```bash
  kubectl apply -f k8s/deployment.yaml -n slower-whisper
  kubectl get pods -n slower-whisper

  # Expected: Pods created, eventually Running
  kubectl wait --for=condition=ready pod -l app=slower-whisper -n slower-whisper --timeout=300s
  ```

- [ ] **Service exposes deployment**
  ```bash
  kubectl apply -f k8s/service.yaml -n slower-whisper
  kubectl get svc -n slower-whisper

  # Expected: Service created with ClusterIP
  ```

- [ ] **Job runs to completion**
  ```bash
  kubectl apply -f k8s/job.yaml -n slower-whisper
  kubectl wait --for=condition=complete job/slower-whisper-batch -n slower-whisper --timeout=600s
  kubectl logs job/slower-whisper-batch -n slower-whisper

  # Expected: Job completes successfully
  ```

- [ ] **CronJob is scheduled**
  ```bash
  kubectl apply -f k8s/cronjob.yaml -n slower-whisper
  kubectl get cronjob -n slower-whisper

  # Expected: CronJob created, schedule visible
  ```

### Resource Limits and Requests

- [ ] **Resource quotas applied**
  ```bash
  kubectl apply -f k8s/gpu-resource-quota.yaml -n slower-whisper
  kubectl get resourcequota -n slower-whisper

  # Expected: Resource quotas set correctly
  ```

- [ ] **Pod resources match requirements**
  ```bash
  kubectl describe pod -l app=slower-whisper -n slower-whisper | grep -A 5 "Requests"

  # Expected: CPU/memory requests and limits set appropriately
  # CPU: 2-4 cores, Memory: 8-16GB (base), 16-32GB (with enrichment)
  ```

### GPU Support (if applicable)

- [ ] **GPU nodes available**
  ```bash
  kubectl get nodes -o json | jq '.items[] | select(.status.capacity."nvidia.com/gpu" != null)'

  # Expected: Lists GPU nodes
  ```

- [ ] **GPU pod scheduling works**
  ```bash
  # Check GPU deployment
  kubectl get pods -l app=slower-whisper-gpu -n slower-whisper
  kubectl describe pod -l app=slower-whisper-gpu -n slower-whisper | grep "nvidia.com/gpu"

  # Expected: Pods scheduled on GPU nodes
  ```

### Cleanup

- [ ] **Test resources cleaned up**
  ```bash
  kubectl delete namespace slower-whisper

  # Expected: Namespace and all resources deleted
  ```

---

## CI/CD Pipeline Verification

### GitHub Actions Workflows Valid

- [ ] **CI workflow syntax valid**
  ```bash
  # Validate workflow syntax
  cat .github/workflows/ci.yml | python -c "import yaml, sys; yaml.safe_load(sys.stdin)"

  # Expected: No YAML syntax errors
  ```

- [ ] **Docker build workflow syntax valid**
  ```bash
  cat .github/workflows/docker-build.yml | python -c "import yaml, sys; yaml.safe_load(sys.stdin)"

  # Expected: No YAML syntax errors
  ```

- [ ] **Release workflow syntax valid**
  ```bash
  cat .github/workflows/release.yml | python -c "import yaml, sys; yaml.safe_load(sys.stdin)"

  # Expected: No YAML syntax errors
  ```

### CI Workflow Tests

- [ ] **Lint job configured correctly**
  - Uses `uv` for dependency management
  - Runs `ruff check` on all Python code
  - Fails on linting errors

- [ ] **Format job configured correctly**
  - Runs `ruff format --check`
  - Fails on formatting issues

- [ ] **Type check job configured correctly**
  - Runs `mypy` on transcription package
  - `continue-on-error: true` (allows warnings)

- [ ] **Test matrix includes Python 3.10, 3.11, 3.12**
  - Each version tested independently
  - System dependencies installed (ffmpeg, libsndfile1)
  - Tests run with pytest and coverage

- [ ] **Integration tests run after lint/format**
  - Dependencies: `[lint, format]`
  - Tests execute successfully

- [ ] **Heavy tests run conditionally**
  - Only on main branch or manual trigger
  - Caches HuggingFace models
  - `continue-on-error: true` (doesn't block release)

### Docker Build Workflow

- [ ] **Matrix builds all variants**
  - `cpu-base`: CPU runtime, base dependencies
  - `cpu`: CPU runtime, enrichment dependencies
  - `gpu-base`: GPU runtime, base dependencies
  - `gpu`: GPU runtime, enrichment dependencies

- [ ] **Container registry login configured**
  - Uses `GITHUB_TOKEN` for ghcr.io
  - Only pushes on tags or main branch

- [ ] **Tagging strategy correct**
  - `latest-{variant}` for main branch
  - `v{version}-{variant}` for tags
  - `{branch}-{sha}-{variant}` for branches

- [ ] **Build cache configured**
  - Uses GitHub Actions cache
  - Scope set per variant

### Manual Workflow Trigger Test

- [ ] **CI workflow runs manually**
  ```bash
  # Trigger via GitHub UI or gh CLI
  gh workflow run ci.yml
  gh run list --workflow=ci.yml

  # Expected: Workflow starts and runs successfully
  ```

- [ ] **Docker workflow runs manually**
  ```bash
  gh workflow run docker-build.yml
  gh run list --workflow=docker-build.yml

  # Expected: Builds complete successfully
  ```

### Required Status Checks

- [ ] **Branch protection configured**
  - Main branch requires:
    - CI workflow to pass
    - Lint and format checks
    - At least one code review

- [ ] **Status checks appear on PRs**
  - All CI jobs visible
  - Results reported correctly

---

## Documentation Verification

### README Completeness

- [ ] **Quick Start section works**
  ```bash
  # Follow README Quick Start exactly
  # 1. Install uv (if not installed)
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # 2. Install dependencies
  cd /path/to/slower-whisper
  uv sync

  # 3. Run transcription
  uv run slower-whisper

  # Expected: All commands succeed
  ```

- [ ] **Installation instructions accurate**
  - System dependencies (ffmpeg) correct
  - Python version requirements stated
  - Dependency groups documented

- [ ] **Configuration examples work**
  - CLI flag examples execute
  - Environment variable examples valid
  - Config file examples load

- [ ] **Usage examples functional**
  - CLI examples run without errors
  - Python API examples execute
  - Output format examples match actual output

- [ ] **Links not broken**
  ```bash
  # Check internal links
  grep -o '\[.*\](.*)' README.md | sed 's/.*(\(.*\))/\1/' | while read link; do
    if [[ $link != http* ]]; then
      [ -f "$link" ] || echo "Broken link: $link"
    fi
  done

  # Expected: No broken internal links
  ```

### API Documentation

- [ ] **`API_QUICK_REFERENCE.md` matches implementation**
  - Function signatures correct
  - Parameter types accurate
  - Return types documented
  - Example code runs

- [ ] **`ARCHITECTURE.md` up to date**
  - Schema version documented
  - Module descriptions accurate
  - Data flow diagrams match code

- [ ] **`AUDIO_ENRICHMENT.md` complete**
  - Prosody features documented
  - Emotion features explained
  - Dependencies listed
  - Examples work

### Contributing Guide

- [ ] **`CONTRIBUTING.md` clear**
  - Development setup instructions work
  - Code style guidelines match tooling
  - Testing instructions accurate
  - PR process documented

- [ ] **Coding standards documented**
  - Linting rules explained
  - Formatting conventions stated
  - Type checking requirements clear

### Example Scripts

- [ ] **Basic transcription example works**
  ```bash
  uv run python examples/basic_transcription.py

  # Expected: Runs without errors (may skip if no audio files)
  ```

- [ ] **Custom config example works**
  ```bash
  uv run python examples/custom_config.py

  # Expected: Loads config correctly
  ```

- [ ] **Enrichment workflow example works**
  ```bash
  uv sync --extra full
  uv run python examples/enrichment_workflow.py

  # Expected: Runs enrichment (may skip if no audio/transcripts)
  ```

- [ ] **Emotion integration example works**
  ```bash
  uv sync --extra emotion
  uv run python examples/emotion_integration.py

  # Expected: Demonstrates emotion extraction
  ```

- [ ] **Single file example works**
  ```bash
  uv run python examples/single_file.py

  # Expected: Processes single file
  ```

---

## Release Readiness

### CHANGELOG Updated

- [ ] **Version section added**
  - Version number matches `pyproject.toml`
  - Release date set
  - Section organized by category (Added, Changed, Fixed, etc.)

- [ ] **All changes documented**
  - New features listed
  - Bug fixes noted
  - Breaking changes highlighted
  - Deprecations announced

- [ ] **Migration guide provided (if needed)**
  - Breaking changes explained
  - Code examples for migration
  - Backward compatibility notes

### Version Tags

- [ ] **Git tag created**
  ```bash
  # Create annotated tag
  git tag -a v1.0.0 -m "Release version 1.0.0"

  # Verify tag
  git tag -l -n1 v1.0.0

  # Expected: Tag created with message
  ```

- [ ] **Tag follows semver**
  - Format: `v{MAJOR}.{MINOR}.{PATCH}`
  - Example: `v1.0.0`

### PyPI Publication (if applicable)

- [ ] **PyPI credentials configured**
  ```bash
  # Check if credentials exist
  cat ~/.pypirc

  # Expected: Contains API token for PyPI
  ```

- [ ] **Package builds successfully**
  ```bash
  uv run python -m build

  # Expected: Creates dist/ with .whl and .tar.gz
  ls dist/
  ```

- [ ] **Package metadata correct**
  ```bash
  # Check package metadata
  tar -tzf dist/slower-whisper-*.tar.gz | grep PKG-INFO
  tar -xzf dist/slower-whisper-*.tar.gz --to-stdout '*/PKG-INFO'

  # Expected: Metadata matches pyproject.toml
  ```

- [ ] **Test PyPI upload works**
  ```bash
  # Upload to test PyPI
  uv run twine upload --repository testpypi dist/*

  # Install from test PyPI
  pip install --index-url https://test.pypi.org/simple/ slower-whisper

  # Expected: Package installs successfully
  ```

- [ ] **Production PyPI upload ready**
  ```bash
  # Upload to production PyPI (DO NOT RUN UNTIL READY!)
  # uv run twine upload dist/*

  # Expected: Package available at https://pypi.org/project/slower-whisper/
  ```

### GitHub Release

- [ ] **Release draft created**
  - Title: `v1.0.0 - Production Release`
  - Tag: `v1.0.0`
  - Description from CHANGELOG
  - Assets attached (if any)

- [ ] **Release notes complete**
  - Summary of major changes
  - Installation instructions
  - Breaking changes highlighted
  - Known issues listed

### Container Registry

- [ ] **Images tagged correctly**
  ```bash
  # Check image tags
  docker images | grep slower-whisper

  # Expected tags:
  # - slower-whisper:latest-cpu
  # - slower-whisper:latest-gpu
  # - slower-whisper:v1.0.0-cpu
  # - slower-whisper:v1.0.0-gpu
  ```

- [ ] **Images pushed to registry**
  ```bash
  # Push to GitHub Container Registry
  docker tag slower-whisper:cpu ghcr.io/steven/slower-whisper:latest-cpu
  docker push ghcr.io/steven/slower-whisper:latest-cpu

  # Expected: Images available at ghcr.io
  ```

---

## Post-Deployment Verification

### Smoke Tests

- [ ] **Fresh install works (base)**
  ```bash
  # Clean environment
  python -m venv test_env
  source test_env/bin/activate

  # Install package
  pip install slower-whisper

  # Run help
  slower-whisper --help

  # Expected: Help displays, no import errors
  deactivate
  rm -rf test_env
  ```

- [ ] **Fresh install works (full)**
  ```bash
  # Clean environment
  python -m venv test_env_full
  source test_env_full/bin/activate

  # Install with enrichment
  pip install slower-whisper[full]

  # Test import
  python -c "from transcription import transcribe_file, enrich_transcript; print('OK')"

  # Expected: Imports succeed
  deactivate
  rm -rf test_env_full
  ```

- [ ] **Docker image smoke test**
  ```bash
  # Pull and run
  docker pull ghcr.io/steven/slower-whisper:latest-cpu
  docker run --rm ghcr.io/steven/slower-whisper:latest-cpu slower-whisper --version

  # Expected: Prints version number
  ```

### User Acceptance Testing

- [ ] **End-to-end transcription workflow**
  ```bash
  # 1. Prepare test audio
  mkdir -p test_project/raw_audio
  cp /path/to/sample.mp3 test_project/raw_audio/

  # 2. Run transcription
  cd test_project
  uv run slower-whisper transcribe --model base --language en

  # 3. Verify outputs
  ls whisper_json/*.json
  ls transcripts/*.txt
  ls transcripts/*.srt

  # Expected: All outputs generated correctly
  ```

- [ ] **End-to-end enrichment workflow**
  ```bash
  # 1. Enrich transcripts
  uv run slower-whisper enrich --enable-prosody --enable-emotion

  # 2. Verify audio_state populated
  cat whisper_json/*.json | jq '.segments[0].audio_state'

  # Expected: audio_state contains prosody and emotion data
  ```

- [ ] **API service smoke test**
  ```bash
  # Start service
  docker run -d -p 8000:8000 --name api-test slower-whisper:api

  # Test health endpoint
  curl http://localhost:8000/health

  # Test transcribe endpoint (with sample audio)
  curl -X POST -F "audio=@sample.mp3" "http://localhost:8000/transcribe?model=base"

  # Cleanup
  docker stop api-test && docker rm api-test

  # Expected: All endpoints respond correctly
  ```

### Performance Validation

- [ ] **Transcription speed acceptable**
  - Base model: ~5-10x realtime on CPU
  - Large model: ~1-2x realtime on GPU
  - No memory leaks during batch processing

- [ ] **Enrichment speed acceptable**
  - Prosody extraction: ~1-2s per 10s segment
  - Emotion extraction: ~0.5-1s per 10s segment (GPU)
  - Batch processing doesn't timeout

### Monitoring and Logging

- [ ] **Logs are informative**
  ```bash
  # Run with verbose logging
  uv run slower-whisper transcribe --verbose

  # Expected: Progress updates, timing stats, no errors
  ```

- [ ] **Error messages are helpful**
  ```bash
  # Test with missing dependencies
  python -c "from transcription import enrich_transcript; enrich_transcript(None, None, None)"

  # Expected: Clear error message about what's missing
  ```

### Security Checks

- [ ] **No secrets in code**
  ```bash
  # Check for common secret patterns
  grep -r "API_KEY\|SECRET\|PASSWORD" --include="*.py" transcription/

  # Expected: No hardcoded secrets
  ```

- [ ] **No vulnerabilities in dependencies**
  ```bash
  # Run security audit (if security extras installed)
  uv pip install pip-audit
  uv run pip-audit

  # Expected: No high/critical vulnerabilities
  ```

- [ ] **Docker images scanned**
  ```bash
  # Scan with Docker Scout or Trivy
  docker scout cves slower-whisper:cpu
  # or: trivy image slower-whisper:cpu

  # Expected: No critical vulnerabilities
  ```

---

## Final Checklist Summary

### Critical (Must Pass Before Release)

- [x] All tests passing (or acceptable failures documented)
- [x] Linting and formatting clean
- [x] Version numbers updated and consistent
- [x] CHANGELOG updated with release notes
- [x] README and API docs accurate
- [x] Docker images build successfully
- [x] CI/CD workflows passing
- [x] Examples work without errors
- [x] Fresh install smoke test passes

### Important (Should Pass Before Release)

- [ ] Type checking clean (warnings acceptable)
- [ ] Kubernetes manifests valid
- [ ] Heavy tests passing (emotion models work)
- [ ] Performance benchmarks acceptable
- [ ] Security scan passes (no critical issues)
- [ ] GitHub release drafted

### Nice to Have (Can Be Fixed Post-Release)

- [ ] All documentation polished
- [ ] All integration tests passing
- [ ] Kubernetes deployment tested on cluster
- [ ] PyPI package published
- [ ] Container images pushed to registry

---

## Deployment Commands Reference

### Quick Deployment Flow

```bash
# 1. Pre-deployment checks
uv run pytest -v
uv run ruff check transcription/ tests/ examples/ benchmarks/
uv run ruff format --check transcription/ tests/ examples/ benchmarks/
uv run mypy transcription/ tests/

# 2. Build Docker images
docker build -t slower-whisper:cpu .
docker build -f Dockerfile.gpu -t slower-whisper:gpu .

# 3. Tag and push (if registry configured)
docker tag slower-whisper:cpu ghcr.io/steven/slower-whisper:latest-cpu
docker push ghcr.io/steven/slower-whisper:latest-cpu

# 4. Create release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# 5. Trigger workflows
gh workflow run docker-build.yml
gh workflow run release.yml

# 6. Post-deployment smoke test
docker run --rm ghcr.io/steven/slower-whisper:latest-cpu slower-whisper --help
```

---

## Rollback Procedure

If issues are discovered post-deployment:

1. **Stop deployment**
   - Cancel any in-progress CI/CD workflows
   - Do not push additional tags

2. **Assess impact**
   - Check error reports and logs
   - Identify affected users/deployments

3. **Revert changes**
   ```bash
   # Revert to previous tag
   docker pull ghcr.io/steven/slower-whisper:v0.9.9-cpu

   # Or delete problematic tag
   git tag -d v1.0.0
   git push origin :refs/tags/v1.0.0
   ```

4. **Communicate issue**
   - Update GitHub release notes
   - Notify users via issue tracker
   - Document root cause

5. **Fix and redeploy**
   - Address root cause
   - Re-run full checklist
   - Deploy fixed version

---

## Notes

- This checklist is version 1.0.0 and will evolve with the project
- Mark items as complete with `[x]` during deployment
- Document any deviations or skipped items with rationale
- Some items require specific infrastructure (GPU nodes, test cluster) and may be skipped if unavailable
- Adjust timings and thresholds based on your hardware and requirements

---

**Deployment Sign-off:**

- Prepared by: ________________
- Date: ________________
- Approved by: ________________
- Release version: ________________

**Post-deployment Notes:**

(Add any observations, issues, or follow-up items here)
