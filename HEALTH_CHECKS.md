# Health Check Endpoints

This document describes the production-ready health check endpoints for the slower-whisper API service.

## Overview

The service now provides three health check endpoints:

1. **`/health`** - Legacy endpoint (deprecated)
2. **`/health/live`** - Liveness probe
3. **`/health/ready`** - Readiness probe

## Endpoint Details

### `/health/live` - Liveness Probe

**Purpose:** Verify the service process is alive and responsive.

**Use Case:** Kubernetes liveness probe, basic uptime monitoring

**Checks:** Minimal - just confirms the process is running

**Response:** Always returns 200 if the service is running

**Example Response:**
```json
{
  "status": "alive",
  "service": "slower-whisper-api",
  "version": "1.5.0",
  "schema_version": "2"
}
```

**Usage:**
```bash
curl http://localhost:8000/health/live
```

**Kubernetes Configuration:**
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

---

### `/health/ready` - Readiness Probe

**Purpose:** Verify the service is ready to handle requests.

**Use Case:** Kubernetes readiness probe, load balancer health checks

**Checks:**
- ✅ **ffmpeg availability** (critical)
- ✅ **faster-whisper import** (critical)
- ⚠️ **CUDA availability** (warning only, not critical)
- ⚠️ **Disk space** (warning only, not critical)

**Response Codes:**
- `200` - Service is ready (all critical checks pass)
- `503` - Service is degraded or not ready (critical checks fail)

**Example Response (Healthy):**
```json
{
  "status": "ready",
  "healthy": true,
  "service": "slower-whisper-api",
  "version": "1.5.0",
  "schema_version": "2",
  "checks": {
    "ffmpeg": {
      "status": "ok",
      "path": "/usr/bin/ffmpeg"
    },
    "faster_whisper": {
      "status": "ok"
    },
    "cuda": {
      "status": "ok",
      "message": "CUDA not required (device=cpu)"
    },
    "disk_space": {
      "status": "ok",
      "cache_root": "/home/user/.cache/slower-whisper",
      "free_gb": 2998.68,
      "total_gb": 7256.25,
      "used_gb": 4183.83,
      "percent_used": 57.7
    }
  }
}
```

**Example Response (Degraded - ffmpeg missing):**
```json
{
  "status": "degraded",
  "healthy": false,
  "service": "slower-whisper-api",
  "version": "1.5.0",
  "schema_version": "2",
  "checks": {
    "ffmpeg": {
      "status": "error",
      "message": "ffmpeg not found on PATH"
    },
    "faster_whisper": {
      "status": "ok"
    },
    "cuda": {
      "status": "ok",
      "message": "CUDA not required (device=cpu)"
    },
    "disk_space": {
      "status": "ok",
      "cache_root": "/home/user/.cache/slower-whisper",
      "free_gb": 2998.68,
      "total_gb": 7256.25,
      "used_gb": 4183.83,
      "percent_used": 57.7
    }
  }
}
```

**Usage:**
```bash
curl http://localhost:8000/health/ready
```

**Kubernetes Configuration:**
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
  successThreshold: 1
```

---

### `/health` - Legacy Endpoint (Deprecated)

**Status:** Deprecated - use `/health/live` or `/health/ready` instead

**Purpose:** Simple health check for backward compatibility

**Response:** Always returns 200 with basic service info

**Example Response:**
```json
{
  "status": "healthy",
  "service": "slower-whisper-api",
  "version": "1.5.0",
  "schema_version": "2"
}
```

---

## Check Details

### ffmpeg Check (Critical)

**What it checks:** Whether `ffmpeg` is available on the system PATH

**Why critical:** Required for audio normalization (converting input audio to 16kHz mono WAV)

**Status values:**
- `ok` - ffmpeg found (includes path to binary)
- `error` - ffmpeg not found

**Recovery:** Install ffmpeg via package manager or container image

---

### faster-whisper Check (Critical)

**What it checks:** Whether `faster-whisper` Python package can be imported

**Why critical:** Required for transcription

**Status values:**
- `ok` - faster-whisper available
- `error` - faster-whisper import failed

**Recovery:** Install faster-whisper: `pip install faster-whisper`

---

### CUDA Check (Warning)

**What it checks:** Whether CUDA is available when GPU acceleration is expected

**Why warning only:** The service falls back to CPU if CUDA is unavailable

**Status values:**
- `ok` - CUDA available or not required (includes device count and name)
- `warning` - CUDA requested but unavailable
- `error` - CUDA check failed unexpectedly

**Configuration:** Set device via environment variable (e.g., `SLOWER_WHISPER_DEVICE=cpu`)

**Note:** Currently defaults to `cpu` to avoid false negatives. In production, set this via environment variable.

---

### Disk Space Check (Warning)

**What it checks:** Available disk space in cache directories

**Why warning only:** Service can run with low disk space, but model downloads may fail

**Thresholds:**
- `ok` - More than 5GB free
- `warning` - Less than 5GB free (model downloads may fail)

**Metrics provided:**
- `cache_root` - Path to cache directory
- `free_gb` - Gigabytes of free space
- `total_gb` - Total disk capacity
- `used_gb` - Gigabytes used
- `percent_used` - Percentage of disk used

---

## Production Best Practices

### Load Balancer Configuration

Configure load balancers to use `/health/ready` for health checks:

**AWS ALB/NLB:**
```
Health Check Path: /health/ready
Healthy Threshold: 2
Unhealthy Threshold: 3
Timeout: 5s
Interval: 10s
Success Codes: 200
```

**GCP Load Balancer:**
```
Health Check:
  Type: HTTP
  Request Path: /health/ready
  Check Interval: 10s
  Timeout: 5s
  Healthy Threshold: 2
  Unhealthy Threshold: 3
```

### Monitoring and Alerts

**Alert on:**
- `/health/ready` returning 503 for more than 2 consecutive checks
- `checks.ffmpeg.status = "error"`
- `checks.faster_whisper.status = "error"`

**Monitor but don't alert on:**
- `checks.cuda.status = "warning"` (CPU fallback works)
- `checks.disk_space.status = "warning"` (service still functional)

### Docker/Kubernetes

**Dockerfile:**
```dockerfile
# Install ffmpeg in the container
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Or for Alpine
RUN apk add --no-cache ffmpeg
```

**Kubernetes Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: slower-whisper-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: slower-whisper-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: SLOWER_WHISPER_DEVICE
          value: "cpu"  # or "cuda" if GPU nodes
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
          successThreshold: 1
```

---

## Testing

Run the test suite:
```bash
uv run pytest tests/test_service_health.py -v
```

Manual testing:
```bash
# Start the service
uvicorn transcription.service:app --host 0.0.0.0 --port 8000

# Test liveness
curl http://localhost:8000/health/live

# Test readiness
curl http://localhost:8000/health/ready

# Check specific status
curl -s http://localhost:8000/health/ready | jq '.checks.ffmpeg'
```

---

## Implementation Notes

**Location:** `/home/steven/code/Python/slower-whisper/transcription/service.py`

**Helper Functions:**
- `_check_ffmpeg()` - Check ffmpeg availability
- `_check_faster_whisper()` - Check faster-whisper import
- `_check_cuda(device)` - Check CUDA availability
- `_check_disk_space()` - Check disk space in cache dirs

**Tests:** `/home/steven/code/Python/slower-whisper/tests/test_service_health.py`

**Type Safety:** All endpoints and helper functions are fully typed and pass mypy strict checks.

**Request Tracing:** All endpoints include `X-Request-ID` header in responses for distributed tracing.
