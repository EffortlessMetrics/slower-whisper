# Logging Documentation

This document describes the logging system in slower-whisper, including configuration, log levels, structured logging, and best practices for capturing logs programmatically.

## Table of Contents

- [Overview](#overview)
- [Logger Names](#logger-names)
- [Log Levels](#log-levels)
- [Configuration](#configuration)
- [Structured Logging](#structured-logging)
- [Common Logging Patterns](#common-logging-patterns)
- [Batch Operation Logs](#batch-operation-logs)
- [Capturing Logs Programmatically](#capturing-logs-programmatically)
- [Best Practices](#best-practices)

---

## Overview

slower-whisper uses Python's standard `logging` module for all log output. The logging system is designed to:

- Provide visibility into transcription and enrichment progress
- Track errors and warnings with context
- Support structured logging with extra fields for machine parsing
- Enable programmatic log capture for monitoring and debugging

All modules follow a consistent pattern:

```python
import logging
logger = logging.getLogger(__name__)
```

This creates module-based loggers like `transcription.pipeline`, `transcription.asr_engine`, etc.

---

## Logger Names

Each module creates a logger using `__name__`, resulting in hierarchical logger names:

| Logger Name | Module | Purpose |
|-------------|--------|---------|
| `transcription.pipeline` | `pipeline.py` | Stage 1 transcription orchestration |
| `transcription.asr_engine` | `asr_engine.py` | Whisper model loading and transcription |
| `transcription.audio_io` | `audio_io.py` | Audio normalization with ffmpeg |
| `transcription.api` | `api.py` | Public API functions |
| `transcription.audio_enrichment` | `audio_enrichment.py` | Stage 2 audio enrichment orchestration |
| `transcription.prosody` | `prosody.py` | Prosodic feature extraction |
| `transcription.emotion` | `emotion.py` | Emotion recognition |
| `transcription.diarization` | `diarization.py` | Speaker diarization |
| `transcription.service` | `service.py` | FastAPI service endpoints |
| `transcription.writers` | `writers.py` | JSON/TXT/SRT output writers |

You can filter logs by module using these logger names (see [Capturing Logs Programmatically](#capturing-logs-programmatically)).

---

## Log Levels

slower-whisper uses four standard log levels:

### DEBUG

**When to use**: Detailed diagnostic information for troubleshooting.

**What it contains**:
- Audio segment extraction details (sample counts, sample rates)
- Feature extraction success confirmations
- Baseline statistics computations
- Internal state details

**Examples**:
```python
logger.debug(
    "Extracted segment [%.2fs - %.2fs]: %d samples at %d Hz",
    segment.start,
    segment.end,
    len(audio_data),
    sample_rate,
)

logger.debug("Prosody extraction succeeded for segment %s", segment.id)
logger.debug("Baseline stats: %s", speaker_baseline)
```

### INFO

**When to use**: Normal operation progress, file counters, summaries.

**What it contains**:
- Pipeline stage headers (`=== Step 2: Loading Whisper model ===`)
- File-by-file progress indicators (`[1/10] interview.wav`)
- Batch operation summaries (processed/skipped/failed counts)
- Real-time factor (RTF) statistics
- Model loading confirmations

**Examples**:
```python
logger.info("=== Step 2: Loading Whisper model %s on %s (%s) ===",
            model_name, device, compute_type)

logger.info("[%d/%d] %s", idx, total, wav.name)

logger.info(
    "Pipeline completed: %d processed, %d skipped, %d failed",
    result.processed,
    result.skipped,
    result.failed,
)

logger.info(
    "  [stats] audio=%.1f min, wall=%.1fs, RTF=%.2fx",
    duration / 60,
    elapsed,
    rtf,
)
```

### WARNING

**When to use**: Recoverable issues, degraded operation, missing dependencies.

**What it contains**:
- Missing optional dependencies (parselmouth, librosa)
- Diarization failures (with fallback to no speaker labels)
- Semantic annotator errors
- Validation errors
- File timestamp comparison failures
- Unexpected speaker counts

**Examples**:
```python
logger.warning("parselmouth not available. Pitch extraction will be limited.")

logger.warning(
    "Diarization failed for %s: %s. Proceeding without speakers/turns.",
    wav_path.name,
    exc,
    exc_info=True,
)

logger.warning("Could not read duration for %s: %s", path.name, exc)
```

### ERROR

**When to use**: Unrecoverable failures for individual files or operations.

**What it contains**:
- Audio normalization failures
- Transcription failures
- Enrichment feature extraction failures
- File I/O errors
- Configuration errors

**Examples**:
```python
logger.error(
    "Failed to normalize %s",
    src.name,
    exc_info=True,
    extra={"file": src.name},
)

logger.error(
    "Failed to transcribe %s: %s",
    wav.name,
    e,
    exc_info=True
)

logger.error("Failed to enrich segment %s: %s", segment.id, e, exc_info=True)
```

**Note**: Batch operations log individual file errors at ERROR level but continue processing. Use `exc_info=True` to include full traceback.

---

## Configuration

### CLI Configuration

The `slower-whisper` CLI configures logging automatically based on the `--progress` flag:

```bash
# Show progress messages (INFO level, file counters)
slower-whisper transcribe --progress

# Hide progress messages (WARNING level only)
slower-whisper transcribe
```

**Implementation** (in `cli.py`):
```python
def _setup_progress_logging(show_progress: bool) -> None:
    import logging
    level = logging.INFO if show_progress else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")
```

### Python API Configuration

When using the Python API directly, you control logging configuration:

#### Basic Configuration

```python
import logging
from transcription import transcribe_directory, TranscriptionConfig

# Show progress messages
logging.basicConfig(level=logging.INFO, format="%(message)s")

config = TranscriptionConfig(model="large-v3", language="en")
transcripts = transcribe_directory("/data/project", config)
```

#### Detailed Configuration

```python
import logging

# Configure root logger with timestamps and module names
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Or configure specific loggers
logging.getLogger("transcription.pipeline").setLevel(logging.DEBUG)
logging.getLogger("transcription.asr_engine").setLevel(logging.INFO)
```

#### Suppress All Logs

```python
import logging

# Suppress all logs except critical errors
logging.basicConfig(level=logging.CRITICAL)
```

#### Custom Handler

```python
import logging
from pathlib import Path

# Write logs to file with timestamps
log_path = Path("logs/transcription.log")
log_path.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),  # Also log to console
    ],
)
```

### Example Scripts Configuration

Example scripts in `examples/workflows/` use consistent logging setup:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
```

---

## Structured Logging

Many log messages include **structured extra fields** for machine-readable context. These fields are accessible via custom log handlers or formatters.

### Extra Fields

Common extra fields used across modules:

| Field | Type | Example | Used By |
|-------|------|---------|---------|
| `file` | str | `"interview.wav"` | audio_io, asr_engine, pipeline |
| `output` | str | `"interview.wav"` | audio_io |
| `device` | str | `"cuda"` | asr_engine, service |
| `compute_type` | str | `"float16"` | asr_engine, service |
| `index` | int | `1` | cli |
| `total` | int | `10` | cli |
| `request_id` | str | `"550e8400-..."` | service |
| `status_code` | int | `200` | service |
| `duration_ms` | float | `1234.56` | service |
| `error` | str | `"File not found"` | cli, service |

### Examples

**Audio normalization with context**:
```python
logger.info(
    "Normalizing audio file",
    extra={"file": src.name, "output": dst.name}
)
```

**Model loading with device info**:
```python
logger.info(
    "=== Step 2: Loading Whisper model %s on %s (%s) ===",
    cfg.model_name,
    cfg.device,
    cfg.compute_type,
    extra={"device": cfg.device, "compute_type": cfg.compute_type},
)
```

**Batch processing with counters**:
```python
logger.info(
    "Enriching single file",
    extra={"file": config.single_file}
)

logger.info(
    "Processing file",
    extra={"index": idx, "total": total, "file": json_path.name},
)
```

**HTTP request tracing** (service.py):
```python
logger.info(
    "Request started: %s %s [request_id=%s]",
    request.method,
    request.url.path,
    request_id,
    extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params),
    },
)
```

### Accessing Extra Fields

Custom log handlers can access extra fields via `record`:

```python
import logging

class StructuredHandler(logging.Handler):
    def emit(self, record):
        # Access extra fields
        file_name = getattr(record, "file", None)
        device = getattr(record, "device", None)

        # Build structured log entry
        log_entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "message": record.getMessage(),
            "file": file_name,
            "device": device,
        }
        print(json.dumps(log_entry))

# Attach to transcription loggers
handler = StructuredHandler()
logging.getLogger("transcription").addHandler(handler)
```

---

## Common Logging Patterns

### Pattern 1: File-by-File Progress

Used in batch operations (pipeline, enrichment):

```python
logger.info("=== Step 3: Transcribing normalized audio ===")
total = len(norm_files)

for idx, wav in enumerate(norm_files, start=1):
    logger.info("[%d/%d] %s", idx, total, wav.name)
    # Process file...
    logger.info("  → JSON: %s", json_path)
    logger.info("  → TXT:  %s", txt_path)
    logger.info("  → SRT:  %s", srt_path)
```

**Output**:
```
=== Step 3: Transcribing normalized audio ===
[1/10] interview.wav
  → JSON: whisper_json/interview.json
  → TXT:  transcripts/interview.txt
  → SRT:  transcripts/interview.srt
[2/10] lecture.wav
...
```

### Pattern 2: Real-Time Factor (RTF) Reporting

Used in transcription pipeline to report performance:

```python
start = time.time()
transcript = engine.transcribe_file(wav)
elapsed = time.time() - start

rtf = elapsed / duration if duration > 0 else 0.0
logger.info(
    "  [stats] audio=%.1f min, wall=%.1fs, RTF=%.2fx",
    duration / 60,
    elapsed,
    rtf,
)
```

**Output**:
```
  [stats] audio=3.2 min, wall=45.3s, RTF=0.24x
```

### Pattern 3: Batch Summary

Used after processing multiple files:

```python
logger.info("=== Summary ===")
logger.info(
    "  transcribed=%d, diarized_only=%d, skipped=%d, failed=%d, total=%d",
    processed,
    diarized_only,
    skipped,
    failed,
    total,
)
if total_audio > 0 and total_time > 0:
    overall_rtf = total_time / total_audio
    logger.info(
        "  audio=%.1f min, wall=%.1f min, RTF=%.2fx",
        total_audio / 60,
        total_time / 60,
        overall_rtf,
    )
logger.info("All done.")
```

**Output**:
```
=== Summary ===
  transcribed=8, diarized_only=0, skipped=2, failed=0, total=10
  audio=25.3 min, wall=6.1 min, RTF=0.24x
All done.
```

### Pattern 4: Graceful Degradation

Used when features fail but processing continues:

```python
try:
    prosody_result = extract_prosody(audio_data, sample_rate, ...)
    audio_state["prosody"] = prosody_result
    audio_state["extraction_status"]["prosody"] = "success"
    logger.debug("Prosody extraction succeeded for segment %s", segment.id)
except Exception as e:
    error_msg = f"Prosody extraction failed: {e}"
    logger.error(error_msg, exc_info=True)
    audio_state["extraction_status"]["prosody"] = "failed"
    audio_state["extraction_status"]["errors"].append(error_msg)
```

### Pattern 5: Model Loading with Fallback

Used when GPU model fails and CPU fallback succeeds:

```python
logger.warning(
    "Whisper model load failed on %s: %s",
    device,
    load_err,
    extra={"device": device},
    exc_info=True,
)
logger.warning(
    "Retrying on %s with compute_type=%s",
    next_device_label,
    next_compute_type,
    extra={"device": next_device, "compute_type": next_compute_type},
)
```

---

## Batch Operation Logs

### Transcription Pipeline

**Key log messages**:

```
=== Step 2: Loading Whisper model large-v3 on cuda (float16) ===
=== Step 3: Transcribing normalized audio ===
[1/10] interview.wav
  [stats] audio=3.2 min, wall=45.3s, RTF=0.24x
  → JSON: whisper_json/interview.json
  → TXT:  transcripts/interview.txt
  → SRT:  transcripts/interview.srt
[2/10] lecture.wav
...
=== Summary ===
  transcribed=8, diarized_only=0, skipped=2, failed=0, total=10
  audio=25.3 min, wall=6.1 min, RTF=0.24x
All done.
```

### Audio Enrichment

**Key log messages**:

```
Starting audio enrichment for 10 transcripts
[1/10] interview.json
Computing speaker baseline statistics...
  Sampled 20 segments for baseline
  Baseline stats: {'pitch': {'median': 180.5}, ...}
  Enriched 45 segments
[2/10] lecture.json
...
=== Audio Enrichment Summary ===
  Enriched: 8 files
  Skipped: 2 files
  Failed: 0 files
All done.
```

### Diarization Logs

**Success case**:
```
[diarize-existing] interview.wav (reusing existing transcript)
  → [diarization-only] whisper_json/interview.json
  → [diarization-only] transcripts/interview.txt
  → [diarization-only] transcripts/interview.srt
```

**Warning case**:
```
Diarization produced no speaker turns for interview.wav
Diarization found 15 speakers for meeting.wav; this may indicate noisy audio or misconfiguration.
```

**Failure case**:
```
Diarization failed for interview.wav: HuggingFace token required. Proceeding without speakers/turns.
```

---

## Capturing Logs Programmatically

### Use Case 1: Log to File for Analysis

```python
import logging
from pathlib import Path
from transcription import transcribe_directory, TranscriptionConfig

# Create log directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure file logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "transcription.log"),
        logging.StreamHandler(),
    ],
)

# Run transcription (logs go to file and console)
config = TranscriptionConfig(model="large-v3", language="en")
transcripts = transcribe_directory("/data/project", config)
```

### Use Case 2: Capture Logs in Memory

```python
import logging
from io import StringIO
from transcription import enrich_directory, EnrichmentConfig

# Create in-memory log stream
log_stream = StringIO()
handler = logging.StreamHandler(log_stream)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

# Attach to transcription loggers
logging.getLogger("transcription").addHandler(handler)

# Run enrichment
config = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
enriched = enrich_directory("/data/project", config)

# Extract logs
log_contents = log_stream.getvalue()
print("Captured logs:")
print(log_contents)
```

### Use Case 3: Filter Logs by Module

```python
import logging
from transcription import transcribe_file, TranscriptionConfig

# Suppress noisy modules, show pipeline and ASR only
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transcription.pipeline").setLevel(logging.INFO)
logging.getLogger("transcription.asr_engine").setLevel(logging.INFO)

config = TranscriptionConfig(model="large-v3")
transcript = transcribe_file("interview.mp3", "/data/project", config)
```

### Use Case 4: JSON Structured Logging

```python
import json
import logging
from transcription import transcribe_directory, TranscriptionConfig

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extra fields if present
        for key in ["file", "device", "request_id", "index", "total"]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)

config = TranscriptionConfig(model="large-v3")
transcripts = transcribe_directory("/data/project", config)
```

**Output**:
```json
{"timestamp": "2025-12-02 10:15:23", "level": "INFO", "logger": "transcription.audio_io", "message": "Normalizing audio file", "file": "interview.wav", "output": "interview.wav"}
{"timestamp": "2025-12-02 10:15:30", "level": "INFO", "logger": "transcription.pipeline", "message": "[1/10] interview.wav", "index": 1, "total": 10, "file": "interview.wav"}
```

### Use Case 5: Monitor Enrichment Errors

```python
import logging
from transcription import enrich_directory, EnrichmentConfig

class ErrorCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.errors = []

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.errors.append({
                "logger": record.name,
                "message": record.getMessage(),
                "file": getattr(record, "file", None),
            })

# Attach error collector
error_handler = ErrorCollector()
error_handler.setLevel(logging.ERROR)
logging.getLogger("transcription").addHandler(error_handler)

# Run enrichment
config = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
enriched = enrich_directory("/data/project", config)

# Check errors
if error_handler.errors:
    print(f"Encountered {len(error_handler.errors)} errors:")
    for err in error_handler.errors:
        print(f"  {err['file']}: {err['message']}")
```

### Use Case 6: HTTP Service Request Tracing

The FastAPI service (`service.py`) uses structured logging for request tracing:

```python
# service.py generates logs like:
# INFO - Request started: POST /transcribe [request_id=550e8400-...]
# INFO - Starting transcription: model=large-v3, language=en, device=cuda
# INFO - Transcription completed successfully: 42 segments
# INFO - Request completed: POST /transcribe -> 200 (12345.67 ms) [request_id=550e8400-...]

# Custom handler to extract request metrics:
import logging

class RequestMetricsHandler(logging.Handler):
    def emit(self, record):
        request_id = getattr(record, "request_id", None)
        duration_ms = getattr(record, "duration_ms", None)

        if request_id and duration_ms:
            # Log to metrics system
            print(f"Request {request_id}: {duration_ms:.2f}ms")

handler = RequestMetricsHandler()
logging.getLogger("transcription.service").addHandler(handler)
```

---

## Best Practices

### 1. Use Module-Based Loggers

**Do**:
```python
import logging
logger = logging.getLogger(__name__)  # Creates transcription.module_name
```

**Don't**:
```python
logger = logging.getLogger("my_logger")  # Custom name breaks hierarchy
```

### 2. Include Context in Log Messages

**Do**:
```python
logger.error(
    "Failed to transcribe %s: %s",
    wav.name,
    e,
    exc_info=True,
    extra={"file": wav.name}
)
```

**Don't**:
```python
logger.error("Transcription failed")  # Missing context
```

### 3. Use Appropriate Log Levels

- **DEBUG**: Internal state, detailed diagnostics
- **INFO**: Progress indicators, normal operation
- **WARNING**: Recoverable issues, degraded operation
- **ERROR**: Unrecoverable failures (but batch continues)
- **CRITICAL**: Reserved (not currently used)

### 4. Use `exc_info=True` for Errors

**Do**:
```python
try:
    result = risky_operation()
except Exception as e:
    logger.error("Operation failed: %s", e, exc_info=True)
```

**Don't**:
```python
logger.error(f"Operation failed: {e}")  # Missing traceback
```

### 5. Use Structured Extra Fields

**Do**:
```python
logger.info(
    "Processing file %s",
    filename,
    extra={"file": filename, "index": idx, "total": total}
)
```

**Don't**:
```python
logger.info(f"Processing file {filename} ({idx}/{total})")  # Hard to parse
```

### 6. Configure Logging Before Importing

**Do**:
```python
import logging
logging.basicConfig(level=logging.INFO)  # Configure first

from transcription import transcribe_directory  # Then import
```

**Don't**:
```python
from transcription import transcribe_directory
logging.basicConfig(level=logging.INFO)  # Too late, loggers already created
```

### 7. Suppress Logs in Tests

```python
import logging
import pytest

@pytest.fixture(autouse=True)
def suppress_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)
```

### 8. Log Performance Metrics

For long-running operations, log timing information:

```python
import time

start = time.time()
result = expensive_operation()
elapsed = time.time() - start

logger.info(
    "Operation completed in %.2fs (RTF=%.2fx)",
    elapsed,
    elapsed / duration,
)
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and module structure
- [CLI Documentation](../README.md#cli-usage) - Command-line interface reference
- [Python API Documentation](../README.md#python-api) - Programmatic usage examples
- Python `logging` module: https://docs.python.org/3/library/logging.html
