# Tier 2 Implementation Plan: Comprehensive Improvements

**Generated**: 2025-12-02
**Based on**: 8 parallel exploration agents analyzing codebase

---

## Executive Summary

8 exploration agents analyzed the codebase post-Tier 2 foundation work. Key findings:

- **Error Handling**: 11 modules use print() instead of logger; 67 bare exception handlers missing exc_info=True
- **Logging**: 31/41 modules lack module-level loggers; inconsistent log levels
- **API Layer**: Functions return list[Transcript] instead of BatchResult models; no structured error reporting
- **Test Coverage**: Zero tests for BatchResult models; minimal caplog usage
- **Configuration**: Legacy configs still in use; missing validation for critical fields
- **Service Layer**: Missing comprehensive health checks; no request validation
- **Documentation**: Batch results, logging, and CLI features undocumented
- **CLI**: --progress flag declared but unimplemented; structured results hidden

---

## Priority 1: Critical User-Facing Issues (High Impact, Quick Wins)

### 1.1 Implement CLI --progress Flag
- **Impact**: Users can track long-running operations
- **Effort**: Low (1-2 hours)
- **Files**: `transcription/cli.py`
- **Action**: Add file counter display `[3/10] processing audio.wav...`

### 1.2 Fix CLI Exit Codes
- **Impact**: CI/CD can detect partial failures
- **Effort**: Low (30 min)
- **Files**: `transcription/cli.py`
- **Action**: Return 1 if any files failed

### 1.3 Expose Structured Results in CLI
- **Impact**: Users see failures/skips, not just success count
- **Effort**: Medium (2-3 hours)
- **Files**: `transcription/cli.py`
- **Action**: Display PipelineBatchResult summary with failures

### 1.4 Add Model Name Validation
- **Impact**: Prevent runtime failures from typos
- **Effort**: Low (1 hour)
- **Files**: `transcription/config.py`
- **Action**: Validate against ALLOWED_WHISPER_MODELS

---

## Priority 2: API Layer Improvements (Breaking Changes for v2.0)

### 2.1 Change API Return Types
- **Impact**: Programmatic callers get structured results
- **Effort**: High (4-6 hours)
- **Files**: `transcription/api.py`, `transcription/models.py`
- **Action**:
  - `transcribe_directory()` → `BatchProcessingResult`
  - `enrich_directory()` → `EnrichmentBatchResult`
  - Add backward compatibility flag

### 2.2 Add Progress Callbacks to API
- **Impact**: Long-running operations can report progress
- **Effort**: Medium (2-3 hours)
- **Files**: `transcription/api.py`
- **Action**: Add `progress_callback: Callable[[int, int, str], None]` parameter

### 2.3 Standardize Error Handling in API
- **Impact**: Consistent error reporting across functions
- **Effort**: Medium (3-4 hours)
- **Files**: `transcription/api.py`
- **Action**: Use structured error types; add exc_info=True; collect errors

---

## Priority 3: Logging Infrastructure (High Impact, Medium Effort)

### 3.1 Add Loggers to 31 Modules
- **Impact**: Consistent logging across codebase
- **Effort**: Medium (3-4 hours)
- **Files**: All modules listed in exploration report
- **Action**: Add `logger = logging.getLogger(__name__)` to each module

### 3.2 Convert print() to logger (11 Modules)
- **Impact**: Structured logging for CLI operations
- **Effort**: Medium (4-5 hours)
- **Files**: `cli.py`, `dogfood.py`, etc.
- **Action**: Replace ~50 print statements with appropriate logger calls

### 3.3 Add exc_info=True to Error Logs
- **Impact**: Full stack traces in error logs
- **Effort**: Low (1-2 hours)
- **Files**: `audio_enrichment.py`, `prosody.py`, `api.py`, `asr_engine.py`
- **Action**: Add exc_info=True to ~20 logger.error() calls

### 3.4 Standardize Log Levels
- **Impact**: Consistent log level semantics
- **Effort**: Low (1-2 hours)
- **Files**: `pipeline.py`, `api.py`, `audio_io.py`
- **Action**: Fix inconsistent warning/info/error usage

---

## Priority 4: Test Coverage (Critical Gaps)

### 4.1 Create BatchResult Models Tests
- **Impact**: Ensure model reliability
- **Effort**: Medium (4-5 hours)
- **Files**: New `tests/test_batch_result_models.py`
- **Action**: Add 15-20 tests covering all model methods

### 4.2 Add Logging Tests (caplog)
- **Impact**: Verify logging behavior
- **Effort**: Medium (3-4 hours)
- **Files**: `test_api_integration.py`, `test_pipeline.py`, `test_audio_enrichment.py`
- **Action**: Add ~20 caplog tests for key logging scenarios

### 4.3 Add Batch Error Scenario Tests
- **Impact**: Ensure error handling works
- **Effort**: Medium (3-4 hours)
- **Files**: `test_api_integration.py`
- **Action**: Add 6+ tests for partial failures, all failures, mixed results

---

## Priority 5: Service Layer Production Readiness

### 5.1 Comprehensive Health Checks
- **Impact**: Production monitoring and k8s readiness probes
- **Effort**: Medium (3-4 hours)
- **Files**: `transcription/service.py`
- **Action**: Separate /health/live and /health/ready; check dependencies

### 5.2 Add Request Validation
- **Impact**: Prevent OOM and invalid uploads
- **Effort**: Low (1-2 hours)
- **Files**: `transcription/service.py`
- **Action**: Validate file sizes, audio format, duration limits

### 5.3 Add Metrics Endpoint
- **Impact**: Prometheus monitoring
- **Effort**: High (4-6 hours)
- **Files**: `transcription/service.py`
- **Action**: Add /metrics endpoint with request counts, durations, errors

### 5.4 Add Request Logging
- **Impact**: Audit trail for uploads
- **Effort**: Low (1 hour)
- **Files**: `transcription/service.py`
- **Action**: Log file metadata (name, size, content-type) for uploads

---

## Priority 6: Configuration System Cleanup

### 6.1 Deprecate Legacy Configs
- **Impact**: Simpler config maintenance
- **Effort**: Medium (3-4 hours)
- **Files**: `transcription/config.py`, `transcription/asr_engine.py`
- **Action**: Add deprecation warnings; create conversion utilities

### 6.2 Add Config Validation
- **Impact**: Catch errors at config time
- **Effort**: Medium (2-3 hours)
- **Files**: `transcription/config.py`
- **Action**: Validate model names, device availability, language codes

### 6.3 Document Environment Variables
- **Impact**: Easier configuration
- **Effort**: Low (1-2 hours)
- **Files**: New `docs/ENVIRONMENT_VARIABLES.md`
- **Action**: List all env vars with descriptions

---

## Priority 7: Documentation (User-Facing)

### 7.1 Batch Results Documentation
- **Impact**: Users can use new features
- **Effort**: Medium (2-3 hours)
- **Files**: `docs/API_QUICK_REFERENCE.md`, model docstrings
- **Action**: Document BatchResult models with examples

### 7.2 Logging Guide
- **Impact**: Users can debug issues
- **Effort**: Medium (2-3 hours)
- **Files**: New `docs/LOGGING.md`
- **Action**: Document log levels, configuration, key messages

### 7.3 CLI Progress Flag Documentation
- **Impact**: Users know how to use --progress
- **Effort**: Low (30 min)
- **Files**: `docs/CLI_REFERENCE.md`
- **Action**: Document --progress flag behavior

### 7.4 Update CHANGELOG for v1.6.0
- **Impact**: Release notes
- **Effort**: Low (1 hour)
- **Files**: `CHANGELOG.md`
- **Action**: Document all Tier 2 features

---

## Implementation Strategy

### Phase 1: Quick Wins (1-2 days)
1. Implement CLI --progress flag
2. Fix CLI exit codes
3. Add model name validation
4. Add exc_info=True to error logs
5. Create batch result tests

### Phase 2: API Changes (2-3 days)
1. Change API return types (breaking change)
2. Update CLI to use structured results
3. Add progress callbacks
4. Add batch error scenario tests

### Phase 3: Infrastructure (3-4 days)
1. Add loggers to all modules
2. Convert print() to logger
3. Standardize log levels
4. Add comprehensive health checks
5. Add request validation

### Phase 4: Documentation (1-2 days)
1. Document batch results
2. Create logging guide
3. Update CHANGELOG
4. Add environment variables reference

---

## Breaking Changes (v2.0.0)

1. **API Return Types**: `transcribe_directory()` and `enrich_directory()` return BatchResult instead of list[Transcript]
2. **Legacy Config Removal**: Remove AsrConfig, AppConfig, LegacyEnrichmentConfig
3. **CLI Print to Logger**: Some stderr output moves to logging framework

---

## Metrics for Success

- **Test Coverage**: BatchResult models at 100%, logging tests added
- **Logging**: All 41 modules have loggers, zero print() in production code
- **API**: All batch operations return structured results
- **Service**: Health checks pass, request validation active
- **Documentation**: All new features documented with examples
- **User Experience**: CLI shows progress and structured results

---

## Estimated Total Effort

- **Priority 1 (Critical)**: 5-7 hours
- **Priority 2 (API)**: 9-13 hours
- **Priority 3 (Logging)**: 9-12 hours
- **Priority 4 (Tests)**: 10-13 hours
- **Priority 5 (Service)**: 9-13 hours
- **Priority 6 (Config)**: 6-9 hours
- **Priority 7 (Docs)**: 6-9 hours

**Total**: 54-76 hours (7-10 days with parallel work)

---

## Next Steps

1. Review and approve this plan
2. Launch parallel implementation agents for independent tasks
3. Create feature branches for breaking changes
4. Implement in priority order
5. Run verification suite after each phase
6. Update documentation alongside code changes
