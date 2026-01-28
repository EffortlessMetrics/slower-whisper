# Test Coverage Gaps Analysis

This document identifies modules in the `transcription/` package that lack dedicated test coverage and provides recommendations for improving test coverage.

## Summary

| Category | Count |
|----------|-------|
| Total modules in `transcription/` | 42 |
| Test files in `tests/` | 38 |
| Modules with dedicated tests | ~26 |
| Modules with coverage gaps | ~14 |

## Coverage Analysis

### Modules WITH Dedicated Test Coverage

| Module | Test File(s) | Coverage Level |
|--------|-------------|----------------|
| `api.py` | `test_api_integration.py` | Good - extensive integration tests |
| `asr_engine.py` | `test_asr_engine.py` | Excellent - resilience and fallback tests |
| `audio_enrichment.py` | `test_audio_enrichment.py` | Good |
| `audio_io.py` | `test_audio_io.py`, `test_audio_io_security.py` | Good - including security tests |
| `audio_rendering.py` | `test_audio_rendering.py` | Good |
| `cache.py` | `test_cache.py` | Good |
| `chunking.py` | `test_chunking.py` | Good |
| `cli.py` | `test_cli_integration.py` | Good - parser and command tests |
| `config.py` | `test_config.py`, `test_config_sources.py`, `test_config_merge_bug.py` | Excellent |
| `diarization.py` | `test_diarization_skeleton.py`, `test_diarization_mapping.py` | Moderate |
| `emotion.py` | `test_emotion.py` | Excellent - 56 tests with mocked dependencies |
| `exporters.py` | `test_exporters.py` | Good |
| `llm_utils.py` | `test_llm_utils.py` | Excellent - typed |
| `meta_utils.py` | `test_meta_utils.py` | Good |
| `prosody.py` | `test_prosody.py` | Good |
| `samples.py` | `test_samples.py` | Good |
| `semantic.py` | `test_semantic.py` | Good |
| `service.py` | `test_api_service.py`, `test_service_health.py` | Good - endpoint and health tests |
| `streaming.py` | `test_streaming.py` | Good |
| `streaming_semantic.py` | `test_streaming_semantics.py` | Good |
| `turn_helpers.py` | `test_turn_helpers.py` | Excellent - typed |
| `turns.py` | `test_turns.py` | Good |
| `speaker_id.py` | `test_speaker_id.py` | Excellent - 72 tests covering all input types |
| `turns_enrich.py` | `test_turns_enrich.py` | Good |
| `writers.py` | `test_writers.py` | Excellent - typed |

### Modules WITH COVERAGE GAPS

#### High Priority (Core Pipeline Components)

| Module | Purpose | Current Coverage | Recommended Tests |
|--------|---------|-----------------|-------------------|
| `pipeline.py` | Core transcription pipeline orchestration | Indirect via `test_api_integration.py` | **Dedicated unit tests needed** |
| `models.py` | Core data models (Segment, Transcript, Word) | Minimal direct testing | **Dedicated unit tests needed** |
| `exceptions.py` | Custom exception hierarchy | None | **Unit tests for exception behavior** |

#### Medium Priority (Supporting Components)

| Module | Purpose | Current Coverage | Recommended Tests |
|--------|---------|-----------------|-------------------|
| `audio_utils.py` | Memory-efficient WAV segment extraction | None | Unit tests for segment extraction |
| `speaker_stats.py` | Per-speaker analytics computation | Indirect via `test_speaker_analytics_pipeline.py` | Dedicated unit tests |
| `models_speakers.py` | Speaker-related data models | None | Unit tests for model serialization |
| `models_turns.py` | Turn-related data models | None | Unit tests for model serialization |
| `types_audio.py` | Audio-related type definitions | None | Type validation tests |

#### Low Priority (Utilities/Internal)

| Module | Purpose | Current Coverage | Recommended Tests |
|--------|---------|-----------------|-------------------|
| `__init__.py` | Package exports | Indirect | N/A - export verification |
| `_import_guards.py` | Import safety guards | None | Import failure tests |
| `benchmarks.py` | Benchmark utilities | None | Benchmark validation tests |
| `dogfood.py` | Internal testing utilities | None | N/A - test infrastructure |
| `dogfood_utils.py` | Internal testing utilities | None | N/A - test infrastructure |
| `streaming_enrich.py` | Streaming enrichment | Partial via `test_streaming.py` | Extended streaming tests |
| `validation.py` | JSON schema validation | Indirect via CLI tests | Unit tests for validators |

## Detailed Gap Analysis

### 1. `pipeline.py` - HIGH PRIORITY

**Purpose**: Core transcription pipeline orchestration with batch processing, progress tracking, and error recovery.

**Current State**: Tested indirectly through `test_api_integration.py` which mocks `run_pipeline()`.

**Missing Coverage**:
- Direct unit tests for `PipelineFileResult` and `PipelineBatchResult` dataclasses
- Tests for `_get_duration_seconds()` edge cases
- Tests for `_build_meta()` metadata construction
- Tests for `run_pipeline()` logic without mocking
- Error handling and recovery scenarios
- Diarization integration paths
- Skip-existing logic verification

**Recommended Test Scenarios**:
```
- test_pipeline_file_result_dataclass
- test_pipeline_batch_result_rtf_calculation
- test_run_pipeline_empty_directory
- test_run_pipeline_skip_existing_json
- test_run_pipeline_diarization_upgrade_existing
- test_run_pipeline_normalization_failure_handling
- test_run_pipeline_transcription_error_continues
- test_build_meta_preserves_asr_metadata
- test_get_duration_seconds_invalid_wav
```

### 2. `models.py` - HIGH PRIORITY

**Purpose**: Core data models including `Segment`, `Transcript`, `Word`, `DiarizationMeta`, etc.

**Current State**: Used extensively but lacks dedicated tests for model behavior.

**Missing Coverage**:
- Dataclass instantiation and defaults
- `to_dict()` method implementations
- Schema version constants
- Word dataclass (v1.8+ feature)
- Optional field handling (speaker, tone, audio_state)
- Serialization round-trips

**Recommended Test Scenarios**:
```
- test_segment_creation_with_defaults
- test_segment_with_audio_state
- test_segment_with_speaker_info
- test_segment_with_words
- test_transcript_creation
- test_transcript_with_diarization_fields
- test_word_dataclass
- test_diarization_meta_to_dict
- test_turn_meta_to_dict
- test_schema_version_constant
```

### 3. `exceptions.py` - HIGH PRIORITY

**Purpose**: Custom exception hierarchy for the package.

**Current State**: No dedicated tests.

**Missing Coverage**:
- Exception inheritance hierarchy
- Exception message formatting
- Exception use in error handling paths

**Recommended Test Scenarios**:
```
- test_slower_whisper_error_base_class
- test_transcription_error_inheritance
- test_enrichment_error_inheritance
- test_configuration_error_inheritance
- test_exception_message_formatting
```

### 4. `audio_utils.py` - MEDIUM PRIORITY

**Purpose**: Memory-efficient WAV segment extraction for audio processing.

**Current State**: No dedicated tests.

**Missing Coverage**:
- `extract_segment_audio()` function
- Edge cases (empty segments, boundary conditions)
- Memory efficiency with large files

**Recommended Test Scenarios**:
```
- test_extract_segment_audio_basic
- test_extract_segment_audio_boundaries
- test_extract_segment_audio_invalid_times
- test_extract_segment_audio_short_file
```

### 5. `emotion.py` - MEDIUM PRIORITY

**Purpose**: Emotion recognition using pre-trained wav2vec2 models.

**Current State**: No dedicated tests (likely due to heavy ML dependencies).

**Missing Coverage**:
- `EmotionRecognizerLike` protocol
- `DummyEmotionRecognizer` fallback
- `get_emotion_recognizer()` factory
- Model availability detection

**Recommended Test Scenarios**:
```
- test_emotion_recognizer_protocol
- test_dummy_emotion_recognizer_returns_neutral
- test_get_emotion_recognizer_fallback
- test_emotion_availability_flag
```

## Test File Recommendations

### Immediate Actions (Create These Files)

1. **`tests/test_pipeline.py`** - Dedicated pipeline unit tests
2. **`tests/test_models_core.py`** - Core data model tests
3. **`tests/test_exceptions.py`** - Exception hierarchy tests

### Future Actions

4. **`tests/test_audio_utils.py`** - Audio segment extraction tests
5. **`tests/test_emotion.py`** - Emotion recognition tests (with mocks)

## Test Markers Reference

When creating new tests, use appropriate pytest markers:

```python
@pytest.mark.slow                    # Skip with -m "not slow"
@pytest.mark.heavy                   # Heavy ML model tests
@pytest.mark.requires_gpu            # Skip with -m "not requires_gpu"
@pytest.mark.requires_enrich         # Requires enrichment dependencies
@pytest.mark.requires_diarization    # Requires pyannote.audio
@pytest.mark.integration             # Integration tests
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run fast tests only
uv run pytest -m "not slow and not heavy"

# Run with coverage
uv run pytest --cov=transcription --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_pipeline.py -v
```

## Metrics Goals

| Metric | Current | Target |
|--------|---------|--------|
| Modules with dedicated tests | ~60% | 80%+ |
| Core modules with tests | ~75% | 100% |
| Function-level coverage | ~70% | 85%+ |

## Document History

- **2024-12-31**: Initial analysis created
