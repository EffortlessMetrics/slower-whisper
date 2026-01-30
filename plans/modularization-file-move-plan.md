# Slower-Whisper Modularization: File-by-File Move Plan

**Version:** 1.0
**Created:** 2026-01-30
**Purpose:** Detailed file-by-file move plan for transforming monolithic slower-whisper into a monorepo with 12+ microlibraries

---

## Executive Summary

This document provides a comprehensive file-by-file move plan for modularizing the slower-whisper project. The plan maps each current file in `transcription/` and `slower_whisper/` to its target microlibrary, organized by phases that respect dependency hierarchies.

**Key Principles:**
- Move contracts (data models, exceptions, events) first as they have no dependencies
- Move foundational packages (device, config, io) next
- Move feature packages (audio, ASR, prosody, emotion) after foundations
- Move orchestration packages (streaming, intel, safety) after features
- Move server/client packages last as they depend on all other packages
- Handle cross-cutting orchestration files (pipeline.py, enrich.py, post_process.py) in meta package
- Optional features (historian/, integrations/, store/) remain separate or excluded

**Target Package Structure:**
```
packages/
├── slower-whisper-contracts/      # Core data models, exceptions, events, schemas
├── slower-whisper-config/       # EnrichmentConfig, env/file parsing
├── slower-whisper-io/            # Writers, readers, transcript persistence
├── slower-whisper-device/         # Device resolution, compute_type
├── slower-whisper-audio/          # Audio loading/resampling/chunking
├── slower-whisper-asr/            # WhisperModel, faster-whisper wrapper, asr_engine
├── slower-whisper-safety/          # Smart formatting, moderation, PII/redaction
├── slower-whisper-intel/           # Role inference, topic segmentation, turn-taking
├── slower-whisper-prosody/         # Prosody + prosody_extended + environment classifier
├── slower-whisper-emotion/         # Speech emotion model wrappers
├── slower-whisper-diarization/      # PyAnnote integration
├── slower-whisper-streaming/       # Streaming state machine, enrich session, events, callbacks
├── slower-whisper-server/          # FastAPI, WebSocket, session registry, auth
├── slower-whisper-client/          # Python SDK for REST/WS
├── slower-whisper/                 # Meta package (orchestrates all)
└── slower-whisper-compat/          # Legacy imports (compatibility shim)
```

---

## Phase 0: Preflight (Workspace Setup)

**Goal:** Set up monorepo workspace structure before moving files.

### Tasks

| ID | Task | Description |
|-----|-------|-------------|
| 0.1 | Create `packages/` directory at repository root |
| 0.2 | Create workspace configuration (pyproject.toml with [tool.poetry.workspace]) |
| 0.3 | Create initial `packages/slower-whisper/` meta package skeleton |
| 0.4 | Create initial `packages/slower-whisper-compat/` package skeleton |
| 0.5 | Set up CI/CD for workspace-level testing |

### Risk Assessment

**Risk Level:** Low

**Mitigations:**
- Workspace setup is non-destructive
- Can be tested with empty packages first
- Rollback simply involves deleting packages/ directory

---

## Phase 1: Extract Contracts

**Goal:** Extract core data models, exceptions, events, and schemas that have no dependencies.

### File Move Mapping

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/exceptions.py` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/exceptions.py` | 1 | None | None |
| `transcription/models.py` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/models.py` | 1 | None | None |
| `transcription/models_speakers.py` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/models_speakers.py` | 1 | `transcription/models.py` | Update to import from `slower_whisper_contracts.models` |
| `transcription/models_turns.py` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/models_turns.py` | 1 | `transcription/models.py` | Update to import from `slower_whisper_contracts.models` |
| `transcription/types_audio.py` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/types_audio.py` | 1 | `transcription/models.py` | Update to import from `slower_whisper_contracts.models` |
| `transcription/outcomes.py` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/outcomes.py` | 1 | `transcription/models.py` | Update to import from `slower_whisper_contracts.models` |
| `transcription/receipt.py` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/receipt.py` | 1 | `transcription/models.py` | Update to import from `slower_whisper_contracts.models` |
| `transcription/ids.py` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/ids.py` | 1 | None | None |
| `transcription/validation.py` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/validation.py` | 1 | `transcription/models.py` | Update to import from `slower_whisper_contracts.models` |
| `transcription/schema/` (all files) | `packages/slower-whisper-contracts/src/slower_whisper_contracts/schema/` | 1 | None | None |
| `transcription/schemas/stream_event.schema.json` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/schemas/stream_event.schema.json` | 1 | None | None |
| `transcription/schemas/transcript-v2.schema.json` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/schemas/transcript-v2.schema.json` | 1 | None | None |
| `transcription/schemas/pr-dossier-v2.schema.json` | `packages/slower-whisper-contracts/src/slower_whisper_contracts/schemas/pr-dossier-v2.schema.json` | 1 | None | None |

### Import Changes Summary

**After Phase 1, update all importing files:**
- `transcription/__init__.py`: Update imports from `transcription.exceptions` → `slower_whisper_contracts.exceptions`
- `transcription/__init__.py`: Update imports from `transcription.models` → `slower_whisper_contracts.models`
- `transcription/__init__.py`: Update imports from `transcription.receipt` → `slower_whisper_contracts.receipt`
- `transcription/__init__.py`: Update imports from `transcription.outcomes` → `slower_whisper_contracts.outcomes`
- `transcription/__init__.py`: Update imports from `transcription.ids` → `slower_whisper_contracts.ids`
- `transcription/__init__.py`: Update imports from `transcription.validation` → `slower_whisper_contracts.validation`
- `transcription/__init__.py`: Update imports from `transcription.types_audio` → `slower_whisper_contracts.types_audio`
- `transcription/__init__.py`: Update imports from `transcription.models_speakers` → `slower_whisper_contracts.models_speakers`
- `transcription/__init__.py`: Update imports from `transcription.models_turns` → `slower_whisper_contracts.models_turns`

### Risk Assessment

**Risk Level:** Low

**Mitigations:**
- Contracts have no external dependencies
- Pure dataclasses and enums
- Easy to test independently
- Changes are mechanical (import path updates)

**Rollback Strategy:**
- Delete `packages/slower-whisper-contracts/` directory
- Revert import changes in `transcription/__init__.py`
- All files remain in original location

---

## Phase 2: Extract Config + IO + Device

**Goal:** Extract foundational packages that depend only on contracts.

### File Move Mapping - Config Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/config.py` | `packages/slower-whisper-config/src/slower_whisper_config/config.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |
| `transcription/transcription_config.py` | `packages/slower-whisper-config/src/slower_whisper_config/transcription_config.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |
| `transcription/enrichment_config.py` | `packages/slower-whisper-config/src/slower_whisper_config/enrichment_config.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |
| `transcription/config_validation.py` | `packages/slower-whisper-config/src/slower_whisper_config/config_validation.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |
| `transcription/config_merge.py` | `packages/slower-whisper-config/src/slower_whisper_config/config_merge.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |
| `transcription/legacy_config.py` | `packages/slower-whisper-config/src/slower_whisper_config/legacy_config.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |

### File Move Mapping - IO Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/writers.py` | `packages/slower-whisper-io/src/slower_whisper_io/writers.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |
| `transcription/transcript_io.py` | `packages/slower-whisper-io/src/slower_whisper_io/transcript_io.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |
| `transcription/exporters.py` | `packages/slower-whisper-io/src/slower_whisper_io/exporters.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |

### File Move Mapping - Device Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/device.py` | `packages/slower-whisper-device/src/slower_whisper_device/device.py` | 2 | `slower-whisper-contracts` | Update imports to use `slower_whisper_contracts.*` |
| `transcription/color_utils.py` | `packages/slower-whisper-device/src/slower_whisper_device/color_utils.py` | 2 | None | None |

### Import Changes Summary

**After Phase 2, update all importing files:**
- `transcription/__init__.py`: Update imports from `transcription.config` → `slower_whisper_config.config`
- `transcription/__init__.py`: Update imports from `transcription.device` → `slower_whisper_device.device`
- `transcription/__init__.py`: Update imports from `transcription.writers` → `slower_whisper_io.writers`
- `transcription/__init__.py`: Update imports from `transcription.exporters` → `slower_whisper_io.exporters`
- `transcription/pipeline.py`: Update imports from `transcription.config` → `slower_whisper_config.config`
- `transcription/pipeline.py`: Update imports from `transcription.device` → `slower_whisper_device.device`
- `transcription/pipeline.py`: Update imports from `transcription.writers` → `slower_whisper_io.writers`
- All other files that import from these modules need similar updates

### Risk Assessment

**Risk Level:** Low-Medium

**Mitigations:**
- Config, IO, and Device have clear, well-defined boundaries
- Dependencies are only on contracts (already moved)
- Can test each package independently
- Import changes are straightforward

**Rollback Strategy:**
- Delete `packages/slower-whisper-config/`, `packages/slower-whisper-io/`, `packages/slower-whisper-device/` directories
- Revert import changes in all affected files
- All files remain in original location

---

## Phase 3: Extract Audio + ASR

**Goal:** Extract audio processing and ASR engine packages.

### File Move Mapping - Audio Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/audio_io.py` | `packages/slower-whisper-audio/src/slower_whisper_audio/audio_io.py` | 3 | `slower-whisper-contracts`, `slower-whisper-device` | Update imports to use new package paths |
| `transcription/audio_utils.py` | `packages/slower-whisper-audio/src/slower_whisper_audio/audio_utils.py` | 3 | `slower-whisper-contracts`, `slower-whisper-device` | Update imports to use new package paths |
| `transcription/audio_health.py` | `packages/slower-whisper-audio/src/slower_whisper_audio/audio_health.py` | 3 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/audio_rendering.py` | `packages/slower-whisper-audio/src/slower_whisper_audio/audio_rendering.py` | 3 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/chunking.py` | `packages/slower-whisper-audio/src/slower_whisper_audio/chunking.py` | 3 | `slower-whisper-contracts` | Update imports to use new package paths |

### File Move Mapping - ASR Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/asr_engine.py` | `packages/slower-whisper-asr/src/slower_whisper_asr/asr_engine.py` | 3 | `slower-whisper-contracts`, `slower-whisper-device`, `slower-whisper-audio` | Update imports to use new package paths |
| `transcription/cache.py` | `packages/slower-whisper-asr/src/slower_whisper_asr/cache.py` | 3 | `slower-whisper-contracts`, `slower-whisper-device` | Update imports to use new package paths |
| `transcription/transcription_orchestrator.py` | `packages/slower-whisper-asr/src/slower_whisper_asr/transcription_orchestrator.py` | 3 | `slower-whisper-contracts`, `slower-whisper-device`, `slower-whisper-audio`, `slower-whisper-asr` | Update imports to use new package paths |
| `transcription/transcription_helpers.py` | `packages/slower-whisper-asr/src/slower_whisper_asr/transcription_helpers.py` | 3 | `slower-whisper-contracts`, `slower-whisper-audio` | Update imports to use new package paths |

### Import Changes Summary

**After Phase 3, update all importing files:**
- `transcription/__init__.py`: Update imports from `transcription.asr_engine` → `slower_whisper_asr.asr_engine`
- `transcription/__init__.py`: Update imports from `transcription.cache` → `slower_whisper_asr.cache`
- `transcription/pipeline.py`: Update imports from `transcription.asr_engine` → `slower_whisper_asr.asr_engine`
- `transcription/pipeline.py`: Update imports from `transcription.audio_io` → `slower_whisper_audio.audio_io`
- All other files that import from these modules need similar updates

### Risk Assessment

**Risk Level:** Medium

**Mitigations:**
- Audio and ASR packages have clear dependencies on Phase 1-2 packages
- Audio package is relatively self-contained
- ASR package depends on faster-whisper (external dep, not affected)
- Can test each package with mock dependencies first

**Rollback Strategy:**
- Delete `packages/slower-whisper-audio/` and `packages/slower-whisper-asr/` directories
- Revert import changes in all affected files
- All files remain in original location

---

## Phase 4: Extract Safety + Intel + Prosody

**Goal:** Extract safety, intelligence, and prosody packages.

### File Move Mapping - Safety Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/smart_formatting.py` | `packages/slower-whisper-safety/src/slower_whisper_safety/smart_formatting.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/privacy.py` | `packages/slower-whisper-safety/src/slower_whisper_safety/privacy.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/safety_layer.py` | `packages/slower-whisper-safety/src/slower_whisper_safety/safety_layer.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/safety_config.py` | `packages/slower-whisper-safety/src/slower_whisper_safety/safety_config.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/renderer.py` | `packages/slower-whisper-safety/src/slower_whisper_safety/renderer.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/content_moderation.py` | `packages/slower-whisper-safety/src/slower_whisper_safety/content_moderation.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |

**Note:** There's ambiguity between `privacy.py` and `safety_layer.py`. Based on code analysis:
- `privacy.py` contains PII detection, redaction, encryption
- `safety_layer.py` contains smart formatting, moderation
- Both belong in safety package, with `safety_layer.py` being the main orchestrator

### File Move Mapping - Intel Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/role_inference.py` | `packages/slower-whisper-intel/src/slower_whisper_intel/role_inference.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/topic_segmentation.py` | `packages/slower-whisper-intel/src/slower_whisper_intel/topic_segmentation.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/turn_taking_policy.py` | `packages/slower-whisper-intel/src/slower_whisper_intel/turn_taking_policy.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/turns.py` | `packages/slower-whisper-intel/src/slower_whisper_intel/turns.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/turns_enrich.py` | `packages/slower-whisper-intel/src/slower_whisper_intel/turns_enrich.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/turn_helpers.py` | `packages/slower-whisper-intel/src/slower_whisper_intel/turn_helpers.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/tts_style.py` | `packages/slower-whisper-intel/src/slower_whisper_intel/tts_style.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/conversation_physics.py` | `packages/slower-whisper-intel/src/slower_whisper_intel/conversation_physics.py` | 4 | `slower-whisper-contracts` | Update imports to use new package paths |

### File Move Mapping - Prosody Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/prosody.py` | `packages/slower-whisper-prosody/src/slower_whisper_prosody/prosody.py` | 4 | `slower-whisper-contracts`, `slower-whisper-audio` | Update imports to use new package paths |
| `transcription/prosody_extended.py` | `packages/slower-whisper-prosody/src/slower_whisper_prosody/prosody_extended.py` | 4 | | Update imports to use new package paths |
| `transcription/environment_classifier.py` | `packages/slower-whisper-prosody/src/slower_whisper_prosody/environment_classifier.py` | 4 | `slower-whisper-contracts`, `slower-whisper-audio` | Update imports to use new package paths |

### Import Changes Summary

**After Phase 4, update all importing files:**
- `transcription/__init__.py`: Update imports from `transcription.smart_formatting` → `slower_whisper_safety.smart_formatting`
- `transcription/__init__.py`: Update imports from `transcription.privacy` → `slower_whisper_safety.privacy`
- `transcription/__init__.py`: Update imports from `transcription.role_inference` → `slower_whisper_intel.role_inference`
- `transcription/__init__.py`: Update imports from `transcription.prosody` → `slower_whisper_prosody.prosody`
- All other files that import from these modules need similar updates

### Risk Assessment

**Risk Level:** Medium

**Mitigations:**
- Safety, Intel, and Prosody packages have minimal dependencies
- Mostly depend on contracts and audio (already moved)
- Optional dependencies (librosa, parselmouth) are handled gracefully
- Can test each package independently

**Rollback Strategy:**
- Delete `packages/slower-whisper-safety/`, `packages/slower-whisper-intel/`, `packages/slower-whisper-prosody/` directories
- Revert import changes in all affected files
- All files remain in original location

---

## Phase 5: Extract Emotion + Diarization

**Goal:** Extract emotion recognition and speaker diarization packages.

### File Move Mapping - Emotion Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/emotion.py` | `packages/slower-whisper-emotion/src/slower_whisper_emotion/emotion.py` | 5 | `slower-whisper-contracts`, `slower-whisper-audio`, `slower-whisper-device` | Update imports to use new package paths |

### File Move Mapping - Diarization Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/diarization.py` | `packages/slower-whisper-diarization/src/slower_whisper_diarization/diarization.py` | 5 | `slower-whisper-contracts`, `slower-whisper-device`, `slower-whisper-audio` | Update imports to use new package paths |
| `transcription/diarization_orchestrator.py` | `packages/slower-whisper-diarization/src/slower_whisper_diarization/diarization_orchestrator.py` | 5 | `slower-whisper-contracts`, `slower-whisper-device`, `slower-whisper-audio` | Update imports to use new package paths |
| `transcription/speaker_id.py` | `packages/slower-whisper-diarization/src/slower_whisper_diarization/speaker_id.py` | 5 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/speaker_identity.py` | `packages/slower-whisper-diarization/src/slower_whisper_diarization/speaker_identity.py` | 5 | `slower-whisper-contracts`, `slower-whisper-device` | Update imports to use new package paths |
| `transcription/speaker_stats.py` | `packages/slower-whisper-diarization/src/slower_whisper_diarization/speaker_stats.py` | 5 | `slower-whisper-contracts` | Update imports to use new package paths |

### Import Changes Summary

**After Phase 5, update all importing files:**
- `transcription/__init__.py`: Update imports from `transcription.emotion` → `slower_whisper_emotion.emotion`
- `transcription/__init__.py`: Update imports from `transcription.diarization` → `slower_whisper_diarization.diarization`
- `transcription/__init__.py`: Update imports from `transcription.speaker_id` → `slower_whisper_diarization.speaker_id`
- All other files that import from these modules need similar updates

### Risk Assessment

**Risk Level:** Medium

**Mitigations:**
- Emotion and Diarization have optional heavy dependencies (torch, pyannote)
- Both packages handle missing dependencies gracefully
- Can test each package with optional deps disabled first
- Clear separation of concerns

**Rollback Strategy:**
- Delete `packages/slower-whisper-emotion/` and `packages/slower-whisper-diarization/` directories
- Revert import changes in all affected files
- All files remain in original location

---

## Phase 6: Extract Streaming Core

**Goal:** Extract streaming state machine, events, and callbacks.

### File Move Mapping - Streaming Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/streaming.py` | `packages/slower-whisper-streaming/src/slower_whisper_streaming/streaming.py` | 6 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/streaming_asr.py` | `packages/slower-whisper-streaming/src/slower_whisper_streaming/streaming_asr.py` | 6 | `slower-whisper-contracts`, `slower-whisper-asr` | Update imports to use new package paths |
| `transcription/streaming_callbacks.py` | `packages/slower-whisper-streaming/src/slower_whisper_streaming/streaming_callbacks.py` | 6 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/streaming_client.py` | `packages/slower-whisper-streaming/src/slower_whisper_streaming/streaming_client.py` | 6 | `slower-whisper-contracts`, `slower-whisper-streaming` | Update imports to use new package paths |
| `transcription/streaming_ws.py` | `packages/slower-whisper-streaming/src/slower_whisper_streaming/streaming_ws.py` | 6 | `slower-whisper-contracts`, `slower-whisper-audio`, `slower-whisper-asr`, `slower-whisper-prosody`, `slower-whisper-intel` | Update imports to use new package paths |
| `transcription/streaming_enrich.py` | `packages/slower-whisper-streaming/src/slower_whisper_streaming/streaming_enrich.py` | 6 | `slower-whisper-contracts`, `slower-whisper-prosody`, `slower-whisper-emotion` | Update imports to use new package paths |
| `transcription/streaming_semantic.py` | `packages/slower-whisper-streaming/src/slower_whisper_streaming/streaming_semantic.py` | 6 | `slower-whisper-contracts`, `slower-whisper-intel` | Update imports to use new package paths |
| `transcription/streaming_safety.py` | `packages/slower-whisper-streaming/src/slower_whisper_streaming/streaming_safety.py` | 6 | `slower-whisper-contracts`, `slower-whisper-safety` | Update imports to use new package paths |
| `transcription/streaming_diarization.py` | `packages/slower-whisper-streaming/src/slower_whisper_streaming/streaming_diarization.py` | 6 | `slower-whisper-contracts`, `slower-whisper-diarization` | Update imports to use new package paths |

### Import Changes Summary

**After Phase 6, update all importing files:**
- `transcription/__init__.py`: Update imports from `transcription.streaming` → `slower_whisper_streaming.streaming`
- `transcription/__init__.py`: Update imports from `transcription.streaming_ws` → `slower_whisper_streaming.streaming_ws`
- All other files that import from these modules need similar updates

### Risk Assessment

**Risk Level:** High

**Mitigations:**
- Streaming is complex with many interdependencies
- Depends on multiple feature packages (ASR, prosody, emotion, diarization, intel, safety)
- Must maintain event envelope protocol contracts
- Comprehensive testing required after move
- Consider moving streaming as last major package before server

**Rollback Strategy:**
- Delete `packages/slower-whisper-streaming/` directory
- Revert import changes in all affected files
- All files remain in original location
- May need to revert streaming protocol version changes if made

---

## Phase 7: Extract Server + Client

**Goal:** Extract FastAPI service and Python SDK.

### File Move Mapping - Server Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/api.py` | `packages/slower-whisper-server/src/slower_whisper_server/api.py` | 7 | `slower-whisper-contracts`, `slower-whisper-config`, `slower-whisper-asr`, `slower-whisper-streaming`, `slower-whisper-intel` | Update imports to use new package paths |
| `transcription/service.py` | `packages/slower-whisper-server/src/slower_whisper_server/service.py` | 7 | `slower-whisper-contracts`, `slower-whisper-config`, `slower-whisper-asr`, `slower-whisper-streaming` | Update imports to use new package paths |
| `transcription/service_enrich.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_enrich.py` | 7 | `slower-whisper-contracts`, `slower-whisper-config`, `slower-whisper-asr` | Update imports to use new package paths |
| `transcription/service_errors.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_errors.py` | 7 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/service_health.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_health.py` | 7 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/service_metrics.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_metrics.py` | 7 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/service_middleware.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_middleware.py` | 7 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/service_serialization.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_serialization.py` | 7 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/service_settings.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_settings.py` | 7 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/service_sse.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_sse.py` | 7 | `slower-whisper-contracts`, `slower-whisper-streaming` | Update imports to use new package paths |
| `transcription/service_streaming.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_streaming.py` | 7 | `slower-whisper-contracts`, `slower-whisper-streaming` | Update imports to use new package paths |
| `transcription/service_transcribe.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_transcribe.py` | 7 | `slower-whisper-contracts`, `slower-whisper-asr` | Update imports to use new package paths |
| `transcription/service_validation.py` | `packages/slower-whisper-server/src/slower_whisper_server/service_validation.py` | 7 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/session_registry.py` | `packages/slower-whisper-server/src/slower_whisper_server/session_registry.py` | 7 | `slower-whisper-contracts` | Update imports to use new package paths |

### File Move Mapping - Client Package

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/streaming_client.py` (duplicate - see streaming package) | `packages/slower-whisper-client/src/slower_whisper_client/client.py` | 7 | `slower-whisper-contracts`, `slower-whisper-streaming` | Update imports to use new package paths |

**Note:** `streaming_client.py` appears in both streaming and client packages. For the client package, create a wrapper that imports from streaming package and provides a simplified SDK interface.

### Import Changes Summary

**After Phase 7, update all importing files:**
- `transcription/__init__.py`: Update imports from `transcription.api` → `slower_whisper_server.api`
- `transcription/__init__.py`: Update imports from `transcription.service` → `slower_whisper_server.service`
- All other files that import from these modules need similar updates

### Risk Assessment

**Risk Level:** High

**Mitigations:**
- Server and Client depend on almost all other packages
- Server has FastAPI dependencies (websockets, uvicorn)
- Must maintain API contract compatibility
- Integration testing required after move
- Consider running server in test mode before deploying

**Rollback Strategy:**
- Delete `packages/slower-whisper-server/` and `packages/slower-whisper-client/` directories
- Revert import changes in all affected files
- All files remain in original location
- May need to revert API changes if made during move

---

## Phase 8: Meta + Compat Packaging

**Goal:** Create meta package that orchestrates all microlibraries and compat package for legacy imports.

### Cross-Cutting Orchestration Files

These files orchestrate multiple packages and belong in the meta package:

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/pipeline.py` | `packages/slower-whisper/src/slower_whisper/pipeline.py` | 8 | All other packages | Update imports to use new package paths |
| `transcription/enrich.py` | `packages/slower-whisper/src/slower_whisper/enrich.py` | 8 | `slower-whisper-config`, `slower-whisper-asr`, `slower-whisper-prosody`, `slower-whisper-emotion`, `slower-whisper-intel` | Update imports to use new package paths |
| `transcription/post_process.py` | `packages/slower-whisper/src/slower_whisper/post_process.py` | 8 | `slower-whisper-contracts`, `slower-whisper-intel` | Update imports to use new package paths |
| `transcription/audio_enrichment.py` | `packages/slower-whisper/src/slower_whisper/audio_enrichment.py` | 8 | `slower-whisper-config`, `slower-whisper-prosody`, `slower-whisper-emotion` | Update imports to use new package paths |
| `transcription/enrichment_orchestrator.py` | `packages/slower-whisper/src/slower_whisper/enrichment_orchestrator.py` | 8 | `slower-whisper-config`, `slower-whisper-prosody`, `slower-whisper-emotion`, `slower-whisper-intel` | Update imports to use new package paths |
| `transcription/cli.py` | `packages/slower-whisper/src/slower_whisper/cli.py` | 8 | All other packages | Update imports to use new package paths |
| `transcription/cli_legacy_transcribe.py` | `packages/slower-whisper/src/slower_whisper/cli_legacy_transcribe.py` | 8 | `slower-whisper-config`, `slower-whisper-asr` | Update imports to use new package paths |

### Semantic and LLM Files

These files involve semantic analysis and LLM integration:

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/semantic.py` | `packages/slower-whisper/src/slower_whisper/semantic.py` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/semantic_adapter.py` | `packages/slower-whisper/src/slower_whisper/semantic_adapter.py` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/semantic_providers/` (all files) | `packages/slower-whisper/src/slower_whisper/semantic_providers/` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/local_llm_provider.py` | `packages/slower-whisper/src/slower_whisper/local_llm_provider.py` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/llm_guardrails.py` | `packages/slower-whisper/src/slower_whisper/llm_guardrails.py` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/llm_utils.py` | `packages/slower-whisper/src/slower_whisper/llm_utils.py` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |

### Utility and Support Files

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/meta_utils.py` | `packages/slower-whisper/src/slower_whisper/meta_utils.py` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/dogfood.py` | `packages/slower-whisper/src/slower_whisper/dogfood.py` | 8 | `slower-whisper-contracts`, `slower-whisper-asr` | Update imports to use new package paths |
| `transcription/dogfood_utils.py` | `packages/slower-whisper/src/slower_whisper/dogfood_utils.py` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/telemetry.py` | `packages/slower-whisper/src/slower_whisper/telemetry.py` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/benchmarks.py` | `packages/slower-whisper/src/slower_whisper/benchmarks.py` | 8 | `slower-whisper-contracts`, `slower-whisper-asr` | Update imports to use new package paths |
| `transcription/benchmark_cli.py` | `packages/slower-whisper/src/slower_whisper/benchmark_cli.py` | 8 | `slower-whisper-contracts`, `slower-whisper-asr` | Update imports to use new package paths |
| `transcription/benchmark/` (all files) | `packages/slower-whisper/src/slower_whisper/benchmark/` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/cli_commands/` (all files) | `packages/slower-whisper/src/slower_whisper/cli_commands/` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/samples.py` | `packages/slower-whisper/src/slower_whisper/samples.py` | 8 | `slower-whisper-contracts` | Update imports to use new package paths |
| `transcription/_import_guards.py` | `packages/slower-whisper/src/slower_whisper/_import_guards.py` | 8 | None | None |
| `transcription/py.typed` | `packages/slower-whisper/src/slower_whisper/py.typed` | 8 | None | None |

### Meta Package __init__.py

The meta package's `__init__.py` will re-export all public APIs from microlibraries:

```python
# Re-export from contracts
from slower_whisper_contracts import (
    Transcript, Segment, Turn, Word, Chunk,
    SlowerWhisperError, TranscriptionError, EnrichmentError, ConfigurationError,
    Receipt, build_receipt,
    # ... all other contracts exports
)

# Re-export from config
from slower_whisper_config import (
    TranscriptionConfig, EnrichmentConfig,
    # ... all other config exports
)

# Re-export from device
from slower_whisper_device import (
    resolve_device, ResolvedDevice,
    # ... all other device exports
)

# Re-export from audio
from slower_whisper_audio import (
    load_audio, normalize_audio,
    # ... all other audio exports
)

# Re-export from ASR
from slower_whisper_asr import (
    transcribe_file, WhisperModel,
    # ... all other ASR exports
)

# Re-export from safety
from slower_whisper_safety import (
    format_smart, redact_pii,
    # ... all other safety exports
)

# Re-export from intel
from slower_whisper_intel import (
    infer_roles, segment_topics,
    # ... all other intel exports
)

# Re-export from prosody
from slower_whisper_prosody import (
    extract_prosody,
    # ... all other prosody exports
)

# Re-export from emotion
from slower_whisper_emotion import (
    extract_emotion_dimensional, extract_emotion_categorical,
    # ... all other emotion exports
)

# Re-export from diarization
from slower_whisper_diarization import (
    Diarizer, assign_speakers,
    # ... all other diarization exports
)

# Re-export from streaming
from slower_whisper_streaming import (
    StreamingSession, EventEnvelope, WebSocketStreamingSession,
    # ... all other streaming exports
)

# Re-export from server
from slower_whisper_server import (
    create_app, run_server,
    # ... all other server exports
)

# Re-export from client
from slower_whisper_client import (
    StreamingClient, create_client,
    # ... all other client exports
)

# Re-export from io
from slower_whisper_io import (
    write_json, write_txt, write_srt,
    # ... all other io exports
)
```

### Compat Package

The compat package provides legacy import paths:

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/__init__.py` (legacy exports) | `packages/slower-whisper-compat/src/slower_whisper_compat/__init__.py` | 8 | `slower-whisper` | Re-export from meta package |

The compat package `__init__.py` will maintain backward compatibility by re-exporting from the meta package with the same import paths as before:

```python
# Legacy compatibility: maintain old import paths
# Users importing from transcription.* will get redirected to new packages

from slower_whisper import (
    # All the same exports that were in transcription/__init__.py
    # This ensures backward compatibility
)
```

### Import Changes Summary

**After Phase 8, update all importing files:**
- All files in `transcription/` that import from moved modules must be updated
- External code importing from `transcription.*` will work via compat package
- Tests may need import path updates

### Risk Assessment

**Risk Level:** High

**Mitigations:**
- Meta package has complex dependencies on all other packages
- Must maintain exact public API contract
- Compat package ensures backward compatibility
- Comprehensive integration testing required
- Version pinning critical (all packages must have compatible versions)

**Rollback Strategy:**
- Delete `packages/slower-whisper/` and `packages/slower-whisper-compat/` directories
- Keep original `transcription/__init__.py` intact
- All other files remain in original location
- May need to revert version changes in pyproject.toml

---

## Phase 9: Optional Features (Separate or Exclude)

**Goal:** Handle optional features that are not part of core modularization.

### Optional Features Mapping

| Source Path | Target Path | Priority | Dependencies | Import Changes Needed |
|--------------|--------------|-----------|---------------|---------------------|
| `transcription/historian/` | **EXCLUDE** - Keep as separate optional feature | 9 | N/A | N/A |
| `transcription/integrations/` | **EXCLUDE** - Keep as separate optional feature | 9 | N/A | N/A |
| `transcription/store/` | **EXCLUDE** - Keep as separate optional feature | 9 | N/A | N/A |

**Decision:** These optional features should remain in the main repository outside the `packages/` directory. They can be:
- Kept as-is in `transcription/` after modularization
- Moved to a separate `optional-features/` directory
- Converted to standalone repositories in the future

### Risk Assessment

**Risk Level:** None

**Mitigations:**
- These are truly optional features
- Not part of core public API
- Can be excluded without breaking changes
- Decision documented in migration guide

---

## Import Change Patterns

### Common Import Transformations

When moving files, imports need to be updated as follows:

**Old Pattern:**
```python
from transcription.models import Transcript, Segment
from transcription.config import TranscriptionConfig
from transcription.device import resolve_device
from transcription.streaming import StreamingSession
```

**New Pattern:**
```python
from slower_whisper_contracts.models import Transcript, Segment
from slower_whisper_config.config import TranscriptionConfig
from slower_whisper_device.device import resolve_device
from slower_whisper_streaming.streaming import StreamingSession
```

### Relative vs Absolute Imports

- Use absolute imports from package roots (e.g., `from slower_whisper_contracts.models import...`)
- Avoid relative imports across package boundaries
- Each package should have its own namespace

### Circular Dependency Prevention

The move order prevents circular dependencies:
1. Contracts have no dependencies
2. Config, IO, Device depend only on contracts
3. Audio, ASR depend on contracts, device, io
4. Safety, Intel, Prosody depend on contracts, audio
5. Emotion, Diarization depend on contracts, audio, device
6. Streaming depends on contracts, audio, asr, prosody, emotion, diarization, intel, safety
7. Server, Client depend on almost everything
8. Meta package orchestrates all packages

---

## Risk Assessment Summary

### Per-Phase Risk Levels

| Phase | Risk Level | Primary Concerns | Mitigations |
|--------|--------------|-------------------|--------------|
| Phase 0: Preflight | Low | Workspace setup errors | Test with empty packages first |
| Phase 1: Contracts | Low | Import path errors | Pure dataclasses, easy to test |
| Phase 2: Config+IO+Device | Low-Medium | Config breaking changes | Clear boundaries, well-tested |
| Phase 3: Audio+ASR | Medium | Audio processing errors | Optional deps handled gracefully |
| Phase 4: Safety+Intel+Prosody | Medium | Feature interaction issues | Minimal interdependencies |
| Phase 5: Emotion+Diarization | Medium | Optional dep failures | Graceful degradation |
| Phase 6: Streaming | High | Event protocol breaking | Comprehensive testing |
| Phase 7: Server+Client | High | API contract breaking | Integration testing |
| Phase 8: Meta+Compat | High | Public API breaking | Version pinning |
| Phase 9: Optional Features | None | Feature isolation | Document exclusion |

### Overall Risk Assessment

**Highest Risk Areas:**
1. **Streaming package** - Complex event protocol with many dependencies
2. **Server package** - Depends on almost all packages, API contract critical
3. **Meta package** - Must maintain exact public API for backward compatibility
4. **Import path updates** - Mechanical but error-prone across many files

**Critical Success Factors:**
1. Maintain exact public API contracts
2. Comprehensive testing at each phase
3. Version compatibility matrix
4. Clear documentation of changes
5. Gradual rollout with ability to rollback

---

## Rollback Strategy

### Per-Phase Rollback

**Phase 0-1 (Low Risk):**
- Delete created package directories
- Revert any import changes
- All files remain in original location
- Minimal impact

**Phase 2-5 (Medium Risk):**
- Delete created package directories
- Revert import changes in affected files
- All files remain in original location
- May need to revert config changes if made

**Phase 6-7 (High Risk):**
- Delete created package directories
- Revert import changes in affected files
- All files remain in original location
- May need to revert protocol version changes
- May need to revert API changes if made

**Phase 8 (Meta Package):**
- Delete `packages/slower-whisper/` directory
- Keep original `transcription/__init__.py` intact
- May need to revert version changes
- Revert pyproject.toml workspace changes

### Full Rollback

If complete rollback is needed:

1. **Stop all CI/CD pipelines**
2. **Delete `packages/` directory entirely**
3. **Revert `pyproject.toml` to pre-modularization state**
4. **Revert any import changes made during modularization**
5. **Verify all tests pass with original structure**
6. **Document rollback reason and lessons learned**

### Rollback Triggers

Consider rollback if:
- Critical bugs discovered that cannot be fixed without reverting
- Performance regression > 20% in core workflows
- Breaking changes detected in public API
- Integration tests fail consistently after a phase
- User reports indicate widespread issues

---

## Testing Strategy

### Per-Phase Testing

**After each phase:**
1. Run unit tests for moved package
2. Run integration tests that depend on moved package
3. Verify import paths work correctly
4. Check that no circular dependencies exist
5. Validate package can be installed independently

**Before next phase:**
1. Ensure all tests from previous phases still pass
2. Verify no breaking changes to public API
3. Check that optional dependencies work correctly

### Final Testing

**After Phase 8:**
1. Run full test suite
2. Test backward compatibility via compat package
3. Test all public API entry points
4. Verify CLI works correctly
5. Verify server starts and responds
6. Test client can connect and stream
7. Validate version compatibility matrix

---

## Migration Guide

### For Users

**No Action Required:**
- Users importing from `transcription.*` will continue to work via compat package
- No code changes needed for existing users
- New users can import directly from specific packages

**For Contributors:**
- Use new package structure when adding features
- Import from specific packages rather than `transcription.*`
- Follow dependency hierarchy when adding new code

### For Package Maintainers

Each microlibrary should:
1. Have its own `pyproject.toml` with version
2. Depend on appropriate packages from workspace
3. Export public API via `__init__.py`
4. Include tests in `tests/` subdirectory
5. Document public API in README

---

## Appendix: Complete File Inventory

### All Files by Current Location

**transcription/ (root files):**
- `__init__.py` → Meta package (after Phase 8)
- `_import_guards.py` → Meta package
- `py.typed` → Meta package

**Contracts (Phase 1):**
- `exceptions.py`, `models.py`, `models_speakers.py`, `models_turns.py`, `types_audio.py`, `outcomes.py`, `receipt.py`, `ids.py`, `validation.py`

**Config (Phase 2):**
- `config.py`, `transcription_config.py`, `enrichment_config.py`, `config_validation.py`, `config_merge.py`, `legacy_config.py`

**IO (Phase 2):**
- `writers.py`, `transcript_io.py`, `exporters.py`

**Device (Phase 2):**
- `device.py`, `color_utils.py`

**Audio (Phase 3):**
- `audio_io.py`, `audio_utils.py`, `audio_health.py`, `audio_rendering.py`, `chunking.py`

**ASR (Phase 3):**
- `asr_engine.py`, `cache.py`, `transcription_orchestrator.py`, `transcription_helpers.py`

**Safety (Phase 4):**
- `smart_formatting.py`, `privacy.py`, `safety_layer.py`, `safety_config.py`, `renderer.py`, `content_moderation.py`

**Intel (Phase 4):**
- `role_inference.py`, `topic_segmentation.py`, `turn_taking_policy.py`, `turns.py`, `turns_enrich.py`, `turn_helpers.py`, `tts_style.py`, `conversation_physics.py`

**Prosody (Phase 4):**
- `prosody.py`, `prosody_extended.py`, `environment_classifier.py`

**Emotion (Phase 5):**
- `emotion.py`

**Diarization (Phase 5):**
- `diarization.py`, `diarization_orchestrator.py`, `speaker_id.py`, `speaker_identity.py`, `speaker_stats.py`

**Streaming (Phase 6):**
- `streaming.py`, `streaming_asr.py`, `streaming_callbacks.py`, `streaming_client.py`, `streaming_ws.py`, `streaming_enrich.py`, `streaming_semantic.py`, `streaming_safety.py`, `streaming_diarization.py`

**Server (Phase 7):**
- `api.py`, `service.py`, `service_enrich.py`, `service_errors.py`, `service_health.py`, `service_metrics.py`, `service_middleware.py`, `service_serialization.py`, `service_settings.py`, `service_sse.py`, `service_streaming.py`, `service_transcribe.py`, `service_validation.py`, `session_registry.py`

**Client (Phase 7):**
- `streaming_client.py` (also in streaming, create wrapper for client package)

**Meta/Orchestration (Phase 8):**
- `pipeline.py`, `enrich.py`, `post_process.py`, `audio_enrichment.py`, `enrichment_orchestrator.py`, `cli.py`, `cli_legacy_transcribe.py`

**Semantic/LLM (Phase 8):**
- `semantic.py`, `semantic_adapter.py`, `semantic_providers/`, `local_llm_provider.py`, `llm_guardrails.py`, `llm_utils.py`

**Utilities (Phase 8):**
- `meta_utils.py`, `dogfood.py`, `dogfood_utils.py`, `telemetry.py`, `benchmarks.py`, `benchmark_cli.py`, `benchmark/`, `cli_commands/`, `samples.py`

**Optional Features (Phase 9 - Exclude):**
- `historian/`, `integrations/`, `store/`

**Schema:**
- `schema/`, `schemas/`

**slower_whisper/ (compatibility layer):**
- `__init__.py`, `compat.py`, `model.py`, `py.typed`

---

## Conclusion

This file-by-file move plan provides a comprehensive roadmap for modularizing slower-whisper into 12+ microlibraries. The plan:

1. Respects dependency hierarchy by moving contracts first
2. Groups moves into logical phases with clear priorities
3. Documents all import changes needed for each file
4. Assesses risks per phase with mitigations
5. Provides rollback strategies for each phase
6. Handles optional features separately
7. Maintains backward compatibility via compat package

**Next Steps:**
1. Review and approve this plan
2. Begin Phase 0 (workspace setup)
3. Execute phases sequentially with testing at each step
4. Monitor for issues and apply rollback if needed

---

**Document Version:** 1.0
**Last Updated:** 2026-01-30
