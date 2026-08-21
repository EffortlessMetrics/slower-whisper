from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    source = file_path.read_text(encoding="utf-8")
    count = source.count(old)
    assert count == 1, f"{path}: expected one anchor, found {count}: {old[:80]!r}"
    file_path.write_text(source.replace(old, new, 1), encoding="utf-8")


# Typed runtime failures.
replace_once(
    "transcription/exceptions.py",
    '''class ASROutputError(ASRError):
    """Raised when an ASR backend returns malformed or invalid output."""

    default_reason_code = "asr_output_invalid"


class EnrichmentError''',
    '''class ASROutputError(ASRError):
    """Raised when an ASR backend returns malformed or invalid output."""

    default_reason_code = "asr_output_invalid"


class RuntimeNotReadyError(TranscriptionError):
    """Raised when the process-owned service runtime cannot serve requests."""

    default_reason_code = "runtime_not_ready"


class RuntimeProfileMismatchError(TranscriptionError):
    """Raised when a request asks the process to load a different ASR profile."""

    default_reason_code = "runtime_profile_mismatch"


class EnrichmentError''',
)

# Private engine injection through the public facade preserves existing patch points.
replace_once(
    "transcription/api.py",
    "from pathlib import Path\n",
    "from pathlib import Path\nfrom typing import TYPE_CHECKING\n",
)
replace_once(
    "transcription/api.py",
    "from .config import EnrichmentConfig, TranscriptionConfig\n",
    '''from .config import EnrichmentConfig, TranscriptionConfig

if TYPE_CHECKING:
    from .asr_engine import TranscriptionEngine
''',
)
replace_once(
    "transcription/api.py",
    '''def transcribe_file(
    audio_path: str | Path,
    root: str | Path,
    config: TranscriptionConfig,
) -> Transcript:''',
    '''def transcribe_file(
    audio_path: str | Path,
    root: str | Path,
    config: TranscriptionConfig,
    *,
    _engine: TranscriptionEngine | None = None,
) -> Transcript:''',
)
replace_once(
    "transcription/api.py",
    '''        maybe_run_diarization=_maybe_run_diarization,
        maybe_build_chunks=_maybe_build_chunks,
    )


def transcribe_bytes''',
    '''        maybe_run_diarization=_maybe_run_diarization,
        maybe_build_chunks=_maybe_build_chunks,
        engine=_engine,
    )


def transcribe_bytes''',
)
replace_once(
    "transcription/api.py",
    '''    *,
    file_name: str = "audio.wav",
) -> Transcript:''',
    '''    *,
    file_name: str = "audio.wav",
    _engine: TranscriptionEngine | None = None,
) -> Transcript:''',
)
replace_once(
    "transcription/api.py",
    '''        maybe_run_diarization=_maybe_run_diarization,
        maybe_build_chunks=_maybe_build_chunks,
    )

    # Set the file_name''',
    '''        maybe_run_diarization=_maybe_run_diarization,
        maybe_build_chunks=_maybe_build_chunks,
        engine=_engine,
    )

    # Set the file_name''',
)

# Orchestrators construct engines only when a caller did not provide one.
replace_once(
    "transcription/transcription_orchestrator.py",
    "from pathlib import Path\n",
    "from pathlib import Path\nfrom typing import TYPE_CHECKING\n",
)
replace_once(
    "transcription/transcription_orchestrator.py",
    "from .writers import load_transcript_from_json\n",
    '''from .writers import load_transcript_from_json

if TYPE_CHECKING:
    from .asr_engine import TranscriptionEngine
''',
)
replace_once(
    "transcription/transcription_orchestrator.py",
    '''    maybe_run_diarization: MaybeRunDiarizationFn,
    maybe_build_chunks: MaybeBuildChunksFn,
) -> Transcript:
    from .asr_engine import TranscriptionEngine
''',
    '''    maybe_run_diarization: MaybeRunDiarizationFn,
    maybe_build_chunks: MaybeBuildChunksFn,
    engine: TranscriptionEngine | None = None,
) -> Transcript:
''',
)
replace_once(
    "transcription/transcription_orchestrator.py",
    '''    engine = TranscriptionEngine(asr_cfg)
    transcript = engine.transcribe_file(norm_wav)
''',
    '''    if engine is None:
        from .asr_engine import TranscriptionEngine

        engine = TranscriptionEngine(asr_cfg)
    transcript = engine.transcribe_file(norm_wav)
''',
)
replace_once(
    "transcription/transcription_orchestrator.py",
    '''    maybe_run_diarization: MaybeRunDiarizationFn,
    maybe_build_chunks: MaybeBuildChunksFn,
) -> Transcript:
    import re
    import tempfile

    from .asr_engine import TranscriptionEngine
''',
    '''    maybe_run_diarization: MaybeRunDiarizationFn,
    maybe_build_chunks: MaybeBuildChunksFn,
    engine: TranscriptionEngine | None = None,
) -> Transcript:
    import re
    import tempfile
''',
)
replace_once(
    "transcription/transcription_orchestrator.py",
    '''        engine = TranscriptionEngine(asr_cfg)
        transcript = engine.transcribe_file(norm_path)
''',
    '''        if engine is None:
            from .asr_engine import TranscriptionEngine

            engine = TranscriptionEngine(asr_cfg)
        transcript = engine.transcribe_file(norm_path)
''',
)

# Canonical schema key plus legacy API key.
replace_once(
    "transcription/service_serialization.py",
    '''        "schema_version": SCHEMA_VERSION,
        "file_name": transcript.file_name,
''',
    '''        "schema_version": SCHEMA_VERSION,
        "file": transcript.file_name,
        "file_name": transcript.file_name,
''',
)

# Structured 409 responses for fixed runtime-profile conflicts.
replace_once(
    "transcription/service_errors.py",
    '''        404: "not_found",
        413: "file_too_large",
''',
    '''        404: "not_found",
        409: "conflict",
        413: "file_too_large",
''',
)

# Replace only the batch REST endpoint; leave the separately tracked SSE implementation intact.
service_path = Path("transcription/service_transcribe.py")
service_source = service_path.read_text(encoding="utf-8")
marker = '''# =============================================================================
# SSE Streaming Transcription Endpoint
# =============================================================================
'''
assert service_source.count(marker) == 1
_suffix = service_source.split(marker, 1)[1]
_prefix = '''"""Transcription endpoints for the API service."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from .api import transcribe_file
from .config import TranscriptionConfig, WhisperTask, validate_compute_type
from .exceptions import (
    ASRInferenceError,
    ASRModelLoadError,
    ASROutputError,
    ASRUnavailableError,
    ConfigurationError,
    RuntimeNotReadyError,
    RuntimeProfileMismatchError,
    TranscriptionError,
)
from .service_errors import create_error_response
from .service_serialization import _transcript_to_dict
from .service_settings import HTTP_413_TOO_LARGE, MAX_AUDIO_SIZE_MB
from .service_sse import _generate_sse_transcription
from .service_validation import save_upload_file_streaming, validate_audio_format

logger = logging.getLogger(__name__)
router = APIRouter()


def _request_runtime(request: Request):
    runtime = getattr(request.app.state, "asr_runtime", None)
    if runtime is not None:
        return runtime
    startup_error = getattr(request.app.state, "asr_startup_error", None)
    context: dict[str, Any] = {"state": "failed" if startup_error else "stopped"}
    if isinstance(startup_error, TranscriptionError):
        context["startup_error"] = startup_error.public_details()
    raise RuntimeNotReadyError(
        "The configured ASR runtime is not ready",
        context=context,
    )


def _asr_error_response(request: Request, exc: TranscriptionError) -> JSONResponse:
    if isinstance(exc, RuntimeProfileMismatchError):
        status_code = 409
        message = "Request does not match the configured ASR profile"
    elif isinstance(exc, (ASRUnavailableError, ASRModelLoadError, RuntimeNotReadyError)):
        status_code = 503
        message = "ASR runtime is not ready"
    elif isinstance(exc, (ASRInferenceError, ASROutputError)):
        status_code = 500
        message = "Transcription failed"
    else:
        status_code = 500
        message = "Transcription failed"
    return create_error_response(
        status_code=status_code,
        error_type=exc.reason_code,
        message=message,
        request_id=getattr(request.state, "request_id", None),
        details=exc.public_details(),
    )


@router.post(
    "/transcribe",
    summary="Transcribe audio file",
    description=(
        "Upload audio and transcribe it with the process-owned ASR profile. "
        "Explicit ASR overrides must match the configured service runtime."
    ),
    tags=["Transcription"],
    response_model=None,
)
async def transcribe_audio(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    request: Request,
    model: Annotated[str | None, Query(description="Configured Whisper model")] = None,
    language: Annotated[str | None, Query(description="Configured language hint")] = None,
    device: Annotated[str | None, Query(description="Configured device ('cuda' or 'cpu')")] = None,
    compute_type: Annotated[str | None, Query(description="Configured compute precision")] = None,
    task: Annotated[str | None, Query(description="Configured task")]=None,
    enable_diarization: Annotated[bool, Query(description="Run speaker diarization")]=False,
    diarization_device: Annotated[str, Query(description="Diarization device")]= "auto",
    min_speakers: Annotated[int | None, Query(ge=1)] = None,
    max_speakers: Annotated[int | None, Query(ge=1)] = None,
    overlap_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    word_timestamps: Annotated[bool | None, Query(description="Configured word timestamps")]=None,
) -> JSONResponse:
    if task is not None and task not in ("transcribe", "translate"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be 'transcribe' or 'translate'.",
        )
    if device is not None and device not in ("cuda", "cpu"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device '{device}'. Must be 'cuda' or 'cpu'.",
        )
    if diarization_device not in ("cuda", "cpu", "auto"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid diarization_device '{diarization_device}'. "
                "Must be 'cuda', 'cpu', or 'auto'."
            ),
        )
    try:
        normalized_compute_type = (
            validate_compute_type(compute_type) if compute_type is not None else None
        )
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid compute_type '{compute_type}'. See logs for details.",
        ) from exc

    try:
        runtime = _request_runtime(request)
        profile = runtime.profile
        resolved_model = model or profile.model
        resolved_language = language if language is not None else profile.language
        resolved_device = device or profile.device
        resolved_compute_type = normalized_compute_type or profile.compute_type
        resolved_task = task or profile.task
        resolved_word_timestamps = (
            word_timestamps if word_timestamps is not None else profile.word_timestamps
        )
        task_value = cast(WhisperTask, resolved_task)

        extra_kwargs: dict[str, Any] = {}
        if overlap_threshold is not None:
            extra_kwargs["overlap_threshold"] = overlap_threshold
        config = TranscriptionConfig(
            model=resolved_model,
            language=resolved_language,
            device=resolved_device,
            compute_type=resolved_compute_type,
            task=task_value,
            beam_size=profile.beam_size,
            vad_min_silence_ms=profile.vad_min_silence_ms,
            skip_existing_json=False,
            enable_diarization=enable_diarization,
            diarization_device=diarization_device,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            word_timestamps=resolved_word_timestamps,
            **extra_kwargs,
        )
        runtime.assert_profile(config)
    except RuntimeProfileMismatchError as exc:
        return _asr_error_response(request, exc)
    except RuntimeNotReadyError as exc:
        return _asr_error_response(request, exc)
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid transcription configuration", exc_info=exc)
        raise HTTPException(
            status_code=400,
            detail="Invalid transcription configuration. Check parameter values.",
        ) from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        safe_suffix = ""
        if audio.filename:
            import re

            ext_match = re.search(r"(\\.[^.]+)$", audio.filename)
            if ext_match:
                ext = ext_match.group(1)
                allowed_extensions = {
                    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"
                }
                if ext.lower() in allowed_extensions:
                    safe_suffix = ext

        import secrets

        audio_path = tmpdir_path / f"audio_{secrets.token_hex(16)}{safe_suffix}"
        try:
            await save_upload_file_streaming(
                audio,
                audio_path,
                max_bytes=MAX_AUDIO_SIZE_MB * 1024 * 1024,
                file_type="audio",
            )
        except HTTPException as exc:
            if exc.status_code == HTTP_413_TOO_LARGE:
                raise HTTPException(
                    status_code=HTTP_413_TOO_LARGE,
                    detail=f"File too large: >{MAX_AUDIO_SIZE_MB} MB",
                ) from exc
            raise

        validate_audio_format(audio_path)
        try:
            logger.info(
                "Starting service transcription: model=%s device=%s compute_type=%s",
                config.model,
                config.device,
                config.compute_type,
            )
            transcript = await runtime.transcribe_file(
                transcribe_file,
                audio_path=audio_path,
                root=tmpdir_path,
                config=config,
            )
        except ConfigurationError as exc:
            logger.error("Configuration error during transcription", exc_info=exc)
            raise HTTPException(
                status_code=400,
                detail="Configuration error during transcription. Check parameter values.",
            ) from exc
        except TranscriptionError as exc:
            logger.error("Transcription failed", exc_info=exc)
            return _asr_error_response(request, exc)
        except Exception as exc:
            logger.exception("Unexpected error during transcription")
            raise HTTPException(
                status_code=500,
                detail="Unexpected error during transcription",
            ) from exc

        return JSONResponse(
            content=_transcript_to_dict(
                transcript,
                include_words=config.word_timestamps,
            ),
            status_code=200,
        )


'''
service_path.write_text(_prefix + marker + _suffix, encoding="utf-8")

# Test-only runtime factory fixture. It keeps endpoint patch points intact.
conftest = Path("tests/conftest.py")
conftest_source = conftest.read_text(encoding="utf-8")
fixture_marker = "\n\n# Service runtime fixture for API unit tests (Issue #623)\n"
assert fixture_marker not in conftest_source
conftest_source += fixture_marker + r'''
class _ServiceTestEngine:
    def __init__(self) -> None:
        from types import SimpleNamespace

        self.cfg = SimpleNamespace(
            model_name="large-v3",
            device="cpu",
            compute_type="int8",
            language=None,
            task="transcribe",
            beam_size=5,
            vad_min_silence_ms=500,
            word_timestamps=False,
        )
        self.model_load_attempts = [
            {
                "device": "cpu",
                "compute_type": "int8",
                "outcome": "selected",
                "reason_code": "ok",
            }
        ]
        self.close_count = 0

    def transcribe_file(self, audio_path):
        from pathlib import Path
        from transcription.models import Transcript

        return Transcript(
            file_name=Path(audio_path).name,
            language="en",
            segments=[],
            meta={
                "asr_backend": "faster-whisper",
                "asr_device": "cpu",
                "asr_compute_type": "int8",
                "asr_model_load_attempts": [dict(item) for item in self.model_load_attempts],
            },
        )

    def close(self) -> None:
        self.close_count += 1


class _PassthroughServiceRuntime:
    def __init__(self) -> None:
        from transcription.service_runtime import RuntimeProfile, RuntimeState

        self.profile = RuntimeProfile(
            model="large-v3",
            device="cpu",
            compute_type="int8",
            language=None,
            task="transcribe",
            beam_size=5,
            vad_min_silence_ms=500,
            word_timestamps=False,
        )
        self.engine = _ServiceTestEngine()
        self.ready = True
        self.state = RuntimeState.READY

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        self.engine.close()

    def assert_profile(self, _config) -> None:
        return None

    def status(self):
        return {
            "state": self.state.value,
            "ready": True,
            "profile": self.profile.to_dict(),
            "selected": self.profile.to_dict(),
            "attempts": [dict(item) for item in self.engine.model_load_attempts],
            "max_concurrency": 1,
            "active_inferences": 0,
            "max_observed_inferences": 1,
            "error": None,
        }

    async def transcribe_file(self, transcribe, *, audio_path, root, config):
        return transcribe(audio_path, root, config, _engine=self.engine)

    async def deep_probe(self, audio_path):
        return self.engine.transcribe_file(audio_path)


@pytest.fixture
def service_runtime_factory():
    return _PassthroughServiceRuntime
'''
conftest.write_text(conftest_source, encoding="utf-8")


def replace_client_fixture(path: str, old_import: str, old_fixture: str, new_fixture: str) -> None:
    replace_once(path, old_import, "from transcription.service import create_app  # noqa: E402\n")
    replace_once(path, old_fixture, new_fixture)


replace_client_fixture(
    "tests/test_service.py",
    "from transcription.service import app  # noqa: E402\n",
    '''@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)
''',
    '''@pytest.fixture
def client(service_runtime_factory) -> TestClient:
    """Create a client with an explicit ready service runtime."""
    application = create_app(runtime_factory=service_runtime_factory)
    with TestClient(application) as test_client:
        yield test_client
''',
)
replace_client_fixture(
    "tests/test_api_service.py",
    "from transcription.service import app  # noqa: E402\n",
    '''@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Patch normalize_all for all tests to avoid calling ffmpeg
    with patch("transcription.audio_io.normalize_all"):
        yield TestClient(app)
''',
    '''@pytest.fixture
def client(service_runtime_factory):
    """Create a client with an explicit ready service runtime."""
    application = create_app(runtime_factory=service_runtime_factory)
    with patch("transcription.audio_io.normalize_all"), TestClient(application) as test_client:
        yield test_client
''',
)
replace_client_fixture(
    "tests/test_security_leak.py",
    "from transcription.service import app\n",
    '''@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
''',
    '''@pytest.fixture
def client(service_runtime_factory) -> TestClient:
    application = create_app(runtime_factory=service_runtime_factory)
    with TestClient(application) as test_client:
        yield test_client
''',
)
replace_client_fixture(
    "tests/test_service_health.py",
    "from transcription.service import app  # noqa: E402\n",
    '''@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)
''',
    '''@pytest.fixture
def client(service_runtime_factory):
    """Create a test client with a ready runtime."""
    application = create_app(runtime_factory=service_runtime_factory)
    with TestClient(application) as test_client:
        yield test_client
''',
)

# Health acceptance now treats the loaded runtime as the critical ASR check.
replace_once(
    "tests/test_service_health.py",
    '''        assert "ffmpeg" in checks
        assert "faster_whisper" in checks
        assert "cuda" in checks
        assert "disk_space" in checks
''',
    '''        assert "ffmpeg" in checks
        assert "resources" in checks
        assert "runtime" in checks
        assert "faster_whisper" in checks
        assert "cuda" in checks
        assert "disk_space" in checks
''',
)
replace_once(
    "tests/test_service_health.py",
    '''        # If any critical check (ffmpeg or faster_whisper) fails, should be 503
        ffmpeg_status = data["checks"]["ffmpeg"]["status"]
        whisper_status = data["checks"]["faster_whisper"]["status"]

        if ffmpeg_status == "error" or whisper_status == "error":
''',
    '''        # Runtime, ffmpeg, and installed resources are the critical checks.
        ffmpeg_status = data["checks"]["ffmpeg"]["status"]
        resource_status = data["checks"]["resources"]["status"]
        runtime_status = data["checks"]["runtime"]["status"]

        if "error" in {ffmpeg_status, resource_status, runtime_status}:
''',
)
replace_once(
    "tests/test_service_health.py",
    '''        # If both critical checks pass, should be 200
        ffmpeg_status = data["checks"]["ffmpeg"]["status"]
        whisper_status = data["checks"]["faster_whisper"]["status"]

        if ffmpeg_status == "ok" and whisper_status == "ok":
''',
    '''        # If every critical check passes, readiness should be 200.
        ffmpeg_status = data["checks"]["ffmpeg"]["status"]
        resource_status = data["checks"]["resources"]["status"]
        runtime_status = data["checks"]["runtime"]["status"]

        if {ffmpeg_status, resource_status, runtime_status} == {"ok"}:
''',
)
