"""Configuration rejection parity across REST, SSE, and the CLI."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from transcription.cli import main  # noqa: E402
from transcription.config import TranscriptionConfig  # noqa: E402
from transcription.service import create_app  # noqa: E402
from transcription.service_runtime import ASRRuntime, RuntimeProfile  # noqa: E402
from transcription.service_transcribe import save_upload_file_streaming  # noqa: E402


@pytest.fixture
def client():
    """Use the production app lifecycle with an inert, process-owned engine."""
    config = TranscriptionConfig(model="tiny", device="cpu", compute_type="int8")
    runtime = ASRRuntime(
        RuntimeProfile.from_config(config),
        engine_factory=lambda cfg: SimpleNamespace(cfg=cfg, close=lambda: None),
    )
    with TestClient(
        create_app(runtime_factory=lambda: runtime), raise_server_exceptions=False
    ) as c:
        yield c


@pytest.fixture
def audio_bytes() -> bytes:
    """Supply a valid tiny PCM container without a fixture or model download."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


@pytest.mark.parametrize("model", ["unknown-model", "/synthetic/private/model", "TINY"])
@pytest.mark.parametrize("endpoint", ["/transcribe", "/transcribe/stream"])
def test_invalid_model_is_bad_request_before_audio_work(
    client: TestClient, audio_bytes: bytes, endpoint: str, model: str
) -> None:
    """Real config rejection must remain a client error, never an internal failure."""
    with (
        patch(
            "transcription.service_transcribe.save_upload_file_streaming",
            wraps=save_upload_file_streaming,
        ) as save,
        patch("transcription.service_transcribe.validate_audio_format") as validate,
        patch(
            "transcription.audio_io.normalize_single",
            side_effect=lambda src, dst: dst.write_bytes(audio_bytes),
        ) as normalize,
        patch("transcription.service_transcribe.transcribe_file") as transcribe,
        patch("transcription.service_transcribe._generate_sse_transcription") as stream,
    ):
        response = client.post(
            endpoint,
            params={"model": model},
            files={"audio": ("synthetic.wav", audio_bytes, "audio/wav")},
            headers={"X-Request-ID": "config-parity-request"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == {
        "type": "bad_request",
        "message": "Invalid transcription configuration. Check parameter values.",
        "status_code": 400,
        "request_id": response.headers["x-request-id"],
    }
    assert body["detail"] == body["error"]["message"]
    assert model not in response.text
    assert response.headers["content-type"] == "application/json"
    for operation in (save, validate, normalize, transcribe, stream):
        operation.assert_not_called()


@pytest.mark.parametrize("model", ["unknown-model", "/synthetic/private/model", "TINY"])
def test_cli_rejects_same_invalid_models_as_user_errors(
    model: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same model strings already produce a normal CLI configuration error."""
    with patch("transcription.cli.api_module.transcribe_directory") as transcribe:
        result = main(["transcribe", "--root", str(tmp_path), "--model", model])
    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err.startswith("Error:")
    assert "Invalid model" in captured.err
    assert "Unexpected error" not in captured.err
    transcribe.assert_not_called()


def test_valid_model_profile_mismatch_stays_conflict(
    client: TestClient, audio_bytes: bytes
) -> None:
    """A supported model differing from the service profile is still HTTP 409."""
    response = client.post(
        "/transcribe",
        params={"model": "base"},
        files={"audio": ("synthetic.wav", audio_bytes, "audio/wav")},
    )
    assert response.status_code == 409
    assert response.json()["error"]["type"] == "runtime_profile_mismatch"


def test_sse_unexpected_config_failure_stays_internal(
    client: TestClient, audio_bytes: bytes
) -> None:
    """Only expected configuration failures become bad requests."""
    with patch(
        "transcription.service_transcribe.TranscriptionConfig",
        side_effect=RuntimeError("synthetic-internal-detail"),
    ):
        response = client.post(
            "/transcribe/stream",
            params={"model": "tiny"},
            files={"audio": ("synthetic.wav", audio_bytes, "audio/wav")},
        )
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "internal_error"
    assert "synthetic-internal-detail" not in response.text


def test_valid_sse_config_reaches_stream_and_cleans_up(
    client: TestClient, audio_bytes: bytes
) -> None:
    """Valid configuration still streams and releases its temporary audio directory."""
    paths: list[Path] = []
    configs: list[TranscriptionConfig] = []

    async def events(path, config, include_words):
        paths.append(path)
        configs.append(config)
        assert path.exists()
        yield 'data: {"type":"SESSION_ENDED"}\n\n'

    with (
        patch("transcription.service_transcribe.validate_audio_format"),
        patch(
            "transcription.audio_io.normalize_single",
            side_effect=lambda src, dst: dst.write_bytes(audio_bytes),
        ),
        patch("transcription.service_transcribe._generate_sse_transcription", events),
    ):
        response = client.post(
            "/transcribe/stream",
            params={"model": "tiny", "word_timestamps": True},
            files={"audio": ("synthetic.wav", audio_bytes, "audio/wav")},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "SESSION_ENDED" in response.text
    assert len(paths) == len(configs) == 1
    assert configs[0].model == "tiny"
    assert configs[0].word_timestamps is True
    assert not paths[0].parent.exists()
