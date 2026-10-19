"""Transcript upload rejection at the helper and multipart service boundaries."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from transcription.service import app  # noqa: E402
from transcription.service_validation import validate_transcript_json  # noqa: E402

INVALID_TRANSCRIPTS = [
    pytest.param(b"null", "must be a JSON object", id="null"),
    pytest.param(b"true", "must be a JSON object", id="true"),
    pytest.param(b"false", "must be a JSON object", id="false"),
    pytest.param(b"42", "must be a JSON object", id="integer"),
    pytest.param(b"1.5", "must be a JSON object", id="float"),
    pytest.param(b'"file_name segments"', "must be a JSON object", id="string"),
    pytest.param(b"[]", "must be a JSON object", id="empty-array"),
    pytest.param(b'["file_name", "segments"]', "must be a JSON object", id="field-array"),
    pytest.param(b"\xff", "Invalid transcript JSON", id="invalid-utf8"),
    pytest.param(b"\xfe\xff\x00", "Invalid transcript JSON", id="truncated-utf16"),
    pytest.param(b"\xff\xfe\x00\x00\x00", "Invalid transcript JSON", id="truncated-utf32"),
    pytest.param(b"{not valid json", "parse error at line 1, column 2", id="malformed-json"),
]


@pytest.mark.parametrize(("payload", "message"), INVALID_TRANSCRIPTS)
def test_invalid_transcript_is_bad_request(payload: bytes, message: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_transcript_json(payload)

    assert exc_info.value.status_code == 400
    assert message in exc_info.value.detail


@pytest.mark.parametrize(("payload", "message"), INVALID_TRANSCRIPTS)
def test_enrich_rejects_invalid_transcript_before_processing(payload: bytes, message: str) -> None:
    client = TestClient(app, raise_server_exceptions=False)
    with (
        patch("transcription.service_enrich.save_upload_file_streaming") as save_audio,
        patch("transcription.service_enrich.validate_audio_format") as probe_audio,
        patch("transcription.api.load_transcript") as load_transcript,
        patch("transcription.service_enrich.enrich_transcript") as enrich_transcript,
    ):
        response = client.post(
            "/enrich",
            files={
                "transcript": ("transcript.json", payload, "application/json"),
                "audio": ("test.wav", b"unused audio", "audio/wav"),
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "bad_request"
    assert body["error"]["status_code"] == 400
    assert body["detail"] == body["error"]["message"]
    assert message in body["detail"]
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    save_audio.assert_not_called()
    probe_audio.assert_not_called()
    load_transcript.assert_not_called()
    enrich_transcript.assert_not_called()


def test_decode_error_does_not_echo_uploaded_bytes(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_transcript_json(b'{"private-note": "private-content-\xff"}')

    assert exc_info.value.status_code == 400
    assert "private-content" not in str(exc_info.value.detail)
    assert "private-content" not in caplog.text


def test_valid_unicode_object_is_accepted() -> None:
    payload = json.dumps(
        {
            "file_name": "会議.wav",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "こんにちは 🌍"}],
        },
        ensure_ascii=False,
    ).encode("utf-8")

    validate_transcript_json(payload)
