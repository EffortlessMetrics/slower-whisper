"""Privacy and shape contract for structured service errors."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi import Request
from fastapi.exceptions import RequestValidationError

from transcription.service_errors import validation_exception_handler


@pytest.mark.asyncio
async def test_validation_error_log_excludes_submitted_values_and_messages() -> None:
    secret = "submitted-secret\r\nforged-log-line"
    error_message = f"Invalid value: {secret}"
    validation_error = RequestValidationError(
        [
            {
                "type": "string_type",
                "loc": ("body", "credential\r\nfield"),
                "msg": error_message,
                "input": secret,
            }
        ],
        body={"credential": secret},
    )
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/transcribe",
            "raw_path": b"/transcribe",
            "query_string": b"",
            "headers": [],
            "client": ("test", 1234),
            "server": ("test", 443),
        }
    )
    request.state.request_id = "request-1"

    with patch("transcription.service_errors.logger.warning") as warning:
        response = await validation_exception_handler(request, validation_error)

    logged = warning.call_args.kwargs["extra"]
    assert logged == {
        "request_id": "request-1",
        "validation_errors": [
            {
                "loc": ["body", "credential\\r\\nfield"],
                "type": "string_type",
            }
        ],
    }
    assert secret not in repr(logged)
    assert error_message not in repr(logged)

    payload = json.loads(response.body)
    assert response.status_code == 422
    assert payload["error"]["details"]["validation_errors"] == [
        {
            "loc": ["body", "credential\r\nfield"],
            "msg": error_message,
            "type": "string_type",
        }
    ]
