"""The sync/async bridge must preserve results and original coroutine failures."""

from __future__ import annotations

import asyncio
import threading

import pytest

from transcription.semantic_adapter import _run_async_safely


def call_in_context(callback, running_loop: bool):
    if not running_loop:
        return callback()

    async def invoke():
        return callback()

    return asyncio.run(invoke())


@pytest.mark.parametrize("running_loop", (False, True))
def test_result_identity_and_one_execution(running_loop):
    result = object()
    calls = []
    parent_thread = threading.get_ident()

    async def operation():
        calls.append(threading.get_ident())
        await asyncio.sleep(0)
        return result

    actual = call_in_context(lambda: _run_async_safely(operation()), running_loop)
    assert actual is result
    assert len(calls) == 1
    assert (calls[0] != parent_thread) is running_loop


@pytest.mark.parametrize("running_loop", (False, True))
@pytest.mark.parametrize("error_type", (RuntimeError, ValueError, asyncio.CancelledError))
def test_original_failure_identity_and_cause_are_preserved(running_loop, error_type):
    cause = LookupError("fixture cause")
    error = error_type("fixture provider failure")
    calls = []

    async def operation():
        calls.append("called")
        await asyncio.sleep(0)
        raise error from cause

    with pytest.raises(error_type) as caught:
        call_in_context(lambda: _run_async_safely(operation()), running_loop)
    assert caught.value is error
    assert caught.value.__cause__ is cause
    assert calls == ["called"]


@pytest.mark.parametrize("running_loop", (False, True))
def test_runtime_error_subclass_is_not_a_loop_discovery_failure(running_loop):
    class ProviderFailure(RuntimeError):
        pass

    error = ProviderFailure("fixture failure")

    async def operation():
        raise error

    with pytest.raises(ProviderFailure) as caught:
        call_in_context(lambda: _run_async_safely(operation()), running_loop)
    assert caught.value is error
