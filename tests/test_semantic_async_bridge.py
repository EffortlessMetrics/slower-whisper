"""Regression coverage for the synchronous semantic provider coroutine bridge."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from typing import Any

import pytest

from transcription.semantic_adapter import _run_async_safely


class ProviderRuntimeError(RuntimeError):
    """A provider-specific failure must not be confused with loop discovery."""


def _call[T](operation: Callable[[], T], running_loop: bool) -> T:
    if not running_loop:
        return operation()

    async def invoke() -> T:
        return operation()

    return asyncio.run(invoke())


@pytest.mark.parametrize("running_loop", [False, True])
def test_result_identity_single_execution_and_loop_isolation(running_loop: bool) -> None:
    result = object()
    caller_thread = threading.get_ident()
    executions: list[tuple[int, asyncio.AbstractEventLoop]] = []

    async def operation() -> object:
        executions.append((threading.get_ident(), asyncio.get_running_loop()))
        await asyncio.sleep(0)
        return result

    coroutine = operation()
    assert _call(lambda: _run_async_safely(coroutine), running_loop) is result
    assert len(executions) == 1
    thread, loop = executions[0]
    assert (thread != caller_thread) is running_loop
    assert loop.is_closed()
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


@pytest.mark.parametrize("running_loop", [False, True])
@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, ProviderRuntimeError, ValueError, TimeoutError, asyncio.CancelledError],
)
def test_original_exception_cause_and_cleanup_survive(
    running_loop: bool, error_type: type[BaseException]
) -> None:
    cause = LookupError("upstream cause")
    error = error_type("provider failure")
    events: list[str] = []

    async def operation() -> None:
        events.append("started")
        try:
            await asyncio.sleep(0)
            raise error from cause
        finally:
            events.append("cleaned up")

    coroutine = operation()
    with pytest.raises(error_type) as caught:
        _call(lambda: _run_async_safely(coroutine), running_loop)
    assert caught.value is error
    assert caught.value.__cause__ is cause
    assert events == ["started", "cleaned up"]
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


@pytest.mark.parametrize("running_loop", [False, True])
def test_successful_call_after_failure_does_not_reuse_coroutine(running_loop: bool) -> None:
    error = ProviderRuntimeError("first operation fails")
    events: list[str] = []

    async def fail() -> None:
        events.append("failure")
        raise error

    async def succeed() -> str:
        events.append("success")
        return "recovered"

    def invoke() -> str:
        with pytest.raises(ProviderRuntimeError) as caught:
            _run_async_safely(fail())
        assert caught.value is error
        return _run_async_safely(succeed())

    assert _call(invoke, running_loop) == "recovered"
    assert events == ["failure", "success"]


@pytest.mark.parametrize("running_loop", [False, True])
def test_runtime_error_during_coroutine_cleanup_is_not_masked(running_loop: bool) -> None:
    error = RuntimeError("cleanup failure")

    async def operation() -> str:
        try:
            return "result that must not escape"
        finally:
            raise error

    with pytest.raises(RuntimeError) as caught:
        _call(lambda: _run_async_safely(operation()), running_loop)
    assert caught.value is error


def test_caller_loop_remains_usable_after_worker_failure() -> None:
    error = ProviderRuntimeError("worker failure")

    async def fail() -> None:
        await asyncio.sleep(0)
        raise error

    async def invoke() -> None:
        caller_loop = asyncio.get_running_loop()
        with pytest.raises(ProviderRuntimeError) as caught:
            _run_async_safely(fail())
        assert caught.value is error
        assert asyncio.get_running_loop() is caller_loop
        assert not caller_loop.is_closed()

        await asyncio.sleep(0)
        assert asyncio.get_running_loop() is caller_loop

    asyncio.run(invoke())


@pytest.mark.parametrize("running_loop", [False, True])
def test_worker_submission_error_is_not_treated_as_loop_discovery(
    running_loop: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import concurrent.futures

    error = RuntimeError("executor is unavailable")
    executions: list[str] = []

    def fail_submission(*args: Any, **kwargs: Any) -> Any:
        raise error

    async def operation() -> str:
        executions.append("called")
        return "done"

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "submit", fail_submission)
    coroutine = operation()
    if running_loop:
        with pytest.raises(RuntimeError) as caught:
            _call(lambda: _run_async_safely(coroutine), running_loop)
        assert caught.value is error
        assert executions == []
    else:
        assert _call(lambda: _run_async_safely(coroutine), running_loop) == "done"
        assert executions == ["called"]

    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
