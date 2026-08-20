"""Unit tests for fail-closed ASR truth and real fallback behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import transcription.asr_engine as asr_engine
from transcription.asr_engine import BackendFactory, TranscriptionEngine
from transcription.config import AsrConfig
from transcription.exceptions import (
    ASRInferenceError,
    ASRModelLoadError,
    ASROutputError,
    ASRUnavailableError,
)
from transcription.models import Transcript
from transcription.writers import write_json, write_srt, write_txt


def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"test audio is not read by injected backends")
    return path


class StaticModel:
    def __init__(
        self,
        segments: Any = None,
        info: Any = None,
    ) -> None:
        self.segments = [] if segments is None else segments
        self.info = SimpleNamespace(language="en") if info is None else info
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, _audio_path: str, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append(dict(kwargs))
        return self.segments, self.info


def static_factory(model: StaticModel) -> BackendFactory:
    def factory(
        _model_name: str,
        _device: str,
        _compute_type: str,
        _download_root: Path,
    ) -> StaticModel:
        return model

    return factory


def config(
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    word_timestamps: bool = False,
) -> AsrConfig:
    return AsrConfig(
        model_name="tiny",
        device=device,
        compute_type=compute_type,
        language=language,
        word_timestamps=word_timestamps,
    )


def segment(
    *,
    start: float = 0.0,
    end: float = 0.5,
    text: str = "real output",
    **extra: Any,
) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text, **extra)


def test_missing_backend_raises_typed_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_error = ImportError("provider detail must remain chained")
    monkeypatch.setattr(asr_engine, "_FASTER_WHISPER_AVAILABLE", False)
    monkeypatch.setattr(asr_engine, "WhisperModel", None)
    monkeypatch.setattr(
        asr_engine,
        "_FASTER_WHISPER_IMPORT_ERROR",
        provider_error,
    )

    with pytest.raises(ASRUnavailableError) as raised:
        TranscriptionEngine(config())

    assert raised.value.reason_code == "asr_backend_unavailable"
    assert raised.value.context == {"backend": "faster-whisper"}
    assert raised.value.__cause__ is provider_error
    assert "provider detail" not in str(raised.value.public_details())


def test_cuda_load_failure_retries_real_cpu_backend(
    tmp_path: Path,
) -> None:
    attempts: list[tuple[str, str]] = []

    def factory(
        _model_name: str,
        device: str,
        compute_type: str,
        _download_root: Path,
    ) -> StaticModel:
        attempts.append((device, compute_type))
        if device != "cpu":
            raise RuntimeError("local CUDA diagnostic")
        return StaticModel([segment(text="cpu result")])

    cfg = config(device="cuda", compute_type="float16")
    engine = TranscriptionEngine(cfg, backend_factory=factory)
    transcript = engine.transcribe_file(audio_file(tmp_path))

    assert attempts == [("cuda", "float16"), ("cpu", "int8")]
    assert cfg.device == "cpu"
    assert cfg.compute_type == "int8"
    assert transcript.full_text == "cpu result"
    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert transcript.meta["asr_device"] == "cpu"
    assert transcript.meta["asr_compute_type"] == "int8"
    assert transcript.meta["asr_model_load_attempts"] == [
        {
            "device": "cuda",
            "compute_type": "float16",
            "outcome": "failed",
            "reason_code": "asr_model_load_failed",
        },
        {
            "device": "cpu",
            "compute_type": "int8",
            "outcome": "selected",
            "reason_code": "ok",
        },
    ]
    assert transcript.meta["asr_model_load_warnings"] == [
        "cuda (float16) load failed"
    ]


def test_cpu_compute_fallback_is_ordered() -> None:
    attempts: list[tuple[str, str]] = []

    def factory(
        _model_name: str,
        device: str,
        compute_type: str,
        _download_root: Path,
    ) -> StaticModel:
        attempts.append((device, compute_type))
        if compute_type != "int8":
            raise RuntimeError("unsupported compute")
        return StaticModel()

    cfg = config(device="cpu", compute_type="float16")
    engine = TranscriptionEngine(cfg, backend_factory=factory)

    assert attempts == [("cpu", "float16"), ("cpu", "int8")]
    assert cfg.compute_type == "int8"
    assert engine.model_load_attempts[-1]["outcome"] == "selected"


def test_all_real_load_attempts_raise_typed_model_error() -> None:
    def factory(
        _model_name: str,
        device: str,
        compute_type: str,
        _download_root: Path,
    ) -> StaticModel:
        raise RuntimeError(f"sensitive provider detail for {device}/{compute_type}")

    with pytest.raises(ASRModelLoadError) as raised:
        TranscriptionEngine(
            config(device="cuda", compute_type="float16"),
            backend_factory=factory,
        )

    error = raised.value
    assert error.reason_code == "asr_model_load_failed"
    assert error.context["backend"] == "faster-whisper"
    assert error.context["model"] == "tiny"
    assert [item["outcome"] for item in error.context["attempts"]] == [
        "failed",
        "failed",
    ]
    assert "sensitive provider detail" not in str(error.public_details())
    assert isinstance(error.__cause__, RuntimeError)


def test_direct_inference_failure_is_not_a_transcript(
    tmp_path: Path,
) -> None:
    class BrokenModel:
        def transcribe(self, _audio_path: str, **_kwargs: Any) -> Any:
            raise RuntimeError("decoder exploded")

    with pytest.raises(ASRInferenceError) as raised:
        TranscriptionEngine(
            config(),
            backend_factory=lambda *_args: BrokenModel(),
        ).transcribe_file(audio_file(tmp_path))

    assert raised.value.reason_code == "asr_inference_failed"
    assert raised.value.context["phase"] == "transcribe"
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_lazy_segment_failure_is_typed_inference_error(
    tmp_path: Path,
) -> None:
    def failing_segments():
        yield segment(text="partial provider output")
        raise RuntimeError("iterator failed")

    model = StaticModel(failing_segments())

    with pytest.raises(ASRInferenceError) as raised:
        TranscriptionEngine(
            config(),
            backend_factory=static_factory(model),
        ).transcribe_file(audio_file(tmp_path))

    assert raised.value.context == {"phase": "segment_iteration"}
    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("result", "violation"),
    [
        (None, "result_shape"),
        (([],), "result_shape"),
        ((None, SimpleNamespace(language="en")), "segments_not_iterable"),
        (("not segments", SimpleNamespace(language="en")), "segment_collection_type"),
    ],
)
def test_invalid_result_shape_is_typed_output_error(
    tmp_path: Path,
    result: Any,
    violation: str,
) -> None:
    class InvalidResultModel:
        def transcribe(self, _audio_path: str, **_kwargs: Any) -> Any:
            return result

    with pytest.raises(ASROutputError) as raised:
        TranscriptionEngine(
            config(),
            backend_factory=lambda *_args: InvalidResultModel(),
        ).transcribe_file(audio_file(tmp_path))

    assert raised.value.reason_code == "asr_output_invalid"
    assert raised.value.context["violation"] == violation


@pytest.mark.parametrize(
    ("bad_segment", "violation"),
    [
        (SimpleNamespace(end=1.0, text="missing start"), "required_shape"),
        (segment(start=float("nan")), "non_finite_timing"),
        (segment(start=-0.1), "negative_timing"),
        (segment(start=1.0, end=0.5), "end_before_start"),
        (SimpleNamespace(start=0.0, end=1.0), "missing_text"),
        (SimpleNamespace(start=0.0, end=1.0, text=123), "text_not_string"),
    ],
)
def test_malformed_segment_is_typed_output_error(
    tmp_path: Path,
    bad_segment: SimpleNamespace,
    violation: str,
) -> None:
    model = StaticModel([bad_segment])
    with pytest.raises(ASROutputError) as raised:
        TranscriptionEngine(
            config(),
            backend_factory=static_factory(model),
        ).transcribe_file(audio_file(tmp_path))

    assert raised.value.context == {
        "segment_index": 0,
        "violation": violation,
    }


def test_real_zero_segment_result_is_successful_empty_transcript(
    tmp_path: Path,
) -> None:
    transcript = TranscriptionEngine(
        config(language="en"),
        backend_factory=static_factory(
            StaticModel([], SimpleNamespace(language="en", duration_after_vad=0.0))
        ),
    ).transcribe_file(audio_file(tmp_path))

    assert transcript.segments == []
    assert transcript.full_text == ""
    assert transcript.language == "en"
    assert transcript.duration_after_vad == 0.0
    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert "asr_fallback_reason" not in transcript.meta
    assert "asr_placeholder_segments" not in transcript.meta


def test_empty_transcript_writers_emit_no_fabricated_content(
    tmp_path: Path,
) -> None:
    transcript = Transcript(
        file_name="silence.wav",
        language="en",
        segments=[],
        meta={"asr_backend": "faster-whisper"},
    )
    json_path = tmp_path / "silence.json"
    txt_path = tmp_path / "silence.txt"
    srt_path = tmp_path / "silence.srt"

    write_json(transcript, json_path)
    write_txt(transcript, txt_path)
    write_srt(transcript, srt_path)

    assert '"segments": []' in json_path.read_text(encoding="utf-8")
    assert txt_path.read_text(encoding="utf-8") == (
        "# File: silence.wav\n# Language: en\n\n"
    )
    assert srt_path.read_text(encoding="utf-8") == ""


def test_legacy_vad_kwargs_are_removed_without_hiding_real_output(
    tmp_path: Path,
) -> None:
    class LegacyModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def transcribe(
            self,
            _audio_path: str,
            **kwargs: Any,
        ) -> tuple[list[SimpleNamespace], SimpleNamespace]:
            self.calls.append(dict(kwargs))
            if "vad_filter" in kwargs:
                raise TypeError(
                    "transcribe() got an unexpected keyword argument 'vad_filter'"
                )
            return [segment(text="legacy result")], SimpleNamespace(language="en")

    model = LegacyModel()
    transcript = TranscriptionEngine(
        config(),
        backend_factory=lambda *_args: model,
    ).transcribe_file(audio_file(tmp_path))

    assert transcript.full_text == "legacy result"
    assert len(model.calls) == 2
    assert "vad_filter" not in model.calls[-1]
    assert "vad_parameters" in model.calls[-1]


def test_partial_vad_support_is_retained(
    tmp_path: Path,
) -> None:
    class PartialVadModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def transcribe(
            self,
            _audio_path: str,
            **kwargs: Any,
        ) -> tuple[list[SimpleNamespace], SimpleNamespace]:
            self.calls.append(dict(kwargs))
            if "vad_parameters" in kwargs:
                raise TypeError(
                    "transcribe() got an unexpected keyword argument "
                    "'vad_parameters'"
                )
            return [segment(text="partial VAD")], SimpleNamespace(language="en")

    model = PartialVadModel()
    engine = TranscriptionEngine(
        config(),
        backend_factory=lambda *_args: model,
    )
    transcript = engine.transcribe_file(audio_file(tmp_path))

    assert transcript.full_text == "partial VAD"
    assert len(model.calls) == 2
    assert "vad_filter" in model.calls[-1]
    assert "vad_parameters" not in model.calls[-1]
    assert engine._supports_vad_filter is True
    assert engine._supports_vad_parameters is False


@pytest.mark.parametrize(
    ("info", "configured_language", "expected"),
    [
        (SimpleNamespace(language="es"), "fr", "es"),
        ({"language": "de"}, "fr", "de"),
        ("it", "fr", "it"),
        (SimpleNamespace(language="   "), "fr", "fr"),
        (SimpleNamespace(), None, "unknown"),
    ],
)
def test_language_normalization(
    tmp_path: Path,
    info: Any,
    configured_language: str | None,
    expected: str,
) -> None:
    transcript = TranscriptionEngine(
        config(language=configured_language),
        backend_factory=static_factory(
            StaticModel([segment(text="language")], info)
        ),
    ).transcribe_file(audio_file(tmp_path))

    assert transcript.language == expected


def test_segments_are_sorted_and_reidentified(
    tmp_path: Path,
) -> None:
    transcript = TranscriptionEngine(
        config(),
        backend_factory=static_factory(
            StaticModel(
                [
                    segment(start=2.0, end=3.0, text="second"),
                    segment(start=0.0, end=1.0, text="first"),
                ]
            )
        ),
    ).transcribe_file(audio_file(tmp_path))

    assert [(item.id, item.text) for item in transcript.segments] == [
        (0, "first"),
        (1, "second"),
    ]


def test_word_timestamps_are_preserved(
    tmp_path: Path,
) -> None:
    words = [
        SimpleNamespace(
            word=" hello",
            start=0.0,
            end=0.4,
            probability=0.9,
        )
    ]
    transcript = TranscriptionEngine(
        config(word_timestamps=True),
        backend_factory=static_factory(
            StaticModel([segment(text="hello", words=words)])
        ),
    ).transcribe_file(audio_file(tmp_path))

    assert transcript.segments[0].words is not None
    assert transcript.segments[0].words[0].word == " hello"


def test_missing_file_and_directory_fail_before_backend_dispatch(
    tmp_path: Path,
) -> None:
    model = StaticModel([segment()])
    engine = TranscriptionEngine(
        config(),
        backend_factory=static_factory(model),
    )

    with pytest.raises(FileNotFoundError):
        engine.transcribe_file(tmp_path / "missing.wav")

    directory = tmp_path / "directory.wav"
    directory.mkdir()
    with pytest.raises(IsADirectoryError):
        engine.transcribe_file(directory)

    assert model.calls == []


def test_transcribe_many_propagates_typed_failure(
    tmp_path: Path,
) -> None:
    first = audio_file(tmp_path)
    second = tmp_path / "second.wav"
    second.write_bytes(b"also not read")

    class FailsSecond:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(
            self,
            _audio_path: str,
            **_kwargs: Any,
        ) -> tuple[list[SimpleNamespace], SimpleNamespace]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second inference failed")
            return [segment(text="first")], SimpleNamespace(language="en")

    engine = TranscriptionEngine(
        config(),
        backend_factory=lambda *_args: FailsSecond(),
    )
    iterator = iter(engine.transcribe_many([first, second]))

    assert next(iterator).full_text == "first"
    with pytest.raises(ASRInferenceError):
        next(iterator)
