"""Unit tests for ASR engine resilience and fallbacks."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import slower_whisper.pipeline.asr_engine as asr_engine
from slower_whisper.pipeline.asr_engine import TranscriptionEngine
from slower_whisper.pipeline.config import AsrConfig
from slower_whisper.pipeline.models import Transcript


def test_model_init_retries_on_cpu_when_cuda_load_fails(monkeypatch, caplog):
    """If GPU init fails, the engine should retry on CPU before going dummy."""
    attempts: list[tuple[str, str]] = []

    class FlakyModel:
        def __init__(self, model_name, device, compute_type, download_root):
            attempts.append((device, compute_type))
            if device != "cpu":
                raise RuntimeError("CUDA unavailable")

        def transcribe(self, *args, **kwargs):
            return [], type("info", (), {"language": "en"})()

    monkeypatch.setattr(asr_engine, "_FASTER_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(asr_engine, "WhisperModel", FlakyModel)

    cfg = AsrConfig(model_name="tiny", device="cuda", compute_type="float16")

    with caplog.at_level(logging.WARNING):
        engine = asr_engine.TranscriptionEngine(cfg)

    # GPU attempt then CPU fallback
    assert attempts[0][0] == "cuda"
    assert attempts[-1][0] == "cpu"
    assert isinstance(engine.model, FlakyModel)
    # Config should reflect the fallback device/compute_type
    assert cfg.device == "cpu"
    assert cfg.compute_type == "int8"
    # Warn user about the retry
    assert "Retrying on CPU" in caplog.text


def test_model_init_retries_safer_compute_type_on_cpu(monkeypatch, caplog):
    """CPU loads with aggressive compute_type should retry with int8 before dummy."""
    attempts: list[tuple[str, str]] = []

    class CpuSensitiveModel:
        def __init__(self, model_name, device, compute_type, download_root):
            attempts.append((device, compute_type))
            if compute_type != "int8":
                raise RuntimeError(f"{compute_type}-unsupported")

        def transcribe(self, *args, **kwargs):
            return [], type("info", (), {"language": "en"})()

    monkeypatch.setattr(asr_engine, "_FASTER_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(asr_engine, "WhisperModel", CpuSensitiveModel)

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="float16")

    with caplog.at_level(logging.WARNING):
        engine = asr_engine.TranscriptionEngine(cfg)

    assert attempts == [("cpu", "float16"), ("cpu", "int8")]
    assert isinstance(engine.model, CpuSensitiveModel)
    assert cfg.compute_type == "int8"  # Updated to reflect the successful fallback
    assert engine.model_load_error is None
    assert "compute_type=int8" in caplog.text


def test_model_init_surfaces_load_warnings(monkeypatch, tmp_path):
    """Load retries that succeed should still expose warnings in metadata."""
    attempts: list[tuple[str, str]] = []

    class FallbackModel:
        def transcribe(self, *args, **kwargs):
            seg = SimpleNamespace(start=0.0, end=0.5, text="ok")
            return [seg], SimpleNamespace(language="en")

    def flaky_loader(self, device, compute_type, download_root):
        attempts.append((device, compute_type))
        if device != "cpu":
            raise RuntimeError("cuda unavailable")
        return FallbackModel()

    monkeypatch.setattr(asr_engine, "_FASTER_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(TranscriptionEngine, "_load_whisper_model", flaky_loader)

    cfg = AsrConfig(model_name="tiny", device="cuda", compute_type="float16")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert attempts == [("cuda", "float16"), ("cpu", "int8")]
    assert engine.model_load_error is None
    assert engine.model_load_warnings == ["cuda (float16) load failed: cuda unavailable"]
    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert transcript.meta["asr_device"] == "cpu"
    assert transcript.meta["asr_compute_type"] == "int8"
    assert transcript.meta["asr_model_load_warnings"] == engine.model_load_warnings


def test_model_init_reports_cpu_fallback_failure(monkeypatch, tmp_path):
    """Aggregated error message should include both GPU and CPU load failures."""

    def failing_loader(self, device, compute_type, download_root):
        raise RuntimeError(f"{device}-{compute_type}-boom")

    monkeypatch.setattr(asr_engine, "_FASTER_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(TranscriptionEngine, "_load_whisper_model", failing_loader)

    cfg = AsrConfig(model_name="tiny", device="cuda", compute_type="float16")
    engine = TranscriptionEngine(cfg)

    assert isinstance(engine.model, asr_engine.DummyWhisperModel)
    assert "cuda (float16) load failed" in engine.model_load_error
    assert "cpu (int8) load failed" in engine.model_load_error

    # Transcription metadata should expose the aggregated failure message
    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)
    assert transcript.meta["asr_backend"] == "dummy"
    assert "cpu (int8) load failed" in transcript.meta["asr_fallback_reason"]


def test_transcribe_falls_back_to_dummy_on_inference_error(tmp_path, monkeypatch, caplog):
    """Runtime inference errors should return dummy output with a warning."""

    class BrokenModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("decoder exploded")

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: BrokenModel())

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    with caplog.at_level(logging.WARNING):
        transcript = engine.transcribe_file(wav_path)

    assert transcript.segments[0].text == "dummy segment"
    assert engine.using_dummy is True
    assert transcript.meta["asr_backend"] == "dummy"
    assert "decoder exploded" in transcript.meta["asr_fallback_reason"]
    assert "Falling back to dummy output" in caplog.text


def test_transcribe_meta_records_actual_backend_for_dummy(tmp_path, monkeypatch):
    """Metadata should reflect the actual backend/device used when falling back."""

    class BrokenModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("decoder exploded")

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: BrokenModel())

    cfg = AsrConfig(model_name="tiny", device="cuda", compute_type="float16")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.meta["asr_backend"] == "dummy"
    assert transcript.meta["asr_device"] == "cpu"
    assert transcript.meta["asr_compute_type"] == "n/a"


def test_transcribe_meta_records_actual_backend_for_real_model(tmp_path, monkeypatch):
    """Metadata should capture the real backend details when inference succeeds."""

    class SimpleModel:
        def transcribe(self, *args, **kwargs):
            seg = SimpleNamespace(start=0.0, end=0.5, text="ok")
            return [seg], SimpleNamespace(language="en")

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: SimpleModel())

    cfg = AsrConfig(model_name="tiny", device="cuda", compute_type="float16")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert transcript.meta["asr_device"] == "cuda"
    assert transcript.meta["asr_compute_type"] == "float16"


def test_using_dummy_resets_after_successful_retry(tmp_path, monkeypatch):
    """using_dummy should reflect the last run, not stick after a single failure."""

    class FlakyModel:
        def __init__(self):
            self.calls = 0

        def transcribe(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first call fails")
            return [], type("info", (), {"language": "en"})()

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: FlakyModel())

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    first = engine.transcribe_file(wav_path)
    assert first.meta["asr_backend"] == "dummy"
    assert engine.using_dummy is True

    second = engine.transcribe_file(wav_path)
    assert second.meta["asr_backend"] == "faster-whisper"
    assert engine.using_dummy is False


def test_transcribe_handles_generator_failure(tmp_path, monkeypatch, caplog):
    """Errors raised while iterating segments should trigger dummy fallback."""

    class StreamingModel:
        def transcribe(self, *args, **kwargs):
            def generator():
                yield SimpleNamespace(start=0.0, end=0.5, text="ok")
                raise RuntimeError("midstream failure")

            return generator(), SimpleNamespace(language="es")

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: StreamingModel())

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    with caplog.at_level(logging.WARNING):
        transcript = engine.transcribe_file(wav_path)

    assert transcript.meta["asr_backend"] == "dummy"
    assert "midstream failure" in transcript.meta["asr_fallback_reason"]
    assert transcript.language == "es"
    assert engine.using_dummy is True
    assert "Whisper inference failed" in caplog.text


def test_transcribe_retries_without_vad_kwargs(tmp_path, monkeypatch):
    """Legacy faster-whisper builds without VAD kwargs should succeed after retry."""

    class LegacyModel:
        def __init__(self):
            self.calls = 0

        def transcribe(self, *args, **kwargs):
            self.calls += 1
            if "vad_filter" in kwargs:
                raise TypeError("transcribe() got an unexpected keyword argument 'vad_filter'")
            seg = SimpleNamespace(start=0.0, end=1.0, text="hi")
            return [seg], SimpleNamespace(language="en")

    model = LegacyModel()
    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: model)

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert engine.using_dummy is False
    assert model.calls == 2  # initial attempt with VAD + retry without


def test_transcribe_retains_supported_vad_filter(tmp_path, monkeypatch):
    """If only vad_parameters are rejected, keep using vad_filter on retry."""

    class PartialVadModel:
        def __init__(self):
            self.calls: list[dict] = []

        def transcribe(self, *args, **kwargs):
            self.calls.append(kwargs)
            if "vad_parameters" in kwargs:
                raise TypeError("transcribe() got an unexpected keyword argument 'vad_parameters'")
            seg = SimpleNamespace(start=0.0, end=1.0, text="hi")
            return [seg], SimpleNamespace(language="en")

    model = PartialVadModel()
    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: model)

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert engine.using_dummy is False
    assert len(model.calls) == 2
    assert "vad_filter" in model.calls[-1]
    assert "vad_parameters" not in model.calls[-1]
    assert engine._supports_vad_filter is True
    assert engine._supports_vad_parameters is False


def test_transcribe_drops_all_vad_kwargs_when_both_rejected(tmp_path, monkeypatch):
    """Sequential VAD kwarg errors should fall back to running without VAD instead of dummy."""

    class NoVadSupportModel:
        def __init__(self):
            self.calls: list[dict] = []

        def transcribe(self, *args, **kwargs):
            self.calls.append(kwargs)
            if "vad_filter" in kwargs:
                raise TypeError("transcribe() got an unexpected keyword argument 'vad_filter'")
            if "vad_parameters" in kwargs:
                raise TypeError("transcribe() got an unexpected keyword argument 'vad_parameters'")
            seg = SimpleNamespace(start=0.0, end=1.0, text="hi")
            return [seg], SimpleNamespace(language="en")

    model = NoVadSupportModel()
    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: model)

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert engine.using_dummy is False
    assert len(model.calls) == 3  # initial with both VAD kwargs, then one, then none
    assert "vad_filter" not in model.calls[-1]
    assert "vad_parameters" not in model.calls[-1]
    assert engine._supports_vad_filter is False
    assert engine._supports_vad_parameters is False


def test_transcribe_recovers_from_invalid_segment_timings(tmp_path, monkeypatch):
    """Bad start/end timestamps should trigger dummy fallback instead of corrupt output."""

    class BadSegmentsModel:
        def transcribe(self, *args, **kwargs):
            bad_seg = SimpleNamespace(start=1.0, end=0.5, text="oops")
            return [bad_seg], SimpleNamespace(language="en")

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: BadSegmentsModel())

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.meta["asr_backend"] == "dummy"
    assert "end < start" in transcript.meta["asr_fallback_reason"]
    assert engine.using_dummy is True


def test_transcribe_defaults_language_when_missing(tmp_path, monkeypatch):
    """Missing language info should fall back to configured language or 'unknown'."""

    class NoLanguageModel:
        def transcribe(self, *args, **kwargs):
            seg = SimpleNamespace(start=0.0, end=1.0, text="hello")
            return [seg], SimpleNamespace()  # no language attribute

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: NoLanguageModel())

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8", language="fr")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.language == "fr"
    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert engine.using_dummy is False


def test_transcribe_uses_language_from_mapping_info(tmp_path, monkeypatch):
    """Language should be read from dict-like info objects too."""

    class DictInfoModel:
        def transcribe(self, *args, **kwargs):
            seg = SimpleNamespace(start=0.0, end=1.0, text="hola")
            return [seg], {"language": "es"}

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: DictInfoModel())

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8", language=None)
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.language == "es"
    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert engine.using_dummy is False


def test_transcribe_accepts_language_string_info(tmp_path, monkeypatch):
    """Language string returned directly as info should be respected."""

    class StringInfoModel:
        def transcribe(self, *args, **kwargs):
            seg = SimpleNamespace(start=0.0, end=1.0, text="hola")
            return [seg], "es"

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: StringInfoModel())

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8", language="fr")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.language == "es"
    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert engine.using_dummy is False


def test_transcribe_trims_blank_language(tmp_path, monkeypatch):
    """Whitespace-only language in info should fall back to configured language."""

    class BlankLanguageModel:
        def transcribe(self, *args, **kwargs):
            seg = SimpleNamespace(start=0.0, end=1.0, text="bonjour")
            return [seg], SimpleNamespace(language="   ")

    monkeypatch.setattr(TranscriptionEngine, "_init_model", lambda self: BlankLanguageModel())

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8", language="de")
    engine = TranscriptionEngine(cfg)

    wav_path = Path(tmp_path) / "clip.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, audio, 16000)

    transcript = engine.transcribe_file(wav_path)

    assert transcript.language == "de"
    assert transcript.meta["asr_backend"] == "faster-whisper"
    assert engine.using_dummy is False


def test_transcribe_raises_for_missing_audio(tmp_path, monkeypatch):
    """Engine should fail fast if the audio file path does not exist."""

    def build_dummy(self):
        return asr_engine.DummyWhisperModel(AsrConfig(device="cpu", compute_type="int8"))

    monkeypatch.setattr(TranscriptionEngine, "_init_model", build_dummy)

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    engine = TranscriptionEngine(cfg)

    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError):
        engine.transcribe_file(missing)


def test_transcribe_rejects_directory_path(tmp_path, monkeypatch):
    """Engine should not attempt transcription on a directory path."""

    def build_dummy(self):
        return asr_engine.DummyWhisperModel(AsrConfig(device="cpu", compute_type="int8"))

    monkeypatch.setattr(TranscriptionEngine, "_init_model", build_dummy)

    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    engine = TranscriptionEngine(cfg)

    directory = tmp_path / "audio_dir.wav"
    directory.mkdir()

    with pytest.raises(IsADirectoryError):
        engine.transcribe_file(directory)


def test_pipeline_meta_keeps_backend_details():
    """Internal metadata builder should preserve ASR backend annotations."""
    from slower_whisper.pipeline.config import AppConfig
    from slower_whisper.pipeline.pipeline import _build_meta

    transcript = Transcript(
        file_name="example.wav",
        language="en",
        segments=[],
        meta={"asr_backend": "dummy", "asr_fallback_reason": "missing faster-whisper"},
    )
    meta = _build_meta(AppConfig(), transcript, Path("audio.wav"), 12.3)

    assert meta["asr_backend"] == "dummy"
    assert "missing faster-whisper" in meta["asr_fallback_reason"]
    assert meta["audio_file"] == "example.wav"


def test_pipeline_meta_prefers_actual_asr_runtime():
    """Metadata should record the actual device/compute_type used by ASR."""
    from slower_whisper.pipeline.config import AppConfig
    from slower_whisper.pipeline.pipeline import _build_meta

    transcript = Transcript(
        file_name="example.wav",
        language="en",
        segments=[],
        meta={
            "asr_backend": "dummy",
            "asr_device": "cpu",
            "asr_compute_type": "n/a",
        },
    )
    cfg = AppConfig()
    cfg.asr.device = "cuda"
    cfg.asr.compute_type = "float16"

    meta = _build_meta(cfg, transcript, Path("audio.wav"), 12.3)

    assert meta["device"] == "cpu"
    assert meta["compute_type"] == "n/a"
    assert meta["asr_backend"] == "dummy"
