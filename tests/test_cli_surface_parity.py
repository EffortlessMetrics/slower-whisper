"""Argparse and wheel-installed CLI parity with the canonical file surface."""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import subprocess
import wave
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft7Validator, FormatChecker

from transcription import _build_info
from transcription.api import transcribe_file
from transcription.cli import main
from transcription.config import AsrConfig, Paths, TranscriptionConfig
from transcription.device import ResolvedDevice
from transcription.models import Segment, Transcript, Word
from transcription.receipt import receipt_stable_projection


class ParityEngine:
    """Deterministic fallback engine shared by direct and CLI surfaces."""

    instances = 0
    closes = 0
    calls = 0
    word_timestamp_values: list[bool] = []

    def __init__(self, cfg: AsrConfig) -> None:
        type(self).instances += 1
        type(self).word_timestamp_values.append(bool(cfg.word_timestamps))
        self.cfg = cfg
        self.model_load_attempts = [
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
        self.cfg.device = "cpu"
        self.cfg.compute_type = "int8"

    def transcribe_file(self, audio_path: Path) -> Transcript:
        type(self).calls += 1
        return Transcript(
            file_name=audio_path.name,
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=0.01,
                    text="hello",
                    words=[
                        Word(
                            word="hello",
                            start=0.0,
                            end=0.01,
                            probability=0.99,
                        ),
                    ],
                ),
            ],
            meta={
                "asr_backend": "faster-whisper",
                "asr_model": self.cfg.model_name,
                "asr_device": self.cfg.device,
                "asr_compute_type": self.cfg.compute_type,
                "asr_model_load_attempts": list(self.model_load_attempts),
            },
        )

    def close(self) -> None:
        type(self).closes += 1


def reset_engine() -> None:
    ParityEngine.instances = 0
    ParityEngine.closes = 0
    ParityEngine.calls = 0
    ParityEngine.word_timestamp_values = []


def config() -> TranscriptionConfig:
    return TranscriptionConfig(
        model="tiny",
        device="cuda",
        compute_type="float16",
        language="en",
        task="transcribe",
        beam_size=3,
        vad_min_silence_ms=400,
        word_timestamps=True,
        skip_existing_json=False,
        enable_chunking=False,
        enable_diarization=False,
    )


def asr_config() -> AsrConfig:
    return AsrConfig(
        model_name="tiny",
        device="cuda",
        compute_type="float16",
        language="en",
        task="transcribe",
        beam_size=3,
        vad_min_silence_ms=400,
        word_timestamps=True,
    )


def resolved_cuda() -> ResolvedDevice:
    return ResolvedDevice(
        device="cuda",
        compute_type="float16",
        requested_device="cuda",
        cuda_available=True,
        cuda_device_count=1,
    )


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(struct.pack("h" * 160, *[0] * 160))
    return buffer.getvalue()


def normalize_all(paths: Paths) -> None:
    paths.norm_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in sorted(paths.raw_dir.iterdir()):
        if raw_path.is_file():
            (paths.norm_dir / f"{raw_path.stem}.wav").write_bytes(wav_bytes())


def prepare_project(root: Path, source_name: str = "surface.mp3") -> Path:
    paths = Paths(root=root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    source = paths.raw_dir / source_name
    source.write_bytes(wav_bytes())
    return source


def cli_arguments(root: Path) -> list[str]:
    return [
        "transcribe",
        "--root",
        str(root),
        "--model",
        "tiny",
        "--device",
        "cuda",
        "--compute-type",
        "float16",
        "--language",
        "en",
        "--task",
        "transcribe",
        "--beam-size",
        "3",
        "--vad-min-silence-ms",
        "400",
        "--word-timestamps",
        "--no-skip-existing-json",
        "--no-enable-chunking",
        "--no-enable-diarization",
    ]


def load_document(root: Path, stem: str = "surface") -> dict[str, Any]:
    path = Paths(root=root).json_dir / f"{stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_schema(name: str) -> dict[str, Any]:
    resource = resources.files("transcription").joinpath("schemas", name)
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_document(document: dict[str, Any]) -> None:
    assert not list(
        Draft7Validator(
            load_schema("transcript-v2.schema.json"),
            format_checker=FormatChecker(),
        ).iter_errors(document)
    )
    assert not list(
        Draft7Validator(
            load_schema("receipt-v1.schema.json"),
            format_checker=FormatChecker(),
        ).iter_errors(document["meta"]["receipt"])
    )


def semantic_projection(document: dict[str, Any]) -> dict[str, Any]:
    meta = document["meta"]
    return {
        "schema_version": document["schema_version"],
        "file": document["file"],
        "audio_file": meta["audio_file"],
        "language": document["language"],
        "segments": document["segments"],
        "runtime": {
            "backend": meta["asr_backend"],
            "model": meta["model_name"],
            "device": meta["device"],
            "compute_type": meta["compute_type"],
            "attempts": meta["asr_model_load_attempts"],
        },
        "receipt": receipt_stable_projection(meta["receipt"]),
    }


def clear_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("SLOWER_WHISPER_"):
            monkeypatch.delenv(key, raising=False)


def install_source_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_config_environment(monkeypatch)
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "cli-parity-1")
    monkeypatch.setattr("transcription.audio_io.normalize_all", normalize_all)
    monkeypatch.setattr("transcription.pipeline.TranscriptionEngine", ParityEngine)
    monkeypatch.setattr(
        "transcription.cli.resolve_device",
        lambda *_args, **_kwargs: resolved_cuda(),
    )
    monkeypatch.setattr(
        "transcription.api._get_wav_duration_seconds",
        lambda _path: 0.01,
    )
    monkeypatch.setattr(
        "transcription.api._maybe_run_diarization",
        lambda transcript, _path, _config: transcript,
    )
    monkeypatch.setattr(
        "transcription.api._maybe_build_chunks",
        lambda transcript, _config: transcript,
    )


def canonical_file_document(
    root: Path,
    source_name: str = "surface.mp3",
) -> dict[str, Any]:
    source = root / source_name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(wav_bytes())
    transcribe_file(
        source,
        root,
        config(),
        _engine=ParityEngine(asr_config()),
    )
    return load_document(root)


def assert_cli_document_matches(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    validate_document(baseline)
    validate_document(candidate)
    assert semantic_projection(candidate) == semantic_projection(baseline)
    assert candidate["file"] == "surface.mp3"
    assert candidate["meta"]["audio_file"] == "surface.mp3"


def test_argparse_transcribe_matches_canonical_file_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_engine()
    install_source_patches(monkeypatch)

    baseline_root = tmp_path / "baseline"
    baseline = canonical_file_document(baseline_root)

    cli_root = tmp_path / "cli"
    prepare_project(cli_root)
    exit_code = main(cli_arguments(cli_root))
    candidate = load_document(cli_root)

    assert exit_code == 0
    assert_cli_document_matches(baseline, candidate)
    assert ParityEngine.instances == 2
    assert ParityEngine.calls == 2
    assert ParityEngine.closes == 1
    assert ParityEngine.word_timestamp_values == [True, True]


def write_startup_injection(
    root: Path,
    lifecycle_path: Path,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sitecustomize.py").write_text(
        "from parity_injection import install\ninstall()\n",
        encoding="utf-8",
    )
    (root / "parity_injection.py").write_text(
        f"""from __future__ import annotations

import atexit
import io
import json
import struct
import wave
from pathlib import Path

LIFECYCLE_PATH = Path({str(lifecycle_path)!r})
STATE = {{
    "instances": 0,
    "calls": 0,
    "closes": 0,
    "word_timestamp_values": [],
}}


class Engine:
    def __init__(self, cfg):
        STATE["instances"] += 1
        STATE["word_timestamp_values"].append(bool(cfg.word_timestamps))
        self.cfg = cfg
        self.model_load_attempts = [
            {{
                "device": "cuda",
                "compute_type": "float16",
                "outcome": "failed",
                "reason_code": "asr_model_load_failed",
            }},
            {{
                "device": "cpu",
                "compute_type": "int8",
                "outcome": "selected",
                "reason_code": "ok",
            }},
        ]
        self.cfg.device = "cpu"
        self.cfg.compute_type = "int8"

    def transcribe_file(self, audio_path):
        STATE["calls"] += 1
        from transcription.models import Segment, Transcript, Word

        return Transcript(
            file_name=Path(audio_path).name,
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=0.01,
                    text="hello",
                    words=[
                        Word(
                            word="hello",
                            start=0.0,
                            end=0.01,
                            probability=0.99,
                        )
                    ],
                )
            ],
            meta={{
                "asr_backend": "faster-whisper",
                "asr_model": self.cfg.model_name,
                "asr_device": self.cfg.device,
                "asr_compute_type": self.cfg.compute_type,
                "asr_model_load_attempts": list(self.model_load_attempts),
            }},
        )

    def close(self):
        STATE["closes"] += 1


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(struct.pack("h" * 160, *([0] * 160)))
    return buffer.getvalue()


def normalize_all(paths):
    paths.norm_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in sorted(paths.raw_dir.iterdir()):
        if raw_path.is_file():
            (paths.norm_dir / f"{{raw_path.stem}}.wav").write_bytes(wav_bytes())


def install():
    from transcription import _build_info
    import transcription.audio_io as audio_io
    import transcription.cli as cli
    import transcription.pipeline as pipeline
    from transcription.device import ResolvedDevice

    _build_info.SOURCE_COMMIT = "abcdef123456"
    _build_info.BUILD_ID = "cli-parity-1"
    audio_io.normalize_all = normalize_all
    pipeline.TranscriptionEngine = Engine
    cli.resolve_device = lambda *_args, **_kwargs: ResolvedDevice(
        device="cuda",
        compute_type="float16",
        requested_device="cuda",
        cuda_available=True,
        cuda_device_count=1,
    )


@atexit.register
def write_lifecycle():
    LIFECYCLE_PATH.write_text(json.dumps(STATE), encoding="utf-8")
""",
        encoding="utf-8",
    )
    return root


@pytest.mark.skipif(
    os.getenv("SLOWER_WHISPER_INSTALLED_CLI_PARITY") != "1",
    reason="installed console transaction runs only in the wheel acceptance lane",
)
def test_wheel_installed_console_matches_canonical_file_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_engine()
    install_source_patches(monkeypatch)

    baseline_root = tmp_path / "baseline"
    baseline = canonical_file_document(baseline_root)

    executable = shutil.which("slower-whisper")
    assert executable is not None

    cli_root = tmp_path / "installed-cli"
    prepare_project(cli_root)
    lifecycle_path = tmp_path / "lifecycle.json"
    injection = write_startup_injection(tmp_path / "injection", lifecycle_path)
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()

    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("SLOWER_WHISPER_")
    }
    environment["PYTHONPATH"] = str(injection)

    process = subprocess.run(
        [executable, *cli_arguments(cli_root)],
        cwd=unrelated,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    candidate = load_document(cli_root)
    assert_cli_document_matches(baseline, candidate)
    assert json.loads(lifecycle_path.read_text(encoding="utf-8")) == {
        "instances": 1,
        "calls": 1,
        "closes": 1,
        "word_timestamp_values": [True],
    }
