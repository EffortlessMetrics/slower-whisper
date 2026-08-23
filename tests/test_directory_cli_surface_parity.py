"""End-to-end directory and console parity against the canonical file surface."""

from __future__ import annotations

import importlib
import inspect
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tomllib
import wave
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner
from jsonschema import Draft7Validator, FormatChecker

from transcription import _build_info
from transcription.api import transcribe_file
from transcription.config import AsrConfig, Paths, TranscriptionConfig
from transcription.models import Segment, Transcript, Word
from transcription.pipeline import run_pipeline
from transcription.receipt import receipt_stable_projection


class ParityEngine:
    """Deterministic operation engine with one explicit fallback."""

    instances = 0
    closes = 0

    def __init__(self, cfg: AsrConfig) -> None:
        type(self).instances += 1
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
                        )
                    ],
                )
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


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack("h" * 160, *[0] * 160))
    return buffer.getvalue()


def normalize_all(paths: Paths) -> None:
    paths.norm_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in sorted(paths.raw_dir.iterdir()):
        (paths.norm_dir / f"{raw_path.stem}.wav").write_bytes(wav_bytes())


def normalize_single(_input: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(wav_bytes())


def patch_operation_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("transcription.asr_engine.TranscriptionEngine", ParityEngine)
    monkeypatch.setattr("transcription.audio_io.normalize_all", normalize_all)
    monkeypatch.setattr("transcription.audio_io.normalize_single", normalize_single)
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

    for module_name in (
        "transcription.pipeline",
        "transcription.cli",
        "transcription.cli_app",
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, "TranscriptionEngine"):
            monkeypatch.setattr(module, "TranscriptionEngine", ParityEngine)
        if hasattr(module, "normalize_all"):
            monkeypatch.setattr(module, "normalize_all", normalize_all)
        if hasattr(module, "normalize_single"):
            monkeypatch.setattr(module, "normalize_single", normalize_single)


def prepare_project(root: Path, source_name: str = "surface.mp3") -> tuple[Paths, Path]:
    paths = Paths(root=root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    source = paths.raw_dir / source_name
    source.write_bytes(wav_bytes())
    return paths, source


def json_output(root: Path, stem: str = "surface") -> dict[str, Any]:
    matches = sorted(root.rglob(f"{stem}.json"))
    assert len(matches) == 1, matches
    return json.loads(matches[0].read_text(encoding="utf-8"))


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


def call_run_pipeline(root: Path, transcription_config: TranscriptionConfig) -> Any:
    signature = inspect.signature(run_pipeline)
    values: dict[str, Any] = {
        "root": root,
        "project_root": root,
        "paths": Paths(root=root),
        "config": transcription_config,
        "transcription_config": transcription_config,
    }
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.name in values:
            value = values[parameter.name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise AssertionError(
                f"Unsupported required run_pipeline parameter: {parameter.name}"
            )
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[parameter.name] = value
    return run_pipeline(*args, **kwargs)


def console_entrypoint() -> tuple[Any, Any]:
    points = [
        point
        for point in metadata.entry_points(group="console_scripts")
        if point.name == "slower-whisper"
    ]
    assert len(points) == 1, points
    point = points[0]
    target = point.load()
    module = importlib.import_module(point.module)

    if isinstance(target, click.Command):
        return target, module

    try:
        import typer
    except ImportError:  # pragma: no cover - Typer is a declared CLI dependency
        typer = None
    if typer is not None:
        for value in vars(module).values():
            if isinstance(value, typer.Typer):
                return typer.main.get_command(value), module

    for value in vars(module).values():
        if isinstance(value, click.Command):
            return value, module
    raise AssertionError(f"Could not resolve Click/Typer command from {point.value}")


def option_token(parameter: click.Option) -> str:
    long_options = [option for option in parameter.opts if option.startswith("--")]
    assert long_options or parameter.opts, parameter.name
    return (long_options or list(parameter.opts))[0]


def parameter_value(name: str, *, root: Path, source: Path) -> str:
    normalized = name.lower()
    if any(word in normalized for word in ("root", "project", "workspace", "directory")):
        return str(root)
    if any(word in normalized for word in ("input", "audio", "file", "source")):
        return str(source)
    if "output" in normalized:
        return str(root)
    deterministic = {
        "model": "tiny",
        "device": "cuda",
        "compute_type": "float16",
        "language": "en",
        "task": "transcribe",
    }
    if normalized in deterministic:
        return deterministic[normalized]
    raise AssertionError(f"Unsupported required CLI parameter: {name}")


def append_command_parameters(
    arguments: list[str],
    command: click.Command,
    *,
    root: Path,
    source: Path,
) -> None:
    deterministic_options = {
        "model": "tiny",
        "device": "cuda",
        "compute_type": "float16",
        "language": "en",
        "task": "transcribe",
    }
    for parameter in command.params:
        if isinstance(parameter, click.Argument):
            if parameter.required:
                arguments.append(parameter_value(parameter.name, root=root, source=source))
            continue
        if not isinstance(parameter, click.Option):
            continue
        name = parameter.name or ""
        if name in deterministic_options:
            arguments.extend([option_token(parameter), deterministic_options[name]])
        elif name == "word_timestamps" and parameter.is_flag and not parameter.default:
            arguments.append(option_token(parameter))
        elif parameter.required:
            arguments.extend(
                [option_token(parameter), parameter_value(name, root=root, source=source)]
            )


def cli_arguments(command: click.Command, *, root: Path, source: Path) -> list[str]:
    arguments: list[str] = []
    selected = command
    if isinstance(command, click.Group):
        append_command_parameters(arguments, command, root=root, source=source)
        assert "transcribe" in command.commands, sorted(command.commands)
        arguments.append("transcribe")
        selected = command.commands["transcribe"]
    append_command_parameters(arguments, selected, root=root, source=source)
    return arguments


def canonical_file_document(
    root: Path,
    source_name: str,
) -> dict[str, Any]:
    source = root / source_name
    source.write_bytes(wav_bytes())
    transcribe_file(
        source,
        root,
        config(),
        _engine=ParityEngine(asr_config()),
    )
    return json_output(root)


def test_directory_and_console_match_the_canonical_file_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ParityEngine.instances = 0
    ParityEngine.closes = 0
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "directory-cli-parity-1")
    patch_operation_runtime(monkeypatch)

    source_name = "surface.mp3"
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    baseline = canonical_file_document(baseline_root, source_name)

    directory_root = tmp_path / "directory"
    prepare_project(directory_root, source_name)
    call_run_pipeline(directory_root, config())
    directory = json_output(directory_root)

    cli_root = tmp_path / "cli"
    _paths, cli_source = prepare_project(cli_root, source_name)
    command, _module = console_entrypoint()
    monkeypatch.chdir(cli_root)
    result = CliRunner().invoke(
        command,
        cli_arguments(command, root=cli_root, source=cli_source),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    cli = json_output(cli_root)

    documents = [baseline, directory, cli]
    for document in documents:
        validate_document(document)
        assert document["file"] == source_name
        assert document["meta"]["audio_file"] == source_name

    projections = [semantic_projection(document) for document in documents]
    assert projections[1:] == projections[:-1]
    assert ParityEngine.instances == 3
    assert ParityEngine.closes == 2


@pytest.mark.skipif(
    os.getenv("SLOWER_WHISPER_INSTALLED_CLI_PARITY") != "1",
    reason="installed console transaction runs only in the wheel acceptance lane",
)
def test_installed_console_script_matches_the_canonical_surface(tmp_path: Path) -> None:
    executable = shutil.which("slower-whisper")
    assert executable is not None

    root = tmp_path / "installed-cli"
    _paths, source = prepare_project(root)
    command, _module = console_entrypoint()
    arguments = cli_arguments(command, root=root, source=source)

    injection = tmp_path / "injection"
    injection.mkdir()
    lifecycle = tmp_path / "lifecycle.json"
    (injection / "sitecustomize.py").write_text(
        "from parity_injection import install\ninstall()\n",
        encoding="utf-8",
    )
    (injection / "parity_injection.py").write_text(
        f'''from __future__ import annotations
import atexit
import json
import os
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

LIFECYCLE = Path({str(lifecycle)!r})
state = {{"instances": 0, "closes": 0}}

class Engine:
    def __init__(self, cfg):
        state["instances"] += 1
        self.cfg = cfg
        self.model_load_attempts = [
            {{"device": "cuda", "compute_type": "float16", "outcome": "failed", "reason_code": "asr_model_load_failed"}},
            {{"device": "cpu", "compute_type": "int8", "outcome": "selected", "reason_code": "ok"}},
        ]
        self.cfg.device = "cpu"
        self.cfg.compute_type = "int8"

    def transcribe_file(self, audio_path):
        from transcription.models import Segment, Transcript, Word
        return Transcript(
            file_name=Path(audio_path).name,
            language="en",
            segments=[Segment(id=0, start=0.0, end=0.01, text="hello", words=[Word(word="hello", start=0.0, end=0.01, probability=0.99)])],
            meta={{
                "asr_backend": "faster-whisper",
                "asr_model": self.cfg.model_name,
                "asr_device": self.cfg.device,
                "asr_compute_type": self.cfg.compute_type,
                "asr_model_load_attempts": list(self.model_load_attempts),
            }},
        )

    def close(self):
        state["closes"] += 1


def wav_bytes():
    import io
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("h" * 160, *([0] * 160)))
    return buffer.getvalue()


def normalize_all(paths):
    paths.norm_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in sorted(paths.raw_dir.iterdir()):
        (paths.norm_dir / f"{{raw_path.stem}}.wav").write_bytes(wav_bytes())


def normalize_single(_input, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(wav_bytes())


def install():
    import transcription.asr_engine as asr_engine
    import transcription.audio_io as audio_io
    asr_engine.TranscriptionEngine = Engine
    audio_io.normalize_all = normalize_all
    audio_io.normalize_single = normalize_single
    try:
        import transcription.api as api
        api._get_wav_duration_seconds = lambda _path: 0.01
        api._maybe_run_diarization = lambda transcript, _path, _config: transcript
        api._maybe_build_chunks = lambda transcript, _config: transcript
    except Exception:
        pass

@atexit.register
def write_lifecycle():
    LIFECYCLE.write_text(json.dumps(state), encoding="utf-8")
''',
        encoding="utf-8",
    )

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(injection)
    environment["SLOWER_WHISPER_SOURCE_COMMIT"] = "abcdef123456"
    process = subprocess.run(
        [executable, *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    document = json_output(root)
    validate_document(document)
    assert document["file"] == "surface.mp3"
    assert document["meta"]["audio_file"] == "surface.mp3"
    assert json.loads(lifecycle.read_text(encoding="utf-8")) == {
        "instances": 1,
        "closes": 1,
    }


def test_console_entrypoint_matches_pyproject_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected = project["project"]["scripts"]["slower-whisper"]
    points = [
        point
        for point in metadata.entry_points(group="console_scripts")
        if point.name == "slower-whisper"
    ]
    assert len(points) == 1
    assert points[0].value == expected
