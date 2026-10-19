"""Resume accounting for unreadable output left by interrupted batch jobs."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transcription import audio_io, cli, pipeline, writers
from transcription.config import AppConfig, AsrConfig, Paths, TranscriptionConfig
from transcription.models import Transcript


@pytest.fixture
def resume_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[AppConfig, MagicMock]:
    paths = Paths(root=tmp_path)
    audio_io.ensure_dirs(paths)
    engine = MagicMock()
    engine.transcribe_file.side_effect = lambda path: Transcript(
        file_name=path.name, language="en", segments=[]
    )
    monkeypatch.setattr(pipeline, "TranscriptionEngine", lambda cfg: engine)
    return AppConfig(paths=paths, asr=AsrConfig(device="cpu"), skip_existing_json=True), engine


def add_normalized_audio(paths: Paths, stem: str) -> None:
    with wave.open(str(paths.norm_dir / f"{stem}.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)


@pytest.mark.parametrize(
    "contents",
    [b"", b'{"segments": [', b"\xff", b"null", b'{"segments": [{}]}'],
    ids=["empty", "truncated", "invalid-utf8", "null-root", "missing-segment-fields"],
)
@pytest.mark.parametrize("with_config", [False, True], ids=["pipeline", "cli-config"])
def test_unreadable_existing_json_is_failed_and_preserved(
    resume_project: tuple[AppConfig, MagicMock], contents: bytes, with_config: bool
) -> None:
    cfg, engine = resume_project
    add_normalized_audio(cfg.paths, "interrupted")
    json_path = cfg.paths.json_dir / "interrupted.json"
    json_path.write_bytes(contents)

    result = pipeline.run_pipeline(cfg, TranscriptionConfig() if with_config else None)

    assert (result.total_files, result.failed, result.skipped, result.processed) == (1, 1, 0, 0)
    assert [(item.file_name, item.status) for item in result.file_results] == [
        ("interrupted.wav", "error")
    ]
    assert "Load failed:" in (result.file_results[0].error_message or "")
    assert json_path.read_bytes() == contents
    assert list(cfg.paths.transcripts_dir.iterdir()) == []
    engine.transcribe_file.assert_not_called()
    engine.close.assert_called_once_with()


def test_resume_continues_after_unreadable_json_and_keeps_legacy_skip(
    resume_project: tuple[AppConfig, MagicMock],
) -> None:
    cfg, engine = resume_project
    for stem in ("a_interrupted", "b_legacy", "c_new"):
        add_normalized_audio(cfg.paths, stem)
    broken_path = cfg.paths.json_dir / "a_interrupted.json"
    broken_path.write_bytes(b'{"segments": [')
    legacy_path = cfg.paths.json_dir / "b_legacy.json"
    legacy = json.dumps({"file_name": "b_legacy.wav", "language": "en", "segments": []})
    legacy_path.write_text(legacy, encoding="utf-8")

    result = pipeline.run_pipeline(cfg)

    assert (result.total_files, result.failed, result.skipped, result.processed) == (3, 1, 1, 1)
    assert [item.status for item in result.file_results] == ["error", "skipped", "success"]
    assert len(result.file_results) == result.total_files
    assert broken_path.read_bytes() == b'{"segments": ['
    assert legacy_path.read_text(encoding="utf-8") == legacy
    engine.transcribe_file.assert_called_once_with(cfg.paths.norm_dir / "c_new.wav")
    assert (
        writers.load_transcript_from_json(cfg.paths.json_dir / "c_new.json").file_name
        == "c_new.wav"
    )
    engine.close.assert_called_once_with()


def test_resume_after_keyboard_interrupt_during_json_write(
    resume_project: tuple[AppConfig, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, engine = resume_project
    add_normalized_audio(cfg.paths, "interrupted")
    json_path = cfg.paths.json_dir / "interrupted.json"
    write_text = Path.write_text

    def interrupt_write(path: Path, data: str, *, encoding: str | None = None) -> int:
        if path == json_path:
            write_text(path, data[: len(data) // 2], encoding="utf-8")
            raise KeyboardInterrupt
        return write_text(path, data, encoding=encoding)

    with monkeypatch.context() as during_write:
        during_write.setattr(Path, "write_text", interrupt_write)
        with pytest.raises(KeyboardInterrupt):
            pipeline.run_pipeline(cfg)

    interrupted_bytes = json_path.read_bytes()
    assert interrupted_bytes
    with pytest.raises(json.JSONDecodeError):
        json.loads(interrupted_bytes)
    engine.close.assert_called_once_with()

    result = pipeline.run_pipeline(cfg)

    assert (result.failed, result.skipped, result.processed) == (1, 0, 0)
    assert json_path.read_bytes() == interrupted_bytes
    engine.transcribe_file.assert_called_once_with(cfg.paths.norm_dir / "interrupted.wav")
    assert engine.close.call_count == 2


def test_cli_resume_returns_failure_for_interrupted_json(
    resume_project: tuple[AppConfig, MagicMock],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, engine = resume_project
    monkeypatch.setenv("NO_COLOR", "1")
    add_normalized_audio(cfg.paths, "interrupted")
    (cfg.paths.json_dir / "interrupted.json").write_bytes(b'{"segments": [')

    exit_code = cli.main(["transcribe", "--root", str(cfg.paths.root), "--device", "cpu"])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Failed:           1" in output
    assert "Skipped:          0" in output
    assert "interrupted.wav: Load failed:" in output
    engine.transcribe_file.assert_not_called()
