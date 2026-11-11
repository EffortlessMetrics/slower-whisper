"""Dry runs predict batch work without mutating the project or loading models."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest

from transcription.cli import build_parser, main
from transcription.cli_commands import dry_run


def _wav(path: Path, seconds: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000 * seconds)
    return path


def _transcript(root: Path, stem: str, states: tuple = (None,), **extra) -> Path:
    path = root / "whisper_json" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "file": f"{stem}.wav",
                "language": "en",
                "segments": [
                    {"id": i, "start": i, "end": i + 1, "text": "hi", "audio_state": state}
                    for i, state in enumerate(states)
                ],
                **extra,
            }
        )
    )
    return path


def _snapshot(root: Path) -> dict:
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.read_bytes())
        if path.is_file()
        else None
        for path in root.rglob("*")
    }


@pytest.fixture(autouse=True)
def no_processing(monkeypatch):
    """Any normal execution path would violate the dry-run guarantee."""

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted model/device initialization or output mutation")

    for target in (
        "transcription.cli.resolve_device",
        "transcription.pipeline.run_pipeline",
        "transcription.asr_engine.TranscriptionEngine.__init__",
        "transcription.audio_io.ensure_dirs",
        "transcription.audio_io.normalize_all",
        "transcription.api.enrich_transcript",
        "transcription.writers.write_json",
    ):
        monkeypatch.setattr(target, forbidden)


@pytest.mark.parametrize("command", ["transcribe", "enrich"])
def test_flag_is_opt_in(command):
    parser = build_parser()
    assert not parser.parse_args([command]).dry_run
    assert parser.parse_args([command, "--dry-run"]).dry_run


def test_transcribe_empty_root_creates_nothing(tmp_path, capsys):
    root = tmp_path / "not-created"
    assert main(["transcribe", "--root", str(root), "--dry-run"]) == 0
    assert not root.exists()
    assert "Would process: 0" in capsys.readouterr().out


def test_transcribe_lists_raw_and_normalized_files_without_writes(tmp_path, capsys):
    _wav(tmp_path / "raw_audio" / "raw.wav", 3)
    _wav(tmp_path / "input_audio" / "normalized.wav", 2)
    (tmp_path / "raw_audio" / "nested").mkdir()
    _wav(tmp_path / "raw_audio" / "nested" / "ignored.wav")
    before = _snapshot(tmp_path)
    assert main(["transcribe", "--root", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "raw.wav: normalize raw.wav, transcribe (3s)" in out
    assert "normalized.wav: transcribe (2s)" in out
    assert "ignored.wav" not in out
    assert "Would process: 2" in out
    assert "Total estimated audio duration: 5s" in out
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("force", [False, True])
def test_transcribe_existing_json_respects_skip_override(tmp_path, capsys, force):
    _wav(tmp_path / "input_audio" / "done.wav")
    _transcript(tmp_path, "done")
    args = ["transcribe", "--root", str(tmp_path), "--dry-run"]
    if force:
        args.append("--no-skip-existing")
    assert main(args) == 0
    out = capsys.readouterr().out
    assert ("done.wav: transcribe (2s)" if force else "done.wav: skip (existing JSON)") in out
    assert f"Would process: {int(force)}" in out


@pytest.mark.parametrize("fresh", [False, True])
def test_transcribe_normalization_uses_mtime_and_effective_audio(tmp_path, capsys, fresh):
    raw = _wav(tmp_path / "raw_audio" / "same.wav", 3)
    normalized = _wav(tmp_path / "input_audio" / "same.wav", 1)
    os.utime(raw, (100, 100))
    os.utime(normalized, (200 if fresh else 50, 200 if fresh else 50))
    before = _snapshot(tmp_path)
    assert main(["transcribe", "--root", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert ("normalize same.wav" in out) is not fresh
    assert f"Total estimated audio duration: {1 if fresh else 3}s" in out
    assert "Would process: 1" in out
    assert _snapshot(tmp_path) == before


def test_normalization_still_happens_when_transcription_is_skipped(tmp_path, capsys):
    _wav(tmp_path / "raw_audio" / "done.wav")
    _transcript(tmp_path, "done")
    assert main(["transcribe", "--root", str(tmp_path), "--dry-run"]) == 0
    assert "normalize done.wav, skip transcription (existing JSON)" in capsys.readouterr().out
    assert not (tmp_path / "input_audio").exists()


def test_transcribe_rejects_casefolded_source_collisions_without_writes(tmp_path, capsys):
    _wav(tmp_path / "raw_audio" / "Same.wav")
    _wav(tmp_path / "raw_audio" / "same.mp3")
    before = _snapshot(tmp_path)
    assert main(["transcribe", "--root", str(tmp_path), "--dry-run"]) == 1
    assert "would overwrite the same normalized WAV" in capsys.readouterr().err
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("status", [None, "failed", "success", "ok"])
def test_transcribe_reports_diarization_of_existing_json(tmp_path, capsys, status):
    _wav(tmp_path / "input_audio" / "existing.wav")
    _transcript(tmp_path, "existing", meta={"diarization": {"status": status}})
    before = _snapshot(tmp_path)
    assert main(["transcribe", "--root", str(tmp_path), "--dry-run", "--enable-diarization"]) == 0
    out = capsys.readouterr().out
    assert ("diarize existing transcript" in out) is (status not in {"success", "ok"})
    assert _snapshot(tmp_path) == before


def test_transcribe_does_not_hide_chunk_updates_behind_skip(tmp_path, capsys):
    _wav(tmp_path / "input_audio" / "existing.wav")
    _transcript(tmp_path, "existing")
    before = _snapshot(tmp_path)
    assert main(["transcribe", "--root", str(tmp_path), "--dry-run", "--enable-chunking"]) == 0
    assert "update chunks, skip transcription" in capsys.readouterr().out
    assert _snapshot(tmp_path) == before


def test_transcribe_reports_invalid_existing_transcript_for_diarization(tmp_path, capsys):
    _wav(tmp_path / "input_audio" / "bad.wav")
    _transcript(tmp_path, "bad").write_text("{bad")
    assert main(["transcribe", "--root", str(tmp_path), "--dry-run", "--enable-diarization"]) == 1
    assert "bad.wav: would fail to read bad.json" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["transcribe", "enrich"])
def test_invalid_config_is_not_ignored_for_preview(tmp_path, capsys, command):
    config = tmp_path / "config.json"
    config.write_text("{bad")
    assert main([command, "--root", str(tmp_path), "--config", str(config), "--dry-run"]) == 1
    assert "configuration" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command,key", [("transcribe", "skip_existing_json"), ("enrich", "skip_existing")]
)
def test_preview_uses_merged_config_and_cli_precedence(tmp_path, capsys, command, key):
    _wav(tmp_path / "input_audio" / "done.wav")
    _transcript(tmp_path, "done", states=({},))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({key: False}))
    args = [command, "--root", str(tmp_path), "--config", str(config), "--dry-run"]
    assert main(args) == 0
    assert "Would process: 1" in capsys.readouterr().out
    assert main([*args, "--skip-existing"]) == 0
    assert "Would skip: 1" in capsys.readouterr().out


def test_enrich_missing_or_empty_json_directory_is_actionable(tmp_path, capsys):
    args = ["enrich", "--root", str(tmp_path), "--dry-run"]
    assert main(args) == 1
    assert "Run transcription first" in capsys.readouterr().err
    assert not (tmp_path / "whisper_json").exists()
    (tmp_path / "whisper_json").mkdir()
    assert main(args) == 1
    assert "No JSON transcript files" in capsys.readouterr().err


def test_enrich_handles_partial_full_and_empty_transcripts_without_writes(tmp_path, capsys):
    for stem, states in [("full", ({}, {})), ("partial", ({}, None)), ("empty", ())]:
        _wav(tmp_path / "input_audio" / f"{stem}.wav")
        _transcript(tmp_path, stem, states=states)
    before = _snapshot(tmp_path)
    assert main(["enrich", "--root", str(tmp_path), "--dry-run", "--device", "auto"]) == 0
    out = capsys.readouterr().out
    assert "full.json: skip (already enriched)" in out
    assert "partial.json: enrich (2 segments, 2s)" in out
    assert "empty.json: enrich (0 segments, 2s)" in out
    assert "Total estimated audio duration: 4s" in out
    assert _snapshot(tmp_path) == before


def test_enrich_reports_missing_audio_and_bad_json_continues_valid_files(tmp_path, capsys):
    _transcript(tmp_path, "missing", states=({},))
    _wav(tmp_path / "input_audio" / "bad.wav")
    _transcript(tmp_path, "bad").write_text("not json")
    _wav(tmp_path / "input_audio" / "good.wav")
    _transcript(tmp_path, "good")
    before = _snapshot(tmp_path)
    assert main(["enrich", "--root", str(tmp_path), "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "missing.json: would fail (audio file not found:" in out
    assert "bad.json: would fail to read transcript" in out
    assert "good.json: enrich" in out
    assert "Would fail: 2" in out
    assert _snapshot(tmp_path) == before


def test_unknown_durations_are_not_silently_zero(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw_audio"
    raw.mkdir()
    (raw / "unknown.mp3").write_bytes(b"not real audio")
    _wav(raw / "known.wav", 3)
    monkeypatch.setattr(dry_run.subprocess, "run", Mock(side_effect=FileNotFoundError))
    assert main(["transcribe", "--root", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "duration unknown" in out
    assert "Total estimated audio duration: 3s" in out
    assert "Duration unknown for 1 file(s)" in out


@pytest.mark.parametrize("value", ["N/A", "nan", "inf", "-1", "", "bad"])
def test_invalid_probe_duration_is_unknown(tmp_path, monkeypatch, value):
    monkeypatch.setattr(
        dry_run.subprocess, "run", Mock(return_value=Mock(returncode=0, stdout=value))
    )
    assert dry_run._audio_duration(tmp_path / "audio.mp3") is None


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("ffprobe", 5)])
def test_probe_failure_is_unknown(tmp_path, monkeypatch, error):
    monkeypatch.setattr(dry_run.subprocess, "run", Mock(side_effect=error))
    assert dry_run._audio_duration(tmp_path / "audio.mp3") is None


def test_probe_uses_timeout_absolute_path_and_local_protocols(tmp_path, monkeypatch):
    probe = Mock(return_value=Mock(returncode=0, stdout="12.5\n"))
    monkeypatch.setattr(dry_run.subprocess, "run", probe)
    path = tmp_path / "-audio.mp3"
    assert dry_run._audio_duration(path) == 12.5
    args, kwargs = probe.call_args
    assert args[0][-1] == str(path.resolve())
    assert args[0][args[0].index("-protocol_whitelist") + 1] == "file,pipe"
    assert kwargs["timeout"] == 5
    assert not kwargs.get("shell", False)


def test_wav_metadata_does_not_require_ffprobe(tmp_path, monkeypatch):
    probe = Mock(side_effect=AssertionError("ffprobe not needed for PCM WAV"))
    monkeypatch.setattr(dry_run.subprocess, "run", probe)
    assert dry_run._audio_duration(_wav(tmp_path / "audio.wav", 4)) == 4
    probe.assert_not_called()


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0s"), (62, "1m 02s"), (3661, "1h 01m 01s"), (None, "duration unknown")],
)
def test_duration_format(seconds, expected):
    assert dry_run._format_duration(seconds) == expected


def test_malformed_wav_chunk_falls_back_to_unknown(tmp_path, monkeypatch):
    path = tmp_path / "broken.wav"
    path.write_bytes(
        b"RIFF" + struct.pack("<I", 16) + b"WAVEJUNK" + struct.pack("<I", 0xFFFFFFFF) + b"\0" * 4
    )
    monkeypatch.setattr(dry_run.subprocess, "run", Mock(side_effect=FileNotFoundError))
    assert dry_run._audio_duration(path) is None


def test_transcribe_chunk_only_read_failure_warns_then_skips(tmp_path, capsys):
    _wav(tmp_path / "input_audio" / "bad.wav")
    _transcript(tmp_path, "bad").write_text("{bad")
    assert main(["transcribe", "--root", str(tmp_path), "--dry-run", "--enable-chunking"]) == 0
    out = capsys.readouterr().out
    assert "warning: could not update chunks" in out
    assert "Would skip: 1" in out
    assert "Would fail: 0" in out
