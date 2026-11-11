"""Current documentation must not send users to removed v1 commands."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_code_examples import CodeExample, check_current_cli, validate_current_cli

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify_code_examples.py"


@pytest.fixture(autouse=True)
def isolated_model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep subprocess import-time cache creation out of the user's home."""
    monkeypatch.setenv("SLOWER_WHISPER_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache/hf"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "cache/torch"))
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.mark.parametrize(
    "command",
    [
        "python transcribe_pipeline.py",
        "uv run python scripts/audio_enrich.py --skip-existing",
        "python3 ./scripts/transcribe_pipeline.py",
        "python3.12 -m scripts.audio_enrich",
        "python -m scripts.transcribe_pipeline",
        "python -m transcription.audio_enrich_cli",
        "python -m transcription.cli_legacy_transcribe",
        "slower-whisper-enrich --enable-prosody",
        "slower-whisper enrich --enrich-config config.json",
    ],
)
def test_removed_commands_are_errors(command: str) -> None:
    """All retired script, module, executable and flag forms fail validation."""
    issues = validate_current_cli(CodeExample("current.md", 12, command, "bash"))
    assert issues
    assert all(issue.severity == "error" for issue in issues)
    assert all(issue.example.line_number == 12 for issue in issues)


@pytest.mark.parametrize(
    "command",
    [
        "slower-whisper transcribe --root .",
        "uv run slower-whisper enrich --root . --config config.json",
        'grep -r "slower-whisper-enrich" .',
        'rg -- "--enrich-config" .',
        'echo "slower-whisper-enrichment-config-v1.8"',
    ],
)
def test_current_commands_and_migration_searches_are_allowed(command: str) -> None:
    """Valid commands, searches and schema identifiers are not stale invocations."""
    assert not validate_current_cli(CodeExample("current.md", 1, command, "bash"))


def test_history_exemption_is_scoped_to_individual_fences(tmp_path: Path) -> None:
    """A historical snippet does not exempt current commands in the same file."""
    (tmp_path / "README.md").write_text(
        "Before v2 (removed):\n```bash historical\npython audio_enrich.py\n```\n"
        "Current:\n```bash\npython audio_enrich.py\n```\n",
        encoding="utf-8",
    )
    issues = check_current_cli(tmp_path)
    assert len(issues) == 1
    assert issues[0].example.line_number == 7


def test_archived_docs_are_excluded_but_python_help_is_checked(tmp_path: Path) -> None:
    """History stays intact while a Python help/error string is still executable guidance."""
    for directory in ("docs/archive", "docs/releases"):
        path = tmp_path / directory / "old.md"
        path.parent.mkdir(parents=True)
        path.write_text("```bash\npython audio_enrich.py\n```\n", encoding="utf-8")
    path = tmp_path / "examples/workflow.py"
    path.parent.mkdir()
    path.write_text('print("python audio_enrich.py")\n', encoding="utf-8")
    issues = check_current_cli(tmp_path)
    assert len(issues) == 1
    assert issues[0].example.source_file == str(path)


def test_cli_verifier_fails_for_current_removed_commands(tmp_path: Path) -> None:
    """The real script exits nonzero, rather than only printing a warning."""
    path = tmp_path / "README.md"
    path.write_text("```bash\npython audio_enrich.py\n```\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(VERIFIER), "--legacy-cli-only"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "README.md:2: Removed v2 CLI" in result.stdout
    path.write_text("```bash\nslower-whisper enrich --root .\n```\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(VERIFIER), "--legacy-cli-only"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "0 error(s)" in result.stdout


def test_repository_current_cli_examples_are_valid() -> None:
    """Scan the actual current README, docs and runnable example sources."""
    assert check_current_cli(ROOT) == []


@pytest.mark.parametrize(
    "command",
    [
        "slower-whisper transcribe --root .",
        "slower-whisper enrich --root .",
        "slower-whisper enrich --config config.json",
    ],
)
def test_replacement_commands_parse_with_real_cli(command: str) -> None:
    """Check replacements against the production parser, without running ASR."""
    from transcription.cli import build_parser

    args = build_parser().parse_args(shlex.split(command)[1:])
    assert args.command in ("transcribe", "enrich")


def test_meeting_help_shows_canonical_prerequisites() -> None:
    """Exercise the argparse help path without loading an audio model."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/workflows/meeting_transcription.py"), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "slower-whisper transcribe --root ." in result.stdout
    assert "slower-whisper enrich --root ." in result.stdout
    assert "reads the generated JSON from whisper_json/" in result.stdout
    assert not validate_current_cli(CodeExample("help", 1, result.stdout, "text"))


@pytest.mark.parametrize("workflow", ["meeting_transcription", "podcast_processing"])
def test_missing_transcript_reports_current_commands(tmp_path: Path, workflow: str) -> None:
    """Execute the missing-transcript branch with an existing input and no model."""
    audio = tmp_path / "meeting with spaces.wav"
    audio.touch()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/workflows" / f"{workflow}.py"),
            "--audio",
            str(audio),
            "--output",
            str(tmp_path / "output"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Transcript not found" in result.stderr
    assert "slower-whisper transcribe --root ." in result.stderr
    assert "slower-whisper enrich --root ." in result.stderr
    assert "raw_audio/" in result.stderr
    assert not validate_current_cli(CodeExample("error", 1, result.stderr, "text"))


def test_llm_walkthrough_prints_current_pipeline() -> None:
    """Run the interactive walkthrough through its next-steps section."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/llm_integration/quick_start.py")],
        input="\n" * 10,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "slower-whisper transcribe --root ." in result.stdout
    assert "slower-whisper enrich --root ." in result.stdout
    assert "whisper_json/your_audio.json" in result.stdout
    assert not validate_current_cli(CodeExample("walkthrough", 1, result.stdout, "text"))


def test_meeting_reads_pipeline_transcript_without_mixing_report_json(tmp_path: Path) -> None:
    """Keep analysis JSON out of the directory consumed by subsequent enrichment."""
    audio = tmp_path / "raw_audio/meeting.wav"
    audio.parent.mkdir()
    audio.touch()
    transcript = tmp_path / "whisper_json/meeting.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "file": "meeting.wav",
                "language": "en",
                "segments": [
                    {
                        "id": 0,
                        "start": 0,
                        "end": 1,
                        "text": "We agreed on the plan.",
                        "audio_state": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    original = transcript.read_bytes()
    output = tmp_path / "outputs/meetings"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/workflows/meeting_transcription.py"),
            "--audio",
            str(audio),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Meeting analysis complete!" in result.stderr
    assert (output / "meeting_engagement.json").is_file()
    assert (output / "meeting_analysis.txt").is_file()
    assert list(transcript.parent.glob("*.json")) == [transcript]
    assert transcript.read_bytes() == original
    normalized = tmp_path / "input_audio/meeting.wav"
    normalized.parent.mkdir()
    normalized.touch()
    enrich = subprocess.run(
        [
            sys.executable,
            "-m",
            "transcription.cli",
            "enrich",
            "--root",
            str(tmp_path),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert enrich.returncode == 0, enrich.stdout + enrich.stderr
    summary = " ".join(enrich.stdout.split())
    assert "Skipped: 1" in summary
    assert "Failed: 0" in summary


@pytest.mark.parametrize("source", ["walkthrough", "index"])
def test_llm_next_steps_run_from_repository_root(tmp_path: Path, source: str) -> None:
    """Execute printed/documented prompt commands with a fixture, never an API call."""
    transcript = tmp_path / "your_audio.json"
    transcript.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "file": "your_audio.wav",
                "language": "en",
                "segments": [{"id": 0, "start": 0, "end": 1, "text": "We agreed on the plan."}],
            }
        ),
        encoding="utf-8",
    )
    if source == "walkthrough":
        walkthrough = subprocess.run(
            [sys.executable, "examples/llm_integration/quick_start.py"],
            cwd=ROOT,
            input="\n" * 10,
            text=True,
            capture_output=True,
            check=True,
        )
        text = walkthrough.stdout
    else:
        text = (ROOT / "examples/llm_integration/INDEX.md").read_text(encoding="utf-8")

    commands = [
        shlex.split(line.strip().removeprefix("$ "))
        for line in text.splitlines()
        if "python " in line and "whisper_json/" in line
    ]
    assert len(commands) == (4 if source == "walkthrough" else 1)
    for command in commands:
        assert (ROOT / command[1]).is_file(), command
        command[0] = sys.executable
        command[2] = str(transcript)
        if ">" in command:
            command = command[: command.index(">")]
        is_api_demo = command[1].endswith("llm_integration_demo.py")
        if is_api_demo:
            command.append("--dry-run")
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=30
        )
        assert result.returncode == 0, result.stderr
        if is_api_demo:
            assert "Loaded transcript: your_audio.wav" in result.stdout
            assert "DRY RUN (prompts only, no API calls)" in result.stdout
            assert "Querying LLM..." not in result.stdout
        else:
            assert "We agreed on the plan." in result.stdout
