"""Dry-run must not create or overwrite output files in ledger CLI modes."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def ledger():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate-pr-ledger.py"
    spec = importlib.util.spec_from_file_location("generate_pr_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", [[], ["--raw"]])
def test_dry_run_preserves_output_and_normal_run_writes(ledger, tmp_path, capsys, mode):
    output = tmp_path / "output.json"
    output.write_text("existing")
    with (
        patch.object(ledger, "_legacy_gather_pr_data", return_value={"diff": "", "number": 123}),
        patch.object(ledger, "_generate_analysis_prompt", return_value="prompt"),
    ):
        with patch.object(
            sys, "argv", ["ledger", "--pr", "123", *mode, "--output", str(output), "--dry-run"]
        ):
            ledger.main()
        assert output.read_text() == "existing"
        assert capsys.readouterr().out
        with patch.object(sys, "argv", ["ledger", "--pr", "123", *mode, "--output", str(output)]):
            ledger.main()
        assert output.read_text() != "existing"


def test_dump_bundle_dry_run_does_not_create_default_directory(
    ledger, tmp_path, monkeypatch, capsys
):
    from transcription import historian
    from transcription.historian import estimation

    class Bundle:
        def to_dict(self):
            return {"number": 123}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(historian, "gather_pr_data", lambda *args, **kwargs: Bundle())
    monkeypatch.setattr(estimation, "compute_bounded_estimate", lambda bundle: None)
    monkeypatch.setattr(sys, "argv", ["ledger", "--pr", "123", "--dump-bundle", "--dry-run"])
    ledger.main()
    assert not (tmp_path / "docs").exists()
    assert '"number": 123' in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["ledger", "--pr", "123", "--dump-bundle"])
    ledger.main()
    assert (tmp_path / "docs/audit/pr/_bundles/123.json").is_file()
