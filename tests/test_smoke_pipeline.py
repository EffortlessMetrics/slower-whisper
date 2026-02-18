"""Smoke tests: full pipeline with real ASR model.

Exercises run_pipeline() end-to-end with a real tiny model, temp directories,
and JSON output validation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

AUDIO_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "asr" / "audio"
CALL_CENTER_WAV = AUDIO_DIR / "call_center_narrowband.wav"


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(120)
class TestRealPipeline:
    """Full pipeline smoke test with real ASR."""

    def test_pipeline_produces_valid_json(self, tmp_path: Path) -> None:
        """run_pipeline should produce schema v2 JSON with real transcription."""
        # Skip if ffmpeg not available (required for audio normalization)
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not found on PATH")

        assert CALL_CENTER_WAV.exists(), f"Missing fixture: {CALL_CENTER_WAV}"

        from transcription.config import AppConfig, AsrConfig, Paths
        from transcription.pipeline import run_pipeline

        # Set up project directory structure
        project_dir = tmp_path / "project"
        input_dir = project_dir / "input_audio"
        input_dir.mkdir(parents=True)
        shutil.copy2(CALL_CENTER_WAV, input_dir / CALL_CENTER_WAV.name)

        cfg = AppConfig(
            paths=Paths(root=project_dir),
            asr=AsrConfig(model_name="tiny", device="cpu", compute_type="int8"),
        )

        result = run_pipeline(cfg)
        assert result.processed >= 1, f"Expected >=1 processed, got {result.processed}"
        assert result.failed == 0, f"Pipeline had {result.failed} failures"

        # Check JSON output
        json_dir = project_dir / "whisper_json"
        json_files = list(json_dir.glob("*.json"))
        assert json_files, f"No JSON files in {json_dir}"

        data = json.loads(json_files[0].read_text())
        assert data["schema_version"] == 2, f"Expected schema v2, got {data.get('schema_version')}"
        assert data.get("language") == "en"
        assert len(data.get("segments", [])) > 0

        # Check keywords in combined text
        full_text = " ".join(s["text"] for s in data["segments"]).lower()
        found = [
            kw for kw in ["support", "password", "email", "account", "help"] if kw in full_text
        ]
        assert found, f"Expected keywords in output, got: {full_text[:300]}"
