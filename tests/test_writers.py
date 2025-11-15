from pathlib import Path
import json
from transcription.models import Segment, Transcript
from transcription import __version__
from transcription.config import AppConfig
from transcription.pipeline import _build_meta
from transcription import writers  # type: ignore[attr-defined]


def test_write_json_shape(tmp_path: Path) -> None:
    seg = Segment(id=0, start=0.0, end=1.0, text="hello")
    t = Transcript(file_name="test.wav", language="en", segments=[seg])
    cfg = AppConfig()
    t.meta = _build_meta(cfg, t, Path("test.wav"), duration_sec=1.0)

    out_path = tmp_path / "out.json"
    writers.write_json(t, out_path)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["file"] == "test.wav"
    assert data["language"] == "en"
    assert "meta" in data
    assert data["meta"]["model_name"] == cfg.asr.model_name
    assert data["meta"]["pipeline_version"] == __version__
    assert len(data["segments"]) == 1
    seg0 = data["segments"][0]
    assert seg0["id"] == 0
    assert seg0["text"] == "hello"
