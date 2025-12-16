import importlib.util
from pathlib import Path

import pytest

from integrations.langchain_loader import SlowerWhisperLoader
from integrations.llamaindex_reader import SlowerWhisperReader
from transcription.chunking import ChunkingConfig
from transcription.models import Segment, Transcript, Turn
from transcription.writers import write_json

pytestmark = pytest.mark.integration


class _FakeDoc:
    def __init__(self, page_content=None, text=None, metadata=None):
        self.page_content = page_content or text
        self.text = text or page_content
        self.metadata = metadata or {}


def _write_sample(tmp_path: Path) -> Path:
    segments = [
        Segment(id=0, start=0.0, end=4.0, text="hello there", speaker={"id": "spk_0"}),
        Segment(id=1, start=4.1, end=7.0, text="general kenobi", speaker={"id": "spk_1"}),
    ]
    turns = [
        Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0],
            start=0.0,
            end=4.0,
            text="hello there",
            metadata={},
        ),
        Turn(
            id="turn_1",
            speaker_id="spk_1",
            segment_ids=[1],
            start=4.1,
            end=7.0,
            text="general kenobi",
            metadata={},
        ),
    ]
    transcript = Transcript(file_name="sample.wav", language="en", segments=segments, turns=turns)
    out_path = tmp_path / "sample.json"
    write_json(transcript, out_path)
    return out_path


def test_langchain_loader_emits_documents(monkeypatch, tmp_path):
    _write_sample(tmp_path)
    cfg = ChunkingConfig(
        target_duration_s=2.0, max_duration_s=4.5, target_tokens=5, pause_split_threshold_s=0.2
    )
    loader = SlowerWhisperLoader(tmp_path, chunking_config=cfg)
    monkeypatch.setattr(loader, "_document_class", lambda: _FakeDoc)

    docs = loader.load()
    assert len(docs) == 2
    assert docs[0].page_content
    assert docs[0].metadata.get("speaker_ids")


def test_llamaindex_reader_emits_documents(monkeypatch, tmp_path):
    _write_sample(tmp_path)
    cfg = ChunkingConfig(
        target_duration_s=2.0, max_duration_s=4.5, target_tokens=5, pause_split_threshold_s=0.2
    )
    reader = SlowerWhisperReader(tmp_path, chunking_config=cfg)
    monkeypatch.setattr(reader, "_document_class", lambda: _FakeDoc)

    docs = reader.load_data()
    assert len(docs) == 2
    assert docs[0].text
    assert docs[0].metadata.get("chunk_id")


def test_langchain_loader_uses_sample_transcript_metadata():
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip(
            "langchain-core not installed; run `uv sync --extra integrations` or `pip install -e '.[integrations]'`"
        )

    sample_path = Path("benchmarks/data/samples/sample_transcript.json")
    loader = SlowerWhisperLoader(sample_path)

    docs = loader.load()
    assert len(docs) >= 1
    meta = docs[0].metadata
    assert meta.get("start_time") is not None
    assert meta.get("end_time") is not None
    assert meta.get("speakers")
