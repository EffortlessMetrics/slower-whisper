"""LlamaIndex reader for slower-whisper transcripts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from transcription import load_transcript
from transcription.chunking import ChunkingConfig, build_chunks
from transcription.models import Chunk, Transcript


@dataclass
class SlowerWhisperReader:
    path: str | Path
    chunking_config: ChunkingConfig | None = None

    def _document_class(self):
        try:
            from llama_index.core import Document  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                'llama-index-core is required for SlowerWhisperReader; install with `uv sync --extra integrations` or `pip install -e ".[integrations]"`.'
            ) from exc
        return Document

    def _iter_transcripts(self) -> Iterable[tuple[Transcript, Path]]:
        path = Path(self.path)
        if path.is_dir():
            for json_path in sorted(path.glob("*.json")):
                yield load_transcript(json_path), json_path
        else:
            yield load_transcript(path), path

    def _normalize_chunks(self, chunks: list[Chunk | dict] | None) -> list[Chunk]:
        if not chunks:
            return []
        normalized: list[Chunk] = []
        for chunk in chunks:
            if isinstance(chunk, Chunk):
                normalized.append(chunk)
            elif isinstance(chunk, dict):
                normalized.append(Chunk.from_dict(chunk))
        return normalized

    def _chunk_text(self, transcript: Transcript, chunk: Chunk) -> str:
        if getattr(chunk, "text", ""):
            return chunk.text

        seg_lookup = {seg.id: seg for seg in transcript.segments}
        texts: list[str] = []
        for seg_id in chunk.segment_ids:
            seg = seg_lookup.get(seg_id)
            if seg and seg.text:
                texts.append(seg.text)
        return " ".join(texts)

    def _ensure_chunks(self, transcript: Transcript) -> list[Chunk]:
        existing = self._normalize_chunks(getattr(transcript, "chunks", None))
        if existing:
            return existing

        cfg = self.chunking_config or ChunkingConfig()
        return build_chunks(transcript, cfg)

    def load_data(self) -> list[object]:
        Document = self._document_class()
        docs: list[object] = []

        for transcript, source_path in self._iter_transcripts():
            chunks = self._ensure_chunks(transcript)
            for chunk in chunks:
                content = self._chunk_text(transcript, chunk)
                metadata = {
                    "source": transcript.file_name,
                    "source_path": str(source_path),
                    "chunk_id": chunk.id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "start_time": chunk.start,
                    "end_time": chunk.end,
                    "speakers": chunk.speaker_ids,
                    "speaker_ids": chunk.speaker_ids,
                    "turn_ids": chunk.turn_ids,
                    "segment_ids": chunk.segment_ids,
                    "language": transcript.language,
                }
                docs.append(Document(text=content, metadata=metadata))

        return docs
