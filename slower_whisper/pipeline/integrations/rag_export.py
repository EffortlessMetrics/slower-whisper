"""RAG bundle export for transcript data.

Provides structured export of transcripts for RAG (Retrieval-Augmented Generation)
ingestion. Supports multiple chunking strategies and optional embedding generation.

Chunking strategies:
- by_segment: One chunk per segment (finest granularity)
- by_speaker_turn: Group consecutive same-speaker segments
- by_time: Fixed time windows (e.g., 30 seconds)
- by_topic: Semantic boundaries (if topic annotations available)

Example usage:
    >>> config = RAGExporterConfig(
    ...     chunking_strategy=ChunkingStrategy.BY_SPEAKER_TURN,
    ...     include_embeddings=True,
    ... )
    >>> exporter = RAGExporter(config)
    >>> bundle = exporter.export(transcript)
    >>> bundle.save("output.json")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import Segment, Transcript

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Chunking strategies for RAG export."""

    BY_SEGMENT = "by_segment"
    BY_SPEAKER_TURN = "by_speaker_turn"
    BY_TIME = "by_time"
    BY_TOPIC = "by_topic"


@dataclass(slots=True)
class RAGChunk:
    """
    A single chunk for RAG ingestion.

    Attributes:
        id: Unique chunk identifier.
        text: Chunk text content.
        metadata: Chunk metadata for filtering/retrieval.
        embedding: Optional embedding vector.
    """

    id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }
        if self.embedding is not None:
            result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGChunk:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


@dataclass
class RAGBundle:
    """
    Complete RAG bundle with chunks and metadata.

    Attributes:
        chunks: List of RAG chunks.
        metadata: Bundle-level metadata.
    """

    chunks: list[RAGChunk]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGBundle:
        """Create from dictionary."""
        return cls(
            chunks=[RAGChunk.from_dict(c) for c in data.get("chunks", [])],
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path | str) -> None:
        """Save bundle to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Saved RAG bundle to %s (%d chunks)", path, len(self.chunks))

    @classmethod
    def load(cls, path: Path | str) -> RAGBundle:
        """Load bundle from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)


@dataclass
class RAGExporterConfig:
    """
    Configuration for RAG export.

    Attributes:
        chunking_strategy: How to split transcript into chunks.
        time_window_seconds: Window size for BY_TIME strategy.
        include_embeddings: Whether to generate embeddings.
        embedding_model: Sentence-transformers model name.
        include_audio_state: Include audio state summaries in metadata.
        include_outcomes: Include semantic outcomes in metadata.
        max_chunk_tokens: Maximum tokens per chunk (soft limit).
    """

    chunking_strategy: ChunkingStrategy = ChunkingStrategy.BY_SPEAKER_TURN
    time_window_seconds: float = 30.0
    include_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    include_audio_state: bool = True
    include_outcomes: bool = True
    max_chunk_tokens: int = 512

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGExporterConfig:
        """Create config from dictionary."""
        strategy = data.get("chunking_strategy", "by_speaker_turn")
        if isinstance(strategy, str):
            strategy = ChunkingStrategy(strategy)
        return cls(
            chunking_strategy=strategy,
            time_window_seconds=data.get("time_window_seconds", 30.0),
            include_embeddings=data.get("include_embeddings", False),
            embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"),
            include_audio_state=data.get("include_audio_state", True),
            include_outcomes=data.get("include_outcomes", True),
            max_chunk_tokens=data.get("max_chunk_tokens", 512),
        )


class RAGExporter:
    """
    Export transcripts as RAG-ready bundles.

    Supports multiple chunking strategies and optional embedding generation.

    Example:
        >>> exporter = RAGExporter(RAGExporterConfig())
        >>> bundle = exporter.export(transcript)
        >>> bundle.save("output.json")
    """

    def __init__(self, config: RAGExporterConfig | None = None) -> None:
        """
        Initialize RAG exporter.

        Args:
            config: Export configuration (uses defaults if not provided).
        """
        self.config = config or RAGExporterConfig()
        self._embedding_model: Any = None

    def _get_embedding_model(self) -> Any:
        """Lazy-load embedding model."""
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.config.embedding_model)
            return self._embedding_model
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        model = self._get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def _get_speaker_id(self, segment: Segment) -> str:
        """Extract speaker ID from segment."""
        speaker = getattr(segment, "speaker", None)
        if speaker is None:
            return "unknown"
        if isinstance(speaker, dict):
            speaker_id = speaker.get("id", "unknown")
            return str(speaker_id) if speaker_id is not None else "unknown"
        return str(speaker)

    def _chunk_by_segment(self, transcript: Transcript) -> list[RAGChunk]:
        """Create one chunk per segment."""
        chunks = []
        for seg in transcript.segments:
            chunk_id = f"chunk_{seg.id:03d}"
            metadata = self._build_chunk_metadata(
                source=transcript.file_name,
                segment_ids=[seg.id],
                speaker=self._get_speaker_id(seg),
                start_time=seg.start,
                end_time=seg.end,
                segment=seg,
            )
            chunks.append(
                RAGChunk(
                    id=chunk_id,
                    text=seg.text.strip(),
                    metadata=metadata,
                )
            )
        return chunks

    def _chunk_by_speaker_turn(self, transcript: Transcript) -> list[RAGChunk]:
        """Group consecutive same-speaker segments."""
        if not transcript.segments:
            return []

        chunks = []
        current_speaker = self._get_speaker_id(transcript.segments[0])
        current_segments: list[Segment] = []
        chunk_counter = 0

        def flush_chunk() -> None:
            nonlocal chunk_counter
            if not current_segments:
                return

            text = " ".join(seg.text.strip() for seg in current_segments)
            segment_ids = [seg.id for seg in current_segments]
            chunk_id = f"chunk_{chunk_counter:03d}"

            metadata = self._build_chunk_metadata(
                source=transcript.file_name,
                segment_ids=segment_ids,
                speaker=current_speaker,
                start_time=current_segments[0].start,
                end_time=current_segments[-1].end,
                segments=current_segments,
            )

            chunks.append(
                RAGChunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                )
            )
            chunk_counter += 1

        for seg in transcript.segments:
            speaker = self._get_speaker_id(seg)
            if speaker != current_speaker:
                flush_chunk()
                current_speaker = speaker
                current_segments = []

            current_segments.append(seg)

        flush_chunk()  # Don't forget the last chunk
        return chunks

    def _chunk_by_time(self, transcript: Transcript) -> list[RAGChunk]:
        """Create fixed time window chunks."""
        if not transcript.segments:
            return []

        window = self.config.time_window_seconds
        chunks = []
        chunk_counter = 0

        current_window_start = 0.0
        current_window_end = window
        current_segments: list[Segment] = []

        def flush_chunk() -> None:
            nonlocal chunk_counter
            if not current_segments:
                return

            text = " ".join(seg.text.strip() for seg in current_segments)
            segment_ids = [seg.id for seg in current_segments]
            speakers = sorted({self._get_speaker_id(seg) for seg in current_segments})
            chunk_id = f"chunk_{chunk_counter:03d}"

            metadata = self._build_chunk_metadata(
                source=transcript.file_name,
                segment_ids=segment_ids,
                speaker=speakers[0] if len(speakers) == 1 else None,
                start_time=current_segments[0].start,
                end_time=current_segments[-1].end,
                segments=current_segments,
            )
            if len(speakers) > 1:
                metadata["speakers"] = speakers

            chunks.append(
                RAGChunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                )
            )
            chunk_counter += 1

        for seg in transcript.segments:
            # Check if segment starts a new window
            while seg.start >= current_window_end:
                flush_chunk()
                current_window_start = current_window_end
                current_window_end = current_window_start + window
                current_segments = []

            current_segments.append(seg)

        flush_chunk()  # Don't forget the last chunk
        return chunks

    def _chunk_by_topic(self, transcript: Transcript) -> list[RAGChunk]:
        """
        Create chunks based on topic boundaries.

        Falls back to speaker turn chunking if no topic annotations available.
        """
        # Check for topic annotations
        annotations = getattr(transcript, "annotations", None)
        if not annotations or "topics" not in annotations:
            logger.warning("No topic annotations found, falling back to speaker turn chunking")
            return self._chunk_by_speaker_turn(transcript)

        topics = annotations["topics"]
        if not topics:
            return self._chunk_by_speaker_turn(transcript)

        chunks = []
        chunk_counter = 0

        for topic in topics:
            # Get segments in this topic's time range
            start = topic.get("start", 0.0)
            end = topic.get("end", transcript.duration)
            topic_name = topic.get("name", "unknown")

            topic_segments = [
                seg for seg in transcript.segments if seg.start >= start and seg.end <= end
            ]

            if not topic_segments:
                continue

            text = " ".join(seg.text.strip() for seg in topic_segments)
            segment_ids = [seg.id for seg in topic_segments]
            speakers = sorted({self._get_speaker_id(seg) for seg in topic_segments})
            chunk_id = f"chunk_{chunk_counter:03d}"

            metadata = self._build_chunk_metadata(
                source=transcript.file_name,
                segment_ids=segment_ids,
                speaker=speakers[0] if len(speakers) == 1 else None,
                start_time=topic_segments[0].start,
                end_time=topic_segments[-1].end,
                segments=topic_segments,
            )
            metadata["topic"] = topic_name
            if len(speakers) > 1:
                metadata["speakers"] = speakers

            chunks.append(
                RAGChunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                )
            )
            chunk_counter += 1

        return chunks

    def _build_chunk_metadata(
        self,
        source: str,
        segment_ids: list[int],
        speaker: str | None,
        start_time: float,
        end_time: float,
        segment: Segment | None = None,
        segments: list[Segment] | None = None,
    ) -> dict[str, Any]:
        """Build metadata dictionary for a chunk."""
        metadata: dict[str, Any] = {
            "source": source,
            "segment_ids": segment_ids,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
        }

        if speaker:
            metadata["speaker"] = speaker

        # Handle single segment
        if segment is not None:
            segments = [segment]

        # Include audio state summary if configured
        if self.config.include_audio_state and segments:
            audio_summary = self._summarize_audio_state(segments)
            if audio_summary:
                metadata["audio_state"] = audio_summary

        # Include outcomes if configured
        if self.config.include_outcomes and segments:
            outcomes = self._extract_outcomes(segments)
            if outcomes:
                metadata["outcomes"] = outcomes

        return metadata

    def _summarize_audio_state(self, segments: list[Segment]) -> dict[str, Any] | None:
        """Summarize audio state across segments."""
        states = [seg.audio_state for seg in segments if seg.audio_state]
        if not states:
            return None

        summary: dict[str, Any] = {}

        # Average prosody values
        pitch_values = []
        energy_values = []
        for state in states:
            prosody = state.get("prosody", {})
            if prosody.get("pitch_median_hz"):
                pitch_values.append(prosody["pitch_median_hz"])
            if prosody.get("energy_median_db"):
                energy_values.append(prosody["energy_median_db"])

        if pitch_values:
            summary["avg_pitch_hz"] = sum(pitch_values) / len(pitch_values)
        if energy_values:
            summary["avg_energy_db"] = sum(energy_values) / len(energy_values)

        # Collect emotions
        emotions = []
        for state in states:
            if "emotion" in state:
                emotions.append(state["emotion"])
        if emotions:
            # Just include the emotions list; could do more aggregation
            summary["emotions"] = emotions

        return summary if summary else None

    def _extract_outcomes(self, segments: list[Segment]) -> list[dict[str, Any]]:
        """Extract outcomes from segments (if annotated)."""
        outcomes = []
        for seg in segments:
            audio_state = seg.audio_state or {}
            seg_outcomes = audio_state.get("outcomes", [])
            for outcome in seg_outcomes:
                outcomes.append(
                    {
                        "segment_id": seg.id,
                        **outcome,
                    }
                )
        return outcomes

    def export(self, transcript: Transcript) -> RAGBundle:
        """
        Export transcript to RAG bundle.

        Args:
            transcript: Transcript to export.

        Returns:
            RAGBundle ready for ingestion.
        """
        # Generate chunks based on strategy
        strategy = self.config.chunking_strategy
        if strategy == ChunkingStrategy.BY_SEGMENT:
            chunks = self._chunk_by_segment(transcript)
        elif strategy == ChunkingStrategy.BY_SPEAKER_TURN:
            chunks = self._chunk_by_speaker_turn(transcript)
        elif strategy == ChunkingStrategy.BY_TIME:
            chunks = self._chunk_by_time(transcript)
        elif strategy == ChunkingStrategy.BY_TOPIC:
            chunks = self._chunk_by_topic(transcript)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        # Generate embeddings if requested
        if self.config.include_embeddings and chunks:
            texts = [c.text for c in chunks]
            embeddings = self._generate_embeddings(texts)
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding

        # Build bundle metadata
        bundle_metadata: dict[str, Any] = {
            "total_chunks": len(chunks),
            "total_duration": transcript.duration,
            "source_file": transcript.file_name,
            "language": transcript.language,
            "speakers": transcript.speaker_ids(),
            "chunking_strategy": strategy.value,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "export_version": "1.0.0",
        }

        if self.config.include_embeddings:
            bundle_metadata["embedding_model"] = self.config.embedding_model

        return RAGBundle(chunks=chunks, metadata=bundle_metadata)

    def export_many(self, transcripts: list[Transcript]) -> list[RAGBundle]:
        """
        Export multiple transcripts to RAG bundles.

        Args:
            transcripts: List of transcripts to export.

        Returns:
            List of RAGBundle objects.
        """
        return [self.export(t) for t in transcripts]


def export_transcript_to_rag(
    transcript: Transcript,
    output_path: Path | str,
    strategy: ChunkingStrategy | str = ChunkingStrategy.BY_SPEAKER_TURN,
    include_embeddings: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> RAGBundle:
    """
    Convenience function to export transcript to RAG bundle.

    Args:
        transcript: Transcript to export.
        output_path: Path to save the bundle.
        strategy: Chunking strategy.
        include_embeddings: Whether to generate embeddings.
        embedding_model: Sentence-transformers model name.

    Returns:
        The exported RAGBundle.
    """
    if isinstance(strategy, str):
        strategy = ChunkingStrategy(strategy)

    config = RAGExporterConfig(
        chunking_strategy=strategy,
        include_embeddings=include_embeddings,
        embedding_model=embedding_model,
    )
    exporter = RAGExporter(config)
    bundle = exporter.export(transcript)
    bundle.save(output_path)
    return bundle
