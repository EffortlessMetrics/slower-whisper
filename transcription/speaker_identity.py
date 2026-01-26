"""Speaker identity system for cross-session speaker recognition.

This module provides stable speaker identification across transcription sessions
by extracting and matching speaker embeddings. It enables mapping diarization
speaker IDs (e.g., SPEAKER_00) to stable, user-defined identities.

**Status**: v1.9.3 - NEW

**Components**:
- SpeakerEmbedder: Extract speaker embeddings from audio segments
- SpeakerRegistry: SQLite-backed storage for known speakers
- Identity mapping: Map diarization IDs to stable identities

**Optional Dependencies**:
The embedding extraction requires one of:
- speechbrain (preferred): Install with `pip install speechbrain`
- resemblyzer: Install with `pip install resemblyzer`

Without these dependencies, embedding extraction will raise ImportError.
The registry and CLI commands remain functional for manual identity management.

**Usage**:
    >>> from transcription.speaker_identity import SpeakerEmbedder, SpeakerRegistry
    >>>
    >>> # Register a known speaker
    >>> embedder = SpeakerEmbedder()
    >>> embedding = embedder.extract_embedding("sample.wav", start=0.0, end=5.0)
    >>>
    >>> registry = SpeakerRegistry("speakers.db")
    >>> speaker_id = registry.register_speaker("Alice", embedding)
    >>>
    >>> # Match unknown speaker
    >>> unknown_embedding = embedder.extract_embedding("meeting.wav", start=10.0, end=15.0)
    >>> match = registry.match_speaker(unknown_embedding, threshold=0.7)
    >>> if match:
    ...     print(f"Matched: {match.name}")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from transcription.models import Transcript

logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Backend Detection
# =============================================================================


def _check_speechbrain_available() -> bool:
    """Check if speechbrain is available for embedding extraction."""
    try:
        from speechbrain.inference import EncoderClassifier  # noqa: F401

        return True
    except ImportError:
        return False


def _check_resemblyzer_available() -> bool:
    """Check if resemblyzer is available for embedding extraction."""
    try:
        from resemblyzer import VoiceEncoder  # noqa: F401

        return True
    except ImportError:
        return False


def get_available_backend() -> str | None:
    """Get the first available embedding backend.

    Returns:
        Backend name ('speechbrain', 'resemblyzer') or None if no backend available.
    """
    if _check_speechbrain_available():
        return "speechbrain"
    if _check_resemblyzer_available():
        return "resemblyzer"
    return None


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class Speaker:
    """Represents a registered speaker with identity information.

    Attributes:
        id: Unique speaker identifier (UUID).
        name: Human-readable name for the speaker.
        embedding: Speaker's voice embedding vector.
        metadata: Optional additional metadata about the speaker.
        created_at: Timestamp when the speaker was registered.
        updated_at: Timestamp when the speaker was last updated.
        sample_count: Number of audio samples used to create/update the embedding.
    """

    id: str
    name: str
    embedding: np.ndarray
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    sample_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize speaker to a JSON-serializable dict."""
        return {
            "id": self.id,
            "name": self.name,
            "embedding": self.embedding.tolist(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "sample_count": self.sample_count,
        }


@dataclass
class SpeakerMatch:
    """Result of speaker matching operation.

    Attributes:
        speaker: The matched Speaker object.
        similarity: Cosine similarity score (0.0-1.0).
    """

    speaker: Speaker
    similarity: float


@dataclass
class MappedSegment:
    """A segment with both original diarization ID and mapped identity.

    Attributes:
        segment_index: Index of the segment in the transcript.
        original_speaker_id: Original diarization speaker ID (e.g., "SPEAKER_00").
        mapped_speaker_id: Stable speaker ID from registry (e.g., UUID).
        mapped_speaker_name: Human-readable name of the matched speaker.
        confidence: Matching confidence score (0.0-1.0).
    """

    segment_index: int
    original_speaker_id: str | None
    mapped_speaker_id: str | None
    mapped_speaker_name: str | None
    confidence: float | None


@dataclass
class MappedTranscript:
    """Transcript with identity mapping applied.

    Attributes:
        transcript: The original transcript (unmodified).
        mappings: List of segment-to-identity mappings.
        speaker_id_map: Mapping from original speaker IDs to registry speaker IDs.
        unmapped_speakers: List of original speaker IDs that couldn't be matched.
    """

    transcript: Any  # Transcript type
    mappings: list[MappedSegment]
    speaker_id_map: dict[str, str]  # original_id -> registry_id
    unmapped_speakers: list[str]


# =============================================================================
# Speaker Embedder
# =============================================================================


class SpeakerEmbedder:
    """Extract speaker embeddings from audio segments.

    Uses either speechbrain or resemblyzer for embedding extraction, with
    automatic backend selection based on availability.

    Embedding vectors are normalized and suitable for cosine similarity comparison.

    Args:
        backend: Embedding backend to use ('speechbrain', 'resemblyzer', or 'auto').
                 'auto' selects the first available backend.
        device: Device for inference ('cpu', 'cuda', or 'auto').

    Raises:
        ImportError: If no embedding backend is available.

    Example:
        >>> embedder = SpeakerEmbedder()
        >>> embedding = embedder.extract_embedding("audio.wav", start=0.0, end=5.0)
        >>> print(f"Embedding shape: {embedding.shape}")
    """

    def __init__(
        self,
        backend: str = "auto",
        device: str = "auto",
    ) -> None:
        resolved_backend = backend if backend != "auto" else get_available_backend()
        if resolved_backend is None:
            raise ImportError(
                "No speaker embedding backend available. "
                "Install speechbrain (`pip install speechbrain`) or "
                "resemblyzer (`pip install resemblyzer`)."
            )
        self._backend: str = resolved_backend

        self._device = device
        self._model: Any = None
        self._model_lock = threading.Lock()

    def _ensure_model(self) -> None:
        """Lazy-load embedding model on first use."""
        if self._model is not None:
            return

        with self._model_lock:
            # Double-checked locking pattern for thread safety
            # Another thread may have initialized while we waited for the lock
            if self._model is not None:
                return  # type: ignore[unreachable]

            if self._backend == "speechbrain":
                self._model = self._load_speechbrain_model()
            elif self._backend == "resemblyzer":
                self._model = self._load_resemblyzer_model()
            else:
                raise ValueError(f"Unknown backend: {self._backend}")

    def _load_speechbrain_model(self) -> Any:
        """Load speechbrain ECAPA-TDNN model."""
        from speechbrain.inference import EncoderClassifier

        device = self._resolve_device()

        # Use the ECAPA-TDNN model trained on VoxCeleb
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(self._get_cache_dir() / "speechbrain"),
            run_opts={"device": device},
        )
        return model

    def _load_resemblyzer_model(self) -> Any:
        """Load resemblyzer VoiceEncoder model."""
        from resemblyzer import VoiceEncoder

        device = self._resolve_device()
        use_cuda = device == "cuda"

        encoder = VoiceEncoder(device="cuda" if use_cuda else "cpu")
        return encoder

    def _resolve_device(self) -> str:
        """Resolve device string for model loading."""
        if self._device != "auto":
            return self._device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def _get_cache_dir(self) -> Path:
        """Get cache directory for model files."""
        from transcription.cache import CachePaths

        try:
            paths = CachePaths.from_env().ensure_dirs()
            cache_dir = paths.root / "speaker_embeddings"
        except Exception:
            cache_dir = Path.home() / ".cache" / "slower-whisper" / "speaker_embeddings"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def extract_embedding(
        self,
        audio_path: str | Path,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> np.ndarray:
        """Extract speaker embedding from audio segment.

        Args:
            audio_path: Path to audio file (WAV format recommended).
            start_time: Start time in seconds (None for beginning of file).
            end_time: End time in seconds (None for end of file).

        Returns:
            Normalized embedding vector as numpy array.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            ValueError: If time range is invalid.
            RuntimeError: If embedding extraction fails.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio segment
        waveform, sample_rate = self._load_audio_segment(audio_path, start_time, end_time)

        # Extract embedding based on backend
        self._ensure_model()

        if self._backend == "speechbrain":
            embedding = self._extract_speechbrain(waveform, sample_rate)
        elif self._backend == "resemblyzer":
            embedding = self._extract_resemblyzer(waveform, sample_rate)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        # Normalize embedding
        normalized: np.ndarray = embedding / (np.linalg.norm(embedding) + 1e-8)

        return normalized

    def _load_audio_segment(
        self,
        audio_path: Path,
        start_time: float | None,
        end_time: float | None,
    ) -> tuple[np.ndarray, int]:
        """Load audio segment from file."""
        import soundfile as sf

        # Get file info
        info = sf.info(str(audio_path))
        sample_rate = info.samplerate
        total_frames = info.frames

        # Calculate frame range
        start_frame = 0
        if start_time is not None:
            start_frame = int(start_time * sample_rate)

        end_frame = total_frames
        if end_time is not None:
            end_frame = min(int(end_time * sample_rate), total_frames)

        if start_frame >= end_frame:
            raise ValueError(f"Invalid time range: start={start_time}, end={end_time}")

        # Load audio segment
        with sf.SoundFile(str(audio_path)) as f:
            f.seek(start_frame)
            frames_to_read = end_frame - start_frame
            waveform = f.read(frames_to_read, dtype="float32")

        # Convert stereo to mono if needed
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)

        return waveform, sample_rate

    def _extract_speechbrain(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Extract embedding using speechbrain."""
        import torch

        # Speechbrain expects torch tensor
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)

        # Resample to 16kHz if needed (speechbrain expects 16kHz)
        if sample_rate != 16000:
            import torchaudio

            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform_tensor = resampler(waveform_tensor)

        # Extract embedding
        embedding = self._model.encode_batch(waveform_tensor)
        result: np.ndarray = embedding.squeeze().cpu().numpy()

        return result

    def _extract_resemblyzer(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Extract embedding using resemblyzer."""
        from resemblyzer import preprocess_wav

        # Preprocess (resample to 16kHz, normalize)
        wav = preprocess_wav(waveform, source_sr=sample_rate)

        # Extract embedding
        result: np.ndarray = self._model.embed_utterance(wav)

        return result

    @property
    def backend(self) -> str:
        """Return the active backend name."""
        return self._backend

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality.

        Returns 192 for speechbrain (ECAPA-TDNN) or 256 for resemblyzer.
        """
        if self._backend == "speechbrain":
            return 192
        elif self._backend == "resemblyzer":
            return 256
        else:
            return 192  # Default


# =============================================================================
# Speaker Registry Schema
# =============================================================================


SPEAKER_REGISTRY_SCHEMA = """
-- Speaker Registry Schema
-- Stores speaker embeddings and metadata for cross-session identification

CREATE TABLE IF NOT EXISTS registry_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO registry_meta (key, value) VALUES ('schema_version', '1');
INSERT OR IGNORE INTO registry_meta (key, value) VALUES ('embedding_backend', 'unknown');

CREATE TABLE IF NOT EXISTS speakers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    metadata_json TEXT,
    sample_count INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_speakers_name ON speakers(name);
CREATE INDEX IF NOT EXISTS idx_speakers_created_at ON speakers(created_at);

-- Audit log for speaker changes
CREATE TABLE IF NOT EXISTS speaker_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id TEXT NOT NULL,
    operation TEXT NOT NULL,  -- 'register', 'update', 'delete'
    old_embedding BLOB,
    new_embedding BLOB,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_speaker_history_speaker_id ON speaker_history(speaker_id);
"""


# =============================================================================
# Speaker Registry
# =============================================================================


class SpeakerRegistry:
    """SQLite-backed registry for storing known speaker identities.

    The registry stores speaker embeddings and metadata, enabling cross-session
    speaker identification by matching new audio against known speakers.

    Can optionally share a database with the ConversationStore by using the
    same database path.

    Args:
        db_path: Path to SQLite database file. If None, uses default path.

    Example:
        >>> registry = SpeakerRegistry("speakers.db")
        >>> speaker_id = registry.register_speaker("Alice", embedding)
        >>> match = registry.match_speaker(unknown_embedding, threshold=0.7)
        >>> if match:
        ...     print(f"Matched: {match.speaker.name} ({match.similarity:.2f})")
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path else self._get_default_path()
        self._conn: sqlite3.Connection | None = None
        self._connect()
        self._init_schema()

    def _get_default_path(self) -> Path:
        """Get default registry database path."""
        store_dir = Path.home() / ".slower-whisper"
        store_dir.mkdir(parents=True, exist_ok=True)
        return store_dir / "speaker_registry.db"

    def _connect(self) -> None:
        """Create database connection."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.row_factory = sqlite3.Row

    def _init_schema(self) -> None:
        """Initialize database schema."""
        if self._conn is None:
            raise RuntimeError("Database not connected")

        # Check if schema exists
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='registry_meta'"
        )
        if cursor.fetchone() is None:
            self._conn.executescript(SPEAKER_REGISTRY_SCHEMA)

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._connect()
        assert self._conn is not None  # Guaranteed by _connect() above
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for transactions."""
        if self._conn is None:
            raise RuntimeError("Database not connected")

        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN")
            yield cursor
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

    @property
    def path(self) -> Path:
        """Return database path."""
        return self._path

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SpeakerRegistry:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def register_speaker(
        self,
        name: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a new speaker with their voice embedding.

        Args:
            name: Human-readable name for the speaker.
            embedding: Speaker's voice embedding vector.
            metadata: Optional additional metadata.

        Returns:
            Unique speaker ID (UUID).

        Raises:
            ValueError: If embedding is invalid.
        """
        if embedding.ndim != 1:
            raise ValueError(f"Embedding must be 1D, got shape {embedding.shape}")

        speaker_id = str(uuid.uuid4())
        embedding_bytes = embedding.astype(np.float32).tobytes()
        embedding_dim = len(embedding)
        metadata_json = json.dumps(metadata) if metadata else None

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO speakers (
                    id, name, embedding, embedding_dim, metadata_json, sample_count
                ) VALUES (?, ?, ?, ?, ?, 1)
                """,
                (speaker_id, name, embedding_bytes, embedding_dim, metadata_json),
            )

            # Log operation
            cursor.execute(
                """
                INSERT INTO speaker_history (speaker_id, operation, new_embedding)
                VALUES (?, 'register', ?)
                """,
                (speaker_id, embedding_bytes),
            )

        logger.info(f"Registered speaker '{name}' with ID {speaker_id}")
        return speaker_id

    def match_speaker(
        self,
        embedding: np.ndarray,
        threshold: float = 0.7,
    ) -> SpeakerMatch | None:
        """Find the best matching speaker for an embedding.

        Uses cosine similarity to compare the input embedding against all
        registered speakers and returns the best match if it exceeds the
        threshold.

        Args:
            embedding: Voice embedding to match.
            threshold: Minimum similarity threshold (0.0-1.0).

        Returns:
            SpeakerMatch with the best match, or None if no match above threshold.
        """
        speakers = self.list_speakers()
        if not speakers:
            return None

        # Normalize input embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        best_match: SpeakerMatch | None = None
        best_similarity = threshold

        for speaker in speakers:
            # Normalize stored embedding
            stored_embedding = speaker.embedding / (np.linalg.norm(speaker.embedding) + 1e-8)

            # Compute cosine similarity
            similarity = float(np.dot(embedding, stored_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = SpeakerMatch(speaker=speaker, similarity=similarity)

        return best_match

    def list_speakers(self) -> list[Speaker]:
        """List all registered speakers.

        Returns:
            List of Speaker objects, ordered by creation date.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, name, embedding, embedding_dim, metadata_json,
                   sample_count, created_at, updated_at
            FROM speakers
            ORDER BY created_at DESC
            """
        )

        speakers = []
        for row in cursor.fetchall():
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None

            created_at = None
            if row["created_at"]:
                try:
                    created_at = datetime.fromisoformat(row["created_at"])
                except ValueError:
                    pass

            updated_at = None
            if row["updated_at"]:
                try:
                    updated_at = datetime.fromisoformat(row["updated_at"])
                except ValueError:
                    pass

            speakers.append(
                Speaker(
                    id=row["id"],
                    name=row["name"],
                    embedding=embedding,
                    metadata=metadata,
                    created_at=created_at,
                    updated_at=updated_at,
                    sample_count=row["sample_count"],
                )
            )

        return speakers

    def get_speaker(self, speaker_id: str) -> Speaker | None:
        """Get a speaker by ID.

        Args:
            speaker_id: The speaker's unique ID.

        Returns:
            Speaker object, or None if not found.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, name, embedding, embedding_dim, metadata_json,
                   sample_count, created_at, updated_at
            FROM speakers
            WHERE id = ?
            """,
            (speaker_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None

        created_at = None
        if row["created_at"]:
            try:
                created_at = datetime.fromisoformat(row["created_at"])
            except ValueError:
                pass

        updated_at = None
        if row["updated_at"]:
            try:
                updated_at = datetime.fromisoformat(row["updated_at"])
            except ValueError:
                pass

        return Speaker(
            id=row["id"],
            name=row["name"],
            embedding=embedding,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            sample_count=row["sample_count"],
        )

    def get_speaker_by_name(self, name: str) -> Speaker | None:
        """Get a speaker by name.

        Args:
            name: The speaker's name.

        Returns:
            Speaker object, or None if not found.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, name, embedding, embedding_dim, metadata_json,
                   sample_count, created_at, updated_at
            FROM speakers
            WHERE name = ?
            """,
            (name,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None

        return Speaker(
            id=row["id"],
            name=row["name"],
            embedding=embedding,
            metadata=metadata,
            sample_count=row["sample_count"],
        )

    def update_speaker(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        strategy: str = "average",
    ) -> bool:
        """Update a speaker's embedding with a new sample.

        This allows improving speaker identification accuracy over time by
        incorporating additional voice samples.

        Args:
            speaker_id: The speaker's unique ID.
            embedding: New voice embedding to incorporate.
            strategy: Update strategy:
                      - 'replace': Replace with new embedding
                      - 'average': Running average of all samples (default)

        Returns:
            True if speaker was updated, False if speaker not found.
        """
        speaker = self.get_speaker(speaker_id)
        if speaker is None:
            return False

        if strategy == "replace":
            new_embedding = embedding
            new_sample_count = 1
        elif strategy == "average":
            # Compute running average
            n = speaker.sample_count
            new_embedding = (speaker.embedding * n + embedding) / (n + 1)
            new_sample_count = n + 1
        else:
            raise ValueError(f"Unknown update strategy: {strategy}")

        # Normalize
        new_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
        embedding_bytes = new_embedding.astype(np.float32).tobytes()
        old_embedding_bytes = speaker.embedding.astype(np.float32).tobytes()

        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE speakers
                SET embedding = ?, sample_count = ?, updated_at = datetime('now')
                WHERE id = ?
                """,
                (embedding_bytes, new_sample_count, speaker_id),
            )

            cursor.execute(
                """
                INSERT INTO speaker_history (speaker_id, operation, old_embedding, new_embedding)
                VALUES (?, 'update', ?, ?)
                """,
                (speaker_id, old_embedding_bytes, embedding_bytes),
            )

        logger.info(f"Updated speaker {speaker_id} (samples: {new_sample_count})")
        return True

    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker from the registry.

        Args:
            speaker_id: The speaker's unique ID.

        Returns:
            True if speaker was deleted, False if not found.
        """
        speaker = self.get_speaker(speaker_id)
        if speaker is None:
            return False

        old_embedding_bytes = speaker.embedding.astype(np.float32).tobytes()

        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM speakers WHERE id = ?",
                (speaker_id,),
            )

            cursor.execute(
                """
                INSERT INTO speaker_history (speaker_id, operation, old_embedding)
                VALUES (?, 'delete', ?)
                """,
                (speaker_id, old_embedding_bytes),
            )

        logger.info(f"Deleted speaker {speaker_id}")
        return True

    def stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dict with speaker_count, total_samples, database_size_bytes, etc.
        """
        conn = self._get_conn()

        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as speaker_count,
                COALESCE(SUM(sample_count), 0) as total_samples,
                MIN(created_at) as oldest,
                MAX(updated_at) as newest
            FROM speakers
            """
        )
        row = cursor.fetchone()

        db_size = self._path.stat().st_size if self._path.exists() else 0

        return {
            "speaker_count": row["speaker_count"],
            "total_samples": row["total_samples"],
            "database_size_bytes": db_size,
            "oldest_registration": row["oldest"],
            "newest_update": row["newest"],
        }


# =============================================================================
# Identity Mapping
# =============================================================================


def map_diarization_to_identity(
    transcript: Transcript,
    registry: SpeakerRegistry,
    embedder: SpeakerEmbedder | None = None,
    audio_path: str | Path | None = None,
    threshold: float = 0.7,
) -> MappedTranscript:
    """Map diarization speaker IDs to stable identities from the registry.

    This function takes a transcript with diarization results and maps the
    transient speaker IDs (e.g., SPEAKER_00, SPEAKER_01) to stable identities
    from the speaker registry.

    There are two modes of operation:

    1. **Without embeddings** (embedder=None): Uses segment text and metadata
       to suggest possible matches but requires manual confirmation.

    2. **With embeddings** (embedder and audio_path provided): Extracts voice
       embeddings from representative segments and matches against the registry.

    Args:
        transcript: Transcript with diarization results.
        registry: Speaker registry with known speakers.
        embedder: Optional SpeakerEmbedder for automatic matching.
        audio_path: Path to audio file (required if embedder is provided).
        threshold: Minimum similarity threshold for matching (0.0-1.0).

    Returns:
        MappedTranscript with identity mappings.

    Example:
        >>> embedder = SpeakerEmbedder()
        >>> registry = SpeakerRegistry("speakers.db")
        >>> result = map_diarization_to_identity(
        ...     transcript, registry, embedder, "meeting.wav"
        ... )
        >>> for orig_id, reg_id in result.speaker_id_map.items():
        ...     print(f"{orig_id} -> {reg_id}")
    """
    from transcription.speaker_id import get_speaker_id

    # Collect unique speaker IDs from transcript
    unique_speakers: dict[str, list[int]] = {}  # speaker_id -> segment indices
    for i, seg in enumerate(transcript.segments):
        speaker = getattr(seg, "speaker", None)
        if speaker is not None:
            speaker_id = get_speaker_id(speaker)
            if speaker_id:
                if speaker_id not in unique_speakers:
                    unique_speakers[speaker_id] = []
                unique_speakers[speaker_id].append(i)

    # Match speakers
    speaker_id_map: dict[str, str] = {}  # original -> registry
    speaker_name_map: dict[str, str] = {}  # original -> name
    unmapped_speakers: list[str] = []

    if embedder is not None and audio_path is not None:
        audio_path = Path(audio_path)

        for orig_speaker_id, segment_indices in unique_speakers.items():
            # Get representative segment for this speaker
            # Use the longest segment for better embedding quality
            best_segment = None
            best_duration = 0.0

            for seg_idx in segment_indices:
                seg = transcript.segments[seg_idx]
                duration = seg.end - seg.start
                if duration > best_duration:
                    best_duration = duration
                    best_segment = seg

            if best_segment is None or best_duration < 1.0:
                # Not enough audio for reliable embedding
                unmapped_speakers.append(orig_speaker_id)
                continue

            try:
                # Extract embedding from representative segment
                embedding = embedder.extract_embedding(
                    audio_path,
                    start_time=best_segment.start,
                    end_time=best_segment.end,
                )

                # Match against registry
                match = registry.match_speaker(embedding, threshold=threshold)

                if match is not None:
                    speaker_id_map[orig_speaker_id] = match.speaker.id
                    speaker_name_map[orig_speaker_id] = match.speaker.name
                else:
                    unmapped_speakers.append(orig_speaker_id)

            except Exception as e:
                logger.warning(f"Failed to extract embedding for {orig_speaker_id}: {e}")
                unmapped_speakers.append(orig_speaker_id)
    else:
        # No embedder - all speakers are unmapped
        unmapped_speakers = list(unique_speakers.keys())

    # Build segment mappings
    mappings: list[MappedSegment] = []
    for i, seg in enumerate(transcript.segments):
        speaker = getattr(seg, "speaker", None)
        segment_speaker_id = get_speaker_id(speaker) if speaker else None

        mapped_id = speaker_id_map.get(segment_speaker_id) if segment_speaker_id else None
        mapped_name = speaker_name_map.get(segment_speaker_id) if segment_speaker_id else None

        # Get confidence from original speaker assignment
        confidence = None
        if isinstance(speaker, dict):
            confidence = speaker.get("confidence")

        mappings.append(
            MappedSegment(
                segment_index=i,
                original_speaker_id=segment_speaker_id,
                mapped_speaker_id=mapped_id,
                mapped_speaker_name=mapped_name,
                confidence=confidence,
            )
        )

    return MappedTranscript(
        transcript=transcript,
        mappings=mappings,
        speaker_id_map=speaker_id_map,
        unmapped_speakers=unmapped_speakers,
    )


def apply_identity_mapping(
    transcript: Transcript,
    mapping: MappedTranscript,
    update_segments: bool = True,
    update_speakers: bool = True,
) -> Transcript:
    """Apply identity mapping to a transcript, updating speaker IDs.

    This creates a modified version of the transcript with stable speaker IDs
    from the registry, preserving the original IDs in metadata.

    Args:
        transcript: Original transcript to update.
        mapping: Identity mapping from map_diarization_to_identity().
        update_segments: Whether to update segment.speaker fields.
        update_speakers: Whether to update transcript.speakers list.

    Returns:
        Updated transcript with stable speaker IDs.

    Note:
        This function modifies the transcript in-place and returns it.
    """

    # Build reverse mapping: segment_index -> mapped info
    segment_mapping = {m.segment_index: m for m in mapping.mappings}

    if update_segments:
        for seg in transcript.segments:
            seg_map = segment_mapping.get(seg.id)
            if seg_map and seg_map.mapped_speaker_id:
                seg.speaker = {
                    "id": seg_map.mapped_speaker_id,
                    "name": seg_map.mapped_speaker_name,
                    "original_id": seg_map.original_speaker_id,
                    "confidence": seg_map.confidence,
                }

    if update_speakers and transcript.speakers:
        new_speakers = []
        for speaker in transcript.speakers:
            orig_id = speaker.get("id")
            if orig_id in mapping.speaker_id_map:
                new_speaker = dict(speaker)
                new_speaker["id"] = mapping.speaker_id_map[orig_id]
                new_speaker["original_id"] = orig_id
                # Look up name
                for s in mapping.mappings:
                    if s.original_speaker_id == orig_id and s.mapped_speaker_name:
                        new_speaker["label"] = s.mapped_speaker_name
                        break
                new_speakers.append(new_speaker)
            else:
                new_speakers.append(speaker)
        transcript.speakers = new_speakers

    return transcript


# =============================================================================
# CLI Integration
# =============================================================================


def build_speakers_parser(subparsers: Any) -> None:
    """Build the speakers subcommand parser.

    Adds the following commands:
    - speakers register: Register a new speaker from audio
    - speakers list: List registered speakers
    - speakers match: Match audio against registered speakers
    - speakers apply: Apply identity mapping to transcript
    - speakers delete: Delete a speaker
    - speakers stats: Show registry statistics

    Args:
        subparsers: The subparsers action from the parent parser.
    """
    p_speakers = subparsers.add_parser(
        "speakers",
        help="Manage speaker identities for cross-session recognition.",
    )

    speakers_subparsers = p_speakers.add_subparsers(dest="speakers_action", required=True)

    # Common registry path argument
    def add_registry_arg(parser: Any) -> None:
        parser.add_argument(
            "--registry",
            type=Path,
            default=None,
            help="Path to speaker registry database (default: ~/.slower-whisper/speaker_registry.db)",
        )

    # =========================================================================
    # speakers register
    # =========================================================================
    p_register = speakers_subparsers.add_parser(
        "register",
        help="Register a new speaker from an audio sample.",
    )
    p_register.add_argument(
        "name",
        type=str,
        help="Name to assign to the speaker.",
    )
    p_register.add_argument(
        "--from-clip",
        type=Path,
        required=True,
        dest="audio_file",
        help="Path to audio file containing speaker's voice.",
    )
    p_register.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds (default: beginning of file).",
    )
    p_register.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds (default: end of file).",
    )
    p_register.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="JSON metadata to attach to speaker.",
    )
    add_registry_arg(p_register)

    # =========================================================================
    # speakers list
    # =========================================================================
    p_list = speakers_subparsers.add_parser(
        "list",
        help="List registered speakers.",
    )
    p_list.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )
    add_registry_arg(p_list)

    # =========================================================================
    # speakers match
    # =========================================================================
    p_match = speakers_subparsers.add_parser(
        "match",
        help="Match an audio sample against registered speakers.",
    )
    p_match.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file to match.",
    )
    p_match.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds.",
    )
    p_match.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds.",
    )
    p_match.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum similarity threshold (default: 0.7).",
    )
    p_match.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top matches to show (default: 3).",
    )
    add_registry_arg(p_match)

    # =========================================================================
    # speakers apply
    # =========================================================================
    p_apply = speakers_subparsers.add_parser(
        "apply",
        help="Apply identity mapping to a transcript.",
    )
    p_apply.add_argument(
        "transcript",
        type=Path,
        help="Path to transcript JSON file.",
    )
    p_apply.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Path to audio file (for embedding-based matching).",
    )
    p_apply.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for mapped transcript (default: overwrite input).",
    )
    p_apply.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum similarity threshold (default: 0.7).",
    )
    add_registry_arg(p_apply)

    # =========================================================================
    # speakers delete
    # =========================================================================
    p_delete = speakers_subparsers.add_parser(
        "delete",
        help="Delete a speaker from the registry.",
    )
    p_delete.add_argument(
        "speaker_id",
        type=str,
        help="Speaker ID to delete.",
    )
    p_delete.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    add_registry_arg(p_delete)

    # =========================================================================
    # speakers stats
    # =========================================================================
    p_stats = speakers_subparsers.add_parser(
        "stats",
        help="Show speaker registry statistics.",
    )
    p_stats.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )
    add_registry_arg(p_stats)


def handle_speakers_command(args: Any) -> int:
    """Handle the speakers subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    import sys

    try:
        registry = SpeakerRegistry(args.registry)

        if args.speakers_action == "register":
            return _handle_register(registry, args)
        elif args.speakers_action == "list":
            return _handle_list(registry, args)
        elif args.speakers_action == "match":
            return _handle_match(registry, args)
        elif args.speakers_action == "apply":
            return _handle_apply(registry, args)
        elif args.speakers_action == "delete":
            return _handle_delete(registry, args)
        elif args.speakers_action == "stats":
            return _handle_stats(registry, args)
        else:
            print(f"Unknown speakers action: {args.speakers_action}", file=sys.stderr)
            return 1

    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("\nInstall speaker embedding support with:")
        print("  pip install speechbrain")
        print("  # or")
        print("  pip install resemblyzer")
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        if "registry" in locals():
            registry.close()


def _handle_register(registry: SpeakerRegistry, args: Any) -> int:
    """Handle the register subcommand."""
    import sys

    # Check for embedding support
    backend = get_available_backend()
    if backend is None:
        print("Error: No speaker embedding backend available.", file=sys.stderr)
        print("\nInstall speechbrain or resemblyzer:", file=sys.stderr)
        print("  pip install speechbrain", file=sys.stderr)
        return 1

    # Extract embedding
    embedder = SpeakerEmbedder(backend=backend)
    embedding = embedder.extract_embedding(
        args.audio_file,
        start_time=args.start,
        end_time=args.end,
    )

    # Parse metadata
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"Invalid metadata JSON: {e}", file=sys.stderr)
            return 1

    # Register speaker
    speaker_id = registry.register_speaker(args.name, embedding, metadata)

    print(f"Registered speaker '{args.name}' with ID: {speaker_id}")
    print(f"Embedding backend: {backend}")
    print(f"Embedding dimension: {len(embedding)}")

    return 0


def _handle_list(registry: SpeakerRegistry, args: Any) -> int:
    """Handle the list subcommand."""
    speakers = registry.list_speakers()

    if args.format == "json":
        import json

        print(json.dumps([s.to_dict() for s in speakers], indent=2))
    else:
        if not speakers:
            print("No speakers registered.")
            return 0

        print(f"{'ID':<36} {'Name':<20} {'Samples':<8} {'Created':<20}")
        print("-" * 84)

        for speaker in speakers:
            created = speaker.created_at.strftime("%Y-%m-%d %H:%M") if speaker.created_at else "N/A"
            print(f"{speaker.id:<36} {speaker.name:<20} {speaker.sample_count:<8} {created:<20}")

    return 0


def _handle_match(registry: SpeakerRegistry, args: Any) -> int:
    """Handle the match subcommand."""
    import sys

    backend = get_available_backend()
    if backend is None:
        print("Error: No speaker embedding backend available.", file=sys.stderr)
        return 1

    embedder = SpeakerEmbedder(backend=backend)
    embedding = embedder.extract_embedding(
        args.audio_file,
        start_time=args.start,
        end_time=args.end,
    )

    # Get all speakers and compute similarities
    speakers = registry.list_speakers()
    if not speakers:
        print("No speakers registered in registry.")
        return 0

    # Compute similarities
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    results = []

    for speaker in speakers:
        stored_embedding = speaker.embedding / (np.linalg.norm(speaker.embedding) + 1e-8)
        similarity = float(np.dot(embedding, stored_embedding))
        results.append((speaker, similarity))

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print(f"\nTop {args.top_k} matches (threshold: {args.threshold}):\n")
    print(f"{'Rank':<6} {'Name':<20} {'Similarity':<12} {'Status':<10}")
    print("-" * 48)

    for i, (speaker, similarity) in enumerate(results[: args.top_k]):
        status = "MATCH" if similarity >= args.threshold else "-"
        print(f"{i + 1:<6} {speaker.name:<20} {similarity:.4f}       {status:<10}")

    return 0


def _handle_apply(registry: SpeakerRegistry, args: Any) -> int:
    """Handle the apply subcommand."""
    import sys

    from transcription.writers import load_transcript_from_json, write_json

    # Load transcript
    transcript = load_transcript_from_json(args.transcript)

    # Set up embedder if audio provided
    embedder = None
    if args.audio:
        backend = get_available_backend()
        if backend:
            embedder = SpeakerEmbedder(backend=backend)
        else:
            print("Warning: No embedding backend available, using manual mapping.", file=sys.stderr)

    # Map identities
    mapping = map_diarization_to_identity(
        transcript,
        registry,
        embedder=embedder,
        audio_path=args.audio,
        threshold=args.threshold,
    )

    # Apply mapping
    updated_transcript = apply_identity_mapping(transcript, mapping)

    # Write output
    output_path = args.output or args.transcript
    write_json(updated_transcript, output_path)

    # Report results
    print(f"\nIdentity mapping applied to {args.transcript}")
    print(f"Output written to: {output_path}")
    print(f"\nMapped speakers ({len(mapping.speaker_id_map)}):")
    for orig_id, reg_id in mapping.speaker_id_map.items():
        # Find name
        name = "Unknown"
        for m in mapping.mappings:
            if m.original_speaker_id == orig_id and m.mapped_speaker_name:
                name = m.mapped_speaker_name
                break
        print(f"  {orig_id} -> {name} ({reg_id[:8]}...)")

    if mapping.unmapped_speakers:
        print(f"\nUnmapped speakers ({len(mapping.unmapped_speakers)}):")
        for speaker_id in mapping.unmapped_speakers:
            print(f"  {speaker_id}")

    return 0


def _handle_delete(registry: SpeakerRegistry, args: Any) -> int:
    """Handle the delete subcommand."""
    import sys

    # Check if speaker exists
    speaker = registry.get_speaker(args.speaker_id)
    if speaker is None:
        print(f"Speaker not found: {args.speaker_id}", file=sys.stderr)
        return 1

    # Confirm deletion
    if not args.force:
        if not sys.stdin.isatty():
            print("Error: Delete requires --force in non-interactive mode.", file=sys.stderr)
            return 1

        confirm = input(f"Delete speaker '{speaker.name}' ({speaker.id})? [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Aborted.")
            return 0

    # Delete speaker
    registry.delete_speaker(args.speaker_id)
    print(f"Deleted speaker '{speaker.name}' ({speaker.id})")

    return 0


def _handle_stats(registry: SpeakerRegistry, args: Any) -> int:
    """Handle the stats subcommand."""
    stats = registry.stats()

    if args.format == "json":
        import json

        print(json.dumps(stats, indent=2))
    else:
        print("Speaker Registry Statistics")
        print("=" * 40)
        print(f"Total speakers:       {stats['speaker_count']:,}")
        print(f"Total samples:        {stats['total_samples']:,}")
        print(f"Database size:        {stats['database_size_bytes'] / 1024:.1f} KB")
        if stats["oldest_registration"]:
            print(f"Oldest registration:  {stats['oldest_registration'][:16]}")
        if stats["newest_update"]:
            print(f"Latest update:        {stats['newest_update'][:16]}")

    return 0


__all__ = [
    # Data types
    "Speaker",
    "SpeakerMatch",
    "MappedSegment",
    "MappedTranscript",
    # Embedder
    "SpeakerEmbedder",
    "get_available_backend",
    # Registry
    "SpeakerRegistry",
    # Mapping functions
    "map_diarization_to_identity",
    "apply_identity_mapping",
    # CLI
    "build_speakers_parser",
    "handle_speakers_command",
]
