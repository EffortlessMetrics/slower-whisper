"""Tests for speaker identity system.

Tests cover:
- SpeakerEmbedder: Embedding extraction (mocked if no backend)
- SpeakerRegistry: CRUD operations
- Identity matching with threshold
- Diarization-to-identity mapping
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcription.speaker_identity import (
    MappedSegment,
    MappedTranscript,
    Speaker,
    SpeakerEmbedder,
    SpeakerMatch,
    SpeakerRegistry,
    apply_identity_mapping,
    get_available_backend,
    map_diarization_to_identity,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_speakers.db"


@pytest.fixture
def registry(temp_db: Path) -> SpeakerRegistry:
    """Create a SpeakerRegistry with temporary database."""
    reg = SpeakerRegistry(temp_db)
    yield reg
    reg.close()


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Create a sample normalized embedding."""
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(192).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def sample_embeddings() -> list[np.ndarray]:
    """Create multiple sample embeddings with known similarity."""
    rng = np.random.default_rng(42)
    base = rng.standard_normal(192).astype(np.float32)
    base = base / np.linalg.norm(base)

    # Create similar embedding (high similarity) - use smaller noise
    # With 0.1 * random noise, similarity ~0.7-0.8
    noise = rng.standard_normal(192).astype(np.float32) * 0.05
    similar = base + noise
    similar = similar / np.linalg.norm(similar)

    # Create different embedding (low similarity)
    different = rng.standard_normal(192).astype(np.float32)
    different = different / np.linalg.norm(different)

    return [base, similar, different]


@pytest.fixture
def mock_transcript():
    """Create a mock transcript with diarization results."""
    from transcription.models import Segment, Transcript

    segments = [
        Segment(
            id=0,
            start=0.0,
            end=5.0,
            text="Hello, how are you?",
            speaker={"id": "spk_0", "confidence": 0.9},
        ),
        Segment(
            id=1,
            start=5.0,
            end=10.0,
            text="I'm doing well, thanks.",
            speaker={"id": "spk_1", "confidence": 0.85},
        ),
        Segment(
            id=2,
            start=10.0,
            end=15.0,
            text="That's great to hear.",
            speaker={"id": "spk_0", "confidence": 0.88},
        ),
        Segment(
            id=3,
            start=15.0,
            end=20.0,
            text="Yes, it is.",
            speaker={"id": "spk_1", "confidence": 0.92},
        ),
    ]

    return Transcript(
        file_name="test_meeting.wav",
        language="en",
        segments=segments,
        speakers=[
            {"id": "spk_0", "label": None, "total_speech_time": 10.0, "num_segments": 2},
            {"id": "spk_1", "label": None, "total_speech_time": 10.0, "num_segments": 2},
        ],
    )


# =============================================================================
# Backend Detection Tests
# =============================================================================


class TestBackendDetection:
    """Tests for embedding backend detection."""

    def test_get_available_backend_returns_string_or_none(self):
        """Backend detection should return a string or None."""
        backend = get_available_backend()
        assert backend is None or isinstance(backend, str)

    def test_get_available_backend_valid_backends(self):
        """If a backend is available, it should be a known backend."""
        backend = get_available_backend()
        if backend is not None:
            assert backend in ("speechbrain", "resemblyzer")


# =============================================================================
# SpeakerRegistry Tests
# =============================================================================


class TestSpeakerRegistry:
    """Tests for SpeakerRegistry CRUD operations."""

    def test_registry_creates_database(self, temp_db: Path):
        """Registry should create database file."""
        registry = SpeakerRegistry(temp_db)
        try:
            assert temp_db.exists()
        finally:
            registry.close()

    def test_registry_creates_schema(self, temp_db: Path):
        """Registry should create required tables."""
        registry = SpeakerRegistry(temp_db)
        try:
            conn = sqlite3.connect(str(temp_db))
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            conn.close()

            assert "speakers" in tables
            assert "registry_meta" in tables
            assert "speaker_history" in tables
        finally:
            registry.close()

    def test_register_speaker(self, registry: SpeakerRegistry, sample_embedding: np.ndarray):
        """Should register a new speaker and return ID."""
        speaker_id = registry.register_speaker("Alice", sample_embedding)

        assert speaker_id is not None
        assert len(speaker_id) == 36  # UUID format

    def test_register_speaker_with_metadata(
        self, registry: SpeakerRegistry, sample_embedding: np.ndarray
    ):
        """Should store metadata with speaker."""
        metadata = {"department": "Engineering", "role": "Developer"}
        speaker_id = registry.register_speaker("Bob", sample_embedding, metadata)

        speaker = registry.get_speaker(speaker_id)
        assert speaker is not None
        assert speaker.metadata == metadata

    def test_get_speaker_by_id(self, registry: SpeakerRegistry, sample_embedding: np.ndarray):
        """Should retrieve speaker by ID."""
        speaker_id = registry.register_speaker("Charlie", sample_embedding)

        speaker = registry.get_speaker(speaker_id)

        assert speaker is not None
        assert speaker.id == speaker_id
        assert speaker.name == "Charlie"
        assert np.allclose(speaker.embedding, sample_embedding, atol=1e-6)

    def test_get_speaker_by_name(self, registry: SpeakerRegistry, sample_embedding: np.ndarray):
        """Should retrieve speaker by name."""
        registry.register_speaker("Diana", sample_embedding)

        speaker = registry.get_speaker_by_name("Diana")

        assert speaker is not None
        assert speaker.name == "Diana"

    def test_get_nonexistent_speaker(self, registry: SpeakerRegistry):
        """Should return None for nonexistent speaker."""
        speaker = registry.get_speaker("nonexistent-id")
        assert speaker is None

    def test_list_speakers_empty(self, registry: SpeakerRegistry):
        """Should return empty list when no speakers registered."""
        speakers = registry.list_speakers()
        assert speakers == []

    def test_list_speakers(self, registry: SpeakerRegistry, sample_embedding: np.ndarray):
        """Should list all registered speakers."""
        registry.register_speaker("Eve", sample_embedding)
        registry.register_speaker("Frank", sample_embedding)
        registry.register_speaker("Grace", sample_embedding)

        speakers = registry.list_speakers()

        assert len(speakers) == 3
        names = {s.name for s in speakers}
        assert names == {"Eve", "Frank", "Grace"}

    def test_update_speaker_replace(
        self, registry: SpeakerRegistry, sample_embeddings: list[np.ndarray]
    ):
        """Should replace embedding when strategy is 'replace'."""
        old_embedding, new_embedding, _ = sample_embeddings
        speaker_id = registry.register_speaker("Henry", old_embedding)

        result = registry.update_speaker(speaker_id, new_embedding, strategy="replace")

        assert result is True
        speaker = registry.get_speaker(speaker_id)
        assert speaker is not None
        assert np.allclose(
            speaker.embedding, new_embedding / np.linalg.norm(new_embedding), atol=1e-6
        )
        assert speaker.sample_count == 1

    def test_update_speaker_average(
        self, registry: SpeakerRegistry, sample_embeddings: list[np.ndarray]
    ):
        """Should average embedding when strategy is 'average'."""
        old_embedding, new_embedding, _ = sample_embeddings
        speaker_id = registry.register_speaker("Ivy", old_embedding)

        result = registry.update_speaker(speaker_id, new_embedding, strategy="average")

        assert result is True
        speaker = registry.get_speaker(speaker_id)
        assert speaker is not None
        assert speaker.sample_count == 2

    def test_update_nonexistent_speaker(
        self, registry: SpeakerRegistry, sample_embedding: np.ndarray
    ):
        """Should return False when updating nonexistent speaker."""
        result = registry.update_speaker("nonexistent-id", sample_embedding)
        assert result is False

    def test_delete_speaker(self, registry: SpeakerRegistry, sample_embedding: np.ndarray):
        """Should delete speaker from registry."""
        speaker_id = registry.register_speaker("Jack", sample_embedding)

        result = registry.delete_speaker(speaker_id)

        assert result is True
        assert registry.get_speaker(speaker_id) is None

    def test_delete_nonexistent_speaker(self, registry: SpeakerRegistry):
        """Should return False when deleting nonexistent speaker."""
        result = registry.delete_speaker("nonexistent-id")
        assert result is False

    def test_stats(self, registry: SpeakerRegistry, sample_embedding: np.ndarray):
        """Should return registry statistics."""
        registry.register_speaker("Kate", sample_embedding)
        registry.register_speaker("Leo", sample_embedding)

        stats = registry.stats()

        assert stats["speaker_count"] == 2
        assert stats["total_samples"] == 2
        assert stats["database_size_bytes"] > 0

    def test_context_manager(self, temp_db: Path, sample_embedding: np.ndarray):
        """Should work as context manager."""
        with SpeakerRegistry(temp_db) as registry:
            speaker_id = registry.register_speaker("Mike", sample_embedding)
            assert speaker_id is not None

    def test_invalid_embedding_shape(self, registry: SpeakerRegistry):
        """Should reject 2D embeddings."""
        embedding_2d = np.random.randn(2, 192).astype(np.float32)

        with pytest.raises(ValueError, match="1D"):
            registry.register_speaker("Invalid", embedding_2d)


# =============================================================================
# Speaker Matching Tests
# =============================================================================


class TestSpeakerMatching:
    """Tests for speaker matching functionality."""

    def test_match_speaker_exact(self, registry: SpeakerRegistry, sample_embedding: np.ndarray):
        """Should match speaker with same embedding (similarity ~1.0)."""
        speaker_id = registry.register_speaker("Nancy", sample_embedding)

        match = registry.match_speaker(sample_embedding, threshold=0.9)

        assert match is not None
        assert match.speaker.id == speaker_id
        assert match.similarity > 0.99  # Should be ~1.0

    def test_match_speaker_similar(
        self, registry: SpeakerRegistry, sample_embeddings: list[np.ndarray]
    ):
        """Should match speaker with similar embedding."""
        base, similar, _ = sample_embeddings
        speaker_id = registry.register_speaker("Oliver", base)

        match = registry.match_speaker(similar, threshold=0.7)

        assert match is not None
        assert match.speaker.id == speaker_id
        assert match.similarity > 0.7

    def test_match_speaker_no_match(
        self, registry: SpeakerRegistry, sample_embeddings: list[np.ndarray]
    ):
        """Should return None when no speaker matches above threshold."""
        base, _, different = sample_embeddings
        registry.register_speaker("Patricia", base)

        # Different embedding should not match with high threshold
        match = registry.match_speaker(different, threshold=0.9)

        assert match is None

    def test_match_speaker_empty_registry(
        self, registry: SpeakerRegistry, sample_embedding: np.ndarray
    ):
        """Should return None when registry is empty."""
        match = registry.match_speaker(sample_embedding, threshold=0.5)
        assert match is None

    def test_match_speaker_multiple_candidates(
        self, registry: SpeakerRegistry, sample_embeddings: list[np.ndarray]
    ):
        """Should return best match among multiple speakers."""
        base, similar, different = sample_embeddings

        registry.register_speaker("Quinn", base)
        registry.register_speaker("Rachel", different)

        match = registry.match_speaker(similar, threshold=0.5)

        assert match is not None
        assert match.speaker.name == "Quinn"

    def test_match_speaker_threshold_boundary(
        self, registry: SpeakerRegistry, sample_embeddings: list[np.ndarray]
    ):
        """Should respect threshold exactly."""
        base, similar, _ = sample_embeddings
        registry.register_speaker("Steve", base)

        # Get actual similarity
        match_low = registry.match_speaker(similar, threshold=0.5)
        assert match_low is not None

        similarity = match_low.similarity

        # Match with threshold just above should fail
        match_high = registry.match_speaker(similar, threshold=similarity + 0.01)
        assert match_high is None


# =============================================================================
# SpeakerEmbedder Tests
# =============================================================================


class TestSpeakerEmbedder:
    """Tests for SpeakerEmbedder (mocked if no backend available)."""

    def test_embedder_raises_without_backend(self):
        """Should raise ImportError if no backend available."""
        with patch("transcription.speaker_identity.get_available_backend", return_value=None):
            with pytest.raises(ImportError, match="No speaker embedding backend"):
                SpeakerEmbedder()

    def test_embedder_selects_available_backend(self):
        """Should use available backend when auto selected."""
        backend = get_available_backend()
        if backend is None:
            pytest.skip("No embedding backend available")

        embedder = SpeakerEmbedder(backend="auto")
        assert embedder.backend == backend

    def test_embedder_backend_property(self):
        """Should expose backend name."""
        backend = get_available_backend()
        if backend is None:
            pytest.skip("No embedding backend available")

        embedder = SpeakerEmbedder(backend=backend)
        assert embedder.backend == backend

    def test_embedder_embedding_dim(self):
        """Should report correct embedding dimension."""
        backend = get_available_backend()
        if backend is None:
            pytest.skip("No embedding backend available")

        embedder = SpeakerEmbedder(backend=backend)

        if backend == "speechbrain":
            assert embedder.embedding_dim == 192
        elif backend == "resemblyzer":
            assert embedder.embedding_dim == 256

    def test_embedder_file_not_found(self):
        """Should raise FileNotFoundError for missing audio."""
        backend = get_available_backend()
        if backend is None:
            pytest.skip("No embedding backend available")

        embedder = SpeakerEmbedder(backend=backend)

        with pytest.raises(FileNotFoundError):
            embedder.extract_embedding("/nonexistent/audio.wav")

    def test_embedder_invalid_time_range(self, tmp_path: Path):
        """Should raise ValueError for invalid time range."""
        backend = get_available_backend()
        if backend is None:
            pytest.skip("No embedding backend available")

        # Create minimal WAV file
        import soundfile as sf

        audio_path = tmp_path / "test.wav"
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        sf.write(str(audio_path), audio, 16000)

        embedder = SpeakerEmbedder(backend=backend)

        with pytest.raises(ValueError, match="Invalid time range"):
            embedder.extract_embedding(audio_path, start_time=5.0, end_time=2.0)


class TestSpeakerEmbedderMocked:
    """Mocked tests for SpeakerEmbedder that don't require actual backends."""

    def test_extract_embedding_normalized(self):
        """Extracted embeddings should be normalized."""
        mock_raw_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with patch(
            "transcription.speaker_identity.get_available_backend", return_value="speechbrain"
        ):
            embedder = SpeakerEmbedder(backend="speechbrain")

            # Mock internal methods
            embedder._ensure_model = MagicMock()
            embedder._load_audio_segment = MagicMock(return_value=(np.random.randn(16000), 16000))
            embedder._extract_speechbrain = MagicMock(return_value=mock_raw_embedding)

            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                Path(f.name).touch()
                embedding = embedder.extract_embedding(f.name)

                assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


# =============================================================================
# Identity Mapping Tests
# =============================================================================


class TestIdentityMapping:
    """Tests for diarization-to-identity mapping."""

    def test_map_without_embedder(self, registry: SpeakerRegistry, mock_transcript):
        """Without embedder, all speakers should be unmapped."""
        result = map_diarization_to_identity(mock_transcript, registry)

        assert isinstance(result, MappedTranscript)
        assert result.speaker_id_map == {}
        assert set(result.unmapped_speakers) == {"spk_0", "spk_1"}

    def test_map_preserves_transcript(self, registry: SpeakerRegistry, mock_transcript):
        """Mapping should preserve original transcript."""
        result = map_diarization_to_identity(mock_transcript, registry)

        assert result.transcript is mock_transcript

    def test_map_creates_segment_mappings(self, registry: SpeakerRegistry, mock_transcript):
        """Should create mapping for each segment."""
        result = map_diarization_to_identity(mock_transcript, registry)

        assert len(result.mappings) == len(mock_transcript.segments)

        for mapping in result.mappings:
            assert isinstance(mapping, MappedSegment)
            assert mapping.segment_index >= 0

    def test_apply_identity_mapping_updates_segments(
        self, registry: SpeakerRegistry, mock_transcript, sample_embedding: np.ndarray
    ):
        """Should update segment speaker IDs after mapping."""
        # Register a speaker
        speaker_id = registry.register_speaker("TestUser", sample_embedding)

        # Create a mapping manually (simulating successful embedding match)
        mapping = MappedTranscript(
            transcript=mock_transcript,
            mappings=[
                MappedSegment(0, "spk_0", speaker_id, "TestUser", 0.9),
                MappedSegment(1, "spk_1", None, None, 0.85),
                MappedSegment(2, "spk_0", speaker_id, "TestUser", 0.88),
                MappedSegment(3, "spk_1", None, None, 0.92),
            ],
            speaker_id_map={"spk_0": speaker_id},
            unmapped_speakers=["spk_1"],
        )

        # Apply mapping
        updated = apply_identity_mapping(mock_transcript, mapping)

        # Check segment 0 was updated
        assert updated.segments[0].speaker["id"] == speaker_id
        assert updated.segments[0].speaker["name"] == "TestUser"
        assert updated.segments[0].speaker["original_id"] == "spk_0"

    def test_apply_identity_mapping_updates_speakers_list(
        self, registry: SpeakerRegistry, mock_transcript, sample_embedding: np.ndarray
    ):
        """Should update transcript.speakers list."""
        speaker_id = registry.register_speaker("TestUser", sample_embedding)

        mapping = MappedTranscript(
            transcript=mock_transcript,
            mappings=[
                MappedSegment(0, "spk_0", speaker_id, "TestUser", 0.9),
                MappedSegment(1, "spk_1", None, None, 0.85),
                MappedSegment(2, "spk_0", speaker_id, "TestUser", 0.88),
                MappedSegment(3, "spk_1", None, None, 0.92),
            ],
            speaker_id_map={"spk_0": speaker_id},
            unmapped_speakers=["spk_1"],
        )

        updated = apply_identity_mapping(mock_transcript, mapping)

        # Find the mapped speaker in speakers list
        mapped_speaker = next((s for s in updated.speakers if s.get("id") == speaker_id), None)
        assert mapped_speaker is not None
        assert mapped_speaker.get("original_id") == "spk_0"


# =============================================================================
# Data Type Tests
# =============================================================================


class TestDataTypes:
    """Tests for data type serialization."""

    def test_speaker_to_dict(self, sample_embedding: np.ndarray):
        """Speaker should serialize to dict."""
        speaker = Speaker(
            id="test-id",
            name="Test",
            embedding=sample_embedding,
            metadata={"key": "value"},
        )

        d = speaker.to_dict()

        assert d["id"] == "test-id"
        assert d["name"] == "Test"
        assert isinstance(d["embedding"], list)
        assert d["metadata"] == {"key": "value"}

    def test_speaker_match_contains_similarity(self, sample_embedding: np.ndarray):
        """SpeakerMatch should contain similarity score."""
        speaker = Speaker(id="test", name="Test", embedding=sample_embedding)
        match = SpeakerMatch(speaker=speaker, similarity=0.95)

        assert match.similarity == 0.95
        assert match.speaker.name == "Test"

    def test_mapped_segment_fields(self):
        """MappedSegment should have all required fields."""
        mapping = MappedSegment(
            segment_index=0,
            original_speaker_id="spk_0",
            mapped_speaker_id="uuid-123",
            mapped_speaker_name="Alice",
            confidence=0.9,
        )

        assert mapping.segment_index == 0
        assert mapping.original_speaker_id == "spk_0"
        assert mapping.mapped_speaker_id == "uuid-123"
        assert mapping.mapped_speaker_name == "Alice"
        assert mapping.confidence == 0.9


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestSpeakerIdentityIntegration:
    """Integration tests that require actual audio files and models."""

    @pytest.fixture
    def audio_file(self, tmp_path: Path) -> Path:
        """Create a test audio file."""
        import soundfile as sf

        audio_path = tmp_path / "test_audio.wav"
        # Generate 5 seconds of audio at 16kHz
        audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
        sf.write(str(audio_path), audio, 16000)
        return audio_path

    def test_full_workflow(self, temp_db: Path, audio_file: Path):
        """Test complete register -> match workflow."""
        backend = get_available_backend()
        if backend is None:
            pytest.skip("No embedding backend available")

        # Create embedder and registry
        embedder = SpeakerEmbedder(backend=backend)
        registry = SpeakerRegistry(temp_db)

        try:
            # Extract embedding and register speaker
            embedding = embedder.extract_embedding(audio_file, start_time=0.0, end_time=3.0)
            speaker_id = registry.register_speaker("TestSpeaker", embedding)

            # Extract another embedding from same file (should match)
            test_embedding = embedder.extract_embedding(audio_file, start_time=1.0, end_time=4.0)

            # Match should succeed (same audio source)
            match = registry.match_speaker(test_embedding, threshold=0.5)

            assert match is not None
            assert match.speaker.id == speaker_id
            assert match.similarity > 0.5

        finally:
            registry.close()
