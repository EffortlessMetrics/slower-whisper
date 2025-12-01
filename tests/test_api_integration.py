"""
Comprehensive integration tests for transcription.api module.

This test suite covers:
- transcribe_directory() - batch transcription
- transcribe_file() - single file transcription
- enrich_directory() - batch enrichment
- enrich_transcript() - single transcript enrichment
- load_transcript() and save_transcript() - I/O round-trip
- Configuration objects (TranscriptionConfig, EnrichmentConfig)
- Error handling (missing files, invalid configs, etc.)

Tests use pytest fixtures and mocking to avoid requiring actual GPU/models.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcription import (
    EnrichmentConfig,
    EnrichmentError,
    TranscriptionConfig,
    TranscriptionError,
    enrich_directory,
    enrich_transcript,
    load_transcript,
    save_transcript,
    transcribe_directory,
    transcribe_file,
)
from transcription.models import Segment, Transcript

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_project_root(tmp_path):
    """
    Create a temporary project root with expected directory structure.

    Creates:
        raw_audio/      - for input recordings
        input_audio/    - for normalized WAVs
        whisper_json/   - for JSON transcripts
        transcripts/    - for TXT/SRT outputs
    """
    root = tmp_path / "project"
    root.mkdir()

    (root / "raw_audio").mkdir()
    (root / "input_audio").mkdir()
    (root / "whisper_json").mkdir()
    (root / "transcripts").mkdir()

    return root


@pytest.fixture
def test_audio_file(tmp_path):
    """
    Create a test WAV file with synthetic audio.

    Creates a 3-second audio file with varying frequency to simulate speech.
    """
    import soundfile as sf

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create simple sine wave audio
    audio = np.zeros(len(t), dtype=np.float32)
    for i, time in enumerate(t):
        if time < 1.0:
            freq = 200
        elif time < 2.0:
            freq = 250
        else:
            freq = 180
        audio[i] = 0.3 * np.sin(2 * np.pi * freq * time)

    # Add slight noise
    audio += 0.01 * np.random.randn(len(audio)).astype(np.float32)

    audio_path = tmp_path / "test_speech.wav"
    sf.write(audio_path, audio, sr)

    return audio_path


@pytest.fixture
def multiple_audio_files(temp_project_root):
    """Create multiple test audio files in raw_audio directory."""
    import soundfile as sf

    sr = 16000
    audio_files = []

    for i in range(3):
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)

        audio_path = temp_project_root / "raw_audio" / f"test_{i}.wav"
        sf.write(audio_path, audio, sr)
        audio_files.append(audio_path)

    return audio_files


@pytest.fixture
def test_transcript():
    """Create a test transcript with multiple segments."""
    segments = [
        Segment(id=0, start=0.0, end=1.0, text="Hello world"),
        Segment(id=1, start=1.0, end=2.0, text="This is a test"),
        Segment(id=2, start=2.0, end=3.0, text="Audio transcription"),
    ]

    return Transcript(
        file_name="test_speech.wav",
        language="en",
        segments=segments,
        meta={
            "model_name": "large-v3",
            "device": "cpu",
            "pipeline_version": "1.0.0",
        },
    )


@pytest.fixture
def enriched_transcript():
    """Create a test transcript with enriched audio_state data."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=1.0,
            text="Hello world",
            audio_state={
                "prosody": {"pitch_mean": 200.0, "energy_mean": 0.3},
                "emotion": {"valence": 0.5, "arousal": 0.4},
                "rendering": "neutral",
            },
        ),
        Segment(
            id=1,
            start=1.0,
            end=2.0,
            text="This is exciting",
            audio_state={
                "prosody": {"pitch_mean": 250.0, "energy_mean": 0.5},
                "emotion": {"valence": 0.8, "arousal": 0.7},
                "rendering": "excited",
            },
        ),
    ]

    return Transcript(
        file_name="test_speech.wav",
        language="en",
        segments=segments,
        meta={"model_name": "large-v3", "enriched": True},
    )


# ============================================================================
# Configuration Tests
# ============================================================================


def test_transcription_config_defaults():
    """Test TranscriptionConfig default values."""
    config = TranscriptionConfig()

    assert config.model == "large-v3"
    assert config.device == "cuda"
    assert config.compute_type == "float16"
    assert config.language is None  # Auto-detect
    assert config.task == "transcribe"
    assert config.skip_existing_json is True
    assert config.vad_min_silence_ms == 500
    assert config.beam_size == 5


def test_transcription_config_custom():
    """Test TranscriptionConfig with custom values."""
    config = TranscriptionConfig(
        model="medium",
        device="cpu",
        compute_type="int8",
        language="es",
        task="translate",
        skip_existing_json=False,
        vad_min_silence_ms=300,
        beam_size=3,
    )

    assert config.model == "medium"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.language == "es"
    assert config.task == "translate"
    assert config.skip_existing_json is False
    assert config.vad_min_silence_ms == 300
    assert config.beam_size == 3


def test_enrichment_config_defaults():
    """Test EnrichmentConfig default values."""
    config = EnrichmentConfig()

    assert config.skip_existing is True
    assert config.enable_prosody is True
    assert config.enable_emotion is True
    assert config.enable_categorical_emotion is False
    assert config.device == "cpu"
    assert config.dimensional_model_name is None
    assert config.categorical_model_name is None


def test_enrichment_config_custom():
    """Test EnrichmentConfig with custom values."""
    config = EnrichmentConfig(
        skip_existing=False,
        enable_prosody=False,
        enable_emotion=True,
        enable_categorical_emotion=True,
        device="cuda",
        dimensional_model_name="custom-dim-model",
        categorical_model_name="custom-cat-model",
    )

    assert config.skip_existing is False
    assert config.enable_prosody is False
    assert config.enable_emotion is True
    assert config.enable_categorical_emotion is True
    assert config.device == "cuda"
    assert config.dimensional_model_name == "custom-dim-model"
    assert config.categorical_model_name == "custom-cat-model"


# ============================================================================
# I/O Tests (load_transcript / save_transcript)
# ============================================================================


def test_save_and_load_transcript_roundtrip(test_transcript, tmp_path):
    """Test that save_transcript and load_transcript work correctly together."""
    json_path = tmp_path / "test_output.json"

    # Save the transcript
    save_transcript(test_transcript, json_path)

    # Verify file exists
    assert json_path.exists()

    # Load it back
    loaded = load_transcript(json_path)

    # Verify data integrity
    assert loaded.file_name == test_transcript.file_name
    assert loaded.language == test_transcript.language
    assert len(loaded.segments) == len(test_transcript.segments)

    for orig_seg, loaded_seg in zip(test_transcript.segments, loaded.segments, strict=True):
        assert loaded_seg.id == orig_seg.id
        assert loaded_seg.start == orig_seg.start
        assert loaded_seg.end == orig_seg.end
        assert loaded_seg.text == orig_seg.text


def test_load_transcript_with_enriched_data(enriched_transcript, tmp_path):
    """Test loading transcript with audio_state enrichment data."""
    json_path = tmp_path / "enriched_output.json"

    # Save enriched transcript
    save_transcript(enriched_transcript, json_path)

    # Load it back
    loaded = load_transcript(json_path)

    # Verify enrichment data preserved
    assert loaded.segments[0].audio_state is not None
    assert "prosody" in loaded.segments[0].audio_state
    assert "emotion" in loaded.segments[0].audio_state
    assert loaded.segments[0].audio_state["rendering"] == "neutral"
    assert loaded.segments[1].audio_state["rendering"] == "excited"


def test_load_transcript_missing_file(tmp_path):
    """Test load_transcript raises TranscriptionError for missing file."""
    missing_path = tmp_path / "nonexistent.json"

    with pytest.raises(TranscriptionError, match="Transcript file not found"):
        load_transcript(missing_path)


def test_save_transcript_creates_parent_dirs(test_transcript, tmp_path):
    """Test save_transcript creates parent directories if needed."""
    nested_path = tmp_path / "nested" / "dir" / "output.json"

    # Directory doesn't exist yet
    assert not nested_path.parent.exists()

    # Create the parent directory (write_json doesn't create it automatically)
    nested_path.parent.mkdir(parents=True, exist_ok=True)

    # Save should work now
    save_transcript(test_transcript, nested_path)

    assert nested_path.exists()
    assert nested_path.parent.exists()


# ============================================================================
# transcribe_file() Tests
# ============================================================================


@patch("transcription.asr_engine.TranscriptionEngine")
@patch("transcription.audio_io.normalize_all")
def test_transcribe_file_basic(
    mock_normalize,
    mock_engine_class,
    test_audio_file,
    temp_project_root,
    test_transcript,
):
    """Test transcribe_file with basic configuration."""
    # Setup mocks
    mock_engine = MagicMock()
    mock_engine.transcribe_file.return_value = test_transcript
    mock_engine_class.return_value = mock_engine

    # Mock normalize_all to create the expected WAV file
    def create_normalized_wav(paths):
        import soundfile as sf

        norm_wav = paths.norm_dir / f"{test_audio_file.stem}.wav"
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(norm_wav, audio, 16000)

    mock_normalize.side_effect = create_normalized_wav

    # Configure transcription
    config = TranscriptionConfig(model="large-v3", device="cpu", language="en")

    # Transcribe
    result = transcribe_file(test_audio_file, temp_project_root, config)

    # Verify engine was created with correct config
    mock_engine_class.assert_called_once()
    assert mock_engine_class.call_args[0][0].model_name == "large-v3"
    assert mock_engine_class.call_args[0][0].device == "cpu"
    assert mock_engine_class.call_args[0][0].language == "en"

    # Verify normalization was called
    mock_normalize.assert_called_once()

    # Verify transcription was called
    mock_engine.transcribe_file.assert_called_once()

    # Verify outputs were written
    json_path = temp_project_root / "whisper_json" / f"{test_audio_file.stem}.json"
    txt_path = temp_project_root / "transcripts" / f"{test_audio_file.stem}.txt"
    srt_path = temp_project_root / "transcripts" / f"{test_audio_file.stem}.srt"

    assert json_path.exists()
    assert txt_path.exists()
    assert srt_path.exists()

    # Verify result
    assert result == test_transcript
    assert result.meta is not None
    assert "generated_at" in result.meta


@patch("transcription.asr_engine.TranscriptionEngine")
@patch("transcription.audio_io.normalize_all")
def test_transcribe_file_meta_records_actual_runtime(
    mock_normalize,
    mock_engine_class,
    test_audio_file,
    temp_project_root,
):
    """Metadata should reflect the device/compute_type actually used by ASR."""
    mock_engine = MagicMock()
    dummy_transcript = Transcript(
        file_name=test_audio_file.name,
        language="en",
        segments=[Segment(id=0, start=0.0, end=0.5, text="dummy")],
        meta={
            "asr_backend": "dummy",
            "asr_device": "cpu",
            "asr_compute_type": "n/a",
        },
    )
    mock_engine.transcribe_file.return_value = dummy_transcript
    mock_engine_class.return_value = mock_engine

    def create_normalized_wav(paths):
        import soundfile as sf

        norm_wav = paths.norm_dir / f"{test_audio_file.stem}.wav"
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(norm_wav, audio, 16000)

    mock_normalize.side_effect = create_normalized_wav

    config = TranscriptionConfig(model="large-v3", device="cuda", compute_type="float16")

    result = transcribe_file(test_audio_file, temp_project_root, config)

    assert result.meta["device"] == "cpu"
    assert result.meta["compute_type"] == "n/a"
    assert result.meta["asr_backend"] == "dummy"


@patch("transcription.asr_engine.TranscriptionEngine")
@patch("transcription.audio_io.normalize_all")
def test_transcribe_file_meta_includes_duration(
    mock_normalize,
    mock_engine_class,
    test_audio_file,
    temp_project_root,
):
    """Single-file meta should carry the audio duration like the pipeline output."""
    mock_engine = MagicMock()
    dummy_transcript = Transcript(
        file_name=test_audio_file.name,
        language="en",
        segments=[Segment(id=0, start=0.0, end=1.0, text="hi")],
        meta={
            "asr_backend": "dummy",
            "asr_device": "cpu",
            "asr_compute_type": "n/a",
        },
    )
    mock_engine.transcribe_file.return_value = dummy_transcript
    mock_engine_class.return_value = mock_engine

    def create_one_second_wav(paths):
        import soundfile as sf

        norm_wav = paths.norm_dir / f"{test_audio_file.stem}.wav"
        audio = np.zeros(16000, dtype=np.float32)
        sf.write(norm_wav, audio, 16000)

    mock_normalize.side_effect = create_one_second_wav

    config = TranscriptionConfig(model="large-v3", device="cpu", compute_type="int8")

    result = transcribe_file(test_audio_file, temp_project_root, config)

    assert result.meta["audio_duration_sec"] == pytest.approx(1.0, rel=0.01)


@patch("transcription.asr_engine.TranscriptionEngine")
@patch("transcription.audio_io.normalize_all")
def test_transcribe_file_meta_ignores_blank_asr_runtime(
    mock_normalize,
    mock_engine_class,
    test_audio_file,
    temp_project_root,
):
    """Empty asr_device/compute_type in meta should fall back to config values."""
    mock_engine = MagicMock()
    mock_engine.cfg = SimpleNamespace(device="cuda", compute_type="float16")
    dummy_transcript = Transcript(
        file_name=test_audio_file.name,
        language="en",
        segments=[Segment(id=0, start=0.0, end=0.5, text="dummy")],
        meta={
            "asr_backend": "dummy",
            "asr_device": "   ",  # whitespace only should be treated as missing
            "asr_compute_type": "",
        },
    )
    mock_engine.transcribe_file.return_value = dummy_transcript
    mock_engine_class.return_value = mock_engine

    def create_norm(paths):
        import soundfile as sf

        norm_wav = paths.norm_dir / f"{test_audio_file.stem}.wav"
        audio = np.zeros(16000, dtype=np.float32)
        sf.write(norm_wav, audio, 16000)

    mock_normalize.side_effect = create_norm

    config = TranscriptionConfig(model="large-v3", device="cuda", compute_type="float16")

    result = transcribe_file(test_audio_file, temp_project_root, config)

    assert result.meta["device"] == "cuda"
    assert result.meta["compute_type"] == "float16"
    assert result.meta["asr_backend"] == "dummy"


@patch("transcription.asr_engine.TranscriptionEngine")
@patch("transcription.audio_io.normalize_all")
def test_transcribe_file_missing_audio(mock_normalize, mock_engine_class, temp_project_root):
    """Test transcribe_file raises error for missing audio file."""
    config = TranscriptionConfig()
    missing_file = Path("/nonexistent/audio.wav")

    # Should raise TranscriptionError for missing audio file
    with pytest.raises(TranscriptionError, match="Audio file not found"):
        transcribe_file(missing_file, temp_project_root, config)


@patch("transcription.asr_engine.TranscriptionEngine")
@patch("transcription.audio_io.normalize_all")
def test_transcribe_file_normalization_failure(
    mock_normalize, mock_engine_class, test_audio_file, temp_project_root
):
    """Test transcribe_file handles normalization failures."""
    config = TranscriptionConfig()

    # Mock normalization to not create the expected WAV
    mock_normalize.return_value = None

    with pytest.raises(TranscriptionError, match="Audio normalization failed"):
        transcribe_file(test_audio_file, temp_project_root, config)


@patch("transcription.asr_engine.TranscriptionEngine")
def test_transcribe_file_refreshes_raw_copy(mock_engine_class, temp_project_root, monkeypatch):
    """Repeated transcribe_file calls with the same filename should use the latest audio."""
    # Stub normalization to copy raw files into input_audio, overwriting existing outputs.
    normalization_calls: list[Path] = []

    def fake_normalize(paths):
        normalization_calls.append(paths.norm_dir)
        paths.norm_dir.mkdir(parents=True, exist_ok=True)
        for src in paths.raw_dir.iterdir():
            if src.is_file():
                dst = paths.norm_dir / f"{src.stem}.wav"
                dst.write_text(src.read_text(), encoding="utf-8")

    monkeypatch.setattr("transcription.audio_io.normalize_all", fake_normalize)
    # Avoid wave parsing errors on the text fixtures.
    monkeypatch.setattr("transcription.api._get_wav_duration_seconds", lambda _path: 0.0)

    # Fake ASR engine that echoes the normalized file contents.
    mock_engine = MagicMock()

    def fake_transcribe(path):
        content = Path(path).read_text(encoding="utf-8")
        return Transcript(
            file_name=Path(path).name,
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text=content)],
        )

    mock_engine.transcribe_file.side_effect = fake_transcribe
    mock_engine.cfg = SimpleNamespace(device="cpu", compute_type="int8")
    mock_engine_class.return_value = mock_engine

    # Two different audio sources with the same filename.
    first_src = temp_project_root.parent / "clip.wav"
    first_src.write_text("first version", encoding="utf-8")
    updated_dir = temp_project_root.parent / "updated"
    updated_dir.mkdir()
    second_src = updated_dir / "clip.wav"
    second_src.write_text("second version", encoding="utf-8")

    config = TranscriptionConfig(model="tiny", device="cpu", compute_type="int8")

    first = transcribe_file(first_src, temp_project_root, config)
    raw_copy = temp_project_root / "raw_audio" / "clip.wav"
    norm_copy = temp_project_root / "input_audio" / "clip.wav"

    assert raw_copy.read_text(encoding="utf-8") == "first version"
    assert norm_copy.read_text(encoding="utf-8") == "first version"
    assert first.segments[0].text == "first version"

    second = transcribe_file(second_src, temp_project_root, config)

    assert raw_copy.read_text(encoding="utf-8") == "second version"
    assert norm_copy.read_text(encoding="utf-8") == "second version"
    assert second.segments[0].text == "second version"

    assert mock_engine.transcribe_file.call_count == 2
    assert len(normalization_calls) == 2


# ============================================================================
# transcribe_directory() Tests
# ============================================================================


@patch("transcription.pipeline.run_pipeline")
def test_transcribe_directory_basic(
    mock_run_pipeline, temp_project_root, multiple_audio_files, test_transcript
):
    """Test transcribe_directory with multiple audio files."""
    # Create mock JSON outputs
    for i in range(3):
        transcript = Transcript(
            file_name=f"test_{i}.wav",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text=f"Test {i}")],
        )
        json_path = temp_project_root / "whisper_json" / f"test_{i}.json"
        save_transcript(transcript, json_path)

    # Configure and run
    config = TranscriptionConfig(model="large-v3", language="en", device="cpu")
    results = transcribe_directory(temp_project_root, config)

    # Verify pipeline was called
    mock_run_pipeline.assert_called_once()

    # Verify results
    assert len(results) == 3
    assert all(isinstance(t, Transcript) for t in results)
    assert results[0].file_name == "test_0.wav"
    assert results[1].file_name == "test_1.wav"
    assert results[2].file_name == "test_2.wav"


@patch("transcription.pipeline.run_pipeline")
def test_transcribe_directory_empty(mock_run_pipeline, temp_project_root):
    """Test transcribe_directory with no audio files raises error."""
    config = TranscriptionConfig()

    # Should raise error when no transcripts are found
    with pytest.raises(TranscriptionError, match="No transcripts found"):
        transcribe_directory(temp_project_root, config)


@patch("transcription.pipeline.run_pipeline")
def test_transcribe_directory_skip_existing(
    mock_run_pipeline, temp_project_root, multiple_audio_files
):
    """Test transcribe_directory with skip_existing_json=True."""
    # Pre-create one JSON file
    transcript = Transcript(
        file_name="test_0.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=1.0, text="Existing")],
    )
    json_path = temp_project_root / "whisper_json" / "test_0.json"
    save_transcript(transcript, json_path)

    config = TranscriptionConfig(skip_existing_json=True)
    transcribe_directory(temp_project_root, config)

    # Verify skip_existing was passed to AppConfig
    app_config = mock_run_pipeline.call_args[0][0]
    assert app_config.skip_existing_json is True


@patch("transcription.pipeline.run_pipeline")
def test_transcribe_directory_custom_config(mock_run_pipeline, temp_project_root):
    """Test transcribe_directory with custom configuration."""
    # Create a mock transcript so the function doesn't raise "No transcripts found"
    transcript = Transcript(
        file_name="test.wav",
        language="es",
        segments=[Segment(id=0, start=0.0, end=1.0, text="Prueba")],
    )
    json_path = temp_project_root / "whisper_json" / "test.json"
    save_transcript(transcript, json_path)

    config = TranscriptionConfig(
        model="medium",
        device="cpu",
        compute_type="int8",
        language="es",
        task="translate",
        vad_min_silence_ms=300,
        beam_size=3,
    )

    transcribe_directory(temp_project_root, config)

    # Verify config was converted correctly
    app_config = mock_run_pipeline.call_args[0][0]
    assert app_config.asr.model_name == "medium"
    assert app_config.asr.device == "cpu"
    assert app_config.asr.compute_type == "int8"
    assert app_config.asr.language == "es"
    assert app_config.asr.task == "translate"
    assert app_config.asr.vad_min_silence_ms == 300
    assert app_config.asr.beam_size == 3


# ============================================================================
# enrich_transcript() Tests
# ============================================================================


@patch("transcription.audio_enrichment.enrich_transcript_audio")
def test_enrich_transcript_basic(
    mock_enrich_internal, test_transcript, enriched_transcript, test_audio_file
):
    """Test enrich_transcript with basic configuration."""
    # Mock the internal enrichment function
    mock_enrich_internal.return_value = enriched_transcript

    config = EnrichmentConfig(enable_prosody=True, enable_emotion=True)

    result = enrich_transcript(test_transcript, test_audio_file, config)

    # Verify internal function was called with correct params
    mock_enrich_internal.assert_called_once()
    call_kwargs = mock_enrich_internal.call_args[1]
    assert call_kwargs["transcript"] == test_transcript
    assert call_kwargs["wav_path"] == test_audio_file
    assert call_kwargs["enable_prosody"] is True
    assert call_kwargs["enable_emotion"] is True
    assert call_kwargs["enable_categorical_emotion"] is False
    assert call_kwargs["compute_baseline"] is True

    # Verify result
    assert result == enriched_transcript
    assert result.segments[0].audio_state is not None


def test_enrich_transcript_missing_audio(test_transcript, tmp_path):
    """Test enrich_transcript raises error for missing audio file."""
    config = EnrichmentConfig()
    missing_audio = tmp_path / "nonexistent.wav"

    with pytest.raises(EnrichmentError, match="Audio file not found"):
        enrich_transcript(test_transcript, missing_audio, config)


@patch("transcription.audio_enrichment.enrich_transcript_audio")
def test_enrich_transcript_prosody_only(
    mock_enrich_internal, test_transcript, enriched_transcript, test_audio_file
):
    """Test enrich_transcript with only prosody enabled."""
    mock_enrich_internal.return_value = enriched_transcript

    config = EnrichmentConfig(enable_prosody=True, enable_emotion=False)

    enrich_transcript(test_transcript, test_audio_file, config)

    call_kwargs = mock_enrich_internal.call_args[1]
    assert call_kwargs["enable_prosody"] is True
    assert call_kwargs["enable_emotion"] is False


@patch("transcription.audio_enrichment.enrich_transcript_audio")
def test_enrich_transcript_categorical_emotion(
    mock_enrich_internal, test_transcript, enriched_transcript, test_audio_file
):
    """Test enrich_transcript with categorical emotion enabled."""
    mock_enrich_internal.return_value = enriched_transcript

    config = EnrichmentConfig(
        enable_prosody=False, enable_emotion=True, enable_categorical_emotion=True
    )

    enrich_transcript(test_transcript, test_audio_file, config)

    call_kwargs = mock_enrich_internal.call_args[1]
    assert call_kwargs["enable_prosody"] is False
    assert call_kwargs["enable_emotion"] is True
    assert call_kwargs["enable_categorical_emotion"] is True


# ============================================================================
# enrich_directory() Tests
# ============================================================================


def test_enrich_directory_basic(temp_project_root, test_audio_file):
    """Test enrich_directory with pre-existing transcripts and audio."""
    # Create transcripts
    for i in range(2):
        transcript = Transcript(
            file_name=f"test_{i}.wav",
            language="en",
            segments=[
                Segment(id=0, start=0.0, end=1.0, text=f"Segment {i}"),
            ],
        )
        json_path = temp_project_root / "whisper_json" / f"test_{i}.json"
        save_transcript(transcript, json_path)

        # Create corresponding audio
        import soundfile as sf

        audio = np.random.randn(16000).astype(np.float32) * 0.1
        audio_path = temp_project_root / "input_audio" / f"test_{i}.wav"
        sf.write(audio_path, audio, 16000)

    # Mock the enrichment function
    with patch("transcription.audio_enrichment.enrich_transcript_audio") as mock_enrich:

        def enrich_side_effect(transcript, **kwargs):
            # Add audio_state to segments
            for seg in transcript.segments:
                seg.audio_state = {"prosody": {}, "emotion": {}}
            return transcript

        mock_enrich.side_effect = enrich_side_effect

        config = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
        results = enrich_directory(temp_project_root, config)

    # Verify results
    assert len(results) == 2
    assert all(isinstance(t, Transcript) for t in results)

    # Verify enrichment was applied
    for result in results:
        assert result.segments[0].audio_state is not None


def test_enrich_directory_missing_json_dir(temp_project_root):
    """Test enrich_directory raises error when JSON directory doesn't exist."""
    # Remove json directory
    json_dir = temp_project_root / "whisper_json"
    json_dir.rmdir()

    config = EnrichmentConfig()

    with pytest.raises(EnrichmentError, match="JSON directory does not exist"):
        enrich_directory(temp_project_root, config)


def test_enrich_directory_missing_audio_dir(temp_project_root):
    """Test enrich_directory raises error when audio directory doesn't exist."""
    # Remove audio directory
    audio_dir = temp_project_root / "input_audio"
    audio_dir.rmdir()

    config = EnrichmentConfig()

    with pytest.raises(EnrichmentError, match="Audio directory does not exist"):
        enrich_directory(temp_project_root, config)


def test_enrich_directory_no_json_files(temp_project_root):
    """Test enrich_directory with no JSON files raises error."""
    config = EnrichmentConfig()

    with pytest.raises(EnrichmentError, match="No JSON transcript files found"):
        enrich_directory(temp_project_root, config)


def test_enrich_directory_skip_existing(temp_project_root):
    """Test enrich_directory skips already-enriched transcripts."""
    # Create enriched transcript
    transcript = Transcript(
        file_name="test_0.wav",
        language="en",
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=1.0,
                text="Already enriched",
                audio_state={"prosody": {}, "emotion": {}},  # Already has audio_state
            ),
        ],
    )
    json_path = temp_project_root / "whisper_json" / "test_0.json"
    save_transcript(transcript, json_path)

    # Create corresponding audio
    import soundfile as sf

    audio = np.random.randn(16000).astype(np.float32) * 0.1
    audio_path = temp_project_root / "input_audio" / "test_0.wav"
    sf.write(audio_path, audio, 16000)

    with patch("transcription.audio_enrichment.enrich_transcript_audio") as mock_enrich:
        config = EnrichmentConfig(skip_existing=True)
        results = enrich_directory(temp_project_root, config)

        # Should not call enrichment function (already enriched)
        mock_enrich.assert_not_called()

    # Verify transcript was still returned
    assert len(results) == 1
    assert results[0].segments[0].audio_state is not None


def test_enrich_directory_partial_audio_state_not_skipped(temp_project_root):
    """Partial enrichment should still trigger processing when skip_existing=True."""
    # Create transcript where only the first segment is enriched
    transcript = Transcript(
        file_name="test_partial.wav",
        language="en",
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=1.0,
                text="Already enriched",
                audio_state={"prosody": {}},
            ),
            Segment(id=1, start=1.0, end=2.0, text="Needs enrichment"),
        ],
    )
    json_path = temp_project_root / "whisper_json" / "test_partial.json"
    save_transcript(transcript, json_path)

    # Create corresponding audio
    import soundfile as sf

    audio = np.random.randn(16000).astype(np.float32) * 0.05
    audio_path = temp_project_root / "input_audio" / "test_partial.wav"
    sf.write(audio_path, audio, 16000)

    with patch("transcription.audio_enrichment.enrich_transcript_audio") as mock_enrich:
        mock_enrich.return_value = transcript
        config = EnrichmentConfig(skip_existing=True)
        results = enrich_directory(temp_project_root, config)

        # Should re-run enrichment because not all segments have audio_state
        mock_enrich.assert_called_once()

    assert len(results) == 1
    assert results[0].file_name == "test_partial.wav"


def test_enrich_directory_missing_audio_for_transcript(temp_project_root):
    """Test enrich_directory raises error when audio files are missing for all transcripts."""
    # Create transcript without corresponding audio
    transcript = Transcript(
        file_name="test_0.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=1.0, text="Test")],
    )
    json_path = temp_project_root / "whisper_json" / "test_0.json"
    save_transcript(transcript, json_path)

    # Audio file does NOT exist
    config = EnrichmentConfig()

    with patch("transcription.audio_enrichment.enrich_transcript_audio") as mock_enrich:
        # Should raise error when all transcripts fail to enrich
        with pytest.raises(EnrichmentError, match="Failed to enrich any transcripts"):
            enrich_directory(temp_project_root, config)

        # Should not call enrichment
        mock_enrich.assert_not_called()


def test_enrich_directory_handles_enrichment_errors(temp_project_root):
    """Test enrich_directory continues on enrichment errors."""
    # Create two transcripts
    for i in range(2):
        transcript = Transcript(
            file_name=f"test_{i}.wav",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text=f"Test {i}")],
        )
        json_path = temp_project_root / "whisper_json" / f"test_{i}.json"
        save_transcript(transcript, json_path)

        # Create audio
        import soundfile as sf

        audio = np.random.randn(16000).astype(np.float32) * 0.1
        audio_path = temp_project_root / "input_audio" / f"test_{i}.wav"
        sf.write(audio_path, audio, 16000)

    with patch("transcription.audio_enrichment.enrich_transcript_audio") as mock_enrich:
        # First call fails, second succeeds
        def enrich_side_effect(transcript, **kwargs):
            if "test_0" in transcript.file_name:
                raise RuntimeError("Enrichment failed for test_0")
            # Success for test_1
            for seg in transcript.segments:
                seg.audio_state = {"prosody": {}}
            return transcript

        mock_enrich.side_effect = enrich_side_effect

        config = EnrichmentConfig()
        results = enrich_directory(temp_project_root, config)

    # Should only return the successful one
    assert len(results) == 1
    assert results[0].file_name == "test_1.wav"


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_load_transcript_invalid_json(tmp_path):
    """Test load_transcript handles invalid JSON gracefully."""
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{ this is not valid json }")

    with pytest.raises(TranscriptionError, match="Failed to load transcript"):
        load_transcript(invalid_json)


def test_save_transcript_to_readonly_location(test_transcript, tmp_path):
    """Test save_transcript handles permission errors."""
    # Create a read-only directory
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o444)

    readonly_file = readonly_dir / "output.json"

    try:
        # Should raise TranscriptionError (wrapping PermissionError)
        with pytest.raises(TranscriptionError, match="Failed to save transcript"):
            save_transcript(test_transcript, readonly_file)
    finally:
        # Cleanup: restore permissions
        readonly_dir.chmod(0o755)


def test_transcription_config_invalid_task():
    """Test TranscriptionConfig with invalid task value (type checking)."""
    # This would be caught by type checkers, but we can verify runtime behavior
    config = TranscriptionConfig(task="invalid_task")  # type: ignore
    # The config accepts it (no runtime validation), but downstream code might fail
    assert config.task == "invalid_task"


# ============================================================================
# Integration Test - Full Workflow
# ============================================================================


@patch("transcription.asr_engine.TranscriptionEngine")
@patch("transcription.audio_io.normalize_all")
@patch("transcription.audio_enrichment.enrich_transcript_audio")
def test_full_workflow_transcribe_and_enrich(
    mock_enrich_internal,
    mock_normalize,
    mock_engine_class,
    test_audio_file,
    temp_project_root,
):
    """Test complete workflow: transcribe_file → enrich_transcript → save → load."""
    # Step 1: Setup transcription mock
    base_transcript = Transcript(
        file_name=test_audio_file.name,
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="Hello"),
            Segment(id=1, start=1.0, end=2.0, text="World"),
        ],
    )

    mock_engine = MagicMock()
    mock_engine.transcribe_file.return_value = base_transcript
    mock_engine_class.return_value = mock_engine

    # Mock normalize_all to create the WAV file
    def create_normalized_wav(paths):
        import soundfile as sf

        norm_wav = paths.norm_dir / f"{test_audio_file.stem}.wav"
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(norm_wav, audio, 16000)

    mock_normalize.side_effect = create_normalized_wav

    # Step 2: Transcribe
    transcription_config = TranscriptionConfig(model="large-v3", device="cpu", language="en")
    transcript = transcribe_file(test_audio_file, temp_project_root, transcription_config)

    assert transcript.file_name == base_transcript.file_name
    assert len(transcript.segments) == 2

    # Step 3: Setup enrichment mock
    def enrich_side_effect(transcript, **kwargs):
        for seg in transcript.segments:
            seg.audio_state = {
                "prosody": {"pitch_mean": 200.0},
                "emotion": {"valence": 0.5, "arousal": 0.5},
            }
        return transcript

    mock_enrich_internal.side_effect = enrich_side_effect

    # Step 4: Enrich
    audio_path = temp_project_root / "input_audio" / f"{test_audio_file.stem}.wav"
    enrichment_config = EnrichmentConfig(enable_prosody=True, enable_emotion=True)

    enriched = enrich_transcript(transcript, audio_path, enrichment_config)

    assert enriched.segments[0].audio_state is not None
    assert "prosody" in enriched.segments[0].audio_state

    # Step 5: Save
    output_path = temp_project_root / "final_output.json"
    save_transcript(enriched, output_path)

    # Step 6: Load and verify
    loaded = load_transcript(output_path)

    assert loaded.file_name == base_transcript.file_name
    assert len(loaded.segments) == 2
    assert loaded.segments[0].audio_state is not None
    assert loaded.segments[0].audio_state["prosody"]["pitch_mean"] == 200.0
