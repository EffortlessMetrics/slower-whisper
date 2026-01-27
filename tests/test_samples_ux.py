from __future__ import annotations
import pytest
from transcription import samples
from transcription.exceptions import SampleExistsError

def test_copy_sample_overwrite_behavior(monkeypatch, tmp_path):
    monkeypatch.setenv("SLOWER_WHISPER_SAMPLES", str(tmp_path))

    dataset_dir = tmp_path / "mini_diarization" / "dataset" / "test"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    audio_file = dataset_dir / "test.wav"
    audio_file.write_bytes(b"original-content")

    project_raw = tmp_path / "raw_audio"
    project_raw.mkdir(parents=True)

    # Pre-existing file with different content
    existing_file = project_raw / "mini_diarization_test.wav"
    existing_file.write_bytes(b"user-modified-content")

    # This should now raise SampleExistsError
    with pytest.raises(SampleExistsError) as excinfo:
        samples.copy_sample_to_project("mini_diarization", project_raw)

    # Verify exception details
    assert existing_file in excinfo.value.existing_files

    # Verify content was NOT overwritten
    assert existing_file.read_bytes() == b"user-modified-content"

    # Now retry with force=True
    copied = samples.copy_sample_to_project("mini_diarization", project_raw, force=True)

    assert copied == [existing_file]
    # Check if content was overwritten
    assert existing_file.read_bytes() == b"original-content"
