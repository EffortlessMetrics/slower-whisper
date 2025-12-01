from __future__ import annotations

import hashlib
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

from transcription import samples


def test_get_samples_cache_dir_override(monkeypatch, tmp_path):
    override = tmp_path / "custom_samples"
    monkeypatch.setenv("SLOWER_WHISPER_SAMPLES", str(override))

    assert samples.get_samples_cache_dir() == override


def test_get_samples_cache_dir_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("SLOWER_WHISPER_SAMPLES", raising=False)
    monkeypatch.setattr(
        samples.CachePaths,
        "from_env",
        classmethod(lambda cls, env=None: SimpleNamespace(root=tmp_path / "cache_root")),
    )

    assert samples.get_samples_cache_dir() == tmp_path / "cache_root" / "samples"


def test_verify_sha256(tmp_path):
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"hello-world")
    expected = hashlib.sha256(b"hello-world").hexdigest()

    assert samples.verify_sha256(file_path, expected)
    assert not samples.verify_sha256(file_path, "deadbeef")


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_extract_archive(tmp_path, archive_format):
    src = tmp_path / "src"
    src.mkdir()
    (src / "file.txt").write_text("payload")

    archive_path = tmp_path / f"archive.{archive_format}"
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.write(src / "file.txt", arcname="file.txt")
    else:
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(src / "file.txt", arcname="file.txt")

    dest = tmp_path / "dest"
    samples.extract_archive(archive_path, dest, archive_format, members=["file.txt"])

    assert (dest / "file.txt").read_text() == "payload"


def test_download_sample_dataset_requires_manual_download(tmp_path):
    with pytest.raises(ValueError):
        samples.download_sample_dataset("mini_diarization", cache_dir=tmp_path)


def test_get_sample_test_files_and_copy(monkeypatch, tmp_path):
    monkeypatch.setenv("SLOWER_WHISPER_SAMPLES", str(tmp_path))

    dataset_dir = tmp_path / "mini_diarization" / "dataset" / "test"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    audio_file = dataset_dir / "test.wav"
    audio_file.write_bytes(b"audio-bytes")

    test_files = samples.get_sample_test_files("mini_diarization")
    assert test_files == [audio_file]

    project_raw = tmp_path / "raw_audio"
    copied = samples.copy_sample_to_project("mini_diarization", project_raw)

    assert copied == [project_raw / "mini_diarization_test.wav"]
    assert copied[0].read_bytes() == b"audio-bytes"


def test_list_sample_datasets_contains_metadata():
    available = samples.list_sample_datasets()
    assert "mini_diarization" in available

    meta = available["mini_diarization"]
    assert meta["source_url"]
    assert "dataset/test/test.wav" in meta["test_files"]
