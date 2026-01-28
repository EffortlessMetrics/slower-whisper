"""Tests for LibriCSS benchmark iterator."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcription.benchmarks import iter_libricss


@pytest.fixture
def libricss_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal LibriCSS directory structure."""
    root = tmp_path / "libricss"
    audio_dir = root / "audio"
    rttm_dir = root / "rttm"
    splits_dir = root / "splits"

    audio_dir.mkdir(parents=True)
    rttm_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    monkeypatch.setenv("SLOWER_WHISPER_BENCHMARKS", str(tmp_path))
    return root


def test_iter_libricss_reads_split_and_rttm(libricss_root: Path) -> None:
    """Iterator loads split list and parses RTTM references."""
    audio_path = libricss_root / "audio" / "meeting_001.wav"
    audio_path.write_bytes(b"")

    rttm_path = libricss_root / "rttm" / "meeting_001.rttm"
    rttm_path.write_text(
        "SPEAKER meeting_001 1 0.00 1.23 <NA> <NA> speaker_0 <NA> <NA>\n"
        "SPEAKER meeting_001 1 1.23 2.00 <NA> <NA> speaker_1 <NA> <NA>\n",
        encoding="utf-8",
    )

    split_file = libricss_root / "splits" / "test.txt"
    split_file.write_text("meeting_001\n", encoding="utf-8")

    samples = list(iter_libricss(split="test"))
    assert len(samples) == 1

    sample = samples[0]
    assert sample.id == "meeting_001"
    assert sample.audio_path == audio_path
    assert sample.reference_speakers is not None
    assert len(sample.reference_speakers) == 2


def test_iter_libricss_missing_root_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Iterator raises when LibriCSS root is missing."""
    monkeypatch.setenv("SLOWER_WHISPER_BENCHMARKS", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        list(iter_libricss())
