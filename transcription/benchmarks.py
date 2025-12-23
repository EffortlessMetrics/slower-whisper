"""Benchmark dataset management for evaluation and quality measurement.

This module provides infrastructure for working with public benchmark datasets
like AMI Meeting Corpus, IEMOCAP, LibriCSS, etc. for evaluating:
- Speaker diarization quality (DER)
- Emotion recognition accuracy
- LLM-based task quality (summaries, action items, QA)

Datasets are expected to be manually staged under:
    $SLOWER_WHISPER_CACHE_ROOT/benchmarks (default: ~/.cache/slower-whisper/benchmarks)

Each dataset should be placed in its own subdirectory with a documented structure.
See docs/AMI_SETUP.md, docs/IEMOCAP_SETUP.md for setup instructions.

Environment variables respected:
- SLOWER_WHISPER_CACHE_ROOT: Root cache directory
- SLOWER_WHISPER_BENCHMARKS: Override benchmarks cache location
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcription.cache import CachePaths

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample from a benchmark dataset.

    Attributes:
        dataset: Dataset name (e.g., "ami", "iemocap", "libricss")
        id: Unique sample identifier within the dataset
        audio_path: Path to audio file
        reference_transcript: Ground truth transcript text (optional)
        reference_summary: Ground truth summary (optional)
        reference_speakers: Ground truth speaker diarization (optional)
            Format: list of dicts with keys: speaker_id, start, end
        reference_emotions: Ground truth emotion labels (optional)
            Format: list of emotion labels per segment
        metadata: Additional dataset-specific metadata
    """

    dataset: str
    id: str
    audio_path: Path
    reference_transcript: str | None = None
    reference_summary: str | None = None
    reference_speakers: list[dict[str, Any]] | None = None
    reference_emotions: list[str] | None = None
    metadata: dict[str, Any] | None = None


def get_benchmarks_root() -> Path:
    """Get the benchmarks cache directory.

    Returns:
        Path to benchmarks cache directory

    Example:
        >>> root = get_benchmarks_root()
        >>> ami_dir = root / "ami"
    """
    # Respect SLOWER_WHISPER_BENCHMARKS if set, otherwise use cache root
    if benchmarks_override := os.environ.get("SLOWER_WHISPER_BENCHMARKS"):
        return Path(benchmarks_override).expanduser()

    paths = CachePaths.from_env()
    return paths.benchmarks_root


def iter_ami_meetings(
    split: str = "test",
    limit: int | None = None,
    require_summary: bool = False,
) -> Iterable[EvalSample]:
    """Iterate over AMI Meeting Corpus samples.

    The AMI corpus should be manually staged under benchmarks_root/ami/ following
    the structure documented in docs/AMI_SETUP.md.

    Expected structure:
        benchmarks_root/ami/
            audio/
                ES2002a.Mix-Headset.wav
                ES2002b.Mix-Headset.wav
                ...
            annotations/
                ES2002a.json  # {transcript: "...", summary: "...", speakers: [...]}
                ES2002b.json
                ...

    Args:
        split: Dataset split ("train", "dev", "test", or "all")
        limit: Maximum number of samples to return (None = all)
        require_summary: Only return samples with reference summaries

    Yields:
        EvalSample objects with AMI meeting data

    Raises:
        FileNotFoundError: If AMI directory not found or structure invalid

    Example:
        >>> for sample in iter_ami_meetings(split="test", limit=5):
        ...     print(f"Meeting {sample.id}: {sample.audio_path}")
    """
    root = get_benchmarks_root() / "ami"
    if not root.exists():
        raise FileNotFoundError(
            f"AMI Meeting Corpus not found at {root}.\n"
            f"Please see docs/AMI_SETUP.md for setup instructions."
        )

    audio_dir = root / "audio"
    annotations_dir = root / "annotations"

    if not audio_dir.exists() or not annotations_dir.exists():
        raise FileNotFoundError(
            f"AMI directory structure invalid. Expected:\n"
            f"  {audio_dir}/\n"
            f"  {annotations_dir}/\n"
            f"See docs/AMI_SETUP.md for correct structure."
        )

    # Load split manifest if available
    split_file = root / f"splits/{split}.txt"
    if split_file.exists():
        meeting_ids = split_file.read_text().strip().split("\n")
    else:
        # Fallback: all audio files
        meeting_ids = [f.stem for f in audio_dir.glob("*.wav")]

    count = 0
    for meeting_id in meeting_ids:
        if limit and count >= limit:
            break

        audio_path = audio_dir / f"{meeting_id}.wav"
        annotation_path = annotations_dir / f"{meeting_id}.json"

        if not audio_path.exists():
            continue

        # Load annotations if available
        reference_transcript = None
        reference_summary = None
        reference_speakers = None
        metadata = {}

        if annotation_path.exists():
            with open(annotation_path) as f:
                annotations = json.load(f)
            reference_transcript = annotations.get("transcript")
            reference_summary = annotations.get("summary")
            reference_speakers = annotations.get("speakers")
            metadata = annotations.get("metadata", {})

        # Skip if summary required but missing
        if require_summary and not reference_summary:
            continue

        yield EvalSample(
            dataset="ami",
            id=meeting_id,
            audio_path=audio_path,
            reference_transcript=reference_transcript,
            reference_summary=reference_summary,
            reference_speakers=reference_speakers,
            metadata=metadata,
        )
        count += 1


def iter_iemocap_clips(
    session: str | None = None,
    limit: int | None = None,
) -> Iterable[EvalSample]:
    """Iterate over IEMOCAP emotion recognition samples.

    IEMOCAP should be manually staged under benchmarks_root/iemocap/ following
    the structure documented in docs/IEMOCAP_SETUP.md.

    Expected structure:
        benchmarks_root/iemocap/
            Session1/
                sentences/wav/...
                dialog/EmoEvaluation/...
            Session2/
            ...

    Args:
        session: Specific session to load ("Session1" through "Session5", or None for all)
        limit: Maximum number of samples to return (None = all)

    Yields:
        EvalSample objects with IEMOCAP clip data

    Raises:
        FileNotFoundError: If IEMOCAP directory not found

    Example:
        >>> for sample in iter_iemocap_clips(session="Session1", limit=10):
        ...     print(f"Clip {sample.id}: {sample.reference_emotions}")
    """
    root = get_benchmarks_root() / "iemocap"
    if not root.exists():
        raise FileNotFoundError(
            f"IEMOCAP dataset not found at {root}.\n"
            f"Please see docs/IEMOCAP_SETUP.md for setup instructions."
        )

    sessions = [session] if session else [f"Session{i}" for i in range(1, 6)]
    count = 0

    for sess in sessions:
        sess_dir = root / sess
        if not sess_dir.exists():
            continue

        wav_dir = sess_dir / "sentences" / "wav"
        eval_dir = sess_dir / "dialog" / "EmoEvaluation"

        if not wav_dir.exists():
            continue

        # Scan WAV files and match to emotion annotations
        for wav_file in wav_dir.rglob("*.wav"):
            if limit and count >= limit:
                return

            clip_id = wav_file.stem

            # Try to find corresponding emotion annotation
            # IEMOCAP format: Ses01F_impro01_F000.wav
            # Annotation file: Ses01F_impro01.txt contains all utterances
            dialog_id = "_".join(clip_id.split("_")[:2])
            eval_file = eval_dir / f"{dialog_id}.txt"

            reference_emotions = None
            if eval_file.exists():
                # Parse emotion annotation file for this clip
                reference_emotions = _parse_iemocap_emotions(eval_file, clip_id)

            yield EvalSample(
                dataset="iemocap",
                id=clip_id,
                audio_path=wav_file,
                reference_emotions=reference_emotions,
                metadata={"session": sess, "dialog_id": dialog_id},
            )
            count += 1


def _parse_iemocap_emotions(eval_file: Path, clip_id: str) -> list[str] | None:
    """Parse IEMOCAP emotion evaluation file for a specific clip.

    IEMOCAP format (simplified):
        [6.2901 - 8.2357]	A	ang	[2.5000, 2.5000, 2.5000]

    Args:
        eval_file: Path to emotion evaluation file
        clip_id: Clip ID to search for (e.g., "Ses01F_impro01_F000")

    Returns:
        List of emotion labels (e.g., ["angry"]) or None if not found
    """
    try:
        with open(eval_file) as f:
            for line in f:
                if clip_id in line:
                    # Extract emotion label (third field)
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        # Map IEMOCAP codes to full labels
                        emotion_map = {
                            "ang": "angry",
                            "hap": "happy",
                            "sad": "sad",
                            "neu": "neutral",
                            "exc": "excited",
                            "fru": "frustrated",
                            "fea": "fearful",
                            "sur": "surprised",
                            "dis": "disgusted",
                            "oth": "other",
                        }
                        emotion = emotion_map.get(emotion_code, emotion_code)
                        return [emotion]
    except Exception as e:
        logger.debug(
            "Failed to parse IEMOCAP emotion annotations from %s (clip %s): %s",
            eval_file,
            clip_id,
            e,
        )
    return None


def iter_librispeech(
    split: str = "dev-clean",
    limit: int | None = None,
) -> Iterable[EvalSample]:
    """Iterate over LibriSpeech ASR evaluation samples.

    LibriSpeech is a large corpus of read English speech (1000 hours) derived from
    LibriVox audiobooks. It's the gold standard for evaluating ASR accuracy (WER).

    The dataset should be manually downloaded and staged under:
        benchmarks_root/librispeech/LibriSpeech/<split>/

    Download instructions:
        LibriSpeech is available from OpenSLR (www.openslr.org/12):

        # Recommended subsets for evaluation (sorted by size):
        # dev-clean (337 MB): Clean dev set, 5.4 hours
        wget https://www.openslr.org/resources/12/dev-clean.tar.gz

        # test-clean (346 MB): Clean test set, 5.4 hours
        wget https://www.openslr.org/resources/12/test-clean.tar.gz

        # dev-other (314 MB): Challenging dev set, 5.3 hours
        wget https://www.openslr.org/resources/12/dev-other.tar.gz

        # test-other (328 MB): Challenging test set, 5.1 hours
        wget https://www.openslr.org/resources/12/test-other.tar.gz

        # Extract to benchmarks directory:
        tar -xzf dev-clean.tar.gz -C ~/.cache/slower-whisper/benchmarks/librispeech/

        This creates the expected structure:
            benchmarks_root/librispeech/LibriSpeech/dev-clean/...

        For WER evaluation, dev-clean and test-clean are recommended (high quality audio).
        For robustness testing, use dev-other and test-other (noisy conditions).

    Expected directory structure:
        benchmarks_root/librispeech/
            LibriSpeech/
                dev-clean/
                    <speaker_id>/
                        <chapter_id>/
                            <speaker_id>-<chapter_id>.trans.txt
                            <speaker_id>-<chapter_id>-<utterance_id>.flac
                            ...

    Args:
        split: Dataset split to load. Options:
            - "dev-clean": Development set, clean speech (default)
            - "test-clean": Test set, clean speech
            - "dev-other": Development set, challenging conditions
            - "test-other": Test set, challenging conditions
            - "train-clean-100": Training subset, 100h clean (for reference)
            - "train-clean-360": Training subset, 360h clean
            - "train-other-500": Training subset, 500h other
        limit: Maximum number of samples to return (None = all)

    Yields:
        EvalSample objects with:
            - dataset="librispeech"
            - id="{speaker_id}-{chapter_id}-{utterance_id}"
            - audio_path: Path to .flac file
            - reference_transcript: Ground truth text
            - metadata: {"split": split, "speaker_id": ..., "chapter_id": ...}

    Raises:
        FileNotFoundError: If LibriSpeech split directory not found

    Example:
        >>> # Evaluate first 10 dev-clean samples
        >>> for sample in iter_librispeech(split="dev-clean", limit=10):
        ...     print(f"{sample.id}: {sample.reference_transcript[:50]}...")

        >>> # Evaluate all test-clean samples
        >>> for sample in iter_librispeech(split="test-clean"):
        ...     wer = evaluate_wer(sample.audio_path, sample.reference_transcript)
    """
    root = get_benchmarks_root() / "librispeech" / "LibriSpeech" / split
    if not root.exists():
        raise FileNotFoundError(
            f"LibriSpeech split '{split}' not found at {root}.\n"
            f"Please download from OpenSLR and extract:\n"
            f"  wget https://www.openslr.org/resources/12/{split}.tar.gz\n"
            f"  tar -xzf {split}.tar.gz -C {get_benchmarks_root()}/librispeech/\n\n"
            f"See docstring of iter_librispeech() for complete setup instructions."
        )

    count = 0

    # Walk the LibriSpeech directory structure: speaker_id/chapter_id/*.trans.txt
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name

        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            chapter_id = chapter_dir.name

            # Find the transcript file for this chapter
            trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            if not trans_file.exists():
                continue

            # Parse transcript file
            # Format: <utterance_id> <transcript text...>
            with open(trans_file, encoding="utf-8") as f:
                for line in f:
                    if limit and count >= limit:
                        return

                    line = line.strip()
                    if not line:
                        continue

                    # Split on first whitespace: utterance_id and rest
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue

                    utterance_id, transcript = parts

                    # Find corresponding .flac audio file
                    audio_path = chapter_dir / f"{utterance_id}.flac"
                    if not audio_path.exists():
                        continue

                    yield EvalSample(
                        dataset="librispeech",
                        id=utterance_id,
                        audio_path=audio_path,
                        reference_transcript=transcript,
                        metadata={
                            "split": split,
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                        },
                    )
                    count += 1


def list_available_benchmarks() -> dict[str, dict[str, Any]]:
    """List available benchmark datasets with status.

    Returns:
        Dict mapping dataset name to status info:
        {
            "ami": {
                "available": True,
                "path": "/path/to/benchmarks/ami",
                "setup_doc": "docs/AMI_SETUP.md",
                "description": "AMI Meeting Corpus for diarization and summarization",
            },
            ...
        }

    Example:
        >>> benchmarks = list_available_benchmarks()
        >>> if benchmarks["ami"]["available"]:
        ...     print(f"AMI is ready at {benchmarks['ami']['path']}")
    """
    root = get_benchmarks_root()

    datasets = {
        "ami": {
            "path": str(root / "ami"),
            "available": (root / "ami").exists(),
            "setup_doc": "docs/AMI_SETUP.md",
            "description": "AMI Meeting Corpus for diarization and summarization evaluation",
            "tasks": ["diarization", "summarization", "action_items"],
        },
        "iemocap": {
            "path": str(root / "iemocap"),
            "available": (root / "iemocap").exists(),
            "setup_doc": "docs/IEMOCAP_SETUP.md",
            "description": "IEMOCAP for emotion recognition evaluation",
            "tasks": ["emotion"],
        },
        "librispeech": {
            "path": str(root / "librispeech"),
            "available": (root / "librispeech" / "LibriSpeech").exists(),
            "setup_doc": "See iter_librispeech() docstring",
            "description": "LibriSpeech ASR corpus for WER evaluation (clean speech)",
            "tasks": ["asr"],
        },
        "libricss": {
            "path": str(root / "libricss"),
            "available": (root / "libricss").exists(),
            "setup_doc": "docs/LIBRICSS_SETUP.md",
            "description": "LibriCSS for overlapping speech and diarization",
            "tasks": ["diarization", "overlap_detection"],
        },
    }

    return datasets
