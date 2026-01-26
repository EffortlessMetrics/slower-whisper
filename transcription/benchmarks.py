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

import csv
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


def iter_commonvoice(
    subset: str = "en_smoke",
    limit: int | None = None,
    validate_only: bool = False,
) -> Iterable[EvalSample]:
    """Iterate over Common Voice dataset samples from a manifest file.

    This function reads samples from a JSON manifest file located at:
        benchmarks/datasets/asr/commonvoice_{subset}/manifest.json

    The manifest should follow the standard benchmark manifest schema with
    a "samples" array containing sample definitions. If the samples array
    is empty and sample_source.method is "selection_file", samples are
    loaded from the referenced selection CSV file instead.

    Args:
        subset: Common Voice subset identifier (e.g., "en_smoke")
        limit: Maximum number of samples to return (None = all)
        validate_only: When True, skip audio file existence checks (for dry-run mode)

    Yields:
        EvalSample objects with:
            - dataset="commonvoice"
            - id=sample["id"]
            - audio_path: Resolved path to audio file
            - reference_transcript: Ground truth text
            - metadata: Additional sample metadata

    Raises:
        FileNotFoundError: If manifest file not found

    Example:
        >>> for sample in iter_commonvoice(subset="en_smoke", limit=5):
        ...     print(f"{sample.id}: {sample.reference_transcript[:50]}...")
    """
    # Manifest is in the project benchmarks directory, not the cache
    # Look relative to this module's location
    module_dir = Path(__file__).parent.parent
    manifest_path = (
        module_dir / "benchmarks" / "datasets" / "asr" / f"commonvoice_{subset}" / "manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Common Voice manifest not found at {manifest_path}.\n"
            f"Please ensure the manifest file exists for subset '{subset}'."
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])

    # Check for selection_file sample_source when samples is empty
    sample_source = manifest.get("sample_source", {})
    if sample_source.get("method") == "selection_file" and not samples:
        # Load samples from selection.csv
        selection_filename = sample_source.get("file", "selection.csv")
        selection_path = manifest_path.parent / selection_filename

        if not selection_path.exists():
            raise FileNotFoundError(
                f"Selection file not found at {selection_path}.\n"
                f"The manifest references sample_source.file='{selection_filename}' "
                f"but the file does not exist."
            )

        yield from _iter_commonvoice_from_selection(
            manifest_path=manifest_path,
            selection_path=selection_path,
            subset=subset,
            manifest=manifest,
            limit=limit,
            validate_only=validate_only,
        )
        return

    if not samples:
        logger.warning(f"No samples found in Common Voice manifest: {manifest_path}")
        return

    count = 0
    for sample in samples:
        if limit and count >= limit:
            return

        sample_id = sample.get("id")
        audio_rel = sample.get("audio")
        reference = sample.get("reference_transcript")

        if not sample_id or not audio_rel:
            logger.debug(f"Skipping sample with missing id or audio: {sample}")
            continue

        # Resolve audio path relative to manifest directory
        audio_path = (manifest_path.parent / audio_rel).resolve()

        if not validate_only and not audio_path.exists():
            logger.warning(
                f"Audio file not staged for sample '{sample_id}': {audio_path}. Skipping sample."
            )
            continue

        yield EvalSample(
            dataset="commonvoice",
            id=sample_id,
            audio_path=audio_path,
            reference_transcript=reference,
            metadata={
                "subset": subset,
                "language": sample.get("language", "en"),
                "duration_s": sample.get("duration_s"),
                "source": sample.get("source"),
                "profile": sample.get("profile"),
            },
        )
        count += 1


def _iter_commonvoice_from_selection(
    manifest_path: Path,
    selection_path: Path,
    subset: str,
    manifest: dict[str, Any],
    limit: int | None = None,
    validate_only: bool = False,
) -> Iterable[EvalSample]:
    """Load Common Voice samples from a selection CSV file.

    The selection CSV is expected to have columns (supports two formats):

    New format (from --generate-selection):
        - path: Audio file path/name from Common Voice (e.g., "common_voice_en_12345.mp3")
        - sentence: Reference transcript text
        - accent: Speaker accent (optional)
        - duration_s: Audio duration in seconds (optional)
        - slot_bucket: Selection bucket (optional)
        - slot_feature: Selection feature (optional)

    Old format (legacy):
        - clip_id: Sample identifier
        - expected_transcript: Reference transcript text

    Audio files are looked up in the cache directory:
        ~/.cache/slower-whisper/benchmarks/commonvoice_{subset}/

    Args:
        manifest_path: Path to the manifest.json file
        selection_path: Path to the selection.csv file
        subset: Common Voice subset identifier
        manifest: Parsed manifest dictionary
        limit: Maximum number of samples to return
        validate_only: When True, skip audio file existence checks

    Yields:
        EvalSample objects
    """
    # Audio files are staged to cache directory
    cache_root = get_benchmarks_root()
    audio_dir = cache_root / f"commonvoice_{subset}"

    # Get audio format from manifest for file extension
    audio_format = manifest.get("audio_format", {})
    audio_encoding = audio_format.get("encoding", "mp3")

    count = 0
    with open(selection_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and count >= limit:
                return

            # Support both new format (path) and old format (clip_id)
            clip_path = row.get("path", "").strip()
            clip_id = row.get("clip_id", "").strip()

            if clip_path:
                # New format: extract filename from path, remove extension for ID
                clip_filename = Path(clip_path).name
                clip_id = Path(clip_filename).stem  # Remove .mp3 extension
            elif not clip_id:
                logger.debug(f"Skipping row with missing path/clip_id: {row}")
                continue

            # Reference transcript can be in 'expected_transcript' or 'sentence'
            reference = row.get("expected_transcript") or row.get("sentence", "")
            reference = reference.strip() if reference else None

            # Build audio path
            # Staged audio is converted to WAV, so try .wav first, then original format
            if clip_path:
                # New format: use the original filename but try .wav extension first
                audio_filename_wav = Path(clip_path).stem + ".wav"
                audio_filename_orig = Path(clip_path).name
            else:
                # Old format: build from clip_id
                audio_filename_wav = f"{clip_id}.wav"
                audio_filename_orig = f"{clip_id}.{audio_encoding}"

            # Try WAV first (staging converts to WAV), then original format
            audio_path = audio_dir / audio_filename_wav
            if not audio_path.exists():
                audio_path = audio_dir / audio_filename_orig

            if not validate_only and not audio_path.exists():
                logger.warning(
                    f"Audio file not staged for sample '{clip_id}': {audio_path}. "
                    f"Run stage_commonvoice.py to download audio files. Skipping sample."
                )
                continue

            # Parse duration if available
            duration_s = None
            if duration_str := row.get("duration_s", "").strip():
                try:
                    duration_s = float(duration_str)
                except ValueError:
                    pass

            yield EvalSample(
                dataset="commonvoice",
                id=clip_id,
                audio_path=audio_path,
                reference_transcript=reference,
                metadata={
                    "subset": subset,
                    "language": manifest.get("meta", {}).get("language", "en"),
                    "duration_s": duration_s,
                    "accent": row.get("accent", "").strip() or None,
                    "notes": row.get("notes", "").strip() or None,
                    "slot_bucket": row.get("slot_bucket", "").strip() or None,
                    "slot_feature": row.get("slot_feature", "").strip() or None,
                    "source": "selection_file",
                },
            )
            count += 1


def iter_smoke_asr(
    limit: int | None = None,
) -> Iterable[EvalSample]:
    """Iterate over ASR smoke test samples.

    Smoke tests are minimal datasets committed to the repository for quick CI
    validation. They use synthetic TTS audio and are always available.

    The samples are defined in:
        benchmarks/datasets/asr/smoke/manifest.json

    Args:
        limit: Maximum number of samples to return (None = all)

    Yields:
        EvalSample objects with:
            - dataset="smoke"
            - id=sample["id"]
            - audio_path: Path to audio file
            - reference_transcript: Ground truth text

    Raises:
        FileNotFoundError: If manifest or audio files not found

    Example:
        >>> for sample in iter_smoke_asr(limit=2):
        ...     print(f"{sample.id}: {sample.reference_transcript[:40]}...")
    """
    # Manifest is in the project benchmarks directory
    module_dir = Path(__file__).parent.parent
    manifest_path = module_dir / "benchmarks" / "datasets" / "asr" / "smoke" / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"ASR smoke manifest not found at {manifest_path}.\n"
            f"This should be committed to the repository."
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    if not samples:
        logger.warning(f"No samples found in ASR smoke manifest: {manifest_path}")
        return

    count = 0
    for sample in samples:
        if limit and count >= limit:
            return

        sample_id = sample.get("id")
        audio_rel = sample.get("audio")
        reference = sample.get("reference_transcript")

        if not sample_id or not audio_rel:
            logger.debug(f"Skipping sample with missing id or audio: {sample}")
            continue

        # Resolve audio path relative to manifest directory
        audio_path = (manifest_path.parent / audio_rel).resolve()

        if not audio_path.exists():
            logger.warning(
                f"Audio file not found for sample '{sample_id}': {audio_path}. Skipping."
            )
            continue

        yield EvalSample(
            dataset="smoke",
            id=sample_id,
            audio_path=audio_path,
            reference_transcript=reference,
            metadata={
                "language": sample.get("language", "en"),
                "duration_s": sample.get("duration_s"),
                "source": sample.get("source"),
                "profile": sample.get("profile"),
                "notes": sample.get("notes"),
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
        "commonvoice_en_smoke": {
            "path": str(
                Path(__file__).parent.parent
                / "benchmarks"
                / "datasets"
                / "asr"
                / "commonvoice_en_smoke"
            ),
            "available": (
                Path(__file__).parent.parent
                / "benchmarks"
                / "datasets"
                / "asr"
                / "commonvoice_en_smoke"
                / "manifest.json"
            ).exists(),
            "setup_doc": "Manifest-based dataset",
            "description": "Common Voice English smoke test subset for quick ASR evaluation",
            "tasks": ["asr"],
        },
        "smoke": {
            "path": str(Path(__file__).parent.parent / "benchmarks" / "datasets" / "asr" / "smoke"),
            "available": (
                Path(__file__).parent.parent
                / "benchmarks"
                / "datasets"
                / "asr"
                / "smoke"
                / "manifest.json"
            ).exists(),
            "setup_doc": "Committed to repository - always available",
            "description": "Minimal ASR smoke tests for CI (synthetic TTS audio)",
            "tasks": ["asr"],
        },
    }

    return datasets
