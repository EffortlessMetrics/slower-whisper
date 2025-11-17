"""
Public API for slower-whisper library.

This module provides a stable, high-level interface for transcription and
enrichment. It wraps the internal pipeline and audio_enrichment modules with
a clean API suitable for programmatic use.

Example usage:
    >>> from transcription import transcribe_directory, enrich_directory
    >>> from transcription import TranscriptionConfig, EnrichmentConfig
    >>>
    >>> # Stage 1: Transcribe
    >>> cfg = TranscriptionConfig(model="large-v3", language="en")
    >>> transcripts = transcribe_directory("/data/project", cfg)
    >>>
    >>> # Stage 2: Enrich
    >>> ecfg = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
    >>> enriched = enrich_directory("/data/project", ecfg)
"""

from __future__ import annotations

from pathlib import Path

from .config import AsrConfig, EnrichmentConfig, Paths, TranscriptionConfig
from .exceptions import EnrichmentError, TranscriptionError
from .models import Transcript
from .writers import load_transcript_from_json, write_json


def transcribe_directory(
    root: str | Path,
    config: TranscriptionConfig,
) -> list[Transcript]:
    """
    Transcribe all audio files under a project root.

    Expected layout:
        raw_audio/      input recordings
        input_audio/    normalized WAVs (auto-created)
        whisper_json/   JSON transcripts (auto-created)
        transcripts/    TXT/SRT outputs (auto-created)

    Args:
        root: Root directory containing raw_audio/ subdirectory
        config: Transcription configuration

    Returns:
        List of Transcript objects (one per audio file)

    Example:
        >>> config = TranscriptionConfig(model="large-v3", language="en")
        >>> transcripts = transcribe_directory("/data/project", config)
        >>> print(f"Transcribed {len(transcripts)} files")
    """
    from .config import AppConfig
    from .pipeline import run_pipeline

    root = Path(root)

    # Convert public config to internal AppConfig
    paths = Paths(root=root)
    asr_cfg = AsrConfig(
        model_name=config.model,
        device=config.device,
        compute_type=config.compute_type,
        vad_min_silence_ms=config.vad_min_silence_ms,
        beam_size=config.beam_size,
        language=config.language,
        task=config.task,
    )
    app_cfg = AppConfig(
        paths=paths,
        asr=asr_cfg,
        skip_existing_json=config.skip_existing_json,
    )

    # Run the pipeline (modifies files on disk)
    run_pipeline(app_cfg)

    # Load and return all transcripts
    json_dir = paths.json_dir
    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        raise TranscriptionError(
            f"No transcripts found in {json_dir}. "
            f"Ensure audio files exist in {paths.raw_dir} and transcription completed successfully."
        )

    transcripts = []
    for json_path in json_files:
        try:
            transcript = load_transcript_from_json(json_path)
            transcripts.append(transcript)
        except Exception as e:
            raise TranscriptionError(
                f"Failed to load transcript from {json_path.name}: {e}. "
                f"The JSON file may be corrupted or have an invalid schema."
            ) from e

    return transcripts


def transcribe_file(
    audio_path: str | Path,
    root: str | Path,
    config: TranscriptionConfig,
) -> Transcript:
    """
    Transcribe a single audio file into the project structure.

    The file will be normalized into root/input_audio/ and the transcript
    written to root/whisper_json/ and root/transcripts/.

    Args:
        audio_path: Path to input audio file (any format supported by ffmpeg)
        root: Root directory for outputs
        config: Transcription configuration

    Returns:
        Transcript object

    Example:
        >>> config = TranscriptionConfig(model="large-v3")
        >>> transcript = transcribe_file("interview.mp3", "/data/project", config)
        >>> print(transcript.segments[0].text)
    """
    from .asr_engine import TranscriptionEngine

    audio_path = Path(audio_path)
    root = Path(root)

    # Validate input
    if not audio_path.exists():
        raise TranscriptionError(
            f"Audio file not found: {audio_path}. "
            f"Please verify the file path and ensure the file exists."
        )

    # For single-file mode, we use the ASR engine directly
    # (pipeline.run_pipeline is designed for batch processing)
    asr_cfg = AsrConfig(
        model_name=config.model,
        device=config.device,
        compute_type=config.compute_type,
        vad_min_silence_ms=config.vad_min_silence_ms,
        beam_size=config.beam_size,
        language=config.language,
        task=config.task,
    )

    # Normalize the audio file
    from . import audio_io

    paths = Paths(root=root)
    audio_io.ensure_dirs(paths)

    # Copy to raw_audio if not already there
    raw_dest = paths.raw_dir / audio_path.name
    if not raw_dest.exists():
        import shutil

        raw_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, raw_dest)

    # Normalize to input_audio
    from .audio_io import normalize_all

    normalize_all(paths)

    # Find the normalized file
    stem = audio_path.stem
    norm_wav = paths.norm_dir / f"{stem}.wav"

    if not norm_wav.exists():
        raise TranscriptionError(
            f"Audio normalization failed: {norm_wav} not found. "
            f"Ensure ffmpeg is installed and the input audio format is supported."
        )

    # Transcribe
    engine = TranscriptionEngine(asr_cfg)
    transcript = engine.transcribe_file(norm_wav)

    # Write outputs
    from . import writers

    json_path = paths.json_dir / f"{stem}.json"
    txt_path = paths.transcripts_dir / f"{stem}.txt"
    srt_path = paths.transcripts_dir / f"{stem}.srt"

    # Build metadata
    from datetime import datetime, timezone

    from . import __version__

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audio_file": transcript.file_name,
        "model_name": config.model,
        "device": config.device,
        "compute_type": config.compute_type,
        "beam_size": config.beam_size,
        "vad_min_silence_ms": config.vad_min_silence_ms,
        "language_hint": config.language,
        "task": config.task,
        "pipeline_version": __version__,
        "root": str(root),
    }
    transcript.meta = meta

    writers.write_json(transcript, json_path)
    writers.write_txt(transcript, txt_path)
    writers.write_srt(transcript, srt_path)

    return transcript


def enrich_directory(
    root: str | Path,
    config: EnrichmentConfig,
) -> list[Transcript]:
    """
    Enrich all transcripts under whisper_json/ with audio-derived features.

    Reads transcripts from root/whisper_json/ and corresponding audio from
    root/input_audio/, enriches them with prosody/emotion features, and
    writes updated transcripts back to whisper_json/.

    Args:
        root: Root directory
        config: Enrichment configuration

    Returns:
        List of enriched Transcript objects

    Example:
        >>> config = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
        >>> enriched = enrich_directory("/data/project", config)
        >>> print(enriched[0].segments[0].audio_state["rendering"])
    """
    root = Path(root)
    paths = Paths(root=root)

    json_dir = paths.json_dir
    audio_dir = paths.norm_dir

    if not json_dir.exists():
        raise EnrichmentError(
            f"JSON directory does not exist: {json_dir}. "
            f"Run transcription first using transcribe_directory() or ensure the project structure is correct."
        )
    if not audio_dir.exists():
        raise EnrichmentError(
            f"Audio directory does not exist: {audio_dir}. "
            f"Run transcription first to generate normalized audio files in {audio_dir}."
        )

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise EnrichmentError(
            f"No JSON transcript files found in {json_dir}. "
            f"Run transcription first using transcribe_directory() to generate transcript files."
        )

    enriched_transcripts = []
    errors = []

    for json_path in json_files:
        stem = json_path.stem
        wav_path = audio_dir / f"{stem}.wav"

        if not wav_path.exists():
            errors.append(f"{json_path.name}: Audio file not found at {wav_path}")
            continue

        # Load transcript
        try:
            transcript = load_transcript_from_json(json_path)
        except Exception as e:
            errors.append(f"{json_path.name}: Failed to load transcript - {e}")
            continue

        # Skip if already enriched and skip_existing=True
        if config.skip_existing:
            if transcript.segments and transcript.segments[0].audio_state is not None:
                enriched_transcripts.append(transcript)
                continue

        # Enrich (lazy import to avoid requiring enrichment dependencies)
        try:
            from .audio_enrichment import enrich_transcript_audio as _enrich_transcript_internal

            enriched = _enrich_transcript_internal(
                transcript=transcript,
                wav_path=wav_path,
                enable_prosody=config.enable_prosody,
                enable_emotion=config.enable_emotion,
                enable_categorical_emotion=config.enable_categorical_emotion,
                compute_baseline=True,
            )

            # Write back
            write_json(enriched, json_path)
            enriched_transcripts.append(enriched)

        except ImportError as e:
            raise EnrichmentError(
                f"Missing required dependencies for audio enrichment. "
                f"Install with: uv sync --extra full or pip install -e '.[full]'. "
                f"Error: {e}"
            ) from e
        except Exception as e:
            errors.append(f"{json_path.name}: Enrichment failed - {e}")

    # If we had errors but got some results, the user may want to know
    if errors and not enriched_transcripts:
        raise EnrichmentError(
            "Failed to enrich any transcripts. Errors encountered:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )

    return enriched_transcripts


def enrich_transcript(
    transcript: Transcript,
    audio_path: str | Path,
    config: EnrichmentConfig,
) -> Transcript:
    """
    Enrich a single Transcript with audio-derived features.

    This is a pure function that does not write to disk. Use with
    load_transcript() and save_transcript() for file-based workflows.

    Args:
        transcript: Transcript object to enrich
        audio_path: Path to corresponding audio file (WAV)
        config: Enrichment configuration

    Returns:
        Enriched Transcript object (new instance)

    Raises:
        EnrichmentError: If audio file not found or enrichment dependencies missing
        EnrichmentError: If enrichment processing fails

    Example:
        >>> from transcription import load_transcript, save_transcript
        >>>
        >>> transcript = load_transcript("transcript.json")
        >>> config = EnrichmentConfig(enable_prosody=True)
        >>> enriched = enrich_transcript(transcript, "audio.wav", config)
        >>> save_transcript(enriched, "enriched.json")
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise EnrichmentError(
            f"Audio file not found: {audio_path}. "
            f"Ensure the audio file exists and the path is correct."
        )

    try:
        from .audio_enrichment import enrich_transcript_audio as _enrich_transcript_internal
    except ImportError as e:
        raise EnrichmentError(
            f"Missing required dependencies for audio enrichment. "
            f"Install with: uv sync --extra full or pip install -e '.[full]'. "
            f"Error: {e}"
        ) from e

    try:
        enriched = _enrich_transcript_internal(
            transcript=transcript,
            wav_path=audio_path,
            enable_prosody=config.enable_prosody,
            enable_emotion=config.enable_emotion,
            enable_categorical_emotion=config.enable_categorical_emotion,
            compute_baseline=True,
        )
    except Exception as e:
        raise EnrichmentError(
            f"Failed to enrich transcript for {audio_path.name}: {e}. "
            f"Check that the audio file is a valid WAV file and enrichment features are properly configured."
        ) from e

    return enriched


# Convenience I/O wrappers for public API


def load_transcript(json_path: str | Path) -> Transcript:
    """
    Load a Transcript from a JSON file.

    Args:
        json_path: Path to JSON transcript file

    Returns:
        Transcript object

    Raises:
        TranscriptionError: If file not found or JSON is invalid

    Example:
        >>> transcript = load_transcript("output.json")
        >>> print(transcript.file_name, transcript.language)
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise TranscriptionError(
            f"Transcript file not found: {json_path}. "
            f"Ensure the file path is correct and the file exists."
        )

    try:
        return load_transcript_from_json(json_path)
    except Exception as e:
        raise TranscriptionError(
            f"Failed to load transcript from {json_path.name}: {e}. "
            f"The JSON file may be corrupted or have an invalid schema."
        ) from e


def save_transcript(transcript: Transcript, json_path: str | Path) -> None:
    """
    Save a Transcript to a JSON file.

    Args:
        transcript: Transcript object to save
        json_path: Output path

    Raises:
        TranscriptionError: If file cannot be written

    Example:
        >>> save_transcript(transcript, "output.json")
    """
    json_path = Path(json_path)

    # Ensure parent directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        write_json(transcript, json_path)
    except Exception as e:
        raise TranscriptionError(
            f"Failed to save transcript to {json_path}: {e}. "
            f"Check that you have write permissions and sufficient disk space."
        ) from e
