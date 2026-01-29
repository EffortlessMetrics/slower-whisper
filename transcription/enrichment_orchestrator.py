"""
Enrichment orchestration extracted from transcription.api.

Holds both:
- helper hooks (speaker analytics + semantic annotator)
- enrichment entrypoints (directory + single transcript)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import EnrichmentConfig, Paths
from .exceptions import EnrichmentError
from .models import AUDIO_STATE_VERSION, SCHEMA_VERSION, Transcript
from .writers import load_transcript_from_json, write_json

logger = logging.getLogger(__name__)

TurnsHaveMetadataFn = Callable[[Any], bool]
RunSpeakerAnalyticsFn = Callable[[Transcript, EnrichmentConfig], Transcript]
RunSemanticAnnotatorFn = Callable[[Transcript, EnrichmentConfig], Transcript]
NeutralAudioStateFn = Callable[[str | None], dict[str, Any]]


def _run_speaker_analytics(  # noqa: N802
    transcript: Transcript, config: EnrichmentConfig
) -> Transcript:
    """Populate turn metadata and speaker_stats when enabled in config."""
    needs_turn_meta = config.enable_turn_metadata or config.enable_speaker_stats

    if needs_turn_meta and not transcript.turns:
        from .turns import build_turns

        transcript = build_turns(transcript, pause_threshold=config.pause_threshold)

    if needs_turn_meta:
        from .turns_enrich import enrich_turns_metadata

        enrich_turns_metadata(transcript)

    if config.enable_speaker_stats:
        from .speaker_stats import compute_speaker_stats

        compute_speaker_stats(transcript)

    return transcript


def _run_semantic_annotator(transcript: Transcript, config: EnrichmentConfig) -> Transcript:
    """Invoke semantic annotator hook when enabled."""
    if not getattr(config, "enable_semantic_annotator", False):
        return transcript

    annotator = getattr(config, "semantic_annotator", None)
    if annotator is None:
        from .semantic import KeywordSemanticAnnotator

        annotator = KeywordSemanticAnnotator()

    try:
        updated = annotator.annotate(transcript)
        return updated or transcript
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic annotator failed: %s", exc, exc_info=True)
        return transcript


def _convert_turn_to_dict(turn: Any) -> dict[str, Any]:
    """Convert a Turn object or dict to a dict for post-processing.

    Args:
        turn: Turn object or dict.

    Returns:
        Dictionary with id, speaker_id, text, start, end.
    """
    if isinstance(turn, dict):
        return turn
    return {
        "id": turn.id,
        "speaker_id": turn.speaker_id,
        "text": turn.text,
        "start": turn.start,
        "end": turn.end,
    }


def _run_post_processors(transcript: Transcript, config: EnrichmentConfig) -> Transcript:
    """Run v2.0 post-processors on transcript using unified PostProcessor.

    Uses the PostProcessor class for unified orchestration of:
    - Safety layer (PII + moderation + formatting)
    - Role inference (agent/customer/facilitator)
    - Topic segmentation with proper finalization
    - Turn-taking evaluation
    - Environment classification
    - Extended prosody analysis

    The function processes segments and turns through PostProcessor,
    calls finalize() to close any open topic chunks, and stores
    results in transcript.annotations.

    Args:
        transcript: Transcript to enrich.
        config: EnrichmentConfig with post-processing settings.

    Returns:
        Enriched transcript with annotations.
    """
    # Use config translation helper to get PostProcessConfig
    pp_cfg = config.to_post_process_config()

    if pp_cfg is None:
        # No post-processing features enabled
        return transcript

    # Ensure annotations dict exists and set schema version
    if transcript.annotations is None:
        transcript.annotations = {}
    transcript.annotations["_schema_version"] = SCHEMA_VERSION

    try:
        from .post_process import PostProcessor, SegmentContext

        # Create unified PostProcessor
        pp = PostProcessor(config=pp_cfg)

        # Process segments
        for idx, seg in enumerate(transcript.segments):
            try:
                # Build segment context
                ctx = SegmentContext(
                    session_id=getattr(transcript, "id", None) or "batch",
                    segment_id=f"seg_{seg.id}" if seg.id is not None else f"seg_{idx}",
                    speaker_id=(seg.speaker.get("id") if seg.speaker else None),
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    audio_state=seg.audio_state or {},
                )

                # Process segment through unified processor
                result = pp.process_segment(ctx)

                # Merge results into audio_state
                result_dict = result.to_dict()
                if result_dict:
                    if seg.audio_state is None:
                        seg.audio_state = {"_schema_version": AUDIO_STATE_VERSION}
                    elif "_schema_version" not in seg.audio_state:
                        seg.audio_state["_schema_version"] = AUDIO_STATE_VERSION
                    seg.audio_state.update(result_dict)

            except Exception as e:
                logger.warning("Segment post-processing failed for seg %d: %s", idx, e)

        # Process turns for role inference and topic segmentation
        if transcript.turns:
            for turn in transcript.turns:
                try:
                    turn_dict = _convert_turn_to_dict(turn)
                    pp.process_turn(turn_dict)
                except Exception as e:
                    logger.warning("Turn post-processing failed: %s", e)

        # CRITICAL: Call finalize() to close open topic chunks and trigger final role inference
        # This ensures:
        # 1. Current topic chunk is closed with proper end_time
        # 2. Role inference is triggered if not already decided
        # 3. Final callbacks are emitted (not used in batch mode, but maintains consistency)
        try:
            pp.finalize()
        except Exception as e:
            logger.warning("Post-processor finalization failed: %s", e)

        # Store role assignments (already computed by finalize())
        if pp_cfg.enable_roles:
            try:
                roles = pp.get_role_assignments()
                if roles:
                    transcript.annotations["roles"] = {
                        speaker_id: assignment.to_dict()
                        for speaker_id, assignment in roles.items()
                    }
            except Exception as e:
                logger.warning("Failed to get role assignments: %s", e)

        # Store topic chunks (finalize() already closed them properly)
        if pp_cfg.enable_topics:
            try:
                topics = pp.get_topics()
                if topics:
                    transcript.annotations["topics"] = [t.to_dict() for t in topics]
            except Exception as e:
                logger.warning("Failed to get topic chunks: %s", e)

    except ImportError as e:
        logger.warning("Post-processing dependencies not available: %s", e)
    except Exception as e:
        logger.warning("Post-processing failed: %s", e, exc_info=True)

    return transcript


def _enrich_directory_impl(
    root: str | Path,
    config: EnrichmentConfig,
    *,
    turns_have_metadata: TurnsHaveMetadataFn,
    run_speaker_analytics: RunSpeakerAnalyticsFn,
    run_semantic_annotator: RunSemanticAnnotatorFn,
) -> list[Transcript]:
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

    enriched_transcripts: list[Transcript] = []
    errors: list[str] = []
    total = len(json_files)

    for idx, json_path in enumerate(json_files, start=1):
        logger.info("[%d/%d] %s", idx, total, json_path.name)
        stem = json_path.stem
        wav_path = audio_dir / f"{stem}.wav"

        if not wav_path.exists():
            errors.append(f"{json_path.name}: Audio file not found at {wav_path}")
            continue

        try:
            transcript = load_transcript_from_json(json_path)
        except Exception as e:
            errors.append(f"{json_path.name}: Failed to load transcript - {e}")
            continue

        audio_ready = bool(transcript.segments) and all(
            seg.audio_state is not None for seg in transcript.segments
        )
        turns_ready = turns_have_metadata(transcript.turns)
        stats_ready = bool(transcript.speaker_stats)
        semantic_ready = not getattr(config, "enable_semantic_annotator", False) or bool(
            (getattr(transcript, "annotations", None) or {}).get("semantic")
        )
        already_enriched = (
            audio_ready
            and (not config.enable_turn_metadata or turns_ready)
            and (not config.enable_speaker_stats or stats_ready)
            and semantic_ready
        )

        if config.skip_existing:
            if already_enriched:
                enriched_transcripts.append(transcript)
                continue

            if audio_ready:
                transcript = run_speaker_analytics(transcript, config)
                transcript = run_semantic_annotator(transcript, config)
                transcript = _run_post_processors(transcript, config)
                write_json(transcript, json_path)
                enriched_transcripts.append(transcript)
                continue

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
            enriched = run_speaker_analytics(enriched, config)
            enriched = run_semantic_annotator(enriched, config)
            enriched = _run_post_processors(enriched, config)

            write_json(enriched, json_path)
            enriched_transcripts.append(enriched)

        except ImportError as e:
            raise EnrichmentError(
                "Missing required dependencies for audio enrichment. "
                'Install with: uv sync --extra full (repo) or pip install "slower-whisper[full]". '
                f"Error: {e}"
            ) from e
        except Exception as e:
            errors.append(f"{json_path.name}: Enrichment failed - {e}")
            logger.warning("Enrichment failed for %s: %s", json_path.name, e, exc_info=True)

    if errors and not enriched_transcripts:
        raise EnrichmentError(
            "Failed to enrich any transcripts. Errors encountered:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )

    return enriched_transcripts


def _enrich_transcript_impl(
    transcript: Transcript,
    audio_path: str | Path,
    config: EnrichmentConfig,
    *,
    turns_have_metadata: TurnsHaveMetadataFn,
    neutral_audio_state: NeutralAudioStateFn,
    run_speaker_analytics: RunSpeakerAnalyticsFn,
    run_semantic_annotator: RunSemanticAnnotatorFn,
) -> Transcript:
    audio_path = Path(audio_path)

    audio_ready = bool(transcript.segments) and all(
        seg.audio_state is not None for seg in transcript.segments
    )
    turns_ready = turns_have_metadata(transcript.turns)
    stats_ready = bool(transcript.speaker_stats)
    semantic_ready = not getattr(config, "enable_semantic_annotator", False) or bool(
        (getattr(transcript, "annotations", None) or {}).get("semantic")
    )
    already_enriched = (
        audio_ready
        and (not config.enable_turn_metadata or turns_ready)
        and (not config.enable_speaker_stats or stats_ready)
        and semantic_ready
    )

    if config.skip_existing and already_enriched:
        return transcript

    if not audio_path.exists():
        raise EnrichmentError(
            f"Audio file not found: {audio_path}. "
            f"Ensure the audio file exists and the path is correct."
        )

    try:
        from .audio_enrichment import enrich_transcript_audio as _enrich_transcript_internal
    except ImportError as e:
        raise EnrichmentError(
            "Missing required dependencies for audio enrichment. "
            'Install with: uv sync --extra full (repo) or pip install "slower-whisper[full]". '
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
        enriched = run_speaker_analytics(enriched, config)
        enriched = run_semantic_annotator(enriched, config)
        enriched = _run_post_processors(enriched, config)
        return enriched
    except Exception as e:
        logger.warning(
            "Enrichment failed for %s: %s; returning neutral audio_state",
            audio_path.name,
            e,
            exc_info=True,
        )
        for segment in transcript.segments:
            segment.audio_state = neutral_audio_state(str(e))
        enriched = run_speaker_analytics(transcript, config)
        enriched = run_semantic_annotator(enriched, config)
        enriched = _run_post_processors(enriched, config)
        return enriched
