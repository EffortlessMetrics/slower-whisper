"""Speaker diarization for transcripts."""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from ...models import Transcript, Segment
from .config import DiarizationConfig

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    Performs speaker diarization and maps speakers to transcript segments.

    Uses pyannote.audio for diarization.
    """

    def __init__(self, config: DiarizationConfig):
        """
        Initialize the speaker diarizer.

        Args:
            config: DiarizationConfig instance
        """
        self.config = config
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """Initialize pyannote diarization pipeline."""
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio not installed. "
                "Install with: pip install pyannote.audio"
            )

        # Get HuggingFace token
        hf_token = self.config.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HuggingFace token not provided. "
                "Set HF_TOKEN environment variable or pass via config. "
                "Get token from: https://huggingface.co/settings/tokens"
            )

        # Load pipeline
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.config.model_name,
                use_auth_token=hf_token
            )

            # Move to specified device
            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.pipeline.to(torch.device("cuda"))
                    logger.info(f"Using CUDA for diarization")
                else:
                    logger.warning("CUDA requested but not available, using CPU")
                    self.pipeline.to(torch.device("cpu"))
            else:
                import torch
                self.pipeline.to(torch.device("cpu"))

            logger.info(f"Initialized diarization pipeline: {self.config.model_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to load diarization pipeline: {e}")

    def annotate(self, transcript: Transcript, audio_path: Path,
                 speaker_mapping: Optional[Dict[str, str]] = None) -> Transcript:
        """
        Annotate transcript with speaker labels.

        Args:
            transcript: Transcript object to enrich
            audio_path: Path to the audio file (WAV format, normalized)
            speaker_mapping: Optional mapping from SPEAKER_XX to human names

        Returns:
            Updated Transcript with segment.speaker populated
        """
        if not transcript.segments:
            logger.warning(f"No segments to diarize in {transcript.file_name}")
            return transcript

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Diarizing {audio_path.name} ({len(transcript.segments)} segments)")

        # Run diarization
        diarization = self._run_diarization(audio_path)

        # Extract speaker turns
        speaker_turns = self._extract_speaker_turns(diarization)
        logger.info(f"Found {len(set(t[2] for t in speaker_turns))} unique speakers")

        # Map speakers to segments
        self._map_speakers_to_segments(transcript.segments, speaker_turns, speaker_mapping)

        # Update metadata
        if transcript.meta is None:
            transcript.meta = {}
        if "enrichments" not in transcript.meta:
            transcript.meta["enrichments"] = {}

        transcript.meta["enrichments"].update({
            "speaker_version": "1.0",
            "diarization_model": self.config.model_name,
            "diarization_timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(f"Diarization complete for {transcript.file_name}")
        return transcript

    def _run_diarization(self, audio_path: Path):
        """
        Run pyannote diarization on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Diarization result from pyannote
        """
        # Build diarization parameters
        params = {}
        if self.config.num_speakers is not None:
            params["num_speakers"] = self.config.num_speakers
        else:
            params["min_speakers"] = self.config.min_speakers
            params["max_speakers"] = self.config.max_speakers

        logger.debug(f"Running diarization with params: {params}")
        diarization = self.pipeline(str(audio_path), **params)
        return diarization

    def _extract_speaker_turns(self, diarization) -> List[Tuple[float, float, str]]:
        """
        Extract speaker turns from diarization result.

        Args:
            diarization: Pyannote diarization result

        Returns:
            List of (start, end, speaker_label) tuples
        """
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))
        return sorted(turns, key=lambda x: x[0])

    def _map_speakers_to_segments(self, segments: List[Segment],
                                   speaker_turns: List[Tuple[float, float, str]],
                                   speaker_mapping: Optional[Dict[str, str]]):
        """
        Map speaker labels to segments based on temporal overlap.

        Args:
            segments: List of Segment objects to annotate
            speaker_turns: List of (start, end, speaker_label) from diarization
            speaker_mapping: Optional mapping from default labels to custom names
        """
        for seg in segments:
            seg_mid = (seg.start + seg.end) / 2  # Use midpoint for matching

            # Find the speaker turn that contains the segment midpoint
            best_speaker = None
            max_overlap = 0.0

            for turn_start, turn_end, speaker in speaker_turns:
                # Calculate overlap
                overlap_start = max(seg.start, turn_start)
                overlap_end = min(seg.end, turn_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker

            # Apply speaker label
            if best_speaker:
                # Format as SPEAKER_XX
                speaker_label = f"SPEAKER_{best_speaker.split('_')[-1] if '_' in best_speaker else best_speaker}"

                # Apply mapping if provided
                if speaker_mapping and speaker_label in speaker_mapping:
                    speaker_label = speaker_mapping[speaker_label]

                seg.speaker = speaker_label
            else:
                seg.speaker = "UNKNOWN"


def load_speaker_mapping(mapping_path: Path) -> Dict[str, str]:
    """
    Load speaker mapping from JSON file.

    Format: {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}

    Args:
        mapping_path: Path to mapping JSON file

    Returns:
        Dictionary mapping default labels to custom names
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)
