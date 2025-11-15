"""
Enrichment utilities for transcripts.

This module is the intended home for second-stage processing such as:
- Tone tagging
- Speaker attribution (diarization)
- Highlighting / topic tagging

These functions operate on Transcript objects or JSON equivalents and
do not touch audio or the ASR engine directly.
"""

from .models import Transcript


def annotate_tone(transcript: Transcript) -> Transcript:
    """
    Placeholder for tone annotation.

    In a future iteration, this function can call an LLM or other model
    to infer tone for each segment and populate the `tone` field.
    """
    # Currently a no-op.
    return transcript


def annotate_speakers(transcript: Transcript) -> Transcript:
    """
    Placeholder for speaker attribution.

    A diarization pipeline could map time-stamped speaker turns onto the
    transcript segments and populate the `speaker` field.
    """
    # Currently a no-op.
    return transcript
