"""
Enrichment package for Stage 2 processing.

This package contains tools for enriching transcripts with:
- Tone analysis
- Speaker diarization
- Analytics and reporting
"""

from .tone.analyzer import ToneAnalyzer
from .speaker.diarizer import SpeakerDiarizer

__all__ = ["ToneAnalyzer", "SpeakerDiarizer"]
