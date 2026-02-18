"""Transcript rendering API for safe, formatted output.

This module provides a unified rendering API that produces text variants
from transcripts based on the desired output mode:

- raw: Original transcribed text
- formatted: Smart-formatted text (ITN-lite: dates, times, currency, etc.)
- safe: Formatted + PII masked + moderation masked

The renderer uses safety processing results when available, and falls back
to original text when not.

Example:
    >>> from transcription.renderer import TranscriptRenderer, RenderMode
    >>> renderer = TranscriptRenderer()
    >>>
    >>> # Render a segment
    >>> text = renderer.render_segment(segment, mode="safe")
    >>>
    >>> # Render a full transcript
    >>> output = renderer.render_transcript(transcript, mode="formatted")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .safety_config import SafetyConfig
from .safety_layer import SafetyProcessor
from .smart_formatting import SmartFormatter, SmartFormatterConfig

if TYPE_CHECKING:
    from .models import Segment, Transcript, Turn

logger = logging.getLogger(__name__)

RenderMode = Literal["raw", "formatted", "safe"]


@dataclass(slots=True)
class RendererConfig:
    """Configuration for transcript rendering.

    Attributes:
        default_mode: Default rendering mode.
        enable_formatting: Enable smart formatting for "formatted" mode.
        enable_safety: Enable safety processing for "safe" mode.
        safety_config: Configuration for safety processing.
        formatting_config: Configuration for smart formatting.
        segment_separator: Separator between segments in full transcript.
        turn_separator: Separator between turns in full transcript.
        include_timestamps: Include timestamps in rendered output.
        include_speakers: Include speaker labels in rendered output.
    """

    default_mode: RenderMode = "safe"
    enable_formatting: bool = True
    enable_safety: bool = True
    safety_config: SafetyConfig | None = None
    formatting_config: SmartFormatterConfig | None = None
    segment_separator: str = " "
    turn_separator: str = "\n\n"
    include_timestamps: bool = False
    include_speakers: bool = True


class TranscriptRenderer:
    """Renderer for transcript text with multiple output modes.

    Provides a unified API for rendering segments, turns, and full transcripts
    with configurable processing modes.

    The renderer uses a hierarchy for "safe" mode:
    1. Pre-computed safety results (segment.audio_state["safety"]["processed_text"])
    2. On-demand safety processing
    3. Fallback to formatted text
    4. Fallback to raw text

    Example:
        >>> renderer = TranscriptRenderer()
        >>>
        >>> # Render with pre-computed safety results
        >>> segment.audio_state = {"safety": {"processed_text": "Meeting at 5:00 PM"}}
        >>> text = renderer.render_segment(segment, mode="safe")
        >>> print(text)
        "Meeting at 5:00 PM"
        >>>
        >>> # Render on-demand
        >>> text = renderer.render_text("five pm meeting", mode="formatted")
        >>> print(text)
        "5:00 PM meeting"
    """

    def __init__(self, config: RendererConfig | None = None):
        """Initialize renderer with configuration.

        Args:
            config: Renderer configuration. Uses defaults if not provided.
        """
        self.config = config or RendererConfig()
        self._formatter: SmartFormatter | None = None
        self._safety_processor: SafetyProcessor | None = None
        self._initialize_processors()

    def _initialize_processors(self) -> None:
        """Initialize processing components."""
        if self.config.enable_formatting:
            fmt_config = self.config.formatting_config or SmartFormatterConfig()
            self._formatter = SmartFormatter(fmt_config)

        if self.config.enable_safety:
            safety_config = self.config.safety_config or SafetyConfig(
                enabled=True,
                enable_pii_detection=True,
                enable_content_moderation=True,
                enable_smart_formatting=True,
                pii_action="mask",
                content_action="mask",
            )
            self._safety_processor = SafetyProcessor(safety_config)

    def render_text(
        self,
        text: str,
        mode: RenderMode | None = None,
    ) -> str:
        """Render arbitrary text with specified mode.

        Args:
            text: Text to render.
            mode: Rendering mode. Uses default if not specified.

        Returns:
            Rendered text string.
        """
        mode = mode or self.config.default_mode

        if mode == "raw":
            return text

        if mode == "formatted":
            if self._formatter:
                return self._formatter.render(text)
            return text

        # mode == "safe" or default
        if self._safety_processor:
            result = self._safety_processor.process(text)
            return result.processed_text
        # Fallback to formatted
        if self._formatter:
            return self._formatter.render(text)
        return text

    def render_segment(
        self,
        segment: Segment | dict[str, Any],
        mode: RenderMode | None = None,
    ) -> str:
        """Render a segment with specified mode.

        Uses pre-computed results from segment.audio_state["safety"] when
        available for "safe" mode.

        Args:
            segment: Segment object or dictionary.
            mode: Rendering mode. Uses default if not specified.

        Returns:
            Rendered text string.
        """
        mode = mode or self.config.default_mode

        # Extract text
        if isinstance(segment, dict):
            text = str(segment.get("text", ""))
            audio_state: dict[str, Any] = segment.get("audio_state") or {}
        else:
            text = segment.text or ""
            audio_state = segment.audio_state or {}

        # Raw mode: return original text
        if mode == "raw":
            return text

        # Check for pre-computed results
        if mode == "safe":
            safety_state: dict[str, Any] = audio_state.get("safety") or {}

            # Use processed_text if available
            processed = safety_state.get("processed_text")
            if processed:
                return str(processed)

            # Check for original_text (means processing happened)
            if safety_state.get("original_text"):
                # Processing happened but no processed_text stored
                # This shouldn't happen, but fall through to on-demand
                pass

        # On-demand processing
        return self.render_text(text, mode)

    def render_turn(
        self,
        turn: Turn | dict[str, Any],
        mode: RenderMode | None = None,
    ) -> str:
        """Render a turn with specified mode.

        Renders the turn's text content. For turns with segment references,
        consider rendering each segment separately.

        Args:
            turn: Turn object or dictionary.
            mode: Rendering mode. Uses default if not specified.

        Returns:
            Rendered text string.
        """
        mode = mode or self.config.default_mode

        # Extract text
        if isinstance(turn, dict):
            text = turn.get("text", "")
        else:
            text = turn.text or ""

        return self.render_text(text, mode)

    def render_transcript(
        self,
        transcript: Transcript | dict[str, Any],
        mode: RenderMode | None = None,
        use_turns: bool = True,
    ) -> str:
        """Render a full transcript with specified mode.

        Args:
            transcript: Transcript object or dictionary.
            mode: Rendering mode. Uses default if not specified.
            use_turns: If True, render by turns; otherwise by segments.

        Returns:
            Rendered transcript string.
        """
        mode = mode or self.config.default_mode
        lines: list[str] = []

        # Extract segments and turns
        if isinstance(transcript, dict):
            segments = transcript.get("segments", [])
            turns = transcript.get("turns", [])
        else:
            segments = transcript.segments or []
            turns = transcript.turns or []

        # Render by turns if available and requested
        if use_turns and turns:
            for turn in turns:
                line = self._render_turn_line(turn, mode)
                lines.append(line)
            return self.config.turn_separator.join(lines)

        # Render by segments
        for segment in segments:
            line = self._render_segment_line(segment, mode)
            lines.append(line)

        return self.config.segment_separator.join(lines)

    def _render_segment_line(
        self,
        segment: Segment | dict[str, Any],
        mode: RenderMode,
    ) -> str:
        """Render a segment with optional metadata.

        Args:
            segment: Segment to render.
            mode: Rendering mode.

        Returns:
            Formatted segment line.
        """
        text = self.render_segment(segment, mode)
        parts: list[str] = []

        # Add timestamp if requested
        if self.config.include_timestamps:
            if isinstance(segment, dict):
                start = segment.get("start", 0.0)
            else:
                start = segment.start
            parts.append(f"[{start:.1f}s]")

        # Add speaker if requested
        if self.config.include_speakers:
            if isinstance(segment, dict):
                speaker = segment.get("speaker", {})
                speaker_id = speaker.get("id") if isinstance(speaker, dict) else None
            else:
                speaker_id = segment.speaker.get("id") if segment.speaker else None

            if speaker_id:
                parts.append(f"{speaker_id}:")

        parts.append(text)
        return " ".join(parts)

    def _render_turn_line(
        self,
        turn: Turn | dict[str, Any],
        mode: RenderMode,
    ) -> str:
        """Render a turn with optional metadata.

        Args:
            turn: Turn to render.
            mode: Rendering mode.

        Returns:
            Formatted turn line.
        """
        text = self.render_turn(turn, mode)
        parts: list[str] = []

        # Add timestamp if requested
        if self.config.include_timestamps:
            if isinstance(turn, dict):
                start = turn.get("start", 0.0)
            else:
                start = turn.start
            parts.append(f"[{start:.1f}s]")

        # Add speaker if requested
        if self.config.include_speakers:
            if isinstance(turn, dict):
                speaker_id = turn.get("speaker_id")
            else:
                speaker_id = turn.speaker_id

            if speaker_id:
                parts.append(f"{speaker_id}:")

        parts.append(text)
        return " ".join(parts)


def render_for_llm(
    transcript: Any,
    include_speakers: bool = True,
    include_timestamps: bool = False,
) -> str:
    """Convenience function to render transcript for LLM consumption.

    Applies safe mode (PII masked, formatted) with sensible defaults
    for LLM prompts.

    Args:
        transcript: Transcript to render.
        include_speakers: Include speaker labels.
        include_timestamps: Include timestamps.

    Returns:
        LLM-ready transcript string.
    """
    config = RendererConfig(
        default_mode="safe",
        include_speakers=include_speakers,
        include_timestamps=include_timestamps,
        turn_separator="\n",
        segment_separator=" ",
    )
    renderer = TranscriptRenderer(config)
    return renderer.render_transcript(transcript)


def render_segment_safe(segment: Any) -> str:
    """Convenience function to get safe-rendered segment text.

    Args:
        segment: Segment to render.

    Returns:
        Safe-rendered text.
    """
    renderer = TranscriptRenderer()
    return renderer.render_segment(segment, mode="safe")


def render_segment_formatted(segment: Any) -> str:
    """Convenience function to get formatted segment text.

    Args:
        segment: Segment to render.

    Returns:
        Formatted text (no PII masking).
    """
    renderer = TranscriptRenderer()
    return renderer.render_segment(segment, mode="formatted")


__all__ = [
    "TranscriptRenderer",
    "RendererConfig",
    "RenderMode",
    "render_for_llm",
    "render_segment_safe",
    "render_segment_formatted",
]
