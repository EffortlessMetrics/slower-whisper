"""Tone analyzer for transcript enrichment."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from ...models import Transcript, Segment
from .config import ToneConfig, TONE_LABELS
from .prompts import build_tone_analysis_prompt, format_segments_for_analysis

logger = logging.getLogger(__name__)


class ToneAnalyzer:
    """
    Analyzes and annotates transcript segments with emotional tone.

    Uses LLM API (Anthropic Claude by default) to classify segment tone.
    """

    def __init__(self, config: ToneConfig):
        """
        Initialize the tone analyzer.

        Args:
            config: ToneConfig instance
        """
        self.config = config
        self.client = None

        # Initialize API client if not using mock
        if config.api_provider == "anthropic":
            self._init_anthropic_client()
        elif config.api_provider == "openai":
            self._init_openai_client()
        elif config.api_provider == "mock":
            logger.info("Using mock tone analyzer (all segments → neutral)")
        else:
            raise ValueError(f"Unsupported API provider: {config.api_provider}")

    def _init_anthropic_client(self):
        """Initialize Anthropic API client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass via config."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Initialized Anthropic client with model: {self.config.model_name}")

    def _init_openai_client(self):
        """Initialize OpenAI API client."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass via config."
            )

        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {self.config.model_name}")

    def annotate(self, transcript: Transcript) -> Transcript:
        """
        Annotate all segments in a transcript with tone labels.

        Args:
            transcript: Transcript object to enrich

        Returns:
            Updated Transcript with segment.tone populated
        """
        if not transcript.segments:
            logger.warning(f"No segments to analyze in {transcript.file_name}")
            return transcript

        logger.info(f"Analyzing tone for {len(transcript.segments)} segments in {transcript.file_name}")

        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(transcript.segments), batch_size):
            batch_end = min(i + batch_size, len(transcript.segments))
            batch = transcript.segments[i:batch_end]

            # Get context if enabled
            context_before = None
            context_after = None
            if self.config.include_context:
                ctx_win = self.config.context_window
                context_before = transcript.segments[max(0, i - ctx_win):i] if i > 0 else None
                context_after = transcript.segments[batch_end:batch_end + ctx_win] if batch_end < len(transcript.segments) else None

            # Analyze batch
            try:
                self._analyze_batch(batch, context_before, context_after)
                logger.debug(f"Analyzed segments {i}-{batch_end-1}")
            except Exception as e:
                logger.error(f"Failed to analyze batch {i}-{batch_end}: {e}")
                # Fallback: mark as neutral
                for seg in batch:
                    seg.tone = "neutral"

        # Update metadata
        if transcript.meta is None:
            transcript.meta = {}
        if "enrichments" not in transcript.meta:
            transcript.meta["enrichments"] = {}

        transcript.meta["enrichments"].update({
            "tone_version": "1.0",
            "tone_model": self.config.model_name,
            "tone_timestamp": datetime.now(timezone.utc).isoformat(),
            "tone_provider": self.config.api_provider,
        })

        logger.info(f"Tone analysis complete for {transcript.file_name}")
        return transcript

    def _analyze_batch(self, batch: List[Segment],
                       context_before: Optional[List[Segment]],
                       context_after: Optional[List[Segment]]):
        """
        Analyze a batch of segments.

        Args:
            batch: Segments to analyze
            context_before: Optional context segments before batch
            context_after: Optional context segments after batch
        """
        if self.config.api_provider == "mock":
            # Mock: assign neutral to all
            for seg in batch:
                seg.tone = "neutral"
            return

        # Format segments for prompt
        segments_text = format_segments_for_analysis(
            batch, 0, len(batch), context_before, context_after
        )

        prompt = build_tone_analysis_prompt(segments_text, self.config.include_context)

        # Call API with retry
        for attempt in range(self.config.max_retries):
            try:
                result = self._call_api(prompt)
                self._apply_results(batch, result)
                return
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait = 2 ** attempt  # Exponential backoff
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

    def _call_api(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Call the LLM API to analyze tone.

        Args:
            prompt: Formatted prompt

        Returns:
            List of tone analysis results
        """
        if self.config.api_provider == "anthropic":
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
        elif self.config.api_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            content = response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported provider: {self.config.api_provider}")

        # Parse JSON response
        try:
            result = json.loads(content.strip())
            if not isinstance(result, list):
                raise ValueError("Expected JSON array")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response as JSON: {content}")
            raise ValueError(f"Invalid JSON response: {e}")

    def _apply_results(self, batch: List[Segment], results: List[Dict[str, Any]]):
        """
        Apply tone analysis results to segments.

        Args:
            batch: Segments that were analyzed
            results: Analysis results from API
        """
        if len(results) != len(batch):
            logger.warning(
                f"Result count mismatch: {len(results)} results for {len(batch)} segments. "
                "Using what we can."
            )

        for i, seg in enumerate(batch):
            if i >= len(results):
                seg.tone = "neutral"
                continue

            result = results[i]
            tone = result.get("tone", "neutral")
            confidence = result.get("confidence", 0.0)

            # Validate tone
            if tone not in TONE_LABELS:
                logger.warning(f"Invalid tone '{tone}' for segment {seg.id}, using 'neutral'")
                tone = "neutral"
                confidence = 0.0

            # Apply confidence threshold
            if confidence < self.config.confidence_threshold:
                tone = "neutral"

            seg.tone = tone
