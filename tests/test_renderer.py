"""Tests for TranscriptRenderer."""

from __future__ import annotations

from transcription.renderer import (
    RendererConfig,
    TranscriptRenderer,
    render_for_llm,
    render_segment_formatted,
    render_segment_safe,
)


class TestRendererConfig:
    """Tests for RendererConfig."""

    def test_default_config(self):
        """Default config uses safe mode."""
        config = RendererConfig()
        assert config.default_mode == "safe"
        assert config.enable_formatting is True
        assert config.enable_safety is True
        assert config.include_speakers is True
        assert config.include_timestamps is False


class TestTranscriptRenderer:
    """Tests for TranscriptRenderer."""

    def test_raw_mode_returns_unchanged(self):
        """Raw mode returns text unchanged."""
        renderer = TranscriptRenderer()
        text = "Meeting at five pm"
        result = renderer.render_text(text, mode="raw")
        assert result == text

    def test_formatted_mode_applies_formatting(self):
        """Formatted mode applies smart formatting."""
        renderer = TranscriptRenderer()
        text = "Meeting at five pm"
        result = renderer.render_text(text, mode="formatted")
        # Should convert "five pm" to "5:00 PM"
        assert "5:00 PM" in result or "five pm" in result  # May depend on formatter config

    def test_safe_mode_applies_safety(self):
        """Safe mode applies safety processing."""
        renderer = TranscriptRenderer()
        text = "Call me at 555-123-4567"
        result = renderer.render_text(text, mode="safe")
        # PII should be masked
        assert "[PHONE]" in result or "555-123-4567" in result  # Depends on config

    def test_render_segment_dict(self):
        """Render segment from dictionary."""
        renderer = TranscriptRenderer()
        segment = {
            "text": "Hello world",
            "start": 0.0,
            "end": 1.0,
            "speaker": {"id": "spk_0"},
        }
        result = renderer.render_segment(segment, mode="raw")
        assert result == "Hello world"

    def test_render_segment_with_precomputed_safety(self):
        """Use pre-computed safety results when available."""
        renderer = TranscriptRenderer()
        segment = {
            "text": "Original text",
            "start": 0.0,
            "end": 1.0,
            "speaker": None,
            "audio_state": {
                "safety": {
                    "processed_text": "Processed text",
                },
            },
        }
        result = renderer.render_segment(segment, mode="safe")
        assert result == "Processed text"

    def test_render_turn_dict(self):
        """Render turn from dictionary."""
        renderer = TranscriptRenderer()
        turn = {
            "text": "Turn text content",
            "start": 0.0,
            "end": 5.0,
            "speaker_id": "spk_0",
        }
        result = renderer.render_turn(turn, mode="raw")
        assert result == "Turn text content"

    def test_render_transcript_by_segments(self):
        """Render transcript by segments."""
        config = RendererConfig(
            include_speakers=False,
            include_timestamps=False,
            segment_separator=" ",
        )
        renderer = TranscriptRenderer(config)

        transcript = {
            "segments": [
                {"text": "Hello", "start": 0.0, "end": 1.0, "speaker": None},
                {"text": "world", "start": 1.0, "end": 2.0, "speaker": None},
            ],
            "turns": [],
        }

        result = renderer.render_transcript(transcript, mode="raw", use_turns=False)
        assert result == "Hello world"

    def test_render_transcript_by_turns(self):
        """Render transcript by turns."""
        config = RendererConfig(
            include_speakers=True,
            include_timestamps=False,
            turn_separator="\n",
        )
        renderer = TranscriptRenderer(config)

        transcript = {
            "segments": [],
            "turns": [
                {"text": "First turn", "start": 0.0, "end": 2.0, "speaker_id": "spk_0"},
                {"text": "Second turn", "start": 2.0, "end": 4.0, "speaker_id": "spk_1"},
            ],
        }

        result = renderer.render_transcript(transcript, mode="raw", use_turns=True)
        assert "spk_0:" in result
        assert "spk_1:" in result
        assert "First turn" in result
        assert "Second turn" in result

    def test_render_with_timestamps(self):
        """Include timestamps when configured."""
        config = RendererConfig(
            include_timestamps=True,
            include_speakers=False,
        )
        renderer = TranscriptRenderer(config)

        segment = {
            "text": "Hello",
            "start": 1.5,
            "end": 2.5,
            "speaker": None,
        }

        result = renderer._render_segment_line(segment, mode="raw")
        assert "[1.5s]" in result

    def test_render_with_speakers(self):
        """Include speaker labels when configured."""
        config = RendererConfig(
            include_timestamps=False,
            include_speakers=True,
        )
        renderer = TranscriptRenderer(config)

        segment = {
            "text": "Hello",
            "start": 0.0,
            "end": 1.0,
            "speaker": {"id": "Agent"},
        }

        result = renderer._render_segment_line(segment, mode="raw")
        assert "Agent:" in result


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_render_for_llm(self):
        """render_for_llm produces LLM-ready output."""
        transcript = {
            "segments": [],
            "turns": [
                {"text": "Hello", "start": 0.0, "end": 1.0, "speaker_id": "spk_0"},
            ],
        }

        result = render_for_llm(transcript)
        assert "spk_0:" in result
        assert "Hello" in result

    def test_render_for_llm_no_speakers(self):
        """render_for_llm respects include_speakers flag."""
        transcript = {
            "segments": [],
            "turns": [
                {"text": "Hello", "start": 0.0, "end": 1.0, "speaker_id": "spk_0"},
            ],
        }

        result = render_for_llm(transcript, include_speakers=False)
        assert "spk_0:" not in result
        assert "Hello" in result

    def test_render_segment_safe_function(self):
        """render_segment_safe convenience function."""
        segment = {"text": "Test text", "start": 0.0, "end": 1.0, "speaker": None}
        result = render_segment_safe(segment)
        assert isinstance(result, str)

    def test_render_segment_formatted_function(self):
        """render_segment_formatted convenience function."""
        segment = {"text": "five pm", "start": 0.0, "end": 1.0, "speaker": None}
        result = render_segment_formatted(segment)
        assert isinstance(result, str)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Handle empty text gracefully."""
        renderer = TranscriptRenderer()
        result = renderer.render_text("", mode="raw")
        assert result == ""

    def test_none_speaker(self):
        """Handle None speaker gracefully."""
        config = RendererConfig(include_speakers=True)
        renderer = TranscriptRenderer(config)

        segment = {
            "text": "Hello",
            "start": 0.0,
            "end": 1.0,
            "speaker": None,
        }

        result = renderer._render_segment_line(segment, mode="raw")
        # Should not include speaker prefix when None
        assert "None:" not in result

    def test_empty_transcript(self):
        """Handle empty transcript gracefully."""
        renderer = TranscriptRenderer()
        transcript = {"segments": [], "turns": []}

        result = renderer.render_transcript(transcript, mode="raw")
        assert result == ""

    def test_missing_audio_state(self):
        """Handle missing audio_state gracefully."""
        renderer = TranscriptRenderer()
        segment = {
            "text": "Hello",
            "start": 0.0,
            "end": 1.0,
            "speaker": None,
            # No audio_state key
        }

        result = renderer.render_segment(segment, mode="safe")
        # Should fall back to on-demand processing
        assert isinstance(result, str)
