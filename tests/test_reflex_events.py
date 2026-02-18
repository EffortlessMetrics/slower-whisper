"""Tests for reflex event payloads (v2.1.0).

This test suite validates the reflex event payload dataclasses:
1. VADActivityPayload creation and fields
2. BargeInPayload creation and fields
3. EndOfTurnHintPayload creation and fields
4. Payloads are proper dataclasses
"""

from __future__ import annotations

import dataclasses

import pytest

from slower_whisper.pipeline.streaming_callbacks import (
    BargeInPayload,
    EndOfTurnHintPayload,
    VADActivityPayload,
)

# =============================================================================
# 1. Test VADActivityPayload Creation and Fields
# =============================================================================


class TestVADActivityPayload:
    """Tests for VADActivityPayload dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """VADActivityPayload can be created with all required fields."""
        payload = VADActivityPayload(
            energy_level=0.5,
            is_speech=True,
            silence_duration_sec=0.0,
        )

        assert payload.energy_level == 0.5
        assert payload.is_speech is True
        assert payload.silence_duration_sec == 0.0

    def test_energy_level_float(self) -> None:
        """energy_level accepts float values."""
        payload = VADActivityPayload(
            energy_level=0.123456,
            is_speech=True,
            silence_duration_sec=0.0,
        )

        assert payload.energy_level == pytest.approx(0.123456, rel=0.001)

    def test_is_speech_boolean(self) -> None:
        """is_speech accepts boolean values."""
        payload_speech = VADActivityPayload(
            energy_level=0.5,
            is_speech=True,
            silence_duration_sec=0.0,
        )
        payload_silence = VADActivityPayload(
            energy_level=0.1,
            is_speech=False,
            silence_duration_sec=2.5,
        )

        assert payload_speech.is_speech is True
        assert payload_silence.is_speech is False

    def test_silence_duration_sec(self) -> None:
        """silence_duration_sec stores duration correctly."""
        payload = VADActivityPayload(
            energy_level=0.05,
            is_speech=False,
            silence_duration_sec=3.5,
        )

        assert payload.silence_duration_sec == 3.5

    def test_speech_state_resets_silence_duration(self) -> None:
        """Typical usage: silence_duration_sec is 0 when speech detected."""
        payload = VADActivityPayload(
            energy_level=0.6,
            is_speech=True,
            silence_duration_sec=0.0,  # Reset when speech detected
        )

        assert payload.is_speech is True
        assert payload.silence_duration_sec == 0.0

    def test_silence_state_accumulates_duration(self) -> None:
        """Typical usage: silence_duration_sec accumulates in silence."""
        payload = VADActivityPayload(
            energy_level=0.01,
            is_speech=False,
            silence_duration_sec=1.5,  # 1.5 seconds of silence
        )

        assert payload.is_speech is False
        assert payload.silence_duration_sec == 1.5

    def test_zero_energy_level(self) -> None:
        """energy_level can be zero."""
        payload = VADActivityPayload(
            energy_level=0.0,
            is_speech=False,
            silence_duration_sec=5.0,
        )

        assert payload.energy_level == 0.0

    def test_high_energy_level(self) -> None:
        """energy_level can be high (e.g., dB scale or normalized)."""
        payload = VADActivityPayload(
            energy_level=0.95,
            is_speech=True,
            silence_duration_sec=0.0,
        )

        assert payload.energy_level == 0.95


# =============================================================================
# 2. Test BargeInPayload Creation and Fields
# =============================================================================


class TestBargeInPayload:
    """Tests for BargeInPayload dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """BargeInPayload can be created with all required fields."""
        payload = BargeInPayload(
            energy=0.6,
            tts_elapsed_sec=2.5,
        )

        assert payload.energy == 0.6
        assert payload.tts_elapsed_sec == 2.5

    def test_energy_field(self) -> None:
        """energy field stores the triggering energy level."""
        payload = BargeInPayload(
            energy=0.75,
            tts_elapsed_sec=1.0,
        )

        assert payload.energy == 0.75

    def test_tts_elapsed_sec_field(self) -> None:
        """tts_elapsed_sec stores time since TTS playback started."""
        payload = BargeInPayload(
            energy=0.5,
            tts_elapsed_sec=3.25,
        )

        assert payload.tts_elapsed_sec == 3.25

    def test_early_barge_in(self) -> None:
        """Early barge-in (user interrupts quickly)."""
        payload = BargeInPayload(
            energy=0.8,
            tts_elapsed_sec=0.5,  # Interrupted after 0.5s of TTS
        )

        assert payload.tts_elapsed_sec == 0.5
        assert payload.energy == 0.8

    def test_late_barge_in(self) -> None:
        """Late barge-in (user waits then interrupts)."""
        payload = BargeInPayload(
            energy=0.6,
            tts_elapsed_sec=10.0,  # Interrupted after 10s of TTS
        )

        assert payload.tts_elapsed_sec == 10.0

    def test_zero_tts_elapsed(self) -> None:
        """tts_elapsed_sec can be zero (immediate barge-in)."""
        payload = BargeInPayload(
            energy=0.9,
            tts_elapsed_sec=0.0,
        )

        assert payload.tts_elapsed_sec == 0.0

    def test_low_energy_barge_in(self) -> None:
        """Low energy barge-in (sensitive threshold)."""
        payload = BargeInPayload(
            energy=0.2,
            tts_elapsed_sec=1.5,
        )

        assert payload.energy == 0.2


# =============================================================================
# 3. Test EndOfTurnHintPayload Creation and Fields
# =============================================================================


class TestEndOfTurnHintPayload:
    """Tests for EndOfTurnHintPayload dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """EndOfTurnHintPayload can be created with all required fields."""
        payload = EndOfTurnHintPayload(
            confidence=0.85,
            silence_duration=1.5,
            terminal_punctuation=True,
            partial_text="Hello, how are you?",
        )

        assert payload.confidence == 0.85
        assert payload.silence_duration == 1.5
        assert payload.terminal_punctuation is True
        assert payload.partial_text == "Hello, how are you?"

    def test_confidence_field(self) -> None:
        """confidence field stores 0.0-1.0 score."""
        payload = EndOfTurnHintPayload(
            confidence=0.95,
            silence_duration=2.0,
            terminal_punctuation=True,
            partial_text="Done.",
        )

        assert payload.confidence == 0.95
        assert 0.0 <= payload.confidence <= 1.0

    def test_low_confidence(self) -> None:
        """Low confidence end-of-turn hint."""
        payload = EndOfTurnHintPayload(
            confidence=0.3,
            silence_duration=0.5,
            terminal_punctuation=False,
            partial_text="I think maybe",
        )

        assert payload.confidence == 0.3

    def test_high_confidence(self) -> None:
        """High confidence end-of-turn hint."""
        payload = EndOfTurnHintPayload(
            confidence=0.99,
            silence_duration=3.0,
            terminal_punctuation=True,
            partial_text="That's all I have to say.",
        )

        assert payload.confidence == 0.99

    def test_silence_duration_field(self) -> None:
        """silence_duration stores trailing silence in seconds."""
        payload = EndOfTurnHintPayload(
            confidence=0.8,
            silence_duration=2.5,
            terminal_punctuation=False,
            partial_text="Hmm",
        )

        assert payload.silence_duration == 2.5

    def test_terminal_punctuation_true(self) -> None:
        """terminal_punctuation is True when punctuation detected."""
        payload = EndOfTurnHintPayload(
            confidence=0.9,
            silence_duration=1.0,
            terminal_punctuation=True,
            partial_text="What do you think?",
        )

        assert payload.terminal_punctuation is True

    def test_terminal_punctuation_false(self) -> None:
        """terminal_punctuation is False when no punctuation."""
        payload = EndOfTurnHintPayload(
            confidence=0.6,
            silence_duration=1.0,
            terminal_punctuation=False,
            partial_text="I was wondering if",
        )

        assert payload.terminal_punctuation is False

    def test_partial_text_field(self) -> None:
        """partial_text stores the transcript that triggered the hint."""
        text = "This is a complete sentence."
        payload = EndOfTurnHintPayload(
            confidence=0.9,
            silence_duration=1.5,
            terminal_punctuation=True,
            partial_text=text,
        )

        assert payload.partial_text == text

    def test_empty_partial_text(self) -> None:
        """partial_text can be empty string."""
        payload = EndOfTurnHintPayload(
            confidence=0.5,
            silence_duration=3.0,
            terminal_punctuation=False,
            partial_text="",
        )

        assert payload.partial_text == ""

    def test_question_mark_punctuation(self) -> None:
        """Question mark detected as terminal punctuation."""
        payload = EndOfTurnHintPayload(
            confidence=0.85,
            silence_duration=0.8,
            terminal_punctuation=True,
            partial_text="Are you there?",
        )

        assert payload.terminal_punctuation is True
        assert payload.partial_text.endswith("?")

    def test_exclamation_mark_punctuation(self) -> None:
        """Exclamation mark detected as terminal punctuation."""
        payload = EndOfTurnHintPayload(
            confidence=0.9,
            silence_duration=0.5,
            terminal_punctuation=True,
            partial_text="Stop right there!",
        )

        assert payload.terminal_punctuation is True
        assert payload.partial_text.endswith("!")


# =============================================================================
# 4. Test That Payloads Are Proper Dataclasses
# =============================================================================


class TestPayloadsAreDataclasses:
    """Tests verifying payloads are proper dataclasses."""

    def test_vad_activity_is_dataclass(self) -> None:
        """VADActivityPayload is a dataclass."""
        assert dataclasses.is_dataclass(VADActivityPayload)

    def test_barge_in_is_dataclass(self) -> None:
        """BargeInPayload is a dataclass."""
        assert dataclasses.is_dataclass(BargeInPayload)

    def test_end_of_turn_hint_is_dataclass(self) -> None:
        """EndOfTurnHintPayload is a dataclass."""
        assert dataclasses.is_dataclass(EndOfTurnHintPayload)

    def test_vad_activity_has_slots(self) -> None:
        """VADActivityPayload uses slots for memory efficiency."""
        assert hasattr(VADActivityPayload, "__slots__")

    def test_barge_in_has_slots(self) -> None:
        """BargeInPayload uses slots for memory efficiency."""
        assert hasattr(BargeInPayload, "__slots__")

    def test_end_of_turn_hint_has_slots(self) -> None:
        """EndOfTurnHintPayload uses slots for memory efficiency."""
        assert hasattr(EndOfTurnHintPayload, "__slots__")

    def test_vad_activity_fields(self) -> None:
        """VADActivityPayload has expected fields."""
        fields = {f.name for f in dataclasses.fields(VADActivityPayload)}
        expected = {"energy_level", "is_speech", "silence_duration_sec"}
        assert fields == expected

    def test_barge_in_fields(self) -> None:
        """BargeInPayload has expected fields."""
        fields = {f.name for f in dataclasses.fields(BargeInPayload)}
        expected = {"energy", "tts_elapsed_sec"}
        assert fields == expected

    def test_end_of_turn_hint_fields(self) -> None:
        """EndOfTurnHintPayload has expected fields."""
        fields = {f.name for f in dataclasses.fields(EndOfTurnHintPayload)}
        expected = {
            "confidence",
            "silence_duration",
            "terminal_punctuation",
            "partial_text",
            "reason_codes",
            "silence_duration_ms",
            "policy_name",
        }
        assert fields == expected


class TestDataclassEquality:
    """Tests for dataclass equality behavior."""

    def test_vad_activity_equality(self) -> None:
        """Two VADActivityPayloads with same values are equal."""
        payload1 = VADActivityPayload(energy_level=0.5, is_speech=True, silence_duration_sec=0.0)
        payload2 = VADActivityPayload(energy_level=0.5, is_speech=True, silence_duration_sec=0.0)

        assert payload1 == payload2

    def test_vad_activity_inequality(self) -> None:
        """Two VADActivityPayloads with different values are not equal."""
        payload1 = VADActivityPayload(energy_level=0.5, is_speech=True, silence_duration_sec=0.0)
        payload2 = VADActivityPayload(energy_level=0.5, is_speech=False, silence_duration_sec=1.0)

        assert payload1 != payload2

    def test_barge_in_equality(self) -> None:
        """Two BargeInPayloads with same values are equal."""
        payload1 = BargeInPayload(energy=0.6, tts_elapsed_sec=2.5)
        payload2 = BargeInPayload(energy=0.6, tts_elapsed_sec=2.5)

        assert payload1 == payload2

    def test_barge_in_inequality(self) -> None:
        """Two BargeInPayloads with different values are not equal."""
        payload1 = BargeInPayload(energy=0.6, tts_elapsed_sec=2.5)
        payload2 = BargeInPayload(energy=0.6, tts_elapsed_sec=3.0)

        assert payload1 != payload2

    def test_end_of_turn_hint_equality(self) -> None:
        """Two EndOfTurnHintPayloads with same values are equal."""
        payload1 = EndOfTurnHintPayload(
            confidence=0.85,
            silence_duration=1.5,
            terminal_punctuation=True,
            partial_text="Hello?",
        )
        payload2 = EndOfTurnHintPayload(
            confidence=0.85,
            silence_duration=1.5,
            terminal_punctuation=True,
            partial_text="Hello?",
        )

        assert payload1 == payload2

    def test_end_of_turn_hint_inequality(self) -> None:
        """Two EndOfTurnHintPayloads with different values are not equal."""
        payload1 = EndOfTurnHintPayload(
            confidence=0.85,
            silence_duration=1.5,
            terminal_punctuation=True,
            partial_text="Hello?",
        )
        payload2 = EndOfTurnHintPayload(
            confidence=0.85,
            silence_duration=1.5,
            terminal_punctuation=False,  # Different
            partial_text="Hello?",
        )

        assert payload1 != payload2


class TestDataclassRepr:
    """Tests for dataclass repr behavior."""

    def test_vad_activity_repr(self) -> None:
        """VADActivityPayload has useful repr."""
        payload = VADActivityPayload(energy_level=0.5, is_speech=True, silence_duration_sec=0.0)

        r = repr(payload)

        assert "VADActivityPayload" in r
        assert "energy_level" in r
        assert "is_speech" in r
        assert "silence_duration_sec" in r

    def test_barge_in_repr(self) -> None:
        """BargeInPayload has useful repr."""
        payload = BargeInPayload(energy=0.6, tts_elapsed_sec=2.5)

        r = repr(payload)

        assert "BargeInPayload" in r
        assert "energy" in r
        assert "tts_elapsed_sec" in r

    def test_end_of_turn_hint_repr(self) -> None:
        """EndOfTurnHintPayload has useful repr."""
        payload = EndOfTurnHintPayload(
            confidence=0.85,
            silence_duration=1.5,
            terminal_punctuation=True,
            partial_text="Test",
        )

        r = repr(payload)

        assert "EndOfTurnHintPayload" in r
        assert "confidence" in r
        assert "silence_duration" in r
        assert "terminal_punctuation" in r
        assert "partial_text" in r
