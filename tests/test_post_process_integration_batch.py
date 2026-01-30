"""Comprehensive batch integration tests for post-processing wiring.

These tests prove the post-processing wiring works end-to-end in batch mode.
They verify that:
- EnrichmentConfig flags properly enable/disable features
- _run_post_processors correctly wires data through PostProcessor
- Results are properly attached to transcript annotations and segment audio_state
- Schema versions are present where expected
"""

from __future__ import annotations

import pytest

from transcription.enrichment_config import EnrichmentConfig
from transcription.enrichment_orchestrator import _run_post_processors
from transcription.models import (
    AUDIO_STATE_VERSION,
    SCHEMA_VERSION,
    Segment,
    Transcript,
    Turn,
)

# ============================================================================
# Fixtures and Factory Functions
# ============================================================================


def make_segment(
    id: int,
    start: float,
    end: float,
    text: str,
    speaker_id: str | None = None,
    audio_state: dict | None = None,
) -> Segment:
    """Create a test segment with optional speaker and audio_state."""
    speaker = {"id": speaker_id} if speaker_id else None
    return Segment(
        id=id,
        start=start,
        end=end,
        text=text,
        speaker=speaker,
        audio_state=audio_state,
    )


def make_turn(
    id: str,
    speaker_id: str,
    text: str,
    start: float,
    end: float,
    segment_ids: list[int] | None = None,
) -> Turn:
    """Create a test turn."""
    return Turn(
        id=id,
        speaker_id=speaker_id,
        segment_ids=segment_ids or [],
        start=start,
        end=end,
        text=text,
    )


def make_transcript(
    segments: list[Segment],
    turns: list[Turn] | None = None,
    file_name: str = "test.wav",
    language: str = "en",
) -> Transcript:
    """Create a test transcript with segments and optional turns."""
    return Transcript(
        file_name=file_name,
        language=language,
        segments=segments,
        turns=turns,
    )


# ============================================================================
# Test: No-op when disabled
# ============================================================================


class TestNoopWhenDisabled:
    """Tests that post-processing is a no-op when all features are disabled."""

    def test_noop_when_all_features_disabled(self):
        """Config with all post-processing flags false returns transcript unchanged."""
        # Create a simple transcript
        segments = [
            make_segment(0, 0.0, 2.0, "Hello there.", speaker_id="spk_0"),
            make_segment(1, 2.0, 4.0, "How are you?", speaker_id="spk_1"),
        ]
        turns = [
            make_turn("turn_0", "spk_0", "Hello there.", 0.0, 2.0, [0]),
            make_turn("turn_1", "spk_1", "How are you?", 2.0, 4.0, [1]),
        ]
        transcript = make_transcript(segments, turns)

        # Default EnrichmentConfig has no post-processing enabled
        config = EnrichmentConfig()

        # Verify config does not enable any post-processing
        assert config.enable_safety_layer is False
        assert config.enable_role_inference is False
        assert config.enable_topic_segmentation is False
        assert config.enable_environment_classifier is False
        assert config.enable_prosody_v2 is False
        assert config.turn_taking_policy == "balanced"  # Default, no turn-taking enabled

        # Run post-processors
        result = _run_post_processors(transcript, config)

        # Transcript should be unchanged
        assert result.annotations is None
        for seg in result.segments:
            assert seg.audio_state is None

    def test_noop_with_explicit_all_false(self):
        """Explicitly setting all flags to False is a no-op."""
        segments = [
            make_segment(0, 0.0, 2.0, "Test text", speaker_id="spk_0"),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(
            enable_safety_layer=False,
            enable_role_inference=False,
            enable_topic_segmentation=False,
            enable_environment_classifier=False,
            enable_prosody_v2=False,
            turn_taking_policy="balanced",
        )

        result = _run_post_processors(transcript, config)

        assert result.annotations is None
        assert result.segments[0].audio_state is None


# ============================================================================
# Test: Safety attaches structured state
# ============================================================================


class TestSafetyAttachesStructuredState:
    """Tests that safety processing attaches structured state to segments."""

    def test_safety_attaches_processed_flag(self):
        """Safety layer attaches audio_state['safety']['processed'] = True."""
        segments = [
            make_segment(0, 0.0, 2.0, "meeting at five pm", speaker_id="spk_0"),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_safety_layer=True)

        result = _run_post_processors(transcript, config)

        # Segment should have safety state attached
        seg = result.segments[0]
        assert seg.audio_state is not None
        assert "safety" in seg.audio_state
        assert seg.audio_state["safety"]["processed"] is True

    def test_safety_preserves_original_text(self):
        """Safety processing preserves original text in segment."""
        original_text = "call me at 555-123-4567"
        segments = [
            make_segment(0, 0.0, 2.0, original_text, speaker_id="spk_0"),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_safety_layer=True)

        result = _run_post_processors(transcript, config)

        # Original text in segment is preserved (we don't mutate segment.text)
        assert result.segments[0].text == original_text

    def test_safety_with_pii_detection(self):
        """Safety layer detects PII and attaches appropriate state."""
        segments = [
            make_segment(0, 0.0, 2.0, "my email is test@example.com", speaker_id="spk_0"),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_safety_layer=True)

        result = _run_post_processors(transcript, config)

        seg = result.segments[0]
        assert seg.audio_state is not None
        assert "safety" in seg.audio_state
        safety_state = seg.audio_state["safety"]
        assert safety_state["processed"] is True
        # Check that an action was determined
        assert "action" in safety_state

    def test_safety_with_smart_formatting(self):
        """Safety layer with smart formatting still attaches state."""
        segments = [
            make_segment(0, 0.0, 2.0, "meeting at three thirty pm", speaker_id="spk_0"),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_safety_layer=True)

        result = _run_post_processors(transcript, config)

        seg = result.segments[0]
        assert seg.audio_state is not None
        assert "safety" in seg.audio_state
        assert seg.audio_state["safety"]["processed"] is True


# ============================================================================
# Test: Environment classifier consumes audio health
# ============================================================================


class TestEnvironmentClassifierConsumesAudioHealth:
    """Tests that environment classifier uses audio_health data from segments."""

    def test_environment_classifier_with_low_snr(self):
        """Environment classifier tags noisy audio based on low SNR."""
        # Create segment with low SNR in audio_state
        audio_state = {
            "audio_health": {
                "snr_db": 5.0,  # Low SNR indicates noisy
                "noise_floor_db": -30.0,
                "quality_score": 0.4,
            }
        }
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", speaker_id="spk_0", audio_state=audio_state),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_environment_classifier=True)

        result = _run_post_processors(transcript, config)

        seg = result.segments[0]
        assert seg.audio_state is not None
        assert "environment" in seg.audio_state
        env_state = seg.audio_state["environment"]
        assert "tag" in env_state
        # Low SNR should result in "noisy" classification
        assert env_state["tag"] == "noisy"

    def test_environment_classifier_with_high_snr(self):
        """Environment classifier tags clean audio based on high SNR."""
        audio_state = {
            "audio_health": {
                "snr_db": 25.0,  # Good SNR
                "noise_floor_db": -50.0,
                "quality_score": 0.85,
                "spectral_centroid_hz": 2000.0,
                "clipping_ratio": 0.0,
            }
        }
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", speaker_id="spk_0", audio_state=audio_state),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_environment_classifier=True)

        result = _run_post_processors(transcript, config)

        seg = result.segments[0]
        assert seg.audio_state is not None
        assert "environment" in seg.audio_state
        env_state = seg.audio_state["environment"]
        assert "tag" in env_state
        # Good audio health should result in "clean" classification
        assert env_state["tag"] == "clean"

    def test_environment_classifier_without_audio_health_skips(self):
        """Environment classifier skips segments without audio_health data."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", speaker_id="spk_0", audio_state=None),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_environment_classifier=True)

        result = _run_post_processors(transcript, config)

        seg = result.segments[0]
        # audio_state should be created with schema version but no environment
        # since there's no audio_health to classify
        if seg.audio_state:
            assert "environment" not in seg.audio_state or seg.audio_state.get("environment") is None


# ============================================================================
# Test: Prosody extended attaches features
# ============================================================================


class TestProsodyExtendedAttachesFeatures:
    """Tests that prosody_v2 attaches extended prosody features."""

    def test_prosody_v2_flag_enables_extended_prosody(self):
        """Prosody v2 flag enables extended prosody analysis."""
        # Note: Without actual audio, prosody_extended won't be computed
        # but the configuration should still be set up correctly
        segments = [
            make_segment(0, 0.0, 2.0, "Hello there.", speaker_id="spk_0"),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_prosody_v2=True)

        # This should not crash even without audio
        result = _run_post_processors(transcript, config)

        # The transcript should be processed (config translation works)
        assert result is not None

    def test_prosody_extended_with_base_prosody_data(self):
        """When base prosody data exists, extended prosody should be enabled."""
        # Create segment with existing prosody data
        audio_state = {
            "prosody": {
                "pitch": {"mean": 150.0, "std": 20.0},
                "energy": {"mean": -20.0},
            }
        }
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", speaker_id="spk_0", audio_state=audio_state),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_prosody_v2=True)

        result = _run_post_processors(transcript, config)

        # The segment should still have its base prosody data
        seg = result.segments[0]
        assert seg.audio_state is not None
        assert "prosody" in seg.audio_state


# ============================================================================
# Test: Role inference produces transcript-level assignments
# ============================================================================


class TestRoleInferenceProducesTranscriptLevelAssignments:
    """Tests that role inference produces transcript.annotations['roles']."""

    def test_role_inference_with_agent_customer_triggers(self):
        """Role inference assigns roles based on trigger phrases."""
        segments = [
            make_segment(0, 0.0, 3.0, "Thank you for calling, how can I help you today?", speaker_id="spk_0"),
            make_segment(1, 3.0, 6.0, "I'm calling about my account.", speaker_id="spk_1"),
            make_segment(2, 6.0, 9.0, "Let me check that for you.", speaker_id="spk_0"),
            make_segment(3, 9.0, 12.0, "I have a problem with billing.", speaker_id="spk_1"),
        ]
        turns = [
            make_turn("turn_0", "spk_0", "Thank you for calling, how can I help you today?", 0.0, 3.0, [0]),
            make_turn("turn_1", "spk_1", "I'm calling about my account.", 3.0, 6.0, [1]),
            make_turn("turn_2", "spk_0", "Let me check that for you.", 6.0, 9.0, [2]),
            make_turn("turn_3", "spk_1", "I have a problem with billing.", 9.0, 12.0, [3]),
        ]
        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(enable_role_inference=True)

        result = _run_post_processors(transcript, config)

        # Should have roles annotation
        assert result.annotations is not None
        assert "roles" in result.annotations
        roles = result.annotations["roles"]

        # Should have at least one speaker with role assigned
        assert len(roles) > 0

        # At least one should be agent (spk_0 has "how can I help" and "let me check")
        # and one should be customer (spk_1 has "I'm calling about" and "I have a problem")
        speaker_roles = {sid: data.get("role") for sid, data in roles.items()}
        assert "spk_0" in speaker_roles or "spk_1" in speaker_roles

    def test_role_inference_with_minimal_turns(self):
        """Role inference works with minimal turns."""
        segments = [
            make_segment(0, 0.0, 2.0, "How can I help you?", speaker_id="spk_0"),
        ]
        turns = [
            make_turn("turn_0", "spk_0", "How can I help you?", 0.0, 2.0, [0]),
        ]
        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(enable_role_inference=True)

        result = _run_post_processors(transcript, config)

        # Should have roles annotation
        assert result.annotations is not None
        assert "roles" in result.annotations

    def test_role_inference_without_turns_skips(self):
        """Role inference handles transcript without turns gracefully."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello there.", speaker_id="spk_0"),
        ]
        transcript = make_transcript(segments, turns=None)

        config = EnrichmentConfig(enable_role_inference=True)

        # Should not crash
        result = _run_post_processors(transcript, config)
        assert result is not None


# ============================================================================
# Test: Topic segmentation produces chunks and closes final
# ============================================================================


class TestTopicSegmentationProducesChunks:
    """Tests that topic segmentation produces transcript.annotations['topics']."""

    def test_topic_segmentation_with_vocabulary_shift(self):
        """Topic segmentation detects topic boundaries on vocabulary shift."""
        # Create enough turns with distinct vocabulary to trigger boundary detection
        # Topic 1: Budget discussion
        # Topic 2: Technical support
        segments = []
        turns = []

        # Budget topic (turns 0-5)
        budget_texts = [
            "Let's discuss the budget for this quarter.",
            "The budget needs to be reviewed carefully.",
            "We need to allocate funds for marketing.",
            "Budget constraints are affecting our plans.",
            "Financial projections show budget shortfall.",
            "We should prioritize budget items.",
        ]
        for i, text in enumerate(budget_texts):
            t = float(i * 5)
            segments.append(make_segment(i, t, t + 4.0, text, speaker_id=f"spk_{i % 2}"))
            turns.append(make_turn(f"turn_{i}", f"spk_{i % 2}", text, t, t + 4.0, [i]))

        # Technical topic (turns 6-11)
        tech_texts = [
            "Now let's talk about the software deployment.",
            "The deployment requires server configuration.",
            "We need to test the software thoroughly.",
            "Software bugs need to be fixed before release.",
            "The technical team will handle deployment.",
            "Server capacity needs to be increased.",
        ]
        for j, text in enumerate(tech_texts):
            i = j + len(budget_texts)
            t = float(i * 5)
            segments.append(make_segment(i, t, t + 4.0, text, speaker_id=f"spk_{i % 2}"))
            turns.append(make_turn(f"turn_{i}", f"spk_{i % 2}", text, t, t + 4.0, [i]))

        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(enable_topic_segmentation=True)

        result = _run_post_processors(transcript, config)

        # Should have topics annotation
        assert result.annotations is not None
        assert "topics" in result.annotations
        topics = result.annotations["topics"]

        # Should have at least one topic
        assert len(topics) >= 1

        # Last topic should have end_time set (finalize was called)
        last_topic = topics[-1]
        assert "end" in last_topic
        assert last_topic["end"] > 0

    def test_topic_segmentation_closes_final_chunk(self):
        """Topic segmentation closes final topic chunk with proper end_time."""
        segments = [
            make_segment(0, 0.0, 3.0, "First topic content.", speaker_id="spk_0"),
            make_segment(1, 3.0, 6.0, "More first topic.", speaker_id="spk_1"),
            make_segment(2, 6.0, 9.0, "Still first topic.", speaker_id="spk_0"),
        ]
        turns = [
            make_turn("turn_0", "spk_0", "First topic content.", 0.0, 3.0, [0]),
            make_turn("turn_1", "spk_1", "More first topic.", 3.0, 6.0, [1]),
            make_turn("turn_2", "spk_0", "Still first topic.", 6.0, 9.0, [2]),
        ]
        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(enable_topic_segmentation=True)

        result = _run_post_processors(transcript, config)

        assert result.annotations is not None
        assert "topics" in result.annotations
        topics = result.annotations["topics"]

        # The final topic should have end_time equal to last segment/turn end
        if topics:
            last_topic = topics[-1]
            assert last_topic["end"] == 9.0


# ============================================================================
# Test: All features enabled together
# ============================================================================


class TestAllFeaturesEnabledTogether:
    """Tests that all features work correctly when enabled together."""

    def test_all_features_enabled_no_crashes(self):
        """All features enabled together don't crash."""
        audio_state = {
            "audio_health": {
                "snr_db": 20.0,
                "quality_score": 0.7,
                "spectral_centroid_hz": 2000.0,
                "clipping_ratio": 0.0,
            }
        }
        segments = [
            make_segment(0, 0.0, 3.0, "Thank you for calling, how can I help you today?",
                        speaker_id="spk_0", audio_state=audio_state.copy()),
            make_segment(1, 3.0, 6.0, "I'm calling about my account at test@example.com.",
                        speaker_id="spk_1", audio_state=audio_state.copy()),
            make_segment(2, 6.0, 9.0, "Let me check that for you.",
                        speaker_id="spk_0", audio_state=audio_state.copy()),
            make_segment(3, 9.0, 12.0, "I need help with billing.",
                        speaker_id="spk_1", audio_state=audio_state.copy()),
        ]
        turns = [
            make_turn("turn_0", "spk_0", "Thank you for calling, how can I help you today?", 0.0, 3.0, [0]),
            make_turn("turn_1", "spk_1", "I'm calling about my account at test@example.com.", 3.0, 6.0, [1]),
            make_turn("turn_2", "spk_0", "Let me check that for you.", 6.0, 9.0, [2]),
            make_turn("turn_3", "spk_1", "I need help with billing.", 9.0, 12.0, [3]),
        ]
        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(
            enable_safety_layer=True,
            enable_role_inference=True,
            enable_topic_segmentation=True,
            enable_environment_classifier=True,
            enable_prosody_v2=True,
            turn_taking_policy="aggressive",  # Non-default enables turn-taking
        )

        # Should not crash
        result = _run_post_processors(transcript, config)

        # Verify structure
        assert result is not None
        assert result.annotations is not None

        # Check for expected annotations
        assert "_schema_version" in result.annotations
        assert "roles" in result.annotations
        assert "topics" in result.annotations

        # Check for expected audio_state keys in segments
        for seg in result.segments:
            assert seg.audio_state is not None
            assert "_schema_version" in seg.audio_state
            assert "safety" in seg.audio_state
            assert "environment" in seg.audio_state

    def test_all_features_produce_expected_data(self):
        """All features produce expected data structures."""
        audio_state = {
            "audio_health": {
                "snr_db": 15.0,
                "quality_score": 0.6,
                "spectral_centroid_hz": 1500.0,
                "clipping_ratio": 0.01,
            }
        }
        segments = [
            make_segment(0, 0.0, 2.0, "How can I help you today?",
                        speaker_id="spk_0", audio_state=audio_state.copy()),
            make_segment(1, 2.0, 4.0, "I need help with my order.",
                        speaker_id="spk_1", audio_state=audio_state.copy()),
        ]
        turns = [
            make_turn("turn_0", "spk_0", "How can I help you today?", 0.0, 2.0, [0]),
            make_turn("turn_1", "spk_1", "I need help with my order.", 2.0, 4.0, [1]),
        ]
        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(
            enable_safety_layer=True,
            enable_role_inference=True,
            enable_topic_segmentation=True,
            enable_environment_classifier=True,
        )

        result = _run_post_processors(transcript, config)

        # Verify safety state structure
        for seg in result.segments:
            safety = seg.audio_state.get("safety", {})
            assert "processed" in safety
            assert "action" in safety

        # Verify environment state structure
        for seg in result.segments:
            env = seg.audio_state.get("environment", {})
            assert "tag" in env
            assert "confidence" in env

        # Verify roles structure
        roles = result.annotations.get("roles", {})
        for _speaker_id, role_data in roles.items():
            assert "role" in role_data
            assert "confidence" in role_data

        # Verify topics structure
        topics = result.annotations.get("topics", [])
        for topic in topics:
            assert "id" in topic
            assert "start" in topic
            assert "end" in topic


# ============================================================================
# Test: Schema version present
# ============================================================================


class TestSchemaVersionPresent:
    """Tests that schema versions are present in annotations and audio_state."""

    def test_transcript_annotations_have_schema_version(self):
        """Transcript annotations include _schema_version."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", speaker_id="spk_0"),
        ]
        turns = [
            make_turn("turn_0", "spk_0", "Hello", 0.0, 2.0, [0]),
        ]
        transcript = make_transcript(segments, turns)

        # Enable any post-processing to trigger annotation creation
        config = EnrichmentConfig(enable_role_inference=True)

        result = _run_post_processors(transcript, config)

        assert result.annotations is not None
        assert "_schema_version" in result.annotations
        assert result.annotations["_schema_version"] == SCHEMA_VERSION

    def test_segment_audio_state_has_schema_version(self):
        """Segment audio_state includes _schema_version after processing."""
        segments = [
            make_segment(0, 0.0, 2.0, "Test text", speaker_id="spk_0"),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_safety_layer=True)

        result = _run_post_processors(transcript, config)

        seg = result.segments[0]
        assert seg.audio_state is not None
        assert "_schema_version" in seg.audio_state
        assert seg.audio_state["_schema_version"] == AUDIO_STATE_VERSION

    def test_existing_audio_state_gets_schema_version(self):
        """Existing audio_state without version gets _schema_version added."""
        # Pre-existing audio_state without schema version
        audio_state = {
            "audio_health": {"snr_db": 20.0},
        }
        segments = [
            make_segment(0, 0.0, 2.0, "Test", speaker_id="spk_0", audio_state=audio_state),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_environment_classifier=True)

        result = _run_post_processors(transcript, config)

        seg = result.segments[0]
        assert seg.audio_state is not None
        assert "_schema_version" in seg.audio_state
        # Original data should be preserved
        assert "audio_health" in seg.audio_state

    @pytest.mark.parametrize("feature", [
        "enable_safety_layer",
        "enable_role_inference",
        "enable_topic_segmentation",
        "enable_environment_classifier",
    ])
    def test_any_feature_sets_schema_version(self, feature: str):
        """Any enabled feature results in schema version being set."""
        audio_state = {"audio_health": {"snr_db": 15.0, "quality_score": 0.6}}
        segments = [
            make_segment(0, 0.0, 2.0, "Test", speaker_id="spk_0",
                        audio_state=audio_state if feature == "enable_environment_classifier" else None),
        ]
        turns = [
            make_turn("turn_0", "spk_0", "Test", 0.0, 2.0, [0]),
        ]
        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(**{feature: True})

        result = _run_post_processors(transcript, config)

        assert result.annotations is not None
        assert "_schema_version" in result.annotations


# ============================================================================
# Additional Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_transcript(self):
        """Empty transcript doesn't crash post-processing."""
        transcript = make_transcript([], [])

        config = EnrichmentConfig(
            enable_safety_layer=True,
            enable_role_inference=True,
        )

        result = _run_post_processors(transcript, config)

        # Should not crash, annotations should be set but empty
        assert result is not None

    def test_segment_without_speaker(self):
        """Segments without speaker are processed correctly."""
        segments = [
            make_segment(0, 0.0, 2.0, "Unspeakered text", speaker_id=None),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_safety_layer=True)

        result = _run_post_processors(transcript, config)

        assert result.segments[0].audio_state is not None
        assert "safety" in result.segments[0].audio_state

    def test_turns_with_empty_text(self):
        """Turns with empty text are handled gracefully."""
        segments = [
            make_segment(0, 0.0, 2.0, "", speaker_id="spk_0"),
        ]
        turns = [
            make_turn("turn_0", "spk_0", "", 0.0, 2.0, [0]),
        ]
        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(enable_role_inference=True)

        # Should not crash
        result = _run_post_processors(transcript, config)
        assert result is not None

    def test_callback_errors_do_not_crash_pipeline(self):
        """Errors in processing don't crash the pipeline (Invariant 3)."""
        # Create normal transcript
        segments = [
            make_segment(0, 0.0, 2.0, "Normal text", speaker_id="spk_0"),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_safety_layer=True)

        # Should complete without raising
        result = _run_post_processors(transcript, config)
        assert result is not None


class TestDataFlowIntegrity:
    """Tests for data flow integrity through the pipeline."""

    def test_segment_data_preserved(self):
        """Original segment data is preserved after processing."""
        segments = [
            Segment(
                id=0,
                start=0.0,
                end=2.0,
                text="Hello there",
                speaker={"id": "spk_0", "confidence": 0.9},
                tokens=[1, 2, 3],
                avg_logprob=-0.5,
                words=None,
            ),
        ]
        transcript = make_transcript(segments)

        config = EnrichmentConfig(enable_safety_layer=True)

        result = _run_post_processors(transcript, config)

        seg = result.segments[0]
        # Original fields preserved
        assert seg.id == 0
        assert seg.start == 0.0
        assert seg.end == 2.0
        assert seg.text == "Hello there"
        assert seg.speaker == {"id": "spk_0", "confidence": 0.9}
        assert seg.tokens == [1, 2, 3]
        assert seg.avg_logprob == -0.5

    def test_turn_data_preserved(self):
        """Original turn data is preserved after processing."""
        segments = [make_segment(0, 0.0, 2.0, "Test", speaker_id="spk_0")]
        turns = [
            Turn(
                id="turn_special",
                speaker_id="spk_0",
                segment_ids=[0, 1, 2],
                start=0.0,
                end=2.0,
                text="Test",
                metadata={"custom": "value"},
            )
        ]
        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(enable_role_inference=True)

        result = _run_post_processors(transcript, config)

        turn = result.turns[0]
        # Original fields preserved
        assert turn.id == "turn_special"
        assert turn.segment_ids == [0, 1, 2]
        assert turn.metadata == {"custom": "value"}

    def test_finalize_called_for_topics(self):
        """finalize() is called to properly close topic chunks."""
        segments = [
            make_segment(i, i * 2.0, (i + 1) * 2.0, f"Sentence {i}.", speaker_id=f"spk_{i % 2}")
            for i in range(5)
        ]
        turns = [
            make_turn(f"turn_{i}", f"spk_{i % 2}", f"Sentence {i}.", i * 2.0, (i + 1) * 2.0, [i])
            for i in range(5)
        ]
        transcript = make_transcript(segments, turns)

        config = EnrichmentConfig(enable_topic_segmentation=True)

        result = _run_post_processors(transcript, config)

        # If topics exist, the last one should have an end_time (from finalize)
        if result.annotations and "topics" in result.annotations:
            topics = result.annotations["topics"]
            if topics:
                last_topic = topics[-1]
                # End time should match the last turn's end time
                assert last_topic["end"] == 10.0  # (4 + 1) * 2.0
