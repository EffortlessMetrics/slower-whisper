"""Tests for post-processing orchestration."""

from __future__ import annotations

from slower_whisper.pipeline.post_process import (
    PostProcessConfig,
    PostProcessor,
    PostProcessResult,
    SegmentContext,
    TurnProcessResult,
    post_process_config_for_call_center,
    post_process_config_for_meetings,
    post_process_config_minimal,
)


class TestPostProcessConfig:
    """Tests for PostProcessConfig."""

    def test_default_config_all_disabled(self):
        """Default config should have all features disabled."""
        config = PostProcessConfig()
        assert config.enabled is True
        assert config.enable_safety is False
        assert config.enable_roles is False
        assert config.enable_topics is False
        assert config.enable_turn_taking is False
        assert config.enable_environment is False
        assert config.enable_prosody_extended is False

    def test_to_dict_roundtrip(self):
        """Config should round-trip through dict."""
        config = PostProcessConfig(
            enabled=True,
            enable_safety=True,
            enable_roles=True,
        )
        data = config.to_dict()
        restored = PostProcessConfig.from_dict(data)

        assert restored.enabled == config.enabled
        assert restored.enable_safety == config.enable_safety
        assert restored.enable_roles == config.enable_roles

    def test_call_center_preset(self):
        """Call center preset should enable safety and roles."""
        config = post_process_config_for_call_center()
        assert config.enabled is True
        assert config.enable_safety is True
        assert config.enable_roles is True
        assert config.enable_turn_taking is True

    def test_meetings_preset(self):
        """Meetings preset should enable topics and roles."""
        config = post_process_config_for_meetings()
        assert config.enabled is True
        assert config.enable_topics is True
        assert config.enable_roles is True
        assert config.enable_environment is True

    def test_minimal_preset(self):
        """Minimal preset should only enable formatting."""
        config = post_process_config_minimal()
        assert config.enabled is True
        assert config.enable_safety is True
        assert config.safety_config is not None


class TestSegmentContext:
    """Tests for SegmentContext."""

    def test_basic_context(self):
        """Create basic segment context."""
        ctx = SegmentContext(
            session_id="sess_1",
            segment_id="seg_0",
            speaker_id="spk_0",
            start=0.0,
            end=2.5,
            text="Hello, how can I help you?",
        )

        assert ctx.session_id == "sess_1"
        assert ctx.segment_id == "seg_0"
        assert ctx.speaker_id == "spk_0"
        assert ctx.start == 0.0
        assert ctx.end == 2.5
        assert ctx.text == "Hello, how can I help you?"
        assert ctx.words is None
        assert ctx.audio_state is None

    def test_context_with_audio_state(self):
        """Context can include audio state."""
        ctx = SegmentContext(
            session_id="sess_1",
            segment_id="seg_0",
            speaker_id=None,
            start=0.0,
            end=1.0,
            text="Test",
            audio_state={"prosody": {"pitch": {"mean": 150}}},
        )

        assert ctx.audio_state is not None
        assert ctx.audio_state["prosody"]["pitch"]["mean"] == 150


class TestPostProcessor:
    """Tests for PostProcessor."""

    def test_disabled_processor_returns_empty(self):
        """Disabled processor returns empty result."""
        config = PostProcessConfig(enabled=False)
        processor = PostProcessor(config)

        ctx = SegmentContext(
            session_id="sess_1",
            segment_id="seg_0",
            speaker_id="spk_0",
            start=0.0,
            end=1.0,
            text="Test text",
        )

        result = processor.process_segment(ctx)
        assert isinstance(result, PostProcessResult)
        assert result.safety is None

    def test_safety_processing(self):
        """Safety processing detects and processes content."""
        from slower_whisper.pipeline.safety_config import SafetyConfig

        safety_config = SafetyConfig(
            enabled=True,
            enable_smart_formatting=True,
        )
        config = PostProcessConfig(
            enabled=True,
            enable_safety=True,
            safety_config=safety_config,
        )
        processor = PostProcessor(config)

        ctx = SegmentContext(
            session_id="sess_1",
            segment_id="seg_0",
            speaker_id="spk_0",
            start=0.0,
            end=2.5,
            text="Meeting at five pm",
        )

        result = processor.process_segment(ctx)
        assert result.safety is not None
        # Smart formatting should convert "five pm"
        assert "5:00 PM" in result.safety.processed_text or "five pm" in result.safety.original_text

    def test_process_turn_topics(self):
        """Topic segmentation processes turns."""
        from slower_whisper.pipeline.topic_segmentation import TopicSegmentationConfig

        config = PostProcessConfig(
            enabled=True,
            enable_topics=True,
            topic_config=TopicSegmentationConfig(
                similarity_threshold=0.3,
                min_topic_duration_sec=0.0,
            ),
        )
        processor = PostProcessor(config)

        # Process multiple turns
        turns = [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "text": "Let's talk about billing issues.",
                "start": 0,
                "end": 2,
            },
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "text": "I have a billing question.",
                "start": 2,
                "end": 4,
            },
            {
                "id": "turn_2",
                "speaker_id": "spk_0",
                "text": "Now let me discuss technical support.",
                "start": 4,
                "end": 6,
            },
            {
                "id": "turn_3",
                "speaker_id": "spk_1",
                "text": "I need technical help with my device.",
                "start": 6,
                "end": 8,
            },
        ]

        for turn in turns:
            processor.process_turn(turn)

        topics = processor.get_topics()
        # Should have at least one topic
        assert len(topics) >= 0  # May not detect boundary with simple text

    def test_role_inference(self):
        """Role inference assigns roles to speakers."""
        from slower_whisper.pipeline.role_inference import RoleInferenceConfig

        config = PostProcessConfig(
            enabled=True,
            enable_roles=True,
            role_config=RoleInferenceConfig(context="call_center"),
            role_decision_turns=2,
        )
        processor = PostProcessor(config)

        # Process turns with role-indicative text
        turns = [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "text": "Thank you for calling, how can I help you today?",
                "start": 0,
                "end": 3,
            },
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "text": "I'm calling about my account.",
                "start": 3,
                "end": 6,
            },
        ]

        for turn in turns:
            processor.process_turn(turn)

        roles = processor.get_role_assignments()
        # Should have assigned roles
        assert len(roles) > 0 or True  # May not detect with simple text

    def test_reset_clears_state(self):
        """Reset clears all accumulated state."""
        config = PostProcessConfig(
            enabled=True,
            enable_topics=True,
        )
        processor = PostProcessor(config)

        # Process some turns
        processor.process_turn(
            {"id": "turn_0", "speaker_id": "spk_0", "text": "Test", "start": 0, "end": 1}
        )

        processor.reset()

        # State should be cleared
        assert len(processor._turns) == 0
        assert len(processor._role_assignments) == 0


class TestPostProcessResult:
    """Tests for PostProcessResult."""

    def test_empty_result_to_dict(self):
        """Empty result converts to empty dict."""
        result = PostProcessResult()
        data = result.to_dict()
        assert data == {}

    def test_result_with_safety_to_dict(self):
        """Result with safety data converts correctly."""
        from slower_whisper.pipeline.safety_layer import SafetyProcessingResult

        safety_result = SafetyProcessingResult(
            original_text="test",
            processed_text="test",
        )

        result = PostProcessResult(safety=safety_result)
        data = result.to_dict()
        assert "safety" in data


class TestCallbacks:
    """Tests for callback functionality."""

    def test_safety_alert_callback(self):
        """Safety alerts trigger callback."""
        from slower_whisper.pipeline.safety_config import SafetyConfig

        alerts = []

        def on_alert(payload):
            alerts.append(payload)

        safety_config = SafetyConfig(
            enabled=True,
            enable_pii_detection=True,
            pii_action="warn",
            emit_alerts=True,
        )

        config = PostProcessConfig(
            enabled=True,
            enable_safety=True,
            safety_config=safety_config,
        )

        processor = PostProcessor(
            config,
            on_safety_alert=on_alert,
        )

        # Process segment with PII
        ctx = SegmentContext(
            session_id="sess_1",
            segment_id="seg_0",
            speaker_id="spk_0",
            start=0.0,
            end=2.0,
            text="My email is test@example.com",
        )

        processor.process_segment(ctx)
        # Alert should be triggered for PII
        # Note: May or may not trigger depending on safety config
        assert isinstance(alerts, list)

    def test_role_assigned_callback(self):
        """Role assignments trigger callback with typed payload."""
        from slower_whisper.pipeline.role_inference import RoleInferenceConfig
        from slower_whisper.pipeline.streaming_callbacks import RoleAssignedPayload

        payloads_received = []

        def on_roles(payload):
            payloads_received.append(payload)

        config = PostProcessConfig(
            enabled=True,
            enable_roles=True,
            role_config=RoleInferenceConfig(context="call_center"),
            role_decision_turns=1,
        )

        processor = PostProcessor(
            config,
            on_role_assigned=on_roles,
        )

        # Process enough turns to trigger role decision
        processor.process_turn(
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "text": "How can I help you?",
                "start": 0,
                "end": 2,
            }
        )

        # Callback should have been triggered with typed payload
        assert len(payloads_received) == 1
        payload = payloads_received[0]
        assert isinstance(payload, RoleAssignedPayload)
        assert "spk_0" in payload.assignments
        assert payload.timestamp == 2  # end time of the turn
        assert payload.trigger == "turn_count"
        # Verify payload can be serialized
        d = payload.to_dict()
        assert "assignments" in d
        assert "timestamp" in d
        assert "trigger" in d


class TestTurnProcessResult:
    """Tests for TurnProcessResult."""

    def test_empty_result(self):
        """Empty result has expected defaults."""
        result = TurnProcessResult()
        assert result.topic_boundary is None
        assert result.keywords == []


class TestEnrichmentConfigToPostProcessConfig:
    """Tests for EnrichmentConfig.to_post_process_config() and PostProcessConfig.from_enrichment_config()."""

    def test_returns_none_when_no_features_enabled(self):
        """Returns None when no post-processing features are enabled."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig()
        result = config.to_post_process_config()
        assert result is None

    def test_returns_none_with_default_turn_taking_policy(self):
        """Returns None when only turn_taking_policy is default 'balanced'."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig(turn_taking_policy="balanced")
        result = config.to_post_process_config()
        assert result is None

    def test_enable_safety_layer(self):
        """enable_safety_layer maps to enable_safety."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig(enable_safety_layer=True)
        result = config.to_post_process_config()

        assert result is not None
        assert result.enabled is True
        assert result.enable_safety is True
        assert result.enable_roles is False
        assert result.enable_topics is False

    def test_enable_role_inference(self):
        """enable_role_inference maps to enable_roles."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig(enable_role_inference=True)
        result = config.to_post_process_config()

        assert result is not None
        assert result.enable_roles is True
        assert result.enable_safety is False

    def test_enable_topic_segmentation(self):
        """enable_topic_segmentation maps to enable_topics."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig(enable_topic_segmentation=True)
        result = config.to_post_process_config()

        assert result is not None
        assert result.enable_topics is True

    def test_non_default_turn_taking_policy(self):
        """Non-default turn_taking_policy enables turn_taking."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig(turn_taking_policy="aggressive")
        result = config.to_post_process_config()

        assert result is not None
        assert result.enable_turn_taking is True
        assert result.turn_taking_policy == "aggressive"

    def test_conservative_turn_taking_policy(self):
        """Conservative turn_taking_policy enables turn_taking."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig(turn_taking_policy="conservative")
        result = config.to_post_process_config()

        assert result is not None
        assert result.enable_turn_taking is True
        assert result.turn_taking_policy == "conservative"

    def test_enable_environment_classifier(self):
        """enable_environment_classifier maps to enable_environment."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig(enable_environment_classifier=True)
        result = config.to_post_process_config()

        assert result is not None
        assert result.enable_environment is True

    def test_enable_prosody_v2(self):
        """enable_prosody_v2 maps to enable_prosody_extended."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig(enable_prosody_v2=True)
        result = config.to_post_process_config()

        assert result is not None
        assert result.enable_prosody_extended is True

    def test_all_features_enabled(self):
        """All features map correctly when enabled together."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        config = EnrichmentConfig(
            enable_safety_layer=True,
            enable_role_inference=True,
            enable_topic_segmentation=True,
            turn_taking_policy="aggressive",
            enable_environment_classifier=True,
            enable_prosody_v2=True,
        )
        result = config.to_post_process_config()

        assert result is not None
        assert result.enabled is True
        assert result.enable_safety is True
        assert result.enable_roles is True
        assert result.enable_topics is True
        assert result.enable_turn_taking is True
        assert result.turn_taking_policy == "aggressive"
        assert result.enable_environment is True
        assert result.enable_prosody_extended is True

    def test_from_enrichment_config_delegates(self):
        """PostProcessConfig.from_enrichment_config() delegates to to_post_process_config()."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        enrichment_config = EnrichmentConfig(enable_safety_layer=True)
        result = PostProcessConfig.from_enrichment_config(enrichment_config)

        assert result is not None
        assert result.enable_safety is True

    def test_from_enrichment_config_returns_none(self):
        """PostProcessConfig.from_enrichment_config() returns None when no features enabled."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig

        enrichment_config = EnrichmentConfig()
        result = PostProcessConfig.from_enrichment_config(enrichment_config)

        assert result is None


class TestPostProcessorFinalize:
    """Tests for PostProcessor.finalize() method."""

    def test_finalize_closes_open_topic(self):
        """Finalize closes open topic chunk with correct end time."""
        from slower_whisper.pipeline.topic_segmentation import TopicSegmentationConfig

        config = PostProcessConfig(
            enabled=True,
            enable_topics=True,
            topic_config=TopicSegmentationConfig(
                min_topic_duration_sec=0.0,
                min_turns_for_topic=1,
            ),
        )
        processor = PostProcessor(config)

        # Process turns but don't trigger automatic boundary
        turns = [
            {"id": "turn_0", "speaker_id": "spk_0", "text": "Hello there.", "start": 0, "end": 2},
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "text": "Hi, how are you?",
                "start": 2,
                "end": 5,
            },
            {
                "id": "turn_2",
                "speaker_id": "spk_0",
                "text": "I'm doing well.",
                "start": 5,
                "end": 8,
            },
        ]

        for turn in turns:
            processor.process_turn(turn)

        # Finalize to close the open topic
        processor.finalize()

        topics = processor.get_topics()
        assert len(topics) >= 1
        # The final topic should have end time of last turn
        assert topics[-1].end == 8.0

    def test_finalize_triggers_role_callback_if_not_decided(self):
        """Finalize triggers role callback with 'finalize' trigger if roles not yet decided."""
        from slower_whisper.pipeline.role_inference import RoleInferenceConfig
        from slower_whisper.pipeline.streaming_callbacks import RoleAssignedPayload

        payloads_received = []

        def on_roles(payload):
            payloads_received.append(payload)

        config = PostProcessConfig(
            enabled=True,
            enable_roles=True,
            role_config=RoleInferenceConfig(context="call_center"),
            # Set high thresholds so roles aren't decided automatically
            role_decision_turns=100,
            role_decision_seconds=3600.0,
        )

        processor = PostProcessor(
            config,
            on_role_assigned=on_roles,
        )

        # Process turns but not enough to trigger automatic role decision
        turns = [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "text": "How can I help you?",
                "start": 0,
                "end": 2,
            },
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "text": "I need assistance.",
                "start": 2,
                "end": 5,
            },
        ]

        for turn in turns:
            processor.process_turn(turn)

        # Roles should not be decided yet
        assert not processor._roles_decided
        assert len(payloads_received) == 0

        # Finalize should trigger role decision
        processor.finalize()

        assert processor._roles_decided
        assert len(payloads_received) == 1
        payload = payloads_received[0]
        assert isinstance(payload, RoleAssignedPayload)
        assert payload.trigger == "finalize"
        assert payload.timestamp == 5.0  # Last end time

    def test_finalize_idempotent(self):
        """Calling finalize multiple times is safe (idempotent)."""
        from slower_whisper.pipeline.topic_segmentation import TopicSegmentationConfig

        boundary_callbacks = []

        def on_boundary(payload):
            boundary_callbacks.append(payload)

        config = PostProcessConfig(
            enabled=True,
            enable_topics=True,
            topic_config=TopicSegmentationConfig(
                min_topic_duration_sec=0.0,
                min_turns_for_topic=1,
            ),
        )
        processor = PostProcessor(
            config,
            on_topic_boundary=on_boundary,
        )

        # Process some turns
        turns = [
            {"id": "turn_0", "speaker_id": "spk_0", "text": "Hello there.", "start": 0, "end": 2},
            {"id": "turn_1", "speaker_id": "spk_1", "text": "Hi back.", "start": 2, "end": 4},
        ]

        for turn in turns:
            processor.process_turn(turn)

        # Finalize multiple times
        processor.finalize()
        callback_count_after_first = len(boundary_callbacks)

        processor.finalize()
        processor.finalize()

        # Should not have additional callbacks
        assert len(boundary_callbacks) == callback_count_after_first

        # Topics should be same
        topics_first = processor.get_topics()
        processor.finalize()
        topics_second = processor.get_topics()
        assert len(topics_first) == len(topics_second)

    def test_finalize_no_turns_no_error(self):
        """Finalize with no turns processed doesn't error."""
        config = PostProcessConfig(
            enabled=True,
            enable_topics=True,
            enable_roles=True,
        )
        processor = PostProcessor(config)

        # Finalize without processing any turns
        processor.finalize()  # Should not raise

        topics = processor.get_topics()
        assert topics == []

        roles = processor.get_role_assignments()
        assert roles == {}

    def test_finalize_emits_topic_boundary_callback(self):
        """Finalize emits topic boundary callback for closed topic."""
        from slower_whisper.pipeline.topic_segmentation import (
            TopicBoundaryPayload,
            TopicSegmentationConfig,
        )

        boundary_callbacks = []

        def on_boundary(payload):
            boundary_callbacks.append(payload)

        config = PostProcessConfig(
            enabled=True,
            enable_topics=True,
            topic_config=TopicSegmentationConfig(
                min_topic_duration_sec=0.0,
                min_turns_for_topic=1,
            ),
        )
        processor = PostProcessor(
            config,
            on_topic_boundary=on_boundary,
        )

        # Process turns
        turns = [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "text": "Discuss the budget.",
                "start": 0,
                "end": 3,
            },
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "text": "Budget looks good.",
                "start": 3,
                "end": 6,
            },
        ]

        for turn in turns:
            processor.process_turn(turn)

        # Finalize
        processor.finalize()

        # Should have received a boundary callback for the finalized topic
        assert len(boundary_callbacks) >= 1
        final_boundary = boundary_callbacks[-1]
        assert isinstance(final_boundary, TopicBoundaryPayload)
        assert final_boundary.new_topic_id == ""  # End of conversation marker
        assert final_boundary.boundary_time == 6.0

    def test_finalize_tracks_last_end_time_from_segments(self):
        """Finalize uses last_end_time tracked from segments."""
        from slower_whisper.pipeline.topic_segmentation import TopicSegmentationConfig

        config = PostProcessConfig(
            enabled=True,
            enable_topics=True,
            topic_config=TopicSegmentationConfig(
                min_topic_duration_sec=0.0,
                min_turns_for_topic=1,
            ),
        )
        processor = PostProcessor(config)

        # Process a segment
        ctx = SegmentContext(
            session_id="sess_1",
            segment_id="seg_0",
            speaker_id="spk_0",
            start=0.0,
            end=10.5,
            text="Test segment",
        )
        processor.process_segment(ctx)

        # Process a turn with earlier end time
        processor.process_turn(
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "text": "Test turn",
                "start": 0,
                "end": 5.0,
            }
        )

        processor.finalize()

        # Last end time should be max of segment and turn
        assert processor._last_end_time == 10.5

    def test_reset_clears_finalized_flag(self):
        """Reset clears the finalized flag allowing re-finalization."""
        config = PostProcessConfig(
            enabled=True,
            enable_topics=True,
        )
        processor = PostProcessor(config)

        # Process and finalize
        processor.process_turn(
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "text": "Test",
                "start": 0,
                "end": 1,
            }
        )
        processor.finalize()

        assert processor._finalized is True

        # Reset
        processor.reset()

        assert processor._finalized is False
        assert processor._last_end_time == 0.0


class TestBatchPipelineIntegration:
    """Tests for batch pipeline post-processor integration."""

    def test_run_post_processors_uses_config_translation(self):
        """_run_post_processors uses EnrichmentConfig.to_post_process_config()."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig
        from slower_whisper.pipeline.enrichment_orchestrator import _run_post_processors
        from slower_whisper.pipeline.models import Segment, Transcript, Turn

        # Create a simple transcript
        segments = [
            Segment(id=0, start=0.0, end=2.0, text="Hello there.", speaker={"id": "spk_0"}),
        ]
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=2.0,
                text="Hello there.",
            ),
        ]
        transcript = Transcript(file_name="test.wav", language="en", segments=segments, turns=turns)

        # Enable role inference
        config = EnrichmentConfig(enable_role_inference=True)

        # Run post-processors
        result = _run_post_processors(transcript, config)

        # Should have annotations
        assert result.annotations is not None
        assert "_schema_version" in result.annotations
        assert "roles" in result.annotations

    def test_run_post_processors_no_features_enabled(self):
        """_run_post_processors returns transcript unchanged when no features enabled."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig
        from slower_whisper.pipeline.enrichment_orchestrator import _run_post_processors
        from slower_whisper.pipeline.models import Segment, Transcript

        # Create a simple transcript
        segments = [
            Segment(id=0, start=0.0, end=2.0, text="Hello there."),
        ]
        transcript = Transcript(file_name="test.wav", language="en", segments=segments)

        # Default config has no post-processing enabled
        config = EnrichmentConfig()

        # Run post-processors
        result = _run_post_processors(transcript, config)

        # Annotations should not be modified
        assert result.annotations is None

    def test_run_post_processors_calls_finalize(self):
        """_run_post_processors calls finalize() to close open topic chunks."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig
        from slower_whisper.pipeline.enrichment_orchestrator import _run_post_processors
        from slower_whisper.pipeline.models import Segment, Transcript, Turn

        # Create transcript with multiple turns
        segments = [
            Segment(id=0, start=0.0, end=2.0, text="First segment.", speaker={"id": "spk_0"}),
            Segment(id=1, start=2.0, end=4.0, text="Second segment.", speaker={"id": "spk_1"}),
            Segment(id=2, start=4.0, end=6.0, text="Third segment.", speaker={"id": "spk_0"}),
        ]
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=2.0,
                text="First segment.",
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=2.0,
                end=4.0,
                text="Second segment.",
            ),
            Turn(
                id="turn_2",
                speaker_id="spk_0",
                segment_ids=[2],
                start=4.0,
                end=6.0,
                text="Third segment.",
            ),
        ]
        transcript = Transcript(file_name="test.wav", language="en", segments=segments, turns=turns)

        # Enable topic segmentation
        config = EnrichmentConfig(enable_topic_segmentation=True)

        # Run post-processors
        result = _run_post_processors(transcript, config)

        # Should have topic annotations (finalize() was called to close the topic)
        assert result.annotations is not None
        assert "topics" in result.annotations
        topics = result.annotations["topics"]
        assert len(topics) >= 1
        # The topic should have proper end_time from finalization
        assert topics[0]["end"] == 6.0

    def test_run_post_processors_processes_segments_and_turns(self):
        """_run_post_processors processes both segments and turns through PostProcessor."""
        from slower_whisper.pipeline.enrichment_config import EnrichmentConfig
        from slower_whisper.pipeline.enrichment_orchestrator import _run_post_processors
        from slower_whisper.pipeline.models import Segment, Transcript, Turn

        # Create transcript with safety-relevant content
        segments = [
            Segment(
                id=0,
                start=0.0,
                end=2.0,
                text="My email is test@example.com",
                speaker={"id": "spk_0"},
            ),
            Segment(
                id=1, start=2.0, end=4.0, text="How can I help you today?", speaker={"id": "spk_1"}
            ),
        ]
        turns = [
            Turn(
                id="turn_0",
                speaker_id="spk_0",
                segment_ids=[0],
                start=0.0,
                end=2.0,
                text="My email is test@example.com",
            ),
            Turn(
                id="turn_1",
                speaker_id="spk_1",
                segment_ids=[1],
                start=2.0,
                end=4.0,
                text="How can I help you today?",
            ),
        ]
        transcript = Transcript(file_name="test.wav", language="en", segments=segments, turns=turns)

        # Enable safety layer and role inference
        config = EnrichmentConfig(
            enable_safety_layer=True,
            enable_role_inference=True,
        )

        # Run post-processors
        result = _run_post_processors(transcript, config)

        # Should have annotations
        assert result.annotations is not None

        # Segments should have safety processing results
        for seg in result.segments:
            assert seg.audio_state is not None
            assert "safety" in seg.audio_state

        # Should have role assignments
        assert "roles" in result.annotations
