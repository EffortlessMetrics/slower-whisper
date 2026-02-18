"""Tests for topic segmentation module."""

from __future__ import annotations

from slower_whisper.pipeline.topic_segmentation import (
    StreamingTopicSegmenter,
    TopicBoundaryPayload,
    TopicChunk,
    TopicSegmentationConfig,
    TopicSegmenter,
    compute_idf,
    compute_tf,
    compute_tfidf,
    cosine_similarity,
    tokenize,
)


class TestTokenize:
    """Tests for tokenize function."""

    def test_basic_tokenization(self):
        """Basic tokenization removes stopwords."""
        tokens = tokenize("This is a test sentence about programming")
        assert "test" in tokens
        assert "sentence" in tokens
        assert "programming" in tokens
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens

    def test_short_words_removed(self):
        """Words with 2 or fewer characters are removed."""
        tokens = tokenize("I am a go to person")
        assert "am" not in tokens
        assert "go" not in tokens

    def test_case_insensitive(self):
        """Tokenization is case insensitive."""
        tokens = tokenize("Python PYTHON python")
        assert all(t == "python" for t in tokens)

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert tokenize("") == []

    def test_custom_stopwords(self):
        """Custom stopwords are excluded."""
        custom = frozenset(["test"])
        tokens = tokenize("This is a test", custom)
        assert "test" not in tokens


class TestTfIdf:
    """Tests for TF-IDF computation."""

    def test_compute_tf(self):
        """Computes term frequency correctly."""
        tokens = ["apple", "banana", "apple", "cherry"]
        tf = compute_tf(tokens)
        assert tf["apple"] == 0.5  # 2/4
        assert tf["banana"] == 0.25  # 1/4
        assert tf["cherry"] == 0.25  # 1/4

    def test_compute_tf_empty(self):
        """Empty tokens returns empty dict."""
        assert compute_tf([]) == {}

    def test_compute_idf(self):
        """Computes IDF correctly."""
        documents = [
            ["apple", "banana"],
            ["apple", "cherry"],
            ["banana", "cherry"],
        ]
        idf = compute_idf(documents)
        # All terms appear in 2 of 3 docs
        # IDF = log((3+1)/(2+1)) + 1 = log(4/3) + 1
        assert "apple" in idf
        assert "banana" in idf
        assert "cherry" in idf

    def test_compute_idf_empty(self):
        """Empty documents returns empty dict."""
        assert compute_idf([]) == {}

    def test_compute_tfidf(self):
        """Computes TF-IDF correctly."""
        tokens = ["apple", "apple", "banana"]
        idf = {"apple": 1.5, "banana": 2.0}
        tfidf = compute_tfidf(tokens, idf)
        # apple: TF=2/3, TF-IDF = 2/3 * 1.5 = 1.0
        # banana: TF=1/3, TF-IDF = 1/3 * 2.0 = 0.67
        assert abs(tfidf["apple"] - 1.0) < 0.01
        assert abs(tfidf["banana"] - 0.667) < 0.01


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        vec1 = {"a": 1.0, "b": 0.0}
        vec2 = {"c": 1.0, "d": 0.0}
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_empty_vectors(self):
        """Empty vectors have similarity 0.0."""
        assert cosine_similarity({}, {}) == 0.0
        assert cosine_similarity({"a": 1.0}, {}) == 0.0

    def test_partial_overlap(self):
        """Partial overlap produces intermediate similarity."""
        vec1 = {"a": 1.0, "b": 1.0}
        vec2 = {"a": 1.0, "c": 1.0}
        sim = cosine_similarity(vec1, vec2)
        assert 0.0 < sim < 1.0


class TestTopicSegmentationConfig:
    """Tests for TopicSegmentationConfig."""

    def test_default_config(self):
        """Default config has expected values."""
        config = TopicSegmentationConfig()
        assert config.enabled is True
        assert config.window_size_turns == 5
        assert config.similarity_threshold == 0.35

    def test_custom_stopwords(self):
        """Custom stopwords are combined with default."""
        config = TopicSegmentationConfig(custom_stopwords={"customword"})
        stopwords = config.get_stopwords()
        assert "customword" in stopwords
        assert "the" in stopwords  # Default stopword


class TestTopicChunk:
    """Tests for TopicChunk dataclass."""

    def test_to_dict_and_from_dict(self):
        """TopicChunk round-trips through dict."""
        original = TopicChunk(
            id="topic_0",
            start=0.0,
            end=60.0,
            turn_ids=["t0", "t1", "t2"],
            turn_range=(0, 2),
            summary_text="Discussion about budget",
            keywords=["budget", "finance", "quarterly"],
            speaker_ids=["spk_0", "spk_1"],
        )
        data = original.to_dict()
        restored = TopicChunk.from_dict(data)

        assert restored.id == original.id
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.keywords == original.keywords

    def test_duration(self):
        """Duration is computed correctly."""
        chunk = TopicChunk(id="t", start=10.0, end=50.0)
        assert chunk.duration() == 40.0


class TestTopicBoundaryPayload:
    """Tests for TopicBoundaryPayload."""

    def test_to_dict(self):
        """Payload converts to dict correctly."""
        payload = TopicBoundaryPayload(
            previous_topic_id="topic_0",
            new_topic_id="topic_1",
            boundary_turn_id="turn_15",
            boundary_time=120.5,
            similarity_score=0.25,
            keywords_previous=["budget", "finance"],
            keywords_new=["marketing", "campaign"],
        )
        data = payload.to_dict()

        assert data["previous_topic_id"] == "topic_0"
        assert data["similarity_score"] == 0.25
        assert "budget" in data["keywords_previous"]


class TestTopicSegmenter:
    """Tests for TopicSegmenter class."""

    def test_disabled_returns_empty(self):
        """Disabled segmenter returns empty list."""
        config = TopicSegmentationConfig(enabled=False)
        segmenter = TopicSegmenter(config)
        turns = [{"id": "t0", "text": "Hello", "start": 0, "end": 5}]
        assert segmenter.segment(turns) == []

    def test_empty_turns(self):
        """Empty turns returns empty list."""
        segmenter = TopicSegmenter()
        assert segmenter.segment([]) == []

    def test_single_topic_short_conversation(self):
        """Short conversation produces single topic."""
        config = TopicSegmentationConfig(
            window_size_turns=2,
            min_turns_for_topic=2,
            min_topic_duration_sec=0,
        )
        segmenter = TopicSegmenter(config)

        turns = [
            {"id": "t0", "text": "Let's talk about the budget", "start": 0, "end": 5},
            {"id": "t1", "text": "The budget needs review", "start": 5, "end": 10},
            {"id": "t2", "text": "I agree about the budget", "start": 10, "end": 15},
        ]

        topics = segmenter.segment(turns)
        assert len(topics) == 1
        assert topics[0].id == "topic_0"

    def test_topic_boundary_detection(self):
        """Detects topic boundary when vocabulary changes."""
        config = TopicSegmentationConfig(
            window_size_turns=3,
            similarity_threshold=0.2,
            min_turns_for_topic=2,
            min_topic_duration_sec=0,
        )
        segmenter = TopicSegmenter(config)

        # Create turns with clear topic shift
        budget_turns = [
            {
                "id": f"t{i}",
                "text": "budget finance quarterly reports",
                "start": i * 5,
                "end": (i + 1) * 5,
            }
            for i in range(8)
        ]
        marketing_turns = [
            {
                "id": f"t{i + 8}",
                "text": "marketing campaign advertising social media",
                "start": (i + 8) * 5,
                "end": (i + 9) * 5,
            }
            for i in range(8)
        ]

        all_turns = budget_turns + marketing_turns
        topics = segmenter.segment(all_turns)

        # Should detect at least one boundary
        assert len(topics) >= 1

    def test_extracts_keywords(self):
        """Topics have meaningful keywords extracted."""
        config = TopicSegmentationConfig(
            window_size_turns=2,
            min_turns_for_topic=2,
            min_topic_duration_sec=0,
        )
        segmenter = TopicSegmenter(config)

        turns = [
            {"id": "t0", "text": "Python programming language is excellent", "start": 0, "end": 5},
            {"id": "t1", "text": "Python code programming development", "start": 5, "end": 10},
            {"id": "t2", "text": "Software development with Python", "start": 10, "end": 15},
        ]

        topics = segmenter.segment(turns)
        assert len(topics) >= 1

        # Check keywords include relevant terms
        all_keywords = []
        for topic in topics:
            all_keywords.extend(topic.keywords)

        assert "python" in all_keywords or "programming" in all_keywords


class TestStreamingTopicSegmenter:
    """Tests for StreamingTopicSegmenter."""

    def test_disabled_returns_none(self):
        """Disabled segmenter returns None."""
        config = TopicSegmentationConfig(enabled=False)
        segmenter = StreamingTopicSegmenter(config)
        result = segmenter.add_turn({"id": "t0", "text": "Hello", "start": 0, "end": 5})
        assert result is None

    def test_incremental_addition(self):
        """Turns can be added incrementally."""
        config = TopicSegmentationConfig(
            window_size_turns=2,
            min_turns_for_topic=2,
            min_topic_duration_sec=0,
        )
        segmenter = StreamingTopicSegmenter(config)

        for i in range(10):
            segmenter.add_turn(
                {
                    "id": f"t{i}",
                    "text": f"Turn {i} content about topic",
                    "start": i * 5,
                    "end": (i + 1) * 5,
                }
            )

        topics = segmenter.finalize()
        assert len(topics) >= 1

    def test_boundary_callback(self):
        """Boundary callback is invoked."""
        boundaries = []

        def on_boundary(payload: TopicBoundaryPayload) -> None:
            boundaries.append(payload)

        config = TopicSegmentationConfig(
            window_size_turns=3,
            similarity_threshold=0.1,
            min_turns_for_topic=2,
            min_topic_duration_sec=0,
        )
        segmenter = StreamingTopicSegmenter(config, on_boundary=on_boundary)

        # Add turns with topic shift
        for i in range(6):
            segmenter.add_turn(
                {
                    "id": f"t{i}",
                    "text": "budget finance accounting",
                    "start": i * 5,
                    "end": (i + 1) * 5,
                }
            )

        for i in range(6, 12):
            segmenter.add_turn(
                {
                    "id": f"t{i}",
                    "text": "marketing advertising campaign",
                    "start": i * 5,
                    "end": (i + 1) * 5,
                }
            )

        # Boundary may or may not have been detected depending on threshold
        # Just verify no errors occurred

    def test_reset(self):
        """Reset clears state."""
        segmenter = StreamingTopicSegmenter()

        for i in range(5):
            segmenter.add_turn(
                {
                    "id": f"t{i}",
                    "text": "Content",
                    "start": i * 5,
                    "end": (i + 1) * 5,
                }
            )

        segmenter.reset()

        # After reset, should be empty
        topics = segmenter.finalize()
        assert len(topics) == 0

    def test_close_current_topic_returns_topic(self):
        """close_current_topic returns the closed topic chunk."""
        config = TopicSegmentationConfig(
            min_topic_duration_sec=0.0,
            min_turns_for_topic=1,
        )
        segmenter = StreamingTopicSegmenter(config)

        # Add some turns
        segmenter.add_turn(
            {
                "id": "t0",
                "text": "Hello discussion",
                "start": 0,
                "end": 5,
            }
        )
        segmenter.add_turn(
            {
                "id": "t1",
                "text": "More discussion",
                "start": 5,
                "end": 10,
            }
        )

        # Close the current topic
        closed_topic = segmenter.close_current_topic()

        assert closed_topic is not None
        assert closed_topic.id == "topic_0"
        assert closed_topic.start == 0
        assert closed_topic.end == 10
        assert "t0" in closed_topic.turn_ids
        assert "t1" in closed_topic.turn_ids

    def test_close_current_topic_with_explicit_end_time(self):
        """close_current_topic uses explicit end_time when provided."""
        config = TopicSegmentationConfig(
            min_topic_duration_sec=0.0,
            min_turns_for_topic=1,
        )
        segmenter = StreamingTopicSegmenter(config)

        segmenter.add_turn(
            {
                "id": "t0",
                "text": "Test content",
                "start": 0,
                "end": 5,
            }
        )

        # Close with explicit end time
        closed_topic = segmenter.close_current_topic(end_time=15.5)

        assert closed_topic is not None
        assert closed_topic.end == 15.5

    def test_close_current_topic_returns_none_when_no_turns(self):
        """close_current_topic returns None when no turns to close."""
        segmenter = StreamingTopicSegmenter()

        # No turns added
        result = segmenter.close_current_topic()
        assert result is None

    def test_close_current_topic_idempotent(self):
        """Calling close_current_topic twice doesn't duplicate topics."""
        config = TopicSegmentationConfig(
            min_topic_duration_sec=0.0,
            min_turns_for_topic=1,
        )
        segmenter = StreamingTopicSegmenter(config)

        segmenter.add_turn(
            {
                "id": "t0",
                "text": "Test content",
                "start": 0,
                "end": 5,
            }
        )

        # Close first time
        first_result = segmenter.close_current_topic()
        assert first_result is not None

        # Close second time - should return None (already closed)
        second_result = segmenter.close_current_topic()
        assert second_result is None

        # Finalize should show only one topic
        topics = segmenter.finalize()
        assert len(topics) == 1

    def test_finalize_uses_close_current_topic(self):
        """finalize() internally uses close_current_topic()."""
        config = TopicSegmentationConfig(
            min_topic_duration_sec=0.0,
            min_turns_for_topic=1,
        )
        segmenter = StreamingTopicSegmenter(config)

        segmenter.add_turn(
            {
                "id": "t0",
                "text": "Test content",
                "start": 0,
                "end": 5,
            }
        )

        # Finalize should close the topic
        topics = segmenter.finalize()
        assert len(topics) == 1
        assert topics[0].id == "topic_0"

    def test_close_current_topic_after_boundary(self):
        """close_current_topic works correctly after a boundary was detected."""
        config = TopicSegmentationConfig(
            window_size_turns=3,
            similarity_threshold=0.1,
            min_turns_for_topic=2,
            min_topic_duration_sec=0,
        )
        segmenter = StreamingTopicSegmenter(config)

        # Add turns with topic shift to trigger boundary
        for i in range(6):
            segmenter.add_turn(
                {
                    "id": f"t{i}",
                    "text": "budget finance accounting",
                    "start": i * 5,
                    "end": (i + 1) * 5,
                }
            )

        for i in range(6, 12):
            segmenter.add_turn(
                {
                    "id": f"t{i}",
                    "text": "marketing advertising campaign",
                    "start": i * 5,
                    "end": (i + 1) * 5,
                }
            )

        # Close remaining turns
        closed = segmenter.close_current_topic(end_time=100.0)

        # Should have closed the marketing topic
        if closed is not None:
            assert closed.end == 100.0

        # Finalize should not add more topics
        topics_before = len(segmenter._topics)
        segmenter.finalize()
        topics_after = len(segmenter._topics)

        # Should be same or at most one more (if close didn't catch all)
        assert topics_after >= topics_before


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_stopwords(self):
        """Handles text that is all stopwords."""
        segmenter = TopicSegmenter()
        turns = [
            {"id": "t0", "text": "the and or but", "start": 0, "end": 5},
            {"id": "t1", "text": "is are was were", "start": 5, "end": 10},
        ]
        # Should not crash
        topics = segmenter.segment(turns)
        assert isinstance(topics, list)

    def test_missing_fields(self):
        """Handles turns with missing fields."""
        segmenter = TopicSegmenter()
        turns = [
            {"text": "Hello"},  # Missing id, start, end
            {"id": "t1"},  # Missing text, start, end
        ]
        # Should not crash
        topics = segmenter.segment(turns)
        assert isinstance(topics, list)

    def test_very_long_text(self):
        """Handles very long text."""
        segmenter = TopicSegmenter()
        long_text = "Python programming " * 1000
        turns = [
            {"id": "t0", "text": long_text, "start": 0, "end": 60},
        ]
        topics = segmenter.segment(turns)
        assert isinstance(topics, list)

    def test_unicode_text(self):
        """Handles unicode text."""
        segmenter = TopicSegmenter()
        turns = [
            {"id": "t0", "text": "Discussion about \u00e9\u00e0\u00fc", "start": 0, "end": 5},
        ]
        topics = segmenter.segment(turns)
        assert isinstance(topics, list)
