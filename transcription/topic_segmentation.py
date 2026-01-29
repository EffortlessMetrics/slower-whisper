"""Topic segmentation for conversation analysis.

This module provides vocabulary-based topic boundary detection using
rolling window TF-IDF similarity to identify when conversation topics
change.

Features:
- Rolling window TF-IDF similarity calculation
- Configurable window sizes and thresholds
- Topic boundary events for streaming
- Topic chunks for transcript storage
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Common English stopwords to exclude from TF-IDF
STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for", "from", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "ll", "m", "me", "might", "more", "most", "must", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "ok", "okay", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "re", "s", "same", "shall", "she", "should", "so", "some", "such",
    "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "um",
    "uh", "under", "until", "up", "ve", "very", "was", "we", "well", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "yeah", "yes", "you", "your", "yours", "yourself",
    "yourselves",
])


@dataclass(slots=True)
class TopicSegmentationConfig:
    """Configuration for topic segmentation.

    Attributes:
        enabled: Whether topic segmentation is enabled.
        window_size_turns: Number of turns in each comparison window.
        similarity_threshold: Below this threshold, a boundary is detected.
        min_topic_duration_sec: Minimum duration before a new topic can start.
        max_topic_duration_sec: Maximum duration before forcing topic boundary.
        min_turns_for_topic: Minimum turns before considering a topic change.
        use_stemming: Whether to apply basic stemming.
        custom_stopwords: Additional stopwords to exclude.
    """

    enabled: bool = True
    window_size_turns: int = 5
    similarity_threshold: float = 0.35
    min_topic_duration_sec: float = 30.0
    max_topic_duration_sec: float = 300.0
    min_turns_for_topic: int = 3
    use_stemming: bool = False  # Basic stemming can be enabled
    custom_stopwords: set[str] = field(default_factory=set)

    def get_stopwords(self) -> frozenset[str]:
        """Get combined stopwords set."""
        return STOPWORDS | frozenset(self.custom_stopwords)


@dataclass(slots=True)
class TopicChunk:
    """A topic segment within a conversation.

    Attributes:
        id: Topic chunk identifier (e.g., "topic_0").
        start: Start time in seconds.
        end: End time in seconds.
        turn_ids: IDs of turns in this topic.
        turn_range: Tuple of (first_turn_index, last_turn_index).
        summary_text: Concatenated text from the topic (for analysis).
        keywords: Top keywords identified in this topic.
        speaker_ids: Speakers participating in this topic.
    """

    id: str
    start: float
    end: float
    turn_ids: list[str] = field(default_factory=list)
    turn_range: tuple[int, int] = (0, 0)
    summary_text: str = ""
    keywords: list[str] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)

    def duration(self) -> float:
        """Get topic duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "turn_ids": list(self.turn_ids),
            "turn_range": list(self.turn_range),
            "summary_text": self.summary_text,
            "keywords": list(self.keywords),
            "speaker_ids": list(self.speaker_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicChunk:
        """Create from dictionary."""
        turn_range = data.get("turn_range", [0, 0])
        return cls(
            id=data.get("id", ""),
            start=data.get("start", 0.0),
            end=data.get("end", 0.0),
            turn_ids=list(data.get("turn_ids", [])),
            turn_range=tuple(turn_range) if len(turn_range) == 2 else (0, 0),
            summary_text=data.get("summary_text", ""),
            keywords=list(data.get("keywords", [])),
            speaker_ids=list(data.get("speaker_ids", [])),
        )


@dataclass(slots=True)
class TopicBoundaryPayload:
    """Payload for topic boundary events.

    Attributes:
        previous_topic_id: ID of the topic that ended.
        new_topic_id: ID of the new topic starting.
        boundary_turn_id: Turn ID where boundary was detected.
        boundary_time: Time of the boundary in seconds.
        similarity_score: Similarity score that triggered boundary.
        keywords_previous: Keywords from previous topic.
        keywords_new: Keywords from new topic (preliminary).
    """

    previous_topic_id: str
    new_topic_id: str
    boundary_turn_id: str
    boundary_time: float
    similarity_score: float
    keywords_previous: list[str] = field(default_factory=list)
    keywords_new: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "previous_topic_id": self.previous_topic_id,
            "new_topic_id": self.new_topic_id,
            "boundary_turn_id": self.boundary_turn_id,
            "boundary_time": self.boundary_time,
            "similarity_score": self.similarity_score,
            "keywords_previous": list(self.keywords_previous),
            "keywords_new": list(self.keywords_new),
        }


def tokenize(text: str, stopwords: frozenset[str] | None = None) -> list[str]:
    """Tokenize text into words, removing stopwords.

    Args:
        text: Text to tokenize.
        stopwords: Set of words to exclude.

    Returns:
        List of tokens.
    """
    if stopwords is None:
        stopwords = STOPWORDS

    # Simple tokenization: lowercase and split on non-alphanumeric
    tokens = re.findall(r"\b[a-z]+\b", text.lower())

    # Remove stopwords and very short tokens
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency for tokens.

    Args:
        tokens: List of tokens.

    Returns:
        Dictionary mapping token to TF score.
    """
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency across documents.

    Args:
        documents: List of tokenized documents.

    Returns:
        Dictionary mapping token to IDF score.
    """
    if not documents:
        return {}

    n_docs = len(documents)
    doc_freq: Counter[str] = Counter()

    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_freq[term] += 1

    # IDF with smoothing: log((N + 1) / (df + 1)) + 1
    return {
        term: math.log((n_docs + 1) / (freq + 1)) + 1
        for term, freq in doc_freq.items()
    }


def compute_tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF scores for tokens.

    Args:
        tokens: List of tokens.
        idf: Pre-computed IDF dictionary.

    Returns:
        Dictionary mapping token to TF-IDF score.
    """
    tf = compute_tf(tokens)
    return {term: tf_score * idf.get(term, 1.0) for term, tf_score in tf.items()}


def cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors.

    Args:
        vec1: First vector as {term: weight}.
        vec2: Second vector as {term: weight}.

    Returns:
        Cosine similarity (0.0 to 1.0).
    """
    if not vec1 or not vec2:
        return 0.0

    # Find common terms
    common_terms = set(vec1.keys()) & set(vec2.keys())

    if not common_terms:
        return 0.0

    # Compute dot product
    dot_product = sum(vec1[t] * vec2[t] for t in common_terms)

    # Compute magnitudes
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


class TopicSegmenter:
    """Topic segmentation engine using TF-IDF similarity.

    Analyzes conversation turns to detect topic boundaries based on
    vocabulary changes between rolling windows.

    Example:
        >>> config = TopicSegmentationConfig()
        >>> segmenter = TopicSegmenter(config)
        >>> turns = [
        ...     {"id": "t0", "text": "Let's discuss the budget", "start": 0, "end": 5},
        ...     {"id": "t1", "text": "The budget needs review", "start": 5, "end": 10},
        ...     ...
        ... ]
        >>> topics = segmenter.segment(turns)
    """

    def __init__(self, config: TopicSegmentationConfig | None = None):
        """Initialize segmenter with configuration.

        Args:
            config: Segmentation configuration.
        """
        self.config = config or TopicSegmentationConfig()
        self._stopwords = self.config.get_stopwords()
        self._current_topic_idx = 0
        self._current_topic_start_time = 0.0
        self._current_topic_turns: list[dict[str, Any]] = []

    def segment(self, turns: list[dict[str, Any]]) -> list[TopicChunk]:
        """Segment turns into topics.

        Args:
            turns: List of turn dictionaries with id, text, start, end.

        Returns:
            List of TopicChunk objects.
        """
        if not self.config.enabled or not turns:
            return []

        topics: list[TopicChunk] = []

        # Tokenize all turns
        turn_tokens = [tokenize(t.get("text", ""), self._stopwords) for t in turns]

        # Compute global IDF
        idf = compute_idf(turn_tokens)

        # Track current topic
        current_topic_start_idx = 0
        current_topic_start_time = turns[0].get("start", 0.0)
        window_size = self.config.window_size_turns

        for i in range(len(turns)):
            # Skip until we have enough turns for comparison
            if i < window_size * 2:
                continue

            # Check if we've reached minimum turns for a topic
            turns_in_topic = i - current_topic_start_idx
            if turns_in_topic < self.config.min_turns_for_topic:
                continue

            # Get current time
            current_time = turns[i].get("start", 0.0)
            topic_duration = current_time - current_topic_start_time

            # Check minimum topic duration
            if topic_duration < self.config.min_topic_duration_sec:
                continue

            # Compute similarity between previous window and current window
            prev_window_tokens = []
            for j in range(max(0, i - window_size * 2), i - window_size):
                prev_window_tokens.extend(turn_tokens[j])

            curr_window_tokens = []
            for j in range(i - window_size, i):
                curr_window_tokens.extend(turn_tokens[j])

            # Compute TF-IDF vectors
            prev_tfidf = compute_tfidf(prev_window_tokens, idf)
            curr_tfidf = compute_tfidf(curr_window_tokens, idf)

            # Compute similarity
            similarity = cosine_similarity(prev_tfidf, curr_tfidf)

            # Check for topic boundary
            should_split = (
                similarity < self.config.similarity_threshold
                or topic_duration >= self.config.max_topic_duration_sec
            )

            if should_split:
                # Create topic chunk for the segment that just ended
                topic_turns = turns[current_topic_start_idx:i]
                topic = self._create_topic_chunk(
                    topic_idx=len(topics),
                    turns=topic_turns,
                    start_idx=current_topic_start_idx,
                    end_idx=i - 1,
                    turn_tokens=turn_tokens[current_topic_start_idx:i],
                    idf=idf,
                )
                topics.append(topic)

                # Start new topic
                current_topic_start_idx = i
                current_topic_start_time = current_time

        # Create final topic for remaining turns
        if current_topic_start_idx < len(turns):
            topic_turns = turns[current_topic_start_idx:]
            topic = self._create_topic_chunk(
                topic_idx=len(topics),
                turns=topic_turns,
                start_idx=current_topic_start_idx,
                end_idx=len(turns) - 1,
                turn_tokens=turn_tokens[current_topic_start_idx:],
                idf=idf,
            )
            topics.append(topic)

        return topics

    def _create_topic_chunk(
        self,
        topic_idx: int,
        turns: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        turn_tokens: list[list[str]],
        idf: dict[str, float],
    ) -> TopicChunk:
        """Create a topic chunk from turns.

        Args:
            topic_idx: Topic index.
            turns: Turns in this topic.
            start_idx: Start turn index.
            end_idx: End turn index.
            turn_tokens: Tokenized turns.
            idf: IDF dictionary.

        Returns:
            TopicChunk instance.
        """
        if not turns:
            return TopicChunk(
                id=f"topic_{topic_idx}",
                start=0.0,
                end=0.0,
            )

        # Collect all tokens for keyword extraction
        all_tokens: list[str] = []
        for tokens in turn_tokens:
            all_tokens.extend(tokens)

        # Get top keywords by TF-IDF
        tfidf = compute_tfidf(all_tokens, idf)
        sorted_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
        keywords = [term for term, _ in sorted_terms[:10]]

        # Collect turn IDs and speaker IDs
        turn_ids = [t.get("id", f"turn_{i}") for i, t in enumerate(turns)]
        speaker_ids = list({t.get("speaker_id", "") for t in turns if t.get("speaker_id")})

        # Concatenate text
        summary_text = " ".join(t.get("text", "") for t in turns)
        if len(summary_text) > 500:
            summary_text = summary_text[:497] + "..."

        return TopicChunk(
            id=f"topic_{topic_idx}",
            start=turns[0].get("start", 0.0),
            end=turns[-1].get("end", 0.0),
            turn_ids=turn_ids,
            turn_range=(start_idx, end_idx),
            summary_text=summary_text,
            keywords=keywords,
            speaker_ids=speaker_ids,
        )

    def get_keywords(
        self,
        text: str,
        top_k: int = 10,
        idf: dict[str, float] | None = None,
    ) -> list[str]:
        """Extract top keywords from text.

        Args:
            text: Text to analyze.
            top_k: Number of keywords to return.
            idf: Optional pre-computed IDF dictionary.

        Returns:
            List of top keywords.
        """
        tokens = tokenize(text, self._stopwords)

        if idf is None:
            # Use TF only if no IDF provided
            tf = compute_tf(tokens)
            sorted_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)
        else:
            tfidf = compute_tfidf(tokens, idf)
            sorted_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)

        return [term for term, _ in sorted_terms[:top_k]]


class StreamingTopicSegmenter:
    """Streaming-compatible topic segmenter.

    Processes turns incrementally and emits topic boundary events
    when topic changes are detected.

    Example:
        >>> segmenter = StreamingTopicSegmenter(config, on_boundary=callback)
        >>> for turn in turn_stream:
        ...     segmenter.add_turn(turn)
        >>> topics = segmenter.finalize()
    """

    def __init__(
        self,
        config: TopicSegmentationConfig | None = None,
        on_boundary: Any | None = None,
    ):
        """Initialize streaming segmenter.

        Args:
            config: Segmentation configuration.
            on_boundary: Callback for topic boundary events.
        """
        self.config = config or TopicSegmentationConfig()
        self._on_boundary = on_boundary
        self._stopwords = self.config.get_stopwords()

        # State
        self._turns: list[dict[str, Any]] = []
        self._turn_tokens: list[list[str]] = []
        self._current_topic_idx = 0
        self._current_topic_start_idx = 0
        self._current_topic_start_time = 0.0
        self._topics: list[TopicChunk] = []
        self._idf: dict[str, float] = {}

    def add_turn(self, turn: dict[str, Any]) -> TopicBoundaryPayload | None:
        """Add a turn and check for topic boundary.

        Args:
            turn: Turn dictionary with id, text, start, end.

        Returns:
            TopicBoundaryPayload if boundary detected, None otherwise.
        """
        if not self.config.enabled:
            return None

        # Tokenize and store
        tokens = tokenize(turn.get("text", ""), self._stopwords)
        self._turns.append(turn)
        self._turn_tokens.append(tokens)

        # Update IDF incrementally
        self._update_idf(tokens)

        # Set start time if first turn
        if len(self._turns) == 1:
            self._current_topic_start_time = turn.get("start", 0.0)

        # Check for boundary
        return self._check_boundary()

    def _update_idf(self, new_tokens: list[str]) -> None:
        """Update IDF with new document.

        Args:
            new_tokens: Tokens from new document.
        """
        # Recompute IDF periodically (every 10 turns for efficiency)
        if len(self._turns) % 10 == 0:
            self._idf = compute_idf(self._turn_tokens)

    def _check_boundary(self) -> TopicBoundaryPayload | None:
        """Check if current turn represents a topic boundary.

        Returns:
            TopicBoundaryPayload if boundary detected.
        """
        i = len(self._turns) - 1
        window_size = self.config.window_size_turns

        # Need enough turns for comparison
        if i < window_size * 2:
            return None

        # Check minimum turns for topic
        turns_in_topic = i - self._current_topic_start_idx
        if turns_in_topic < self.config.min_turns_for_topic:
            return None

        # Check minimum topic duration
        current_time = self._turns[i].get("start", 0.0)
        topic_duration = current_time - self._current_topic_start_time

        if topic_duration < self.config.min_topic_duration_sec:
            return None

        # Compute similarity
        prev_window_tokens = []
        for j in range(max(0, i - window_size * 2), i - window_size):
            prev_window_tokens.extend(self._turn_tokens[j])

        curr_window_tokens = []
        for j in range(i - window_size, i):
            curr_window_tokens.extend(self._turn_tokens[j])

        prev_tfidf = compute_tfidf(prev_window_tokens, self._idf)
        curr_tfidf = compute_tfidf(curr_window_tokens, self._idf)

        similarity = cosine_similarity(prev_tfidf, curr_tfidf)

        # Check for boundary
        should_split = (
            similarity < self.config.similarity_threshold
            or topic_duration >= self.config.max_topic_duration_sec
        )

        if should_split:
            return self._create_boundary(i, similarity, prev_tfidf, curr_tfidf)

        return None

    def _create_boundary(
        self,
        turn_idx: int,
        similarity: float,
        prev_tfidf: dict[str, float],
        curr_tfidf: dict[str, float],
    ) -> TopicBoundaryPayload:
        """Create topic boundary and update state.

        Args:
            turn_idx: Index of boundary turn.
            similarity: Similarity score.
            prev_tfidf: TF-IDF of previous window.
            curr_tfidf: TF-IDF of current window.

        Returns:
            TopicBoundaryPayload.
        """
        # Create topic chunk for completed topic
        topic_turns = self._turns[self._current_topic_start_idx:turn_idx]
        topic_tokens = self._turn_tokens[self._current_topic_start_idx:turn_idx]

        topic = self._create_topic_chunk(
            topic_idx=self._current_topic_idx,
            turns=topic_turns,
            start_idx=self._current_topic_start_idx,
            end_idx=turn_idx - 1,
            turn_tokens=topic_tokens,
        )
        self._topics.append(topic)

        # Extract keywords
        prev_keywords = sorted(prev_tfidf.items(), key=lambda x: x[1], reverse=True)[:5]
        curr_keywords = sorted(curr_tfidf.items(), key=lambda x: x[1], reverse=True)[:5]

        # Create payload
        payload = TopicBoundaryPayload(
            previous_topic_id=f"topic_{self._current_topic_idx}",
            new_topic_id=f"topic_{self._current_topic_idx + 1}",
            boundary_turn_id=self._turns[turn_idx].get("id", f"turn_{turn_idx}"),
            boundary_time=self._turns[turn_idx].get("start", 0.0),
            similarity_score=similarity,
            keywords_previous=[k for k, _ in prev_keywords],
            keywords_new=[k for k, _ in curr_keywords],
        )

        # Update state for new topic
        self._current_topic_idx += 1
        self._current_topic_start_idx = turn_idx
        self._current_topic_start_time = self._turns[turn_idx].get("start", 0.0)

        # Emit callback
        if self._on_boundary is not None:
            try:
                self._on_boundary(payload)
            except Exception:
                pass  # Callbacks should not crash pipeline

        return payload

    def _create_topic_chunk(
        self,
        topic_idx: int,
        turns: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        turn_tokens: list[list[str]],
    ) -> TopicChunk:
        """Create topic chunk from accumulated turns."""
        if not turns:
            return TopicChunk(id=f"topic_{topic_idx}", start=0.0, end=0.0)

        # Collect all tokens
        all_tokens: list[str] = []
        for tokens in turn_tokens:
            all_tokens.extend(tokens)

        # Get keywords
        tfidf = compute_tfidf(all_tokens, self._idf)
        sorted_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
        keywords = [term for term, _ in sorted_terms[:10]]

        # Collect metadata
        turn_ids = [t.get("id", "") for t in turns]
        speaker_ids = list({t.get("speaker_id", "") for t in turns if t.get("speaker_id")})
        summary_text = " ".join(t.get("text", "") for t in turns)[:500]

        return TopicChunk(
            id=f"topic_{topic_idx}",
            start=turns[0].get("start", 0.0),
            end=turns[-1].get("end", 0.0),
            turn_ids=turn_ids,
            turn_range=(start_idx, end_idx),
            summary_text=summary_text,
            keywords=keywords,
            speaker_ids=speaker_ids,
        )

    def finalize(self) -> list[TopicChunk]:
        """Finalize segmentation and return all topics.

        Returns:
            List of TopicChunk objects.
        """
        # Create final topic from remaining turns
        if self._current_topic_start_idx < len(self._turns):
            final_turns = self._turns[self._current_topic_start_idx:]
            final_tokens = self._turn_tokens[self._current_topic_start_idx:]

            topic = self._create_topic_chunk(
                topic_idx=self._current_topic_idx,
                turns=final_turns,
                start_idx=self._current_topic_start_idx,
                end_idx=len(self._turns) - 1,
                turn_tokens=final_tokens,
            )
            self._topics.append(topic)

        return self._topics

    def reset(self) -> None:
        """Reset segmenter state for new conversation."""
        self._turns = []
        self._turn_tokens = []
        self._current_topic_idx = 0
        self._current_topic_start_idx = 0
        self._current_topic_start_time = 0.0
        self._topics = []
        self._idf = {}


__all__ = [
    "TopicSegmenter",
    "StreamingTopicSegmenter",
    "TopicSegmentationConfig",
    "TopicChunk",
    "TopicBoundaryPayload",
    "tokenize",
    "compute_tf",
    "compute_idf",
    "compute_tfidf",
    "cosine_similarity",
]
