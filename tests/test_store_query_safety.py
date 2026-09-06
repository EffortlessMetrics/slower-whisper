"""Security contract tests for ConversationStore query ordering."""

from pathlib import Path
import json

import pytest

from transcription.store import ConversationStore, QueryError, StoreQuery, TextQuery


ATTACK_PAYLOADS = [
    "unknown_column",
    "lower(text)",
    "CASE WHEN 1=1 THEN start_time ELSE end_time END",
    "(SELECT name FROM sqlite_master LIMIT 1)",
    "start_time -- comment",
    "start_time\nDESC",
    "start_time; DROP TABLE segments; --",
    "s.start_time",
]


@pytest.fixture
def populated_store(tmp_path: Path) -> ConversationStore:
    """Create a deterministic store with sortable values."""
    transcript_path = tmp_path / "store-query-safety.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "file": "store-query-safety.wav",
                "language": "en",
                "segments": [
                    {
                        "id": 0,
                        "start": 3.0,
                        "end": 4.0,
                        "text": "gamma meeting",
                        "speaker": {"id": "spk_b", "confidence": 0.8},
                    },
                    {
                        "id": 1,
                        "start": 1.0,
                        "end": 2.0,
                        "text": "alpha meeting",
                        "speaker": {"id": "spk_a", "confidence": 0.9},
                    },
                    {
                        "id": 2,
                        "start": 2.0,
                        "end": 3.0,
                        "text": "beta meeting",
                        "speaker": {"id": "spk_c", "confidence": 0.7},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    store = ConversationStore(tmp_path / "store.db")
    store.ingest(transcript_path)
    yield store
    store.close()


@pytest.mark.parametrize(
    "order_by",
    [
        "start_time",
        "end_time",
        "segment_index",
        "text",
        "speaker_id",
        "speaker_confidence",
    ],
)
def test_supported_public_sort_keys(
    populated_store: ConversationStore,
    order_by: str,
) -> None:
    """Every declared public sort key resolves through the safe mapping."""
    results = populated_store.search(StoreQuery(order_by=order_by))
    assert len(results) == 3


def test_sort_direction_is_preserved(populated_store: ConversationStore) -> None:
    """Allowlisting identifiers must not change ASC/DESC semantics."""
    ascending = populated_store.search(StoreQuery(order_by="start_time"))
    descending = populated_store.search(StoreQuery(order_by="start_time", order_desc=True))

    assert [hit["start_time"] for hit in ascending] == [1.0, 2.0, 3.0]
    assert [hit["start_time"] for hit in descending] == [3.0, 2.0, 1.0]


@pytest.mark.parametrize("payload", ATTACK_PAYLOADS)
def test_expression_shaped_order_by_is_rejected_before_query_execution(
    populated_store: ConversationStore,
    payload: str,
) -> None:
    """Caller text cannot become ORDER BY syntax."""
    with pytest.raises(QueryError, match="Invalid order_by column"):
        populated_store.search(StoreQuery(order_by=payload))

    assert populated_store.stats()["segment_count"] == 3


def test_text_search_owns_rank_even_when_order_by_contains_sql(
    populated_store: ConversationStore,
) -> None:
    """FTS ranking is internal and ignores the unused public sort field."""
    results = populated_store.search(
        StoreQuery(
            text=TextQuery("meeting", match_type="phrase"),
            order_by="start_time; DROP TABLE segments; --",
        )
    )

    assert len(results) == 3
    assert populated_store.stats()["segment_count"] == 3
