"""Blank text is an absent filter, including ordering and pagination (#648)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from transcription.store.store import SQLiteConversationStore
from transcription.store.types import (
    DateRangeQuery,
    IngestOptions,
    QueryError,
    SpeakerQuery,
    StoreQuery,
    TextQuery,
    TimeRangeQuery,
    TranscriptQuery,
)


@pytest.fixture
def search_store(tmp_path: Path) -> Iterator[SQLiteConversationStore]:
    with SQLiteConversationStore(tmp_path / "search.db") as store:
        documents = [
            {
                "file": "first.wav",
                "language": "en",
                "segments": [
                    {
                        "id": 0,
                        "start": 40,
                        "end": 44,
                        "text": "alpha beta gamma delta",
                        "speaker": "A",
                    },
                    {"id": 1, "start": 5, "end": 7, "text": "alpha", "speaker": "A"},
                    {"id": 2, "start": 25, "end": 31, "text": "omega", "speaker": "B"},
                    {"id": 3, "start": 12, "end": 17, "text": "beta alpha alpha", "speaker": "A"},
                    {"id": 4, "start": 60, "end": 70, "text": "kappa", "speaker": "B"},
                ],
            },
            {
                "file": "second.wav",
                "language": "fr",
                "segments": [
                    {"id": 0, "start": 9, "end": 10, "text": "alpha delta", "speaker": "A"},
                    {"id": 1, "start": 35, "end": 39, "text": "epsilon", "speaker": "C"},
                ],
            },
        ]
        for index, document in enumerate(documents):
            path = tmp_path / f"transcript-{index}.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            result = store.ingest(path, IngestOptions(generate_transcript_id=False))
            assert result.status == "success"
            store._get_conn().execute(
                "UPDATE transcripts SET ingested_at = ? WHERE transcript_id = ?",
                ("2025-01-01" if index == 0 else "2020-01-01", result.transcript_id),
            )
        yield store


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", "\u2003"])
@pytest.mark.parametrize("order_by", ["start_time", "end_time", "text"])
@pytest.mark.parametrize("descending", [False, True])
def test_blank_search_honors_ordinary_sort(
    search_store: SQLiteConversationStore,
    blank: str,
    order_by: Literal["start_time", "end_time", "text"],
    descending: bool,
) -> None:
    query = StoreQuery(order_by=order_by, order_desc=descending)
    expected = search_store.search(query)
    assert len(expected) == 7
    values = [hit[order_by] for hit in expected]
    assert values == sorted(values, reverse=descending)
    assert search_store.search(replace(query, text=TextQuery(blank))) == expected


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", "\u2003"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("limit,offset", [(100, 0), (2, 0), (2, 1), (1, 2)])
def test_blank_search_preserves_combined_filters_and_pagination(
    search_store: SQLiteConversationStore,
    blank: str,
    descending: bool,
    limit: int,
    offset: int,
) -> None:
    first = next(t for t in search_store.list_transcripts() if t["file_name"] == "first.wav")
    query = StoreQuery(
        time_range=TimeRangeQuery(start=0, end=80),
        speakers=SpeakerQuery(speaker_ids=["B", "C"], exclude=True),
        transcripts=TranscriptQuery(
            transcript_ids=[first["transcript_id"]], file_names=["first.wav"], languages=["en"]
        ),
        date_range=DateRangeQuery(after="2024-01-01", before="2027-01-01"),
        order_by="end_time",
        order_desc=descending,
        limit=limit,
        offset=offset,
    )
    expected = search_store.search(query)
    assert expected  # Do not accidentally prove equality of two empty queries.
    assert all(hit["speaker_id"] == "A" for hit in expected)
    assert search_store.search(replace(query, text=TextQuery(blank))) == expected


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", "\u2003"])
def test_blank_search_never_builds_or_executes_an_fts_query(
    search_store: SQLiteConversationStore, blank: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    statements: list[str] = []
    search_store._get_conn().set_trace_callback(statements.append)

    def unexpected_fts(self: TextQuery) -> str:
        pytest.fail("Blank text must not invoke the FTS query builder")

    monkeypatch.setattr(TextQuery, "to_fts_query", unexpected_fts)
    result = search_store.search(StoreQuery(text=TextQuery(blank), order_by="end_time"))
    assert len(result) == 7
    assert all("MATCH" not in sql.upper() for sql in statements)
    assert all(hit["rank"] == 0 for hit in result)


@pytest.mark.parametrize("descending", [False, True])
def test_nonempty_search_keeps_internal_rank_order(
    search_store: SQLiteConversationStore, descending: bool
) -> None:
    query = StoreQuery(text=TextQuery(" alpha "), order_by="end_time", order_desc=descending)
    hits = search_store.search(query)
    assert len(hits) == 4
    assert hits == search_store.search(replace(query, order_by="start_time"))
    ranks = [hit["rank"] for hit in hits]
    assert ranks == sorted(ranks, reverse=descending)
    assert len(set(ranks)) > 1
    assert all("<mark>" in hit["snippet"] for hit in hits)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", "\u2003"])
@pytest.mark.parametrize("order_by", ["not_a_sort_key", "rank"])
def test_blank_text_preserves_sort_key_validation(
    search_store: SQLiteConversationStore, blank: str, order_by: str
) -> None:
    """A blank TextQuery cannot bypass the non-FTS sort-key contract."""
    with pytest.raises(QueryError, match="Invalid order_by column"):
        search_store.search(StoreQuery(text=TextQuery(text=blank), order_by=order_by))
