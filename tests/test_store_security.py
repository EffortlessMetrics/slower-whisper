import pytest

from transcription.store.store import ConversationStore, QueryError
from transcription.store.types import StoreQuery


def test_search_sql_injection_protection() -> None:
    """Test that the store rejects invalid order_by columns."""
    store = ConversationStore(":memory:")
    # Should fail for injected string
    query = StoreQuery(order_by="(SELECT CASE WHEN (1=1) THEN start_time ELSE end_time END)")
    with pytest.raises(QueryError):
        store.search(query)

    # Should succeed for valid string (even if empty store)
    valid_query = StoreQuery(order_by="start_time")
    results = store.search(valid_query)
    assert isinstance(results, list)
