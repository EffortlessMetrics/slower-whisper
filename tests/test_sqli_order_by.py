import pytest

from transcription.store.store import SQLiteConversationStore
from transcription.store.types import QueryError, StoreQuery


def test_sql_injection_order_by():
    store = SQLiteConversationStore.open(":memory:")
    # Initialize basic schema needed for the test
    # (open already creates tables automatically now)

    # Try a syntax error or malicious payload
    query = StoreQuery(text=None, order_by="INVALID_COLUMN_OR_INJECTION")
    with pytest.raises(QueryError, match="Invalid order_by column"):
        store.search(query)
