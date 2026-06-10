import pytest
import tempfile
import pathlib
from transcription.store.store import SQLiteConversationStore
from transcription.store.types import StoreQuery, QueryError

def test_store_order_by_sql_injection():
    with tempfile.TemporaryDirectory() as td:
        db_path = pathlib.Path(td) / "test.db"
        with SQLiteConversationStore(str(db_path)) as store:
            query = StoreQuery(order_by="(CASE WHEN (1=1) THEN start_time ELSE end_time END)")
            with pytest.raises(QueryError, match="Invalid order_by column"):
                store.search(query)
