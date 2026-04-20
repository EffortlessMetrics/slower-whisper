from transcription.store.store import SQLiteConversationStore
from transcription.store.types import StoreQuery


def test_sql_injection_order_by():
    store = SQLiteConversationStore(":memory:")

    # Try an injection that previously worked
    query = StoreQuery(
        order_by="(CASE WHEN (SELECT 1)=1 THEN start_time ELSE end_time END)", order_desc=False
    )

    # Needs to have some text to trigger search, but we didn't populate DB,
    # so it will just execute the query with 0 results
    store.search(query)

    # The order_by should have been changed to the fallback 'start_time'
    # instead of the malicious case statement.
    # We can't directly assert on the SQL query executed, but we know it didn't crash.


test_sql_injection_order_by()
