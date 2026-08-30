import re

with open("tests/test_conversation_store.py", "r") as f:
    content = f.read()

# Make sure `from transcription.store.types import StoreQuery` is imported
if "from transcription.store.types import StoreQuery" not in content:
    # Use re.sub to inject StoreQuery and QueryError in the import block
    # Actually, we can just replace `QueryFilter,` with `QueryFilter,\n    QueryError,\n    StoreQuery,`
    content = re.sub(
        r"(from transcription\.store\.types import \()",
        r"\1\n    QueryError,",
        content
    )

    content = re.sub(
        r"(from transcription\.store\.types import IngestError, QueryError)",
        r"from transcription.store.types import IngestError",
        content
    )

    with open("tests/test_conversation_store.py", "w") as f:
        f.write(content)
