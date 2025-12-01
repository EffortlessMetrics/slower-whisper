"""Minimal LlamaIndex reader example for slower-whisper chunks."""

from __future__ import annotations

from integrations.llamaindex_reader import SlowerWhisperReader


def main() -> None:
    reader = SlowerWhisperReader("whisper_json")
    docs = reader.load_data()

    print(f"Loaded {len(docs)} LlamaIndex documents")
    for doc in docs[:3]:
        meta = doc.metadata or {}
        print(
            "-",
            meta.get("chunk_id"),
            meta.get("source"),
            f"[{meta.get('start'):.2f}-{meta.get('end'):.2f}]",
            "speakers=",
            ",".join(meta.get("speaker_ids", [])),
        )

    # Example integration (requires llama-index-core + embedding/model choices):
    # from llama_index.core import VectorStoreIndex
    # index = VectorStoreIndex.from_documents(docs)
    # query_engine = index.as_query_engine()
    # print(query_engine.query("What did the customer ask for?"))


if __name__ == "__main__":
    main()
