"""Minimal LangChain loader example for slower-whisper chunks."""

from __future__ import annotations

from slower_whisper.adapters.langchain_loader import SlowerWhisperLoader


def main() -> None:
    loader = SlowerWhisperLoader("whisper_json")
    docs = loader.load()

    print(f"Loaded {len(docs)} LangChain documents")
    for doc in docs[:3]:
        print(
            "-",
            doc.metadata.get("chunk_id"),
            doc.metadata.get("source"),
            f"[{doc.metadata.get('start'):.2f}-{doc.metadata.get('end'):.2f}]",
            "speakers=",
            ",".join(doc.metadata.get("speaker_ids", [])),
        )

    # Example integration (requires embeddings/vector store):
    # from langchain_community.vectorstores import FAISS
    # from langchain_openai import OpenAIEmbeddings
    # vecstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    # results = vecstore.similarity_search("What did the agent promise?", k=3)
    # for res in results:
    #     print(res.page_content[:200], res.metadata)


if __name__ == "__main__":
    main()
