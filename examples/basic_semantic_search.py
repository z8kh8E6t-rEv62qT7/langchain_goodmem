"""Existing-space add/search workflow for ``GoodMemVectorStore``.

This example is the runnable source of truth for the most common workflow:
bind the vector store to an existing GoodMem space, add multiple documents,
then run chunk-level semantic retrieval with and without scores.

Prerequisites:

- install the package with ``pip install -e .``
- provide a valid GoodMem API key and base URL
- create or identify a GoodMem space ahead of time

Run the example by editing the placeholder connection values and ``space_id``,
then execute:

::

    ./.venv/bin/python examples/basic_semantic_search.py

Expected output:

- created memory IDs from ``add_documents(...)``
- one chunk-level retrieval listing
- one scored retrieval listing

Search visibility is eventually consistent. Fresh writes may not appear in
semantic retrieval immediately after the add call returns.
"""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_goodmem import GoodMemConnection, GoodMemVectorStore


def main() -> None:
    connection = GoodMemConnection(
        api_key="gm_your_api_key",
        base_url="https://api.goodmem.ai",
    )

    vectorstore = GoodMemVectorStore(
        space_id="your-space-id",
        connection=connection,
    )

    memory_ids = vectorstore.add_documents(
        [
            Document(
                page_content="GoodMem stores and retrieves semantically related content.",
                metadata={"topic": "product", "lang": "en"},
            ),
            Document(
                page_content="LangChain integrations can expose GoodMem as a native VectorStore.",
                metadata={"topic": "integration", "lang": "en"},
            ),
        ]
    )
    print("Created memory IDs:", memory_ids)

    documents = vectorstore.similarity_search(
        "How does GoodMem work with LangChain?",
        k=2,
        filter="metadata.lang == 'en'",
    )
    print("Chunk-level search:")
    for document in documents:
        print("content:", document.page_content)
        print("id:", document.id)
        print("metadata:", document.metadata)
        print()

    scored_results = vectorstore.similarity_search_with_score(
        "How does GoodMem work with LangChain?",
        k=2,
        filter="metadata.lang == 'en'",
    )
    for document, score in scored_results:
        print("content:", document.page_content)
        print("id:", document.id)
        print("score:", score)
        print("metadata:", document.metadata)
        print()


if __name__ == "__main__":
    main()
