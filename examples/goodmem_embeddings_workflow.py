"""Embeddings-driven create workflow for ``GoodMemEmbeddings``.

This example shows how to expose a GoodMem-managed embedder as a LangChain
``Embeddings`` object, use it as the create-time source for
``GoodMemVectorStore.create(...)``, and then run both direct embedding and
semantic retrieval calls.

Prerequisites:

- install the optional embeddings dependency with
  ``pip install -e '.[openai]'``
- provide ``GOODMEM_API_KEY`` and ``GOODMEM_BASE_URL``
- identify a compatible GoodMem embedder ID

Credential resolution:

- preferred path: the selected GoodMem embedder already stores a readable
  inline API key
- fallback path: set ``GOODMEM_EMBEDDINGS_API_KEY`` when the embedder does not
  expose a readable inline secret or uses another credential form

If your upstream endpoint expects an explicit dimensions field, also export
``GOODMEM_EMBEDDINGS_DIMENSIONS`` with the same value declared on the GoodMem
embedder.

Run the example with:

::

    ./.venv/bin/python examples/goodmem_embeddings_workflow.py

Search visibility is eventually consistent. A fresh write may need time before
semantic retrieval returns the new chunk.
"""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_goodmem import GoodMemConnection, GoodMemEmbeddings, GoodMemVectorStore


def main() -> None:
    connection = GoodMemConnection.from_env()

    embeddings = GoodMemEmbeddings(
        embedder_id="your-embedder-id",
        connection=connection,
    )

    vectorstore = GoodMemVectorStore.create(
        name="langchain-embeddings-demo",
        connection=connection,
        embedding=embeddings,
    )

    vectorstore.add_documents(
        [
            Document(
                page_content="GoodMem can expose an OPENAI-compatible embedder to LangChain.",
                metadata={"topic": "embeddings"},
            )
        ]
    )

    query_vector = embeddings.embed_query("How does GoodMemEmbeddings work?")
    print("Embedding length:", len(query_vector))

    documents = vectorstore.similarity_search(
        "How does GoodMemEmbeddings work?",
        k=1,
    )
    print("Chunk-level result:")
    for document in documents:
        print("content:", document.page_content)
        print("id:", document.id)
        print("metadata:", document.metadata)


if __name__ == "__main__":
    main()
