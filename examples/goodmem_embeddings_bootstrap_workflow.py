"""Clean-slate bootstrap workflow for ``GoodMemEmbeddings.ensure_from_env(...)``.

This example shows how to start from a GoodMem deployment that may not already
have a suitable embedder configured for LangChain use.

The bootstrap helper keeps the package's LangChain-facing scope intentionally
narrow:

- it finds one compatible GoodMem embedder when one already exists
- it creates one compatible GoodMem embedder when none exists
- it then returns a normal ``GoodMemEmbeddings`` instance that can be used
  directly or passed into ``GoodMemVectorStore.create(...)``

This helper is not a general GoodMem resource-management surface. It exists to
close the clean-slate onboarding gap for the package's ``OPENAI``-compatible
embeddings workflow. If you set ``GOODMEM_EMBEDDER_ID`` to reuse an existing
embedder, keep the bootstrap environment below present and aligned with that
embedder so configuration drift is rejected explicitly.

Required environment:

- ``GOODMEM_API_KEY``
- ``GOODMEM_BASE_URL``
- ``GOODMEM_EMBEDDINGS_BASE_URL``
- ``GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER``
- ``GOODMEM_EMBEDDINGS_DIMENSIONS``

Optional environment:

- ``GOODMEM_EMBEDDER_ID`` to reuse one existing compatible embedder, but only
  when the required bootstrap values below still match its endpoint, model, and
  dimensionality
- ``GOODMEM_EMBEDDINGS_API_KEY`` when the upstream API key must be stored on a
  newly created embedder or forwarded because the resolved embedder does not
  expose a readable inline secret

Run the example with:

::

    ./.venv/bin/python examples/goodmem_embeddings_bootstrap_workflow.py
"""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_goodmem import GoodMemConnection, GoodMemEmbeddings, GoodMemVectorStore


def main() -> None:
    connection = GoodMemConnection.from_env()
    embeddings = GoodMemEmbeddings.ensure_from_env(connection=connection)

    vectorstore = GoodMemVectorStore.create(
        name="langchain-bootstrap-demo",
        connection=connection,
        embedding=embeddings,
    )

    vectorstore.add_documents(
        [
            Document(
                page_content=(
                    "GoodMemEmbeddings.ensure_from_env can bootstrap a compatible "
                    "embedder before creating a LangChain-ready store."
                ),
                metadata={"topic": "bootstrap"},
            )
        ]
    )

    query_vector = embeddings.embed_query(
        "How does the GoodMem bootstrap embeddings helper work?"
    )
    print("embedder id:", embeddings.embedder_id)
    print("embedding length:", len(query_vector))

    documents = vectorstore.similarity_search(
        "How does the GoodMem bootstrap embeddings helper work?",
        k=1,
    )
    print("Chunk-level result:")
    for document in documents:
        print("content:", document.page_content)
        print("id:", document.id)
        print("metadata:", document.metadata)


if __name__ == "__main__":
    main()
