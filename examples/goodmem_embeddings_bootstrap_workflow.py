"""Clean-slate bootstrap workflow with ``GoodMemResources``.

This example shows how to start from a GoodMem deployment that may not already
have a suitable embedder configured for LangChain use.

The resources facade covers the GoodMem pieces needed by the normal LangChain
RAG/search path:

- it finds one compatible GoodMem embedder when one already exists
- it creates one compatible GoodMem embedder when none exists
- it creates a space using that embedder
- it returns a ready-to-use ``GoodMemVectorStore``

This helper is not a full GoodMem admin surface. API keys, server init,
migrations, system operations, and LLM/reranker/OCR/extension administration
remain in GoodMem's SDK, CLI, and UI.

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

from langchain_goodmem import GoodMemResources


def main() -> None:
    resources = GoodMemResources.from_env()
    vectorstore = resources.bootstrap_vector_store(
        space_name="langchain-bootstrap-demo",
        endpoint_url=_required_env("GOODMEM_EMBEDDINGS_BASE_URL"),
        model_identifier=_required_env("GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER"),
        dimensionality=int(_required_env("GOODMEM_EMBEDDINGS_DIMENSIONS")),
        upstream_api_key=_optional_env("GOODMEM_EMBEDDINGS_API_KEY"),
    )
    embeddings = vectorstore.embeddings
    if embeddings is None:
        raise RuntimeError("bootstrap_vector_store did not retain embeddings.")

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

def _required_env(name: str) -> str:
    import os

    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"{name} must be set.")
    return value.strip()


def _optional_env(name: str) -> str | None:
    import os

    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


if __name__ == "__main__":
    main()
