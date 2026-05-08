"""Live integration test for the embeddings-driven create workflow.

This suite validates the end-to-end path where ``GoodMemEmbeddings`` backs both
direct embedding calls and ``GoodMemVectorStore.create(...)``.

Coverage goals:

- resolve one compatible GoodMem embedder and exercise ``embed_query(...)``
- create a new GoodMem space from the embeddings adapter
- write a document and retrieve the expected chunk through semantic search

Required base environment:

- ``GOODMEM_API_KEY``
- ``GOODMEM_BASE_URL``

Additional environment expectations:

- either ``GOODMEM_EMBEDDER_ID`` or enough embeddings provisioning inputs for
  ``ensure_test_embedder(...)`` to find or create a compatible embedder
- ``GOODMEM_EMBEDDINGS_API_KEY`` when GoodMem cannot expose a readable inline
  upstream secret
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.documents import Document

from langchain_goodmem import GoodMemEmbeddings, GoodMemVectorStore
from tests._integration_live_support import (
    LiveSpaceResource,
    cleanup_live_resources,
    embedder_has_inline_api_key,
    embedding_integration_config,
    ensure_test_embedder,
    integration_client,
    integration_connection,
    poll_similarity_search,
)


@pytest.mark.integration
def test_live_embeddings_driven_create_and_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = integration_connection()
    upstream_api_key = embedding_integration_config()
    client = integration_client(connection)
    embedder = ensure_test_embedder(
        client,
        marker="langchain-goodmem-live-embeddings-embedder",
    )
    marker = f"live-embeddings-store-{uuid.uuid4().hex}"
    requested_memory_id = str(uuid.uuid4())
    space_name = f"langchain-goodmem-embeddings-{uuid.uuid4().hex[:8]}"
    created_space: LiveSpaceResource | None = None

    try:
        resolved_embedder = client.embedders.get(id=embedder.embedder_id)
        expected_dimensions = resolved_embedder.dimensionality
        has_inline_api_key = embedder_has_inline_api_key(resolved_embedder)

        if upstream_api_key is None and not has_inline_api_key:
            pytest.skip(
                "GoodMemEmbeddings live tests require GOODMEM_EMBEDDINGS_API_KEY "
                "or an embedder with a readable inline API key."
            )

        if has_inline_api_key:
            monkeypatch.delenv("GOODMEM_EMBEDDINGS_API_KEY", raising=False)

        embeddings = GoodMemEmbeddings(
            embedder_id=embedder.embedder_id,
            connection=connection,
        )
        query_vector = embeddings.embed_query(
            "GoodMem embeddings-enabled store integration query."
        )
        assert len(query_vector) == expected_dimensions
        assert all(isinstance(value, float) for value in query_vector)

        store = GoodMemVectorStore.create(
            name=space_name,
            connection=connection,
            embedding=embeddings,
        )
        created_space = LiveSpaceResource(
            space_id=store.space_id,
            created_by_test=True,
        )

        assert store.embeddings is embeddings

        returned_ids = store.add_documents(
            [
                Document(
                    page_content=(
                        f"LangChain GoodMem embeddings-enabled store test {marker}. "
                        "Semantic search should find this memory."
                    ),
                    metadata={"integration_marker": marker, "flow": "embeddings"},
                )
            ],
            ids=[requested_memory_id],
        )

        assert returned_ids == [requested_memory_id]

        results = poll_similarity_search(
            store,
            f"LangChain GoodMem embeddings-enabled store test {marker}",
            expected_memory_id=requested_memory_id,
        )
        target = next(
            doc
            for doc in results
            if doc.metadata["_goodmem_memory_id"] == requested_memory_id
        )
        assert marker in target.page_content
        assert target.id == target.metadata["_goodmem_chunk_id"]
        assert target.metadata["integration_marker"] == marker
        assert target.metadata["_goodmem_memory_id"] == requested_memory_id
        assert target.metadata["_goodmem_space_id"] == store.space_id
    finally:
        cleanup_live_resources(
            client,
            space=created_space,
            embedder=embedder,
        )
