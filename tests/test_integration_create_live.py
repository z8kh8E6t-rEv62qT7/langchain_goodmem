"""Live integration test for the create-helper workflow.

This suite validates that ``GoodMemVectorStore.create(...)`` can provision a
new GoodMem space, write a memory, and retrieve the expected chunk through
semantic search.

Coverage goals:

- create a space through the package-owned create helper
- attach one explicit ``GoodMemSpaceEmbedder`` to the new space
- verify that a created memory becomes discoverable and returns the expected
  GoodMem metadata fields

Required base environment:

- ``GOODMEM_API_KEY``
- ``GOODMEM_BASE_URL``

Optional resource reuse and provisioning inputs:

- ``GOODMEM_EMBEDDER_ID`` to reuse an existing compatible embedder
- ``GOODMEM_EMBEDDINGS_BASE_URL`` and ``GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER``
  when the helper needs to auto-provision an embedder
"""

from __future__ import annotations

import uuid

import pytest

from langchain_goodmem import GoodMemSpaceEmbedder, GoodMemVectorStore
from tests._integration_live_support import (
    LiveSpaceResource,
    cleanup_live_resources,
    ensure_test_embedder,
    integration_client,
    integration_connection,
    poll_similarity_search,
)


@pytest.mark.integration
def test_live_create_helper_add_and_search() -> None:
    connection = integration_connection()
    client = integration_client(connection)
    embedder = ensure_test_embedder(
        client,
        marker="langchain-goodmem-live-create-embedder",
    )

    marker = f"live-create-{uuid.uuid4().hex}"
    requested_memory_id = str(uuid.uuid4())
    space_name = f"langchain-goodmem-create-{uuid.uuid4().hex[:8]}"
    created_space: LiveSpaceResource | None = None

    try:
        store = GoodMemVectorStore.create(
            name=space_name,
            embedders=[GoodMemSpaceEmbedder(embedder_id=embedder.embedder_id)],
            connection=connection,
        )
        created_space = LiveSpaceResource(
            space_id=store.space_id,
            created_by_test=True,
        )

        assert store.space_id
        assert store.embeddings is None

        returned_ids = store.add_texts(
            [
                (
                    f"LangChain GoodMem integration create helper test {marker}. "
                    "This should be discoverable by semantic retrieval."
                )
            ],
            metadatas=[{"integration_marker": marker, "flow": "create"}],
            ids=[requested_memory_id],
        )

        assert returned_ids == [requested_memory_id]

        results = poll_similarity_search(
            store,
            f"LangChain GoodMem integration create helper test {marker}",
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
