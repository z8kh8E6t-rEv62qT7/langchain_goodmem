"""Live integration test for the existing-space workflow.

This suite validates that ``GoodMemVectorStore`` can bind to an existing
GoodMem space, write one document, and later retrieve the expected chunk
through semantic search.

Coverage goals:

- attach to an existing or temporary GoodMem space
- add documents with caller-provided IDs
- verify chunk-level result metadata and scored-search behavior

Required base environment:

- ``GOODMEM_API_KEY``
- ``GOODMEM_BASE_URL``

Optional resource reuse:

- ``GOODMEM_SPACE_ID`` to reuse an existing space instead of creating one
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.documents import Document

from langchain_goodmem import GoodMemVectorStore
from tests._integration_live_support import (
    cleanup_live_resources,
    ensure_existing_space,
    integration_client,
    integration_connection,
    poll_similarity_search,
)


@pytest.mark.integration
def test_live_existing_space_add_and_search() -> None:
    connection = integration_connection()
    client = integration_client(connection)
    existing_space = ensure_existing_space(client)
    marker = f"live-existing-{uuid.uuid4().hex}"
    requested_memory_id = str(uuid.uuid4())

    store = GoodMemVectorStore(
        space_id=existing_space.space.space_id,
        connection=connection,
    )

    try:
        returned_ids = store.add_documents(
            [
                Document(
                    page_content=(
                        f"LangChain GoodMem integration existing space test {marker}. "
                        "Semantic search should find this memory."
                    ),
                    metadata={"integration_marker": marker, "flow": "existing"},
                )
            ],
            ids=[requested_memory_id],
        )

        assert returned_ids == [requested_memory_id]

        results = poll_similarity_search(
            store,
            f"LangChain GoodMem integration existing space test {marker}",
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
        assert target.metadata["_goodmem_space_id"] == existing_space.space.space_id

        scored_results = store.similarity_search_with_score(
            f"LangChain GoodMem integration existing space test {marker}",
            k=5,
        )
        matching = [
            (doc, score)
            for doc, score in scored_results
            if doc.metadata["_goodmem_memory_id"] == requested_memory_id
        ]
        assert matching, "Expected the created memory to appear in scored results."
        doc, score = matching[0]
        assert marker in doc.page_content
        assert isinstance(score, float)
        assert doc.id == doc.metadata["_goodmem_chunk_id"]
        assert doc.metadata["_goodmem_score"] == score
    finally:
        cleanup_live_resources(
            client,
            space=existing_space.space,
            embedder=existing_space.embedder,
        )
