"""Unit tests for ``GoodMemVectorStore``.

This suite covers the package's main LangChain-facing workflow using fake
transport implementations.

Coverage goals:

- existing-space construction and create-helper behavior
- create-time embedder selection and rejection of unsupported public options
- local validation for texts, documents, metadata, IDs, and unexpected keyword
  arguments
- duplicate-ID handling, partial batch failures, backend response ordering, and
  chunk-level search result mapping

It remains a dedicated suite because vector-store behavior spans the broadest
public surface and needs focused coverage for write/search regressions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_goodmem import (
    GoodMemAPIError,
    GoodMemBatchPartialFailureError,
    GoodMemBatchWriteResultItem,
    GoodMemConfigurationError,
    GoodMemConnection,
    GoodMemDuplicateIDError,
    GoodMemEmbeddings,
    GoodMemSpaceEmbedder,
    GoodMemVectorStore,
)


@dataclass(frozen=True)
class FakeSpaceResponse:
    space_id: str


@dataclass(frozen=True)
class FakeSDKErrorDetail:
    code: int | None
    message: str | None


@dataclass(frozen=True)
class FakeSDKMemory:
    memory_id: str


@dataclass(frozen=True)
class FakeBatchMemoryResult:
    success: bool
    request_index: int | None
    memory: FakeSDKMemory | None = None
    memory_id: str | None = None
    error: FakeSDKErrorDetail | None = None


@dataclass(frozen=True)
class FakeBatchMemoryResponse:
    results: list[FakeBatchMemoryResult]


@dataclass(frozen=True)
class FakeMemoryDefinition:
    space_id: str
    metadata: Any


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    memory_id: str
    chunk_text: str
    metadata: Any


@dataclass(frozen=True)
class FakeChunkReference:
    memory_index: int
    relevance_score: float
    chunk: FakeChunk


@dataclass(frozen=True)
class FakeRetrievedItem:
    chunk: FakeChunkReference | None


@dataclass(frozen=True)
class FakeRetrieveEvent:
    memory_definition: Any | None = None
    retrieved_item: FakeRetrievedItem | None = None


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


@dataclass
class FakeTransport:
    create_response: FakeSpaceResponse = field(
        default_factory=lambda: FakeSpaceResponse(space_id="created-space-id")
    )
    batch_response: FakeBatchMemoryResponse = field(
        default_factory=lambda: FakeBatchMemoryResponse(results=[])
    )
    retrieve_events: list[FakeRetrieveEvent] = field(default_factory=list)
    retrieve_event_batches: list[list[FakeRetrieveEvent]] = field(default_factory=list)
    create_exception: Exception | None = None
    batch_exception: Exception | None = None
    retrieve_exception: Exception | None = None
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    batch_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieve_calls: list[dict[str, Any]] = field(default_factory=list)

    def create_space(self, request: Any) -> FakeSpaceResponse:
        self.create_calls.append(
            {
                "name": request.name,
                "space_embedders": request.space_embedders,
            }
        )
        if self.create_exception is not None:
            raise self.create_exception
        return self.create_response

    def batch_create_memories(
        self,
        *,
        space_id: str,
        writes: list[Any],
    ) -> FakeBatchMemoryResponse:
        self.batch_calls.append(
            {
                "space_id": space_id,
                "writes": [
                    {
                        "page_content": write.page_content,
                        "metadata": write.metadata,
                        "memory_id": write.memory_id,
                        "content_type": write.content_type,
                    }
                    for write in writes
                ],
            }
        )
        if self.batch_exception is not None:
            raise self.batch_exception
        return self.batch_response

    @contextmanager
    def retrieve_memories(
        self,
        *,
        space_id: str,
        query: str,
        k: int,
        filter_expression: str | None = None,
    ) -> Any:
        self.retrieve_calls.append(
            {
                "space_id": space_id,
                "query": query,
                "k": k,
                "filter_expression": filter_expression,
            }
        )
        if self.retrieve_exception is not None:
            raise self.retrieve_exception
        if self.retrieve_event_batches:
            yield list(self.retrieve_event_batches.pop(0))
            return
        yield list(self.retrieve_events)


def _connection() -> GoodMemConnection:
    return GoodMemConnection(
        api_key="gm-key",
        base_url="https://goodmem.example",
        verify="custom-ca.pem",
    )


def _patch_transports(
    monkeypatch: pytest.MonkeyPatch,
    *transports: FakeTransport,
) -> list[GoodMemConnection]:
    queue = list(transports)
    connections: list[GoodMemConnection] = []

    def fake_create_transport(connection: GoodMemConnection) -> FakeTransport:
        connections.append(connection)
        if not queue:
            raise AssertionError("No fake transport queued for GoodMemVectorStore.")
        return queue.pop(0)

    monkeypatch.setattr(
        "langchain_goodmem.vectorstores._create_transport",
        fake_create_transport,
    )
    return connections


def test_constructor_uses_connection_and_defaults_to_no_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    _patch_transports(monkeypatch, transport)

    store = GoodMemVectorStore(
        space_id=" space-123 ",
        connection=_connection(),
    )

    assert store.space_id == "space-123"
    assert store.embeddings is None


def test_constructor_rejects_blank_space_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transports(monkeypatch, FakeTransport())

    with pytest.raises(GoodMemConfigurationError, match="space_id"):
        GoodMemVectorStore(space_id="  ", connection=_connection())


def test_constructor_does_not_accept_embedding_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transports(monkeypatch, FakeTransport())

    with pytest.raises(TypeError, match="embedding"):
        GoodMemVectorStore(
            space_id="space-123",
            connection=_connection(),
            embedding=FakeEmbeddings(),  # type: ignore[call-arg]
        )


def test_create_returns_bound_store_and_records_space_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        create_response=FakeSpaceResponse(space_id="created-space"),
        batch_response=FakeBatchMemoryResponse(
            results=[
                FakeBatchMemoryResult(
                    success=True,
                    request_index=0,
                    memory_id="memory-1",
                )
            ]
        )
    )
    connections = _patch_transports(monkeypatch, transport)
    connection = _connection()

    store = GoodMemVectorStore.create(
        name="docs-space",
        embedders=[
            GoodMemSpaceEmbedder(
                embedder_id=" embedder-1 ",
                default_retrieval_weight=0.75,
            )
        ],
        connection=connection,
    )

    assert transport.create_calls == [
        {
            "name": "docs-space",
            "space_embedders": [
                GoodMemSpaceEmbedder(
                    embedder_id="embedder-1",
                    default_retrieval_weight=0.75,
                )
            ],
        }
    ]
    assert connections == [connection]
    assert store.space_id == "created-space"
    assert store.embeddings is None

    returned_ids = store.add_texts(["created store works"], ids=["memory-1"])
    assert returned_ids == ["memory-1"]
    assert transport.batch_calls == [
        {
            "space_id": "created-space",
            "writes": [
                {
                    "page_content": "created store works",
                    "metadata": {},
                    "memory_id": "memory-1",
                    "content_type": None,
                }
            ],
        }
    ]


def test_create_rejects_invalid_embedder_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transports(monkeypatch, FakeTransport())

    with pytest.raises(GoodMemConfigurationError, match="non-empty list"):
        GoodMemVectorStore.create(
            name="docs-space",
            embedders=[],
            connection=_connection(),
        )

    with pytest.raises(GoodMemConfigurationError, match="index 0"):
        GoodMemVectorStore.create(
            name="docs-space",
            embedders=[object()],  # type: ignore[list-item]
            connection=_connection(),
        )

    with pytest.raises(
        GoodMemConfigurationError,
        match="requires either embedders or embedding",
    ):
        GoodMemVectorStore.create(
            name="docs-space",
            connection=_connection(),
        )

    with pytest.raises(
        GoodMemConfigurationError,
        match="requires a GoodMemEmbeddings instance",
    ):
        GoodMemVectorStore.create(
            name="docs-space",
            connection=_connection(),
            embedding=FakeEmbeddings(),  # type: ignore[arg-type]
        )


def test_create_infers_space_embedder_from_goodmem_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        create_response=FakeSpaceResponse(space_id="created-space")
    )
    _patch_transports(monkeypatch, transport)
    monkeypatch.setattr(
        "langchain_goodmem.embeddings._create_transport",
        lambda connection: object(),
    )
    embedding = GoodMemEmbeddings(
        embedder_id=" embedder-123 ",
        connection=_connection(),
    )

    store = GoodMemVectorStore.create(
        name="docs-space",
        connection=_connection(),
        embedding=embedding,
    )

    assert transport.create_calls == [
        {
            "name": "docs-space",
            "space_embedders": [
                GoodMemSpaceEmbedder(embedder_id="embedder-123")
            ],
        }
    ]
    assert store.space_id == "created-space"
    assert store.embeddings is embedding


def test_create_does_not_accept_service_specific_space_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transports(monkeypatch, FakeTransport())

    with pytest.raises(TypeError, match="public_read"):
        GoodMemVectorStore.create(
            name="docs-space",
            embedders=[GoodMemSpaceEmbedder(embedder_id="embedder-1")],
            connection=_connection(),
            public_read=False,
        )

    monkeypatch.setattr(
        "langchain_goodmem.embeddings._create_transport",
        lambda connection: object(),
    )
    embedding = GoodMemEmbeddings(
        embedder_id="embedder-123",
        connection=_connection(),
    )

    with pytest.raises(
        GoodMemConfigurationError,
        match="accepts either embedders or embedding, not both",
    ):
        GoodMemVectorStore.create(
            name="docs-space",
            embedders=[GoodMemSpaceEmbedder(embedder_id="embedder-1")],
            connection=_connection(),
            embedding=embedding,
        )


def test_from_texts_builds_store_and_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        batch_response=FakeBatchMemoryResponse(
            results=[
                FakeBatchMemoryResult(success=True, request_index=0, memory_id="memory-1"),
                FakeBatchMemoryResult(success=True, request_index=1, memory_id="memory-2"),
            ]
        )
    )
    _patch_transports(monkeypatch, transport)
    embedding = FakeEmbeddings()

    store = GoodMemVectorStore.from_texts(
        ["alpha", "beta"],
        embedding,
        metadatas=[{"source": "one"}, {"source": "two"}],
        connection=_connection(),
        space_id="space-123",
        ids=["memory-1", "memory-2"],
    )

    assert store.embeddings is None
    assert transport.batch_calls == [
        {
            "space_id": "space-123",
            "writes": [
                {
                    "page_content": "alpha",
                    "metadata": {"source": "one"},
                    "memory_id": "memory-1",
                    "content_type": None,
                },
                {
                    "page_content": "beta",
                    "metadata": {"source": "two"},
                    "memory_id": "memory-2",
                    "content_type": None,
                },
            ],
        }
    ]


def test_add_documents_maps_metadata_and_document_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        batch_response=FakeBatchMemoryResponse(
            results=[
                FakeBatchMemoryResult(success=True, request_index=0, memory_id="doc-1"),
                FakeBatchMemoryResult(success=True, request_index=1, memory_id="generated-2"),
            ]
        )
    )
    _patch_transports(monkeypatch, transport)
    store = GoodMemVectorStore(space_id="space-123", connection=_connection())

    returned_ids = store.add_documents(
        [
            Document(id="doc-1", page_content="alpha", metadata={"source": "one"}),
            Document(page_content="beta", metadata={"source": "two"}),
        ]
    )

    assert returned_ids == ["doc-1", "generated-2"]
    assert transport.batch_calls == [
        {
            "space_id": "space-123",
            "writes": [
                {
                    "page_content": "alpha",
                    "metadata": {"source": "one"},
                    "memory_id": "doc-1",
                    "content_type": None,
                },
                {
                    "page_content": "beta",
                    "metadata": {"source": "two"},
                    "memory_id": None,
                    "content_type": None,
                },
            ],
        }
    ]


def test_add_documents_accepts_backend_results_without_request_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        batch_response=FakeBatchMemoryResponse(
            results=[
                FakeBatchMemoryResult(success=True, request_index=None, memory_id="doc-1"),
                FakeBatchMemoryResult(
                    success=True,
                    request_index=None,
                    memory_id="generated-2",
                ),
            ]
        )
    )
    _patch_transports(monkeypatch, transport)
    store = GoodMemVectorStore(space_id="space-123", connection=_connection())

    returned_ids = store.add_documents(
        [
            Document(id="doc-1", page_content="alpha", metadata={"source": "one"}),
            Document(page_content="beta", metadata={"source": "two"}),
        ]
    )

    assert returned_ids == ["doc-1", "generated-2"]


def test_add_documents_and_add_texts_validate_inputs_before_transport_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    _patch_transports(monkeypatch, transport)
    store = GoodMemVectorStore(space_id="space-123", connection=_connection())

    with pytest.raises(ValueError, match="page_content"):
        store.add_documents([Document(page_content="  ", metadata={})])

    with pytest.raises(ValueError, match="Document.id"):
        store.add_documents([Document(id="", page_content="alpha", metadata={})])

    with pytest.raises(ValueError, match="iterable of strings"):
        store.add_texts("alpha")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="texts at index 0"):
        store.add_texts([""])

    with pytest.raises(ValueError, match="metadatas must match"):
        store.add_texts(["one", "two"], metadatas=[{"a": 1}])

    with pytest.raises(ValueError, match="metadatas at index 0"):
        store.add_texts(["one"], metadatas=[["not", "a", "mapping"]])  # type: ignore[list-item]

    with pytest.raises(GoodMemDuplicateIDError, match="Duplicate memory IDs"):
        store.add_texts(["one", "two"], ids=["dup-id", "dup-id"])

    assert transport.batch_calls == []


def test_add_documents_propagates_batch_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial_failure = GoodMemBatchPartialFailureError(
        "partial",
        created_ids=["created-1"],
        results=[
            GoodMemBatchWriteResultItem(
                request_index=0,
                success=True,
                memory_id="created-1",
                error_code=None,
                error_message=None,
            ),
            GoodMemBatchWriteResultItem(
                request_index=1,
                success=False,
                memory_id=None,
                error_code=13,
                error_message="backend failed",
            ),
        ],
    )
    transport = FakeTransport(batch_exception=partial_failure)
    _patch_transports(monkeypatch, transport)
    store = GoodMemVectorStore(space_id="space-123", connection=_connection())

    with pytest.raises(GoodMemBatchPartialFailureError) as exc_info:
        store.add_documents(
            [
                Document(page_content="alpha", metadata={"rank": 1}),
                Document(page_content="beta", metadata={"rank": 2}),
            ],
            ids=["memory-1", "memory-2"],
        )

    assert exc_info.value is partial_failure


def test_add_texts_rejects_invalid_request_ordering_from_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        batch_response=FakeBatchMemoryResponse(
            results=[
                FakeBatchMemoryResult(success=True, request_index=0, memory_id="created-1"),
                FakeBatchMemoryResult(success=True, request_index=0, memory_id="created-2"),
            ]
        )
    )
    _patch_transports(monkeypatch, transport)
    store = GoodMemVectorStore(space_id="space-123", connection=_connection())

    with pytest.raises(GoodMemAPIError, match="duplicate request_index 0"):
        store.add_texts(["alpha", "beta"], ids=["memory-1", "memory-2"])


def test_similarity_search_maps_documents_and_merged_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        retrieve_events=[
            FakeRetrieveEvent(
                memory_definition=FakeMemoryDefinition(
                    space_id="space-123",
                    metadata={"shared": "memory", "memory_only": "memory"},
                )
            ),
            FakeRetrieveEvent(
                retrieved_item=FakeRetrievedItem(
                    chunk=FakeChunkReference(
                        memory_index=0,
                        relevance_score=0.91,
                        chunk=FakeChunk(
                            chunk_id="chunk-9",
                            memory_id="memory-42",
                            chunk_text="the matched chunk text",
                            metadata={"shared": "chunk", "chunk_only": "chunk"},
                        ),
                    )
                )
            ),
        ]
    )
    _patch_transports(monkeypatch, transport)
    store = GoodMemVectorStore(space_id="space-123", connection=_connection())

    results = store.similarity_search(
        "chunk text",
        k=1,
        filter="metadata.topic == 'docs'",
    )

    assert results == [
        Document(
            id="chunk-9",
            page_content="the matched chunk text",
            metadata={
                "shared": "chunk",
                "memory_only": "memory",
                "chunk_only": "chunk",
                "_goodmem_chunk_id": "chunk-9",
                "_goodmem_memory_id": "memory-42",
                "_goodmem_space_id": "space-123",
            },
        )
    ]
    assert transport.retrieve_calls == [
        {
            "space_id": "space-123",
            "query": "chunk text",
            "k": 1,
            "filter_expression": "metadata.topic == 'docs'",
        }
    ]


def test_similarity_search_with_score_includes_score_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        retrieve_events=[
            FakeRetrieveEvent(
                memory_definition=FakeMemoryDefinition(
                    space_id="space-123",
                    metadata={"topic": "semantics"},
                )
            ),
            FakeRetrieveEvent(
                retrieved_item=FakeRetrievedItem(
                    chunk=FakeChunkReference(
                        memory_index=0,
                        relevance_score=0.82,
                        chunk=FakeChunk(
                            chunk_id="chunk-1",
                            memory_id="memory-1",
                            chunk_text="matched chunk",
                            metadata={},
                        ),
                    )
                )
            ),
        ]
    )
    _patch_transports(monkeypatch, transport)
    store = GoodMemVectorStore(space_id="space-123", connection=_connection())

    results = store.similarity_search_with_score("find memory")

    assert results == [
        (
            Document(
                id="chunk-1",
                page_content="matched chunk",
                metadata={
                    "topic": "semantics",
                    "_goodmem_chunk_id": "chunk-1",
                    "_goodmem_memory_id": "memory-1",
                    "_goodmem_space_id": "space-123",
                    "_goodmem_score": 0.82,
                },
            ),
            0.82,
        )
    ]


def test_similarity_search_validates_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    _patch_transports(monkeypatch, transport)
    store = GoodMemVectorStore(space_id="space-123", connection=_connection())

    with pytest.raises(ValueError, match="k must be greater than 0"):
        store.similarity_search("find memory", k=0)

    with pytest.raises(ValueError, match="k must be an integer"):
        store.similarity_search_with_score("find memory", k="2")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="k must be an integer"):
        store.similarity_search("find memory", k=True)  # type: ignore[arg-type]

    assert transport.retrieve_calls == []


def test_empty_inputs_return_empty_without_transport_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    _patch_transports(monkeypatch, transport)
    store = GoodMemVectorStore(space_id="space-123", connection=_connection())

    assert store.add_documents([]) == []
    assert store.add_texts([]) == []
    assert transport.batch_calls == []
