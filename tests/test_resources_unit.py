"""Unit tests for the public GoodMem resource facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from langchain_goodmem import (
    GoodMemAPIError,
    GoodMemConnection,
    GoodMemConfigurationError,
    GoodMemDuplicateIDError,
    GoodMemEmbedderInfo,
    GoodMemEmbeddings,
    GoodMemMemoryInfo,
    GoodMemResources,
    GoodMemSpaceEmbedder,
    GoodMemSpaceInfo,
    GoodMemVectorStore,
)

_DEFAULT_DELETE_BATCH_RESPONSE = object()


@dataclass(frozen=True)
class FakeRawEmbedder:
    embedder_id: str = "embedder-1"
    display_name: str = "OpenAI embedder"
    provider_type: Any = "OPENAI"
    endpoint_url: str = "https://embeddings.example"
    model_identifier: str = "text-embedding-3-small"
    dimensionality: int = 1536
    supported_modalities: tuple[Any, ...] = ("TEXT",)


@dataclass(frozen=True)
class FakeRawSpace:
    space_id: str = "space-1"
    name: str = "Docs"
    labels: dict[str, str] = field(default_factory=lambda: {"env": "test"})
    space_embedders: tuple[Any, ...] = field(
        default_factory=lambda: (SimpleNamespace(embedder_id="embedder-1"),)
    )


@dataclass(frozen=True)
class FakeRawMemory:
    memory_id: str = "memory-1"
    space_id: str = "space-1"
    metadata: dict[str, Any] = field(default_factory=lambda: {"topic": "docs"})
    original_content: str | bytes | None = "hello"
    processing_status: str = "COMPLETED"


@dataclass(frozen=True)
class FakeDeleteResult:
    success: bool
    error: Any = None


@dataclass(frozen=True)
class FakeBatchDeleteResponse:
    results: list[FakeDeleteResult]


@dataclass
class FakeAutoPagingList:
    data: list[Any]
    next_token: str | None = None
    pages: dict[str, "FakeAutoPagingList"] = field(default_factory=dict)
    fetch_calls: list[str] = field(default_factory=list)
    _max_items: int | None = None
    _response_model: object = field(default_factory=object)
    _items_field: str = "data"
    _fetch_fn: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fetch_fn = self._fetch_page

    def _fetch_page(self, token: str | None) -> "FakeAutoPagingList":
        if token is None:
            raise AssertionError("unexpected empty pagination token")
        self.fetch_calls.append(token)
        try:
            return self.pages[token]
        except KeyError as exc:  # pragma: no cover - defensive test guard
            raise AssertionError(f"unexpected pagination token: {token}") from exc

    def __iter__(self) -> Any:
        yielded = 0
        for item in self.data:
            if self._max_items is not None and yielded >= self._max_items:
                return
            yield item
            yielded += 1

        next_token = self.next_token
        while next_token is not None:
            if self._max_items is not None and yielded >= self._max_items:
                return
            page = self._fetch_page(next_token)
            for item in page.data:
                if self._max_items is not None and yielded >= self._max_items:
                    return
                yield item
                yielded += 1
            next_token = page.next_token


@dataclass
class FakeResourceTransport:
    embedder: FakeRawEmbedder = field(default_factory=FakeRawEmbedder)
    space: FakeRawSpace = field(default_factory=FakeRawSpace)
    memory: FakeRawMemory = field(default_factory=FakeRawMemory)
    delete_batch_response: Any = _DEFAULT_DELETE_BATCH_RESPONSE
    create_embedder_calls: list[Any] = field(default_factory=list)
    get_embedder_calls: list[dict[str, Any]] = field(default_factory=list)
    list_embedder_calls: list[dict[str, Any]] = field(default_factory=list)
    delete_embedder_calls: list[dict[str, Any]] = field(default_factory=list)
    create_space_calls: list[Any] = field(default_factory=list)
    get_space_calls: list[dict[str, Any]] = field(default_factory=list)
    list_space_calls: list[dict[str, Any]] = field(default_factory=list)
    delete_space_calls: list[dict[str, Any]] = field(default_factory=list)
    create_memory_calls: list[Any] = field(default_factory=list)
    get_memory_calls: list[dict[str, Any]] = field(default_factory=list)
    list_memory_calls: list[dict[str, Any]] = field(default_factory=list)
    delete_memory_calls: list[dict[str, Any]] = field(default_factory=list)
    delete_memories_calls: list[dict[str, Any]] = field(default_factory=list)

    def create_embedder(self, request: Any) -> FakeRawEmbedder:
        self.create_embedder_calls.append(request)
        return self.embedder

    def get_embedder(self, *, embedder_id: str) -> FakeRawEmbedder:
        self.get_embedder_calls.append({"embedder_id": embedder_id})
        return self.embedder

    def list_embedders(self, **kwargs: Any) -> list[FakeRawEmbedder]:
        self.list_embedder_calls.append(kwargs)
        return [self.embedder]

    def delete_embedder(self, *, embedder_id: str) -> None:
        self.delete_embedder_calls.append({"embedder_id": embedder_id})

    def create_space(self, request: Any) -> FakeRawSpace:
        self.create_space_calls.append(request)
        return self.space

    def get_space(self, *, space_id: str) -> FakeRawSpace:
        self.get_space_calls.append({"space_id": space_id})
        return self.space

    def list_spaces(self, **kwargs: Any) -> list[FakeRawSpace]:
        self.list_space_calls.append(kwargs)
        return [self.space]

    def delete_space(self, *, space_id: str) -> None:
        self.delete_space_calls.append({"space_id": space_id})

    def create_memory(self, request: Any) -> FakeRawMemory:
        self.create_memory_calls.append(request)
        return self.memory

    def get_memory(self, *, memory_id: str, include_content: bool = False) -> FakeRawMemory:
        self.get_memory_calls.append(
            {"memory_id": memory_id, "include_content": include_content}
        )
        return self.memory

    def list_memories(self, **kwargs: Any) -> list[FakeRawMemory]:
        self.list_memory_calls.append(kwargs)
        return [self.memory]

    def delete_memory(self, *, memory_id: str) -> None:
        self.delete_memory_calls.append({"memory_id": memory_id})

    def delete_memories(self, *, memory_ids: list[str]) -> Any:
        self.delete_memories_calls.append({"memory_ids": list(memory_ids)})
        if self.delete_batch_response is not _DEFAULT_DELETE_BATCH_RESPONSE:
            return self.delete_batch_response
        return FakeBatchDeleteResponse(
            results=[FakeDeleteResult(success=True) for _ in memory_ids]
        )


def _connection() -> GoodMemConnection:
    return GoodMemConnection(
        api_key="gm-key",
        base_url="https://goodmem.example",
        verify="custom-ca.pem",
    )


def _resources(transport: FakeResourceTransport) -> GoodMemResources:
    return GoodMemResources._from_transport(
        connection=_connection(),
        transport=transport,
    )


def test_embedder_resource_methods_return_stable_info() -> None:
    transport = FakeResourceTransport()
    resources = _resources(transport)

    created = resources.create_embedder(
        endpoint_url=" https://embeddings.example ",
        model_identifier=" text-embedding-3-small ",
        dimensionality=1536,
        upstream_api_key=" sk-test ",
    )
    loaded = resources.get_embedder(" embedder-1 ")
    listed = resources.list_embedders(
        label={"env": "test"},
        owner_id=" owner-1 ",
        provider_type=" OPENAI ",
    )
    resources.delete_embedder(" embedder-1 ")

    assert created == GoodMemEmbedderInfo(
        embedder_id="embedder-1",
        display_name="OpenAI embedder",
        provider_type="OPENAI",
        endpoint_url="https://embeddings.example",
        model_identifier="text-embedding-3-small",
        dimensionality=1536,
        supported_modalities=("TEXT",),
    )
    assert loaded == created
    assert listed == [created]
    create_request = transport.create_embedder_calls[0]
    assert create_request.endpoint_url == "https://embeddings.example"
    assert create_request.model_identifier == "text-embedding-3-small"
    assert create_request.dimensionality == 1536
    assert create_request.api_key == "sk-test"
    assert transport.get_embedder_calls == [{"embedder_id": "embedder-1"}]
    assert transport.list_embedder_calls == [
        {
            "label": {"env": "test"},
            "owner_id": "owner-1",
            "provider_type": "OPENAI",
        }
    ]
    assert transport.delete_embedder_calls == [{"embedder_id": "embedder-1"}]


def test_space_resource_methods_return_stable_info() -> None:
    transport = FakeResourceTransport()
    resources = _resources(transport)

    created = resources.create_space(
        " docs ",
        embedders=[GoodMemSpaceEmbedder(embedder_id=" embedder-1 ")],
        labels={"env": "test"},
    )
    loaded = resources.get_space(" space-1 ")
    listed = resources.list_spaces(
        label={"env": "test"},
        name_filter=" docs* ",
        max_items=5,
    )
    resources.delete_space(" space-1 ")

    assert created == GoodMemSpaceInfo(
        space_id="space-1",
        name="Docs",
        labels={"env": "test"},
        embedder_ids=("embedder-1",),
    )
    assert loaded == created
    assert listed == [created]
    create_request = transport.create_space_calls[0]
    assert create_request.name == "docs"
    assert create_request.labels == {"env": "test"}
    assert create_request.space_embedders == [
        GoodMemSpaceEmbedder(embedder_id="embedder-1")
    ]
    assert transport.get_space_calls == [{"space_id": "space-1"}]
    assert transport.list_space_calls == [
        {
            "label": {"env": "test"},
            "name_filter": "docs*",
            "max_items": 5,
        }
    ]
    assert transport.delete_space_calls == [{"space_id": "space-1"}]


def test_memory_resource_methods_return_stable_info() -> None:
    transport = FakeResourceTransport()
    resources = _resources(transport)

    created = resources.create_memory(
        " space-1 ",
        " hello ",
        metadata={"topic": "docs"},
        memory_id=" memory-1 ",
    )
    loaded = resources.get_memory(" memory-1 ", include_content=True)
    listed = resources.list_memories(
        " space-1 ",
        filter=" val('$.topic') = 'docs' ",
        max_items=10,
    )
    resources.delete_memory(" memory-1 ")
    resources.delete_memories([" memory-1 ", "memory-2"])

    assert created == GoodMemMemoryInfo(
        memory_id="memory-1",
        space_id="space-1",
        metadata={"topic": "docs"},
        content="hello",
        processing_status="COMPLETED",
    )
    assert loaded == created
    assert listed == [created]
    create_request = transport.create_memory_calls[0]
    assert create_request.space_id == "space-1"
    assert create_request.content == " hello "
    assert create_request.metadata == {"topic": "docs"}
    assert create_request.memory_id == "memory-1"
    assert transport.get_memory_calls == [
        {"memory_id": "memory-1", "include_content": True}
    ]
    assert transport.list_memory_calls == [
        {
            "space_id": "space-1",
            "filter_expression": "val('$.topic') = 'docs'",
            "max_items": 10,
        }
    ]
    assert transport.delete_memory_calls == [{"memory_id": "memory-1"}]
    assert transport.delete_memories_calls == [
        {"memory_ids": ["memory-1", "memory-2"]}
    ]


def test_from_env_builds_connection_and_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _connection()
    transport = FakeResourceTransport()
    from_env_calls: list[dict[str, Any]] = []

    def fake_from_env(*, verify: bool | str = True) -> GoodMemConnection:
        from_env_calls.append({"verify": verify})
        return connection

    monkeypatch.setattr(
        "langchain_goodmem.resources.GoodMemConnection.from_env",
        fake_from_env,
    )
    monkeypatch.setattr(
        "langchain_goodmem.resources._create_transport",
        lambda actual_connection: transport
        if actual_connection is connection
        else pytest.fail("unexpected connection"),
    )

    resources = GoodMemResources.from_env(verify=False)

    assert resources._connection is connection
    assert resources._transport is transport
    assert from_env_calls == [{"verify": False}]


def test_resource_info_normalizes_missing_backend_fields() -> None:
    transport = FakeResourceTransport(
        embedder=SimpleNamespace(),
        space=SimpleNamespace(),
        memory=SimpleNamespace(),
    )
    resources = _resources(transport)

    assert resources.get_embedder("embedder-1") == GoodMemEmbedderInfo(
        embedder_id=None,
        display_name=None,
        provider_type=None,
        endpoint_url=None,
        model_identifier=None,
        dimensionality=None,
        supported_modalities=(),
    )
    assert resources.get_space("space-1") == GoodMemSpaceInfo(
        space_id=None,
        name=None,
        labels={},
        embedder_ids=(),
    )
    assert resources.get_memory("memory-1") == GoodMemMemoryInfo(
        memory_id=None,
        space_id=None,
        metadata={},
        content=None,
        processing_status=None,
    )


def test_resource_lists_accept_sdk_data_wrappers() -> None:
    transport = FakeResourceTransport()
    transport.list_embedders = lambda **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        data=[transport.embedder]
    )
    transport.list_spaces = lambda **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        data=[transport.space]
    )
    transport.list_memories = lambda **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        data=[transport.memory]
    )
    resources = _resources(transport)

    assert len(resources.list_embedders()) == 1
    assert len(resources.list_spaces()) == 1
    assert len(resources.list_memories(space_id="space-1")) == 1


def test_resource_lists_auto_paginate_sdk_page_like_results() -> None:
    transport = FakeResourceTransport()

    space_fetch_calls: list[str] = []
    memory_fetch_calls: list[str] = []
    transport.list_spaces = lambda **kwargs: FakeAutoPagingList(  # type: ignore[method-assign]
        data=[transport.space],
        next_token="spaces-token-2",
        pages={
            "spaces-token-2": FakeAutoPagingList(
                data=[FakeRawSpace(space_id="space-2", name="Docs 2")],
            )
        },
        fetch_calls=space_fetch_calls,
    )
    transport.list_memories = lambda **kwargs: FakeAutoPagingList(  # type: ignore[method-assign]
        data=[transport.memory],
        next_token="memories-token-2",
        pages={
            "memories-token-2": FakeAutoPagingList(
                data=[FakeRawMemory(memory_id="memory-2", original_content="hello again")],
            )
        },
        fetch_calls=memory_fetch_calls,
    )
    resources = _resources(transport)

    spaces = resources.list_spaces()
    memories = resources.list_memories(space_id="space-1")

    assert [space.space_id for space in spaces] == ["space-1", "space-2"]
    assert [memory.memory_id for memory in memories] == ["memory-1", "memory-2"]
    assert space_fetch_calls == ["spaces-token-2"]
    assert memory_fetch_calls == ["memories-token-2"]


def test_resource_lists_respect_max_items_across_sdk_pages() -> None:
    transport = FakeResourceTransport()

    space_fetch_calls: list[str] = []
    memory_fetch_calls: list[str] = []
    transport.list_spaces = lambda **kwargs: FakeAutoPagingList(  # type: ignore[method-assign]
        data=[transport.space],
        next_token="spaces-token-2",
        pages={
            "spaces-token-2": FakeAutoPagingList(
                data=[
                    FakeRawSpace(space_id="space-2", name="Docs 2"),
                    FakeRawSpace(space_id="space-3", name="Docs 3"),
                ],
            )
        },
        fetch_calls=space_fetch_calls,
        _max_items=2,
    )
    transport.list_memories = lambda **kwargs: FakeAutoPagingList(  # type: ignore[method-assign]
        data=[transport.memory],
        next_token="memories-token-2",
        pages={
            "memories-token-2": FakeAutoPagingList(
                data=[
                    FakeRawMemory(memory_id="memory-2", original_content="hello again"),
                    FakeRawMemory(memory_id="memory-3", original_content="hello later"),
                ],
            )
        },
        fetch_calls=memory_fetch_calls,
        _max_items=2,
    )
    resources = _resources(transport)

    spaces = resources.list_spaces(max_items=2)
    memories = resources.list_memories(space_id="space-1", max_items=2)

    assert [space.space_id for space in spaces] == ["space-1", "space-2"]
    assert [memory.memory_id for memory in memories] == ["memory-1", "memory-2"]
    assert space_fetch_calls == ["spaces-token-2"]
    assert memory_fetch_calls == ["memories-token-2"]


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (None, "did not return a response"),
        (object(), "did not return batch results"),
        (SimpleNamespace(results=None), "did not return batch results"),
        (FakeBatchDeleteResponse(results=[]), "result count that did not match"),
    ],
)
def test_delete_memories_rejects_malformed_batch_delete_responses(
    response: Any,
    match: str,
) -> None:
    transport = FakeResourceTransport(delete_batch_response=response)
    resources = _resources(transport)

    with pytest.raises(GoodMemAPIError, match=match):
        resources.delete_memories(["memory-1"])

    assert transport.delete_memories_calls == [{"memory_ids": ["memory-1"]}]


def test_resource_methods_validate_local_inputs_before_transport_calls() -> None:
    transport = FakeResourceTransport()
    resources = _resources(transport)

    with pytest.raises(ValueError, match="endpoint_url"):
        resources.create_embedder(
            endpoint_url="",
            model_identifier="text-embedding-3-small",
            dimensionality=1536,
        )

    with pytest.raises(ValueError, match="positive integer"):
        resources.create_embedder(
            endpoint_url="https://embeddings.example",
            model_identifier="text-embedding-3-small",
            dimensionality=0,
        )

    with pytest.raises(GoodMemConfigurationError, match="either embedders or embedding"):
        resources.create_space(name="docs")

    with pytest.raises(GoodMemConfigurationError, match="either embedders or embedding"):
        resources.create_space(
            name="docs",
            embedders=[GoodMemSpaceEmbedder(embedder_id="embedder-1")],
            embedding=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(GoodMemConfigurationError, match="GoodMemEmbeddings"):
        resources.create_space(
            name="docs",
            embedding=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="labels"):
        resources.create_space(
            name="docs",
            embedders=[GoodMemSpaceEmbedder(embedder_id="embedder-1")],
            labels={"bad": 1},  # type: ignore[dict-item]
        )

    with pytest.raises(ValueError, match="labels"):
        resources.list_spaces(label={1: "bad"})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="max_items"):
        resources.list_memories(space_id="space-1", max_items=0)

    with pytest.raises(ValueError, match="content"):
        resources.create_memory(space_id="space-1", content="")

    with pytest.raises(ValueError, match="include_content"):
        resources.get_memory("memory-1", include_content="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="memory_ids"):
        resources.delete_memories([])

    with pytest.raises(ValueError, match="memory_ids"):
        resources.delete_memories("memory-1")  # type: ignore[arg-type]

    with pytest.raises(GoodMemDuplicateIDError, match="Duplicate memory IDs"):
        resources.delete_memories([" memory-1 ", "memory-1"])

    assert transport.create_embedder_calls == []
    assert transport.create_space_calls == []
    assert transport.delete_memories_calls == []


def test_bootstrap_vector_store_rejects_missing_created_space_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeResourceTransport(space=FakeRawSpace(space_id=None))  # type: ignore[arg-type]
    resources = _resources(transport)
    monkeypatch.setattr(
        "langchain_goodmem.embeddings._create_transport",
        lambda connection: object(),
    )
    embedding = GoodMemEmbeddings(
        embedder_id="embedder-1",
        connection=_connection(),
    )
    monkeypatch.setattr(
        "langchain_goodmem.resources.GoodMemEmbeddings.ensure",
        staticmethod(lambda **kwargs: embedding),
    )

    with pytest.raises(GoodMemConfigurationError, match="space ID"):
        resources.bootstrap_vector_store(
            "docs",
            "https://embeddings.example",
            "text-embedding-3-small",
            1536,
        )


def test_bootstrap_vector_store_creates_space_and_returns_bound_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeResourceTransport(space=FakeRawSpace(space_id="boot-space"))
    resources = _resources(transport)
    monkeypatch.setattr(
        "langchain_goodmem.embeddings._create_transport",
        lambda connection: object(),
    )
    embedding = GoodMemEmbeddings(
        embedder_id="embedder-1",
        connection=_connection(),
    )
    ensure_calls: list[dict[str, Any]] = []

    def fake_ensure(**kwargs: Any) -> GoodMemEmbeddings:
        ensure_calls.append(kwargs)
        return embedding

    monkeypatch.setattr(
        "langchain_goodmem.resources.GoodMemEmbeddings.ensure",
        staticmethod(fake_ensure),
    )

    store = resources.bootstrap_vector_store(
        "docs",
        "https://embeddings.example",
        "text-embedding-3-small",
        1536,
        upstream_api_key="sk-test",
        embedder_display_name="demo embedder",
    )

    assert isinstance(store, GoodMemVectorStore)
    assert store.space_id == "boot-space"
    assert store.embeddings is embedding
    assert ensure_calls == [
        {
            "connection": resources._connection,
            "endpoint_url": "https://embeddings.example",
            "model_identifier": "text-embedding-3-small",
            "dimensionality": 1536,
            "upstream_api_key": "sk-test",
            "display_name": "demo embedder",
        }
    ]
    create_request = transport.create_space_calls[0]
    assert create_request.name == "docs"
    assert create_request.space_embedders == [
        GoodMemSpaceEmbedder(embedder_id="embedder-1")
    ]
