"""Unit tests for the GoodMem transport boundary.

This suite validates the package layer that maps normalized requests onto the
official GoodMem SDK and converts SDK exceptions back into package-owned
errors.

Coverage goals:

- create-space request mapping
- API error normalization, including duplicate-resource handling and truncated
  backend detail
- streamed retrieval setup failures, context-manager failures, and preservation
  of consumer exceptions raised while iterating the stream

The suite stands alone because transport correctness depends on edge cases in
SDK interaction rather than on public LangChain-facing behavior.
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import pytest
from goodmem.errors import APIError, ConflictError, GoodMemError

from langchain_goodmem import (
    GoodMemAPIError,
    GoodMemConfigurationError,
    GoodMemConnection,
    GoodMemDuplicateIDError,
    GoodMemSpaceEmbedder,
)
from langchain_goodmem._internal.transport import (
    GoodMemTransport,
    _describe_generic_backend_failure,
    _normalize_sdk_api_error,
)
from langchain_goodmem._internal.types import (
    GoodMemEmbedderBootstrapRequest,
    GoodMemMemoryCreateRequest,
    GoodMemSpaceCreateRequest,
)


class ConsumerError(Exception):
    pass


class ConsumerApiError(APIError):
    pass


class FakeRetrieveStream:
    def __init__(
        self,
        *,
        events: Any = None,
        enter_exception: Exception | None = None,
        exit_exception: Exception | None = None,
    ) -> None:
        self.events = [] if events is None else events
        self.enter_exception = enter_exception
        self.exit_exception = exit_exception
        self.enter_calls = 0
        self.exit_calls = 0
        self.exit_args: list[
            tuple[type[BaseException] | None, BaseException | None]
        ] = []

    def __enter__(self) -> Any:
        self.enter_calls += 1
        if self.enter_exception is not None:
            raise self.enter_exception
        return self.events

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: Any,
    ) -> bool:
        self.exit_calls += 1
        self.exit_args.append((exc_type, exc))
        if self.exit_exception is not None:
            raise self.exit_exception
        return False


class FakeMemoriesClient:
    def __init__(
        self,
        *,
        response: Any = None,
        stream: FakeRetrieveStream | None = None,
        retrieve_exception: Exception | None = None,
    ) -> None:
        self.response = response
        self.stream = stream
        self.retrieve_exception = retrieve_exception
        self.create_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.batch_delete_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self.response

    def get(self, **kwargs: Any) -> Any:
        self.get_calls.append(kwargs)
        return self.response

    def list(self, **kwargs: Any) -> list[Any]:
        self.list_calls.append(kwargs)
        return [self.response]

    def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)

    def batch_delete(self, **kwargs: Any) -> Any:
        self.batch_delete_calls.append(kwargs)
        return self.response

    def retrieve(self, **kwargs: Any) -> FakeRetrieveStream:
        self.retrieve_calls.append(kwargs)
        if self.retrieve_exception is not None:
            raise self.retrieve_exception
        assert self.stream is not None
        return self.stream


class FakeSpacesClient:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.create_calls.append(kwargs)
        return {"space_id": "space-123"}

    def get(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        return {"space_id": "space-123"}

    def list(self, **kwargs: Any) -> list[dict[str, str]]:
        self.list_calls.append(kwargs)
        return [{"space_id": "space-123"}]

    def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)


class FakeClient:
    def __init__(
        self,
        memories: FakeMemoriesClient,
        spaces: FakeSpacesClient | None = None,
        embedders: object | None = None,
    ) -> None:
        self.memories = memories
        self.spaces = spaces or FakeSpacesClient()
        self.embedders = embedders or FakeEmbeddersClient(object())


def _make_transport(
    *,
    stream: FakeRetrieveStream | None = None,
    retrieve_exception: Exception | None = None,
) -> tuple[GoodMemTransport, FakeMemoriesClient]:
    memories = FakeMemoriesClient(
        stream=stream,
        retrieve_exception=retrieve_exception,
    )
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = APIError
    transport._goodmem_error_type = GoodMemError
    transport._client = FakeClient(memories)
    return transport, memories


class ErroringSpacesClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def create(self, **kwargs: Any) -> Any:
        raise self.exc

    def get(self, **kwargs: Any) -> Any:
        raise self.exc

    def list(self, **kwargs: Any) -> Any:
        raise self.exc

    def delete(self, **kwargs: Any) -> Any:
        raise self.exc


class ErroringMemoriesClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def batch_create(self, **kwargs: Any) -> Any:
        raise self.exc

    def create(self, **kwargs: Any) -> Any:
        raise self.exc

    def get(self, **kwargs: Any) -> Any:
        raise self.exc

    def list(self, **kwargs: Any) -> Any:
        raise self.exc

    def delete(self, **kwargs: Any) -> Any:
        raise self.exc

    def batch_delete(self, **kwargs: Any) -> Any:
        raise self.exc


class ErroringEmbeddersClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def get(self, **kwargs: Any) -> Any:
        raise self.exc

    def list(self, **kwargs: Any) -> Any:
        raise self.exc

    def create(self, **kwargs: Any) -> Any:
        raise self.exc

    def delete(self, **kwargs: Any) -> Any:
        raise self.exc


class FakeEmbeddersClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def get(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response

    def list(self, **kwargs: Any) -> Any:
        self.list_calls.append(kwargs)
        return [self.response]

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self.response

    def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)


def test_create_space_maps_package_owned_embedder_config_to_sdk_types() -> None:
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = APIError
    transport._goodmem_error_type = GoodMemError
    spaces = FakeSpacesClient()
    transport._client = FakeClient(FakeMemoriesClient(), spaces=spaces)

    response = transport.create_space(
        GoodMemSpaceCreateRequest(
            name="docs-space",
            space_embedders=[
                GoodMemSpaceEmbedder(
                    embedder_id="embedder-123",
                    default_retrieval_weight=0.5,
                )
            ],
        )
    )

    assert response == {"space_id": "space-123"}
    assert len(spaces.create_calls) == 1
    create_call = spaces.create_calls[0]
    assert create_call["name"] == "docs-space"
    assert create_call["labels"] is None
    embedder = create_call["space_embedders"][0]
    assert embedder.embedder_id == "embedder-123"
    assert embedder.default_retrieval_weight == 0.5


def test_space_resource_methods_map_to_sdk_calls() -> None:
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    spaces = FakeSpacesClient()
    transport._client = FakeClient(FakeMemoriesClient(), spaces=spaces)

    assert transport.get_space(space_id="space-123") == {"space_id": "space-123"}
    assert transport.list_spaces(
        label={"env": "test"},
        name_filter="demo*",
        max_items=3,
    ) == [{"space_id": "space-123"}]
    assert transport.delete_space(space_id="space-123") is None

    assert spaces.get_calls == [{"id": "space-123"}]
    assert spaces.list_calls == [
        {
            "label": {"env": "test"},
            "name_filter": "demo*",
            "max_items": 3,
        }
    ]
    assert spaces.delete_calls == [{"id": "space-123"}]


def test_memory_resource_methods_map_to_sdk_calls() -> None:
    response = object()
    memories = FakeMemoriesClient(response=response)
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = FakeClient(memories)

    assert (
        transport.create_memory(
            GoodMemMemoryCreateRequest(
                space_id="space-123",
                content="hello",
                metadata={"topic": "docs"},
                memory_id="memory-1",
            )
        )
        is response
    )
    assert transport.get_memory(memory_id="memory-1", include_content=True) is response
    assert transport.list_memories(
        space_id="space-123",
        filter_expression="val('$.topic') = 'docs'",
        max_items=2,
    ) == [response]
    assert transport.delete_memory(memory_id="memory-1") is None
    assert transport.delete_memories(memory_ids=["memory-1", "memory-2"]) is response

    assert memories.create_calls == [
        {
            "space_id": "space-123",
            "original_content": "hello",
            "metadata": {"topic": "docs"},
            "memory_id": "memory-1",
        }
    ]
    assert memories.get_calls == [{"id": "memory-1", "include_content": True}]
    assert memories.list_calls == [
        {
            "space_id": "space-123",
            "filter": "val('$.topic') = 'docs'",
            "max_items": 2,
        }
    ]
    assert memories.delete_calls == [{"id": "memory-1"}]
    selectors = memories.batch_delete_calls[0]["requests"]
    assert [selector.memory_id for selector in selectors] == [
        "memory-1",
        "memory-2",
    ]


def test_constructor_builds_sdk_client_from_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import goodmem

    captured: dict[str, Any] = {}

    class FakeSdkClient:
        def __init__(self, *, base_url: str, api_key: str, verify: bool | str) -> None:
            captured.update(
                {
                    "base_url": base_url,
                    "api_key": api_key,
                    "verify": verify,
                }
            )

    monkeypatch.setattr(goodmem, "Goodmem", FakeSdkClient)

    transport = GoodMemTransport(
        GoodMemConnection(
            api_key="gm-key",
            base_url="https://goodmem.example",
            verify="custom-ca.pem",
        )
    )

    assert captured == {
        "base_url": "https://goodmem.example",
        "api_key": "gm-key",
        "verify": "custom-ca.pem",
    }
    assert isinstance(transport._client, FakeSdkClient)


def test_constructor_missing_goodmem_dependency_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "goodmem":
            raise ImportError("No module named 'goodmem'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(GoodMemConfigurationError, match="goodmem"):
        GoodMemTransport(
            GoodMemConnection(api_key="gm-key", base_url="https://goodmem.example")
        )


def test_normalize_sdk_api_error_preserves_status_and_body_detail() -> None:
    exc = APIError(
        'HTTP 422: {"detail":"bad filter"}',
        status_code=422,
        body='{"detail":"bad filter"}',
    )

    normalized = _normalize_sdk_api_error(exc)

    assert isinstance(normalized, GoodMemAPIError)
    assert str(normalized) == (
        'GoodMem request failed with status 422: {"detail":"bad filter"}'
    )


def test_normalize_sdk_api_error_truncates_backend_body_to_sdk_limit() -> None:
    body = "x" * 600
    exc = APIError(
        f"HTTP 500: {body[:500]}",
        status_code=500,
        body=body,
    )

    normalized = _normalize_sdk_api_error(exc)

    assert isinstance(normalized, GoodMemAPIError)
    assert str(normalized) == f"GoodMem request failed with status 500: {body[:500]}"


def test_normalize_sdk_api_error_keeps_duplicate_mapping() -> None:
    exc = APIError(
        "HTTP 409: already exists",
        status_code=409,
        body="already exists",
    )

    normalized = _normalize_sdk_api_error(exc)

    assert isinstance(normalized, GoodMemDuplicateIDError)
    assert str(normalized) == "GoodMem reported that the resource already exists."


def test_normalize_sdk_api_error_uses_message_when_body_is_missing() -> None:
    exc_without_detail = APIError("HTTP 500", status_code=500, body=None)
    exc_with_suffix = APIError(
        "HTTP 500 backend exploded",
        status_code=500,
        body=None,
    )
    exc_without_status = Exception("backend exploded")

    normalized_without_detail = _normalize_sdk_api_error(exc_without_detail)
    normalized_with_suffix = _normalize_sdk_api_error(exc_with_suffix)
    normalized_without_status = _normalize_sdk_api_error(exc_without_status)

    assert str(normalized_without_detail) == "GoodMem request failed with status 500."
    assert (
        str(normalized_with_suffix)
        == "GoodMem request failed with status 500: backend exploded"
    )
    assert str(normalized_without_status) == "GoodMem request failed: backend exploded"


@pytest.mark.parametrize(
    ("exc", "expected_type", "match"),
    [
        (
            ConflictError("already exists", status_code=409, body="already exists"),
            GoodMemDuplicateIDError,
            "requested space already exists",
        ),
        (
            APIError("HTTP 422: bad create", status_code=422, body="bad create"),
            GoodMemAPIError,
            "status 422: bad create",
        ),
        (
            GoodMemError("space backend boom"),
            GoodMemAPIError,
            "space backend boom",
        ),
        (
            RuntimeError("space runtime boom"),
            GoodMemAPIError,
            "space runtime boom",
        ),
    ],
)
def test_create_space_normalizes_backend_failures(
    exc: Exception,
    expected_type: type[Exception],
    match: str,
) -> None:
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(spaces=ErroringSpacesClient(exc))

    with pytest.raises(expected_type, match=match):
        transport.create_space(
            GoodMemSpaceCreateRequest(
                name="docs-space",
                space_embedders=[GoodMemSpaceEmbedder(embedder_id="embedder-123")],
            )
        )


@pytest.mark.parametrize(
    ("exc", "expected_type", "match"),
    [
        (
            ConflictError("already exists", status_code=409, body="already exists"),
            GoodMemDuplicateIDError,
            "memory ID already exists",
        ),
        (
            APIError("HTTP 422: bad write", status_code=422, body="bad write"),
            GoodMemAPIError,
            "status 422: bad write",
        ),
        (
            GoodMemError("memory backend boom"),
            GoodMemAPIError,
            "memory backend boom",
        ),
        (
            RuntimeError("memory runtime boom"),
            GoodMemAPIError,
            "memory runtime boom",
        ),
    ],
)
def test_batch_create_memories_normalizes_backend_failures(
    exc: Exception,
    expected_type: type[Exception],
    match: str,
) -> None:
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(memories=ErroringMemoriesClient(exc))

    with pytest.raises(expected_type, match=match):
        transport.batch_create_memories(space_id="space-123", writes=[])


@pytest.mark.parametrize(
    ("exc", "expected_type", "match"),
    [
        (
            APIError("HTTP 404: missing", status_code=404, body="missing"),
            GoodMemAPIError,
            "status 404: missing",
        ),
        (
            GoodMemError("embedder backend boom"),
            GoodMemAPIError,
            "embedder backend boom",
        ),
        (
            RuntimeError("embedder runtime boom"),
            GoodMemAPIError,
            "embedder runtime boom",
        ),
    ],
)
def test_get_embedder_normalizes_backend_failures(
    exc: Exception,
    expected_type: type[Exception],
    match: str,
) -> None:
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(embedders=ErroringEmbeddersClient(exc))

    with pytest.raises(expected_type, match=match):
        transport.get_embedder(embedder_id="embedder-123")


def test_retrieve_memories_preserves_consumer_exceptions() -> None:
    stream = FakeRetrieveStream(events=["event-1", "event-2"])
    transport, memories = _make_transport(stream=stream)

    with pytest.raises(ConsumerError, match="consumer boom"):
        with transport.retrieve_memories(
            space_id="space-123",
            query="hello",
            k=2,
            filter_expression="topic = 'docs'",
        ) as events:
            assert events == ["event-1", "event-2"]
            raise ConsumerError("consumer boom")

    assert len(memories.retrieve_calls) == 1
    retrieve_call = memories.retrieve_calls[0]
    assert retrieve_call["message"] == "hello"
    assert retrieve_call["requested_size"] == 2
    assert retrieve_call["fetch_memory"] is True
    assert retrieve_call["fetch_memory_content"] is False
    assert retrieve_call["stream"] is True
    space_key = retrieve_call["space_keys"][0]
    assert space_key.space_id == "space-123"
    assert space_key.filter == "topic = 'docs'"
    assert stream.enter_calls == 1
    assert stream.exit_calls == 1
    exit_type, exit_exc = stream.exit_args[0]
    assert exit_type is ConsumerError
    assert isinstance(exit_exc, ConsumerError)
    assert str(exit_exc) == "consumer boom"


def test_retrieve_memories_yields_events_without_consumer_exception() -> None:
    stream = FakeRetrieveStream(events=["event-1", "event-2"])
    transport, memories = _make_transport(stream=stream)

    with transport.retrieve_memories(
        space_id="space-123",
        query="hello",
        k=2,
    ) as events:
        assert events == ["event-1", "event-2"]

    assert len(memories.retrieve_calls) == 1
    assert stream.enter_calls == 1
    assert stream.exit_calls == 1


def test_retrieve_memories_preserves_api_error_consumer_exceptions() -> None:
    stream = FakeRetrieveStream(events=["event-1"])
    transport, _ = _make_transport(stream=stream)
    consumer_exc = ConsumerApiError(
        "HTTP 429: consumer boom",
        status_code=429,
        body="consumer boom",
    )

    with pytest.raises(ConsumerApiError) as exc_info:
        with transport.retrieve_memories(
            space_id="space-123",
            query="hello",
            k=1,
        ) as events:
            assert events == ["event-1"]
            raise consumer_exc

    assert exc_info.value is consumer_exc


def test_retrieve_memories_preserves_goodmem_error_consumer_exceptions() -> None:
    stream = FakeRetrieveStream(events=["event-1"])
    transport, _ = _make_transport(stream=stream)
    consumer_exc = GoodMemError("consumer backend boom")

    with pytest.raises(GoodMemError) as exc_info:
        with transport.retrieve_memories(
            space_id="space-123",
            query="hello",
            k=1,
        ) as events:
            assert events == ["event-1"]
            raise consumer_exc

    assert exc_info.value is consumer_exc


@pytest.mark.parametrize(
    ("exc", "expected_type", "match"),
    [
        (
            APIError("HTTP 422: bad filter", status_code=422, body="bad filter"),
            GoodMemAPIError,
            "GoodMem request failed with status 422: bad filter",
        ),
        (
            GoodMemError("backend setup boom"),
            GoodMemAPIError,
            "GoodMem request failed: backend setup boom",
        ),
        (
            RuntimeError("socket setup boom"),
            GoodMemAPIError,
            "GoodMem request failed: socket setup boom",
        ),
    ],
)
def test_retrieve_memories_normalizes_setup_failures(
    exc: Exception,
    expected_type: type[Exception],
    match: str,
) -> None:
    transport, memories = _make_transport(retrieve_exception=exc)

    with pytest.raises(expected_type, match=match):
        with transport.retrieve_memories(
            space_id="space-123",
            query="hello",
            k=2,
        ):
            pytest.fail("retrieve_memories should not yield when setup fails")

    assert len(memories.retrieve_calls) == 1


@pytest.mark.parametrize(
    ("stream", "expected_type", "match"),
    [
        (
            FakeRetrieveStream(
                enter_exception=APIError(
                    "HTTP 504: stream enter timeout",
                    status_code=504,
                    body="stream enter timeout",
                )
            ),
            GoodMemAPIError,
            "GoodMem request failed with status 504: stream enter timeout",
        ),
        (
            FakeRetrieveStream(exit_exception=GoodMemError("stream exit backend boom")),
            GoodMemAPIError,
            "GoodMem request failed: stream exit backend boom",
        ),
        (
            FakeRetrieveStream(exit_exception=RuntimeError("stream exit crash")),
            GoodMemAPIError,
            "GoodMem request failed: stream exit crash",
        ),
    ],
)
def test_retrieve_memories_normalizes_stream_context_manager_failures(
    stream: FakeRetrieveStream,
    expected_type: type[Exception],
    match: str,
) -> None:
    transport, _ = _make_transport(stream=stream)

    with pytest.raises(expected_type, match=match):
        with transport.retrieve_memories(
            space_id="space-123",
            query="hello",
            k=2,
        ) as events:
            assert events == []

    assert stream.enter_calls == 1
    assert stream.exit_calls == (0 if stream.enter_exception is not None else 1)


def test_get_embedder_successfully_returns_sdk_response() -> None:
    response = object()
    embedders = FakeEmbeddersClient(response)
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(embedders=embedders)

    assert transport.get_embedder(embedder_id="embedder-123") is response
    assert embedders.calls == [{"id": "embedder-123"}]


def test_list_embedders_successfully_returns_sdk_response() -> None:
    response = object()
    embedders = FakeEmbeddersClient(response)
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(embedders=embedders)

    assert transport.list_embedders() == [response]
    assert embedders.list_calls == [{}]


def test_list_embedders_forwards_resource_filters() -> None:
    response = object()
    embedders = FakeEmbeddersClient(response)
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(embedders=embedders)

    assert transport.list_embedders(
        label={"env": "test"},
        owner_id="owner-1",
        provider_type="OPENAI",
    ) == [response]

    call = embedders.list_calls[0]
    assert call["label"] == {"env": "test"}
    assert call["owner_id"] == "owner-1"
    assert getattr(call["provider_type"], "value", call["provider_type"]) == "OPENAI"


def test_delete_embedder_maps_to_sdk_call() -> None:
    response = object()
    embedders = FakeEmbeddersClient(response)
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(embedders=embedders)

    assert transport.delete_embedder(embedder_id="embedder-123") is None

    assert embedders.delete_calls == [{"id": "embedder-123"}]


def test_create_embedder_maps_bootstrap_request_to_sdk_types() -> None:
    response = object()
    embedders = FakeEmbeddersClient(response)
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(embedders=embedders)

    result = transport.create_embedder(
        GoodMemEmbedderBootstrapRequest(
            display_name="langchain-goodmem-openai",
            endpoint_url="https://embeddings.example",
            model_identifier="text-embedding-3-large",
            dimensionality=1024,
            api_key="upstream-key",
        )
    )

    assert result is response
    assert len(embedders.create_calls) == 1
    create_call = embedders.create_calls[0]
    assert create_call["display_name"] == "langchain-goodmem-openai"
    assert create_call["endpoint_url"] == "https://embeddings.example"
    assert create_call["model_identifier"] == "text-embedding-3-large"
    assert create_call["dimensionality"] == 1024
    assert (
        getattr(create_call["provider_type"], "value", create_call["provider_type"])
        == "OPENAI"
    )
    supported_modalities = create_call["supported_modalities"]
    assert len(supported_modalities) == 1
    assert getattr(supported_modalities[0], "value", supported_modalities[0]) == "TEXT"
    assert create_call["api_key"] == "upstream-key"


def test_get_embedder_normalizes_generic_failures_without_message() -> None:
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(
        embedders=ErroringEmbeddersClient(RuntimeError(""))
    )

    with pytest.raises(GoodMemAPIError, match="GoodMem request failed\\.$"):
        transport.get_embedder(embedder_id="embedder-123")


@pytest.mark.parametrize(
    ("method_name", "call", "expected_type", "match"),
    [
        (
            "list_embedders",
            lambda transport: transport.list_embedders(),
            GoodMemAPIError,
            "status 404: missing",
        ),
        (
            "create_embedder",
            lambda transport: transport.create_embedder(
                GoodMemEmbedderBootstrapRequest(
                    display_name="langchain-goodmem-openai",
                    endpoint_url="https://embeddings.example",
                    model_identifier="text-embedding-3-large",
                    dimensionality=1024,
                )
            ),
            GoodMemAPIError,
            "status 404: missing",
        ),
    ],
)
def test_bootstrap_embedder_transport_normalizes_backend_failures(
    method_name: str,
    call: Any,
    expected_type: type[Exception],
    match: str,
) -> None:
    exc = APIError("HTTP 404: missing", status_code=404, body="missing")
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(embedders=ErroringEmbeddersClient(exc))

    with pytest.raises(expected_type, match=match):
        call(transport)


@pytest.mark.parametrize(
    ("client_attr", "client", "call", "match"),
    [
        (
            "spaces",
            ErroringSpacesClient(
                APIError("HTTP 500: space boom", status_code=500, body="space boom")
            ),
            lambda transport: transport.get_space(space_id="space-123"),
            "status 500: space boom",
        ),
        (
            "spaces",
            ErroringSpacesClient(GoodMemError("space backend boom")),
            lambda transport: transport.delete_space(space_id="space-123"),
            "space backend boom",
        ),
        (
            "memories",
            ErroringMemoriesClient(
                APIError("HTTP 500: memory boom", status_code=500, body="memory boom")
            ),
            lambda transport: transport.get_memory(memory_id="memory-1"),
            "status 500: memory boom",
        ),
        (
            "memories",
            ErroringMemoriesClient(RuntimeError("memory runtime boom")),
            lambda transport: transport.delete_memories(memory_ids=["memory-1"]),
            "memory runtime boom",
        ),
        (
            "embedders",
            ErroringEmbeddersClient(
                APIError(
                    "HTTP 500: embedder boom", status_code=500, body="embedder boom"
                )
            ),
            lambda transport: transport.delete_embedder(embedder_id="embedder-123"),
            "status 500: embedder boom",
        ),
    ],
)
def test_resource_transport_methods_normalize_backend_failures(
    client_attr: str,
    client: object,
    call: Any,
    match: str,
) -> None:
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(**{client_attr: client})

    with pytest.raises(GoodMemAPIError, match=match):
        call(transport)


def test_create_embedder_maps_conflict_to_duplicate_error() -> None:
    exc = ConflictError("already exists", status_code=409, body="already exists")
    transport = GoodMemTransport.__new__(GoodMemTransport)
    transport._api_error_type = APIError
    transport._conflict_error_type = ConflictError
    transport._goodmem_error_type = GoodMemError
    transport._client = SimpleNamespace(embedders=ErroringEmbeddersClient(exc))

    with pytest.raises(
        GoodMemDuplicateIDError, match="requested embedder already exists"
    ):
        transport.create_embedder(
            GoodMemEmbedderBootstrapRequest(
                display_name="langchain-goodmem-openai",
                endpoint_url="https://embeddings.example",
                model_identifier="text-embedding-3-large",
                dimensionality=1024,
            )
        )


def test_normalize_sdk_api_error_handles_empty_message_without_status() -> None:
    normalized = _normalize_sdk_api_error(Exception(""))

    assert isinstance(normalized, GoodMemAPIError)
    assert str(normalized) == "GoodMem request failed."


def test_normalize_sdk_api_error_strips_http_prefix_without_colon() -> None:
    exc = APIError("HTTP 500 backend exploded", status_code=500, body=None)

    normalized = _normalize_sdk_api_error(exc)

    assert str(normalized) == "GoodMem request failed with status 500: backend exploded"


def test_normalize_sdk_api_error_strips_http_prefix_with_colon() -> None:
    exc = APIError("HTTP 500: backend exploded", status_code=500, body=None)

    normalized = _normalize_sdk_api_error(exc)

    assert str(normalized) == "GoodMem request failed with status 500: backend exploded"


def test_normalize_sdk_api_error_uses_non_prefixed_message_detail() -> None:
    exc = APIError("backend exploded", status_code=500, body=None)

    normalized = _normalize_sdk_api_error(exc)

    assert str(normalized) == "GoodMem request failed with status 500: backend exploded"


def test_describe_generic_backend_failure_handles_blank_messages() -> None:
    assert _describe_generic_backend_failure(Exception("")) == "GoodMem request failed."
