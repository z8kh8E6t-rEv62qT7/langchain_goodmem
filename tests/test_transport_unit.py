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

from typing import Any

import pytest
from goodmem.errors import APIError, GoodMemError

from langchain_goodmem import GoodMemAPIError, GoodMemDuplicateIDError, GoodMemSpaceEmbedder
from langchain_goodmem._internal.types import GoodMemSpaceCreateRequest
from langchain_goodmem._internal.transport import GoodMemTransport, _normalize_sdk_api_error


class ConsumerError(Exception):
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
        self.exit_args: list[tuple[type[BaseException] | None, BaseException | None]] = []

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
        stream: FakeRetrieveStream | None = None,
        retrieve_exception: Exception | None = None,
    ) -> None:
        self.stream = stream
        self.retrieve_exception = retrieve_exception
        self.retrieve_calls: list[dict[str, Any]] = []

    def retrieve(self, **kwargs: Any) -> FakeRetrieveStream:
        self.retrieve_calls.append(kwargs)
        if self.retrieve_exception is not None:
            raise self.retrieve_exception
        assert self.stream is not None
        return self.stream


class FakeSpacesClient:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.create_calls.append(kwargs)
        return {"space_id": "space-123"}


class FakeClient:
    def __init__(
        self,
        memories: FakeMemoriesClient,
        spaces: FakeSpacesClient | None = None,
    ) -> None:
        self.memories = memories
        self.spaces = spaces or FakeSpacesClient()


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
    embedder = create_call["space_embedders"][0]
    assert getattr(embedder, "embedder_id") == "embedder-123"
    assert getattr(embedder, "default_retrieval_weight") == 0.5


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
    assert getattr(space_key, "space_id") == "space-123"
    assert getattr(space_key, "filter") == "topic = 'docs'"
    assert stream.enter_calls == 1
    assert stream.exit_calls == 1
    exit_type, exit_exc = stream.exit_args[0]
    assert exit_type is ConsumerError
    assert isinstance(exit_exc, ConsumerError)
    assert str(exit_exc) == "consumer boom"


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
