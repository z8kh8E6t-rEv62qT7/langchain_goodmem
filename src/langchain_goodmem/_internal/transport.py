"""Official GoodMem SDK transport boundary with normalized exceptions.

This module is the only package layer that talks directly to the GoodMem SDK.

Responsibilities:

- construct the SDK client from ``GoodMemConnection``
- map package-owned request shapes onto SDK request objects
- normalize SDK-specific exceptions into package-owned errors
- preserve consumer exceptions when retrieval streams are consumed lazily
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..connection import GoodMemConnection
from ..errors import GoodMemAPIError, GoodMemConfigurationError, GoodMemDuplicateIDError
from .types import GoodMemSpaceCreateRequest, GoodMemWriteRequest

_SDK_ERROR_DETAIL_LIMIT = 500


class GoodMemTransport:
    """Official GoodMem SDK backed transport implementation."""

    def __init__(self, connection: GoodMemConnection) -> None:
        try:
            from goodmem import Goodmem
            from goodmem.errors import APIError, ConflictError, GoodMemError
        except ImportError as exc:
            raise GoodMemConfigurationError(
                "The 'goodmem' dependency is required to use langchain-goodmem integrations."
            ) from exc

        self._api_error_type = APIError
        self._conflict_error_type = ConflictError
        self._goodmem_error_type = GoodMemError
        self._client = Goodmem(
            base_url=connection.base_url,
            api_key=connection.api_key,
            verify=connection.verify,
        )

    def create_space(self, request: GoodMemSpaceCreateRequest) -> Any:
        """Create a GoodMem space from one normalized request payload.

        Args:
            request: Package-owned create-space payload.

        Returns:
            The raw GoodMem SDK response object for the created space.

        Raises:
            GoodMemDuplicateIDError: If GoodMem reports that the space already
                exists.
            GoodMemAPIError: If the SDK rejects the request or raises any other
                backend failure.
        """
        from goodmem.types import SpaceEmbedderConfig

        try:
            return self._client.spaces.create(
                name=request.name,
                space_embedders=[
                    SpaceEmbedderConfig(
                        embedder_id=embedder.embedder_id,
                        default_retrieval_weight=embedder.default_retrieval_weight,
                    )
                    for embedder in request.space_embedders
                ],
            )
        except self._conflict_error_type as exc:
            raise GoodMemDuplicateIDError(
                "GoodMem reported that the requested space already exists."
            ) from exc
        except self._api_error_type as exc:
            raise _normalize_sdk_api_error(exc) from exc
        except self._goodmem_error_type as exc:
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive SDK guard
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc

    def batch_create_memories(
        self,
        *,
        space_id: str,
        writes: list[GoodMemWriteRequest],
    ) -> Any:
        """Create one batch of GoodMem memories from normalized write payloads.

        Args:
            space_id: Target GoodMem space ID.
            writes: Package-owned memory-write payloads in request order.

        Returns:
            The raw GoodMem SDK batch-create response.

        Raises:
            GoodMemDuplicateIDError: If GoodMem reports duplicate memory IDs.
            GoodMemAPIError: If the SDK rejects the request or raises any other
                backend failure.
        """
        from goodmem import MemoryCreationRequest

        try:
            return self._client.memories.batch_create(
                requests=[
                    MemoryCreationRequest(
                        memory_id=write.memory_id,
                        space_id=space_id,
                        original_content=write.page_content,
                        content_type=write.content_type,
                        metadata=write.metadata or None,
                    )
                    for write in writes
                ]
            )
        except self._conflict_error_type as exc:
            raise GoodMemDuplicateIDError(
                "GoodMem reported that the memory ID already exists."
            ) from exc
        except self._api_error_type as exc:
            raise _normalize_sdk_api_error(exc) from exc
        except self._goodmem_error_type as exc:
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive SDK guard
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc

    @contextmanager
    def retrieve_memories(
        self,
        *,
        space_id: str,
        query: str,
        k: int,
        filter_expression: str | None = None,
    ) -> Iterator[Any]:
        """Yield one GoodMem retrieval stream while normalizing setup failures.

        Args:
            space_id: Target GoodMem space ID.
            query: Semantic retrieval query text.
            k: Requested result count forwarded to GoodMem.
            filter_expression: Optional raw GoodMem filter expression string.

        Yields:
            The SDK-managed stream of retrieval events.

        Raises:
            GoodMemAPIError: If stream setup or stream teardown fails for a
                backend reason. Consumer exceptions raised while iterating the
                yielded events are preserved as-is.
        """
        from goodmem.types import SpaceKey

        try:
            stream = self._client.memories.retrieve(
                message=query,
                requested_size=k,
                space_keys=[SpaceKey(space_id=space_id, filter=filter_expression)],
                fetch_memory=True,
                fetch_memory_content=False,
                stream=True,
            )
        except self._api_error_type as exc:
            raise _normalize_sdk_api_error(exc) from exc
        except self._goodmem_error_type as exc:
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc
        except Exception as exc:
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc

        consumer_exception: Exception | None = None
        try:
            with stream as events:
                try:
                    yield events
                except Exception as exc:
                    consumer_exception = exc
                    raise
        except self._api_error_type as exc:
            if exc is consumer_exception:
                raise
            raise _normalize_sdk_api_error(exc) from exc
        except self._goodmem_error_type as exc:
            if exc is consumer_exception:
                raise
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc
        except Exception as exc:
            if exc is consumer_exception:
                raise
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc

    def get_embedder(self, *, embedder_id: str) -> Any:
        """Load one GoodMem embedder response by ID.

        Args:
            embedder_id: GoodMem embedder identifier to resolve.

        Returns:
            The raw GoodMem SDK embedder response.

        Raises:
            GoodMemAPIError: If GoodMem rejects the lookup or any other backend
                failure occurs.
        """
        try:
            return self._client.embedders.get(id=embedder_id)
        except self._api_error_type as exc:
            raise _normalize_sdk_api_error(exc) from exc
        except self._goodmem_error_type as exc:
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive SDK guard
            raise GoodMemAPIError(_describe_generic_backend_failure(exc)) from exc


def _normalize_sdk_api_error(exc: Exception) -> GoodMemAPIError:
    """Convert one GoodMem SDK API exception into a package-owned error."""
    status = getattr(exc, "status_code", None)
    if status == 409:
        return GoodMemDuplicateIDError("GoodMem reported that the resource already exists.")
    detail = _sdk_api_error_detail(exc, status=status)
    if status is not None:
        if detail is not None:
            return GoodMemAPIError(f"GoodMem request failed with status {status}: {detail}")
        return GoodMemAPIError(f"GoodMem request failed with status {status}.")
    if detail is not None:
        return GoodMemAPIError(f"GoodMem request failed: {detail}")
    return GoodMemAPIError("GoodMem request failed.")


def _describe_generic_backend_failure(exc: Exception) -> str:
    """Describe one non-API backend failure with bounded detail text."""
    message = _bounded_detail_text(str(exc))
    if message is not None:
        return f"GoodMem request failed: {message}"
    return "GoodMem request failed."


def _bounded_detail_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    return text[:_SDK_ERROR_DETAIL_LIMIT]


def _sdk_api_error_detail(exc: Exception, *, status: int | None) -> str | None:
    body_detail = _bounded_detail_text(getattr(exc, "body", None))
    if body_detail is not None:
        return body_detail

    message_detail = _bounded_detail_text(str(exc))
    if message_detail is None:
        return None
    if status is None:
        return message_detail

    status_prefix = f"HTTP {status}"
    if message_detail == status_prefix:
        return None
    if message_detail.startswith(f"{status_prefix}:"):
        stripped = message_detail[len(status_prefix) + 1 :].strip()
        return stripped or None
    if message_detail.startswith(status_prefix):
        stripped = message_detail[len(status_prefix) :].strip()
        return stripped or None
    return message_detail
