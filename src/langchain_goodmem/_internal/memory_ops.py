"""Private GoodMem memory-operation helpers built on the transport boundary.

This module implements the behavior layer between the public vector-store API
and the transport adapter.

Responsibilities:

- turn normalized write requests into stable batch outcomes
- preserve request ordering even when the backend returns explicit
  ``request_index`` values
- decode streamed retrieval events into flat ``GoodMemSearchHit`` values
- normalize duplicate-ID and partial-success behavior into package-owned errors
"""

from __future__ import annotations

from typing import Any

from ..errors import (
    GoodMemAPIError,
    GoodMemBatchPartialFailureError,
    GoodMemBatchWriteResultItem,
    GoodMemDuplicateIDError,
)
from .types import (
    GoodMemSearchHit,
    GoodMemSpaceCreateRequest,
    GoodMemWriteRequest,
    SupportsMemoryOperationsTransport,
)


def create_space(
    transport: SupportsMemoryOperationsTransport,
    request: GoodMemSpaceCreateRequest,
) -> str:
    """Create a GoodMem space through the transport boundary.

    Args:
        transport: Transport implementation exposing create-space behavior.
        request: Normalized create-space payload.

    Returns:
        The non-empty ``space_id`` string extracted from the transport
        response for the newly created GoodMem space.

    Raises:
        GoodMemAPIError: If GoodMem does not return a usable ``space_id``.
    """
    space = transport.create_space(request)
    space_id = getattr(space, "space_id", None)
    if not isinstance(space_id, str) or not space_id:
        raise GoodMemAPIError("GoodMem create_space did not return a space_id.")
    return space_id


def add_memories(
    transport: SupportsMemoryOperationsTransport,
    *,
    space_id: str,
    writes: list[GoodMemWriteRequest],
) -> list[str]:
    """Create one batch of GoodMem memories.

    Args:
        transport: Transport implementation exposing batch-create behavior.
        space_id: Target GoodMem space ID.
        writes: Normalized write payloads in caller order.

    Returns:
        Memory IDs in the original request order.

    Raises:
        GoodMemDuplicateIDError: If GoodMem reports duplicate IDs.
        GoodMemBatchPartialFailureError: If only part of the batch succeeds.
        GoodMemAPIError: If the response shape is invalid or the backend rejects
            the batch.
    """
    response = transport.batch_create_memories(space_id=space_id, writes=writes)
    return _resolve_batch_create_response(response, writes=writes)


def delete_memories(
    transport: SupportsMemoryOperationsTransport,
    *,
    memory_ids: list[str],
) -> None:
    """Delete a batch of GoodMem memories.

    Args:
        transport: Transport implementation exposing batch-delete behavior.
        memory_ids: GoodMem memory IDs to delete.

    Raises:
        GoodMemAPIError: If the response is missing, malformed, reports failed
            selectors, or does not match the requested delete count.
    """
    response = transport.delete_memories(memory_ids=memory_ids)
    _raise_for_failed_batch_delete_response(response, expected_length=len(memory_ids))


def search_memories(
    transport: SupportsMemoryOperationsTransport,
    *,
    space_id: str,
    query: str,
    k: int,
    filter_expression: str | None = None,
) -> list[GoodMemSearchHit]:
    """Run chunk-level semantic retrieval and flatten the streamed response.

    Args:
        transport: Transport implementation exposing retrieval behavior.
        space_id: Target GoodMem space ID.
        query: Semantic retrieval query.
        k: Maximum number of hits to collect.
        filter_expression: Optional raw GoodMem filter expression string.

    Returns:
        Flat ``GoodMemSearchHit`` values in retrieval order.
    """
    with transport.retrieve_memories(
        space_id=space_id,
        query=query,
        k=k,
        filter_expression=filter_expression,
    ) as events:
        return _collect_search_hits(
            events,
            fallback_space_id=space_id,
            k=k,
        )


def _resolve_batch_create_response(
    response: Any,
    *,
    writes: list[GoodMemWriteRequest],
) -> list[str]:
    results = list(response.results)
    if len(results) != len(writes):
        raise GoodMemAPIError(
            "GoodMem batch_create returned a result count that did not match "
            "the input write count."
        )

    ordered_results = _order_batch_create_results(
        results,
        expected_length=len(writes),
    )
    batch_results = _build_batch_write_result_items(ordered_results, writes=writes)
    return _resolve_batch_write_outcome(batch_results)


def _raise_for_failed_batch_delete_response(
    response: Any,
    *,
    expected_length: int,
) -> None:
    if response is None:
        raise GoodMemAPIError("GoodMem batch_delete did not return a response.")

    results = getattr(response, "results", None)
    if results is None:
        raise GoodMemAPIError("GoodMem batch_delete did not return batch results.")

    result_list = list(results)
    if len(result_list) != expected_length:
        raise GoodMemAPIError(
            "GoodMem batch_delete returned a result count that did not match "
            "the input delete count."
        )

    failed_indices: list[str] = []
    failed_messages: list[str] = []
    for index, result in enumerate(result_list):
        if getattr(result, "success", False):
            continue
        failed_indices.append(str(index))
        error = getattr(result, "error", None)
        message = getattr(error, "message", None)
        if isinstance(message, str) and message:
            failed_messages.append(message)

    if not failed_indices:
        return

    index_list = ", ".join(failed_indices)
    if len(failed_indices) == 1:
        detail = f"GoodMem failed to delete memory at index {index_list}"
    else:
        detail = f"GoodMem failed to delete memories at indices {index_list}"
    if failed_messages:
        detail = f"{detail}: {'; '.join(failed_messages)}"
    raise GoodMemAPIError(f"{detail}.")


def _collect_search_hits(
    events: Any,
    *,
    fallback_space_id: str,
    k: int,
) -> list[GoodMemSearchHit]:
    memory_definitions: list[Any] = []
    hits: list[GoodMemSearchHit] = []

    for event in events:
        hit = _decode_retrieve_event(
            event,
            memory_definitions=memory_definitions,
            fallback_space_id=fallback_space_id,
        )
        if hit is None:
            continue

        hits.append(hit)
        if len(hits) >= k:
            break

    return hits


def _decode_retrieve_event(
    event: Any,
    *,
    memory_definitions: list[Any],
    fallback_space_id: str,
) -> GoodMemSearchHit | None:
    if event.memory_definition is not None:
        memory_definitions.append(event.memory_definition)

    if event.retrieved_item is None or event.retrieved_item.chunk is None:
        return None

    chunk_reference = event.retrieved_item.chunk
    chunk = chunk_reference.chunk
    memory_definition = _resolve_memory_definition(
        memory_definitions,
        chunk_reference.memory_index,
    )
    return GoodMemSearchHit(
        chunk_id=chunk.chunk_id,
        memory_id=chunk.memory_id,
        space_id=_resolve_search_space_id(
            memory_definition,
            fallback_space_id=fallback_space_id,
        ),
        page_content=chunk.chunk_text,
        metadata=_resolve_metadata(memory_definition, chunk),
        score=float(chunk_reference.relevance_score),
    )


def _resolve_search_space_id(
    memory_definition: Any | None,
    *,
    fallback_space_id: str,
) -> str:
    if memory_definition is not None:
        space_id = getattr(memory_definition, "space_id", None)
        if isinstance(space_id, str) and space_id:
            return space_id
    return fallback_space_id


def _is_duplicate_error(code: int | None, message: str | None) -> bool:
    if code in {6, 409}:
        return True
    normalized = (message or "").lower()
    return "already exists" in normalized or "already_exists" in normalized


def _build_batch_write_result_items(
    ordered_results: list[Any],
    *,
    writes: list[GoodMemWriteRequest],
) -> list[GoodMemBatchWriteResultItem]:
    batch_results: list[GoodMemBatchWriteResultItem] = []

    for index, (result, write) in enumerate(zip(ordered_results, writes, strict=True)):
        if result.success:
            memory_id = _resolve_created_memory_id(result, write=write)
            batch_results.append(
                GoodMemBatchWriteResultItem(
                    request_index=index,
                    success=True,
                    memory_id=memory_id,
                    error_code=None,
                    error_message=None,
                )
            )
            continue

        detail = result.error
        batch_results.append(
            GoodMemBatchWriteResultItem(
                request_index=index,
                success=False,
                memory_id=None,
                error_code=getattr(detail, "code", None),
                error_message=getattr(detail, "message", None),
            )
        )

    return batch_results


def _resolve_created_memory_id(
    result: Any,
    *,
    write: GoodMemWriteRequest,
) -> str:
    memory_id = None
    if result.memory is not None:
        memory_id = result.memory.memory_id
    if memory_id is None:
        memory_id = result.memory_id or write.memory_id
    if memory_id is None:
        raise GoodMemAPIError("GoodMem created a memory but did not return a memory ID.")
    return memory_id


def _resolve_batch_write_outcome(
    results: list[GoodMemBatchWriteResultItem],
) -> list[str]:
    created_ids = [result.memory_id for result in results if result.success]
    failure_results = [result for result in results if not result.success]
    assert all(memory_id is not None for memory_id in created_ids)
    resolved_created_ids = [memory_id for memory_id in created_ids if memory_id is not None]

    if not failure_results:
        return resolved_created_ids

    if resolved_created_ids:
        failure_indices = ", ".join(str(result.request_index) for result in failure_results)
        raise GoodMemBatchPartialFailureError(
            "GoodMem batch_create partially succeeded; failed input indices: "
            f"{failure_indices}. Inspect created_ids and results on "
            "GoodMemBatchPartialFailureError before retrying or reconciling.",
            created_ids=resolved_created_ids,
            results=results,
        )

    if all(_result_is_duplicate_error(result) for result in failure_results):
        if len(failure_results) == 1:
            failure_index = failure_results[0].request_index
            raise GoodMemDuplicateIDError(
                f"GoodMem rejected memory at index {failure_index} because the "
                "memory ID already exists."
            )

        failure_indices = ", ".join(str(result.request_index) for result in failure_results)
        raise GoodMemDuplicateIDError(
            "GoodMem rejected memories at indices "
            f"{failure_indices} because the memory IDs already exist."
        )

    first_failure = failure_results[0]
    detail_message = (
        first_failure.error_message
        if first_failure.error_message
        else "GoodMem rejected a batched memory creation request."
    )
    if len(failure_results) == 1:
        raise GoodMemAPIError(
            "GoodMem failed to create memory at index "
            f"{first_failure.request_index}: {detail_message}"
        )

    failure_indices = ", ".join(str(result.request_index) for result in failure_results)
    raise GoodMemAPIError(
        "GoodMem failed to create memories at indices "
        f"{failure_indices}. First error at index {first_failure.request_index}: "
        f"{detail_message}"
    )


def _result_is_duplicate_error(result: GoodMemBatchWriteResultItem) -> bool:
    return _is_duplicate_error(result.error_code, result.error_message)


def _order_batch_create_results(
    results: list[Any],
    *,
    expected_length: int,
) -> list[Any]:
    request_indices = [getattr(result, "request_index", None) for result in results]
    if all(index is None for index in request_indices):
        return list(results)

    ordered_results: list[Any | None] = [None] * expected_length

    for response_index, (result, request_index) in enumerate(
        zip(results, request_indices, strict=True)
    ):
        if not isinstance(request_index, int) or isinstance(request_index, bool):
            raise GoodMemAPIError(
                "GoodMem batch_create returned a result without a valid request_index "
                f"at response index {response_index}."
            )
        if request_index < 0 or request_index >= expected_length:
            raise GoodMemAPIError(
                "GoodMem batch_create returned an out-of-range request_index "
                f"{request_index} at response index {response_index}."
            )
        if ordered_results[request_index] is not None:
            raise GoodMemAPIError(
                "GoodMem batch_create returned duplicate request_index "
                f"{request_index}."
            )

        ordered_results[request_index] = result

    missing_indices = [
        index for index, result in enumerate(ordered_results) if result is None
    ]
    if missing_indices:
        missing_list = ", ".join(str(index) for index in missing_indices)
        raise GoodMemAPIError(
            "GoodMem batch_create did not return results for input indices: "
            f"{missing_list}."
        )

    return [result for result in ordered_results if result is not None]


def _resolve_memory_definition(
    memory_definitions: list[Any],
    memory_index: int,
) -> Any | None:
    if 0 <= memory_index < len(memory_definitions):
        return memory_definitions[memory_index]
    return None


def _resolve_metadata(memory_definition: Any | None, chunk: Any) -> dict[str, Any]:
    resolved_metadata: dict[str, Any] = {}

    memory_metadata = (
        getattr(memory_definition, "metadata", None)
        if memory_definition is not None
        else None
    )
    if isinstance(memory_metadata, dict):
        resolved_metadata.update(memory_metadata)

    chunk_metadata = getattr(chunk, "metadata", None)
    if isinstance(chunk_metadata, dict):
        resolved_metadata.update(chunk_metadata)

    return resolved_metadata
