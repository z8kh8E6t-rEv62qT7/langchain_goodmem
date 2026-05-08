"""Shared private validators used by public entry points.

This module keeps local input validation in package-owned code so public entry
points can reject malformed values before transport code or the GoodMem SDK is
invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ..errors import GoodMemConfigurationError, GoodMemDuplicateIDError
from ..space_embedders import GoodMemSpaceEmbedder

_STRICT_STRING_ADAPTER = TypeAdapter(str)
_STRICT_BOOL_OR_STRING_ADAPTER = TypeAdapter(bool | str)
_STRICT_OPTIONAL_STRING_ADAPTER = TypeAdapter(str | None)
_STRICT_METADATA_MAPPING_ADAPTER = TypeAdapter(Mapping[str, Any])
_STRICT_EMBEDDER_LIST_ADAPTER = TypeAdapter(list[GoodMemSpaceEmbedder])
_STRICT_INTEGER_ADAPTER = TypeAdapter(int)


def require_non_empty_trimmed_string(
    value: str | None,
    *,
    error_message: str,
    exception_type: type[Exception],
) -> str:
    """Require one strict string input to be present and non-blank.

    Args:
        value: Candidate string value to validate.
        error_message: Message used when validation fails.
        exception_type: Exception class raised for invalid input.

    Returns:
        The trimmed string value.

    Raises:
        Exception: An instance of ``exception_type`` when ``value`` is missing,
            not a string, or blank after trimming.
    """
    try:
        normalized_value = _STRICT_STRING_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise exception_type(error_message) from exc

    normalized = normalized_value.strip()
    if not normalized:
        raise exception_type(error_message)
    return normalized


def require_verify_value(verify: bool | str) -> bool | str:
    """Validate the ``verify`` shape accepted by ``GoodMemConnection``.

    Args:
        verify: TLS verification value supplied by the caller.

    Returns:
        Either the original boolean value or a validated non-empty string path.

    Raises:
        GoodMemConfigurationError: If ``verify`` is neither a boolean nor a
            non-empty string.
    """
    try:
        validated = _STRICT_BOOL_OR_STRING_ADAPTER.validate_python(verify, strict=True)
    except ValidationError as exc:
        raise GoodMemConfigurationError(
            "GoodMemConnection requires verify to be a boolean or a non-empty string."
        ) from exc

    if isinstance(validated, bool):
        return validated
    return require_non_empty_trimmed_string(
        validated,
        error_message=(
            "GoodMemConnection requires verify to be a boolean or a non-empty string."
        ),
        exception_type=GoodMemConfigurationError,
    )


def require_space_id(space_id: str) -> str:
    """Validate one existing-space identifier.

    Args:
        space_id: Existing GoodMem space identifier supplied by the caller.

    Returns:
        The trimmed GoodMem space identifier.

    Raises:
        GoodMemConfigurationError: If ``space_id`` is missing, not a string, or
            blank.
    """
    return require_non_empty_trimmed_string(
        space_id,
        error_message=(
            "GoodMemVectorStore requires a non-empty space_id passed explicitly."
        ),
        exception_type=GoodMemConfigurationError,
    )


def require_embedder_id(embedder_id: str) -> str:
    """Validate one explicit GoodMem embedder identifier.

    Args:
        embedder_id: Explicit GoodMem embedder identifier supplied by the
            caller.

    Returns:
        The trimmed GoodMem embedder identifier.

    Raises:
        GoodMemConfigurationError: If ``embedder_id`` is missing, not a string,
            or blank.
    """
    return require_non_empty_trimmed_string(
        embedder_id,
        error_message=(
            "GoodMemEmbeddings requires a non-empty embedder_id passed explicitly."
        ),
        exception_type=GoodMemConfigurationError,
    )


def validate_text_inputs(
    texts: list[Any],
    *,
    label: str,
    exception_type: type[Exception],
    field_name: str | None = None,
) -> list[str]:
    """Validate one list of text inputs and preserve original ordering.

    Args:
        texts: Candidate text values to validate.
        label: Logical collection name used in validation errors.
        exception_type: Exception class raised when validation fails.
        field_name: Optional field label appended to error messages for nested
            sources such as ``Document.page_content``.

    Returns:
        The validated string values in original order, preserving their
        original text content.

    Raises:
        Exception: An instance of ``exception_type`` when any value is not a
            string or is blank after trimming.
    """
    validated_texts: list[str] = []
    for index, text in enumerate(texts):
        try:
            validated_text = _STRICT_STRING_ADAPTER.validate_python(text, strict=True)
        except ValidationError as exc:
            if field_name is None:
                raise exception_type(
                    f"{label} at index {index} must be a non-empty string."
                ) from exc
            raise exception_type(
                f"{label} at index {index} must have a non-empty {field_name} string."
            ) from exc

        if not validated_text.strip():
            if field_name is None:
                raise exception_type(
                    f"{label} at index {index} must be a non-empty string."
                )
            raise exception_type(
                f"{label} at index {index} must have a non-empty {field_name} string."
            )
        validated_texts.append(validated_text)
    return validated_texts


def validate_lengths(
    label: str,
    expected_length: int,
    *,
    metadatas: list[Any] | None = None,
    ids: list[str | None] | None = None,
) -> None:
    """Validate metadata and ID list lengths against one text/document count.

    Args:
        label: Logical collection name used in validation errors.
        expected_length: Required number of entries.
        metadatas: Optional metadata list aligned to the source collection.
        ids: Optional ID list aligned to the source collection.

    Returns:
        ``None`` when both optional lists match ``expected_length``.

    Raises:
        ValueError: If either optional list length does not match
            ``expected_length``.
    """
    if metadatas is not None and len(metadatas) != expected_length:
        raise ValueError(
            f"The number of metadatas must match the number of {label}. "
            f"Got {len(metadatas)} metadatas and {expected_length} {label}."
        )
    if ids is not None and len(ids) != expected_length:
        raise ValueError(
            f"The number of ids must match the number of {label}. "
            f"Got {len(ids)} ids and {expected_length} {label}."
        )


def normalize_metadatas(
    metadatas: list[Mapping[str, Any] | None] | None,
) -> list[dict[str, Any]] | None:
    """Normalize metadata mappings into plain dictionaries.

    Args:
        metadatas: Optional metadata values aligned to a write input
            collection.

    Returns:
        ``None`` when ``metadatas`` is ``None``; otherwise one plain-dictionary
        metadata value per input item, with ``None`` entries converted to empty
        dictionaries.

    Raises:
        ValueError: If any non-``None`` metadata value is not a mapping.
    """
    if metadatas is None:
        return None

    normalized: list[dict[str, Any]] = []
    for index, metadata in enumerate(metadatas):
        if metadata is None:
            normalized.append({})
            continue
        try:
            validated_metadata = _STRICT_METADATA_MAPPING_ADAPTER.validate_python(
                metadata,
                strict=True,
            )
        except ValidationError as exc:
            raise ValueError(f"metadatas at index {index} must be a mapping or None.")
        normalized.append(dict(validated_metadata))
    return normalized


def normalize_space_embedders(
    embedders: list[GoodMemSpaceEmbedder],
) -> list[GoodMemSpaceEmbedder]:
    """Validate and normalize create-time space-embedder declarations.

    Args:
        embedders: Candidate create-time GoodMem space-embedder declarations.

    Returns:
        A validated list of ``GoodMemSpaceEmbedder`` values.

    Raises:
        GoodMemConfigurationError: If ``embedders`` is empty or contains
            values other than ``GoodMemSpaceEmbedder`` instances.
    """
    if not embedders:
        raise GoodMemConfigurationError(
            "GoodMemVectorStore.create requires embedders to be a non-empty list "
            "of GoodMemSpaceEmbedder values."
        )

    try:
        validated_embedders = _STRICT_EMBEDDER_LIST_ADAPTER.validate_python(
            embedders,
            strict=True,
        )
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first_location = errors[0].get("loc", ())
            if first_location and isinstance(first_location[0], int):
                raise GoodMemConfigurationError(
                    "GoodMemVectorStore.create requires embedders to contain only "
                    "GoodMemSpaceEmbedder values. "
                    f"Invalid value at index {first_location[0]}."
                ) from exc
        raise GoodMemConfigurationError(
            "GoodMemVectorStore.create requires embedders to be a non-empty list "
            "of GoodMemSpaceEmbedder values."
        ) from exc

    return list(validated_embedders)


def normalize_optional_ids(
    ids: list[str | None] | None,
    *,
    source: str,
    exception_type: type[Exception],
) -> list[str | None] | None:
    """Normalize one aligned list of optional strict-create memory IDs.

    Args:
        ids: Optional aligned memory ID list.
        source: Logical source name used in validation errors.
        exception_type: Exception class raised when validation fails.

    Returns:
        ``None`` when ``ids`` is ``None``; otherwise a normalized list whose
        entries are either ``None`` or non-empty strings.

    Raises:
        Exception: An instance of ``exception_type`` when any value is neither
            ``None`` nor a non-empty string.
    """
    if ids is None:
        return None

    return [
        _normalize_optional_id(
            value,
            source=source,
            index=index,
            exception_type=exception_type,
        )
        for index, value in enumerate(ids)
    ]


def validate_duplicate_ids(ids: list[str | None] | None) -> None:
    """Reject repeated non-``None`` memory IDs in one local write call.

    Args:
        ids: Optional aligned memory ID list to inspect.

    Returns:
        ``None`` when ``ids`` is ``None`` or when all non-``None`` IDs are
        unique.

    Raises:
        GoodMemDuplicateIDError: If the same non-``None`` memory ID appears
            more than once.
    """
    if ids is None:
        return

    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in ids:
        if value is None:
            continue
        if value in seen:
            duplicates.add(value)
            continue
        seen.add(value)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise GoodMemDuplicateIDError(
            f"Duplicate memory IDs were provided for strict-create writes: {duplicate_list}."
        )


def validate_similarity_search_inputs(
    *,
    query: str,
    k: int,
    filter_expression: str | None,
) -> str:
    """Validate the public similarity-search input shape.

    Args:
        query: Candidate semantic retrieval query text.
        k: Requested result count.
        filter_expression: Optional raw GoodMem filter expression string.

    Returns:
        The validated query string.

    Raises:
        ValueError: If ``query`` is blank, ``k`` is not a positive integer, or
            ``filter_expression`` is neither ``None`` nor a string.
    """
    try:
        validated_query = _STRICT_STRING_ADAPTER.validate_python(query, strict=True)
    except ValidationError as exc:
        raise ValueError("query must be a non-empty string.")
    if not validated_query.strip():
        raise ValueError("query must be a non-empty string.")

    try:
        validated_k = _STRICT_INTEGER_ADAPTER.validate_python(k, strict=True)
    except ValidationError as exc:
        raise ValueError("k must be an integer.") from exc
    if validated_k <= 0:
        raise ValueError("k must be greater than 0.")

    try:
        _STRICT_OPTIONAL_STRING_ADAPTER.validate_python(
            filter_expression,
            strict=True,
        )
    except ValidationError as exc:
        raise ValueError(
            "filter must be a raw GoodMem filter expression string or None."
        ) from exc

    return validated_query


def raise_for_unexpected_kwargs(operation: str, kwargs: dict[str, Any]) -> None:
    """Reject service-specific keyword arguments that the package does not support.

    Args:
        operation: Operation name used in the error message.
        kwargs: Keyword arguments provided by the caller.

    Returns:
        ``None`` when ``kwargs`` is empty.

    Raises:
        ValueError: If any unsupported keyword arguments are present.
    """
    if not kwargs:
        return

    argument_list = ", ".join(sorted(kwargs))
    if len(kwargs) == 1:
        raise ValueError(
            f"{operation} got an unexpected keyword argument: {argument_list}."
        )
    raise ValueError(
        f"{operation} got unexpected keyword arguments: {argument_list}."
    )


def _normalize_optional_id(
    value: str | None,
    *,
    source: str,
    index: int,
    exception_type: type[Exception],
) -> str | None:
    try:
        validated = _STRICT_OPTIONAL_STRING_ADAPTER.validate_python(value, strict=True)
    except ValidationError as exc:
        raise exception_type(f"{source} at index {index} must be a string or None.")
    if validated is None:
        return None
    if not validated.strip():
        raise exception_type(
            f"{source} at index {index} must be None or a non-empty string."
        )
    return validated
