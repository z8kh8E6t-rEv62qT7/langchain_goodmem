"""GoodMem resource helpers for normal LangChain RAG workflows.

``GoodMemResources`` is the public GoodMem-specific companion to the
LangChain-facing ``GoodMemVectorStore`` and ``GoodMemEmbeddings`` classes. It
covers the resources needed to get from a clean GoodMem instance to a first
query without asking users to import the GoodMem SDK directly:

- embedders used by LangChain embedding and retrieval flows
- spaces that scope writes and searches
- memories written and deleted through LangChain workflows

The facade intentionally stops at the RAG/search boundary. It does not expose
broader GoodMem platform administration such as API key lifecycle management,
server initialization, migrations, system operations, or LLM/reranker/OCR
administration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._internal.memory_ops import delete_memories
from ._internal.providers import default_bootstrap_display_name
from ._internal.transport import GoodMemTransport
from ._internal.types import (
    GoodMemEmbedderBootstrapRequest,
    GoodMemMemoryCreateRequest,
    GoodMemSpaceCreateRequest,
    SupportsResourceOperationsTransport,
)
from ._internal.validators import (
    normalize_metadatas,
    normalize_optional_ids,
    normalize_space_embedders,
    require_non_empty_trimmed_string,
    validate_duplicate_ids,
    validate_text_inputs,
)
from .connection import GoodMemConnection
from .embeddings import GoodMemEmbeddings
from .errors import GoodMemConfigurationError
from .space_embedders import GoodMemSpaceEmbedder
from .vectorstores import GoodMemVectorStore


@dataclass(frozen=True)
class GoodMemEmbedderInfo:
    """Stable public summary of a GoodMem embedder.

    Args:
        embedder_id: GoodMem embedder identifier, or ``None`` if missing from
            an unexpected backend response.
        display_name: User-facing embedder name.
        provider_type: Provider type such as ``OPENAI``.
        endpoint_url: Upstream embeddings endpoint URL.
        model_identifier: Upstream embeddings model identifier.
        dimensionality: Embedding vector dimensionality.
        supported_modalities: Supported content modalities.
    """

    embedder_id: str | None
    display_name: str | None
    provider_type: str | None
    endpoint_url: str | None
    model_identifier: str | None
    dimensionality: int | None
    supported_modalities: tuple[str, ...]


@dataclass(frozen=True)
class GoodMemSpaceInfo:
    """Stable public summary of a GoodMem space.

    Args:
        space_id: GoodMem space identifier, or ``None`` if missing from an
            unexpected backend response.
        name: User-facing space name.
        labels: Space labels normalized to a plain dictionary.
        embedder_ids: GoodMem embedder IDs attached to the space.
    """

    space_id: str | None
    name: str | None
    labels: dict[str, str]
    embedder_ids: tuple[str, ...]


@dataclass(frozen=True)
class GoodMemMemoryInfo:
    """Stable public summary of a GoodMem memory.

    Args:
        memory_id: GoodMem memory identifier, or ``None`` if missing from an
            unexpected backend response.
        space_id: Parent GoodMem space identifier.
        metadata: Memory metadata normalized to a plain dictionary.
        content: Original content when the backend response includes it.
        processing_status: GoodMem processing status when available.
    """

    memory_id: str | None
    space_id: str | None
    metadata: dict[str, Any]
    content: str | bytes | None
    processing_status: str | None


def _create_transport(connection: GoodMemConnection) -> GoodMemTransport:
    return GoodMemTransport(connection)


class GoodMemResources:
    """Facade for GoodMem resources used by LangChain RAG workflows.

    Args:
        connection: Shared GoodMem transport configuration.
    """

    def __init__(self, connection: GoodMemConnection) -> None:
        self._connection = connection
        self._transport = _create_transport(connection)

    @classmethod
    def from_env(cls, *, verify: bool | str = True) -> GoodMemResources:
        """Build a resources facade from ``GOODMEM_API_KEY`` and ``GOODMEM_BASE_URL``.

        Args:
            verify: TLS verification setting forwarded to
                ``GoodMemConnection.from_env``.

        Returns:
            A resources facade using environment-provided GoodMem credentials.
        """
        return cls(GoodMemConnection.from_env(verify=verify))

    @classmethod
    def _from_transport(
        cls,
        *,
        connection: GoodMemConnection,
        transport: SupportsResourceOperationsTransport,
    ) -> GoodMemResources:
        resources = cls.__new__(cls)
        resources._connection = connection
        resources._transport = transport
        return resources

    def create_embedder(
        self,
        *,
        endpoint_url: str,
        model_identifier: str,
        dimensionality: int,
        upstream_api_key: str | None = None,
        display_name: str | None = None,
    ) -> GoodMemEmbedderInfo:
        """Create one ``OPENAI``-compatible GoodMem embedder.

        Args:
            endpoint_url: Upstream provider endpoint URL.
            model_identifier: Upstream embedding model identifier.
            dimensionality: Required embedding dimensionality.
            upstream_api_key: Optional upstream API key stored on the embedder.
            display_name: Optional user-facing name. A stable package default
                is used when omitted.

        Returns:
            A package-owned embedder summary.

        Raises:
            ValueError: If local inputs are malformed.
            GoodMemDuplicateIDError: If GoodMem reports that the embedder
                already exists.
            GoodMemAPIError: If GoodMem rejects the create request.
        """
        request = GoodMemEmbedderBootstrapRequest(
            display_name=_normalize_optional_text(display_name)
            or default_bootstrap_display_name(),
            endpoint_url=_require_text(endpoint_url, "endpoint_url"),
            model_identifier=_require_text(model_identifier, "model_identifier"),
            dimensionality=_require_positive_int(dimensionality, "dimensionality"),
            api_key=_normalize_optional_text(upstream_api_key),
        )
        return _embedder_info_from_raw(self._transport.create_embedder(request))

    def get_embedder(self, embedder_id: str) -> GoodMemEmbedderInfo:
        """Return one GoodMem embedder by ID.

        Args:
            embedder_id: GoodMem embedder identifier to load.

        Returns:
            A package-owned embedder summary.

        Raises:
            ValueError: If ``embedder_id`` is blank or not a string.
            GoodMemAPIError: If GoodMem rejects the lookup.
        """
        return _embedder_info_from_raw(
            self._transport.get_embedder(
                embedder_id=_require_text(embedder_id, "embedder_id"),
            )
        )

    def list_embedders(
        self,
        *,
        label: Mapping[str, str] | None = None,
        owner_id: str | None = None,
        provider_type: str | None = None,
    ) -> list[GoodMemEmbedderInfo]:
        """List embedders visible to the current GoodMem credentials.

        Args:
            label: Optional GoodMem label filter.
            owner_id: Optional GoodMem owner ID filter.
            provider_type: Optional GoodMem provider type filter such as
                ``OPENAI``.

        Returns:
            Package-owned embedder summaries.

        Raises:
            ValueError: If local filter inputs have invalid shapes.
            GoodMemAPIError: If GoodMem rejects the listing.
        """
        raw_embedders = self._transport.list_embedders(
            label=_normalize_optional_labels(label),
            owner_id=_normalize_optional_text(owner_id),
            provider_type=_normalize_optional_text(provider_type),
        )
        return [
            _embedder_info_from_raw(embedder) for embedder in _iter_items(raw_embedders)
        ]

    def delete_embedder(self, embedder_id: str) -> None:
        """Delete one GoodMem embedder by ID.

        Args:
            embedder_id: GoodMem embedder identifier to delete.

        Raises:
            ValueError: If ``embedder_id`` is blank or not a string.
            GoodMemAPIError: If GoodMem rejects the delete.
        """
        self._transport.delete_embedder(
            embedder_id=_require_text(embedder_id, "embedder_id"),
        )

    def create_space(
        self,
        name: str,
        embedders: list[GoodMemSpaceEmbedder] | None = None,
        embedding: GoodMemEmbeddings | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> GoodMemSpaceInfo:
        """Create one GoodMem space for LangChain writes and retrieval.

        Exactly one embedder source is accepted: explicit ``embedders`` or a
        single ``GoodMemEmbeddings`` instance.

        Args:
            name: Requested GoodMem space name.
            embedders: Explicit GoodMem space-embedder declarations.
            embedding: Existing ``GoodMemEmbeddings`` instance whose embedder
                ID should be attached to the new space.
            labels: Optional GoodMem labels for the new space.

        Returns:
            A package-owned space summary.

        Raises:
            ValueError: If ``name`` or ``labels`` are invalid.
            GoodMemConfigurationError: If embedder inputs are missing, mixed, or
                incompatible.
            GoodMemAPIError: If GoodMem rejects the create request.
        """
        resolved_embedders = _resolve_space_embedders(
            embedders=embedders,
            embedding=embedding,
        )
        raw_space = self._transport.create_space(
            GoodMemSpaceCreateRequest(
                name=_require_text(name, "name"),
                space_embedders=resolved_embedders,
                labels=_normalize_optional_labels(labels),
            )
        )
        return _space_info_from_raw(raw_space)

    def get_space(self, space_id: str) -> GoodMemSpaceInfo:
        """Return one GoodMem space by ID.

        Args:
            space_id: GoodMem space identifier to load.

        Returns:
            A package-owned space summary.

        Raises:
            ValueError: If ``space_id`` is blank or not a string.
            GoodMemAPIError: If GoodMem rejects the lookup.
        """
        return _space_info_from_raw(
            self._transport.get_space(space_id=_require_text(space_id, "space_id"))
        )

    def list_spaces(
        self,
        *,
        label: Mapping[str, str] | None = None,
        name_filter: str | None = None,
        max_items: int | None = None,
    ) -> list[GoodMemSpaceInfo]:
        """List spaces visible to the current GoodMem credentials.

        Args:
            label: Optional GoodMem label filter.
            name_filter: Optional GoodMem name filter.
            max_items: Optional positive result limit.

        Returns:
            Package-owned space summaries. When GoodMem returns a paginated SDK
            list object, this method follows subsequent pages automatically
            until all visible results are collected or ``max_items`` is
            reached.

        Raises:
            ValueError: If local filter inputs have invalid shapes.
            GoodMemAPIError: If GoodMem rejects the listing.
        """
        raw_spaces = self._transport.list_spaces(
            label=_normalize_optional_labels(label),
            name_filter=_normalize_optional_text(name_filter),
            max_items=_normalize_optional_positive_int(max_items, "max_items"),
        )
        return [
            _space_info_from_raw(space)
            for space in _iter_resource_list_items(raw_spaces)
        ]

    def delete_space(self, space_id: str) -> None:
        """Delete one GoodMem space by ID.

        Args:
            space_id: GoodMem space identifier to delete.

        Raises:
            ValueError: If ``space_id`` is blank or not a string.
            GoodMemAPIError: If GoodMem rejects the delete.
        """
        self._transport.delete_space(space_id=_require_text(space_id, "space_id"))

    def create_memory(
        self,
        space_id: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> GoodMemMemoryInfo:
        """Create one GoodMem memory in a space.

        Args:
            space_id: Target GoodMem space identifier.
            content: Non-empty memory content. The original text is preserved
                after validation.
            metadata: Optional memory metadata mapping.
            memory_id: Optional caller-supplied GoodMem memory ID.

        Returns:
            A package-owned memory summary.

        Raises:
            ValueError: If inputs are malformed or blank where required.
            GoodMemDuplicateIDError: If GoodMem reports that ``memory_id``
                already exists.
            GoodMemAPIError: If GoodMem rejects the create request.
        """
        metadata_list = normalize_metadatas([metadata])
        assert metadata_list is not None
        raw_memory = self._transport.create_memory(
            GoodMemMemoryCreateRequest(
                space_id=_require_text(space_id, "space_id"),
                content=validate_text_inputs(
                    [content],
                    label="content",
                    exception_type=ValueError,
                )[0],
                metadata=metadata_list[0],
                memory_id=_normalize_optional_id(memory_id, "memory_id"),
            )
        )
        return _memory_info_from_raw(raw_memory)

    def get_memory(
        self,
        memory_id: str,
        *,
        include_content: bool = False,
    ) -> GoodMemMemoryInfo:
        """Return one GoodMem memory by ID.

        Args:
            memory_id: GoodMem memory identifier to load.
            include_content: Whether to request original memory content from
                GoodMem when the backend supports it.

        Returns:
            A package-owned memory summary.

        Raises:
            ValueError: If ``memory_id`` or ``include_content`` is invalid.
            GoodMemAPIError: If GoodMem rejects the lookup.
        """
        if not isinstance(include_content, bool):
            raise ValueError("include_content must be a boolean.")
        return _memory_info_from_raw(
            self._transport.get_memory(
                memory_id=_require_text(memory_id, "memory_id"),
                include_content=include_content,
            )
        )

    def list_memories(
        self,
        space_id: str,
        filter: str | None = None,
        max_items: int | None = None,
    ) -> list[GoodMemMemoryInfo]:
        """List memories in one GoodMem space.

        Args:
            space_id: GoodMem space identifier to list memories from.
            filter: Optional raw GoodMem memory filter expression.
            max_items: Optional positive result limit.

        Returns:
            Package-owned memory summaries. When GoodMem returns a paginated SDK
            list object, this method follows subsequent pages automatically
            until all visible results are collected or ``max_items`` is
            reached.

        Raises:
            ValueError: If local filter inputs have invalid shapes.
            GoodMemAPIError: If GoodMem rejects the listing.
        """
        raw_memories = self._transport.list_memories(
            space_id=_require_text(space_id, "space_id"),
            filter_expression=_normalize_optional_text(filter),
            max_items=_normalize_optional_positive_int(max_items, "max_items"),
        )
        return [
            _memory_info_from_raw(memory)
            for memory in _iter_resource_list_items(raw_memories)
        ]

    def delete_memory(self, memory_id: str) -> None:
        """Delete one GoodMem memory by ID.

        Args:
            memory_id: GoodMem memory identifier to delete.

        Raises:
            ValueError: If ``memory_id`` is blank or not a string.
            GoodMemAPIError: If GoodMem rejects the delete.
        """
        self._transport.delete_memory(memory_id=_require_text(memory_id, "memory_id"))

    def delete_memories(self, memory_ids: list[str]) -> None:
        """Delete multiple GoodMem memories by ID.

        Args:
            memory_ids: Non-empty GoodMem memory IDs to delete.

        Raises:
            ValueError: If ``memory_ids`` is empty or contains invalid values.
            GoodMemDuplicateIDError: If duplicate memory IDs are supplied after
                trimming.
            GoodMemAPIError: If GoodMem rejects the delete request or returns a
                malformed batch-delete response.
        """
        delete_memories(
            self._transport,
            memory_ids=_normalize_required_id_list(memory_ids, "memory_ids"),
        )

    def bootstrap_vector_store(
        self,
        space_name: str,
        endpoint_url: str,
        model_identifier: str,
        dimensionality: int,
        upstream_api_key: str | None = None,
        embedder_display_name: str | None = None,
    ) -> GoodMemVectorStore:
        """Create or reuse an embedder, create a space, and return a vector store.

        Args:
            space_name: Requested GoodMem space name.
            endpoint_url: Upstream embeddings endpoint URL.
            model_identifier: Upstream embeddings model identifier.
            dimensionality: Required embedding dimensionality.
            upstream_api_key: Optional upstream API key used when a compatible
                embedder must be created or when the resolved embedder does not
                expose a readable inline credential.
            embedder_display_name: Optional display name used when creating a
                new compatible embedder.

        Returns:
            A ``GoodMemVectorStore`` bound to the newly created space with the
            resolved ``GoodMemEmbeddings`` instance retained on
            ``store.embeddings``.

        Raises:
            ValueError: If bootstrap inputs are malformed.
            GoodMemConfigurationError: If no compatible embedder can be used or
                the created space response is missing a usable ID.
            GoodMemAPIError: If GoodMem rejects an embedder or space request.
        """
        embeddings = GoodMemEmbeddings.ensure(
            connection=self._connection,
            endpoint_url=endpoint_url,
            model_identifier=model_identifier,
            dimensionality=dimensionality,
            upstream_api_key=upstream_api_key,
            display_name=embedder_display_name,
        )
        space = self.create_space(
            name=space_name,
            embedding=embeddings,
        )
        if space.space_id is None:
            raise GoodMemConfigurationError(
                "GoodMemResources.bootstrap_vector_store could not resolve "
                "the created space ID."
            )
        return GoodMemVectorStore._from_transport(
            space_id=space.space_id,
            embedding=embeddings,
            transport=self._transport,
        )


def _resolve_space_embedders(
    *,
    embedders: list[GoodMemSpaceEmbedder] | None,
    embedding: GoodMemEmbeddings | None,
) -> list[GoodMemSpaceEmbedder]:
    if embedders is not None and embedding is not None:
        raise GoodMemConfigurationError(
            "GoodMemResources.create_space accepts either embedders or "
            "embedding, not both."
        )
    if embedders is not None:
        return normalize_space_embedders(embedders)
    if embedding is None:
        raise GoodMemConfigurationError(
            "GoodMemResources.create_space requires either embedders or embedding."
        )
    if not isinstance(embedding, GoodMemEmbeddings):
        raise GoodMemConfigurationError(
            "GoodMemResources.create_space(embedding=...) requires a "
            "GoodMemEmbeddings instance."
        )
    return [GoodMemSpaceEmbedder(embedder_id=embedding.embedder_id)]


def _embedder_info_from_raw(embedder: Any) -> GoodMemEmbedderInfo:
    return GoodMemEmbedderInfo(
        embedder_id=_optional_text(getattr(embedder, "embedder_id", None)),
        display_name=_optional_text(getattr(embedder, "display_name", None)),
        provider_type=_optional_text(
            _enum_value(getattr(embedder, "provider_type", None))
        ),
        endpoint_url=_optional_text(getattr(embedder, "endpoint_url", None)),
        model_identifier=_optional_text(getattr(embedder, "model_identifier", None)),
        dimensionality=_optional_int(getattr(embedder, "dimensionality", None)),
        supported_modalities=tuple(
            value
            for value in (
                _optional_text(_enum_value(modality))
                for modality in getattr(embedder, "supported_modalities", ()) or ()
            )
            if value is not None
        ),
    )


def _space_info_from_raw(space: Any) -> GoodMemSpaceInfo:
    return GoodMemSpaceInfo(
        space_id=_optional_text(getattr(space, "space_id", None)),
        name=_optional_text(getattr(space, "name", None)),
        labels=_labels_from_raw(getattr(space, "labels", None)),
        embedder_ids=tuple(
            embedder_id
            for embedder_id in (
                _optional_text(getattr(embedder, "embedder_id", None))
                for embedder in getattr(space, "space_embedders", ()) or ()
            )
            if embedder_id is not None
        ),
    )


def _memory_info_from_raw(memory: Any) -> GoodMemMemoryInfo:
    return GoodMemMemoryInfo(
        memory_id=_optional_text(getattr(memory, "memory_id", None)),
        space_id=_optional_text(getattr(memory, "space_id", None)),
        metadata=dict(getattr(memory, "metadata", None) or {}),
        content=getattr(memory, "original_content", None),
        processing_status=_optional_text(getattr(memory, "processing_status", None)),
    )


def _iter_resource_list_items(raw_items: Any) -> list[Any]:
    if _is_auto_paging_sdk_list(raw_items):
        return list(raw_items)
    return _iter_items(raw_items)


def _iter_items(raw_items: Any) -> list[Any]:
    if raw_items is None:
        return []
    data = getattr(raw_items, "data", None)
    if data is not None:
        return list(data)
    if isinstance(raw_items, str | bytes | dict):
        return [raw_items]
    try:
        return list(raw_items)
    except TypeError:
        return [raw_items]


def _is_auto_paging_sdk_list(raw_items: Any) -> bool:
    if raw_items is None or not hasattr(raw_items, "__iter__"):
        return False
    return all(
        hasattr(raw_items, attr)
        for attr in (
            "data",
            "next_token",
            "_fetch_fn",
            "_response_model",
            "_items_field",
        )
    )


def _require_text(value: Any, field_name: str) -> str:
    return require_non_empty_trimmed_string(
        value,
        error_message=f"{field_name} must be a non-empty string.",
        exception_type=ValueError,
    )


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return _require_text(value, "value")


def _normalize_optional_id(value: Any, field_name: str) -> str | None:
    normalized = normalize_optional_ids(
        [value],
        source=field_name,
        exception_type=ValueError,
    )
    if normalized is None:
        return None
    if normalized[0] is None:
        return None
    return normalized[0].strip()


def _normalize_required_id_list(values: list[str], field_name: str) -> list[str]:
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a list of memory ID strings.")
    normalized = normalize_optional_ids(
        values,
        source=field_name,
        exception_type=ValueError,
    )
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one memory ID.")
    if any(value is None for value in normalized):
        raise ValueError(f"{field_name} must contain only non-empty memory IDs.")
    resolved = [value.strip() for value in normalized if value is not None]
    validate_duplicate_ids(resolved)
    return resolved


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _normalize_optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, field_name)


def _normalize_optional_labels(
    labels: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if labels is None:
        return None
    if not isinstance(labels, Mapping):
        raise ValueError("labels must be a mapping of strings to strings.")
    normalized: dict[str, str] = {}
    for key, value in labels.items():
        if not isinstance(key, str) or not key:
            raise ValueError("labels must contain only non-empty string keys.")
        if not isinstance(value, str):
            raise ValueError("labels must contain only string values.")
        normalized[key] = value
    return normalized


def _labels_from_raw(labels: Any) -> dict[str, str]:
    if not isinstance(labels, Mapping):
        return {}
    return {str(key): str(value) for key, value in labels.items()}


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
