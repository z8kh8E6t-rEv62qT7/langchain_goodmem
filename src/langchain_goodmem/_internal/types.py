"""Private normalized request and response models.

The dataclasses and protocols in this module are package-owned internal
contracts. They keep validation, behavior, and transport concerns separated
without exposing raw GoodMem SDK model classes to upper layers.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..space_embedders import GoodMemSpaceEmbedder


@dataclass(frozen=True)
class GoodMemWriteRequest:
    """Normalized write payload used by the vector store.

    Attributes:
        page_content: Validated text content that will be stored as the memory
            body.
        metadata: Normalized metadata dictionary attached to the memory.
        memory_id: Optional strict-create memory ID supplied by the caller.
        content_type: Optional transport-level content type override.
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_id: str | None = None
    content_type: str | None = None


@dataclass(frozen=True)
class GoodMemSearchHit:
    """Normalized semantic search hit returned to the vector store.

    Attributes:
        chunk_id: GoodMem chunk identifier that becomes ``Document.id``.
        memory_id: Parent memory identifier preserved in metadata.
        space_id: Originating GoodMem space identifier.
        page_content: Retrieved chunk text.
        metadata: Merged memory-level and chunk-level metadata.
        score: Retrieval relevance score.
    """

    chunk_id: str
    memory_id: str
    space_id: str
    page_content: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class GoodMemSpaceCreateRequest:
    """Normalized create-space request passed through the transport boundary.

    Attributes:
        name: Requested GoodMem space name.
        space_embedders: Package-owned embedder declarations to attach to the
            created space.
    """

    name: str
    space_embedders: list[GoodMemSpaceEmbedder]


@dataclass(frozen=True)
class GoodMemEmbedderConfig:
    """Normalized embedder metadata used by ``GoodMemEmbeddings``.

    Attributes:
        embedder_id: GoodMem embedder identifier.
        provider_type: Normalized provider type such as ``OPENAI``.
        endpoint_url: Upstream provider endpoint URL.
        api_path: Optional provider-specific embeddings path segment.
        model_identifier: Upstream embedding model identifier.
        dimensionality: Configured embedding dimensionality.
        supported_modalities: Normalized modality list exposed by GoodMem.
        credential_kind: Credential mechanism reported by GoodMem, when any.
        inline_api_key: Readable inline API key exposed by GoodMem, when any.
    """

    embedder_id: str
    provider_type: str
    endpoint_url: str
    api_path: str | None
    model_identifier: str
    dimensionality: int
    supported_modalities: tuple[str, ...]
    credential_kind: str | None = None
    inline_api_key: str | None = None


class SupportsMemoryOperationsTransport(Protocol):
    """Narrow transport surface used by vector-store memory operations.

    This protocol keeps the vector-store and memory-operation layers decoupled
    from the full GoodMem client surface.
    """

    def create_space(self, request: GoodMemSpaceCreateRequest) -> object:
        """Create one GoodMem space from a normalized request payload."""
        ...

    def batch_create_memories(
        self,
        *,
        space_id: str,
        writes: list[GoodMemWriteRequest],
    ) -> object:
        """Create one batch of GoodMem memories in the target space."""
        ...

    def retrieve_memories(
        self,
        *,
        space_id: str,
        query: str,
        k: int,
        filter_expression: str | None = None,
    ) -> AbstractContextManager[Iterable[Any]]:
        """Return a context-managed stream of GoodMem retrieval events."""
        ...


class SupportsEmbedderTransport(Protocol):
    """Narrow transport surface used by embeddings/provider operations.

    This protocol isolates embedder lookup from memory-operation behavior so
    tests can stub provider setup independently.
    """

    def get_embedder(self, *, embedder_id: str) -> object:
        """Load one GoodMem embedder response by ID."""
        ...
