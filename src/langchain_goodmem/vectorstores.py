"""LangChain vector-store integration backed by GoodMem semantic retrieval.

This module is the main public workflow surface for the repository.

Two primary usage modes are supported:

- bind to an existing GoodMem space with ``GoodMemVectorStore(...)``
- create a new GoodMem space with ``GoodMemVectorStore.create(...)``

Write behavior is intentionally explicit:

- caller-provided IDs use strict create-if-absent semantics
- ``Document.id`` is used as a fallback memory ID source in
  ``add_documents(...)``
- duplicate IDs in a single call raise ``GoodMemDuplicateIDError`` before any
  backend call

Search behavior is chunk-level:

- ``Document.id`` in search results is the GoodMem ``chunk_id``
- parent memory and space identifiers remain available in ``Document.metadata``
- scored results also expose ``_goodmem_score`` in metadata

The create helper intentionally keeps a minimal LangChain-facing surface. If
you need GoodMem-specific space options such as labels, owner settings, public
access controls, or explicit space IDs, create the space through the GoodMem
SDK first and then attach a ``GoodMemVectorStore`` to the resulting
``space_id``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ._internal.memory_ops import add_memories, create_space, search_memories
from ._internal.transport import GoodMemTransport
from ._internal.types import (
    GoodMemSearchHit,
    GoodMemSpaceCreateRequest,
    GoodMemWriteRequest,
    SupportsMemoryOperationsTransport,
)
from ._internal.validators import (
    normalize_metadatas,
    normalize_optional_ids,
    normalize_space_embedders,
    raise_for_unexpected_kwargs,
    require_space_id,
    validate_duplicate_ids,
    validate_lengths,
    validate_similarity_search_inputs,
    validate_text_inputs,
)
from .connection import GoodMemConnection
from .embeddings import GoodMemEmbeddings
from .errors import GoodMemConfigurationError
from .space_embedders import GoodMemSpaceEmbedder


def _create_transport(connection: GoodMemConnection) -> GoodMemTransport:
    return GoodMemTransport(connection)


class GoodMemVectorStore(VectorStore):
    """LangChain vector store for GoodMem-managed semantic add/search.

    Args:
        space_id: Existing GoodMem space identifier to bind to.
        connection: Shared GoodMem transport configuration.

    Attributes:
        space_id: Trimmed GoodMem space identifier.
        embeddings: Optional retained embeddings object. Existing-space stores
            expose ``None`` because retrieval happens on the GoodMem side.

    Raises:
        GoodMemConfigurationError: If ``space_id`` is blank.
    """

    def __init__(
        self,
        space_id: str,
        connection: GoodMemConnection,
    ) -> None:
        self._initialize(
            space_id=space_id,
            embedding=None,
            transport=_create_transport(connection),
        )

    @property
    def embeddings(self) -> Embeddings | None:
        """Return the retained embeddings object when one exists."""
        return self._embedding

    @classmethod
    def create(
        cls,
        *,
        name: str,
        embedders: list[GoodMemSpaceEmbedder] | None = None,
        connection: GoodMemConnection,
        embedding: GoodMemEmbeddings | None = None,
    ) -> "GoodMemVectorStore":
        """Create a new GoodMem space and return a bound vector store.

        Exactly one create-time embedder source is allowed:

        - ``embedders=[GoodMemSpaceEmbedder(...)]``
        - ``embedding=GoodMemEmbeddings(...)``

        Args:
            name: Requested GoodMem space name.
            embedders: Explicit GoodMem space-embedder declarations.
            connection: Shared GoodMem transport configuration.
            embedding: GoodMem-managed embeddings adapter whose embedder ID
                should be attached to the new space.

        Returns:
            A vector store bound to the created space.

        Raises:
            GoodMemConfigurationError: If embedder inputs are missing, mixed, or
                incompatible with the create helper.
            GoodMemAPIError: If GoodMem rejects the create request.
        """
        resolved_embedders, resolved_embedding = _resolve_create_embedding_inputs(
            embedders=embedders,
            embedding=embedding,
        )
        transport = _create_transport(connection)
        created_space_id = create_space(
            transport,
            GoodMemSpaceCreateRequest(
                name=name,
                space_embedders=resolved_embedders,
            ),
        )
        return cls._from_transport(
            space_id=created_space_id,
            embedding=resolved_embedding,
            transport=transport,
        )

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add LangChain documents to the bound GoodMem space.

        Args:
            documents: Documents whose ``page_content`` becomes GoodMem memory
                content.
            ids: Optional explicit memory IDs. When omitted, the method falls
                back to ``Document.id`` values.
            **kwargs: No service-specific keyword arguments are supported.

        Returns:
            Created or accepted memory IDs in request order. Empty input returns
            an empty list without contacting GoodMem.

        Raises:
            ValueError: If document text, IDs, or unexpected keyword arguments
                fail local validation.
            GoodMemDuplicateIDError: If duplicate strict-create IDs are
                supplied or the backend reports duplicate IDs.
            GoodMemBatchPartialFailureError: If a batch partially succeeds.
            GoodMemAPIError: If GoodMem rejects the write.
        """
        raise_for_unexpected_kwargs("GoodMemVectorStore.add_documents", kwargs)
        if not documents:
            return []

        validated_texts = validate_text_inputs(
            [document.page_content for document in documents],
            label="documents",
            field_name="page_content",
            exception_type=ValueError,
        )
        resolved_ids = _resolve_write_ids(
            label="documents",
            expected_length=len(documents),
            ids=ids if ids is not None else [document.id for document in documents],
            source="ids" if ids is not None else "Document.id",
        )

        writes = [
            GoodMemWriteRequest(
                page_content=validated_texts[index],
                metadata=dict(document.metadata),
                memory_id=resolved_ids[index] if resolved_ids is not None else None,
            )
            for index, document in enumerate(documents)
        ]
        return add_memories(self._transport, space_id=self.space_id, writes=writes)

    def add_texts(
        self,
        texts: list[str] | tuple[str, ...] | Any,
        metadatas: list[Mapping[str, Any] | None] | None = None,
        *,
        ids: list[str | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add raw texts to the bound GoodMem space.

        Args:
            texts: Iterable of non-empty strings. Passing one plain string is
                rejected because it is ambiguous with an iterable of characters.
            metadatas: Optional metadata mappings aligned to ``texts``.
            ids: Optional explicit memory IDs aligned to ``texts``.
            **kwargs: No service-specific keyword arguments are supported.

        Returns:
            Created or accepted memory IDs in request order. Empty input returns
            an empty list without contacting GoodMem.

        Raises:
            ValueError: If input shapes, text values, metadata values, IDs, or
                unexpected keyword arguments fail local validation.
            GoodMemDuplicateIDError: If duplicate strict-create IDs are
                supplied or the backend reports duplicate IDs.
            GoodMemBatchPartialFailureError: If a batch partially succeeds.
            GoodMemAPIError: If GoodMem rejects the write.
        """
        raise_for_unexpected_kwargs("GoodMemVectorStore.add_texts", kwargs)
        if isinstance(texts, str):
            raise ValueError("texts must be an iterable of strings, not a single string.")

        texts_list = list(texts)
        if not texts_list:
            return []

        validated_texts = validate_text_inputs(
            texts_list,
            label="texts",
            exception_type=ValueError,
        )
        validate_lengths("texts", len(texts_list), metadatas=metadatas, ids=ids)
        resolved_ids = _resolve_write_ids(
            label="texts",
            expected_length=len(texts_list),
            ids=ids,
            source="ids",
        )

        metadata_list = (
            normalize_metadatas(metadatas)
            if metadatas is not None
            else [{} for _ in validated_texts]
        )
        writes = [
            GoodMemWriteRequest(
                page_content=text,
                metadata=metadata_list[index],
                memory_id=resolved_ids[index] if resolved_ids is not None else None,
            )
            for index, text in enumerate(validated_texts)
        ]
        return add_memories(self._transport, space_id=self.space_id, writes=writes)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return semantic matches for ``query``.

        Args:
            query: Non-empty semantic query string.
            k: Maximum number of chunk-level ``Document`` values to return.
            filter: Optional raw GoodMem filter expression string.
            **kwargs: No additional search keyword arguments are supported.

        Returns:
            Chunk-level ``Document`` values whose metadata includes
            ``_goodmem_chunk_id``, ``_goodmem_memory_id``, and
            ``_goodmem_space_id``.

        Raises:
            ValueError: If the query shape, ``k``, filter, or keyword arguments
                fail local validation.
            GoodMemAPIError: If GoodMem rejects the retrieval request.
        """
        raise_for_unexpected_kwargs("GoodMemVectorStore.similarity_search", kwargs)
        hits = self._search_hits(
            query=query,
            k=k,
            filter_expression=filter,
        )
        return [_document_from_hit(hit, include_score=False) for hit in hits]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return semantic matches and scores for ``query``.

        Args:
            query: Non-empty semantic query string.
            k: Maximum number of scored chunk-level matches to return.
            filter: Optional raw GoodMem filter expression string.
            **kwargs: No additional search keyword arguments are supported.

        Returns:
            Tuples of ``(Document, score)`` where the returned document metadata
            also includes ``_goodmem_score``.

        Raises:
            ValueError: If the query shape, ``k``, filter, or keyword arguments
                fail local validation.
            GoodMemAPIError: If GoodMem rejects the retrieval request.
        """
        raise_for_unexpected_kwargs(
            "GoodMemVectorStore.similarity_search_with_score",
            kwargs,
        )
        hits = self._search_hits(
            query=query,
            k=k,
            filter_expression=filter,
        )
        return [
            (_document_from_hit(hit, include_score=True), hit.score)
            for hit in hits
        ]

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return self.similarity_search_with_score(query, k=k, **kwargs)

    def _select_relevance_score_fn(self) -> Any:
        return lambda score: score

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[Mapping[str, Any] | None] | None = None,
        *,
        connection: GoodMemConnection,
        space_id: str,
        ids: list[str | None] | None = None,
        **kwargs: Any,
    ) -> "GoodMemVectorStore":
        """Write texts into an existing GoodMem space and return a bound store.

        This compatibility helper matches the LangChain ``VectorStore``
        convention, but it does not create a new GoodMem space and it does not
        retain the provided ``embedding`` object on the returned store.

        Args:
            texts: Texts to write into an existing GoodMem space.
            embedding: Accepted for LangChain compatibility. The returned store
                does not retain it.
            metadatas: Optional metadata mappings aligned to ``texts``.
            connection: Shared GoodMem transport configuration.
            space_id: Existing GoodMem space identifier.
            ids: Optional explicit memory IDs aligned to ``texts``.
            **kwargs: No service-specific keyword arguments are supported.

        Returns:
            A vector store bound to the existing GoodMem space after the write.

        Raises:
            ValueError: If the write inputs or keyword arguments fail validation.
            GoodMemOperationError: If the backend rejects the write.
        """
        raise_for_unexpected_kwargs("GoodMemVectorStore.from_texts", kwargs)
        vectorstore = cls(
            space_id=space_id,
            connection=connection,
        )
        vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
        return vectorstore

    def _search_hits(
        self,
        *,
        query: str,
        k: int,
        filter_expression: str | None,
    ) -> list[GoodMemSearchHit]:
        validate_similarity_search_inputs(
            query=query,
            k=k,
            filter_expression=filter_expression,
        )
        return search_memories(
            self._transport,
            space_id=self.space_id,
            query=query,
            k=k,
            filter_expression=filter_expression,
        )

    @classmethod
    def _from_transport(
        cls,
        *,
        space_id: str,
        embedding: Embeddings | None,
        transport: SupportsMemoryOperationsTransport,
    ) -> "GoodMemVectorStore":
        store = cls.__new__(cls)
        store._initialize(
            space_id=space_id,
            embedding=embedding,
            transport=transport,
        )
        return store

    def _initialize(
        self,
        *,
        space_id: str,
        embedding: Embeddings | None,
        transport: SupportsMemoryOperationsTransport,
    ) -> None:
        self.space_id = require_space_id(space_id)
        self._embedding = embedding
        self._transport = transport


def _document_from_hit(hit: GoodMemSearchHit, *, include_score: bool) -> Document:
    metadata = dict(hit.metadata)
    metadata["_goodmem_chunk_id"] = hit.chunk_id
    metadata["_goodmem_memory_id"] = hit.memory_id
    metadata["_goodmem_space_id"] = hit.space_id
    if include_score:
        metadata["_goodmem_score"] = hit.score

    return Document(
        id=hit.chunk_id,
        page_content=hit.page_content,
        metadata=metadata,
    )


def _resolve_create_embedding_inputs(
    *,
    embedders: list[GoodMemSpaceEmbedder] | None,
    embedding: GoodMemEmbeddings | None,
) -> tuple[list[GoodMemSpaceEmbedder], GoodMemEmbeddings | None]:
    if embedders is not None and embedding is not None:
        raise GoodMemConfigurationError(
            "GoodMemVectorStore.create accepts either embedders or embedding, not both."
        )
    if embedders is not None:
        return normalize_space_embedders(embedders), None
    if embedding is None:
        raise GoodMemConfigurationError(
            "GoodMemVectorStore.create requires either embedders or embedding."
        )
    if not isinstance(embedding, GoodMemEmbeddings):
        raise GoodMemConfigurationError(
            "GoodMemVectorStore.create(embedding=...) requires a GoodMemEmbeddings instance."
        )
    return [GoodMemSpaceEmbedder(embedder_id=embedding.embedder_id)], embedding


def _resolve_write_ids(
    *,
    label: str,
    expected_length: int,
    ids: list[str | None] | None,
    source: str,
) -> list[str | None] | None:
    validate_lengths(label, expected_length, ids=ids)
    normalized_ids = normalize_optional_ids(
        ids,
        source=source,
        exception_type=ValueError,
    )
    validate_duplicate_ids(normalized_ids)
    return normalized_ids
