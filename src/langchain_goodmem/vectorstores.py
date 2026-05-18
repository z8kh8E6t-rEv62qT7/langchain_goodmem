"""LangChain vector-store integration backed by GoodMem semantic retrieval.

This module is the main public workflow surface for the repository.

Two primary usage modes are supported:

- bind to an existing GoodMem space with ``GoodMemVectorStore(...)``
- create a new GoodMem space with ``GoodMemVectorStore.create(...)``

GoodMem uses three related concepts that map differently to LangChain:

- a ``space`` is the search scope
- a ``memory`` is one stored item of source content plus metadata
- a ``chunk`` is a searchable fragment produced by GoodMem from a memory

Write methods in this module create memories. Retrieval methods return chunks.
One memory can therefore produce multiple search results over time.

Write behavior is intentionally explicit:

- caller-provided IDs are treated as memory IDs with strict create-if-absent
  semantics
- ``Document.id`` is used as a fallback memory ID source in
  ``add_documents(...)``
- duplicate IDs in a single call raise ``GoodMemDuplicateIDError`` before any
  backend call

Search behavior is chunk-level:

- ``Document.id`` in search results is the GoodMem ``chunk_id``
- the parent memory ID is preserved in ``Document.metadata`` as
  ``_goodmem_memory_id``
- the originating space ID is preserved in ``Document.metadata`` as
  ``_goodmem_space_id``
- scored results also expose ``_goodmem_score`` in metadata

Metadata filters are passed through to GoodMem unchanged. They operate on the
memory-level metadata JSON attached to each stored memory, not on LangChain
``Document.metadata`` after conversion and not on chunk text. The documented
GoodMem filter language uses helpers such as ``val('$.field')`` and
``exists('$.array[*]')``. A minimal equality example is
``val('$.lang') = 'en'``.

GoodMem processes new memories asynchronously before they become searchable, so
fresh writes may not appear in retrieval results immediately.

The create helper intentionally keeps a minimal LangChain-facing surface. If
you need GoodMem resource helpers around the normal RAG/search path, use
``GoodMemResources``. Broader platform administration remains outside this
LangChain integration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ._internal.memory_ops import (
    add_memories,
    create_space,
    delete_memories,
    search_memories,
)
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

    Each instance is bound to exactly one GoodMem space. ``add_documents(...)``
    and ``add_texts(...)`` write memories into that space. Search methods return
    chunk-level LangChain ``Document`` values derived from GoodMem retrieval
    hits, which means the returned ``Document.id`` identifies the matching
    chunk, not the original written memory.

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
        """Return the retained embeddings object when one exists.

        Returns:
            The ``GoodMemEmbeddings`` instance retained by
            ``GoodMemVectorStore.create(..., embedding=...)``, or ``None`` for
            existing-space stores, stores created with explicit
            ``embedders=[GoodMemSpaceEmbedder(...)]``, and
            ``from_texts(...)`` compatibility stores.
        """
        return self._embedding

    @classmethod
    def create(
        cls,
        *,
        name: str,
        embedders: list[GoodMemSpaceEmbedder] | None = None,
        connection: GoodMemConnection,
        embedding: GoodMemEmbeddings | None = None,
    ) -> GoodMemVectorStore:
        """Create a new GoodMem space and return a bound vector store.

        Exactly one create-time embedder source is allowed:

        - ``embedders=[GoodMemSpaceEmbedder(...)]``
        - ``embedding=GoodMemEmbeddings(...)``

        Choose ``embedders=...`` when you already know the GoodMem embedder ID
        or IDs that should be attached to the new space and you only need
        server-side retrieval behavior.

        Choose ``embedding=...`` when you want the same GoodMem embedder to
        serve both roles:

        - attach its ``embedder_id`` to the new space
        - remain available locally as ``store.embeddings`` for LangChain code
          that also calls ``embed_query(...)`` or ``embed_documents(...)``

        In other words, ``embedding=GoodMemEmbeddings(...)`` is the
        single-embedder shorthand that also preserves a usable LangChain
        ``Embeddings`` object on the returned store.

        Args:
            name: Requested GoodMem space name.
            embedders: Explicit GoodMem space-embedder declarations. Use this
                when you want to declare one or more existing GoodMem embedder
                IDs directly.
            connection: Shared GoodMem transport configuration.
            embedding: GoodMem-managed embeddings adapter whose embedder ID
                should be attached to the new space and then retained on the
                returned store as ``store.embeddings``.

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

        Each input document becomes one GoodMem memory. GoodMem may later split
        that memory into one or more searchable chunks during asynchronous
        processing.

        Args:
            documents: Documents whose ``page_content`` becomes GoodMem memory
                content.
            ids: Optional explicit memory IDs. When omitted, the method falls
                back to ``Document.id`` values.
            **kwargs: No service-specific keyword arguments are supported.

        Returns:
            Created or accepted memory IDs in request order. These are memory
            IDs, not chunk IDs. Empty input returns an empty list without
            contacting GoodMem.

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

        Each input text becomes one GoodMem memory. GoodMem may later split that
        memory into one or more searchable chunks during asynchronous
        processing.

        Args:
            texts: Iterable of non-empty strings. Passing one plain string is
                rejected because it is ambiguous with an iterable of characters.
            metadatas: Optional metadata mappings aligned to ``texts``.
            ids: Optional explicit memory IDs aligned to ``texts``.
            **kwargs: No service-specific keyword arguments are supported.

        Returns:
            Created or accepted memory IDs in request order. These are memory
            IDs, not chunk IDs. Empty input returns an empty list without
            contacting GoodMem.

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
            raise ValueError(
                "texts must be an iterable of strings, not a single string."
            )

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
            filter: Optional GoodMem memory-metadata filter expression string.
                The expression is passed to GoodMem unchanged and is evaluated
                before chunk results are returned. Use the documented GoodMem
                filter language, for example ``val('$.lang') = 'en'`` or
                ``exists('$.tags[*]')``.
            **kwargs: No additional search keyword arguments are supported.

        Returns:
            Chunk-level ``Document`` values. ``Document.id`` is the GoodMem
            chunk ID. ``Document.metadata`` includes ``_goodmem_chunk_id``,
            ``_goodmem_memory_id``, and ``_goodmem_space_id`` so callers can
            trace the hit back to the parent memory and space.

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
            filter: Optional GoodMem memory-metadata filter expression string.
                The expression is passed to GoodMem unchanged and is evaluated
                before chunk results are returned. Use the documented GoodMem
                filter language, for example ``val('$.lang') = 'en'`` or
                ``exists('$.tags[*]')``.
            **kwargs: No additional search keyword arguments are supported.

        Returns:
            Tuples of ``(Document, score)`` where the returned document metadata
            also includes ``_goodmem_score``. ``Document.id`` is still the
            GoodMem chunk ID rather than the parent memory ID.

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
            (_document_from_hit(hit, include_score=True), hit.score) for hit in hits
        ]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """Delete GoodMem memories by memory ID.

        Args:
            ids: GoodMem memory IDs to delete. ``None`` is rejected because it
                would imply deleting an entire space, which this LangChain
                adapter does not expose implicitly.
            **kwargs: No service-specific keyword arguments are supported.

        Returns:
            ``True`` when the delete request succeeds. Empty ID lists return
            ``True`` without contacting GoodMem.

        Raises:
            ValueError: If ``ids`` is ``None``, contains blank values, contains
                invalid values, or if unexpected keyword arguments are passed.
            GoodMemDuplicateIDError: If duplicate memory IDs are supplied.
            GoodMemAPIError: If GoodMem rejects the delete request or returns a
                malformed batch-delete response.
        """
        raise_for_unexpected_kwargs("GoodMemVectorStore.delete", kwargs)
        if ids is None:
            raise ValueError(
                "GoodMemVectorStore.delete requires explicit memory IDs; "
                "ids=None is not supported."
            )
        if isinstance(ids, str):
            raise ValueError("ids must be a list of memory ID strings, not a string.")

        normalized_ids = normalize_optional_ids(
            list(ids),
            source="ids",
            exception_type=ValueError,
        )
        if not normalized_ids:
            return True
        if any(memory_id is None for memory_id in normalized_ids):
            raise ValueError("ids must contain only non-empty memory ID strings.")
        resolved_ids = [
            memory_id.strip() for memory_id in normalized_ids if memory_id is not None
        ]
        validate_duplicate_ids(resolved_ids)
        delete_memories(self._transport, memory_ids=resolved_ids)
        return True

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
    ) -> GoodMemVectorStore:
        """Write texts into an existing GoodMem space and return a bound store.

        This compatibility helper matches the LangChain ``VectorStore``
        convention, but it does not create a new GoodMem space. The provided
        ``embedding`` argument is accepted only because LangChain expects it in
        this constructor shape; this package neither uses it for the write nor
        retains it on the returned store.

        Args:
            texts: Texts to write into an existing GoodMem space.
            embedding: Accepted for LangChain compatibility only. GoodMem
                retrieval for this helper still runs on the GoodMem side, so the
                argument is ignored for both the write and the returned store.
            metadatas: Optional metadata mappings aligned to ``texts``.
            connection: Shared GoodMem transport configuration.
            space_id: Existing GoodMem space identifier.
            ids: Optional explicit memory IDs aligned to ``texts``.
            **kwargs: No service-specific keyword arguments are supported.

        Returns:
            A vector store bound to the existing GoodMem space after the write.

        Raises:
            ValueError: If the write inputs or keyword arguments fail
                validation.
            GoodMemConfigurationError: If ``space_id`` is blank.
            GoodMemDuplicateIDError: If duplicate strict-create IDs are
                supplied or the backend reports duplicate IDs.
            GoodMemBatchPartialFailureError: If a batch partially succeeds.
            GoodMemAPIError: If GoodMem rejects the write.
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
    ) -> GoodMemVectorStore:
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
            "GoodMemVectorStore.create(embedding=...) requires a "
            "GoodMemEmbeddings instance."
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
