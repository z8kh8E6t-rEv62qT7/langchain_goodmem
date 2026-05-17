"""langchain-goodmem package entry point and quick-start guide.

This package exposes a small, explicit LangChain-facing surface for GoodMem:

- ``GoodMemConnection`` for shared API/TLS configuration
- ``GoodMemResources`` for GoodMem resources needed by normal RAG workflows
- ``GoodMemVectorStore`` for writing memories and retrieving semantic matches
- ``GoodMemEmbeddings`` for GoodMem-managed ``OPENAI``-compatible embedders
- ``GoodMemSpaceEmbedder`` for create-time space/embedder declarations

GoodMem concepts used throughout this package:

- a ``space`` is the top-level container you write into and search within
- a ``memory`` is one stored item of source content plus metadata
- GoodMem processes each memory asynchronously and may split it into one or
  more retrievable ``chunk`` values

That distinction matters when you read search results:

- write methods such as ``add_documents(...)`` and ``add_texts(...)`` return
  memory IDs
- search methods return LangChain ``Document`` values whose ``Document.id`` is
  the GoodMem chunk ID for the matching chunk
- the parent memory ID and originating space ID remain available in
  ``Document.metadata`` under ``_goodmem_memory_id`` and ``_goodmem_space_id``

Python 3.10 or newer is required.

Install the core package:

::

    pip install -e .

Install optional extras only when needed:

- ``pip install -e '.[openai]'`` for ``GoodMemEmbeddings``
- ``pip install -e '.[test]'`` for the unit and live test suites
- ``pip install -e '.[docs]'`` for the Sphinx documentation build

Provide GoodMem credentials either directly in code or through environment
variables consumed by ``GoodMemConnection.from_env()``:

- ``GOODMEM_API_KEY``
- ``GOODMEM_BASE_URL``

Choose a store-binding flow:

- if you already have a GoodMem space ID, bind directly with
  ``GoodMemVectorStore(space_id=..., ...)``
- if you need the package to create a new space for you, use
  ``GoodMemVectorStore.create(...)``
- if you need to manage the GoodMem resources around the LangChain workflow,
  use ``GoodMemResources``

Choose an embeddings flow:

- if you already have one compatible GoodMem embedder ID, construct
  ``GoodMemEmbeddings(embedder_id=..., ...)`` directly
- if you are starting from a clean slate and need the package to find or
  create one compatible ``OPENAI``-style embedder first, use
  ``GoodMemEmbeddings.ensure(...)`` or ``GoodMemEmbeddings.ensure_from_env()``
  and, when reusing ``GOODMEM_EMBEDDER_ID`` through the environment helper,
  keep the bootstrap environment present and aligned with that embedder

``GoodMemResources`` covers the GoodMem resources needed for normal
LangChain RAG/search workflows: embedders, spaces, memories, and search
bootstrap. It intentionally leaves broader platform administration to GoodMem's
SDK, CLI, and UI.

Minimal existing-space add-and-search flow:

::

    from langchain_core.documents import Document
    from langchain_goodmem import GoodMemConnection, GoodMemVectorStore

    connection = GoodMemConnection.from_env()
    store = GoodMemVectorStore(
        space_id="your-existing-space-id",
        connection=connection,
    )
    memory_ids = store.add_documents(
        [Document(page_content="GoodMem stores semantically searchable memories.")],
        ids=["memory-1"],
    )
    results = store.similarity_search(
        "How does GoodMem work with LangChain?",
        k=1,
    )
    print("created memory:", memory_ids[0])
    print("matching chunk:", results[0].id)
    print("parent memory:", results[0].metadata["_goodmem_memory_id"])

Search visibility is eventually consistent because GoodMem processes written
memories before they become searchable.

For fuller workflows:

- see ``examples/basic_semantic_search.py`` for existing-space usage
- see ``examples/goodmem_embeddings_bootstrap_workflow.py`` for clean-slate
  embeddings bootstrap usage
- see ``GoodMemVectorStore.create`` for create-helper usage
- see ``examples/goodmem_embeddings_workflow.py`` for embeddings-driven usage
"""

from __future__ import annotations

from .connection import GoodMemConnection
from .embeddings import GoodMemEmbeddings
from .errors import (
    GoodMemAPIError,
    GoodMemBatchPartialFailureError,
    GoodMemBatchWriteResultItem,
    GoodMemConfigurationError,
    GoodMemDuplicateIDError,
    GoodMemOperationError,
    LangChainGoodMemError,
)
from .resources import (
    GoodMemEmbedderInfo,
    GoodMemMemoryInfo,
    GoodMemResources,
    GoodMemSpaceInfo,
)
from .space_embedders import GoodMemSpaceEmbedder
from .vectorstores import GoodMemVectorStore

__all__ = [
    "GoodMemAPIError",
    "GoodMemBatchPartialFailureError",
    "GoodMemBatchWriteResultItem",
    "GoodMemConfigurationError",
    "GoodMemConnection",
    "GoodMemDuplicateIDError",
    "GoodMemEmbedderInfo",
    "GoodMemEmbeddings",
    "GoodMemMemoryInfo",
    "GoodMemOperationError",
    "GoodMemResources",
    "GoodMemSpaceEmbedder",
    "GoodMemSpaceInfo",
    "GoodMemVectorStore",
    "LangChainGoodMemError",
]
