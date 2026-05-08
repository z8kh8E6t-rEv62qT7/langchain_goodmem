"""langchain-goodmem package entry point and quick-start guide.

This package exposes GoodMem semantic search to LangChain code through a small,
explicit public surface:

- ``GoodMemConnection`` for shared transport configuration
- ``GoodMemVectorStore`` for write and chunk-level semantic retrieval workflows
- ``GoodMemEmbeddings`` for GoodMem-managed ``OPENAI``-compatible embedders
- ``GoodMemSpaceEmbedder`` for create-time embedder declarations

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

Minimal add-and-search flow:

::

    from langchain_core.documents import Document
    from langchain_goodmem import GoodMemConnection, GoodMemVectorStore

    connection = GoodMemConnection.from_env()
    store = GoodMemVectorStore(
        space_id="your-space-id",
        connection=connection,
    )
    store.add_documents(
        [Document(page_content="GoodMem stores semantically searchable memories.")],
        ids=["memory-1"],
    )
    results = store.similarity_search(
        "How does GoodMem work with LangChain?",
        k=1,
    )
    print(results[0].page_content)

For fuller workflows:

- see ``examples/basic_semantic_search.py`` for existing-space usage
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
from .space_embedders import GoodMemSpaceEmbedder
from .vectorstores import GoodMemVectorStore

__all__ = [
    "GoodMemAPIError",
    "GoodMemBatchPartialFailureError",
    "GoodMemBatchWriteResultItem",
    "GoodMemConfigurationError",
    "GoodMemConnection",
    "GoodMemDuplicateIDError",
    "GoodMemEmbeddings",
    "GoodMemOperationError",
    "GoodMemSpaceEmbedder",
    "GoodMemVectorStore",
    "LangChainGoodMemError",
]
