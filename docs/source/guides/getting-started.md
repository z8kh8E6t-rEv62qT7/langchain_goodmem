# Getting Started

Use this page to build the right mental model before you look at the individual
APIs.

The most important thing to understand is that this package is a focused
LangChain wrapper around a subset of GoodMem:

- it writes memories into a GoodMem space
- it retrieves chunk-level semantic matches from that space
- it optionally exposes a GoodMem-managed embedder as a LangChain
  `Embeddings` object
- it exposes `GoodMemResources` for the embedders, spaces, and memories needed
  by normal LangChain RAG/search workflows

## Core Capabilities

- `GoodMemVectorStore` binds to an existing GoodMem space or creates a new one
- `GoodMemResources` manages the embedders, spaces, and memories needed for
  normal LangChain RAG/search workflows without exposing the full GoodMem SDK
- `GoodMemEmbeddings` exposes a GoodMem-managed `OPENAI`-compatible embedder as
  a LangChain `Embeddings` implementation
- the integration keeps create, write, search, filter, and error semantics
  explicit for LangChain callers

## What GoodMem Means Here

GoodMem has a few core concepts that matter when you use this package:

- a `space` is the container you write into and search within
- a `memory` is one stored source item plus metadata
- GoodMem processes a memory asynchronously and may split it into one or more
  searchable `chunk` values

That means write and search APIs return different identifiers:

- write calls return memory IDs
- search calls return chunk-level `Document` values
- `Document.id` in search results is the matching chunk ID
- the parent memory ID and space ID stay available in `Document.metadata`
  under `_goodmem_memory_id` and `_goodmem_space_id`

## Quick Start

If your GoodMem instance is already initialized, reachable, and you have a
valid API key, the shortest clean-slate path looks like this:

```python
from langchain_core.documents import Document

from langchain_goodmem import GoodMemResources

resources = GoodMemResources.from_env()
store = resources.bootstrap_vector_store(
    space_name="langchain-demo",
    endpoint_url="https://embeddings.example/v1",
    model_identifier="text-embedding-3-small",
    dimensionality=1536,
)
store.add_documents([Document(page_content="GoodMem works with LangChain.")])
results = store.similarity_search("How does GoodMem work with LangChain?", k=1)
```

For a complete runnable version, see
`examples/goodmem_embeddings_bootstrap_workflow.py` in the repository.

## Which Entry Point Should I Use?

- Use `GoodMemVectorStore(space_id=..., connection=...)` when you already have a
  GoodMem space ID.
- Use `GoodMemVectorStore.create(...)` when you want this package to create a
  new space for you.
- Use `GoodMemResources(...)` when you need to create/list/get/delete the
  embedders, spaces, or memories used by the normal LangChain workflow.
- Use `GoodMemEmbeddings(...)` only when you need local LangChain embedding
  calls or when `GoodMemVectorStore.create(..., embedding=...)` should retain a
  LangChain `Embeddings` object on the returned store.
- Use `GoodMemEmbeddings.ensure(...)` or `GoodMemEmbeddings.ensure_from_env(...)`
  when you want the package to bridge the clean-slate setup gap by finding or
  creating one compatible `OPENAI`-style GoodMem embedder before returning a
  normal `GoodMemEmbeddings` instance. `ensure_from_env(...)` also supports
  `GOODMEM_EMBEDDER_ID` reuse when the bootstrap environment is present and
  explicitly matches that embedder.

## Two Startup Paths

For embeddings workflows, there are two intentionally different ways to begin:

- Existing-resource path: if you already have a compatible GoodMem embedder,
  construct `GoodMemEmbeddings(embedder_id=..., connection=...)` directly.
- Clean-slate bootstrap path: if you do not yet have a compatible embedder,
  use `GoodMemEmbeddings.ensure(...)` or `GoodMemEmbeddings.ensure_from_env(...)`.
  If you set `GOODMEM_EMBEDDER_ID` for `ensure_from_env(...)`, still provide
  the bootstrap environment and keep it aligned with that embedder so config
  drift is caught early.

`GoodMemResources.bootstrap_vector_store(...)` is the one-shot clean-slate path:
it can resolve a compatible embedder, create a space, and return a ready-to-use
`GoodMemVectorStore`. Broader platform administration such as API keys, server
init, migrations, system operations, and LLM/reranker/OCR/extension management
stays in GoodMem's SDK, CLI, and UI.

## Before You Start

- You need a running GoodMem server or deployment.
- You need `GOODMEM_API_KEY` and `GOODMEM_BASE_URL`, or you can pass those
  values directly into `GoodMemConnection(...)`.
- If you plan to use `GoodMemEmbeddings`, install the optional extra with
  `pip install -e '.[openai]'`.
- If you plan to use the bootstrap helpers, also provide
  `GOODMEM_EMBEDDINGS_BASE_URL`, `GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER`, and
  `GOODMEM_EMBEDDINGS_DIMENSIONS`. Set `GOODMEM_EMBEDDINGS_API_KEY` when the
  upstream provider key must be stored on a newly created embedder or forwarded
  because the resolved embedder does not expose a readable inline secret. When
  you also set `GOODMEM_EMBEDDER_ID`, those bootstrap values are still
  required and must match the selected embedder exactly.

If you are new to GoodMem itself, the official product documentation is the best
place to learn the platform concepts and server setup:

- [GoodMem documentation](https://docs.goodmem.ai/docs/)
- [GoodMem filter expressions reference](https://docs.goodmem.ai/docs/reference/filter-expressions/)

The package module docstring below is the canonical quick-start narrative for
the public surface.

```{eval-rst}
.. automodule:: langchain_goodmem
```
