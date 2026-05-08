# langchain-goodmem

`langchain-goodmem` is a LangChain integration for GoodMem semantic search.
It keeps the LangChain-facing surface intentionally small and focuses on the
core workflows this package actually supports: create or bind to a GoodMem
space, write memories, and retrieve semantic matches.

## What GoodMem Means Here

GoodMem has a few core concepts that matter when you use this package:

- a `space` is the container you write into and search within
- a `memory` is one stored source item plus metadata
- GoodMem processes a memory asynchronously and may split it into one or more
  searchable `chunk` values

That means write and search APIs return different identifiers:

- `add_documents(...)` and `add_texts(...)` return memory IDs
- `similarity_search(...)` returns LangChain `Document` values whose
  `Document.id` is the matching chunk ID
- the parent memory ID and space ID stay available in `Document.metadata`
  under `_goodmem_memory_id` and `_goodmem_space_id`

## Core Capabilities

- `GoodMemVectorStore` binds to an existing GoodMem space or creates a new one
- `GoodMemEmbeddings` exposes a GoodMem-managed `OPENAI`-compatible embedder as
  a LangChain `Embeddings` implementation
- the integration keeps create, write, search, filter, and error semantics
  explicit for LangChain callers

## Which Entry Point Should I Use?

- Use `GoodMemVectorStore(space_id=..., connection=...)` when you already have a
  GoodMem space ID from the GoodMem SDK, CLI, or web console.
- Use `GoodMemVectorStore.create(...)` when you want this package to create a
  new space for you.
- Use `GoodMemEmbeddings(...)` only when you need local LangChain embedding
  calls or when `GoodMemVectorStore.create(..., embedding=...)` should retain a
  LangChain `Embeddings` object on the returned store.

## Before You Start

- You need a running GoodMem server or deployment.
- You need `GOODMEM_API_KEY` and `GOODMEM_BASE_URL`, or you can pass those
  values directly into `GoodMemConnection(...)`.
- If you plan to use `GoodMemEmbeddings`, install the optional extra with
  `pip install -e '.[openai]'`.

If you are new to GoodMem itself, the official product documentation is the best
place to learn the platform concepts and server setup:

- [GoodMem documentation](https://docs.goodmem.ai/docs/)
- [GoodMem filter expressions reference](https://docs.goodmem.ai/docs/reference/filter-expressions/)

## Documentation

- [Hosted docs](https://z8kh8E6t-rEv62qT7.github.io/langchain_goodmem/)
- [Docs maintenance guide](docs/README.md)

## Examples

- [Existing-space semantic search example](examples/basic_semantic_search.py)
- [Embeddings-driven create example](examples/goodmem_embeddings_workflow.py)

## Local Docs Build

```bash
pip install -e '.[docs]'
./.venv/bin/sphinx-build -W -b html docs/source docs/_build/html
```
