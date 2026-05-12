# langchain-goodmem

`langchain-goodmem` is a LangChain integration for GoodMem semantic search.
It keeps the LangChain-facing surface intentionally small and focuses on the
core workflows this package actually supports: create or bind to a GoodMem
space, write memories, retrieve semantic matches, and optionally expose one
compatible GoodMem embedder as a LangChain `Embeddings` object.

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

The bootstrap helpers are intentionally narrow. They only cover the package's
own `OPENAI`-compatible embeddings workflow, and they do not turn
`langchain-goodmem` into a general GoodMem resource CRUD layer.

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

## Documentation

- [Hosted docs](https://z8kh8E6t-rEv62qT7.github.io/langchain_goodmem/)
- [Docs maintenance guide](docs/README.md)

## Examples

- [Existing-space semantic search example](examples/basic_semantic_search.py)
- [Clean-slate embeddings bootstrap example](examples/goodmem_embeddings_bootstrap_workflow.py)
- [Embeddings-driven create example](examples/goodmem_embeddings_workflow.py)

## Local Docs Build

```bash
pip install -e '.[docs]'
./.venv/bin/sphinx-build -W -b html docs/source docs/_build/html
```
