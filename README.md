# langchain-goodmem

`langchain-goodmem` is a LangChain integration for GoodMem semantic search.
It keeps the LangChain-facing surface focused on chunk-level semantic add and
search workflows.

## Core Capabilities

- `GoodMemVectorStore` binds to an existing GoodMem space or creates a new one
- `GoodMemEmbeddings` exposes a GoodMem-managed `OPENAI`-compatible embedder as
  a LangChain `Embeddings` implementation
- the integration keeps create, write, search, and error semantics explicit for
  LangChain callers

## Documentation

- [Hosted docs](https://z8kh8E6t-rEv62qT7.github.io/langchain_goodmem/)
- [Docs maintenance guide](docs/README.md)

## Examples

- [Existing-space semantic search example](examples/basic_semantic_search.py)
- [Embeddings-driven create example](examples/goodmem_embeddings_workflow.py)

## Local Docs Build

```bash
pip install -e '.[docs]'
./.venv/bin/sphinx-build -W -b html docs/source docs/build/html
```
