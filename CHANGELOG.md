# Changelog

## 0.2.0

- Added `GoodMemResources` to cover the GoodMem resource workflow around normal
  LangChain RAG/search usage.
- Added clean-slate bootstrap from an initialized GoodMem instance so the
  package can create or reuse a compatible embedder, create a space, and return
  a ready-to-use `GoodMemVectorStore`.
- Added explicit memory deletion through `GoodMemVectorStore.delete(...)` using
  GoodMem memory IDs.
- Added unit coverage and documentation for the new resource, bootstrap, and
  deletion paths.

## 0.1.4

- Polished embeddings bootstrap behavior and contributor-facing documentation.

## 0.1.3

- Added clean-slate bootstrap for compatible embeddings workflows, plus broader
  bootstrap test coverage.

## 0.1.2

- Moved the project docs to Sphinx with docstring-driven API/reference content
  and refreshed examples.

## 0.1.1

- Expanded unit coverage and tightened vector store, embeddings, and transport
  behavior.

## 0.1.0

- Initial public release of the LangChain integration for GoodMem.
