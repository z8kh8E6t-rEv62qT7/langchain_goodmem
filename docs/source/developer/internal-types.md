# Internal Types

These types are not public API, but they are important to the internal design.

Why they exist:

- they keep upper layers from depending directly on raw GoodMem SDK models
- they give validators, behavior helpers, and transport adapters a shared
  package-owned vocabulary
- they make unit tests easier to write because protocols can be faked without
  recreating the whole SDK surface

Pay special attention to:

- `GoodMemWriteRequest` for normalized write inputs
- `GoodMemSearchHit` for the chunk-level shape that eventually becomes a
  LangChain `Document`
- `GoodMemSpaceCreateRequest`, `GoodMemMemoryCreateRequest`,
  `GoodMemEmbedderBootstrapRequest`, and `GoodMemEmbedderConfig` for the
  resource, bootstrap, and embeddings paths
- the transport protocols for the smallest callable surface each upper layer
  actually needs

```{eval-rst}
.. automodule:: langchain_goodmem._internal.types
   :members:
```
