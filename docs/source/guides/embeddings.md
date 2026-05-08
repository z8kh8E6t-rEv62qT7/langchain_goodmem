# Embeddings Workflow

`GoodMemEmbeddings` is the right tool only for the workflows that truly need a
local LangChain `Embeddings` object.

Use it when:

- your application explicitly calls `embed_query(...)` or `embed_documents(...)`
- you want `GoodMemVectorStore.create(..., embedding=...)` to retain that
  embeddings object on the returned store

You do not need `GoodMemEmbeddings` just to search an existing GoodMem space.
`GoodMemVectorStore(space_id=..., ...)` can retrieve directly from GoodMem
without any local embeddings adapter.

The example below shows the full embeddings-driven path, and the API section
after it explains the exact setup and failure behavior.

```{eval-rst}
.. automodule:: goodmem_embeddings_workflow
```

```{eval-rst}
.. autoclass:: langchain_goodmem.embeddings.GoodMemEmbeddings
   :members:
   :no-index:
```
