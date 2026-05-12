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

There are two embeddings startup paths:

- Existing-resource path: use `GoodMemEmbeddings(embedder_id=..., connection=...)`
  when you already know the compatible GoodMem embedder ID you want.
- Clean-slate bootstrap path: use `GoodMemEmbeddings.ensure(...)` or
  `GoodMemEmbeddings.ensure_from_env(...)` when you need the package to find or
  create one compatible `OPENAI` embedder first. When you reuse
  `GOODMEM_EMBEDDER_ID` through `ensure_from_env(...)`, the bootstrap
  environment must still be present and must match that embedder exactly.

The bootstrap helpers are intentionally narrow. They exist to close the
clean-slate onboarding gap for the package's own embeddings workflow, not to
turn this repository into a general GoodMem resource CRUD layer.

The examples below show both clean-slate bootstrap and the direct
embeddings-driven path, and the API section after them explains the exact setup
and failure behavior.

```{eval-rst}
.. automodule:: goodmem_embeddings_bootstrap_workflow
```

```{eval-rst}
.. automodule:: goodmem_embeddings_workflow
```

```{eval-rst}
.. autoclass:: langchain_goodmem.embeddings.GoodMemEmbeddings
   :members:
   :no-index:
```
