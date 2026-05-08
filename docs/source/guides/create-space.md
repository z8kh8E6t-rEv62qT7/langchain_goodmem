# Create A Space

Use `GoodMemVectorStore.create(...)` when you want this package to create a new
GoodMem space and immediately hand back a bound LangChain `VectorStore`.

This helper is intentionally narrow. It focuses on the LangChain-facing pieces
that this repository can explain clearly:

- the space name
- which GoodMem embedder or embedders should be attached
- whether the returned store should also retain a usable LangChain
  `Embeddings` object

Choose the create-time embedder input this way:

- use `embedders=[GoodMemSpaceEmbedder(...)]` when you already know the GoodMem
  embedder IDs that should be attached to the space and you only need
  server-side retrieval
- use `embedding=GoodMemEmbeddings(...)` when the same GoodMem embedder should
  both back the new space and remain available locally as `store.embeddings`

If you need broader GoodMem space controls such as labels, ownership,
public-read settings, client-specified IDs, or chunking configuration, create
the space through the GoodMem SDK or API first and then bind
`GoodMemVectorStore` to the resulting `space_id`.

```{eval-rst}
.. automethod:: langchain_goodmem.vectorstores.GoodMemVectorStore.create
   :no-index:
```

```{eval-rst}
.. autoclass:: langchain_goodmem.space_embedders.GoodMemSpaceEmbedder
   :no-index:
```
