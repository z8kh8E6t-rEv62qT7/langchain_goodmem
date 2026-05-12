# Troubleshooting

When this repository misbehaves, the failures usually fall into one of a small
number of buckets:

- import or dependency problems
  `goodmem` is required for transport-backed workflows, and `langchain-openai`
  is required for `GoodMemEmbeddings`
- missing environment variables
  public usage needs `GOODMEM_API_KEY` and `GOODMEM_BASE_URL`; live embeddings
  coverage and `GoodMemEmbeddings.ensure_from_env(...)` may also need
  `GOODMEM_EMBEDDINGS_*`, and `GOODMEM_EMBEDDER_ID` reuse still requires those
  bootstrap values to match the selected embedder
- filter expression mistakes
  GoodMem filters are their own language and operate on memory-level metadata,
  not on chunk-level result objects
- eventual consistency surprises
  a successful write does not guarantee that the memory is immediately
  searchable, because GoodMem still needs to process it into chunks

Good references while debugging:

- the public `embeddings` and `vectorstores` module docstrings for user-facing
  setup and failure behavior
- `tests._integration_live_support` for the exact live-test environment policy
- the official [GoodMem filter expressions reference](https://docs.goodmem.ai/docs/reference/filter-expressions/)
  when a filter is rejected by the backend

```{eval-rst}
.. automodule:: langchain_goodmem.embeddings
   :no-index:
```

```{eval-rst}
.. automodule:: langchain_goodmem.vectorstores
   :no-index:
```

```{eval-rst}
.. automodule:: tests._integration_live_support
   :no-index:
```
