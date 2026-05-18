# Examples

The runnable scripts under `examples/` are the operational source of truth for
end-to-end usage. Use the surrounding guide pages first if you want the mental
model, then jump into these scripts when you want concrete runnable flows.

- `examples/basic_semantic_search.py`
  Existing-space workflow: bind to one GoodMem space, add documents, and run
  chunk-level semantic search.
- `examples/goodmem_embeddings_bootstrap_workflow.py`
  Clean-slate bootstrap workflow: start from an initialized GoodMem instance,
  resolve or create a compatible embedder, create a space, and run the first
  query.
- `examples/goodmem_embeddings_workflow.py`
  Embeddings-driven create workflow: use `GoodMemEmbeddings` directly when you
  want a retained LangChain `Embeddings` object as part of the setup.
