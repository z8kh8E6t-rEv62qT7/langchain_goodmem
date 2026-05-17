# Manage GoodMem RAG Resources

Use `GoodMemResources` when you want the LangChain integration to manage the
GoodMem resources needed for the normal RAG/search workflow: embedders, spaces,
memories, and a clean-slate bootstrap path.

This facade is deliberately narrower than the full GoodMem SDK. It does not
cover API key lifecycle, server init, migrations, system operations, or
LLM/reranker/OCR/extension administration.

```{eval-rst}
.. automodule:: langchain_goodmem.resources
   :members:
   :no-index:
```
