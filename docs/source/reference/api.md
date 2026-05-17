# API Reference

The pages below are generated from the package's exported modules.

Reading tips:

- start with `langchain_goodmem.vectorstores` if you want the main LangChain
  workflow
- read `langchain_goodmem.embeddings` only if you need local embedding calls
- read `langchain_goodmem.resources` if you need the GoodMem embedders, spaces,
  and memories used by a normal LangChain RAG workflow
- use `langchain_goodmem.space_embedders` to understand the `embedders=` path of
  `GoodMemVectorStore.create(...)`
- read `langchain_goodmem.errors` if you need to catch and distinguish package
  errors precisely

```{eval-rst}
.. autosummary::
   :signatures: short

   langchain_goodmem.connection
   langchain_goodmem.resources
   langchain_goodmem.vectorstores
   langchain_goodmem.embeddings
   langchain_goodmem.space_embedders
   langchain_goodmem.errors
```

```{eval-rst}
.. automodule:: langchain_goodmem.connection
   :members:
   :no-index:
```

```{eval-rst}
.. automodule:: langchain_goodmem.resources
   :members:
   :no-index:
```

```{eval-rst}
.. automodule:: langchain_goodmem.vectorstores
   :members:
   :no-index:
```

```{eval-rst}
.. automodule:: langchain_goodmem.embeddings
   :members:
   :no-index:
```

```{eval-rst}
.. automodule:: langchain_goodmem.space_embedders
   :members:
   :no-index:
```

```{eval-rst}
.. automodule:: langchain_goodmem.errors
   :members:
   :no-index:
```
