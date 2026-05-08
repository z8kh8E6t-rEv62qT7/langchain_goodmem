# Connect To An Existing Space

This is the best starting workflow for most LangChain users.

Choose this path when you already have a GoodMem `space_id` and you want to:

- write texts or LangChain `Document` values into that space
- run semantic retrieval without managing embeddings locally
- keep the LangChain integration as thin as possible

Important behavior to keep in mind while you read the example below:

- each write call creates one or more GoodMem memories
- each search result is a chunk-level LangChain `Document`
- `Document.id` in search results is the GoodMem chunk ID, not the memory ID
- metadata filters operate on the stored memory metadata using the GoodMem
  filter language, for example `val('$.lang') = 'en'`

```{eval-rst}
.. automodule:: basic_semantic_search
```
