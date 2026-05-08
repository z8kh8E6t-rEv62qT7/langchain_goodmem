# User Guides

Read these pages in order if you are new to the package:

- `Getting Started` explains the GoodMem concepts that show up in the public
  API and helps you choose the right entry point.
- `Connect To An Existing Space` is the simplest workflow when you already have
  a GoodMem space.
- `Create A Space` explains the package-owned create helper and when to choose
  `embedders=` versus `embedding=`.
- `Embeddings Workflow` explains when `GoodMemEmbeddings` is necessary and when
  it is not.

Two cross-cutting rules are worth remembering before you dive in:

- writes create GoodMem memories, while searches return chunk-level LangChain
  `Document` values
- GoodMem metadata filters apply to memory-level JSON metadata, not to
  chunk-level result objects

```{toctree}
:maxdepth: 1

getting-started
existing-space
create-space
embeddings
```
