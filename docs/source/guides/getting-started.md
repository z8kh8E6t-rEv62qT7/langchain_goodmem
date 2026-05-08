# Getting Started

Use this page to build the right mental model before you look at the individual
APIs.

The most important thing to understand is that this package is a focused
LangChain wrapper around a subset of GoodMem:

- it writes memories into a GoodMem space
- it retrieves chunk-level semantic matches from that space
- it optionally exposes a GoodMem-managed embedder as a LangChain
  `Embeddings` object

If you only remember one distinction, remember this one:

- write calls return memory IDs
- search calls return chunk-level `Document` values

The package module docstring below is the canonical quick-start narrative for
the public surface.

```{eval-rst}
.. automodule:: langchain_goodmem
```
