# langchain-goodmem Documentation

This documentation set is organized for two audiences:

- users who want to call GoodMem from LangChain without learning the whole
  GoodMem SDK first
- contributors who need to understand how the package layers, tests, and
  transport boundary fit together

If you are new to GoodMem, start with the guides section first. It explains the
GoodMem concepts that matter for this package: spaces, memories, chunks,
embedders, and memory-level metadata filters.

If you already know the workflows and want exact signatures, semantics, and
exception behavior, jump to the API reference.

If you want to change the codebase, the developer docs describe the test matrix,
module layering, internal normalized types, and live integration helpers.
Start with `Developer Docs -> Developer Getting Started` if you are onboarding
to the repository itself.

```{toctree}
:maxdepth: 2
:caption: Documentation

guides/README
reference/README
developer/README
```

The runnable examples under `examples/` remain the operational source of truth
for end-to-end workflows, but the surrounding pages in this site explain the
mental model you need before you run them.

For GoodMem platform setup, server installation, and the full filter language,
refer to the official GoodMem docs:

- [GoodMem documentation](https://docs.goodmem.ai/docs/)
- [Filter expressions reference](https://docs.goodmem.ai/docs/reference/filter-expressions/)
