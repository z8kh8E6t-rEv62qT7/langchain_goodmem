# Developer Docs

This section is for contributors who need to understand how the package is put
together, how it is tested, and where GoodMem-specific behavior enters the
stack.

If you are new to the codebase, start with `Developer Getting Started` first.
The rest of this section is meant to be deeper reference material after that
orientation pass.

Start here if you are asking questions like:

- which layer owns validation, transport, and behavior normalization
- where chunk-level search results are turned into LangChain `Document` values
- how to run unit versus live integration tests
- which environment variables the live tests expect
- which internal dataclasses and protocols are intentionally package-owned

Important boundary:

- everything under `src/langchain_goodmem/_internal` is implementation-facing
  and may change more freely than the public API
- the public contract is defined by the modules documented in the reference
  section, not by the internal helper types documented here

```{toctree}
:maxdepth: 1

getting-started
testing
troubleshooting
architecture
internal-types
```
