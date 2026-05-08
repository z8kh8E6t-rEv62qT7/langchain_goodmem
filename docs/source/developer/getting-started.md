# Developer Getting Started

This page is the recommended first stop for contributors who are new to the
repository.

## What This Package Is

`langchain-goodmem` is a focused LangChain wrapper around a subset of GoodMem.
It is not a full GoodMem SDK replacement.

The package is intentionally narrow:

- it exposes GoodMem spaces as LangChain-facing vector-store workflows
- it exposes one compatible class of GoodMem-managed embedders as LangChain
  `Embeddings`
- it keeps validation, transport mapping, and error normalization inside
  package-owned layers so behavior stays explicit and testable

If you need the full GoodMem platform surface such as advanced space settings,
broader API coverage, or direct resource administration, contributors should
expect to use the official GoodMem SDK or API alongside this package rather
than growing `langchain-goodmem` into a full mirror of the platform.

## GoodMem Concepts That Matter To Contributors

These concepts appear throughout both the user-facing API and the internal
layers:

- a `space` is the top-level container you write into and search within
- a `memory` is one stored item of source content plus metadata
- GoodMem processes a memory asynchronously and may split it into one or more
  searchable `chunk` values
- an `embedder` is a GoodMem-managed embedding configuration that can be
  attached to a space and, in some cases, exposed locally as LangChain
  `Embeddings`
- metadata filters apply to memory-level JSON metadata through the GoodMem
  filter language, not to chunk-level LangChain result objects
- search is eventually consistent because writes need to be processed into
  retrievable chunks before they appear in semantic search

The most important identifier distinction in this repository is:

- write APIs return memory IDs
- search APIs return LangChain `Document` values whose `Document.id` is the
  matching chunk ID

## Layer Map

Read and reason about the repository in this order:

1. public entry points
   `src/langchain_goodmem/connection.py`,
   `src/langchain_goodmem/vectorstores.py`,
   `src/langchain_goodmem/embeddings.py`,
   `src/langchain_goodmem/space_embedders.py`,
   `src/langchain_goodmem/errors.py`
2. validators
   `src/langchain_goodmem/_internal/validators.py`
3. behavior layer
   `src/langchain_goodmem/_internal/memory_ops.py`
4. transport layer
   `src/langchain_goodmem/_internal/transport.py`
5. internal normalized types
   `src/langchain_goodmem/_internal/types.py`
6. tests
   `tests/` plus the live helper module in `tests/_integration_live_support.py`

The intent behind this split is simple:

- public modules define the supported behavior
- validators reject malformed caller input before the SDK is touched
- behavior helpers turn raw GoodMem responses into stable package semantics
- the transport layer is the only place that talks directly to the official
  GoodMem SDK
- internal types let upper layers depend on package-owned contracts instead of
  raw SDK models

For more detail after this overview, continue to [Internal Architecture](architecture.md)
and [Internal Types](internal-types.md).

## Where To Read First

Start from the entrypoint that matches the change you want to make:

- public behavior
  Read `src/langchain_goodmem/vectorstores.py` and
  `src/langchain_goodmem/embeddings.py`, then confirm expected behavior in the
  public docs under [API Reference](../reference/api.md).
- transport boundary
  Read `src/langchain_goodmem/_internal/transport.py` and
  `tests/test_transport_unit.py`.
- internal normalized types
  Read `src/langchain_goodmem/_internal/types.py` and then the companion
  [Internal Types](internal-types.md) page.
- unit tests
  Start with [Testing](testing.md), then narrow to `tests/test_connection_unit.py`,
  `tests/test_embeddings_unit.py`, `tests/test_transport_unit.py`, or
  `tests/test_vectorstore_unit.py`.
- live integration helpers
  Read `tests/_integration_live_support.py` and the
  [Test Helpers](test-helpers.md) page before changing live coverage.

## Local Setup And Verification

Use the minimum setup needed to read, test, and rebuild docs locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pip install -e '.[docs]'
```

Rebuild docs:

```bash
./.venv/bin/sphinx-build -W -E -b text docs/source docs/_build/text
./.venv/bin/sphinx-build -W -E -b html docs/source docs/_build/html
```

Run the unit suite:

```bash
./.venv/bin/python -m pytest \
  tests/test_connection_unit.py \
  tests/test_embeddings_unit.py \
  tests/test_transport_unit.py \
  tests/test_vectorstore_unit.py
```

Run the live integration suite:

```bash
./.venv/bin/python -m pytest \
  tests/test_integration_existing_space_live.py \
  tests/test_integration_create_live.py \
  tests/test_integration_embeddings_live.py \
  -m integration
```

## How To Choose The Right Test Surface

Use unit tests when:

- you are changing validation logic
- you are changing vector-store behavior or result mapping
- you are changing embeddings setup or provider compatibility rules
- you are changing transport normalization and want fast feedback without a live
  GoodMem deployment

Use live integration tests when:

- you are changing behavior that depends on the real GoodMem SDK or server
- you need to confirm eventual-consistency handling
- you need confidence that create, write, retrieval, and embedder wiring still
  works end to end

If a change touches both public behavior and the SDK boundary, start with unit
tests to pin down semantics, then run the live suite for end-to-end confidence.

## Common First Tasks

If you are changing vector-store behavior:

- start with `src/langchain_goodmem/vectorstores.py`
- then review `src/langchain_goodmem/_internal/memory_ops.py`
- verify with `tests/test_vectorstore_unit.py`

If you are changing embeddings behavior:

- start with `src/langchain_goodmem/embeddings.py`
- then review `src/langchain_goodmem/_internal/providers.py`
- verify with `tests/test_embeddings_unit.py`

If you are changing transport mapping or SDK exception handling:

- start with `src/langchain_goodmem/_internal/transport.py`
- verify with `tests/test_transport_unit.py`

If you are changing docs only:

- start with the relevant page under `docs/source/`
- rebuild both text and HTML docs
- confirm that any referenced public semantics still match the module docstrings

## Boundaries And Non-Goals

Everything under `src/langchain_goodmem/_internal` is implementation-facing.
It exists to keep the public package surface small and explicit, not to serve as
its own stable API.

The public contract lives in the pages generated from:

- `src/langchain_goodmem/connection.py`
- `src/langchain_goodmem/vectorstores.py`
- `src/langchain_goodmem/embeddings.py`
- `src/langchain_goodmem/space_embedders.py`
- `src/langchain_goodmem/errors.py`

Contributors should treat the reference docs as the source of truth for
externally visible behavior and treat the internal developer pages as guidance
for how the implementation is organized today.
