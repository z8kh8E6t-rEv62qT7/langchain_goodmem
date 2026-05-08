# Documentation Guide

This repository uses Sphinx to build a docstring-driven documentation site from
the sources under `docs/source/`.

The site keeps a small set of Markdown navigation pages, but the canonical
content for API reference, guides, and internal developer details now lives in
Python docstrings under:

- `src/langchain_goodmem/`
- `examples/`
- `tests/`

Build the docs locally with:

```bash
pip install -e '.[docs]'
sphinx-build -W -b html docs/source docs/build/html
```

## Layout

```text
docs/
  README.md
  source/
    conf.py
    index.md
    guides/
      README.md
      getting-started.md
      existing-space.md
      create-space.md
      embeddings.md
    reference/
      README.md
      api.md
    developer/
      README.md
      testing.md
      troubleshooting.md
      architecture.md
      internal-types.md
```

Navigation rules:

- `README.md` is the GitHub and package front door
- `docs/source/index.md` is the Sphinx root document
- `guides/` is for user workflows
- `reference/` is for public API details
- `developer/` is for testing and implementation-facing documentation
- Markdown pages should stay thin and mostly contain navigation plus autodoc
  directives
- Canonical behavior and usage text should live in Google-style docstrings, not
  in long hand-authored Markdown pages

## Adding Or Updating Pages

- Put end-user workflow docstrings on existing public modules or runnable
  example modules under `examples/`.
- Put public API details on exported classes, methods, functions, and exception
  types under `src/langchain_goodmem/`.
- Put maintainer and implementation-facing details on `_internal` modules and
  test helper modules.
- Update the relevant section landing page and `docs/source/index.md` whenever
  you add a new page or change a canonical page for a workflow.
- Prefer one canonical docstring source per workflow so that getting-started,
  existing-space usage, create-space usage, and embeddings guidance do not
  drift across multiple files.

## Moving Or Renaming Pages

- Keep filenames in lowercase kebab-case for ordinary docs pages, and use
  `README.md` for docs landing pages.
- When a page moves, update inbound links from:
  - `README.md`
  - `docs/source/index.md`
  - the section landing page that owns the document
- Do not leave duplicate content behind at the old path. Either move the page
  cleanly or rewrite the canonical page in place.

## Linking Conventions

- Use relative Markdown links inside the docs tree so the source remains easy
  to browse on GitHub.
- Use Sphinx autodoc directives inside Markdown wrappers when the rendered page
  should come from Python docstrings.
- Keep headings stable unless there is a clear reason to rename them, so links
  remain predictable.

## Examples As The Source Of Truth

The runnable examples live in `examples/`:

- `examples/basic_semantic_search.py`
- `examples/goodmem_embeddings_workflow.py`

Docs may quote short snippets from those workflows, but the scripts are the
source of truth for runnable examples. When example behavior changes, update the
script docstring first and then adjust any surrounding navigation wrappers to
match.
