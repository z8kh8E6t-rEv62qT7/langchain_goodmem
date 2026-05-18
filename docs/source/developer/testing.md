# Testing

The repository keeps its test matrix split by responsibility instead of mixing
everything into one giant suite.

Use this page to decide which test surface you need:

- `Static Quality Checks` for linting, import ordering, and repository-wide
  formatting checks on `src` and `tests`
- `Unit Tests` for fast feedback on validation, transport mapping, vector-store
  behavior, and embeddings behavior with fakes
- `Live Integration Tests` for end-to-end checks against a real GoodMem server
- `Test Helpers` for the shared environment contract and cleanup policy behind
  the live suite

Local quality commands:

```bash
ruff check src tests
ruff format --check src tests
```

The package-level `tests` docstring below contains the canonical commands for
setting up the contributor environment and running the suite.

```{eval-rst}
.. automodule:: tests
```

```{toctree}
:maxdepth: 1

unit-tests
live-integration-tests
test-helpers
```

This section stays focused on how to run the test matrix and where to find more
detailed coverage notes for each suite.
