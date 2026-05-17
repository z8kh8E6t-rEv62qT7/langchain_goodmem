# Internal Architecture

The package is intentionally layered so that public LangChain behavior can stay
explicit and testable.

Read the codebase from top to bottom in this order:

1. public entry points
   `connection.py`, `resources.py`, `vectorstores.py`, `embeddings.py`,
   `space_embedders.py`, and `errors.py`
2. validators
   local input validation that rejects malformed values before the SDK is
   touched
3. behavior layer
   memory-operation helpers that normalize batch writes and streamed retrieval
4. transport layer
   the only layer that talks directly to the official GoodMem SDK
5. internal normalized types
   dataclasses and protocols that keep the layers decoupled

That split is what lets the repository test most edge cases without requiring a
live GoodMem deployment for every change.

The embeddings bootstrap path follows the same layering rule:

- `embeddings.py` owns the public `GoodMemEmbeddings.ensure(...)` and
  `ensure_from_env(...)` entry points
- `resources.py` owns the public `GoodMemResources` facade for embedders,
  spaces, memories, and vector-store bootstrap
- `_internal.providers` owns bootstrap matching, request normalization, and
  readiness checks
- `_internal.transport` owns the SDK calls needed by the resource facade and
  LangChain-facing classes
- `_internal.types` owns the package-local bootstrap request and the narrow
  transport protocols used by upper layers

```{eval-rst}
.. automodule:: langchain_goodmem._internal.memory_ops
   :members:
```

```{eval-rst}
.. automodule:: langchain_goodmem._internal.providers
   :members:
```

```{eval-rst}
.. automodule:: langchain_goodmem._internal.validators
   :members:
```

```{eval-rst}
.. automodule:: langchain_goodmem._internal.transport
   :members:
```
