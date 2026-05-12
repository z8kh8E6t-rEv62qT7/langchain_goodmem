# Unit Tests

The unit suite is split by package layer so failures stay local and easy to
interpret.

Suite map:

- `test_connection_unit` checks public connection normalization and environment
  loading
- `test_embeddings_unit` isolates `GoodMemEmbeddings` with fake transports and
  fake upstream providers, including bootstrap matching and ensure-from-env
  behavior
- `test_transport_unit` checks the SDK boundary and error normalization logic,
  including the minimal embedder bootstrap mapping
- `test_vectorstore_unit` covers the main LangChain-facing write and retrieval
  behavior

If you are debugging a regression, start with the narrowest suite that still
touches the failing layer before you run the broader matrix.

```{eval-rst}
.. automodule:: tests.test_connection_unit
```

```{eval-rst}
.. automodule:: tests.test_embeddings_unit
```

```{eval-rst}
.. automodule:: tests.test_transport_unit
```

```{eval-rst}
.. automodule:: tests.test_vectorstore_unit
```
