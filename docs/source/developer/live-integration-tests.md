# Live Integration Tests

The live suite is the place where the package proves it still works against a
real GoodMem deployment instead of only against fakes.

The three tests cover one path each:

- existing-space usage
- package-owned create-helper usage
- embeddings-driven create usage

Keep these constraints in mind:

- the suite assumes a reachable GoodMem deployment and valid credentials
- semantic retrieval is eventually consistent, so the helpers poll instead of
  assuming immediate visibility after a write
- temporary spaces and embedders created by the tests are cleaned up, but reused
  resources from environment variables are intentionally left alone

```{eval-rst}
.. automodule:: tests.test_integration_existing_space_live
```

```{eval-rst}
.. automodule:: tests.test_integration_create_live
```

```{eval-rst}
.. automodule:: tests.test_integration_embeddings_live
```
