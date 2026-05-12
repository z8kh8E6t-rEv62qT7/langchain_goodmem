# Test Helpers

This page documents the shared helper module behind the live integration suite.

It is the best place to look when you need to answer questions like:

- which environment variables are required or optional for live coverage
- how reusable versus temporary GoodMem resources are chosen
- where cleanup happens
- how the tests wait for eventually consistent retrieval results
- how compatible embedders are reused or provisioned for live bootstrap and
  embeddings coverage

One subtle but important rule lives here: `GOODMEM_VERIFY` is interpreted only
by the live-test helpers, not by `GoodMemConnection.from_env()`.

```{eval-rst}
.. automodule:: tests._integration_live_support
```
