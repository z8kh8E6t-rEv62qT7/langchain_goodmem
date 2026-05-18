"""Test matrix and contributor entry points for ``langchain-goodmem``.

Set up the local contributor environment once:

::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e '.[test]'

Run the static quality checks with:

::

    ./.venv/bin/ruff check src tests
    ./.venv/bin/ruff format --check src tests

Run the unit suite with:

::

    ./.venv/bin/python -m pytest \
      tests/test_connection_unit.py \
      tests/test_embeddings_unit.py \
      tests/test_resources_unit.py \
      tests/test_transport_unit.py \
      tests/test_vectorstore_unit.py

Run the live integration suite with:

::

    ./.venv/bin/python -m pytest \
      tests/test_integration_existing_space_live.py \
      tests/test_integration_create_live.py \
      tests/test_integration_embeddings_live.py \
      -m integration

Coverage focus:

- ``GoodMemConnection.from_env()``
- ``GoodMemResources`` resource CRUD and clean-slate vector-store bootstrap
- vector-store create, write, and chunk-search flows
- embeddings configuration, upstream credential resolution, bootstrap behavior,
  and create-time integration behavior
- transport error normalization
- one live existing-space path, one live create-helper path, and one live
  embeddings bootstrap plus embeddings-driven path
"""
