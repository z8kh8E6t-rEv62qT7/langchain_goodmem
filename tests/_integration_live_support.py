"""Live integration-test helpers and environment contract.

This module centralizes the resource lifecycle and environment policy for the
repository's live GoodMem tests.

Required base environment:

- ``GOODMEM_API_KEY``
- ``GOODMEM_BASE_URL``

Optional reuse controls:

- ``GOODMEM_SPACE_ID`` to reuse an existing space for existing-space coverage
- ``GOODMEM_EMBEDDER_ID`` to reuse an existing compatible embedder

Optional auto-provisioning inputs for embeddings coverage:

- ``GOODMEM_EMBEDDINGS_API_KEY``
- ``GOODMEM_EMBEDDINGS_BASE_URL``
- ``GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER``
- ``GOODMEM_EMBEDDINGS_DIMENSIONS``
- ``GOODMEM_EMBEDDINGS_PROVIDER_TYPE``

Resource lifecycle:

- existing resources referenced through environment variables are reused and
  not deleted
- temporary spaces and embedders created by the tests are cleaned up at the end
  of the run
- cleanup failures fail the test run so resource leaks stay visible

Additional behavior:

- ``GOODMEM_VERIFY`` is parsed only here, not by ``GoodMemConnection.from_env``
- semantic retrieval is eventually consistent, so tests poll until the expected
  chunk becomes visible or the timeout is reached
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time

from goodmem import Goodmem
from goodmem.errors import ConflictError
from goodmem.types import Modality, ProviderType, SpaceEmbedderConfig
from langchain_core.documents import Document
import pytest

from langchain_goodmem import GoodMemConfigurationError, GoodMemConnection, GoodMemVectorStore


@dataclass(frozen=True)
class LiveEmbedderResource:
    embedder_id: str
    created_by_test: bool


@dataclass(frozen=True)
class LiveSpaceResource:
    space_id: str
    created_by_test: bool


@dataclass(frozen=True)
class LiveExistingSpaceResource:
    space: LiveSpaceResource
    embedder: LiveEmbedderResource | None = None


def integration_connection() -> GoodMemConnection:
    verify_raw = os.getenv("GOODMEM_VERIFY", "false").strip().lower()
    verify = verify_raw not in {"0", "false", "no", "off"}
    try:
        return GoodMemConnection.from_env(verify=verify)
    except GoodMemConfigurationError:
        pytest.skip(
            "Live integration tests require GOODMEM_API_KEY and GOODMEM_BASE_URL."
        )


def integration_client(connection: GoodMemConnection) -> Goodmem:
    return Goodmem(
        base_url=connection.base_url,
        api_key=connection.api_key,
        verify=connection.verify,
    )


def embedding_integration_config() -> str | None:
    upstream_api_key = os.getenv("GOODMEM_EMBEDDINGS_API_KEY")
    if upstream_api_key is None or not upstream_api_key.strip():
        return None
    return upstream_api_key.strip()


def embedder_has_inline_api_key(embedder: object) -> bool:
    credentials = getattr(embedder, "credentials", None)
    api_key = getattr(credentials, "api_key", None) if credentials is not None else None
    inline_secret = getattr(api_key, "inline_secret", None) if api_key is not None else None
    return isinstance(inline_secret, str) and bool(inline_secret.strip())


def ensure_test_embedder(
    client: Goodmem,
    *,
    marker: str,
) -> LiveEmbedderResource:
    embedder_id = os.getenv("GOODMEM_EMBEDDER_ID")
    if embedder_id:
        return LiveEmbedderResource(embedder_id=embedder_id, created_by_test=False)

    provider_api_key = os.getenv("GOODMEM_EMBEDDINGS_API_KEY")
    endpoint_url = os.getenv("GOODMEM_EMBEDDINGS_BASE_URL")
    model_identifier = os.getenv("GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER")
    provider_type_name = os.getenv("GOODMEM_EMBEDDINGS_PROVIDER_TYPE", "OPENAI")
    dimensionality_raw = os.getenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    if not endpoint_url or not model_identifier:
        pytest.skip(
            "Live create-helper coverage requires GOODMEM_EMBEDDER_ID or "
            "GOODMEM_EMBEDDINGS_BASE_URL plus GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER."
        )
    if provider_type_name.strip().upper() != "OPENAI":
        pytest.skip(
            "Live embeddings coverage currently supports only "
            "GOODMEM_EMBEDDINGS_PROVIDER_TYPE=OPENAI."
        )
    dimensionality = int(dimensionality_raw)
    provider_type = ProviderType("OPENAI")

    existing = client.embedders.list()
    for item in existing:
        if (
            item.provider_type == provider_type
            and item.endpoint_url == endpoint_url
            and item.model_identifier == model_identifier
            and item.dimensionality == dimensionality
        ):
            if provider_api_key:
                if item.credentials is None or getattr(item.credentials, "api_key", None) is None:
                    continue
            return LiveEmbedderResource(
                embedder_id=item.embedder_id,
                created_by_test=False,
            )

    create_kwargs = {
        "display_name": marker,
        "model_identifier": model_identifier,
        "endpoint_url": endpoint_url,
        "provider_type": provider_type,
        "dimensionality": dimensionality,
        "supported_modalities": [Modality.TEXT],
    }
    if provider_api_key:
        create_kwargs["api_key"] = provider_api_key

    try:
        created = client.embedders.create(**create_kwargs)
        return LiveEmbedderResource(
            embedder_id=created.embedder_id,
            created_by_test=True,
        )
    except ConflictError as exc:
        body = getattr(exc, "body", None)
        if body:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("existingResourceId"):
                return LiveEmbedderResource(
                    embedder_id=str(parsed["existingResourceId"]),
                    created_by_test=False,
                )
        raise


def ensure_existing_space(client: Goodmem) -> LiveExistingSpaceResource:
    space_id = os.getenv("GOODMEM_SPACE_ID")
    if space_id:
        return LiveExistingSpaceResource(
            space=LiveSpaceResource(space_id=space_id, created_by_test=False),
        )

    embedder = ensure_test_embedder(
        client,
        marker="langchain-goodmem-live-existing-embedder",
    )

    name = f"langchain-goodmem-existing-{os.urandom(4).hex()}"
    try:
        space = client.spaces.create(
            name=name,
            space_embedders=[SpaceEmbedderConfig(embedder_id=embedder.embedder_id)],
            public_read=False,
        )
    except Exception:
        if embedder.created_by_test:
            client.embedders.delete(id=embedder.embedder_id)
        raise

    return LiveExistingSpaceResource(
        space=LiveSpaceResource(space_id=space.space_id, created_by_test=True),
        embedder=embedder,
    )


def cleanup_live_resources(
    client: Goodmem,
    *,
    space: LiveSpaceResource | None = None,
    embedder: LiveEmbedderResource | None = None,
) -> None:
    cleanup_errors: list[str] = []

    if space is not None and space.created_by_test:
        try:
            client.spaces.delete(id=space.space_id)
        except Exception as exc:
            cleanup_errors.append(
                f"failed to delete live test space {space.space_id}: {exc}"
            )

    if embedder is not None and embedder.created_by_test:
        try:
            client.embedders.delete(id=embedder.embedder_id)
        except Exception as exc:
            cleanup_errors.append(
                f"failed to delete live test embedder {embedder.embedder_id}: {exc}"
            )

    if cleanup_errors:
        raise AssertionError("Live integration cleanup failed: " + " | ".join(cleanup_errors))


def poll_similarity_search(
    store: GoodMemVectorStore,
    query: str,
    *,
    filter_expression: str | None = None,
    expected_memory_id: str | None = None,
    timeout_seconds: float = 45.0,
) -> list[Document]:
    deadline = time.time() + timeout_seconds
    last_results: list[Document] = []

    while time.time() < deadline:
        last_results = store.similarity_search(
            query,
            k=5,
            filter=filter_expression,
        )
        if not last_results:
            time.sleep(1.0)
            continue

        if expected_memory_id is None:
            return last_results

        if any(
            doc.metadata.get("_goodmem_memory_id") == expected_memory_id
            for doc in last_results
        ):
            return last_results

        time.sleep(1.0)

    pytest.fail(
        "Timed out waiting for GoodMem semantic retrieval to return the expected "
        f"memory. Last results: {last_results!r}"
    )
