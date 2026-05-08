"""Unit tests for ``GoodMemEmbeddings``.

This suite exercises the embeddings adapter without talking to a real GoodMem
deployment or upstream provider.

Coverage goals:

- explicit ``embedder_id`` validation and lazy transport/provider setup
- upstream API-key resolution from inline credentials or environment fallbacks
- dimensions validation, provider-shape compatibility checks, and optional
  dependency failures
- normalization of provider factory failures, request failures, and embedder
  lookup failures

The suite is kept separate because the embeddings adapter has a distinct set of
boundary conditions from vector-store writes and searches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings

from langchain_goodmem import (
    GoodMemAPIError,
    GoodMemConfigurationError,
    GoodMemConnection,
    GoodMemEmbeddings,
)


@dataclass(frozen=True)
class FakeRawEmbedder:
    embedder_id: str = "embedder-123"
    provider_type: str = "OPENAI"
    endpoint_url: str = "https://embeddings.example"
    api_path: str | None = "/embeddings"
    model_identifier: str = "text-embedding-3-large"
    dimensionality: int = 1024
    supported_modalities: tuple[str, ...] = ("TEXT",)
    credentials: Any | None = None


@dataclass(frozen=True)
class FakeApiKeyCredentials:
    inline_secret: str | None = None


@dataclass(frozen=True)
class FakeEndpointCredentials:
    kind: str
    api_key: FakeApiKeyCredentials | None = None
    gcp_adc: object | None = None


class FakeProviderEmbeddings(Embeddings):
    def __init__(
        self,
        *,
        document_result: list[list[float]] | None = None,
        query_result: list[float] | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.document_result = document_result or [[0.1, 0.2]]
        self.query_result = query_result or [0.3, 0.4]
        self.exception = exception
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        if self.exception is not None:
            raise self.exception
        return self.document_result

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        if self.exception is not None:
            raise self.exception
        return self.query_result


@dataclass
class FakeTransport:
    embedder: FakeRawEmbedder = field(default_factory=FakeRawEmbedder)
    exception: Exception | None = None
    get_calls: list[str] = field(default_factory=list)

    def get_embedder(self, *, embedder_id: str) -> FakeRawEmbedder:
        self.get_calls.append(embedder_id)
        if self.exception is not None:
            raise self.exception
        return self.embedder


def _connection() -> GoodMemConnection:
    return GoodMemConnection(api_key="gm-key", base_url="https://goodmem.example")


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: FakeTransport,
) -> list[GoodMemConnection]:
    connections: list[GoodMemConnection] = []

    def fake_create_transport(connection: GoodMemConnection) -> FakeTransport:
        connections.append(connection)
        return transport

    monkeypatch.setattr(
        "langchain_goodmem.embeddings._create_transport",
        fake_create_transport,
    )
    return connections


def test_constructor_requires_non_empty_embedder_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, FakeTransport())

    with pytest.raises(GoodMemConfigurationError, match="embedder_id"):
        GoodMemEmbeddings(embedder_id="  ", connection=_connection())


def test_embed_documents_returns_empty_without_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    _patch_transport(monkeypatch, transport)

    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    assert embeddings.embed_documents([]) == []
    assert transport.get_calls == []


def test_embed_query_uses_connection_and_caches_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "env-fallback-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport(
        embedder=FakeRawEmbedder(
            credentials=FakeEndpointCredentials(
                kind="CREDENTIAL_KIND_API_KEY",
                api_key=FakeApiKeyCredentials(inline_secret="stored-inline-key"),
            )
        )
    )
    connections = _patch_transport(monkeypatch, transport)
    provider = FakeProviderEmbeddings(query_result=[0.1, 0.2])
    factory_calls: list[dict[str, Any]] = []

    def fake_openai_embeddings(*, model: str, **kwargs: Any) -> Embeddings:
        factory_calls.append({"model": model, "kwargs": kwargs})
        return provider

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings",
        fake_openai_embeddings,
    )

    embeddings = GoodMemEmbeddings(embedder_id=" embedder-123 ", connection=_connection())

    assert embeddings.embed_query("hello") == [0.1, 0.2]
    assert embeddings.embed_query("again") == [0.1, 0.2]
    assert connections == [_connection()]
    assert not hasattr(embeddings, "_connection")
    assert transport.get_calls == ["embedder-123"]
    assert len(factory_calls) == 1
    assert factory_calls[0] == {
        "model": "text-embedding-3-large",
        "kwargs": {
            "api_key": "stored-inline-key",
            "base_url": "https://embeddings.example",
            "dimensions": 1024,
            "check_embedding_ctx_length": False,
        },
    }
    assert provider.query_calls == ["hello", "again"]


@pytest.mark.parametrize(
    ("api_path", "expected_base_url"),
    [
        (None, "https://embeddings.example"),
        ("/embeddings", "https://embeddings.example"),
        ("/v1/embeddings", "https://embeddings.example/v1"),
    ],
)
def test_supported_api_path_shapes_derive_expected_base_url(
    monkeypatch: pytest.MonkeyPatch,
    api_path: str | None,
    expected_base_url: str,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.delenv("GOODMEM_EMBEDDINGS_DIMENSIONS", raising=False)
    transport = FakeTransport(embedder=FakeRawEmbedder(api_path=api_path))
    _patch_transport(monkeypatch, transport)
    factory_calls: list[dict[str, Any]] = []

    def fake_openai_embeddings(*, model: str, **kwargs: Any) -> Embeddings:
        factory_calls.append({"model": model, "kwargs": kwargs})
        return FakeProviderEmbeddings(query_result=[0.5, 0.6])

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings",
        fake_openai_embeddings,
    )

    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    assert embeddings.embed_query("hello") == [0.5, 0.6]
    assert factory_calls[0]["kwargs"]["base_url"] == expected_base_url
    assert "dimensions" not in factory_calls[0]["kwargs"]


def test_upstream_embedding_env_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOODMEM_EMBEDDINGS_API_KEY", raising=False)
    monkeypatch.delenv("GOODMEM_EMBEDDINGS_DIMENSIONS", raising=False)
    transport = FakeTransport()
    _patch_transport(monkeypatch, transport)
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(
        GoodMemConfigurationError,
        match="does not expose any credentials",
    ):
        embeddings.embed_query("hello")


def test_embedder_inline_api_key_works_without_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOODMEM_EMBEDDINGS_API_KEY", raising=False)
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport(
        embedder=FakeRawEmbedder(
            credentials=FakeEndpointCredentials(
                kind="CREDENTIAL_KIND_API_KEY",
                api_key=FakeApiKeyCredentials(inline_secret="stored-inline-key"),
            )
        )
    )
    _patch_transport(monkeypatch, transport)
    factory_calls: list[dict[str, Any]] = []

    def fake_openai_embeddings(*, model: str, **kwargs: Any) -> Embeddings:
        factory_calls.append({"model": model, "kwargs": kwargs})
        return FakeProviderEmbeddings(query_result=[0.7, 0.8])

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings",
        fake_openai_embeddings,
    )
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    assert embeddings.embed_query("hello") == [0.7, 0.8]
    assert factory_calls[0]["kwargs"]["api_key"] == "stored-inline-key"


def test_env_fallback_is_used_when_embedder_has_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport(embedder=FakeRawEmbedder(credentials=None))
    _patch_transport(monkeypatch, transport)
    factory_calls: list[dict[str, Any]] = []

    def fake_openai_embeddings(*, model: str, **kwargs: Any) -> Embeddings:
        factory_calls.append({"model": model, "kwargs": kwargs})
        return FakeProviderEmbeddings(query_result=[0.9, 1.0])

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings",
        fake_openai_embeddings,
    )
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    assert embeddings.embed_query("hello") == [0.9, 1.0]
    assert factory_calls[0]["kwargs"]["api_key"] == "upstream-key"


def test_api_key_credentials_without_inline_secret_require_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOODMEM_EMBEDDINGS_API_KEY", raising=False)
    monkeypatch.delenv("GOODMEM_EMBEDDINGS_DIMENSIONS", raising=False)
    transport = FakeTransport(
        embedder=FakeRawEmbedder(
            credentials=FakeEndpointCredentials(
                kind="CREDENTIAL_KIND_API_KEY",
                api_key=FakeApiKeyCredentials(inline_secret=None),
            )
        )
    )
    _patch_transport(monkeypatch, transport)
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(
        GoodMemConfigurationError,
        match="uses API-key credentials, but GoodMem did not expose a readable inline secret",
    ):
        embeddings.embed_query("hello")


def test_non_inline_credentials_require_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOODMEM_EMBEDDINGS_API_KEY", raising=False)
    monkeypatch.delenv("GOODMEM_EMBEDDINGS_DIMENSIONS", raising=False)
    transport = FakeTransport(
        embedder=FakeRawEmbedder(
            credentials=FakeEndpointCredentials(
                kind="CREDENTIAL_KIND_GCP_ADC",
                gcp_adc=object(),
            )
        )
    )
    _patch_transport(monkeypatch, transport)
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(
        GoodMemConfigurationError,
        match="uses CREDENTIAL_KIND_GCP_ADC credentials",
    ):
        embeddings.embed_query("hello")


@pytest.mark.parametrize(
    ("raw_dimensions", "match"),
    [
        ("bad", "must be an integer"),
        ("0", "greater than 0"),
        ("1536", "does not match the GoodMem embedder dimensionality 1024"),
    ],
)
def test_dimensions_env_is_validated(
    monkeypatch: pytest.MonkeyPatch,
    raw_dimensions: str,
    match: str,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", raw_dimensions)
    transport = FakeTransport()
    _patch_transport(monkeypatch, transport)
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(GoodMemConfigurationError, match=match):
        embeddings.embed_query("hello")


@pytest.mark.parametrize(
    ("embedder", "match"),
    [
        (FakeRawEmbedder(provider_type="TEI"), "OPENAI"),
        (FakeRawEmbedder(supported_modalities=("IMAGE",)), "TEXT modality"),
        (FakeRawEmbedder(endpoint_url=""), "endpoint_url"),
        (FakeRawEmbedder(model_identifier=""), "model_identifier"),
        (FakeRawEmbedder(api_path="/v1/custom"), "api_path"),
        (FakeRawEmbedder(dimensionality=0), "dimensionality"),
    ],
)
def test_embedder_config_validation_rejects_incompatible_embedders(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeRawEmbedder,
    match: str,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport(embedder=embedder)
    _patch_transport(monkeypatch, transport)
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(GoodMemConfigurationError, match=match):
        embeddings.embed_query("hello")


def test_embed_documents_validates_texts_without_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    _patch_transport(monkeypatch, transport)
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(GoodMemConfigurationError, match="non-empty string"):
        embeddings.embed_documents([""])

    with pytest.raises(GoodMemConfigurationError, match="non-empty string"):
        embeddings.embed_query("  ")
    assert transport.get_calls == []


def test_embed_documents_delegate_through_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport(embedder=FakeRawEmbedder(api_path="/v1/embeddings"))
    _patch_transport(monkeypatch, transport)
    provider = FakeProviderEmbeddings(document_result=[[1.0, 1.0], [2.0, 2.0]])

    def fake_openai_embeddings(*, model: str, **kwargs: Any) -> Embeddings:
        return provider

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings",
        fake_openai_embeddings,
    )
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    result = embeddings.embed_documents(["one", "two"])

    assert result == [[1.0, 1.0], [2.0, 2.0]]
    assert provider.document_calls == [["one", "two"]]


def test_embed_documents_accepts_large_batches_without_local_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport(embedder=FakeRawEmbedder(api_path="/v1/embeddings"))
    _patch_transport(monkeypatch, transport)
    texts = [f"text-{index}" for index in range(2049)]
    expected_result = [[float(index)] for index in range(len(texts))]
    provider = FakeProviderEmbeddings(document_result=expected_result)

    def fake_openai_embeddings(*, model: str, **kwargs: Any) -> Embeddings:
        return provider

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings",
        fake_openai_embeddings,
    )
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    result = embeddings.embed_documents(texts)

    assert result == expected_result
    assert provider.document_calls == [texts]


def test_provider_failures_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport()
    _patch_transport(monkeypatch, transport)
    provider = FakeProviderEmbeddings(exception=RuntimeError("provider boom"))

    def fake_openai_embeddings(*, model: str, **kwargs: Any) -> Embeddings:
        return provider

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings",
        fake_openai_embeddings,
    )
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(
        GoodMemAPIError,
        match="Upstream embeddings request failed: provider boom",
    ):
        embeddings.embed_query("hello")


def test_provider_factory_failures_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport()
    _patch_transport(monkeypatch, transport)

    def fake_openai_embeddings(*, model: str, **kwargs: Any) -> Embeddings:
        raise RuntimeError("factory boom")

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings",
        fake_openai_embeddings,
    )
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(
        GoodMemAPIError,
        match=(
            "Failed to initialize the LangChain OpenAI-compatible embeddings "
            "provider: factory boom"
        ),
    ):
        embeddings.embed_query("hello")


def test_missing_optional_openai_dependency_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport()
    _patch_transport(monkeypatch, transport)

    def fake_import_module(name: str) -> Any:
        if name == "langchain_openai":
            raise ImportError("No module named 'langchain_openai'")
        raise AssertionError(f"Unexpected module import: {name}")

    monkeypatch.setattr(
        "langchain_goodmem._internal.providers.importlib.import_module",
        fake_import_module,
    )
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(
        GoodMemConfigurationError,
        match=r"pip install langchain-goodmem\[openai\]",
    ):
        embeddings.embed_query("hello")


def test_embedder_lookup_failures_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport(exception=GoodMemAPIError("backend failed"))
    _patch_transport(monkeypatch, transport)
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(GoodMemAPIError, match="backend failed"):
        embeddings.embed_query("hello")


def test_missing_openai_embeddings_class_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_API_KEY", "upstream-key")
    monkeypatch.setenv("GOODMEM_EMBEDDINGS_DIMENSIONS", "1024")
    transport = FakeTransport()
    _patch_transport(monkeypatch, transport)

    class FakeModule:
        pass

    monkeypatch.setattr(
        "langchain_goodmem._internal.providers.importlib.import_module",
        lambda name: FakeModule(),
    )
    embeddings = GoodMemEmbeddings(embedder_id="embedder-123", connection=_connection())

    with pytest.raises(
        GoodMemConfigurationError,
        match="could not load OpenAIEmbeddings",
    ):
        embeddings.embed_query("hello")
