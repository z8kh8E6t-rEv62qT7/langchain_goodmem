"""LangChain embeddings adapter for GoodMem-managed embedders.

``GoodMemEmbeddings`` turns one existing GoodMem embedder resource into a
LangChain ``Embeddings`` implementation, but only when that GoodMem embedder is
compatible with an ``OPENAI``-style upstream embeddings endpoint.

Use this adapter when you need local LangChain embedding calls such as
``embed_query(...)`` or ``embed_documents(...)``, or when you want
``GoodMemVectorStore.create(..., embedding=...)`` to both create a space and
retain a LangChain ``Embeddings`` object on the returned store.

You do not need ``GoodMemEmbeddings`` just to search an existing GoodMem space.
``GoodMemVectorStore(space_id=..., ...)`` can retrieve directly from GoodMem
without any local embeddings object.

Credential resolution order:

1. readable inline API key stored on the GoodMem embedder
2. ``GOODMEM_EMBEDDINGS_API_KEY``

For clean-slate onboarding, ``GoodMemEmbeddings.ensure(...)`` and
``GoodMemEmbeddings.ensure_from_env(...)`` can find or create one compatible
GoodMem embedder before returning a normal ``GoodMemEmbeddings`` instance.
``ensure_from_env(...)`` also supports ``GOODMEM_EMBEDDER_ID`` reuse, but when
that path is used the bootstrap environment must still be present and must
match the selected embedder so configuration drift stays explicit. For broader
RAG-resource setup such as spaces, memories, and one-shot vector-store
bootstrap, use ``GoodMemResources``.

Additional environment support:

- ``GOODMEM_EMBEDDINGS_DIMENSIONS`` remains optional, but when set it must
  match the dimensionality declared on the selected GoodMem embedder
- ``GOODMEM_EMBEDDINGS_BASE_URL``, ``GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER``, and
  ``GOODMEM_EMBEDDINGS_DIMENSIONS`` are required by
  ``GoodMemEmbeddings.ensure_from_env(...)`` even when that helper reuses an
  existing ``GOODMEM_EMBEDDER_ID``

Common setup failures covered by this module include:

- missing optional ``langchain-openai`` dependency
- unsupported provider type or modality
- upstream credential forwarding that GoodMem cannot perform directly
- provider initialization or request failures, which normalize to
  ``GoodMemAPIError``
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from langchain_core.embeddings import Embeddings

from ._internal.providers import (
    create_provider_embeddings,
    default_bootstrap_display_name,
    ensure_embedder,
    format_provider_failure_message,
    load_embedder_config,
)
from ._internal.transport import GoodMemTransport
from ._internal.types import GoodMemEmbedderBootstrapRequest, GoodMemEmbedderConfig
from ._internal.validators import require_embedder_id, validate_text_inputs
from .connection import GoodMemConnection
from .errors import GoodMemAPIError, GoodMemConfigurationError

ProviderResult = TypeVar("ProviderResult")


def _create_transport(connection: GoodMemConnection) -> GoodMemTransport:
    return GoodMemTransport(connection)


class GoodMemEmbeddings(Embeddings):
    """LangChain embeddings implementation backed by a GoodMem embedder.

    This adapter loads the GoodMem embedder lazily on the first real embedding
    request, validates that the upstream provider shape is compatible, and then
    delegates to ``langchain_openai.OpenAIEmbeddings``.

    ``embedder_id`` is the identifier of a GoodMem embedder resource, not the
    raw upstream model name. GoodMem is responsible for storing the provider
    connection details that this adapter resolves on demand.

    Args:
        embedder_id: Explicit GoodMem embedder ID to load on demand.
        connection: Shared GoodMem transport configuration.

    Raises:
        GoodMemConfigurationError: If ``embedder_id`` is blank.
    """

    def __init__(self, embedder_id: str, connection: GoodMemConnection) -> None:
        self.embedder_id = require_embedder_id(embedder_id)
        self._transport = _create_transport(connection)
        self._embedder_config: GoodMemEmbedderConfig | None = None
        self._provider_embeddings: Embeddings | None = None

    @classmethod
    def ensure(
        cls,
        *,
        connection: GoodMemConnection,
        endpoint_url: str,
        model_identifier: str,
        dimensionality: int,
        upstream_api_key: str | None = None,
        display_name: str | None = None,
    ) -> GoodMemEmbeddings:
        """Find or create one compatible GoodMem embedder and return it.

        This helper resolves one ``OPENAI``-compatible embedder that can back
        ``GoodMemEmbeddings``. Use ``GoodMemResources`` when you also need to
        manage spaces, memories, or one-shot vector-store bootstrap.

        Args:
            connection: Shared GoodMem transport configuration.
            endpoint_url: Upstream provider endpoint URL.
            model_identifier: Upstream embedding model identifier.
            dimensionality: Required embedding dimensionality.
            upstream_api_key: Optional upstream API key stored on a newly
                created embedder. When omitted, the helper falls back to
                ``GOODMEM_EMBEDDINGS_API_KEY`` for creation.
            display_name: Optional display name used only when a new embedder
                must be created.

        Returns:
            A ready-to-use ``GoodMemEmbeddings`` instance bound to the matched
            or created GoodMem embedder.

        Raises:
            GoodMemConfigurationError: If the bootstrap inputs are invalid, if
                multiple compatible embedders are found, or if the resolved
                embedder is incompatible with ``GoodMemEmbeddings``.
            GoodMemAPIError: If GoodMem rejects the bootstrap lookup or create
                operations.
        """
        transport = _create_transport(connection)
        resolved_api_key = _normalize_optional_env_or_value(
            upstream_api_key,
            env_name="GOODMEM_EMBEDDINGS_API_KEY",
        )
        resolved_config = ensure_embedder(
            transport,
            request=GoodMemEmbedderBootstrapRequest(
                display_name=display_name or default_bootstrap_display_name(),
                endpoint_url=endpoint_url,
                model_identifier=model_identifier,
                dimensionality=dimensionality,
                api_key=resolved_api_key,
            ),
            upstream_api_key_override=resolved_api_key,
        )
        embeddings = cls(embedder_id=resolved_config.embedder_id, connection=connection)
        embeddings._embedder_config = resolved_config
        embeddings._provider_embeddings = create_provider_embeddings(
            resolved_config,
            upstream_api_key_override=resolved_api_key,
        )
        return embeddings

    @classmethod
    def ensure_from_env(
        cls,
        *,
        connection: GoodMemConnection | None = None,
        verify: bool | str = True,
    ) -> GoodMemEmbeddings:
        """Build one bootstrap-backed embeddings adapter from environment.

        This helper reuses the current ``GOODMEM_EMBEDDINGS_*`` contract to
        resolve one compatible embedder without introducing a larger resource
        management surface. When ``GOODMEM_EMBEDDER_ID`` is set, the helper
        reuses that embedder only after confirming the bootstrap environment
        still matches the embedder's endpoint, model, and dimensionality.

        Args:
            connection: Optional explicit GoodMem connection. When omitted, the
                helper uses ``GoodMemConnection.from_env()``.
            verify: TLS verification setting used only when ``connection`` is
                omitted and a new connection must be built from environment.

        Returns:
            A ready-to-use ``GoodMemEmbeddings`` instance bound to the matched
            or created GoodMem embedder.

        Raises:
            GoodMemConfigurationError: If required bootstrap environment values
                are missing or invalid.
            GoodMemAPIError: If GoodMem rejects the bootstrap lookup or create
                operations.
        """
        resolved_connection = connection or GoodMemConnection.from_env(verify=verify)
        embedder_id = _normalize_optional_env_or_value(
            None,
            env_name="GOODMEM_EMBEDDER_ID",
        )
        endpoint_url = _require_env_text(
            "GOODMEM_EMBEDDINGS_BASE_URL",
            error_message=(
                "GoodMemEmbeddings.ensure_from_env() requires "
                "GOODMEM_EMBEDDINGS_BASE_URL."
            ),
        )
        model_identifier = _require_env_text(
            "GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER",
            error_message=(
                "GoodMemEmbeddings.ensure_from_env() requires "
                "GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER."
            ),
        )
        dimensionality = _require_env_int(
            "GOODMEM_EMBEDDINGS_DIMENSIONS",
            error_message=(
                "GoodMemEmbeddings.ensure_from_env() requires "
                "GOODMEM_EMBEDDINGS_DIMENSIONS to be a positive integer."
            ),
        )
        if embedder_id is not None:
            resolved_embedder = load_embedder_config(
                _create_transport(resolved_connection),
                embedder_id=embedder_id,
            )
            _validate_bootstrap_env_matches_embedder(
                resolved_embedder,
                endpoint_url=endpoint_url,
                model_identifier=model_identifier,
                dimensionality=dimensionality,
            )
            embeddings = cls(
                embedder_id=resolved_embedder.embedder_id,
                connection=resolved_connection,
            )
            embeddings._embedder_config = resolved_embedder
            return embeddings
        return cls.ensure(
            connection=resolved_connection,
            endpoint_url=endpoint_url,
            model_identifier=model_identifier,
            dimensionality=dimensionality,
            upstream_api_key=_normalize_optional_env_or_value(
                None,
                env_name="GOODMEM_EMBEDDINGS_API_KEY",
            ),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts through the resolved upstream provider.

        Args:
            texts: Non-empty text inputs to embed. These calls do not write any
                memories into GoodMem; they only use GoodMem as the source of
                embedder configuration.

        Returns:
            A list of embedding vectors. Empty input returns an empty list
            without contacting GoodMem or the upstream provider.

        Raises:
            GoodMemConfigurationError: If any text input is blank or if the
                selected embedder cannot be configured.
            GoodMemAPIError: If the upstream provider initialization or request
                fails.
        """
        if not texts:
            return []
        validate_text_inputs(
            texts,
            label="texts",
            exception_type=GoodMemConfigurationError,
        )
        return self._invoke_provider(lambda provider: provider.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string through the resolved upstream provider.

        Args:
            text: Non-empty query text. This call does not retrieve from or
                write to a GoodMem space; it only resolves the configured
                upstream embedder.

        Returns:
            One embedding vector.

        Raises:
            GoodMemConfigurationError: If the query is blank or the embedder
                setup is invalid.
            GoodMemAPIError: If provider initialization or the upstream request
                fails.
        """
        validate_text_inputs(
            [text],
            label="texts",
            exception_type=GoodMemConfigurationError,
        )
        return self._invoke_provider(lambda provider: provider.embed_query(text))

    def _invoke_provider(
        self,
        operation: Callable[[Embeddings], ProviderResult],
    ) -> ProviderResult:
        """Run one operation against the lazily initialized provider adapter.

        Args:
            operation: Callable that receives the resolved upstream
                ``Embeddings`` implementation.

        Returns:
            The result returned by the provider operation, such as one
            embedding vector or a list of embedding vectors.

        Raises:
            GoodMemConfigurationError: If embedder setup fails before the
                provider call can succeed.
            GoodMemAPIError: If the upstream provider fails or raises an
                unexpected exception.
        """
        try:
            return operation(self._get_provider_embeddings())
        except GoodMemConfigurationError:
            raise
        except GoodMemAPIError:
            raise
        except Exception as exc:
            raise GoodMemAPIError(
                format_provider_failure_message(
                    "Upstream embeddings request failed",
                    exc,
                )
            ) from exc

    def _get_embedder_config(self) -> GoodMemEmbedderConfig:
        """Load and cache the normalized GoodMem embedder configuration.

        Returns:
            The validated ``GoodMemEmbedderConfig`` for ``self.embedder_id``.
        """
        if self._embedder_config is None:
            self._embedder_config = load_embedder_config(
                self._transport,
                embedder_id=self.embedder_id,
            )
        return self._embedder_config

    def _get_provider_embeddings(self) -> Embeddings:
        """Load and cache the upstream LangChain embeddings implementation.

        Returns:
            The upstream ``Embeddings`` implementation derived from the GoodMem
            embedder configuration.
        """
        if self._provider_embeddings is None:
            self._provider_embeddings = create_provider_embeddings(
                self._get_embedder_config(),
            )
        return self._provider_embeddings


def _normalize_optional_env_or_value(value: str | None, *, env_name: str) -> str | None:
    """Return one trimmed explicit value or trimmed environment fallback.

    Args:
        value: Optional explicit string supplied by the caller.
        env_name: Environment variable consulted when ``value`` is missing or
            blank.

    Returns:
        The trimmed explicit value when present; otherwise the trimmed
        environment value; otherwise ``None``.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    env_value = os.getenv(env_name)
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip()
    return None


def _require_env_text(env_name: str, *, error_message: str) -> str:
    """Load one required non-empty environment value.

    Args:
        env_name: Environment variable name to read.
        error_message: Configuration error message raised for missing or blank
            values.

    Returns:
        The trimmed environment value.

    Raises:
        GoodMemConfigurationError: If the environment variable is missing or
            blank.
    """
    value = os.getenv(env_name)
    if value is None or not value.strip():
        raise GoodMemConfigurationError(error_message)
    return value.strip()


def _require_env_int(env_name: str, *, error_message: str) -> int:
    """Load one required positive integer environment value.

    Args:
        env_name: Environment variable name to read.
        error_message: Configuration error message raised for missing, blank,
            non-integer, or non-positive values.

    Returns:
        The parsed positive integer value.

    Raises:
        GoodMemConfigurationError: If the environment variable does not contain
            a positive integer.
    """
    value = _require_env_text(env_name, error_message=error_message)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise GoodMemConfigurationError(error_message) from exc
    if parsed <= 0:
        raise GoodMemConfigurationError(error_message)
    return parsed


def _validate_bootstrap_env_matches_embedder(
    embedder: GoodMemEmbedderConfig,
    *,
    endpoint_url: str,
    model_identifier: str,
    dimensionality: int,
) -> None:
    """Require bootstrap environment values to match a reused embedder.

    Args:
        embedder: Resolved embedder selected through ``GOODMEM_EMBEDDER_ID``.
        endpoint_url: Required bootstrap endpoint URL from environment.
        model_identifier: Required bootstrap model identifier from environment.
        dimensionality: Required bootstrap dimensionality from environment.

    Raises:
        GoodMemConfigurationError: If the reused embedder does not exactly
            match the declared bootstrap environment values.
    """
    if embedder.endpoint_url != endpoint_url:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings.ensure_from_env() found that GOODMEM_EMBEDDER_ID "
            "does not match GOODMEM_EMBEDDINGS_BASE_URL."
        )
    if embedder.model_identifier != model_identifier:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings.ensure_from_env() found that GOODMEM_EMBEDDER_ID "
            "does not match GOODMEM_EMBEDDINGS_MODEL_IDENTIFIER."
        )
    if embedder.dimensionality != dimensionality:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings.ensure_from_env() found that GOODMEM_EMBEDDER_ID "
            "does not match GOODMEM_EMBEDDINGS_DIMENSIONS."
        )
