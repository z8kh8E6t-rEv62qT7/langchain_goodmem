"""LangChain embeddings adapter for GoodMem-managed embedders.

``GoodMemEmbeddings`` exposes a GoodMem embedder as a LangChain
``Embeddings`` implementation when that embedder is compatible with an
``OPENAI``-style upstream embeddings endpoint.

Credential resolution order:

1. readable inline API key stored on the GoodMem embedder
2. ``GOODMEM_EMBEDDINGS_API_KEY``

Additional environment support:

- ``GOODMEM_EMBEDDINGS_DIMENSIONS`` remains optional, but when set it must
  match the dimensionality declared on the selected GoodMem embedder

Common setup failures covered by this module include:

- missing optional ``langchain-openai`` dependency
- unsupported provider type or modality
- upstream credential forwarding that GoodMem cannot perform directly
- provider initialization or request failures, which normalize to
  ``GoodMemAPIError``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from langchain_core.embeddings import Embeddings

from ._internal.providers import (
    create_provider_embeddings,
    format_provider_failure_message,
    load_embedder_config,
)
from ._internal.transport import GoodMemTransport
from ._internal.types import GoodMemEmbedderConfig
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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts through the resolved upstream provider.

        Args:
            texts: Non-empty text inputs to embed.

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
            text: Non-empty query text.

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
