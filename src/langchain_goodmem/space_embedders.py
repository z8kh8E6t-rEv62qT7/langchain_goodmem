"""Create-time embedder declarations for ``GoodMemVectorStore.create``.

This module is the direct-space-configuration counterpart to
``langchain_goodmem.embeddings``:

- use ``GoodMemSpaceEmbedder`` when you only need to attach one or more
  existing GoodMem embedder IDs to a newly created space
- use ``GoodMemEmbeddings`` when you also need a local LangChain
  ``Embeddings`` object
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import GoodMemConfigurationError


@dataclass(frozen=True)
class GoodMemSpaceEmbedder:
    """Public create-time embedder declaration.

    Use this dataclass when ``GoodMemVectorStore.create(...)`` should create a
    new space from one or more explicit GoodMem embedder IDs instead of
    inferring a single embedder from ``GoodMemEmbeddings``.

    This is the right choice when you already know which GoodMem embedder IDs
    the space should use for retrieval and you do not need a retained local
    LangChain ``Embeddings`` object on the returned store.

    Args:
        embedder_id: Non-empty GoodMem embedder identifier.
        default_retrieval_weight: Optional retrieval weight attached to this
            embedder in the created space. GoodMem uses these weights when a
            space has multiple embedders and retrieval needs to blend their
            results. Omit the value to let GoodMem apply its default weight.

    Attributes:
        embedder_id: Trimmed GoodMem embedder identifier.
        default_retrieval_weight: Retrieval weight normalized to ``float`` when
            provided.

    Raises:
        GoodMemConfigurationError: If ``embedder_id`` is blank or
            ``default_retrieval_weight`` is not numeric.
    """

    embedder_id: str
    default_retrieval_weight: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "embedder_id", _normalize_embedder_id(self.embedder_id)
        )
        object.__setattr__(
            self,
            "default_retrieval_weight",
            _normalize_default_retrieval_weight(self.default_retrieval_weight),
        )


def _normalize_embedder_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoodMemConfigurationError(
            "GoodMemSpaceEmbedder requires a non-empty embedder_id."
        )
    return value.strip()


def _normalize_default_retrieval_weight(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GoodMemConfigurationError(
            "GoodMemSpaceEmbedder default_retrieval_weight must be a number or None."
        )
    return float(value)
