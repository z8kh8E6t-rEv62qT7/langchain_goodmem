"""Package-local exceptions exported by ``langchain_goodmem``.

The package keeps a small, explicit error model:

- configuration errors for local validation or missing setup
- operational errors for backend, transport, or duplicate-ID failures
- batch-partial-failure details so callers can reconcile successful writes
"""

from __future__ import annotations

from dataclasses import dataclass


class LangChainGoodMemError(Exception):
    """Base class for all package-specific exceptions."""


class GoodMemConfigurationError(LangChainGoodMemError):
    """Raised when required GoodMem configuration is missing or invalid."""


class GoodMemOperationError(LangChainGoodMemError):
    """Raised when a GoodMem-backed operation fails."""


class GoodMemDuplicateIDError(GoodMemOperationError):
    """Raised when a memory ID collides with an existing or repeated ID."""


class GoodMemAPIError(GoodMemOperationError):
    """Raised when the GoodMem backend returns an API or transport failure."""


@dataclass(frozen=True)
class GoodMemBatchWriteResultItem:
    """Stable per-item batch write result surfaced to callers.

    Attributes:
        request_index: Input position for the original write request.
        success: Whether the corresponding write succeeded.
        memory_id: Created or caller-provided memory ID when available.
        error_code: Backend error code for failures, when exposed.
        error_message: Backend error message for failures, when exposed.
    """

    request_index: int
    success: bool
    memory_id: str | None
    error_code: int | None
    error_message: str | None


class GoodMemBatchPartialFailureError(GoodMemAPIError):
    """Raised when a batch write partially succeeds and partially fails.

    Attributes:
        created_ids: Successfully created memory IDs from the same batch.
        results: Per-item batch result details in request order.
    """

    def __init__(
        self,
        message: str,
        *,
        created_ids: list[str],
        results: list[GoodMemBatchWriteResultItem],
    ) -> None:
        super().__init__(message)
        self.created_ids = list(created_ids)
        self.results = list(results)
