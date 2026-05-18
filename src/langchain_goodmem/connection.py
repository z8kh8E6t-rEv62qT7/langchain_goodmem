"""Public GoodMem connection configuration.

``GoodMemConnection`` is the caller-facing transport configuration shared by
the vector-store and embeddings entry points.

Use ``GoodMemConnection(...)`` when you want to pass credentials and TLS
settings explicitly in code. Use ``GoodMemConnection.from_env()`` when your
process already provides:

- ``GOODMEM_API_KEY``
- ``GOODMEM_BASE_URL``

``GOODMEM_VERIFY`` is intentionally not read by this module. Pass ``verify=``
explicitly when you need a custom TLS setting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ._internal.validators import require_non_empty_trimmed_string, require_verify_value
from .errors import GoodMemConfigurationError


@dataclass(frozen=True)
class GoodMemConnection:
    """Normalized GoodMem transport configuration.

    Args:
        api_key: GoodMem API key used for all SDK-backed requests.
        base_url: Base URL for the GoodMem API or deployment.
        verify: TLS verification setting forwarded to the GoodMem SDK client.
            Use ``True`` or ``False`` for standard verification behavior, or a
            non-empty string path when the SDK should use a custom CA bundle.

    Attributes:
        api_key: Trimmed GoodMem API key.
        base_url: Trimmed GoodMem base URL.
        verify: Validated TLS verification setting.

    Raises:
        GoodMemConfigurationError: If any field is missing, blank, or uses an
            unsupported ``verify`` shape.
    """

    api_key: str
    base_url: str
    verify: bool | str = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "api_key",
            require_non_empty_trimmed_string(
                self.api_key,
                error_message="GoodMemConnection requires a non-empty api_key.",
                exception_type=GoodMemConfigurationError,
            ),
        )
        object.__setattr__(
            self,
            "base_url",
            require_non_empty_trimmed_string(
                self.base_url,
                error_message="GoodMemConnection requires a non-empty base_url.",
                exception_type=GoodMemConfigurationError,
            ),
        )
        object.__setattr__(self, "verify", require_verify_value(self.verify))

    @classmethod
    def from_env(cls, *, verify: bool | str = True) -> GoodMemConnection:
        """Build a connection from GoodMem process environment variables.

        Args:
            verify: TLS verification setting forwarded to the constructor after
                ``GOODMEM_API_KEY`` and ``GOODMEM_BASE_URL`` are loaded.

        Returns:
            A validated ``GoodMemConnection`` instance.

        Raises:
            GoodMemConfigurationError: If either required environment variable
                is missing or blank.
        """

        api_key = os.getenv("GOODMEM_API_KEY")
        base_url = os.getenv("GOODMEM_BASE_URL")
        if api_key is None or not api_key.strip():
            raise GoodMemConfigurationError(
                "GoodMemConnection.from_env() requires GOODMEM_API_KEY."
            )
        if base_url is None or not base_url.strip():
            raise GoodMemConfigurationError(
                "GoodMemConnection.from_env() requires GOODMEM_BASE_URL."
            )
        return cls(api_key=api_key, base_url=base_url, verify=verify)
