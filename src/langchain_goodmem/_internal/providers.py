"""Private helpers for the GoodMem embeddings path.

This module owns the provider-facing logic behind ``GoodMemEmbeddings``.

Responsibilities:

- normalize the GoodMem embedder response into ``GoodMemEmbedderConfig``
- validate that the selected embedder can back an ``OPENAI``-compatible
  LangChain embeddings adapter
- normalize and validate bootstrap requests used by
  ``GoodMemEmbeddings.ensure(...)``
- find compatible existing embedders before create-once bootstrap falls back to
  provisioning a new one
- resolve credentials and optional dimensions overrides
- verify that bootstrap-resolved embedders are immediately usable by the
  current embeddings path
- build the upstream provider ``Embeddings`` implementation lazily
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from langchain_core.embeddings import Embeddings

from ..errors import GoodMemAPIError, GoodMemConfigurationError
from .types import (
    GoodMemEmbedderBootstrapRequest,
    GoodMemEmbedderConfig,
    SupportsEmbedderTransport,
)

_OPENAI_EXTRA_INSTALL_HINT = "Install it with `pip install langchain-goodmem[openai]`."
_DEFAULT_BOOTSTRAP_DISPLAY_NAME = "langchain-goodmem-openai"


def load_embedder_config(
    transport: SupportsEmbedderTransport,
    *,
    embedder_id: str,
) -> GoodMemEmbedderConfig:
    """Load and validate one GoodMem embedder configuration.

    Args:
        transport: Transport implementation exposing embedder lookup.
        embedder_id: GoodMem embedder ID to resolve.

    Returns:
        A normalized, validated embedder configuration.

    Raises:
        GoodMemConfigurationError: If the embedder shape is incompatible with
            ``GoodMemEmbeddings``.
        GoodMemAPIError: If GoodMem lookup fails.
    """
    config = _to_embedder_config(transport.get_embedder(embedder_id=embedder_id))
    validate_compatible_embedder_config(config)
    return config


def create_provider_embeddings(
    embedder: GoodMemEmbedderConfig,
    *,
    upstream_api_key_override: str | None = None,
) -> Embeddings:
    """Build the upstream LangChain embeddings provider for one embedder.

    Args:
        embedder: Normalized GoodMem embedder configuration.
        upstream_api_key_override: Optional explicit upstream API key that
            takes precedence over inline or environment-based resolution.

    Returns:
        An ``Embeddings`` implementation backed by
        ``langchain_openai.OpenAIEmbeddings``.

    Raises:
        GoodMemConfigurationError: If optional dependencies or upstream
            credentials are missing.
        GoodMemAPIError: If provider initialization fails.
    """
    openai_embeddings_cls = _load_openai_embeddings_class()
    upstream_api_key = resolve_upstream_api_key(
        embedder,
        upstream_api_key_override=upstream_api_key_override,
    )
    dimensions = resolve_upstream_dimensions(embedder)
    provider_kwargs: dict[str, Any] = {
        "api_key": upstream_api_key,
        "base_url": build_provider_base_url(embedder),
        "check_embedding_ctx_length": False,
    }
    if dimensions is not None:
        provider_kwargs["dimensions"] = dimensions

    try:
        return openai_embeddings_cls(
            model=embedder.model_identifier,
            **provider_kwargs,
        )
    except Exception as exc:
        raise GoodMemAPIError(
            format_provider_failure_message(
                (
                    "Failed to initialize the LangChain OpenAI-compatible "
                    "embeddings provider"
                ),
                exc,
            )
        ) from exc


def ensure_embedder(
    transport: SupportsEmbedderTransport,
    *,
    request: GoodMemEmbedderBootstrapRequest,
    upstream_api_key_override: str | None = None,
) -> GoodMemEmbedderConfig:
    """Find or create one compatible embedder and verify it is usable.

    Args:
        transport: Transport implementation exposing embedder bootstrap
            operations.
        request: Normalized bootstrap request describing the target embedder.
        upstream_api_key_override: Optional explicit upstream API key used
            during readiness validation when the resolved embedder does not
            expose a readable inline secret.

    Returns:
        The validated embedder configuration that satisfies
        ``GoodMemEmbeddings`` requirements. Existing matches are revalidated
        for compatibility and readiness before they are reused.

    Raises:
        GoodMemConfigurationError: If the bootstrap request is invalid, if more
            than one compatible embedder exists, or if the resolved embedder is
            unusable.
        GoodMemAPIError: If GoodMem list, create, or get operations fail.
    """
    validated_request = validate_bootstrap_request(request)
    matches = find_matching_embedders(
        transport,
        request=validated_request,
    )
    if len(matches) > 1:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings.ensure found multiple compatible embedders. "
            "Narrow the bootstrap inputs so one embedder matches the requested "
            "provider_type, endpoint_url, model_identifier, and dimensionality."
        )

    if matches:
        resolved = matches[0]
    else:
        created = transport.create_embedder(validated_request)
        resolved = _to_embedder_config(created)

    validate_compatible_embedder_config(resolved)
    resolve_upstream_api_key(
        resolved,
        upstream_api_key_override=upstream_api_key_override,
    )
    return resolved


def default_bootstrap_display_name() -> str:
    """Return the fixed default display name for bootstrap-created embedders."""
    return _DEFAULT_BOOTSTRAP_DISPLAY_NAME


def validate_bootstrap_request(
    request: GoodMemEmbedderBootstrapRequest,
) -> GoodMemEmbedderBootstrapRequest:
    """Validate and normalize one bootstrap request.

    Args:
        request: Candidate bootstrap request supplied by the public entry
            point.

    Returns:
        A normalized bootstrap request.

    Raises:
        GoodMemConfigurationError: If any field is unsupported, blank, or
            invalid for the current ``GoodMemEmbeddings`` bootstrap flow.
    """
    provider_type = _normalize_required_text(
        request.provider_type,
        error_message=("GoodMemEmbeddings.ensure requires provider_type to be OPENAI."),
    ).upper()
    if provider_type != "OPENAI":
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings.ensure currently supports only OPENAI provider_type."
        )

    supported_modalities = tuple(
        _normalize_required_text(
            modality,
            error_message=(
                "GoodMemEmbeddings.ensure requires supported_modalities to contain "
                "non-empty strings."
            ),
        ).upper()
        for modality in request.supported_modalities
    )
    if not supported_modalities:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings.ensure requires supported_modalities to be non-empty."
        )
    if "TEXT" not in supported_modalities:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings.ensure requires supported_modalities to include TEXT."
        )

    try:
        dimensionality = int(request.dimensionality)
    except (TypeError, ValueError) as exc:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings.ensure requires dimensionality to be a positive integer."
        ) from exc
    if dimensionality <= 0:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings.ensure requires dimensionality to be a positive integer."
        )

    return GoodMemEmbedderBootstrapRequest(
        display_name=_normalize_required_text(
            request.display_name,
            error_message=(
                "GoodMemEmbeddings.ensure requires display_name to be a "
                "non-empty string."
            ),
        ),
        endpoint_url=_normalize_required_text(
            request.endpoint_url,
            error_message=(
                "GoodMemEmbeddings.ensure requires endpoint_url to be a "
                "non-empty string."
            ),
        ),
        model_identifier=_normalize_required_text(
            request.model_identifier,
            error_message=(
                "GoodMemEmbeddings.ensure requires model_identifier to be a "
                "non-empty string."
            ),
        ),
        dimensionality=dimensionality,
        provider_type=provider_type,
        supported_modalities=supported_modalities,
        api_key=_normalize_optional_text(request.api_key),
    )


def find_matching_embedders(
    transport: SupportsEmbedderTransport,
    *,
    request: GoodMemEmbedderBootstrapRequest,
) -> list[GoodMemEmbedderConfig]:
    """Return compatible embedders visible to the current client.

    Args:
        transport: Transport implementation exposing embedder listing.
        request: Normalized bootstrap request describing the desired embedder.

    Returns:
        Compatible embedder configurations in transport-provided order.
    """
    matches: list[GoodMemEmbedderConfig] = []
    for raw_embedder in transport.list_embedders():
        config = _to_embedder_config(raw_embedder)
        if not embedder_matches_bootstrap_request(config, request=request):
            continue
        try:
            validate_compatible_embedder_config(config)
        except GoodMemConfigurationError:
            continue
        matches.append(config)
    return matches


def embedder_matches_bootstrap_request(
    embedder: GoodMemEmbedderConfig,
    *,
    request: GoodMemEmbedderBootstrapRequest,
) -> bool:
    """Check whether one embedder satisfies the requested bootstrap target."""
    if embedder.provider_type != request.provider_type:
        return False
    if embedder.endpoint_url != request.endpoint_url:
        return False
    if embedder.model_identifier != request.model_identifier:
        return False
    if embedder.dimensionality != request.dimensionality:
        return False
    embedder_modalities = {
        modality.upper() for modality in embedder.supported_modalities
    }
    return set(request.supported_modalities).issubset(embedder_modalities)


def format_provider_failure_message(prefix: str, exc: Exception) -> str:
    """Format one provider failure message with bounded detail text.

    Args:
        prefix: Stable message prefix that identifies the failing operation.
        exc: Original exception raised by the provider or setup path.

    Returns:
        A human-readable error string that includes bounded exception detail
        text when any detail is available.
    """
    detail = _bounded_detail_text(str(exc))
    if detail is None:
        return f"{prefix}."
    return f"{prefix}: {detail}"


def resolve_upstream_api_key(
    embedder: GoodMemEmbedderConfig,
    *,
    upstream_api_key_override: str | None = None,
) -> str:
    """Resolve the upstream provider API key for one embedder configuration.

    Args:
        embedder: Normalized GoodMem embedder configuration.
        upstream_api_key_override: Optional explicit upstream API key that
            takes precedence over inline or environment-based resolution.

    Returns:
        The API key that should be forwarded to the upstream embeddings
        provider.

    Raises:
        GoodMemConfigurationError: If neither inline credentials nor the
            environment fallback can satisfy the selected embedder.
    """
    override_api_key = _normalize_optional_text(upstream_api_key_override)
    if override_api_key is not None:
        return override_api_key

    inline_api_key = _normalize_optional_text(embedder.inline_api_key)
    if inline_api_key is not None:
        return inline_api_key

    env_api_key = _normalize_optional_text(os.getenv("GOODMEM_EMBEDDINGS_API_KEY"))
    if env_api_key is not None:
        return env_api_key

    if embedder.credential_kind in {None, "CREDENTIAL_KIND_UNSPECIFIED"}:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings could not find an upstream API key. "
            "The selected GoodMem embedder does not expose any credentials. "
            "Store a readable inline API key on the GoodMem embedder or set "
            "GOODMEM_EMBEDDINGS_API_KEY."
        )

    if embedder.credential_kind == "CREDENTIAL_KIND_API_KEY":
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings could not find an upstream API key. "
            "The selected GoodMem embedder uses API-key credentials, but GoodMem "
            "did not expose a readable inline secret. Set "
            "GOODMEM_EMBEDDINGS_API_KEY as a fallback."
        )

    raise GoodMemConfigurationError(
        "GoodMemEmbeddings could not find an upstream API key. "
        "The selected GoodMem embedder uses "
        f"{embedder.credential_kind} credentials, which GoodMemEmbeddings cannot "
        "forward directly. Set GOODMEM_EMBEDDINGS_API_KEY as a fallback."
    )


def resolve_upstream_dimensions(embedder: GoodMemEmbedderConfig) -> int | None:
    """Resolve the optional dimensions override for the upstream provider.

    Args:
        embedder: Normalized GoodMem embedder configuration.

    Returns:
        The configured dimensions override, or ``None`` when the environment
        variable is unset.

    Raises:
        GoodMemConfigurationError: If the environment value is invalid or does
            not match the GoodMem embedder dimensionality.
    """
    raw = os.getenv("GOODMEM_EMBEDDINGS_DIMENSIONS")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dimensions = int(raw)
    except ValueError as exc:
        raise GoodMemConfigurationError(
            "GOODMEM_EMBEDDINGS_DIMENSIONS must be an integer."
        ) from exc
    if dimensions <= 0:
        raise GoodMemConfigurationError(
            "GOODMEM_EMBEDDINGS_DIMENSIONS must be greater than 0."
        )
    if dimensions != embedder.dimensionality:
        raise GoodMemConfigurationError(
            "GOODMEM_EMBEDDINGS_DIMENSIONS="
            f"{dimensions} does not match the GoodMem embedder dimensionality "
            f"{embedder.dimensionality}."
        )
    return dimensions


def validate_compatible_embedder_config(embedder: GoodMemEmbedderConfig) -> None:
    """Validate that a GoodMem embedder can back ``GoodMemEmbeddings``.

    Args:
        embedder: Normalized GoodMem embedder configuration.

    Returns:
        ``None`` when the embedder configuration satisfies all provider
        compatibility requirements.

    Raises:
        GoodMemConfigurationError: If provider type, modality, endpoint, model,
            dimensions, or API path assumptions are not satisfied.
    """
    if embedder.provider_type != "OPENAI":
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings currently supports only OPENAI provider_type embedders."
        )
    if "TEXT" not in embedder.supported_modalities:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings requires an embedder that supports TEXT modality."
        )
    if not embedder.model_identifier.strip():
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings requires the GoodMem embedder to define "
            "model_identifier."
        )
    if not embedder.endpoint_url.strip():
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings requires the GoodMem embedder to define endpoint_url."
        )
    if embedder.dimensionality <= 0:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings requires the GoodMem embedder dimensionality "
            "to be greater than 0."
        )
    api_path = (embedder.api_path or "").strip()
    if api_path and not api_path.endswith("/embeddings"):
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings requires the GoodMem embedder api_path to be empty "
            "or end with '/embeddings'."
        )


def build_provider_base_url(embedder: GoodMemEmbedderConfig) -> str:
    """Build the upstream provider base URL from endpoint and API path.

    Args:
        embedder: Normalized GoodMem embedder configuration.

    Returns:
        The upstream provider base URL expected by
        ``langchain_openai.OpenAIEmbeddings`` after removing any terminal
        ``/embeddings`` path segment.
    """
    endpoint_url = embedder.endpoint_url.rstrip("/")
    api_path = (embedder.api_path or "").strip()
    if not api_path:
        return endpoint_url

    normalized_path = api_path if api_path.startswith("/") else f"/{api_path}"
    if normalized_path == "/embeddings":
        return endpoint_url

    base_path = normalized_path.removesuffix("/embeddings").rstrip("/")
    if not base_path:
        return endpoint_url
    return f"{endpoint_url}{base_path}"


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _bounded_detail_text(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    return text


def _normalize_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def _normalize_required_text(value: Any, *, error_message: str) -> str:
    text = _normalize_optional_text(value)
    if text is None:
        raise GoodMemConfigurationError(error_message)
    return text


def _load_openai_embeddings_class() -> type[Embeddings]:
    try:
        module = importlib.import_module("langchain_openai")
    except ImportError as exc:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings requires the optional 'openai' dependency. "
            f"{_OPENAI_EXTRA_INSTALL_HINT}"
        ) from exc

    openai_embeddings_cls = getattr(module, "OpenAIEmbeddings", None)
    if openai_embeddings_cls is None:
        raise GoodMemConfigurationError(
            "GoodMemEmbeddings could not load OpenAIEmbeddings from the optional "
            f"'openai' dependency. {_OPENAI_EXTRA_INSTALL_HINT}"
        )
    return openai_embeddings_cls


def _to_embedder_config(embedder: Any) -> GoodMemEmbedderConfig:
    credentials = getattr(embedder, "credentials", None)
    credential_kind = None
    if credentials is not None and getattr(credentials, "kind", None) is not None:
        credential_kind = _enum_value(credentials.kind)
    api_key_credentials = getattr(credentials, "api_key", None)
    inline_api_key = _normalize_optional_text(
        getattr(api_key_credentials, "inline_secret", None)
    )
    return GoodMemEmbedderConfig(
        embedder_id=embedder.embedder_id,
        provider_type=_enum_value(embedder.provider_type),
        endpoint_url=embedder.endpoint_url,
        api_path=embedder.api_path,
        model_identifier=embedder.model_identifier,
        dimensionality=int(embedder.dimensionality),
        supported_modalities=tuple(
            _enum_value(modality) for modality in embedder.supported_modalities
        ),
        credential_kind=credential_kind,
        inline_api_key=inline_api_key,
    )
