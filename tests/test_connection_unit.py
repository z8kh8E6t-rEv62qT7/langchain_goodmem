"""Unit tests for ``GoodMemConnection``.

This suite focuses on the public connection configuration layer.

Coverage goals:

- constructor normalization for ``api_key``, ``base_url``, and ``verify``
- rejection of blank or invalid caller-provided values
- environment-driven construction through ``GoodMemConnection.from_env()``

This suite stays separate from transport or vector-store tests because it
validates the package's smallest public configuration contract in isolation.
"""

from __future__ import annotations

import pytest

from langchain_goodmem import GoodMemConfigurationError, GoodMemConnection


def test_connection_constructor_trims_values_and_preserves_verify() -> None:
    connection = GoodMemConnection(
        api_key=" gm-key ",
        base_url=" https://goodmem.example ",
        verify=" custom-ca.pem ",
    )

    assert connection.api_key == "gm-key"
    assert connection.base_url == "https://goodmem.example"
    assert connection.verify == "custom-ca.pem"


def test_connection_constructor_rejects_blank_values() -> None:
    with pytest.raises(GoodMemConfigurationError, match="api_key"):
        GoodMemConnection(api_key="", base_url="https://goodmem.example")

    with pytest.raises(GoodMemConfigurationError, match="base_url"):
        GoodMemConnection(api_key="gm-key", base_url="  ")

    with pytest.raises(GoodMemConfigurationError, match="verify"):
        GoodMemConnection(
            api_key="gm-key",
            base_url="https://goodmem.example",
            verify="",
        )


def test_from_env_trims_values_and_accepts_explicit_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_API_KEY", " env-key ")
    monkeypatch.setenv("GOODMEM_BASE_URL", " https://goodmem.example ")

    connection = GoodMemConnection.from_env(verify=False)

    assert connection == GoodMemConnection(
        api_key="env-key",
        base_url="https://goodmem.example",
        verify=False,
    )


@pytest.mark.parametrize(
    ("env_name", "env_value", "match"),
    [
        ("GOODMEM_API_KEY", None, "GOODMEM_API_KEY"),
        ("GOODMEM_API_KEY", "   ", "GOODMEM_API_KEY"),
        ("GOODMEM_BASE_URL", None, "GOODMEM_BASE_URL"),
        ("GOODMEM_BASE_URL", "   ", "GOODMEM_BASE_URL"),
    ],
)
def test_from_env_requires_goodmem_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str | None,
    match: str,
) -> None:
    monkeypatch.setenv("GOODMEM_API_KEY", "env-key")
    monkeypatch.setenv("GOODMEM_BASE_URL", "https://goodmem.example")
    if env_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, env_value)

    with pytest.raises(GoodMemConfigurationError, match=match):
        GoodMemConnection.from_env()
