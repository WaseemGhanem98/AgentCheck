"""Write an explicit local AgentCheck configuration without importing the target."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, get_args

from .artifacts import create_private_file, replace_private_file
from .config import (
    CONFIG_FILENAME,
    DEFAULT_ENTRYPOINT,
    AgentCheckConfig,
    entrypoint_location,
)
from .errors import ConfigurationError


DEFAULT_ADAPTER = "openai_agents"
_ENTRYPOINT_FORM = (
    "entrypoint must use the form 'relative/path.py:attribute' "
    "or the explicit factory form 'relative/path.py:attribute()'"
)
SUPPORTED_ADAPTERS: tuple[str, ...] = get_args(
    AgentCheckConfig.model_fields["adapter"].annotation
)


def resolve_initialization_root(target: str | os.PathLike[str]) -> Path:
    """Resolve the directory ``init`` writes into.

    Unlike :func:`agentcheck.config.normalize_target` a file is rejected rather
    than reinterpreted as its parent, so the written location is always the path
    the caller named.
    """

    try:
        resolved = Path(target).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError(f"target path cannot be resolved: {exc}") from exc
    if not resolved.is_dir():
        raise ConfigurationError(
            f"target must be an existing directory: {resolved}; "
            "create the directory first, then run agentcheck init again"
        )
    return resolved


def _validated_config(adapter: str, entrypoint: str) -> AgentCheckConfig:
    if adapter not in SUPPORTED_ADAPTERS:
        raise ConfigurationError(
            f"unsupported adapter: {adapter!r}; supported adapters: "
            f"{', '.join(SUPPORTED_ADAPTERS)}"
        )
    document: dict[str, Any] = {"adapter": adapter, "entrypoint": entrypoint}
    try:
        return AgentCheckConfig.model_validate(document)
    except ValueError as exc:
        raise ConfigurationError(
            _ENTRYPOINT_FORM
        ) from exc


def _encoded_config(config: AgentCheckConfig) -> bytes:
    payload = config.model_dump(mode="json", exclude_none=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _write_config_file(destination: Path, payload: bytes, *, force: bool) -> None:
    try:
        if force:
            replace_private_file(destination, payload)
        else:
            create_private_file(destination, payload)
    except FileExistsError as exc:
        raise ConfigurationError(
            f"{destination.name} already exists at {destination}; "
            "re-run with --force to replace it"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(f"unable to write {destination}: {exc}") from exc


def write_initial_config(
    root: str | os.PathLike[str],
    *,
    adapter: str = DEFAULT_ADAPTER,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    force: bool = False,
) -> Path:
    """Write a validated ``agentcheck.json`` at ``root`` and return its path.

    The document is serialized from a constructed :class:`AgentCheckConfig`, so an
    unwritable value fails before any file is created. The target agent is never
    imported and its entrypoint is not required to exist yet.
    """

    resolved_root = resolve_initialization_root(root)
    config = _validated_config(adapter, entrypoint)
    entrypoint_location(resolved_root, config.entrypoint)
    destination = resolved_root / CONFIG_FILENAME
    _write_config_file(destination, _encoded_config(config), force=force)
    return destination


__all__ = [
    "DEFAULT_ADAPTER",
    "SUPPORTED_ADAPTERS",
    "resolve_initialization_root",
    "write_initial_config",
]
