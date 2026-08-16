"""Write an explicit local AgentCheck configuration without importing the target."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, get_args

from .config import (
    CONFIG_FILENAME,
    DEFAULT_ENTRYPOINT,
    AgentCheckConfig,
    entrypoint_location,
)
from .errors import ConfigurationError


DEFAULT_ADAPTER = "openai_agents"
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
        raise ConfigurationError(f"target must be an existing directory: {resolved}")
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
            "entrypoint must use the form 'relative/path.py:attribute'"
        ) from exc


def _encoded_config(config: AgentCheckConfig) -> bytes:
    payload = config.model_dump(mode="json")
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _create_new_file(path: Path, payload: bytes) -> None:
    """Create ``path`` with mode ``0600``, failing if anything already exists there."""

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _write_config_file(destination: Path, payload: bytes, *, force: bool) -> None:
    if not force:
        try:
            _create_new_file(destination, payload)
        except FileExistsError as exc:
            raise ConfigurationError(
                f"{destination.name} already exists at {destination}; "
                "re-run with --force to replace it"
            ) from exc
        except OSError as exc:
            raise ConfigurationError(f"unable to write {destination}: {exc}") from exc
        return

    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(6)}.tmp")
    try:
        _create_new_file(temporary, payload)
        # Replacing the path itself never follows a symlink, so a hostile link at
        # the destination cannot redirect the write outside the target.
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
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
