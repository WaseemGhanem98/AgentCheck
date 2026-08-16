"""Load the explicit local AgentCheck target used by the Phase 1 CLI.

The MVP intentionally avoids repository-wide code discovery.  A directory target
uses ``agentcheck.json`` when present and otherwise defaults to
``agent.py:agent``.  Direct ``file.py`` and ``file.py:attribute`` references are
also accepted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


class TargetLoadError(RuntimeError):
    """Raised when a configured local target cannot be imported unambiguously."""


def _split_entrypoint(value: str, *, base: Path) -> tuple[Path, str]:
    module_value, separator, attribute = value.partition(":")
    attribute = attribute if separator else "agent"
    if not module_value:
        raise TargetLoadError("entrypoint must include a Python file")
    if not attribute or not attribute.isidentifier():
        raise TargetLoadError(f"invalid entrypoint attribute: {attribute!r}")

    module_path = Path(module_value)
    if not module_path.is_absolute():
        module_path = base / module_path
    module_path = module_path.resolve()
    if module_path.suffix != ".py":
        raise TargetLoadError("Phase 1 entrypoints must reference a .py file")
    if not module_path.is_file():
        raise TargetLoadError(f"entrypoint file does not exist: {module_path}")
    return module_path, attribute


def resolve_entrypoint(target: str | Path) -> tuple[Path, str]:
    """Resolve a directory/config/direct-file target without importing it."""

    raw_target = str(target)
    if ":" in raw_target:
        possible_file, _ = raw_target.rsplit(":", 1)
        if possible_file.endswith(".py"):
            return _split_entrypoint(raw_target, base=Path.cwd())

    path = Path(raw_target).resolve()
    if path.is_file():
        if path.suffix != ".py":
            raise TargetLoadError(f"target file must be Python: {path}")
        return path, "agent"
    if not path.is_dir():
        raise TargetLoadError(f"target path does not exist: {path}")

    config_path = path / "agentcheck.json"
    entrypoint = "agent.py:agent"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TargetLoadError(f"invalid {config_path.name}: {exc}") from exc
        if not isinstance(config, dict):
            raise TargetLoadError(f"{config_path.name} must contain a JSON object")
        configured = config.get("entrypoint", entrypoint)
        if not isinstance(configured, str) or not configured.strip():
            raise TargetLoadError(
                "agentcheck.json entrypoint must be a non-empty string"
            )
        entrypoint = configured.strip()
    module_path, attribute = _split_entrypoint(entrypoint, base=path)
    try:
        module_path.relative_to(path)
    except ValueError as exc:
        raise TargetLoadError(
            "configured entrypoint must remain inside the target directory"
        ) from exc
    return module_path, attribute


@contextmanager
def _temporary_import_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


def _load_module(module_path: Path) -> ModuleType:
    digest = hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_agentcheck_target_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise TargetLoadError(f"could not create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with _temporary_import_path(module_path.parent):
            spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise TargetLoadError(f"failed to import {module_path}: {exc}") from exc
    return module


def load_target(target: str | Path) -> tuple[Any, str]:
    """Import the configured object and return it with canonical provenance."""

    module_path, attribute = resolve_entrypoint(target)
    module = _load_module(module_path)
    if not hasattr(module, attribute):
        raise TargetLoadError(f"{module_path} does not export {attribute!r}")
    value = getattr(module, attribute)
    return value, f"{module_path}:{attribute}"


def inspect_target(target: str | Path) -> Any:
    """Load and inspect a supported local agent through the OpenAI adapter."""

    from agentcheck.adapters.openai_agents import OpenAIAgentsAdapter

    agent, source = load_target(target)
    return OpenAIAgentsAdapter().inspect(agent, source=source)


__all__ = [
    "TargetLoadError",
    "inspect_target",
    "load_target",
    "resolve_entrypoint",
]
