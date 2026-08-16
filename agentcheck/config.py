from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import ConfigurationError


CONFIG_FILENAME = "agentcheck.json"
DEFAULT_ENTRYPOINT = "agent.py:agent"
_ENTRYPOINT_RE = re.compile(r"^(?P<path>[^:]+\.py):(?P<attribute>[A-Za-z_][A-Za-z0-9_]*)$")
_SAFE_ENVIRONMENT = ("LANG", "LC_ALL", "TZ", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR")


class AgentCheckConfig(BaseModel):
    """Versioned local target configuration for the Phase 1 CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["agentcheck.config.v1"] = "agentcheck.config.v1"
    adapter: Literal["openai_agents"] = "openai_agents"
    entrypoint: str = DEFAULT_ENTRYPOINT
    suite: Literal["account_support_v1"] = "account_support_v1"
    seed: int = Field(default=1729, ge=0, le=2**63 - 1)
    max_concurrency: int = Field(default=2, ge=1, le=16)
    environment_allowlist: tuple[str, ...] = ()
    include_instructions_in_report: bool = False
    artifacts_directory: str = ".agentcheck"

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        normalized = value.strip()
        if _ENTRYPOINT_RE.fullmatch(normalized) is None:
            raise ValueError("entrypoint must use the form 'relative/path.py:attribute'")
        return normalized

    @field_validator("environment_allowlist")
    @classmethod
    def validate_environment_allowlist(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        result: list[str] = []
        for value in values:
            name = value.strip()
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
                raise ValueError("environment allowlist entries must be uppercase variable names")
            if name not in result:
                result.append(name)
        return tuple(result)

    @field_validator("artifacts_directory")
    @classmethod
    def validate_artifacts_directory(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("artifacts_directory must be a safe relative path")
        return path.as_posix()


def normalize_target(target: str | os.PathLike[str]) -> Path:
    path = Path(target).expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"target does not exist: {path}")
    if path.is_file():
        if path.suffix != ".py":
            raise ConfigurationError("a file target must be a Python source file")
        return path.parent
    if not path.is_dir():
        raise ConfigurationError(f"target is not a directory: {path}")
    return path


def load_config(target: str | os.PathLike[str]) -> tuple[Path, AgentCheckConfig]:
    root = normalize_target(target)
    config_path = root / CONFIG_FILENAME
    if config_path.exists():
        try:
            raw: Any = json.loads(config_path.read_text(encoding="utf-8"))
            config = AgentCheckConfig.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ConfigurationError(f"invalid {CONFIG_FILENAME}: {exc}") from exc
    else:
        config = AgentCheckConfig()
    resolve_entrypoint(root, config.entrypoint)
    return root, config


def entrypoint_location(root: Path, entrypoint: str) -> tuple[Path, str]:
    """Resolve a contained entrypoint without requiring the source to exist.

    Configuration commands must describe a target they are not allowed to import
    or even require on disk yet; existence is the caller's separate concern.
    """

    match = _ENTRYPOINT_RE.fullmatch(entrypoint)
    if match is None:  # already validated; retained for direct callers
        raise ConfigurationError("entrypoint must use the form 'relative/path.py:attribute'")
    resolved_root = root.resolve()
    source = (resolved_root / match.group("path")).resolve()
    try:
        source.relative_to(resolved_root)
    except ValueError as exc:
        raise ConfigurationError("entrypoint must remain inside the target directory") from exc
    return source, match.group("attribute")


def resolve_entrypoint(root: Path, entrypoint: str) -> tuple[Path, str]:
    source, attribute = entrypoint_location(root, entrypoint)
    if not source.is_file():
        raise ConfigurationError(f"entrypoint source does not exist: {source}")
    return source, attribute


def child_environment(config: AgentCheckConfig) -> dict[str, str]:
    """Build the explicit worker environment; provider credentials are absent by default."""

    allowed = set(_SAFE_ENVIRONMENT).union(config.environment_allowlist)
    environment = {name: os.environ[name] for name in sorted(allowed) if name in os.environ}
    environment.update(
        {
            "AGENTCHECK_CHILD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment
