from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import ConfigurationError


CONFIG_FILENAME = "agentcheck.json"
DEFAULT_ENTRYPOINT = "agent.py:agent"
_ENTRYPOINT_RE = re.compile(
    r"^(?P<path>[^:]+\.py):(?P<attribute>[A-Za-z_][A-Za-z0-9_]*)(?P<factory>\(\))?$"
)
_SAFE_ENVIRONMENT = ("LANG", "LC_ALL", "TZ", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR")
_MAX_REALIZATION_CALLS = 32
_MAX_REALIZATION_RETRIES = 2
_ENTRYPOINT_FORM = (
    "entrypoint must use the form 'relative/path.py:attribute' "
    "or the explicit factory form 'relative/path.py:attribute()'"
)


class ParsedEntrypoint(NamedTuple):
    """Explicit configured object or zero-argument factory inside the target."""

    path: str
    attribute: str
    factory: bool


class LlmRealizationConfig(BaseModel):
    """Optional nested block. Disabled unless enabled is true and CLI consents."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    enabled: bool = False
    provider: Literal["openai"] = "openai"
    model: str = Field(default="gpt-4o-mini", min_length=1, max_length=200)
    max_calls: int = Field(default=8, ge=1, le=_MAX_REALIZATION_CALLS)
    max_retries: int = Field(default=1, ge=0, le=_MAX_REALIZATION_RETRIES)


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
    # Deny-by-default egress. Stripping credentials is not a boundary: a target
    # that configures its own provider base_url (Ollama, vLLM, a staging host)
    # otherwise reaches it for real during evaluation.
    allow_network: bool = False
    include_instructions_in_report: bool = False
    artifacts_directory: str = ".agentcheck"
    suite_path: str | None = None
    policy_packs: tuple[str, ...] | None = None
    store_path: str | None = None
    max_cases: int | None = Field(default=None, ge=1, le=256)
    llm_realization: LlmRealizationConfig | None = None
    python_executable: str | None = None

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        normalized = value.strip()
        parse_entrypoint(normalized)
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

    @field_validator("suite_path")
    @classmethod
    def validate_suite_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if not value.strip():
            raise ValueError("suite_path must not be empty")
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("suite_path must be a safe relative path")
        return path.as_posix()

    @field_validator("policy_packs")
    @classmethod
    def validate_policy_packs(
        cls, values: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if values is None:
            return None
        result: list[str] = []
        for value in values:
            name = value.strip()
            if not name:
                raise ValueError("policy pack names must not be empty")
            path = Path(name)
            if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise ValueError("policy pack names must be a safe relative path")
            normalized = path.as_posix()
            if normalized not in result:
                result.append(normalized)
        return tuple(result)

    @field_validator("store_path")
    @classmethod
    def validate_store_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if not value.strip():
            raise ValueError("store_path must not be empty")
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("store_path must be a safe relative path")
        return path.as_posix()

    @field_validator("python_executable")
    @classmethod
    def validate_python_executable(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("python_executable must not be empty")
        if "\x00" in text:
            raise ValueError("python_executable must not contain NUL")
        path = Path(text)
        if path.is_absolute():
            return text
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("python_executable relative path must be a safe relative path")
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


def contained_path(root: Path, relative: str) -> Path:
    """Resolve a relative path that must stay inside the target directory.

    Symlinks are resolved before the containment check, so a link pointing out
    of the target is refused for reads and writes alike.
    """

    if not relative.strip():
        raise ConfigurationError("path must not be empty")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ConfigurationError("path must be relative to the target directory")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ConfigurationError("path must be a safe relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ConfigurationError("path must remain inside the target directory") from exc
    return resolved


def parse_entrypoint(value: str) -> ParsedEntrypoint:
    """Parse an explicit attribute or zero-argument factory entrypoint."""

    match = _ENTRYPOINT_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(_ENTRYPOINT_FORM)
    return ParsedEntrypoint(
        path=match.group("path"),
        attribute=match.group("attribute"),
        factory=match.group("factory") == "()",
    )


def entrypoint_location(root: Path, entrypoint: str) -> tuple[Path, str]:
    """Resolve a contained entrypoint without requiring the source to exist.

    Configuration commands must describe a target they are not allowed to import
    or even require on disk yet; existence is the caller's separate concern.
    """

    try:
        parsed = parse_entrypoint(entrypoint)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    resolved_root = root.resolve()
    source = (resolved_root / parsed.path).resolve()
    try:
        source.relative_to(resolved_root)
    except ValueError as exc:
        raise ConfigurationError("entrypoint must remain inside the target directory") from exc
    return source, parsed.attribute


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


def apply_python_executable(
    config: AgentCheckConfig, python_executable: str | None
) -> AgentCheckConfig:
    """Return config with an optional CLI interpreter override applied."""

    if python_executable is None:
        return config
    text = python_executable.strip()
    if not text:
        raise ConfigurationError("python_executable must not be empty")
    try:
        return config.model_copy(update={"python_executable": text})
    except ValueError as exc:
        raise ConfigurationError(f"invalid python_executable: {exc}") from exc


def resolve_python_executable(root: Path, config: AgentCheckConfig) -> Path:
    """Resolve the worker interpreter without inheriting host PYTHONPATH.

    ``None`` uses the interpreter that invoked AgentCheck, preserving a venv
    wrapper executable (do not follow it to the system Python). A relative
    path must be lexically inside the target (typically ``.venv/bin/python``).
    A venv interpreter may symlink outside the target; that is an explicit
    trust decision, not automatic host site-packages inheritance. Absolute
    paths are also explicit opt-in. The returned path is the file that will
    be executed, not a flattened symlink target.
    """

    configured = config.python_executable
    if configured is None:
        return Path(sys.executable)
    raw = Path(configured)
    if raw.is_absolute():
        candidate = raw.expanduser()
        _require_interpreter_file(candidate)
        return candidate
    if any(part in {"", ".", ".."} for part in raw.parts):
        raise ConfigurationError(
            "python_executable relative path must be a safe relative path"
        )
    resolved_root = root.resolve()
    declared = resolved_root / raw
    try:
        declared.relative_to(resolved_root)
    except ValueError as exc:
        raise ConfigurationError(
            "python_executable must remain inside the target directory"
        ) from exc
    _require_interpreter_file(declared)
    return declared


def _require_interpreter_file(path: Path) -> None:
    if not path.is_file():
        raise ConfigurationError(
            f"python_executable is not an existing interpreter file: {path}"
        )
    if os.name == "posix" and not os.access(path, os.X_OK):
        raise ConfigurationError(f"python_executable is not executable: {path}")
