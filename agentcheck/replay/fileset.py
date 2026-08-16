"""Bounded source-file-set identity for replay and shrink.

This inventory hashes bytes on disk. It never imports a target module, never
executes target code, and never follows outbound symlinks. Git is used only as
``ls-files`` / ``rev-parse`` plumbing when the target is inside a repository.
Generated ``agentcheck-suite.json`` and ``agentcheck-baseline.json`` are not
source identity.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from agentcheck.domain import ContractModel, canonical_hash
from agentcheck.errors import ConfigurationError


_GIT_ENV_PREFIX = "GIT_"


def git_command_env() -> dict[str, str]:
    """Environment for git plumbing. Drops GIT_* so GIT_DIR cannot retarget us."""

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_GIT_ENV_PREFIX)
    }


SOURCE_FILE_SET_CONTRACT_VERSION: Literal["agentcheck.source_file_set.v1"] = (
    "agentcheck.source_file_set.v1"
)
MAX_SOURCE_FILES = 256
MAX_SOURCE_FILE_BYTES = 1 * 1024 * 1024
MAX_WALK_DEPTH = 8
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]*$")
_INCLUDED_SUFFIXES = (".py", ".json", ".toml", ".yaml", ".yml")
_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "__pycache__",
        ".agentcheck",
        "dist",
        "build",
        "node_modules",
        ".eggs",
    }
)
_EXCLUDED_FILE_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".env.test",
        "agentcheck-suite.json",
        "agentcheck-baseline.json",
    }
)
_EXCLUDED_SUFFIXES = (
    ".sqlite",
    ".db",
    ".pyc",
    ".pyo",
    ".html",
    ".so",
    ".dylib",
    ".dll",
    ".whl",
    ".egg",
)


class SourceFileEntry(ContractModel):
    """One inventoried source or config file, relative to the AgentCheck target."""

    path: str = Field(min_length=1, max_length=500)
    digest: str = Field(min_length=8, max_length=80)

    @model_validator(mode="after")
    def validate_entry(self) -> "SourceFileEntry":
        if not _is_safe_relative_path(self.path):
            raise ValueError(f"source file path is not a safe relative path: {self.path}")
        if not self.digest.startswith("sha256:"):
            raise ValueError("source file digest must be a sha256 digest")
        return self


class SourceFileSet(ContractModel):
    """Deterministic inventory of relevant source and config files."""

    schema_version: Literal["agentcheck.source_file_set.v1"] = (
        SOURCE_FILE_SET_CONTRACT_VERSION
    )
    mode: Literal["git_tracked", "local_files"]
    complete: bool
    files: tuple[SourceFileEntry, ...] = Field(min_length=1, max_length=MAX_SOURCE_FILES)
    fingerprint: str = ""

    def expected_fingerprint(self) -> str:
        return canonical_hash(
            self.model_dump(mode="json", exclude={"fingerprint"})
        )

    @model_validator(mode="after")
    def validate_identity(self) -> "SourceFileSet":
        paths = [item.path for item in self.files]
        if tuple(sorted(paths)) != tuple(paths):
            raise ValueError("source file entries must be sorted by path")
        if len(paths) != len(set(paths)):
            raise ValueError("source file paths must be unique")
        expected = self.expected_fingerprint()
        if self.fingerprint and not _digest_equal(self.fingerprint, expected):
            raise ValueError("source file-set fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        return self


def collect_source_file_set(root: Path) -> SourceFileSet:
    """Inventory relevant files under ``root`` without importing the target."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise ConfigurationError(f"target is not a directory: {root}")
    tracked = _git_tracked_paths(resolved)
    git_relevant = (
        [path for path in tracked if _is_relevant_source_path(path)]
        if tracked is not None
        else []
    )
    if git_relevant:
        relevant = git_relevant
        mode: Literal["git_tracked", "local_files"] = "git_tracked"
    else:
        relevant = [path for path in _local_paths(resolved) if _is_relevant_source_path(path)]
        mode = "local_files"
    if not relevant:
        raise ConfigurationError(
            "unable to establish a source file-set; no inventoriable source or "
            "config files were found under the target"
        )
    if len(relevant) > MAX_SOURCE_FILES:
        raise ConfigurationError(
            f"source file-set exceeds the {MAX_SOURCE_FILES} file bound"
        )
    entries = tuple(
        SourceFileEntry(path=path, digest=_hash_contained_file(resolved, path))
        for path in sorted(relevant)
    )
    return SourceFileSet(mode=mode, complete=True, files=entries)


def describe_file_set_mismatch(bound: SourceFileSet, live: SourceFileSet) -> str:
    """Return a fail-closed reason when two inventories are not identical."""

    if _digest_equal(bound.fingerprint, live.fingerprint):
        return ""
    bound_paths = {item.path: item.digest for item in bound.files}
    live_paths = {item.path: item.digest for item in live.files}
    missing = sorted(set(bound_paths) - set(live_paths))
    extra = sorted(set(live_paths) - set(bound_paths))
    modified = sorted(
        path
        for path, digest in bound_paths.items()
        if path in live_paths and live_paths[path] != digest
    )
    if missing:
        return f"source file missing since replay binding: {missing[0]}"
    if modified:
        return f"source file changed since replay binding: {modified[0]}"
    if extra:
        return f"unexpected source file since replay binding: {extra[0]}"
    if bound.mode != live.mode:
        return (
            f"source inventory mode {live.mode} does not match bound mode {bound.mode}"
        )
    return "source file-set fingerprint does not match the replay manifest"


def _is_relevant_source_path(relative: str) -> bool:
    if not _is_safe_relative_path(relative):
        return False
    parts = relative.split("/")
    if any(part in _EXCLUDED_DIR_NAMES or part.endswith(".egg-info") for part in parts[:-1]):
        return False
    name = parts[-1]
    if name in _EXCLUDED_FILE_NAMES or name.startswith(".env"):
        return False
    if any(name.endswith(suffix) for suffix in _EXCLUDED_SUFFIXES):
        return False
    return any(name.endswith(suffix) for suffix in _INCLUDED_SUFFIXES)


def _is_safe_relative_path(relative: str) -> bool:
    if _SAFE_RELATIVE.fullmatch(relative) is None or "\\" in relative:
        return False
    if relative.endswith("/") or "//" in relative:
        return False
    parts = relative.split("/")
    if any(not part or part.startswith(".") for part in parts):
        return False
    return True


def _git_tracked_paths(root: Path) -> tuple[str, ...] | None:
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=git_command_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "."],
            check=False,
            capture_output=True,
            timeout=5,
            env=git_command_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if listed.returncode != 0:
        return None
    paths: list[str] = []
    for raw in listed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8", errors="strict").replace("\\", "/")
        if relative.startswith("./"):
            relative = relative[2:]
        if not _is_safe_relative_path(relative):
            continue
        paths.append(relative)
    return tuple(paths)


def _local_paths(root: Path) -> tuple[str, ...]:
    collected: list[str] = []
    root_resolved = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        try:
            relative_dir = current.resolve().relative_to(root_resolved)
        except ValueError as exc:
            raise ConfigurationError(
                "refusing to follow a path that leaves the target directory"
            ) from exc
        depth = 0 if relative_dir == Path(".") else len(relative_dir.parts)
        if depth > MAX_WALK_DEPTH:
            raise ConfigurationError(
                f"source file-set walk exceeded the {MAX_WALK_DEPTH} directory depth bound"
            )
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in _EXCLUDED_DIR_NAMES
            and not name.endswith(".egg-info")
            and not name.startswith(".")
        )
        for name in dirnames:
            child = current / name
            if child.is_symlink():
                raise ConfigurationError(
                    f"refusing to follow source directory symlink {child}"
                )
        for name in sorted(filenames):
            child = current / name
            if relative_dir == Path("."):
                relative = name
            else:
                relative = "/".join((*relative_dir.parts, name))
            if not _is_safe_relative_path(relative):
                continue
            collected.append(relative)
    return tuple(collected)


def _hash_contained_file(root: Path, relative: str) -> str:
    if not _is_safe_relative_path(relative):
        raise ConfigurationError(f"source file path is not a safe relative path: {relative}")
    candidate = root / Path(*relative.split("/"))
    if candidate.is_symlink():
        raise ConfigurationError(
            f"refusing to follow source file symlink at {relative}"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ConfigurationError(
            f"unable to read source file {relative} for replay binding: {exc}"
        ) from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(MAX_SOURCE_FILE_BYTES + 1)
    except OSError as exc:
        raise ConfigurationError(
            f"unable to read source file {relative} for replay binding: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > MAX_SOURCE_FILE_BYTES:
        raise ConfigurationError(
            f"source file {relative} exceeds the {MAX_SOURCE_FILE_BYTES} byte bound"
        )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def omit_absent_file_set(data: dict[str, Any]) -> dict[str, Any]:
    """Drop a missing file-set so Slice 1 manifests keep their fingerprints."""

    if data.get("file_set") is None:
        data.pop("file_set", None)
    return data


def _digest_equal(left: str, right: str) -> bool:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if len(left_bytes) != len(right_bytes):
        hmac.compare_digest(left_bytes, left_bytes)
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


__all__ = [
    "MAX_SOURCE_FILE_BYTES",
    "MAX_SOURCE_FILES",
    "MAX_WALK_DEPTH",
    "SOURCE_FILE_SET_CONTRACT_VERSION",
    "SourceFileEntry",
    "SourceFileSet",
    "collect_source_file_set",
    "describe_file_set_mismatch",
    "git_command_env",
    "omit_absent_file_set",
]
