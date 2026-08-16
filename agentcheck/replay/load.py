"""Load a replay manifest as untrusted input.

Frozen suites, disclosure artifacts, SQLite databases, and HTML reports are
refused before any model parse. Loading performs no import and no network
access.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentcheck.config import contained_path
from agentcheck.errors import ConfigurationError

from .manifest import (
    MAX_MANIFEST_BYTES,
    REPLAY_MANIFEST_CONTRACT_VERSION,
    ReplayManifest,
)


_SQLITE_MAGIC = b"SQLite format 3"
_REFUSED_CONTRACTS = {
    "agentcheck.frozen_suite.v1": "frozen suites are not replay manifests",
    "agentcheck.suite.v1": "disclosure suite.json artifacts are not replay manifests",
    "agentcheck.summary.v1": "summary artifacts are not replay manifests",
    "agentcheck.agent_spec.v1": "agent-spec artifacts are not replay manifests",
    "agentcheck.invalid_scenarios.v1": (
        "invalid-scenario artifacts are not replay manifests"
    ),
}


def _read_untrusted_bytes(path: Path) -> bytes:
    if path.is_symlink():
        raise ConfigurationError(f"refusing to follow symlink at {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfigurationError(f"unable to read replay manifest {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ConfigurationError(f"unable to read replay manifest {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _refuse_wrong_artifact(path: Path, raw: bytes) -> None:
    if raw.startswith(_SQLITE_MAGIC):
        raise ConfigurationError(
            f"{path.name} is a SQLite database; SQLite is an index, not a replay manifest"
        )
    head = raw.lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        raise ConfigurationError(
            f"{path.name} is an HTML document; reports are not replay manifests"
        )


def _parse_replay_manifest(path: Path) -> ReplayManifest:
    raw = _read_untrusted_bytes(path)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ConfigurationError(
            f"{path.name} exceeds the {MAX_MANIFEST_BYTES} byte replay-manifest limit"
        )
    _refuse_wrong_artifact(path, raw)
    try:
        text = raw.decode("utf-8")
        document = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid replay manifest {path.name}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError(
            f"invalid replay manifest {path.name}: the document must be a JSON object"
        )
    declared = document.get("schema_version")
    if declared in _REFUSED_CONTRACTS:
        raise ConfigurationError(
            f"{path.name} is {declared}; {_REFUSED_CONTRACTS[declared]}"
        )
    if declared != REPLAY_MANIFEST_CONTRACT_VERSION:
        raise ConfigurationError(
            f"unsupported replay manifest contract {declared!r}; this build reads "
            f"{REPLAY_MANIFEST_CONTRACT_VERSION}"
        )
    try:
        return ReplayManifest.model_validate_json(text)
    except ValueError as exc:
        raise ConfigurationError(f"invalid replay manifest {path.name}: {exc}") from exc


def load_replay_manifest(root: Path, relative: str) -> ReplayManifest:
    """Load and integrity-check a contained replay manifest."""

    unresolved = root.resolve() / Path(relative)
    if unresolved.is_symlink():
        raise ConfigurationError(f"refusing to follow symlink at {unresolved}")
    path = contained_path(root, relative)
    return _parse_replay_manifest(path)


def load_replay_manifest_path(path: Path) -> ReplayManifest:
    """Load an already-contained path. Used by tests that supply an absolute file."""

    return _parse_replay_manifest(path)


__all__ = [
    "load_replay_manifest",
    "load_replay_manifest_path",
]
