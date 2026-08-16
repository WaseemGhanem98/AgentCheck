"""Load a baseline document as untrusted input.

HTML, SQLite, replay manifests, frozen suites, and disclosure artifacts are
refused before model parse. Loading performs no import and no network access.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentcheck.config import contained_path
from agentcheck.errors import ConfigurationError

from .contract import BASELINE_CONTRACT_VERSION, MAX_BASELINE_BYTES, EvaluationBaseline


_SQLITE_MAGIC = b"SQLite format 3"
_REFUSED_CONTRACTS = {
    "agentcheck.replay_manifest.v1": "replay manifests are not evaluation baselines",
    "agentcheck.frozen_suite.v1": "frozen suites are not evaluation baselines",
    "agentcheck.suite.v1": "disclosure suite.json artifacts are not evaluation baselines",
    "agentcheck.summary.v1": "summary artifacts are not evaluation baselines",
    "agentcheck.human_review.v1": "human reviews are not evaluation baselines",
    "agentcheck.shrink_result.v1": "shrink results are not evaluation baselines",
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
        raise ConfigurationError(f"unable to read baseline {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read(MAX_BASELINE_BYTES + 1)
    except OSError as exc:
        raise ConfigurationError(f"unable to read baseline {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _refuse_wrong_artifact(path: Path, raw: bytes) -> None:
    if raw.startswith(_SQLITE_MAGIC):
        raise ConfigurationError(
            f"{path.name} is a SQLite database; SQLite is an index, not a baseline"
        )
    head = raw.lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        raise ConfigurationError(
            f"{path.name} is an HTML document; reports are not evaluation baselines"
        )


def parse_baseline(path: Path) -> EvaluationBaseline:
    raw = _read_untrusted_bytes(path)
    if len(raw) > MAX_BASELINE_BYTES:
        raise ConfigurationError(
            f"{path.name} exceeds the {MAX_BASELINE_BYTES} byte baseline limit"
        )
    _refuse_wrong_artifact(path, raw)
    try:
        text = raw.decode("utf-8")
        document = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid baseline {path.name}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError(
            f"invalid baseline {path.name}: the document must be a JSON object"
        )
    declared = document.get("schema_version")
    if declared in _REFUSED_CONTRACTS:
        raise ConfigurationError(
            f"{path.name} is {declared}; {_REFUSED_CONTRACTS[declared]}"
        )
    if declared != BASELINE_CONTRACT_VERSION:
        raise ConfigurationError(
            f"unsupported baseline contract {declared!r}; this build reads "
            f"{BASELINE_CONTRACT_VERSION}"
        )
    try:
        return EvaluationBaseline.model_validate_json(text)
    except ValueError as exc:
        raise ConfigurationError(f"invalid baseline {path.name}: {exc}") from exc


def load_baseline(root: Path, relative: str) -> EvaluationBaseline:
    """Load and integrity-check a contained baseline."""

    unresolved = root.resolve() / Path(relative)
    if unresolved.is_symlink():
        raise ConfigurationError(f"refusing to follow symlink at {unresolved}")
    path = contained_path(root, relative)
    if not path.is_file():
        raise ConfigurationError(f"baseline was not found: {relative}")
    return parse_baseline(path)


__all__ = ["load_baseline", "parse_baseline"]
